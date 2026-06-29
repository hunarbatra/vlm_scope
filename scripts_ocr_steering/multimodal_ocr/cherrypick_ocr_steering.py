#!/usr/bin/env python3
"""
Cherry-pick OCR steering wins and build collages for paper figures.

For each steering target (layer, feature, category):
  1. Loads results.json from caa_per_category_ocr.py
  2. Finds best alpha/gamma config (highest Δcat with |ΔCtrl| < 2%)
  3. Identifies samples where steered=correct AND baseline=incorrect
  4. Saves images + per-sample JSON
  5. Builds a comparison collage: [image | question | baseline resp | steered resp]

Output per target:
  {OUT_DIR}/L{layer}_F{feature}_{cat}/
    wins.json          — per-sample records
    win_01.jpg, ...    — cropped/resized source images
    collage.png        — full comparison collage
    collage.jpg

Usage (generates for all completed targets):
    python3 cherrypick_ocr_steering.py
    python3 cherrypick_ocr_steering.py --layer 19 --feature 10089 --category "Scene Text-centric VQA"
"""
import os, sys, json, argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

CAA_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/caa_per_category")
DOCS_DIR = Path("/home/hbatra/vision-language-scope/docs/imgs/ocr_steering")

# 5 steering targets
TARGETS = [
    {"layer": 19, "feature": 10089, "category": "Scene Text-centric VQA"},
    {"layer": 17, "feature": 13602, "category": "Non-Semantic Text Recognition"},
    {"layer": 21, "feature": 9577,  "category": "Digit String Recognition"},
    {"layer": 19, "feature": 14093, "category": "Irregular Text Recognition"},
    {"layer": 20, "feature": 10687, "category": "Key Information Extraction"},
]

THUMB_W = 320
GAP = 14
CAPTION_H = 210
MAX_WINS = 20  # save up to 20 wins per target so user can pick


def _font(sz, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        try: return ImageFont.truetype(p, sz)
        except: continue
    return ImageFont.load_default()


def _wrap(text, font, max_w, draw):
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if draw.textlength(test, font=font) <= max_w:
            cur.append(w)
        else:
            if cur: lines.append(" ".join(cur))
            cur = [w]
    if cur: lines.append(" ".join(cur))
    return lines


def render_row(img, record, font_lg, font_md, font_sm):
    """Render one sample row: [thumbnail | question+GT | baseline | steered]."""
    w, h = img.size
    new_h = int(THUMB_W * h / w)
    img_th = img.resize((THUMB_W, new_h), Image.LANCZOS)

    text_col_w = THUMB_W * 3 + GAP * 3
    panel_h = max(new_h, CAPTION_H) + 10
    panel_w = THUMB_W + GAP + text_col_w
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    panel.paste(img_th, (0, (panel_h - new_h) // 2))

    d = ImageDraw.Draw(panel)
    x = THUMB_W + GAP

    question = record["question"]
    gt = record.get("gt", "")
    base_resp = record.get("baseline_resp", "")
    steer_resp = record.get("steered_resp", "")

    # Question
    d.text((x, 4), "Question:", fill="#444", font=font_sm)
    q_lines = _wrap(question, font_md, text_col_w - 8, d)
    y = 20
    for ln in q_lines[:4]:
        d.text((x, y), ln, fill="black", font=font_md); y += 20
    d.text((x, y + 2), f"GT: {gt}", fill="#555", font=font_sm); y += 22

    # Two columns
    col_w = (text_col_w - GAP) // 2
    # Baseline
    d.rectangle((x, y, x + col_w, y + 72), outline="#c33", width=2)
    d.text((x + 6, y + 5), "BEFORE — baseline", fill="#c33", font=font_sm)
    resp_lines = _wrap(base_resp[:120], font_md, col_w - 12, d)
    ry = y + 22
    for ln in resp_lines[:3]:
        d.text((x + 6, ry), ln, fill="#a00", font=font_md); ry += 18
    d.text((x + 6, y + 56), "✗  incorrect", fill="#c33", font=font_sm)
    # Steered
    x2 = x + col_w + GAP
    d.rectangle((x2, y, x2 + col_w, y + 72), outline="#063", width=2)
    d.text((x2 + 6, y + 5), "AFTER — Recipe D steered", fill="#063", font=font_sm)
    s_lines = _wrap(steer_resp[:120], font_md, col_w - 12, d)
    ry2 = y + 22
    for ln in s_lines[:3]:
        d.text((x2 + 6, ry2), ln, fill="#063", font=font_md); ry2 += 18
    d.text((x2 + 6, y + 56), "✓  correct", fill="#063", font=font_sm)

    return panel


def build_collage(records, imgs, target, best_key, base_acc, best_acc, out_dir):
    font_lg = _font(20, bold=True)
    font_md = _font(14)
    font_sm = _font(11)

    rows = []
    for r, img in zip(records, imgs):
        try:
            rows.append(render_row(img, r, font_lg, font_md, font_sm))
        except Exception as e:
            print(f"  [skip row] {e}")
    if not rows: return

    pad = 20
    title_h = 72
    body_w = max(p.width for p in rows) + pad * 2
    body_h = sum(p.height for p in rows) + pad * (len(rows) + 1) + title_h

    canvas = Image.new("RGB", (body_w, body_h), "white")
    d = ImageDraw.Draw(canvas)

    alpha = best_key.split("_")[0][1:]
    gamma = best_key.split("_")[1][1:]
    d.text((pad, 8), f"OCR Steering - L{target['layer']}/F{target['feature']} - '{target['category']}'",
           fill="black", font=_font(22, bold=True))
    d.text((pad, 36), f"Recipe D: α={alpha} γ={gamma}  |  base={base_acc:.1f}%  →  steered={best_acc:.1f}%  (Δ={best_acc-base_acc:+.1f}%)",
           fill="#444", font=font_md)
    d.text((pad, 56), f"Showing {len(rows)} steered-correct / baseline-incorrect wins",
           fill="#666", font=font_sm)

    y = title_h
    for p in rows:
        canvas.paste(p, (pad, y))
        y += p.height + pad

    out_png = out_dir / "collage.png"
    out_jpg = out_dir / "collage.jpg"
    canvas.save(out_png, "PNG")
    canvas.save(out_jpg, "JPEG", quality=92)
    print(f"  [OK] {out_png} ({out_png.stat().st_size // 1024}K)")
    print(f"  [OK] {out_jpg} ({out_jpg.stat().st_size // 1024}K)")


def process_target(target, ocr):
    layer, feature, category = target["layer"], target["feature"], target["category"]
    cat_short = category.replace(" ", "_").replace("-", "_")
    caa_subdir = CAA_DIR / f"L{layer}_F{feature}_{cat_short}"
    results_path = caa_subdir / "results.json"

    if not results_path.exists():
        print(f"[SKIP] {results_path} not found"); return

    data = json.load(open(results_path))
    base_acc = data.get("base_cat", {}).get("acc", 0)
    base_results = data.get("base_cat_results", [])

    if not base_results:
        print(f"[SKIP] No base_cat_results in {results_path}"); return

    # Find best alpha/gamma config: highest Δcat with |ΔCtrl| < 2%
    best_key, best_delta, best_acc, best_ctrl = None, -999, 0, 0
    for k, v in data.items():
        if not (k.startswith("a") and "delta_cat" in v): continue
        ctrl_ok = abs(v.get("delta_ctrl", 0)) < 2.0
        if ctrl_ok and v["delta_cat"] > best_delta:
            best_key = k
            best_delta = v["delta_cat"]
            best_acc = v["acc_cat"]
            best_ctrl = v.get("delta_ctrl", 0)

    if best_key is None:
        print(f"[SKIP] No valid alpha/gamma config (all |ΔCtrl| >= 2%) for {cat_short}")
        # Fall back to any config
        for k, v in data.items():
            if not (k.startswith("a") and "delta_cat" in v): continue
            if v["delta_cat"] > best_delta:
                best_key = k; best_delta = v["delta_cat"]
                best_acc = v["acc_cat"]; best_ctrl = v.get("delta_ctrl", 0)
        if best_key is None: return

    best_config = data[best_key]
    steer_results = best_config.get("results", [])
    if not steer_results:
        print(f"[SKIP] No per-sample results in {best_key}"); return

    print(f"\n[{cat_short}] best={best_key}  Δcat={best_delta:+.1f}%  ΔCtrl={best_ctrl:+.2f}%", flush=True)

    # Map si → baseline correct
    base_map = {r["si"]: r for r in base_results}
    steer_map = {r["si"]: r for r in steer_results}

    wins = []
    for si, s_res in steer_map.items():
        if not s_res["correct"]: continue
        b_res = base_map.get(si)
        if b_res is None or b_res["correct"]: continue  # baseline was also correct
        wins.append({
            "si": si,
            "question": s_res.get("question", ""),
            "question_type": s_res.get("question_type", category),
            "baseline_resp": b_res.get("response", ""),
            "steered_resp": s_res.get("response", ""),
            "gt": str(ocr[si].get("answer", [""])[0] if isinstance(ocr[si].get("answer"), list) else ocr[si].get("answer", "")),
        })

    print(f"  wins: {len(wins)} (steered correct + baseline wrong)", flush=True)
    if not wins:
        print(f"  [NOTE] No wins found — steering may not have improved this subset"); return

    # Sort by steered response confidence proxy: prefer non-empty responses
    wins = wins[:MAX_WINS]

    # Output dir
    out_dir = DOCS_DIR / f"L{layer}_F{feature}_{cat_short}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-win images
    saved = []
    saved_imgs = []
    for j, r in enumerate(wins):
        img = ocr[r["si"]].get("image")
        if img is None: continue
        try:
            img = img.convert("RGB")
            img_path = out_dir / f"win_{j+1:02d}_si{r['si']}.jpg"
            # Resize to reasonable size for display
            mw = 800
            if img.width > mw:
                img = img.resize((mw, int(mw * img.height / img.width)), Image.LANCZOS)
            img.save(img_path, "JPEG", quality=92)
            r["image_path"] = str(img_path)
            saved.append(r)
            saved_imgs.append(img)
        except Exception as e:
            print(f"  [skip si={r['si']}] {e}")

    wins_path = out_dir / "wins.json"
    with open(wins_path, "w") as f:
        json.dump({"target": target, "best_config": best_key,
                   "base_acc": base_acc, "steered_acc": best_acc,
                   "delta_cat": best_delta, "delta_ctrl": best_ctrl,
                   "n_wins": len(saved), "wins": saved}, f, indent=2)
    print(f"  wins JSON: {wins_path}", flush=True)

    # Build collage (show up to 10 rows to keep file manageable)
    build_collage(saved[:10], saved_imgs[:10], target, best_key,
                  base_acc, best_acc, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--feature", type=int, default=None)
    ap.add_argument("--category", type=str, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import load_dataset

    targets = TARGETS
    if args.layer is not None:
        targets = [t for t in TARGETS
                   if t["layer"] == args.layer and t["feature"] == args.feature]

    print("[INFO] Loading OCR-Bench...", flush=True)
    ocr = load_dataset("echo840/OCRBench", split="test")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for t in targets:
        print(f"\n{'='*60}", flush=True)
        print(f"Processing L{t['layer']}/F{t['feature']} — {t['category']}", flush=True)
        try:
            process_target(t, ocr)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    print("\n[DONE] Cherry-pick complete.", flush=True)


if __name__ == "__main__":
    main()
