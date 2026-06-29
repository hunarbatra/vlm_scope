#!/usr/bin/env python3
"""
Build a collage showing baseline vs steered (Recipe D) predictions on cherry-picked
VSR samples for L12_F2257 ("facing").

Reads:
  /home/hbatra/vision-language-scope/docs/imgs/spatial_steering/cherrypick_results.json
  individual JPEGs under same dir
Writes:
  /home/hbatra/vision-language-scope/docs/imgs/spatial_steering/collage.png   (full)
  /home/hbatra/vision-language-scope/docs/imgs/spatial_steering/collage.jpg
  /home/hbatra/vision-language-scope/docs/spatial_steering_examples.md
"""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

_DOCS = Path("/home/hbatra/vision-language-scope/docs")
_IMGS = _DOCS / "imgs"

# Default to facing run; override with --suffix
OUT_DIR = _IMGS / "spatial_steering"
DOC_PATH = _DOCS / "spatial_steering_examples.md"
JSON_PATH = OUT_DIR / "cherrypick_results.json"

THUMB_W = 360
GAP = 16
CAPTION_H = 200
LABEL_H = 28


def _font(sz, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        try: return ImageFont.truetype(p, sz)
        except Exception: continue
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


def render_pair(record, font_lg, font_md, font_sm):
    """Render one row: image | caption | baseline pred | steered pred."""
    img_path = Path("/home/hbatra/vision-language-scope") / record["image_path"]
    img = Image.open(img_path).convert("RGB")
    # thumbnail
    w, h = img.size
    new_h = int(THUMB_W * h / w)
    img_th = img.resize((THUMB_W, new_h), Image.LANCZOS)

    # build a panel for this row
    panel_h = max(new_h, CAPTION_H)
    text_w = THUMB_W * 2 + GAP * 2
    panel_w = THUMB_W + GAP + text_w
    panel = Image.new("RGB", (panel_w, panel_h + 40), "white")
    panel.paste(img_th, (0, (panel_h - new_h) // 2))

    d = ImageDraw.Draw(panel)
    x = THUMB_W + GAP

    cap = record["caption"]
    label = "Yes" if record["label"] == 1 else "No"
    rel = record.get("relation", "")
    base_pred = "Yes" if record["baseline_pred"] == 1 else "No"
    steer_pred = "Yes" if record["steered_pred"] == 1 else "No"
    base_conf = record.get("baseline_conf", 0)
    steer_conf = record.get("steered_conf", 0)

    # Caption header
    d.text((x, 4), f"Statement", fill="#444", font=font_sm)
    cap_lines = _wrap(f"“{cap}”", font_md, text_w - 10, d)
    y = 22
    for ln in cap_lines:
        d.text((x, y), ln, fill="black", font=font_md); y += 22
    d.text((x, y + 4), f"relation: {rel}    |    ground truth: {label}",
           fill="#666", font=font_sm)
    y += 30

    # Two columns
    col_w = (text_w - GAP) // 2
    # Baseline column
    d.rectangle((x, y, x + col_w, y + 60), outline="#c33", width=2)
    d.text((x + 8, y + 6), "BEFORE — pt-448 baseline", fill="#c33", font=font_md)
    d.text((x + 8, y + 30), f"→ {base_pred}   ({base_conf*100:.0f}% conf)   ✗",
           fill="#a00", font=font_lg)
    # Steered column
    x2 = x + col_w + GAP
    d.rectangle((x2, y, x2 + col_w, y + 60), outline="#063", width=2)
    d.text((x2 + 8, y + 6), "AFTER — Spatial Steering (Recipe D)",
           fill="#063", font=font_md)
    d.text((x2 + 8, y + 30), f"→ {steer_pred}   ({steer_conf*100:.0f}% conf)   ✓",
           fill="#063", font=font_lg)
    return panel


def main():
    global OUT_DIR, DOC_PATH, JSON_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", type=str, default="",
                    help='e.g. "close_to" → reads /docs/imgs/spatial_steering_close_to/, writes /docs/spatial_steering_examples_close_to.md')
    args = ap.parse_args()
    if args.suffix:
        OUT_DIR = _IMGS / f"spatial_steering_{args.suffix}"
        DOC_PATH = _DOCS / f"spatial_steering_examples_{args.suffix}.md"
        JSON_PATH = OUT_DIR / "cherrypick_results.json"
    if not JSON_PATH.exists():
        print(f"[ERROR] missing {JSON_PATH}; run cherrypick_recipe_D_steering.py first")
        return
    data = json.load(open(JSON_PATH))
    wins = data["wins"]
    if not wins:
        print("[ERROR] no wins in cherrypick output")
        return

    font_lg = _font(22, bold=True)
    font_md = _font(15)
    font_sm = _font(12)

    # Render each row
    rows = []
    for r in wins:
        try:
            rows.append(render_pair(r, font_lg, font_md, font_sm))
        except Exception as e:
            print(f"  skip vi={r['vi']}: {e}")
    if not rows: return

    pad = 24
    title_h = 80
    body_w = max(p.width for p in rows) + pad * 2
    body_h = sum(p.height for p in rows) + pad * (len(rows) + 1) + title_h

    canvas = Image.new("RGB", (body_w, body_h), "white")
    d = ImageDraw.Draw(canvas)
    title_font = _font(26, bold=True)
    sub_font = _font(15)
    d.text((pad, 10),
           f"Spatial Steering — VSR examples (feature L{data['feature']['layer']}/F{data['feature']['feature']}, relation “{', '.join(data['feature']['relations'])}”)",
           fill="black", font=title_font)
    rec = data["recipe_D"]
    d.text((pad, 44),
           f"Recipe D = BACKBONE α={rec['alpha']} @ layers {rec['layers']}  +  γ={rec['gamma']} · W_dec[F] @ L{data['feature']['layer']}",
           fill="#444", font=sub_font)
    d.text((pad, 60),
           f"Eval set: {data['n_eval']} R(F)∩test samples · base acc {100*data['n_correct_base']/data['n_eval']:.1f}% · steered acc {100*data['n_correct_steer']/data['n_eval']:.1f}% · {data['n_wins']} wins, {data['n_losses']} losses",
           fill="#444", font=sub_font)

    y = title_h
    for p in rows:
        canvas.paste(p, (pad, y))
        y += p.height + pad

    out_png = OUT_DIR / "collage.png"
    out_jpg = OUT_DIR / "collage.jpg"
    canvas.save(out_png, "PNG")
    canvas.save(out_jpg, "JPEG", quality=92)
    print(f"[OK] wrote {out_png} ({out_png.stat().st_size//1024}K)")
    print(f"[OK] wrote {out_jpg} ({out_jpg.stat().st_size//1024}K)")

    # Doc
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat = data["feature"]
    md = []
    md.append(f"# Spatial Steering — qualitative wins for L{feat['layer']}/F{feat['feature']} (“{', '.join(feat['relations'])}”)")
    md.append("")
    md.append("**Recipe D** (BACKBONE multi-layer CAA + W_dec[F] feature amplification):")
    md.append(f"- BACKBONE: α = {rec['alpha']} · unit(v_meanpool[L]) at every L ∈ {rec['layers']}")
    md.append(f"- + γ = {rec['gamma']} · W_dec[L{feat['layer']}, F={feat['feature']}] at L{feat['layer']} only")
    md.append("")
    md.append(f"**Headline (R(F)∩test, n={data['n_eval']})**: baseline {100*data['n_correct_base']/data['n_eval']:.2f}% → steered {100*data['n_correct_steer']/data['n_eval']:.2f}% (Δ {100*(data['n_correct_steer']-data['n_correct_base'])/data['n_eval']:+.2f}pp; {data['n_wins']} wins, {data['n_losses']} losses).")
    md.append("")
    md.append(f"**Collage**: ![collage]({OUT_DIR.relative_to(_DOCS)}/collage.png)")
    md.append("")
    md.append("## Cherry-picked wins (steered = correct, baseline = incorrect)")
    md.append("")
    for j, r in enumerate(wins):
        gt = "Yes" if r["label"] == 1 else "No"
        bp = "Yes" if r["baseline_pred"] == 1 else "No"
        sp = "Yes" if r["steered_pred"] == 1 else "No"
        md.append(f"### {j+1}. “{r['caption']}”  (relation: *{r.get('relation','')}*  ·  GT: **{gt}**)")
        md.append(f"![win{j+1}]({r['image_path']})")
        md.append("")
        md.append(f"- **Baseline (pt-448)**: → **{bp}** ({r['baseline_conf']*100:.0f}% conf)  ❌")
        md.append(f"- **Steered (Recipe D)**: → **{sp}** ({r['steered_conf']*100:.0f}% conf)  ✓")
        md.append("")
    DOC_PATH.write_text("\n".join(md))
    print(f"[OK] wrote {DOC_PATH} ({DOC_PATH.stat().st_size//1024}K)")


if __name__ == "__main__":
    main()
