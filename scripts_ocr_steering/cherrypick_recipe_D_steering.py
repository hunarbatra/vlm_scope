#!/usr/bin/env python3
"""
Cherry-pick samples where Recipe D (BACKBONE multi-layer + γ·W_dec[F])
beats the baseline pt-448 on VSR test split.

Recipe D for L12_F2257 ("facing"):
  - BACKBONE: α=1.0 · unit(v_meanpool[L]) injected at each L in {4,6,9,11,12,13,14,15}
  - + γ=7.0 · W_dec[L12, F=2257] at L12 only
  → gave +15.62% on R(F)∩test (n=64) vs baseline 50.00%

This script:
  1. loads pt-448
  2. for each VSR test sample where the feature fires (R(F)∩test):
     a. baseline forward (no steering) → pred
     b. Recipe D steered forward → pred
  3. cherry-picks samples where steered=correct AND baseline=incorrect
  4. writes per-sample JSON + saves images + builds a 2-column collage

Output:
  /home/hbatra/vision-language-scope/docs/spatial_steering_examples.md
  /home/hbatra/vision-language-scope/docs/imgs/spatial_steering/*.{jpg,png}
"""
import os, sys, json, gc, hashlib, warnings, argparse
from pathlib import Path
from io import BytesIO
import torch
import requests
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

PT_MODEL = "google/paligemma2-3b-pt-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
ACTS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
OUT_DIR = Path("/home/hbatra/vision-language-scope/docs/imgs/spatial_steering")

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"

TRAIN_END = 8777
LAYERS = [4, 6, 9, 11, 12, 13, 14, 15]   # BACKBONE
ALPHA = 1.0
FEATURE = {"layer": 12, "feature": 2257, "key": "L12_F2257", "relations": ["facing"]}
GAMMA = 7.0
N_TARGET_WINS = 16  # split as 8 label=1 (No→Yes) + 8 label=0 (Yes→No) when possible


def _build_vsr_prompt(s):
    return ("Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
            f"Statement: {s.strip()}\nAnswer:")


def _get_yes_no_ids(tok):
    y, n = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        tt = tok.encode(t, add_special_tokens=False)
        if tt: y.add(tt[0])
    for t in [" No", "No", " no", "NO"]:
        tt = tok.encode(t, add_special_tokens=False)
        if tt: n.add(tt[0])
    overlap = y & n; y -= overlap; n -= overlap
    return y, n


def _predict(logits, yids, nids):
    p = torch.softmax(logits.float(), dim=-1)
    yp = p[list(yids)].sum().item() if yids else 1e-9
    np_ = p[list(nids)].sum().item() if nids else 1e-9
    pred = 1 if (yp / (yp + np_) if yp + np_ > 0 else 0.5) > 0.5 else 0
    conf = max(yp, np_) / max(yp + np_, 1e-9)
    return pred, conf


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
        img.save(cp, "JPEG"); return img
    except Exception: return None


def compute_meanpool_caa(vsr_labels, layer):
    pos = neg = None; pn = nn = 0
    for vi in range(TRAIN_END):
        p = MEANPOOL_DIR / f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if layer not in d: continue
        v = d[layer].float()
        if int(vsr_labels[vi]) == 1:
            pos = v.clone() if pos is None else pos + v; pn += 1
        else:
            neg = v.clone() if neg is None else neg + v; nn += 1
    if pos is None or neg is None: return None
    return pos / pn - neg / nn


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][fi].float()


def main():
    global FEATURE, GAMMA, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-eval", type=int, default=300, help="cap test samples to evaluate")
    ap.add_argument("--n-wins", type=int, default=N_TARGET_WINS)
    ap.add_argument("--layer", type=int, default=FEATURE["layer"])
    ap.add_argument("--feature-id", type=int, default=FEATURE["feature"])
    ap.add_argument("--relations", type=str, default=",".join(FEATURE["relations"]),
                    help="comma-separated relation list for this feature")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix for output files; default writes to OUT_DIR root")
    args = ap.parse_args()
    # Override globals from CLI so the rest of the script uses them
    FEATURE = {
        "layer": args.layer, "feature": args.feature_id,
        "key": f"L{args.layer}_F{args.feature_id}",
        "relations": [r.strip() for r in args.relations.split(",") if r.strip()],
    }
    GAMMA = args.gamma
    if args.out_suffix:
        OUT_DIR = OUT_DIR.parent / f"spatial_steering_{args.out_suffix}"

    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"[INFO] Loading VSR (all splits)...", flush=True)
    vsr = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", split=s)
        for s in ["train", "validation", "test"]
    ])
    vsr_labels = [int(vsr[vi].get("label", 0)) for vi in range(len(vsr))]
    print(f"[INFO] VSR total: {len(vsr)} samples", flush=True)

    # R(F): samples where this feature fires (from cached acts)
    acts_path = ACTS_DIR / f"acts_{FEATURE['key']}.json"
    acts = json.load(open(acts_path))
    rf_indices = sorted(int(vi) for vi in acts.get("acts", {}).keys())
    rf_test = [vi for vi in rf_indices if vi >= TRAIN_END]
    print(f"[INFO] R(F)∩test for {FEATURE['key']}: {len(rf_test)} samples (relations={FEATURE['relations']})", flush=True)

    # Cap evaluation set
    eval_indices = rf_test[:args.max_eval]
    print(f"[INFO] Evaluating on first {len(eval_indices)} R(F)∩test samples", flush=True)

    # Compute CAA vectors at each backbone layer
    print(f"[INFO] Building CAA meanpool vectors at layers {LAYERS}...", flush=True)
    caa = {}
    for L in LAYERS:
        v = compute_meanpool_caa(vsr_labels, L)
        if v is None:
            print(f"  [WARN] no CAA for L{L}"); continue
        u = v / v.norm().clamp(min=1e-8)
        caa[L] = u
        print(f"  L{L}: ||v||={v.norm():.3f}  unit_norm built", flush=True)

    # W_dec for L12_F2257
    w = _load_wdec(FEATURE["layer"], FEATURE["feature"])
    if w is None:
        print("[ERROR] W_dec not loaded"); return
    w_unit = w / w.norm().clamp(min=1e-8)
    print(f"[INFO] W_dec[L{FEATURE['layer']}/F{FEATURE['feature']}] loaded, ||W||={w.norm():.3f}", flush=True)

    # Load model
    print(f"[INFO] Loading {PT_MODEL} on {device}...", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)
    dtype = next(model.parameters()).dtype

    # Build per-layer α-scaled steer vectors
    sv = {}
    for L, u in caa.items():
        sv[L] = (u * ALPHA).to(dtype).to(device)
    # Also augment L12 with γ·W_dec
    if FEATURE["layer"] in sv:
        sv[FEATURE["layer"]] = (sv[FEATURE["layer"]] + GAMMA * w_unit.to(dtype).to(device))
        print(f"[INFO] L{FEATURE['layer']} steer aug: + γ={GAMMA}·W_dec[F={FEATURE['feature']}]", flush=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    img_end_ref = [0]

    def make_hook(s):
        def f(m, i, o):
            ie = img_end_ref[0]
            h = o[0] if isinstance(o, tuple) else o
            h[0, ie:] = h[0, ie:] + s.unsqueeze(0)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        return f

    wins = []
    losses = []
    n_correct_base = n_correct_steer = 0
    print(f"\n[EVAL] running baseline + Recipe D on {len(eval_indices)} samples...", flush=True)

    for i, vi in enumerate(eval_indices):
        ex = vsr[vi]
        img = _load_image(ex)
        if img is None: continue
        caption = str(ex.get("caption", "")).strip()
        label = int(ex.get("label", 0))
        prompt = _build_vsr_prompt(caption)

        try:
            iids, attn, pv = process_vlm_inputs(img, prompt, proc, model, device=device)
            _, img_end_ref[0] = get_image_token_positions(iids)

            # Baseline
            with torch.no_grad():
                out_b = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
            pred_b, conf_b = _predict(out_b.logits[0, -1, :], yids, nids)

            # Steered (Recipe D)
            hooks = []
            for L, s in sv.items():
                hooks.append(model.model.language_model.layers[L].register_forward_hook(make_hook(s)))
            with torch.no_grad():
                out_s = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
            for h in hooks:
                try: h.remove()
                except: pass
            pred_s, conf_s = _predict(out_s.logits[0, -1, :], yids, nids)
        except Exception as e:
            print(f"  [vi={vi}] error: {e}", flush=True); continue

        ok_b = int(pred_b == label)
        ok_s = int(pred_s == label)
        n_correct_base += ok_b
        n_correct_steer += ok_s

        sample_out = {
            "vi": vi, "caption": caption, "label": label,
            "relation": ex.get("relation", ""),
            "image_link": ex.get("image_link", ""),
            "baseline_pred": pred_b, "baseline_conf": conf_b, "baseline_correct": ok_b,
            "steered_pred": pred_s, "steered_conf": conf_s, "steered_correct": ok_s,
        }
        if ok_s and not ok_b:
            wins.append(sample_out)
        elif ok_b and not ok_s:
            losses.append(sample_out)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(eval_indices)}]  base_acc={100*n_correct_base/(i+1):.1f}%  "
                  f"steer_acc={100*n_correct_steer/(i+1):.1f}%  wins={len(wins)} losses={len(losses)}",
                  flush=True)

    n = len(eval_indices)
    print(f"\n[FINAL] base={100*n_correct_base/n:.2f}%  steer={100*n_correct_steer/n:.2f}%  "
          f"Δ={100*(n_correct_steer-n_correct_base)/n:+.2f}pp  wins={len(wins)} losses={len(losses)}", flush=True)

    # Cherry-pick top wins, BALANCED across label=1 (No→Yes) and label=0 (Yes→No)
    def _margin(r): return r["steered_conf"] - (1 - r["baseline_conf"])
    wins_pos = sorted([w for w in wins if w["label"] == 1], key=lambda r: -_margin(r))
    wins_neg = sorted([w for w in wins if w["label"] == 0], key=lambda r: -_margin(r))
    half = max(1, args.n_wins // 2)
    chosen_pos = wins_pos[:half]
    chosen_neg = wins_neg[:half]
    # If one class is short, top up from the other
    deficit = args.n_wins - len(chosen_pos) - len(chosen_neg)
    if deficit > 0:
        if len(wins_pos) > len(chosen_pos):
            chosen_pos += wins_pos[len(chosen_pos):len(chosen_pos)+deficit]
        elif len(wins_neg) > len(chosen_neg):
            chosen_neg += wins_neg[len(chosen_neg):len(chosen_neg)+deficit]
    # Interleave so collage alternates Yes/No examples
    chosen = []
    for a, b in zip(chosen_pos, chosen_neg):
        chosen.append(a); chosen.append(b)
    leftover = chosen_pos[len(chosen_neg):] + chosen_neg[len(chosen_pos):]
    chosen += leftover
    print(f"\n[CHERRY] selected {len(chosen)} wins  "
          f"({len(chosen_pos)} No→Yes, {len(chosen_neg)} Yes→No)  "
          f"available pool: {len(wins_pos)} pos / {len(wins_neg)} neg", flush=True)

    # Save images + per-sample JSON
    saved_records = []
    for j, r in enumerate(chosen):
        ex = vsr[r["vi"]]
        img = _load_image(ex)
        if img is None: continue
        img_path = OUT_DIR / f"win_{j+1:02d}_vi{r['vi']}.jpg"
        img.save(img_path, "JPEG", quality=92)
        r["image_path"] = str(img_path.relative_to(Path('/home/hbatra/vision-language-scope')))
        saved_records.append(r)
        print(f"  win_{j+1:02d}: vi={r['vi']} '{r['caption']}' "
              f"label={r['label']} base={r['baseline_pred']} steer={r['steered_pred']}", flush=True)

    json_path = OUT_DIR / "cherrypick_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "feature": FEATURE,
            "recipe_D": {"layers": LAYERS, "alpha": ALPHA, "gamma": GAMMA},
            "n_eval": n, "n_correct_base": n_correct_base, "n_correct_steer": n_correct_steer,
            "n_wins": len(wins), "n_losses": len(losses),
            "wins": saved_records,
            "losses_count_only": len(losses),
        }, f, indent=2)
    print(f"\n[INFO] cherrypick JSON: {json_path}", flush=True)
    print(f"[INFO] Run build_steering_collage.py next to render the comparison figure.", flush=True)


if __name__ == "__main__":
    main()
