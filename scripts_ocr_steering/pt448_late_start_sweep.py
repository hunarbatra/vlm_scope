#!/usr/bin/env python3
"""
Late-layer injection sweep — test all 8 features with start_layer fixed to 15.

Hypothesis from L15/F220 result: L15 peaks at +2.09% (new record) because its natural
start_layer=15 restricts injection to layers 15-25 (late reasoning layers). Features
with earlier natural start layers (L4, L6, L9, L11, L12, L14) broadcast through all
or most layers, causing interference with early feature formation on unrelated samples.

This script re-runs all 8 features with start_layer=15, α ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
on full VSR. Compares directly to their natural-start performance.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_late_start_sweep/
Usage:  CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_late_start_sweep.py
"""

import os, sys, json, warnings, math
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_PT    = "google/paligemma2-3b-pt-448"
N_LAYERS    = 26
LATE_START  = 15   # fixed injection window: layers 15-25
CAA_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_late_start_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# All 8 features — natural start shown for reference
FEATURES = {
    "L4_F14233":  {"layer": 4,  "feature": 14233, "natural_start": 0},
    "L14_F10561": {"layer": 14, "feature": 10561, "natural_start": 0},
    "L12_F2257":  {"layer": 12, "feature": 2257,  "natural_start": 1},
    "L15_F220":   {"layer": 15, "feature": 220,   "natural_start": 15},  # control: same as natural
    "L11_F12278": {"layer": 11, "feature": 12278, "natural_start": 5},
    "L9_F387":    {"layer": 9,  "feature": 387,   "natural_start": 1},
    "L6_F7539":   {"layer": 6,  "feature": 7539,  "natural_start": 1},
    "L9_F7540":   {"layer": 9,  "feature": 7540,  "natural_start": 9},
}

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# Previously known best Δ at natural start (for comparison column)
NATURAL_BEST = {
    "L4_F14233":  1.88, "L14_F10561": 1.88, "L12_F2257":  1.72,
    "L15_F220":   2.09, "L11_F12278": 1.90, "L9_F387":    0.42,
    "L6_F7539":   1.90, "L9_F7540":   0.00,
}


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]: toks = tok.encode(t, add_special_tokens=False); yes_ids.update(toks[:1] if toks else [])
    for t in [" No","No"," no","NO"]:  toks = tok.encode(t, add_special_tokens=False); no_ids.update(toks[:1] if toks else [])
    ov = yes_ids & no_ids; yes_ids -= ov; no_ids -= ov
    return yes_ids, no_ids

def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n; p = max(y/d if d > 0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1-p, 1e-7))

def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = __import__("hashlib").md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    if cp.exists():
        try: return Image.open(cp).convert("RGB")
        except: pass
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        cp.parent.mkdir(parents=True, exist_ok=True); img.save(cp)
        return img
    except: return None


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    indices = list(range(N))

    print("[INFO] Loading pt-448...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    print("[INFO] Loading CAA vectors (all features)...", flush=True)
    all_vecs = {}
    for feat_key, cfg in FEATURES.items():
        layer = cfg["layer"]
        feature = cfg["feature"]
        path = CAA_DIR / f"caa_L{layer}_F{feature}.pt"
        if not path.exists():
            print(f"  [WARN] Missing {path}", flush=True); continue
        saved = torch.load(path, map_location="cpu")
        caa_data = saved.get("caa_data", {})
        vecs = {}
        for l, ld in caa_data.items():
            v = ld.get("v_caa_norm")
            if v is not None:
                vecs[int(l)] = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)
        all_vecs[feat_key] = vecs
        print(f"  [LOADED] {feat_key}", flush=True)

    # Shared baseline (run once)
    result_path_base = OUT_DIR / "baseline.json"
    if result_path_base.exists():
        with open(result_path_base) as f:
            base_res = json.load(f)
        base_acc = base_res["base_acc"]
        print(f"[SKIP] Baseline already computed: {base_acc:.2f}%", flush=True)
    else:
        print("\n[INFO] Running baseline...", flush=True)
        correct = total = 0; margins = []
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            lbl = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, _build_vsr_prompt(str(ex.get("caption", ""))),
                    processor_pt, model_pt, device=device)
                with torch.inference_mode():
                    out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, m = _pm(out.logits[0, -1, :], yes_ids, no_ids)
                margins.append(m if lbl == 1 else -m)
            except Exception: pred = 0; margins.append(0.0)
            total += 1; correct += (pred == lbl)
        base_acc = correct / max(total, 1) * 100
        base_mg = sum(margins) / max(len(margins), 1)
        print(f"  BASE: {base_acc:.2f}%  N={total}", flush=True)
        with open(result_path_base, "w") as f:
            json.dump({"base_acc": base_acc, "base_margin": base_mg, "n": total}, f)

    # Per-feature late-start sweep
    summary = {}
    for feat_key, cfg in FEATURES.items():
        result_path = OUT_DIR / f"late_start_{feat_key}.json"
        if result_path.exists():
            print(f"[SKIP] {feat_key} already done", flush=True)
            with open(result_path) as f:
                r = json.load(f)
            summary[feat_key] = r
            continue

        vecs = all_vecs.get(feat_key)
        if vecs is None:
            print(f"[SKIP] {feat_key} no vectors", flush=True); continue

        natural_start = cfg["natural_start"]
        natural_best = NATURAL_BEST.get(feat_key, 0)
        print(f"\n[FEATURE] {feat_key}  natural_start={natural_start}  late_start={LATE_START}", flush=True)
        print(f"  Natural-start best known: +{natural_best:.2f}%", flush=True)

        alpha_results = {}
        best_delta = -999; best_alpha = None
        for alpha in ALPHAS:
            correct = total = 0; margins = []
            for vi in indices:
                ex = vsr_all[vi]; img = _load_image(ex)
                if img is None: continue
                lbl = int(ex.get("label", 0))
                try:
                    iids, attn, pv = process_vlm_inputs(
                        img, _build_vsr_prompt(str(ex.get("caption", ""))),
                        processor_pt, model_pt, device=device)
                    _, img_end = get_image_token_positions(iids)
                    with nns_pt.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                        for l in range(LATE_START, N_LAYERS):
                            if l not in vecs: continue
                            v_l = vecs[l]
                            v_col = v_l.unsqueeze(1)
                            lo = nns_pt.model.language_model.layers[l].output[0][0, img_end:]
                            ones = (lo @ v_col) * 0.0 + 1.0
                            lo += alpha * ones * v_l
                        logits_s = nns_pt.output.logits.save()
                    pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                    margins.append(m if lbl == 1 else -m)
                except Exception: pred = 0; margins.append(0.0)
                total += 1; correct += (pred == lbl)
            acc = correct / max(total, 1) * 100
            da = acc - base_acc
            mg = sum(margins) / max(len(margins), 1)
            marker = " *** BEST ***" if da > best_delta else ""
            if da > best_delta: best_delta = da; best_alpha = alpha
            vs_natural = da - natural_best
            print(f"  α={alpha}: {acc:.2f}% (Δ={da:+.2f}%  vs_natural={vs_natural:+.2f}%) mg={mg:.3f}{marker}", flush=True)
            alpha_results[str(alpha)] = {"acc": acc, "delta_acc": da, "margin": mg}

        result = {
            "feat_key": feat_key, "natural_start": natural_start, "late_start": LATE_START,
            "natural_best_known": natural_best, "base_acc": base_acc,
            "best_alpha": best_alpha, "best_delta": best_delta,
            "best_acc": base_acc + best_delta, "alphas": alpha_results,
        }
        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        summary[feat_key] = result
        print(f"  [SAVED] {result_path}  best={best_delta:+.2f}% vs natural={best_delta-natural_best:+.2f}pp", flush=True)

    # Print summary table
    print("\n" + "="*70, flush=True)
    print(f"{'Feature':14s}  {'Nat.start':9s}  {'Nat.best':8s}  {'Late.best':9s}  {'Δ vs nat':8s}", flush=True)
    print("-"*70, flush=True)
    for fk, r in sorted(summary.items()):
        nat = r.get("natural_best_known", 0)
        lb = r.get("best_delta", 0)
        print(f"{fk:14s}  {r.get('natural_start'):9d}  {nat:+8.2f}%  {lb:+9.2f}%  {lb-nat:+8.2f}pp", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
