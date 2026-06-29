#!/usr/bin/env python3
"""
Full-VSR caa_sae_down alpha sweep for remaining features not in universal_sweep:
L11/F12278, L9/F387, L14/F10561, L13/F15219, L9/F7540

Each feature tested at alpha sweep [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
on FULL VSR dataset (N=10972) to find which single feature injection on full VSR works best.

Context: individual-optimal alpha applied to full VSR always collapses (48.77%).
This sweep finds whether a much smaller alpha can give positive full-VSR gains.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_universal_sweep/fullvsr_remaining.json

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 pt448_caa_fullvsr_remaining.py
"""

import os, sys, json, hashlib, warnings, math
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_PT    = "google/paligemma2-3b-pt-448"
N_LAYERS    = 26
CAA_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_universal_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Features NOT in GPU5's sweep, using caa_sae_down (start_layer=1 for all except L9/F7540 start=9)
FEATURES = [
    # (layer_idx, feature_idx, start_layer, alphas_to_test)
    (11, 12278, 5,  [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    (9,  387,   1,  [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    (14, 10561, 0,  [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    (13, 15219, 0,  [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
    (9,  7540,  9,  [0.01, 0.05, 0.1, 0.25, 0.5]),  # known harmful; test very low alpha
]


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No","No"," no","NO"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
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
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB"); img.save(cp, "JPEG")
        return img
    except Exception: return None


def eval_full_vsr(all_indices, vsr_all, layer_vecs, alpha, start_layer,
                  nns_model, yes_ids, no_ids, processor, model_dtype, device):
    from utils import process_vlm_inputs, get_image_token_positions
    correct = total = 0; margins = []
    for vi in all_indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        try:
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                 processor, nns_model._module, device=device)
            _, img_end = get_image_token_positions(iids)
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in range(start_layer, N_LAYERS):
                    v_l = layer_vecs[l]
                    v_col = v_l.unsqueeze(1)
                    lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                    ones = (lo @ v_col) * 0.0 + 1.0
                    lo += alpha * ones * v_l
                logits_s = nns_model.output.logits.save()
            pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
            margins.append(m if label == 1 else -m)
        except Exception: pred = 0; margins.append(0.0)
        total += 1; correct += (pred == label)
        if total % 2000 == 0: print(f"  {total}", flush=True)
    acc = correct / max(total, 1) * 100
    mg = sum(margins) / max(len(margins), 1)
    return acc, mg, total


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    result_path = OUT_DIR / "fullvsr_remaining.json"
    if result_path.exists():
        print("[SKIP] Results already exist", flush=True)
        with open(result_path) as f: results = json.load(f)
        _print_summary(results)
        return

    print(f"[INFO] Loading {MODEL_PT}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PT)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    all_indices = list(range(len(vsr_all)))

    # Load feature vectors
    feature_vecs = {}
    for layer_idx, feature_idx, start_layer, _ in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists():
            print(f"[SKIP] No CAA for {key}", flush=True); continue
        caa_data = torch.load(caa_path)["caa_data"]
        layer_vecs = {}
        for l in range(N_LAYERS):
            if l in caa_data:
                layer_vecs[l] = caa_data[l]["v_caa_norm"].to(model_dtype).to(device)
            else:
                layer_vecs[l] = caa_data[layer_idx]["v_caa_norm"].to(model_dtype).to(device)
        feature_vecs[key] = layer_vecs
        print(f"[LOADED] {key}", flush=True)

    # Baseline
    print("\n[BASELINE] Full VSR...", flush=True)
    correct = total = 0; margins = []
    for vi in all_indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        try:
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                 processor, model_raw, device=device)
            with torch.inference_mode():
                out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
            pred, m = _pm(out.logits[0, -1, :], yes_ids, no_ids)
            margins.append(m if label == 1 else -m)
        except Exception: pred = 0; margins.append(0.0)
        total += 1; correct += (pred == label)
        if total % 2000 == 0: print(f"  baseline {total}", flush=True)
    base_acc = correct / max(total, 1) * 100
    base_mg = sum(margins) / max(len(margins), 1)
    print(f"[BASELINE] {base_acc:.2f}% margin={base_mg:.3f} N={total}", flush=True)

    results = {"baseline": {"acc": base_acc, "margin": base_mg, "n": total}, "features": {}}

    for layer_idx, feature_idx, start_layer, alphas in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        if key not in feature_vecs: continue
        layer_vecs = feature_vecs[key]
        print(f"\n[FEATURE] {key} start={start_layer}", flush=True)
        results["features"][key] = {"start": start_layer, "alphas": {}}
        best_da = -999; best_a = None
        for alpha in alphas:
            acc, mg, n = eval_full_vsr(all_indices, vsr_all, layer_vecs, alpha, start_layer,
                                        nns_model, yes_ids, no_ids, processor, model_dtype, device)
            da = acc - base_acc
            marker = " *** NEW BEST ***" if da > best_da else ""
            if da > best_da: best_da = da; best_a = alpha
            print(f"  alpha={alpha}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f}{marker}", flush=True)
            results["features"][key]["alphas"][str(alpha)] = {"acc": acc, "delta_acc": da, "margin": mg}
        results["features"][key]["best_alpha"] = best_a
        results["features"][key]["best_delta"] = best_da
        print(f"  BEST: alpha={best_a} Δ={best_da:+.2f}%", flush=True)

    with open(result_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved to {result_path}", flush=True)
    _print_summary(results)


def _print_summary(results):
    base = results["baseline"]["acc"]
    print(f"\n{'='*60}")
    print("Full-VSR Remaining Features Sweep Summary")
    print(f"{'='*60}")
    print(f"Baseline: {base:.2f}%")
    for key, r in results.get("features", {}).items():
        bd = r.get("best_delta"); ba = r.get("best_alpha")
        if bd is not None:
            print(f"  {key}: best Δ={bd:+.2f}% @ alpha={ba}")


if __name__ == "__main__":
    main()
