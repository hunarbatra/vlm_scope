#!/usr/bin/env python3
"""
Scaled combined injection sweep.

The combined all-8 injection may over-steer because each feature uses its
individual optimal alpha (calibrated for solo injection). When combining, the
total steering magnitude is the sum of all features, which may be too large.

This script tests alpha scaling factors [0.1, 0.25, 0.5, 0.75, 1.0] applied
uniformly to all features' alphas in the all-8 combined injection.

For each scale factor s:
  Effective alpha[i] = s * optimal_alpha[i]

Also tests two subsets:
  - Top-3 (L4+L14+L12): high-confidence features only
  - Top-5 (L4+L14+L12+L15+L11): original combined set

All evaluated on full VSR dataset.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_scaled_combined/

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 pt448_caa_scaled_combined.py
"""

import os, sys, json, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_PT    = "google/paligemma2-3b-pt-448"
N_LAYERS    = 26
CAA_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_scaled_combined")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALL8 = [
    (4,  14233, 1.0,  0),
    (14, 10561, 2.0,  0),
    (12, 2257,  1.0,  1),
    (15, 220,   0.75, 15),
    (11, 12278, 0.5,  5),
    (9,  387,   0.5,  1),
    (6,  7539,  1.5,  1),
    (9,  7540,  0.25, 9),
]

TOP3_KEYS = {(4,14233),(14,10561),(12,2257)}
TOP5_KEYS = {(4,14233),(14,10561),(12,2257),(15,220),(11,12278)}

SCALE_FACTORS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


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


def eval_combined(indices, vsr_all, injection_configs, nns_model, yes_ids, no_ids,
                  processor, base_module, model_dtype, device):
    from utils import process_vlm_inputs, get_image_token_positions
    correct = total = 0; margins = []
    for vi in indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        try:
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                 processor, base_module, device=device)
            _, img_end = get_image_token_positions(iids)
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for layer_vecs, alpha, start_layer in injection_configs:
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
        if total % 1000 == 0: print(f"  {total}", flush=True)
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

    result_path = OUT_DIR / "scaled_combined_results.json"
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

    # Load all feature vectors
    feature_vecs = {}
    for layer_idx, feature_idx, alpha_opt, start_layer in ALL8:
        key = f"L{layer_idx}_F{feature_idx}"
        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists(): continue
        caa_data = torch.load(caa_path)["caa_data"]
        layer_vecs = {}
        for l in range(N_LAYERS):
            if l in caa_data:
                layer_vecs[l] = caa_data[l]["v_caa_norm"].to(model_dtype).to(device)
            else:
                layer_vecs[l] = caa_data[layer_idx]["v_caa_norm"].to(model_dtype).to(device)
        feature_vecs[key] = (layer_vecs, alpha_opt, start_layer)
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
        if total % 1000 == 0: print(f"  baseline {total}", flush=True)
    base_acc = correct / max(total, 1) * 100
    base_mg = sum(margins) / max(len(margins), 1)
    print(f"[BASELINE] {base_acc:.2f}% margin={base_mg:.3f} N={total}", flush=True)

    results = {"baseline": {"acc": base_acc, "margin": base_mg, "n": total}, "sweeps": {}}

    # Sweep scale factors for three groups
    groups = {
        "top3": [(k, v) for (li, fi, *_), (k, v) in zip(ALL8, [(f"L{li}_F{fi}", feature_vecs.get(f"L{li}_F{fi}")) for li,fi,_,_ in ALL8]) if (li,fi) in TOP3_KEYS and v is not None],
        "top5": [(k, v) for (li, fi, *_), (k, v) in zip(ALL8, [(f"L{li}_F{fi}", feature_vecs.get(f"L{li}_F{fi}")) for li,fi,_,_ in ALL8]) if (li,fi) in TOP5_KEYS and v is not None],
        "all8": [(f"L{li}_F{fi}", feature_vecs.get(f"L{li}_F{fi}")) for li,fi,_,_ in ALL8 if feature_vecs.get(f"L{li}_F{fi}") is not None],
    }

    for group_name, group_features in groups.items():
        print(f"\n[GROUP] {group_name} ({len(group_features)} features)", flush=True)
        results["sweeps"][group_name] = {}
        best_da = -999; best_scale = None
        for scale in SCALE_FACTORS:
            configs = [(lv, alpha_opt * scale, start_layer) for (k, (lv, alpha_opt, start_layer)) in group_features]
            acc_s, mg_s, n_eval = eval_combined(all_indices, vsr_all, configs, nns_model, yes_ids, no_ids,
                                                 processor, nns_model._module, model_dtype, device)
            da = acc_s - base_acc
            marker = " *** NEW BEST ***" if da > best_da else ""
            if da > best_da: best_da = da; best_scale = scale
            print(f"  scale={scale:.2f}: {acc_s:.2f}% (Δ={da:+.2f}%) margin={mg_s:.3f}{marker}", flush=True)
            results["sweeps"][group_name][str(scale)] = {"acc": acc_s, "delta_acc": da, "margin": mg_s}
        results["sweeps"][group_name]["best_scale"] = best_scale
        results["sweeps"][group_name]["best_delta"] = best_da
        print(f"  BEST: scale={best_scale} Δ={best_da:+.2f}%", flush=True)

    with open(result_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved to {result_path}", flush=True)
    _print_summary(results)


def _print_summary(results):
    base = results["baseline"]["acc"]
    print(f"\n{'='*70}")
    print("Scaled Combined Injection Summary")
    print(f"{'='*70}")
    print(f"Baseline: {base:.2f}%")
    for group, data in results.get("sweeps", {}).items():
        best_s = data.get("best_scale"); best_d = data.get("best_delta")
        if best_s is not None:
            print(f"  {group}: best scale={best_s} → Δ={best_d:+.2f}%")


if __name__ == "__main__":
    main()
