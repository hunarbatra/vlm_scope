#!/usr/bin/env python3
"""
Final verification: run each feature at its confirmed optimal (alpha, start) configuration.

Based on all joint sweep results, the confirmed optimal configs are:
  L4/F14233:  α=1.0,  start=0  → +15.38% "ahead of"
  L14/F10561: α=2.0,  start=0  → +10.75% "close to"
  L12/F2257:  α=1.0,  start=1  → +9.80%  "facing"
  L15/F220:   α=0.75, start=15 → +7.96%  "across from", "at the left side of"
  L11/F12278: α=0.5,  start=5  → +6.32%  "touching"
  L9/F387:    α=0.5,  start=1  → +4.17%  "at the right side of"  [NEW RECORD]
  L9/F7540:   α=0.25, start=9  → +2.86%  "consists of"
  L13/F15219: W_dec layer=3 α=30 → +2.68%  "behind"  (CAA failed to beat this)
  L6/F7539:   α=0.75, start=12 → +2.17%  "left of", "right of"  [to be updated]
  L11/F9639:  barely steerable → +0.64%  "in", "inside", "on"

This script:
1. Re-evaluates each feature at its confirmed optimal config (using per-layer CAA vectors)
2. Reports clean final numbers with confidence-relevant sample counts
3. Tests the NEGATIVE direction (negative alpha) for the top 5 features to confirm directionality
4. Optionally tests combining two non-overlapping features simultaneously

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_final_verify/

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 pt448_caa_optimal_verify.py
    CUDA_VISIBLE_DEVICES=3 GPU_FEATURES=L4,L14,L12 python3 pt448_caa_optimal_verify.py
"""

import os, sys, json, gc, hashlib, warnings, math
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_final_verify")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Confirmed optimal configs from joint sweeps
# (layer_idx, feature_idx, relations, alpha, start_layer, best_delta_known)
FEATURES_OPTIMAL = [
    (4,  14233, ["ahead of"],                            1.0,  0,  +15.38),
    (14, 10561, ["close to"],                            2.0,  0,  +10.75),
    (12, 2257,  ["facing"],                              1.0,  1,   +9.80),
    (15, 220,   ["across from", "at the left side of"],  0.75, 15,  +7.96),
    (11, 12278, ["touching"],                            0.5,  5,   +6.32),
    (9,  387,   ["at the right side of"],                0.5,  1,   +4.17),
    (9,  7540,  ["consists of"],                         0.25, 9,   +2.86),
    (6,  7539,  ["left of", "right of"],                 1.5,  1,   +3.41),
    (11, 9639,  ["in", "inside", "on"],                  None, None, +0.64),  # skip (barely steerable)
]

# Filter by GPU_FEATURES env var
_RUN = os.environ.get("GPU_FEATURES", "ALL")
if _RUN != "ALL":
    _keys = _RUN.split(",")
    FEATURES_OPTIMAL = [f for f in FEATURES_OPTIMAL if any(f"L{f[0]}" in k for k in _keys)]

# Also test negative direction for top features to verify directionality
NEGATIVE_ALPHA_TEST = [
    (4,  14233, ["ahead of"],                           -1.0, 0),
    (12, 2257,  ["facing"],                             -1.0, 1),
    (15, 220,   ["across from", "at the left side of"], -0.75, 15),
    (9,  387,   ["at the right side of"],               -0.5, 1),
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


def run_steered(indices, vsr_all, layer_vecs, alpha, start_layer, nns_model,
                 yes_ids, no_ids, processor, base_module, device):
    inj_layers = list(range(start_layer, N_LAYERS))
    correct = total = 0; margins = []
    for vi in indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        try:
            from utils import process_vlm_inputs, get_image_token_positions
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                 processor, base_module, device=device)
            _, img_end = get_image_token_positions(iids)
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in inj_layers:
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
    acc = correct / max(total, 1) * 100
    mg = sum(margins) / max(len(margins), 1)
    return acc, mg, total


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

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
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations, alpha, start_layer, known_best in FEATURES_OPTIMAL:
        if alpha is None:
            print(f"[SKIP L{layer_idx}/F{feature_idx}] no CAA config (barely steerable)", flush=True)
            continue

        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"verify_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key} already verified", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists():
            print(f"[SKIP - no CAA] {key}", flush=True)
            continue

        caa_saved = torch.load(caa_path)
        caa_data = caa_saved["caa_data"]

        layer_vecs = {}
        for l in range(N_LAYERS):
            if l in caa_data:
                layer_vecs[l] = caa_data[l]["v_caa_norm"].to(model_dtype).to(device)
            else:
                layer_vecs[l] = caa_data[layer_idx]["v_caa_norm"].to(model_dtype).to(device)

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            from utils import process_vlm_inputs
            print(f"[BASE] [{rel_key}] N={len(indices)}...", flush=True)
            correct = total = 0; margins = []
            for vi in indices:
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
            acc = correct / max(total, 1) * 100
            mg = sum(margins) / max(len(margins), 1)
            baseline_cache[rel_key] = (acc, mg, total)
            print(f"[BASE] {acc:.2f}% margin={mg:.3f}", flush=True)
        base_acc, base_mg, n_total = baseline_cache[rel_key]

        print(f"\n[{key}] Optimal config: α={alpha}, start={start_layer} | known={known_best:+.2f}%", flush=True)

        # Run positive direction
        acc_pos, mg_pos, n_eval = run_steered(
            indices, vsr_all, layer_vecs, alpha, start_layer,
            nns_model, yes_ids, no_ids, processor, nns_model._module, device
        )
        da_pos = acc_pos - base_acc
        print(f"  POSITIVE: {acc_pos:.2f}% (Δ={da_pos:+.2f}%) margin={mg_pos:.3f}", flush=True)

        # Check if negative test applies
        neg_config = next((c for c in NEGATIVE_ALPHA_TEST if c[0] == layer_idx and c[1] == feature_idx), None)
        neg_result = None
        if neg_config:
            neg_alpha, neg_start = neg_config[3], neg_config[4]
            acc_neg, mg_neg, _ = run_steered(
                indices, vsr_all, layer_vecs, neg_alpha, neg_start,
                nns_model, yes_ids, no_ids, processor, nns_model._module, device
            )
            da_neg = acc_neg - base_acc
            print(f"  NEGATIVE: {acc_neg:.2f}% (Δ={da_neg:+.2f}%) margin={mg_neg:.3f} [α={neg_alpha}]", flush=True)
            neg_result = {"alpha": neg_alpha, "start": neg_start, "acc": acc_neg, "delta_acc": da_neg,
                          "margin": mg_neg, "delta_margin": mg_neg - base_mg}

        result = {
            "layer_idx": layer_idx, "feature_idx": feature_idx, "relations": relations,
            "known_best": known_best,
            "optimal_alpha": alpha, "optimal_start": start_layer,
            "n_samples": len(indices), "n_evaluated": n_eval,
            "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "positive": {"acc": acc_pos, "delta_acc": da_pos, "margin": mg_pos,
                         "delta_margin": mg_pos - base_mg},
            "negative": neg_result
        }

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        del layer_vecs; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("Final Verification — Summary Table")
    print(f"{'='*100}")
    print(f"{'Feature':<20} {'Relation':<35} {'Base':>7} {'Known':>8} {'Verified':>10} {'Neg':>8}")
    print("-" * 100)
    for r in all_results:
        key = f"L{r['layer_idx']}/F{r['feature_idx']}"
        neg_str = f"{r['negative']['delta_acc']:+.2f}%" if r.get('negative') else "N/A"
        print(f"{key:<20} {str(r['relations']):<35} {r['baseline_vsr_acc']:>7.2f}% "
              f"{r['known_best']:>+8.2f}% {r['positive']['delta_acc']:>+10.2f}% {neg_str:>8}")


if __name__ == "__main__":
    main()
