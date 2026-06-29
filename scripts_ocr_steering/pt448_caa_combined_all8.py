#!/usr/bin/env python3
"""
Combined multi-feature CAA injection — ALL 8 steerable features.

Extends pt448_caa_combined_injection.py (top-5) to include all 8 verified features.

All 8 confirmed optimal configs:
  L4/F14233:  α=1.0,  start=0  (+15.38% on "ahead of",        N=39)
  L14/F10561: α=2.0,  start=0  (+10.75% on "close to",        N=93)
  L12/F2257:  α=1.0,  start=1  (+9.80%  on "facing",          N=306)
  L15/F220:   α=0.75, start=15 (+7.96%  on "across from/left",N=515)
  L11/F12278: α=0.5,  start=5  (+6.32%  on "touching",        N=1281)
  L9/F387:    α=0.5,  start=1  (+4.17%  on "right side of",   N=480)
  L6/F7539:   α=1.5,  start=1  (+3.41%  on "left/right of",   N=323)
  L9/F7540:   α=0.25, start=9  (+2.86%  on "consists of",     N=35)

Tests:
1. Baseline on full VSR
2. Each feature on its own relation subset (verify individual deltas)
3. All 8 simultaneously on full VSR (key combined injection test)
4. Top-3 (L4+L14+L12) combined on full VSR
5. Top-5 (add L15+L11) combined on full VSR
6. Cross-relation: each feature applied to full VSR

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_combined_all8/

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 pt448_caa_combined_all8.py
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_combined_all8")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

FEATURES = [
    # (layer_idx, feature_idx, optimal_relations, alpha, start_layer)
    (4,  14233, ["ahead of"],                            1.0,  0),
    (14, 10561, ["close to"],                            2.0,  0),
    (12, 2257,  ["facing"],                              1.0,  1),
    (15, 220,   ["across from", "at the left side of"],  0.75, 15),
    (11, 12278, ["touching"],                            0.5,  5),
    (9,  387,   ["at the right side of"],                0.5,  1),
    (6,  7539,  ["left of", "right of"],                 1.5,  1),
    (9,  7540,  ["consists of"],                         0.25, 9),
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


def evaluate_dataset(indices, vsr_all, injection_configs, nns_model, yes_ids, no_ids,
                     processor, base_module, model_dtype, device):
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
    acc = correct / max(total, 1) * 100
    mg = sum(margins) / max(len(margins), 1)
    return acc, mg, total


def run_baseline(indices, vsr_all, processor, model_raw, yes_ids, no_ids, device):
    from utils import process_vlm_inputs
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
        if total % 500 == 0: print(f"  baseline {total}", flush=True)
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

    result_path = OUT_DIR / "combined_all8_results.json"
    if result_path.exists():
        print("[SKIP] Results already exist, loading...", flush=True)
        with open(result_path) as f:
            results = json.load(f)
        print(json.dumps(results, indent=2))
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
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)
    all_indices = list(range(len(vsr_all)))

    # Load all CAA vectors
    feature_configs = []
    for layer_idx, feature_idx, relations, alpha, start_layer in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
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
        feature_configs.append((key, layer_idx, feature_idx, relations, alpha, start_layer, layer_vecs))
        print(f"[LOADED] {key}: α={alpha}, start={start_layer}", flush=True)

    results = {}

    # Step 1: Baseline on full VSR
    print("\n[BASELINE] Full VSR...", flush=True)
    base_all_acc, base_all_mg, n_base = run_baseline(all_indices, vsr_all, processor, model_raw, yes_ids, no_ids, device)
    print(f"[BASELINE] {base_all_acc:.2f}% margin={base_all_mg:.3f} N={n_base}", flush=True)
    results["baseline_all"] = {"acc": base_all_acc, "margin": base_all_mg, "n": n_base}

    # Step 2: Each feature on its own relation subset
    print("\n[INDIVIDUAL] Each feature on its relations...", flush=True)
    results["individual"] = {}
    for key, layer_idx, feature_idx, relations, alpha, start_layer, layer_vecs in feature_configs:
        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue
        base_acc, _, _ = run_baseline(indices, vsr_all, processor, model_raw, yes_ids, no_ids, device)
        acc_s, mg_s, n_eval = evaluate_dataset(
            indices, vsr_all, [(layer_vecs, alpha, start_layer)],
            nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device
        )
        da = acc_s - base_acc
        print(f"  {key} ({relations}): {base_acc:.2f}% → {acc_s:.2f}% (Δ={da:+.2f}%)", flush=True)
        results["individual"][key] = {
            "relations": relations, "alpha": alpha, "start": start_layer,
            "n": n_eval, "base_acc": base_acc, "steered_acc": acc_s, "delta_acc": da
        }

    # Step 3: Combined injection groups
    print("\n[COMBINED TOP-3] L4+L14+L12 on full VSR...", flush=True)
    top3_configs = [(lv, alpha, start) for (key, _, _, _, alpha, start, lv) in feature_configs if key in ["L4_F14233","L14_F10561","L12_F2257"]]
    acc_3, mg_3, n_3 = evaluate_dataset(all_indices, vsr_all, top3_configs, nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device)
    da_3 = acc_3 - base_all_acc
    print(f"[TOP-3] {acc_3:.2f}% (Δ={da_3:+.2f}%)", flush=True)
    results["combined_top3"] = {"features": ["L4_F14233","L14_F10561","L12_F2257"], "acc": acc_3, "delta_acc": da_3, "margin": mg_3, "n": n_3}

    print("\n[COMBINED TOP-5] L4+L14+L12+L15+L11 on full VSR...", flush=True)
    top5_keys = ["L4_F14233","L14_F10561","L12_F2257","L15_F220","L11_F12278"]
    top5_configs = [(lv, alpha, start) for (key, _, _, _, alpha, start, lv) in feature_configs if key in top5_keys]
    acc_5, mg_5, n_5 = evaluate_dataset(all_indices, vsr_all, top5_configs, nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device)
    da_5 = acc_5 - base_all_acc
    print(f"[TOP-5] {acc_5:.2f}% (Δ={da_5:+.2f}%)", flush=True)
    results["combined_top5"] = {"features": top5_keys, "acc": acc_5, "delta_acc": da_5, "margin": mg_5, "n": n_5}

    print("\n[COMBINED ALL-8] All 8 features on full VSR...", flush=True)
    all8_configs = [(lv, alpha, start) for (_, _, _, _, alpha, start, lv) in feature_configs]
    acc_8, mg_8, n_8 = evaluate_dataset(all_indices, vsr_all, all8_configs, nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device)
    da_8 = acc_8 - base_all_acc
    print(f"[ALL-8] {acc_8:.2f}% (Δ={da_8:+.2f}%)", flush=True)
    results["combined_all8"] = {"features": [f[0] for f in feature_configs], "acc": acc_8, "delta_acc": da_8, "margin": mg_8, "n": n_8, "base_acc": base_all_acc}

    # Step 4: Cross-relation effect
    print("\n[CROSS-RELATION] Each feature on full VSR...", flush=True)
    results["cross_relation"] = {}
    for key, layer_idx, feature_idx, relations, alpha, start_layer, layer_vecs in feature_configs:
        acc_s, mg_s, n_eval = evaluate_dataset(
            all_indices, vsr_all, [(layer_vecs, alpha, start_layer)],
            nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device
        )
        da = acc_s - base_all_acc
        print(f"  {key}: full VSR {acc_s:.2f}% (Δ={da:+.2f}%)", flush=True)
        results["cross_relation"][key] = {"acc": acc_s, "delta_acc": da, "margin": mg_s, "n": n_eval, "base_acc": base_all_acc}

    with open(result_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[DONE] Results saved to {result_path}", flush=True)

    print(f"\n{'='*80}")
    print("Combined All-8 Injection Summary")
    print(f"{'='*80}")
    print(f"Baseline (full VSR): {base_all_acc:.2f}%")
    print(f"\nIndividual on own relations:")
    for key, r in results["individual"].items():
        print(f"  {key}: Δ={r['delta_acc']:+.2f}%")
    print(f"\nCross-relation (each on full VSR):")
    for key, r in results["cross_relation"].items():
        print(f"  {key}: Δ={r['delta_acc']:+.2f}%")
    print(f"\nCombined top-3 on full VSR: Δ={results['combined_top3']['delta_acc']:+.2f}%")
    print(f"Combined top-5 on full VSR: Δ={results['combined_top5']['delta_acc']:+.2f}%")
    print(f"Combined all-8 on full VSR: Δ={results['combined_all8']['delta_acc']:+.2f}%")


if __name__ == "__main__":
    main()
