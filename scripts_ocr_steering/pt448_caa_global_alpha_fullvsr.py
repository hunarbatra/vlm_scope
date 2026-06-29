#!/usr/bin/env python3
"""
Global-alpha caa_sae_down evaluation on FULL VSR.

Tests a SINGLE fixed α=0.5 (global hyperparameter) applied to ALL 10 features
via caa_sae_down (inject each feature's own-layer CAA from its SAE layer → L25).
This mirrors how CAA papers use a single global alpha selected on a validation set.

Also tests α=0.25 and α=1.0 for sensitivity analysis.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_global_alpha/

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 pt448_caa_global_alpha_fullvsr.py
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_global_alpha")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# All 10 features with start layers; skipping L9/F7540 (caa_sae_down harmful)
FEATURES = [
    # (layer_idx, feature_idx, start_layer, own_relations)
    (4,  14233, 0, ["ahead of"]),
    (14, 10561, 0, ["close to"]),
    (12, 2257,  1, ["facing"]),
    (15, 220,  15, ["across from", "at the left side of"]),
    (11, 12278, 5, ["touching"]),
    (9,  387,   1, ["at the right side of"]),
    (6,  7539,  1, ["left of", "right of"]),
    (9,  7540,  9, ["consists of"]),   # included for completeness; expected ~0
    (13, 15219, 0, ["behind"]),
    (11, 9639,  0, ["in", "inside", "on"]),   # if CAA vector available
]

GLOBAL_ALPHAS = [0.25, 0.5, 1.0]  # test sensitivity


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


def eval_feature_global_alpha(all_indices, vsr_all, layer_vecs, alpha, start_layer,
                               nns_model, yes_ids, no_ids, processor, model_dtype, device,
                               own_relations):
    """Evaluate a feature with a fixed global alpha on all VSR samples.
    Returns: (acc_all_vsr, delta_all, acc_own_relation, delta_own, n_own)
    """
    from utils import process_vlm_inputs, get_image_token_positions
    import re
    
    correct_all = total_all = 0
    correct_own = total_own = 0
    base_correct_own = 0
    margins = []
    
    for vi in all_indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        is_own = ex.get("relation", "") in own_relations
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
        total_all += 1; correct_all += (pred == label)
        if is_own:
            total_own += 1; correct_own += (pred == label)
        if total_all % 2000 == 0: print(f"  {total_all}", flush=True)
    
    acc_all = correct_all / max(total_all, 1) * 100
    acc_own = correct_own / max(total_own, 1) * 100
    return acc_all, acc_own, total_own


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    result_path = OUT_DIR / "global_alpha_results.json"
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
    for layer_idx, feature_idx, start_layer, own_rels in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists(): print(f"[SKIP] No CAA for {key}", flush=True); continue
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
    # Also compute per-own-relation baselines
    own_rel_base = {}
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
        rel = ex.get("relation", "")
        if rel not in own_rel_base: own_rel_base[rel] = [0, 0]
        own_rel_base[rel][0] += (pred == label); own_rel_base[rel][1] += 1
        if total % 2000 == 0: print(f"  baseline {total}", flush=True)
    base_acc = correct / max(total, 1) * 100
    base_mg = sum(margins) / max(len(margins), 1)
    print(f"[BASELINE] {base_acc:.2f}% margin={base_mg:.3f} N={total}", flush=True)

    results = {"baseline": {"acc": base_acc, "margin": base_mg, "n": total}, "features": {}}

    # Test each feature at each global alpha
    for layer_idx, feature_idx, start_layer, own_rels in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        if key not in feature_vecs: continue
        layer_vecs = feature_vecs[key]
        print(f"\n[FEATURE] {key} start={start_layer} own_rels={own_rels}", flush=True)
        results["features"][key] = {"start": start_layer, "own_relations": own_rels, "alphas": {}}
        
        # Get own-relation baseline acc
        own_base_correct = sum(own_rel_base.get(r, [0,0])[0] for r in own_rels)
        own_base_total = sum(own_rel_base.get(r, [0,0])[1] for r in own_rels)
        own_base_acc = own_base_correct / max(own_base_total, 1) * 100
        results["features"][key]["own_base_acc"] = own_base_acc
        results["features"][key]["n_own"] = own_base_total
        
        for alpha in GLOBAL_ALPHAS:
            print(f"  [α={alpha}]", flush=True)
            acc_all, acc_own, n_own = eval_feature_global_alpha(
                all_indices, vsr_all, layer_vecs, alpha, start_layer,
                nns_model, yes_ids, no_ids, processor, model_dtype, device, own_rels)
            da_all = acc_all - base_acc
            da_own = acc_own - own_base_acc
            print(f"    full_VSR: {acc_all:.2f}% (Δ={da_all:+.2f}%)  own_rel: {acc_own:.2f}% (Δ={da_own:+.2f}%)", flush=True)
            results["features"][key]["alphas"][str(alpha)] = {
                "acc_full_vsr": acc_all, "delta_full_vsr": da_all,
                "acc_own_rel": acc_own, "delta_own_rel": da_own
            }

    with open(result_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved to {result_path}", flush=True)
    _print_summary(results)


def _print_summary(results):
    base = results["baseline"]["acc"]
    print(f"\n{'='*80}")
    print(f"Global-Alpha caa_sae_down Summary (baseline={base:.2f}%)")
    print(f"{'='*80}")
    print(f"{'Feature':<15} {'Own Relation':<30} {'α=0.25 ΔfVSR':>13} {'α=0.5 ΔfVSR':>12} {'α=1.0 ΔfVSR':>12}")
    print('-'*85)
    for key, r in results.get("features", {}).items():
        rels = ", ".join(r.get("own_relations", []))
        alphas = r.get("alphas", {})
        d025 = alphas.get("0.25", {}).get("delta_full_vsr", float('nan'))
        d050 = alphas.get("0.5", {}).get("delta_full_vsr", float('nan'))
        d100 = alphas.get("1.0", {}).get("delta_full_vsr", float('nan'))
        print(f"{key:<15} {rels:<30} {d025:>+13.2f}% {d050:>+12.2f}% {d100:>+12.2f}%")


if __name__ == "__main__":
    main()
