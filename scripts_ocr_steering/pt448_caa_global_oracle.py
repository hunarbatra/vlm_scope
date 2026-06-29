#!/usr/bin/env python3
"""
Global-oracle evaluation: per-sample inject best feature at FIXED α=0.5.

Uses per-relation best-feature map derived from pt448_caa_per_relation_steer.py
but with all features constrained to α=0.5 (global hyperparameter).

Three modes:
1. global_oracle: use best feature per relation at α=0.5, skip SKIP relations
2. global_fixed_L6: inject L6/F7539 everywhere at α=0.5 (best cross-relation steerer)
3. global_fixed_best3: inject best of {L6, L9/F387, L15/F220, L4} per relation at α=0.5

Also uses a partial per-relation map from GPU3's current results.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_global_oracle/

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 pt448_caa_global_oracle.py
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_global_oracle")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"
PER_RELATION_JSON = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_per_relation/per_relation_steer.json")

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

GLOBAL_ALPHA = 0.5

# All features with start layers
ALL_FEATURES = [
    (4,  14233, 0), (14, 10561, 0), (12, 2257, 1), (15, 220, 15),
    (11, 12278, 5), (9,  387,   1), (6,  7539,  1), (9,  7540,  9),
    (13, 15219, 0), (11, 9639,  0),
]

# Partial per-relation best-feature map (from GPU3 results so far)
# Updated from per_relation_steer.log
PARTIAL_BEST_MAP = {
    "above":              "L15_F220",
    "across from":        "L6_F7539",
    "adjacent to":        "L9_F387",
    "against":            None,   # all features hurt; skip
    "ahead of":           "L4_F14233",
    "alongside":          "L6_F7539",
    "at the back of":     "L6_F7539",
    "at the edge of":     None,   # all ≤0; skip
    "at the left side of":"L15_F220",
    "at the right side of":"L9_F387",
    "at the side of":     "L9_F387",
    "attached to":        "L9_F387",
    "away from":          "L15_F220",
    "behind":             "L4_F14233",
    "below":              "L6_F7539",
    # relations not yet in GPU3 — use None (no injection) to be conservative
}
SKIP_RELATIONS = {"against", "at the edge of"}


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


def run_mode(mode_name, all_indices, vsr_all, feature_vecs, feature_starts,
             nns_model, yes_ids, no_ids, processor, base_module, model_dtype, device,
             get_feature_fn):
    """Run a specific injection mode. get_feature_fn(relation) -> key or None."""
    from utils import process_vlm_inputs, get_image_token_positions
    correct = total = n_injected = n_skipped = 0; margins = []
    for vi in all_indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        relation = ex.get("relation", "")
        feat_key = get_feature_fn(relation)
        try:
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                 processor, base_module, device=device)
            if feat_key is None or feat_key not in feature_vecs:
                with torch.inference_mode():
                    out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, m = _pm(out.logits[0, -1, :], yes_ids, no_ids)
                n_skipped += 1
            else:
                layer_vecs = feature_vecs[feat_key]
                start_layer = feature_starts[feat_key]
                _, img_end = get_image_token_positions(iids)
                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l in range(start_layer, N_LAYERS):
                        v_l = layer_vecs[l]
                        v_col = v_l.unsqueeze(1)
                        lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        ones = (lo @ v_col) * 0.0 + 1.0
                        lo += GLOBAL_ALPHA * ones * v_l
                    logits_s = nns_model.output.logits.save()
                pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                n_injected += 1
            margins.append(m if label == 1 else -m)
        except Exception: pred = 0; margins.append(0.0)
        total += 1; correct += (pred == label)
        if total % 1000 == 0: print(f"  {mode_name} {total} (inj={n_injected}, skip={n_skipped})", flush=True)
    acc = correct / max(total, 1) * 100
    mg = sum(margins) / max(len(margins), 1)
    return acc, mg, total, n_injected, n_skipped


def main():
    global model_raw
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    result_path = OUT_DIR / "global_oracle_results.json"
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
    feature_vecs = {}; feature_starts = {}
    for layer_idx, feature_idx, start_layer in ALL_FEATURES:
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
        feature_vecs[key] = layer_vecs
        feature_starts[key] = start_layer
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

    results = {"baseline": {"acc": base_acc, "margin": base_mg, "n": total}, 
               "global_alpha": GLOBAL_ALPHA, "modes": {}}

    # Mode 1: Global oracle (partial per-relation map, α=0.5)
    print(f"\n[MODE] global_oracle (partial map, {len([v for v in PARTIAL_BEST_MAP.values() if v])} relations mapped)", flush=True)
    acc, mg, n, n_inj, n_skip = run_mode(
        "global_oracle", all_indices, vsr_all, feature_vecs, feature_starts,
        nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device,
        lambda rel: PARTIAL_BEST_MAP.get(rel, None))
    da = acc - base_acc
    print(f"[RESULT] {acc:.2f}% (Δ={da:+.2f}%) injected={n_inj} skipped={n_skip}", flush=True)
    results["modes"]["global_oracle"] = {"acc": acc, "delta_acc": da, "margin": mg, "n_injected": n_inj, "n_skipped": n_skip}

    # Mode 2: Fixed L6 everywhere (α=0.5) — best single cross-steerer
    print(f"\n[MODE] fixed_L6_everywhere (L6/F7539, α={GLOBAL_ALPHA})", flush=True)
    acc, mg, n, n_inj, n_skip = run_mode(
        "fixed_L6", all_indices, vsr_all, feature_vecs, feature_starts,
        nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device,
        lambda rel: "L6_F7539")
    da = acc - base_acc
    print(f"[RESULT] {acc:.2f}% (Δ={da:+.2f}%)", flush=True)
    results["modes"]["fixed_L6_everywhere"] = {"acc": acc, "delta_acc": da, "margin": mg}

    # Mode 3: Fixed L4/F14233 everywhere — highest own-relation gain
    print(f"\n[MODE] fixed_L4_everywhere (L4/F14233, α={GLOBAL_ALPHA})", flush=True)
    acc, mg, n, n_inj, n_skip = run_mode(
        "fixed_L4", all_indices, vsr_all, feature_vecs, feature_starts,
        nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device,
        lambda rel: "L4_F14233")
    da = acc - base_acc
    print(f"[RESULT] {acc:.2f}% (Δ={da:+.2f}%)", flush=True)
    results["modes"]["fixed_L4_everywhere"] = {"acc": acc, "delta_acc": da, "margin": mg}

    with open(result_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved to {result_path}", flush=True)
    _print_summary(results)


def _print_summary(results):
    base = results["baseline"]["acc"]
    alpha = results.get("global_alpha", "?")
    print(f"\n{'='*60}")
    print(f"Global-Oracle Summary (α={alpha}, baseline={base:.2f}%)")
    print(f"{'='*60}")
    for name, r in results.get("modes", {}).items():
        da = r.get("delta_acc", 0)
        n_inj = r.get("n_injected", "?")
        print(f"  {name}: {r['acc']:.2f}% (Δ={da:+.2f}%) injected={n_inj}")


if __name__ == "__main__":
    main()
