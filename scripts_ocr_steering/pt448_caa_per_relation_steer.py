#!/usr/bin/env python3
"""
Per-relation steering analysis.

For each VSR relation, tests all 8 steerable features at their optimal configs
and reports which feature best steers each relation.

This answers:
- Does steering for "facing" (L12/F2257) also help "behind" or hurt "above"?
- Which relations have zero steerable features (need new feature search)?
- What is the theoretical ceiling if we always apply the best feature per relation?

All 8 features at confirmed optimal configs:
  L4/F14233:  α=1.0,  start=0  | L14/F10561: α=2.0,  start=0
  L12/F2257:  α=1.0,  start=1  | L15/F220:   α=0.75, start=15
  L11/F12278: α=0.5,  start=5  | L9/F387:    α=0.5,  start=1
  L6/F7539:   α=1.5,  start=1  | L9/F7540:   α=0.25, start=9

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_per_relation/

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 pt448_caa_per_relation_steer.py
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_per_relation")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

FEATURES = [
    (4,  14233, ["ahead of"],                            1.0,  0),
    (14, 10561, ["close to"],                            2.0,  0),
    (12, 2257,  ["facing"],                              1.0,  1),
    (15, 220,   ["across from", "at the left side of"],  0.75, 15),
    (11, 12278, ["touching"],                            0.5,  5),
    (9,  387,   ["at the right side of"],                0.5,  1),
    (6,  7539,  ["left of", "right of"],                 1.5,  1),
    (9,  7540,  ["consists of"],                         0.25, 9),
]

MIN_SAMPLES = 20


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


def eval_indices(indices, vsr_all, layer_vecs, alpha, start_layer, nns_model,
                 yes_ids, no_ids, processor, base_module, model_dtype, device):
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
    return acc, total


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    result_path = OUT_DIR / "per_relation_steer.json"
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
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    # Load all feature vectors
    feature_vecs = {}
    feature_meta = []
    for layer_idx, feature_idx, relations, alpha, start_layer in FEATURES:
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
        feature_meta.append((key, layer_idx, feature_idx, relations, alpha, start_layer))
        print(f"[LOADED] {key}", flush=True)

    # Get baseline per relation
    relations_to_test = [r for r, idxs in relation_indices.items() if len(idxs) >= MIN_SAMPLES]
    baseline_cache = {}
    results = {"relations": {}}

    for relation in sorted(relations_to_test):
        indices = relation_indices[relation]
        print(f"\n[RELATION] '{relation}' N={len(indices)}", flush=True)

        # Baseline
        correct = total = 0
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                     processor, model_raw, device=device)
                with torch.inference_mode():
                    out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, _ = _pm(out.logits[0, -1, :], yes_ids, no_ids)
            except Exception: pred = 0
            total += 1; correct += (pred == label)
        base_acc = correct / max(total, 1) * 100
        baseline_cache[relation] = base_acc
        print(f"  BASE: {base_acc:.2f}%", flush=True)

        # Each feature
        feature_results = {}
        for key, layer_idx, feature_idx, feat_relations, alpha, start_layer in feature_meta:
            acc_s, n_eval = eval_indices(
                indices, vsr_all, feature_vecs[key], alpha, start_layer,
                nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device
            )
            da = acc_s - base_acc
            marker = " ← OWN" if relation in feat_relations else ""
            print(f"  {key}: {acc_s:.2f}% (Δ={da:+.2f}%){marker}", flush=True)
            feature_results[key] = {"acc": acc_s, "delta_acc": da, "is_own_relation": relation in feat_relations}

        best_key = max(feature_results, key=lambda k: feature_results[k]["delta_acc"])
        best_da = feature_results[best_key]["delta_acc"]
        results["relations"][relation] = {
            "n": len(indices), "n_eval": total, "base_acc": base_acc,
            "features": feature_results,
            "best_feature": best_key, "best_delta": best_da
        }

    with open(result_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved to {result_path}", flush=True)
    _print_summary(results)


def _print_summary(results):
    print(f"\n{'='*90}")
    print("Per-Relation Steering Summary")
    print(f"{'='*90}")
    print(f"{'Relation':<35} {'N':>5} {'Base':>7} {'BestΔ':>8} {'BestFeature':<20}")
    print("-" * 90)
    for rel, r in sorted(results["relations"].items(), key=lambda x: -x[1]["best_delta"]):
        print(f"{rel:<35} {r['n']:>5} {r['base_acc']:>7.2f}% {r['best_delta']:>+8.2f}% {r['best_feature']:<20}")


if __name__ == "__main__":
    main()
