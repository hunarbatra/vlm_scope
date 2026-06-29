#!/usr/bin/env python3
"""
Fine-grained alpha sweep for the best CAA strategies found in Exp 21.

Key finding from Exp 21:
  - caa_sae_down (inject CAA at layers SAE_layer→25, flat) gives +15.38% for L4/F14233 @ α=1
  - caa_all_ml (inject at all 26 layers, 0.7 decay) gives +8.50% for L12/F2257 @ α=10

This script does finer alpha sweeps around those optima, plus tests:
  - caa_sae_down: fine sweep α ∈ [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
  - caa_all_ml: fine sweep α ∈ [5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 30.0]

Runs on ALL 10 features to find which ones benefit from CAA injection.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_fine_sweep/

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 pt448_caa_fine_sweep.py
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

MODEL_PT       = "google/paligemma2-3b-pt-448"
N_LAYERS       = 26
CAA_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_fine_sweep")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# All 10 features
FEATURES = [
    (4,  14233, ["ahead of"],                            "sae_only_down", 4.0,  +10.26),
    (12, 2257,  ["facing"],                              "all_ml",       50.0,  +3.92),
    (11, 12278, ["touching"],                            "single",       25.0,  +3.36),
    (9,  387,   ["at the right side of"],                "decay_fwd_ra",  2.0,  +3.12),
    (15, 220,   ["across from", "at the left side of"], "sae_only_up",   2.0,  +3.11),
    (9,  7540,  ["consists of"],                         "single",       10.0,  +2.86),
    (14, 10561, ["close to"],                            "all_ml",        2.0,  +2.15),
    (13, 15219, ["behind"],                              "downstream_ml",30.0,  +2.12),
    (6,  7539,  ["left of", "right of"],                 "topK_ml",      20.0,  +1.24),
    (11, 9639,  ["in", "inside", "on"],                  "answer",       10.0,  +0.73),
]

# Fine sweep alphas for each strategy
ALPHA_SAE_DOWN = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
ALPHA_ALL_ML   = [1.0, 2.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0]
DECAY_ML       = 0.7


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


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

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

    for layer_idx, feature_idx, relations, prior_strat, prior_alpha, prior_best in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"caafine_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists():
            print(f"[SKIP - no CAA] {key}", flush=True)
            continue

        caa_saved = torch.load(caa_path)
        caa_data = caa_saved["caa_data"]
        v_caa = caa_data[layer_idx]["v_caa_norm"].to(model_dtype).to(device)
        v_col = v_caa.unsqueeze(1)

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
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

        result = {
            "layer_idx": layer_idx, "feature_idx": feature_idx, "relations": relations,
            "prior_best": prior_best, "prior_strat": prior_strat, "prior_alpha": prior_alpha,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "strategies": {}
        }

        strategies_to_test = [
            ("caa_sae_down_fine", ALPHA_SAE_DOWN, {l: 1.0 for l in range(layer_idx, N_LAYERS)}),
            ("caa_all_ml_fine",   ALPHA_ALL_ML,   {l: DECAY_ML ** abs(l - layer_idx) for l in range(N_LAYERS)}),
        ]

        print(f"\n[{key}] prior_best={prior_best:+.2f}%", flush=True)

        for strat_name, alpha_range, layer_weights in strategies_to_test:
            print(f"  [STRAT] {strat_name}", flush=True)
            strat_res = {"alphas": {}}

            for alpha in alpha_range:
                correct = total = 0; margins = []
                for vi in indices:
                    ex = vsr_all[vi]; img = _load_image(ex)
                    if img is None: continue
                    label = int(ex.get("label", 0))
                    try:
                        iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                             processor, nns_model._module, device=device)
                        _, img_end = get_image_token_positions(iids)
                        with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                            for l, w in layer_weights.items():
                                lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                                ones = (lo @ v_col) * 0.0 + 1.0
                                lo += alpha * w * ones * v_caa
                            logits_s = nns_model.output.logits.save()
                        pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                        margins.append(m if label == 1 else -m)
                    except Exception: pred = 0; margins.append(0.0)
                    total += 1; correct += (pred == label)
                acc = correct / max(total, 1) * 100
                mg = sum(margins) / max(len(margins), 1)
                da = acc - base_acc; dm = mg - base_mg
                strat_res["alphas"][str(alpha)] = {"acc": acc, "delta_acc": da, "margin": mg, "delta_margin": dm}
                print(f"    α={alpha:+7.3f}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f}", flush=True)

            best_a = max(strat_res["alphas"].items(), key=lambda x: x[1]["delta_acc"])
            strat_res["best_alpha"] = best_a[0]
            strat_res["best_delta_acc"] = best_a[1]["delta_acc"]
            print(f"  >> {strat_name}: best Δ={best_a[1]['delta_acc']:+.2f}% @ α={best_a[0]}", flush=True)
            result["strategies"][strat_name] = strat_res

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        del v_caa, v_col; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("CAA Fine Sweep Summary")
    print(f"{'='*100}")
    print(f"{'Feature':<20} {'Relation':<30} {'Prior_W_dec':>12} {'sae_down':>10} {'all_ml':>10}")
    print("-" * 100)
    for r in all_results:
        key = f"L{r['layer_idx']}/F{r['feature_idx']}"
        sd_best = r['strategies'].get('caa_sae_down_fine', {}).get('best_delta_acc', 0)
        ml_best = r['strategies'].get('caa_all_ml_fine', {}).get('best_delta_acc', 0)
        print(f"{key:<20} {str(r['relations']):<30} {r['prior_best']:>+12.2f}% {sd_best:>+10.2f}% {ml_best:>+10.2f}%")


if __name__ == "__main__":
    main()
