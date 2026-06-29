#!/usr/bin/env python3
"""
Per-layer injection sweep: for each feature, inject W_dec[F] at EVERY individual layer
independently to find the most causally effective injection layer in pt-448.

Motivation: SAE was trained on mix-448 at layer L. But the W_dec direction may be
most effective for pt-448 at a DIFFERENT layer (the features may be represented at
different depths in the two models). This experiment finds the optimal injection layer.

Method:
- For each feature, inject at single layer l (l=0..25) at a fixed alpha
- Alpha chosen as the best alpha from prior single-layer experiments
- Reports accuracy delta vs. baseline for each layer

Features tested (top-10 spatial):
  Using best individual alpha from prior experiments.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_layer_sweep/

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 pt448_layer_sweep.py
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

MODEL_NAME     = "google/paligemma2-3b-pt-448"
N_LAYERS       = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_layer_sweep")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Top-10 features with best confirmed alpha from prior experiments
FEATURES = [
    # (layer_idx, feature_idx, relations, sae_layer, best_alpha, prior_best_delta)
    (4,  14233, ["ahead of"],                            4,   4.0,  +10.26),
    (12, 2257,  ["facing"],                              12,  50.0,  +3.92),
    (11, 12278, ["touching"],                            11,  25.0,  +3.36),
    (9,  387,   ["at the right side of"],                9,   2.0,  +3.12),
    (15, 220,   ["across from", "at the left side of"], 15,   2.0,  +3.11),
    (9,  7540,  ["consists of"],                         9,  10.0,  +2.86),
    (14, 10561, ["close to"],                           14,   2.0,  +2.15),
    (13, 15219, ["behind"],                             13,  30.0,  +2.12),
    (6,  7539,  ["left of", "right of"],                 6,  20.0,  +1.24),
    (11, 9639,  ["in", "inside", "on"],                 11,  10.0,  +0.73),
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


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {MODEL_NAME}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
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

    for layer_idx, feature_idx, relations, sae_layer, alpha, prior_best in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"lsweep_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            print(f"\n[BASE] [{rel_key}] N={len(indices)}...", flush=True)
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

        # Load SAE W_dec for this feature
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()
        fv_col = fv.unsqueeze(1)

        print(f"\n[LAYER SWEEP] {key} {relations} α={alpha} (prior best {prior_best:+.2f}% @ sae_layer={sae_layer})", flush=True)

        result = {
            "layer_idx": layer_idx, "feature_idx": feature_idx, "relations": relations,
            "sae_layer": sae_layer, "alpha": alpha, "prior_best": prior_best,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "per_injection_layer": {}
        }

        for inj_layer in range(N_LAYERS):
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
                        lo = nns_model.model.language_model.layers[inj_layer].output[0][0, img_end:]
                        ones = (lo @ fv_col) * 0.0 + 1.0
                        lo += alpha * ones * fv
                        logits_s = nns_model.output.logits.save()
                    pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                    margins.append(m if label == 1 else -m)
                except Exception: pred = 0; margins.append(0.0)
                total += 1; correct += (pred == label)

            acc = correct / max(total, 1) * 100
            mg = sum(margins) / max(len(margins), 1)
            da = acc - base_acc; dm = mg - base_mg
            result["per_injection_layer"][str(inj_layer)] = {
                "acc": acc, "delta_acc": da, "margin": mg, "delta_margin": dm
            }
            print(f"  inj_layer={inj_layer:2d}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f}", flush=True)

        # Find best injection layer
        best_layer = max(result["per_injection_layer"].items(),
                         key=lambda x: x[1]["delta_acc"])
        result["best_injection_layer"] = int(best_layer[0])
        result["best_delta_acc"] = best_layer[1]["delta_acc"]
        print(f"\n  BEST: inj_layer={best_layer[0]} Δ={best_layer[1]['delta_acc']:+.2f}%"
              f"  (SAE layer was {sae_layer}, {'+' if int(best_layer[0]) >= sae_layer else ''}"
              f"{int(best_layer[0]) - sae_layer:+d} layers shift)", flush=True)

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        del fv, fv_col; torch.cuda.empty_cache(); gc.collect()

    print(f"\n{'='*110}")
    print("Per-Layer Injection Sweep — Summary")
    print(f"{'='*110}")
    print(f"{'Feature':<20} {'Relation':<30} {'SAE_L':>6} {'BestL':>6} {'Shift':>6} {'Best_Δ':>8} {'Prior':>8}")
    print("-" * 110)
    for r in all_results:
        key = f"L{r['layer_idx']}/F{r['feature_idx']}"
        shift = r['best_injection_layer'] - r['sae_layer']
        print(f"{key:<20} {str(r['relations']):<30} {r['sae_layer']:>6} "
              f"{r['best_injection_layer']:>6} {shift:>+6} {r['best_delta_acc']:>+8.2f}% {r['prior_best']:>+8.2f}%")


if __name__ == "__main__":
    main()
