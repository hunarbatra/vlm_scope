#!/usr/bin/env python3
"""
Extended alpha sweep for the three hardest features — those with weakest injection gains.

Current best (from all prior experiments):
  L11/F9639  in/inside/on   +0.73% @ answer α=10  (N=1101)
  L6/F7539   left/right of  +1.24% @ topK_ml α=20  (N=323)
  L13/F15219 behind         +2.12% @ downstream_ml α=30 (N=709)

This script extends the alpha range significantly and tests more strategies to see
if these features can be pushed further. Also tests sae_only_down/up which were
best for some other features.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hard_features_ext/

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 pt448_hard_features_ext.py
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
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hard_features_ext")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

DECAY_ML = 0.7
DECAY_RA = 0.85

HARD_FEATURES = [
    (11, 9639,  ["in", "inside", "on"],   +0.73, "answer", 10.0),
    (6,  7539,  ["left of", "right of"],  +1.24, "topK_ml", 20.0),
    (13, 15219, ["behind"],               +2.12, "downstream_ml", 30.0),
]

# Per-feature strategy × alpha combinations
SWEEPS = {
    "L11_F9639": {
        "all_ml":      [1, 2, 5, 10, 20, 30, 50, 70, 100],
        "downstream_ml": [5, 10, 20, 30, 50, 70, 100],
        "answer":      [5, 10, 20, 30, 50, 70, 100],
        "single":      [5, 10, 20, 30, 50],
        "sae_only_down": [2, 5, 10, 20, 30, 50],
    },
    "L6_F7539": {
        "topK_ml":     [5, 10, 20, 30, 50, 70, 100],
        "all_ml":      [5, 10, 20, 30, 50, 70, 100],
        "downstream_ml": [5, 10, 20, 30, 50],
        "sae_only_down": [2, 5, 10, 20, 30],
        "sae_only_up": [2, 5, 10, 20, 30],
    },
    "L13_F15219": {
        "downstream_ml": [10, 20, 30, 50, 70, 100],
        "all_ml":        [10, 20, 30, 50, 70, 100],
        "sae_only_down": [5, 10, 20, 30, 50],
        "single":        [5, 10, 20, 30, 50],
    },
}


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
    url = ex.get("image_link","")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB"); img.save(cp,"JPEG")
        return img
    except Exception: return None

def _get_layer_weights(strategy, sae_layer):
    if strategy == "single":
        return {sae_layer: 1.0}
    elif strategy == "sae_only_down":
        return {l: 1.0 for l in range(sae_layer, N_LAYERS)}
    elif strategy == "sae_only_up":
        return {l: 1.0 for l in range(0, sae_layer + 1)}
    elif strategy == "decay_fwd_ra":
        return {l: DECAY_RA ** max(l - sae_layer, 0) for l in range(N_LAYERS)}
    elif strategy == "all_ml":
        return {l: DECAY_ML ** abs(l - sae_layer) for l in range(N_LAYERS)}
    elif strategy == "downstream_ml":
        return {l: DECAY_ML ** (l - sae_layer) for l in range(sae_layer, N_LAYERS)}
    elif strategy == "topK_ml":
        return {l: DECAY_ML ** abs(l - sae_layer) for l in range(max(0, sae_layer-2), min(N_LAYERS, sae_layer+3))}
    elif strategy == "answer":
        return {l: 1.0 for l in range(21, N_LAYERS)}
    return {sae_layer: 1.0}


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
    data_files = {"train":"train.jsonl","dev":"dev.jsonl","test":"test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train","dev","test"]
    ])
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation","")].append(vi)

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations, prior_best, prior_strat, prior_alpha in HARD_FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        sweep_key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"hardext_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

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
                label = int(ex.get("label",0))
                try:
                    iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                         processor, model_raw, device=device)
                    with torch.inference_mode():
                        out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                    pred, m = _pm(out.logits[0,-1,:], yes_ids, no_ids)
                    margins.append(m if label==1 else -m)
                except Exception: pred=0; margins.append(0.0)
                total+=1; correct+=(pred==label)
            acc = correct/max(total,1)*100
            mg = sum(margins)/max(len(margins),1)
            baseline_cache[rel_key] = (acc, mg, total)
            print(f"[BASE] {acc:.2f}% margin={mg:.3f}", flush=True)
        base_acc, base_mg, _ = baseline_cache[rel_key]

        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()
        fv_col = fv.unsqueeze(1)

        result = {
            "layer": layer_idx, "feature": feature_idx, "relations": relations,
            "prior_best_delta": prior_best, "prior_strategy": prior_strat, "prior_alpha": prior_alpha,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "strategies": {}
        }

        sweeps = SWEEPS.get(sweep_key, {})
        for strategy, alpha_range in sweeps.items():
            print(f"\n[STRAT] {key} {strategy}  prior={prior_strat}@{prior_alpha} best={prior_best:+.2f}%", flush=True)
            lw = _get_layer_weights(strategy, layer_idx)
            strat_res = {"alphas": {}}
            for alpha in alpha_range:
                print(f"  [INJECT] {key} {strategy} α={alpha:+g}...", flush=True)
                correct = total = 0; margins = []
                for vi in indices:
                    ex = vsr_all[vi]; img = _load_image(ex)
                    if img is None: continue
                    label = int(ex.get("label",0))
                    try:
                        iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                             processor, nns_model._module, device=device)
                        _, img_end = get_image_token_positions(iids)
                        with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                            for l, w in lw.items():
                                lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                                ones = (lo @ fv_col) * 0.0 + 1.0
                                lo += (alpha * w) * ones * fv
                            logits_s = nns_model.output.logits.save()
                        pred, m = _pm(logits_s[0,-1,:], yes_ids, no_ids)
                        margins.append(m if label==1 else -m)
                    except Exception: pred=0; margins.append(0.0)
                    total+=1; correct+=(pred==label)
                acc = correct/max(total,1)*100; mg = sum(margins)/max(len(margins),1)
                da = acc - base_acc; dm = mg - base_mg
                strat_res["alphas"][str(alpha)] = {"acc":acc,"delta_acc":da,"margin":mg,"delta_margin":dm}
                print(f"    α={alpha:+6.1f}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f} (Δ={dm:+.3f})", flush=True)
            result["strategies"][strategy] = strat_res

        with open(result_path,"w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n{'='*110}")
    print("Hard Features Extended Alpha Sweep")
    print(f"{'='*110}")
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        print(f"\n{key} {r['relations']} base={r['baseline_vsr_acc']:.2f}% (prior best: {r['prior_best_delta']:+.2f}% @ {r['prior_strategy']} α={r['prior_alpha']})")
        best_overall = (None, None, -999)
        for strat, sv in r["strategies"].items():
            best_a = max(sv["alphas"].items(), key=lambda x: x[1]["delta_acc"])
            if best_a[1]["delta_acc"] > best_overall[2]:
                best_overall = (strat, best_a[0], best_a[1]["delta_acc"])
            print(f"  {strat}: best Δ={best_a[1]['delta_acc']:+.2f}% @ α={best_a[0]}")
        print(f"  OVERALL BEST: {best_overall[0]} α={best_overall[1]} Δ={best_overall[2]:+.2f}%")


if __name__ == "__main__":
    main()
