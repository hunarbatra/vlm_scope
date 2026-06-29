#!/usr/bin/env python3
"""
Late-layer CAA injection experiment.

Based on finding that CAA vector norms are highest at layers 22-25 (near output),
this script tests injecting the CAA vector at those late layers instead of at the
SAE layer. The hypothesis is that injecting closer to the answer token position
in pt-448's late layers may be more effective.

Also tests: inject the W_dec vector at late layers (L21-25) for comparison.

Uses precomputed CAA vectors from pt448_caa_steering.py Phase 1.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_late_layer_caa/

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 pt448_late_layer_caa.py
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
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
CAA_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_late_layer_caa")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Top features with prior best W_dec results for comparison
TOP_FEATURES = [
    (4,  14233, ["ahead of"],                            4,  +10.26, "sae_only_down", 4.0),
    (12, 2257,  ["facing"],                              12, +3.92,  "all_ml",       50.0),
    (11, 12278, ["touching"],                            11, +3.36,  "single",       25.0),
    (9,  387,   ["at the right side of"],                9,  +3.12,  "decay_fwd_ra",  2.0),
    (15, 220,   ["across from", "at the left side of"], 15, +3.11,  "sae_only_up",   2.0),
    (13, 15219, ["behind"],                             13, +2.12,  "downstream_ml",30.0),
]

# Injection strategies to test
# late_caa: inject L24 CAA vector at layers 21-25 with unit weights
# late_caa_single: inject L24 CAA vector at layer 24 only
# late_wdec: inject W_dec at layers 21-25 (reference)
# highest_caa: inject CAA vector at the layer with highest CAA norm

ALPHA_RANGE = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


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

    for layer_idx, feature_idx, relations, sae_layer, prior_best, prior_strat, prior_alpha in TOP_FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"latecaa_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        # Load precomputed CAA vectors
        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists():
            print(f"[SKIP - no CAA] {key}", flush=True)
            continue

        caa_saved = torch.load(caa_path)
        caa_data = caa_saved["caa_data"]

        # Find best CAA layer (highest norm)
        best_caa_layer = max(caa_data.items(), key=lambda x: x[1]["norm"])
        best_caa_l = best_caa_layer[0]
        best_caa_norm = best_caa_layer[1]["norm"]
        print(f"\n[{key}] Best CAA layer: L{best_caa_l} (norm={best_caa_norm:.3f})", flush=True)

        # Get CAA vectors at best layer and at L24 (late decision layer)
        # Use normalized vectors
        v_best = caa_data[best_caa_l]["v_caa_norm"]
        v_late = caa_data[24]["v_caa_norm"] if 24 in caa_data else caa_data[best_caa_l]["v_caa_norm"]
        v_sae  = caa_data[layer_idx]["v_caa_norm"]

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

        # Load W_dec for comparison
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()
        fv_col = fv.unsqueeze(1)

        result = {
            "layer_idx": layer_idx, "feature_idx": feature_idx, "relations": relations,
            "prior_best": prior_best, "prior_strat": prior_strat, "prior_alpha": prior_alpha,
            "best_caa_layer": best_caa_l, "best_caa_norm": best_caa_norm,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "strategies": {}
        }

        # Test strategies
        strategies_to_test = [
            # (name, injection_layers, vector_key, description)
            ("late_caa_answer",   list(range(21, 26)), "v_late",  "CAA@L24 injected at layers 21-25"),
            ("late_caa_single24", [24],                "v_late",  "CAA@L24 at layer 24 only"),
            ("best_caa_single",   [best_caa_l],        "v_best",  f"CAA@best_layer={best_caa_l}"),
            ("late_wdec_answer",  list(range(21, 26)), "v_wdec",  "W_dec at layers 21-25 (reference)"),
        ]

        vecs = {"v_late": v_late.to(model_dtype).to(device),
                "v_best": v_best.to(model_dtype).to(device),
                "v_wdec": fv}

        for strat_name, inj_layers, vec_key, desc in strategies_to_test:
            print(f"\n  [STRAT] {key} {strat_name}: {desc}", flush=True)
            strat_res = {"alphas": {}}
            v_steer = vecs[vec_key]
            v_col = v_steer.unsqueeze(1)

            for alpha in ALPHA_RANGE:
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
                            for l in inj_layers:
                                lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                                ones = (lo @ v_col) * 0.0 + 1.0
                                lo += alpha * ones * v_steer
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
            print(f"  >> {strat_name}: best Δ={best_a[1]['delta_acc']:+.2f}% @ α={best_a[0]}", flush=True)
            result["strategies"][strat_name] = strat_res

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        del fv, fv_col, vecs; torch.cuda.empty_cache(); gc.collect()

    print(f"\n{'='*110}")
    print("Late-Layer CAA Injection — Summary")
    print(f"{'='*110}")
    for r in all_results:
        key = f"L{r['layer_idx']}/F{r['feature_idx']}"
        print(f"\n{key} [{r['relations']}] prior_best={r['prior_best']:+.2f}% base={r['baseline_vsr_acc']:.2f}%")
        print(f"  Best CAA layer: L{r['best_caa_layer']} (norm={r['best_caa_norm']:.3f})")
        for strat, sv in r["strategies"].items():
            best = max(sv["alphas"].items(), key=lambda x: x[1]["delta_acc"])
            print(f"  {strat}: best Δ={best[1]['delta_acc']:+.2f}% @ α={best[0]}")


if __name__ == "__main__":
    main()
