#!/usr/bin/env python3
"""
Fine sweep to boost mix-448 for the two features that showed improvement.

mix448_3tap_alllayer found:
  L12/F2257 "facing":    +3.92% @ alpha=+0.1  (3tap all-26L)
  L15/F220 "across from":+2.52% @ alpha=+0.05 (3tap all-26L)

These features CAN be pushed beyond mix-448 training! This script tests:
  - Fine alpha around the known optimal
  - 3tap (attn_out + mlp_out + layer_out) at all 26 layers
  - Single layer at SAE layer only (fewer taps = safer)
  - Residual only at all 26 layers (middle ground)

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/mix448_boostable_fine/

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 mix448_boostable_fine.py
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

MODEL_NAME     = "google/paligemma2-3b-mix-448"
N_LAYERS       = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix448_boostable_fine")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

FEATURES = [
    (12, 2257,  ["facing"],                         3.92, 0.1),
    (15, 220,   ["across from","at the left side of"], 2.52, 0.05),
]

ALPHA_RANGES = {
    "3tap_alllayer": [0.01, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5],
    "single_layer":  [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    "residual_all":  [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
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


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {MODEL_NAME} (mix-448 boost)...", flush=True)
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

    for layer_idx, feature_idx, relations, prior_best, prior_alpha in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"mixboost_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        indices = []
        for r in relations: indices.extend(relation_indices.get(r,[]))
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
            "prior_best_delta": prior_best, "prior_best_alpha": prior_alpha,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "model": "mix-448", "strategies": {}
        }

        for strategy, alpha_range in ALPHA_RANGES.items():
            print(f"\n[STRAT] {key} {strategy}", flush=True)
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
                            for l in range(N_LAYERS):
                                if strategy == "single_layer" and l != layer_idx: continue
                                ones_proxy = None
                                if strategy in ("3tap_alllayer", "residual_all"):
                                    lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                                    ones = (lo @ fv_col) * 0.0 + 1.0
                                    lo += alpha * ones * fv
                                if strategy == "3tap_alllayer":
                                    ao = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                                    ones_a = (ao @ fv_col) * 0.0 + 1.0
                                    ao += alpha * ones_a * fv
                                    mo = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                                    ones_m = (mo @ fv_col) * 0.0 + 1.0
                                    mo += alpha * ones_m * fv
                                if strategy == "single_layer":
                                    lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                                    ones = (lo @ fv_col) * 0.0 + 1.0
                                    lo += alpha * ones * fv
                            logits_s = nns_model.output.logits.save()
                        pred, m = _pm(logits_s[0,-1,:], yes_ids, no_ids)
                        margins.append(m if label==1 else -m)
                    except Exception: pred=0; margins.append(0.0)
                    total+=1; correct+=(pred==label)
                acc = correct/max(total,1)*100; mg = sum(margins)/max(len(margins),1)
                da = acc - base_acc; dm = mg - base_mg
                strat_res["alphas"][str(alpha)] = {"acc":acc,"delta_acc":da,"margin":mg,"delta_margin":dm}
                print(f"    α={alpha:+g}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f} (Δ={dm:+.3f})", flush=True)
            result["strategies"][strategy] = strat_res

        with open(result_path,"w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n{'='*100}")
    print("mix-448 Boostable Features Fine Sweep")
    print(f"{'='*100}")
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        print(f"{key} {r['relations']} base={r['baseline_vsr_acc']:.2f}% (prior best: {r['prior_best_delta']:+.2f}%)")
        for strat, sv in r["strategies"].items():
            best_a, best_v = max(sv["alphas"].items(), key=lambda x: x[1]["delta_acc"])
            print(f"  {strat}: best Δ={best_v['delta_acc']:+.2f}% @ α={best_a}")


if __name__ == "__main__":
    main()
