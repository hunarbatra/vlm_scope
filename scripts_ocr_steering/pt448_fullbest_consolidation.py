#!/usr/bin/env python3
"""
Consolidation run: apply the single best method per feature, all 10 features.

Takes the overall winner per feature from all prior experiments and runs it
in one clean script for a final, definitive result table.

Best method per feature (from grand leaderboard):
  L4/F14233  ahead of       sae_only_down  α=4.0  (+10.26%)
  L12/F2257  facing         all-26L 0.7    α=50.0 (+3.92%)
  L11/F12278 touching       single layer   α=20.0 (+3.36% via precomp ~ +3.20% single)
  L9/F387    right side of  decay_fwd_ra   α=2.0  (+3.12%)
  L15/F220   across from    sae_only_up    α=5.0  (+3.11% approx)
  L13/F15219 behind         downstream 0.7 α=30.0 (+2.12%)
  L9/F7540   consists of    single         α=10.0 (+2.86%)
  L14/F10561 close to       all-26L 0.7    α=2.0  (+2.15%)
  L6/F7539   left/right of  topK ±2 0.7    α=20.0 (+1.24%)
  L11/F9639  in/inside/on   answer L21-25  α=10.0 (+0.73%)

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fullbest/

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 pt448_fullbest_consolidation.py
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
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fullbest")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

DECAY_ML = 0.7
DECAY_RA = 0.85

# (layer, feature, relations, strategy, alpha, prior_best_delta, xfer_ratio)
BEST_PER_FEATURE = [
    (4,  14233, ["ahead of"],                         "sae_only_down",  4.0,  +10.26, 1.00),
    (12, 2257,  ["facing"],                           "all_ml",        50.0,   +3.92, 0.00),
    (11, 12278, ["touching"],                         "single",        20.0,   +3.20, 0.40),
    (9,  387,   ["at the right side of"],             "decay_fwd_ra",   2.0,   +3.12, 0.07),
    (15, 220,   ["across from","at the left side of"],"sae_only_up",    5.0,   +3.11, 0.27),
    (13, 15219, ["behind"],                           "downstream_ml", 30.0,   +2.12, 0.00),
    (9,  7540,  ["consists of"],                      "single",        10.0,   +2.86, 0.25),
    (14, 10561, ["close to"],                         "all_ml",         2.0,   +2.15, 0.41),
    (6,  7539,  ["left of","right of"],               "topK_ml",       20.0,   +1.24, 0.48),
    (11, 9639,  ["in","inside","on"],                 "answer",        10.0,   +0.73, 0.24),
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

    for layer_idx, feature_idx, relations, strategy, alpha, prior_best, xfer in BEST_PER_FEATURE:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"best_{key}.json"
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

        lw = _get_layer_weights(strategy, layer_idx)
        fv_col = fv.unsqueeze(1)

        print(f"[INJECT] {key} {strategy} α={alpha:+g} xfer={xfer}x (prior_best={prior_best:+.2f}%)...", flush=True)
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
        print(f"  {key}: {acc:.2f}% (Δ={da:+.2f}%, prior={prior_best:+.2f}%) margin={mg:.3f} (Δ={dm:+.3f})", flush=True)

        result = {
            "layer": layer_idx, "feature": feature_idx, "relations": relations,
            "strategy": strategy, "alpha": alpha, "transfer_ratio": xfer,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "acc": acc, "delta_acc": da, "margin": mg, "delta_margin": dm,
            "prior_best_delta": prior_best, "replicated": abs(da - prior_best) < 1.0
        }
        with open(result_path,"w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n{'='*110}")
    print("pt-448 Best Method Per Feature — Consolidation Run")
    print(f"{'='*110}")
    header = f"{'L/F':<14} {'Relation':<28} {'N':>5} {'base':>7} {'α':>6} {'strat':<16} {'xfer':>5} {'Δacc':>7} {'prior':>7} {'match':>6}"
    print(header); print("-"*110)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rel = "; ".join(r["relations"])[:27]
        match = "✓" if r.get("replicated") else "✗"
        row = (f"{key:<14} {rel:<28} {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}% "
               f"{r['alpha']:>5.1f} {r['strategy']:<16} {r['transfer_ratio']:>4.2f}× "
               f"{r['delta_acc']:>+6.2f}% {r['prior_best_delta']:>+6.2f}% {match:>6}")
        print(row)

    import csv
    csv_path = OUT_DIR / "fullbest_summary.csv"
    with open(csv_path,"w",newline="") as f:
        fields = ["layer","feature","relations","strategy","alpha","transfer_ratio",
                  "n_samples","baseline_vsr_acc","delta_acc","delta_margin","prior_best_delta","replicated"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k,"") for k in fields})
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
