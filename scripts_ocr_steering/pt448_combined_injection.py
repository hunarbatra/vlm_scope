#!/usr/bin/env python3
"""
Combined multi-feature injection into pt-448.

Simultaneously inject W_dec directions for multiple spatial features in a
single forward pass. Tests whether per-feature accuracy gains are additive
or whether features interfere when injected together.

Uses best (layer, strategy, alpha) per feature discovered in prior experiments:
  L4/F14233  "ahead of"       — sae_only_down,  alpha=4.0  (+10.26%)
  L11/F12278 "touching"       — single (layer 11), alpha=20.0  (+3.20%)
  L9/F387    "right side of"  — residual decay_fwd, alpha=2.0  (+3.12%)
  L12/F2257  "facing"         — all (26L decay 0.7), alpha=50.0 (+3.92%)
  L14/F10561 "close to"       — sae_only_down, alpha=1.0  (+2.15%)

Tests:
  - Each feature in isolation (verify) at its best known alpha
  - Pairs: ahead_of + touching, ahead_of + facing, right_side + touching
  - Triple: ahead_of + touching + facing
  - All 5 simultaneously

For combined, each feature uses its own optimal (layer, strategy, alpha).

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_combined_injection/

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 pt448_combined_injection.py
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
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_combined_injection")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

DECAY_ML = 0.7  # multilayer decay constant
DECAY_RA = 0.85 # residual-alllayer decay constant

# Best config per feature: (layer, feature, relations, strategy, alpha, best_delta)
FEATURES = {
    "ahead_of":    (4,  14233, ["ahead of"],              "sae_only_down",  4.0,  +10.26),
    "touching":    (11, 12278, ["touching"],               "single",        20.0,   +3.20),
    "right_side":  (9,  387,   ["at the right side of"],  "decay_fwd_ra",   2.0,   +3.12),
    "facing":      (12, 2257,  ["facing"],                "all_ml",        50.0,   +3.92),
    "close_to":    (14, 10561, ["close to"],               "sae_only_down",  1.0,   +2.15),
}

# Relation groups to test (VSR subset for each combination)
COMBOS = [
    # name, feature_keys, relations (union)
    ("ahead_of",               ["ahead_of"],                        ["ahead of"]),
    ("touching",               ["touching"],                        ["touching"]),
    ("right_side",             ["right_side"],                      ["at the right side of"]),
    ("facing",                 ["facing"],                          ["facing"]),
    ("close_to",               ["close_to"],                        ["close to"]),
    ("ahead_touch",            ["ahead_of", "touching"],            ["ahead of", "touching"]),
    ("ahead_face",             ["ahead_of", "facing"],              ["ahead of", "facing"]),
    ("right_touch",            ["right_side", "touching"],          ["at the right side of", "touching"]),
    ("ahead_touch_face",       ["ahead_of", "touching", "facing"],  ["ahead of", "touching", "facing"]),
    ("all5",                   list(FEATURES.keys()),               ["ahead of","touching","at the right side of","facing","close to"]),
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
    """Return {layer: weight} for the given strategy."""
    if strategy == "single":
        return {sae_layer: 1.0}
    elif strategy == "sae_only_down":
        return {l: 1.0 for l in range(sae_layer, N_LAYERS)}
    elif strategy == "decay_fwd_ra":   # residual-alllayer decay_fwd (0.85)
        return {l: DECAY_RA ** max(l - sae_layer, 0) for l in range(N_LAYERS)}
    elif strategy == "all_ml":         # multilayer "all" (0.7 from sae_layer)
        return {l: DECAY_ML ** abs(l - sae_layer) for l in range(N_LAYERS)}
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

    # Pre-load all feature vectors
    print("[INFO] Loading SAE feature vectors...", flush=True)
    fvs = {}  # key -> (fv, layer_weights, alpha)
    for fname, (layer_idx, feat_idx, _, strategy, alpha, _) in FEATURES.items():
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feat_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        lw = _get_layer_weights(strategy, layer_idx)
        fvs[fname] = (fv, lw, alpha)
        del sae; torch.cuda.empty_cache()
        print(f"  Loaded {fname} (L{layer_idx}/F{feat_idx} {strategy} α={alpha})", flush=True)

    baseline_cache = {}
    all_results = []

    for combo_name, feat_keys, relations in COMBOS:
        result_path = OUT_DIR / f"combined_{combo_name}.json"
        if result_path.exists():
            print(f"[SKIP] {combo_name}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue
        indices = list(set(indices))  # deduplicate

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

        print(f"[INJECT] combo={combo_name} features={feat_keys} N={len(indices)}...", flush=True)
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
                    # Accumulate injections per layer from all active features
                    for l in range(N_LAYERS):
                        for fk in feat_keys:
                            fv, lw, alpha = fvs[fk]
                            w = lw.get(l, 0.0)
                            if w < 1e-8: continue
                            fv_col = fv.unsqueeze(1)
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
        print(f"  {combo_name}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f} (Δ={dm:+.3f})", flush=True)

        result = {
            "combo": combo_name, "features": feat_keys, "relations": relations,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "acc": acc, "delta_acc": da, "margin": mg, "delta_margin": dm,
            "feature_configs": {fk: {
                "layer": FEATURES[fk][0], "feature": FEATURES[fk][1],
                "strategy": FEATURES[fk][3], "alpha": FEATURES[fk][4],
                "individual_best": FEATURES[fk][5]
            } for fk in feat_keys}
        }
        with open(result_path,"w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("Combined Multi-Feature Injection Results")
    print(f"{'='*100}")
    header = f"{'Combo':<22} {'Features':<40} {'N':>5} {'Base':>7} {'Δacc':>7} {'Δmargin':>8}"
    print(header); print("-"*100)
    for r in all_results:
        row = (f"{r['combo']:<22} {','.join(r['features']):<40} {r['n_samples']:>5} "
               f"{r['baseline_vsr_acc']:>6.1f}% {r['delta_acc']:>+6.2f}% {r['delta_margin']:>+7.3f}")
        print(row)

    import csv
    csv_path = OUT_DIR / "combined_injection_summary.csv"
    with open(csv_path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["combo","features","n_samples","baseline_vsr_acc","delta_acc","delta_margin"])
        w.writeheader()
        for r in all_results:
            w.writerow({"combo":r["combo"],"features":";".join(r["features"]),
                        "n_samples":r["n_samples"],"baseline_vsr_acc":f"{r['baseline_vsr_acc']:.2f}",
                        "delta_acc":f"{r['delta_acc']:.2f}","delta_margin":f"{r['delta_margin']:.3f}"})
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
