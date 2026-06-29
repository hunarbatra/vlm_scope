#!/usr/bin/env python3
"""
Fine-grained decay_fwd sweep for the 4 high-transfer features (≥0.40×).

Rationale: GPU 5 (residual_alllayer) showed decay_fwd α=2 → +3.12% for L9/F387
(0.07× transfer). High-transfer features (L4/F14233 1.00×, L6/F7539 0.48×,
L14/F10561 0.41×, L11/F12278 0.40×) may respond to different alpha ranges
since they already encode the feature at a higher level.

This script tests:
  - decay_fwd: inject all 26 layers, alpha * 0.85^max(l-sae_layer,0) at each layer
  - sae_only_down: inject only from SAE layer to 25 (downstream propagation)
with a fine-grained alpha grid [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

Single model on GPU 7 only — no interference.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_highxfer_decayfwd/

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 pt448_highxfer_decayfwd_sweep.py
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
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_highxfer_decayfwd")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Fine-grained alpha grid around the known optimal (decay_fwd α=2 best so far)
INJECTION_ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
STRATEGIES = ["decay_fwd", "sae_only_down"]
DECAY = 0.85

# Only high-transfer features (≥0.40× cross-stage transfer ratio)
HIGH_XFER = [
    (4,  14233, ["ahead of"],          1.00),
    (6,  7539,  ["left of", "right of"], 0.48),
    (14, 10561, ["close to"],           0.41),
    (11, 12278, ["touching"],           0.40),
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

def _layer_weights(strategy, sae_layer, n_layers, decay):
    if strategy == "decay_fwd":
        return {l: decay ** max(l - sae_layer, 0) for l in range(n_layers)}
    elif strategy == "sae_only_down":
        return {l: 1.0 for l in range(sae_layer, n_layers)}
    return {l: 1.0 for l in range(n_layers)}


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {MODEL_NAME} (high-xfer sweep, GPU 7)...", flush=True)
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

    for layer_idx, feature_idx, relations, xfer_ratio in HIGH_XFER:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"highxfer_{key}.json"
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

        result = {
            "layer": layer_idx, "feature": feature_idx, "relations": relations,
            "transfer_ratio": xfer_ratio, "n_samples": len(indices),
            "baseline_vsr_acc": base_acc, "baseline_margin": base_mg, "strategies": {}
        }

        fv_col = fv.unsqueeze(1)
        for strategy in STRATEGIES:
            lw = _layer_weights(strategy, layer_idx, N_LAYERS, DECAY)
            print(f"\n[STRAT] {key} strategy={strategy} xfer={xfer_ratio:.2f}×", flush=True)
            strat_res = {"alphas": {}, "n_layers_active": sum(1 for w in lw.values() if w > 1e-6)}

            for alpha in INJECTION_ALPHAS:
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
                strat_res["alphas"][str(alpha)] = {
                    "acc": acc, "delta_acc": da, "margin": mg, "delta_margin": dm
                }
                print(f"    α={alpha:+g}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f} (Δ={dm:+.3f})", flush=True)

            result["strategies"][strategy] = strat_res

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n{'='*120}")
    print("pt-448 High-Transfer Fine-Grained Sweep (decay_fwd + sae_only_down)")
    print(f"{'='*120}")
    header = f"{'L/F':<12} {'Relations':<28} {'Xfer':>5} {'N':>5} {'Base':>7}"
    for s in STRATEGIES: header += f"  best_{s[:8]:>10}"
    print(header); print("-"*120)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rels = "; ".join(r["relations"])[:27]
        row = f"{key:<12} {rels:<28} {r['transfer_ratio']:>4.2f}× {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}%"
        for s in STRATEGIES:
            best = max((v["delta_acc"] for v in r["strategies"].get(s,{}).get("alphas",{}).values()), default=None)
            best_a = max(r["strategies"].get(s,{}).get("alphas",{}).items(),
                        key=lambda x: x[1]["delta_acc"], default=(None,{}))[0] if best else None
            row += f"  {best:>+8.2f}@α={best_a}" if best is not None else f"  {'--':>12}"
        print(row)

    import csv
    csv_path = OUT_DIR / "highxfer_decayfwd_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer","feature","relations","transfer_ratio","n_samples","baseline_vsr_acc"]
        for s in STRATEGIES:
            for a in INJECTION_ALPHAS: fieldnames += [f"{s}_a{a}_dacc", f"{s}_a{a}_dmargin"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in all_results:
            row = {"layer": r["layer"], "feature": r["feature"],
                   "relations": "; ".join(r["relations"]),
                   "transfer_ratio": r["transfer_ratio"],
                   "n_samples": r["n_samples"],
                   "baseline_vsr_acc": f"{r['baseline_vsr_acc']:.2f}"}
            for s in STRATEGIES:
                for a in INJECTION_ALPHAS:
                    v = r["strategies"].get(s, {}).get("alphas", {}).get(str(a), {})
                    row[f"{s}_a{a}_dacc"] = f"{v.get('delta_acc', ''):.2f}" if v else ""
                    row[f"{s}_a{a}_dmargin"] = f"{v.get('delta_margin', ''):.3f}" if v else ""
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
