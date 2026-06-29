#!/usr/bin/env python3
"""
Multi-layer W_dec injection for pt-448 using mix-448 spatial feature directions.

Improvement over pt448_feature_injection.py: instead of injecting only at the
native SAE layer, this script tests multiple injection strategies:

  strategy="single"      — inject only at layer_L (same as original)
  strategy="downstream"  — inject at all layers L..25 with alpha decay
  strategy="all"         — inject at all layers 0..25 with decay from L
  strategy="answer"      — inject at last 5 layers (21..25) only
  strategy="topK"        — inject at layer_L and ±2 neighbouring layers

Decay model (for multi-layer strategies):
    effective_alpha(l) = alpha * decay ^ |l - L_sae|
    decay = 0.7 by default (tunable)

This captures the intuition that the feature's causal effect is strongest at
its native SAE layer but propagates downstream through residual additions.

Also adds log-odds margin as a secondary metric (continuous signal vs. binary
accuracy) to detect sub-threshold gains that don't flip enough examples.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_multilayer_injection/

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 pt448_multilayer_injection.py
"""

import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
from collections import defaultdict
import math

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_NAME     = "google/paligemma2-3b-pt-448"
N_LAYERS       = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_multilayer_injection")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Finer grid focused on the regime where single-layer injection worked best (+5-20)
# Plus extended range to catch peaks for slow-rising features like "touching"
INJECTION_ALPHAS = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]

# Decay factor per layer away from SAE layer
LAYER_DECAY = 0.7

# Injection strategies
STRATEGIES = ["single", "downstream", "all", "answer", "topK"]

TOP10 = [
    (9,  387,   ["at the right side of"]),
    (14, 10561, ["close to"]),
    (11, 12278, ["touching"]),
    (9,  7540,  ["consists of"]),
    (4,  14233, ["ahead of"]),
    (6,  7539,  ["left of", "right of"]),
    (11, 9639,  ["in", "inside", "on"]),
    (13, 15219, ["behind"]),
    (15, 220,   ["across from", "at the left side of"]),
    (12, 2257,  ["facing"]),
]


def _build_vsr_prompt(statement):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\nAnswer:"
    )


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap; no_ids -= overlap
    return yes_ids, no_ids


def _predict_and_margin(logits, yes_ids, no_ids):
    """Returns (prediction: 0/1, log_odds_margin: float)."""
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n
    p_yes = y / d if d > 0 else 0.5
    p_yes = max(p_yes, 1e-7)
    p_no  = 1.0 - p_yes
    p_no  = max(p_no,  1e-7)
    margin = math.log(p_yes / p_no)   # positive = predicts Yes
    return (1 if p_yes > 0.5 else 0), margin


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists():
            return Image.open(cp).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None


def _get_injection_layers(strategy, sae_layer, n_layers=26):
    """Return list of (layer_idx, alpha_scale) pairs for a given strategy."""
    if strategy == "single":
        return [(sae_layer, 1.0)]
    elif strategy == "downstream":
        # Inject from SAE layer to end, with decay
        return [(l, LAYER_DECAY ** (l - sae_layer)) for l in range(sae_layer, n_layers)]
    elif strategy == "all":
        # Inject at all layers, decay from SAE layer
        return [(l, LAYER_DECAY ** abs(l - sae_layer)) for l in range(n_layers)]
    elif strategy == "answer":
        # Last 5 layers only — where answer token is assembled
        return [(l, 1.0) for l in range(max(0, n_layers - 5), n_layers)]
    elif strategy == "topK":
        # SAE layer ±2
        layers = [sae_layer + d for d in [-2, -1, 0, 1, 2] if 0 <= sae_layer + d < n_layers]
        return [(l, LAYER_DECAY ** abs(l - sae_layer)) for l in layers]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


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
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    print("[INFO] Loading VSR (all splits)...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    def get_filtered_indices(relations):
        idxs = []
        for r in relations:
            idxs.extend(relation_indices.get(r, []))
        return idxs

    def run_vsr_baseline(indices):
        correct = total = 0
        margins = []
        for vi in indices:
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model_raw, device=device
                )
                with torch.inference_mode():
                    out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                   pixel_values=pixel_values, use_cache=False)
                pred, margin = _predict_and_margin(out.logits[0, -1, :], yes_ids, no_ids)
                # positive margin = predicts Yes; label=1 means correct = positive margin
                correct_margin = margin if label == 1 else -margin
                margins.append(correct_margin)
            except Exception:
                pred = 0
                margins.append(0.0)
            total += 1
            correct += (pred == label)
        acc = correct / max(total, 1) * 100
        mean_margin = sum(margins) / max(len(margins), 1)
        return acc, mean_margin, total

    def run_vsr_injected(indices, fv, layer_scales, alpha):
        """
        Inject alpha * layer_scale[l] * fv at each layer l in layer_scales.
        layer_scales: list of (layer_idx, scale_factor)
        """
        correct = total = 0
        margins = []
        fv_col = fv.unsqueeze(1)  # (d, 1)
        for vi in indices:
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, nns_model._module, device=device
                )
                _, img_end = get_image_token_positions(input_ids)
                with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                     pixel_values=pixel_values):
                    for l_idx, l_scale in layer_scales:
                        layer_out = nns_model.model.language_model.layers[l_idx].output[0][0, img_end:]
                        ones = (layer_out @ fv_col) * 0.0 + 1.0
                        layer_out += (alpha * l_scale) * ones * fv
                    logits_s = nns_model.output.logits.save()
                pred, margin = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
                correct_margin = margin if label == 1 else -margin
                margins.append(correct_margin)
            except Exception:
                pred = 0
                margins.append(0.0)
            total += 1
            correct += (pred == label)
        acc = correct / max(total, 1) * 100
        mean_margin = sum(margins) / max(len(margins), 1)
        return acc, mean_margin, total

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations in TOP10:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"multilayer_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key} already done", flush=True)
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        indices = get_filtered_indices(relations)
        if not indices:
            print(f"[WARN] {key}: no VSR samples for {relations}", flush=True)
            continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            print(f"[BASE] pt-448 baseline for [{rel_key}] (N={len(indices)})...", flush=True)
            acc, margin, n = run_vsr_baseline(indices)
            baseline_cache[rel_key] = (acc, margin, n)
            print(f"[BASE] Baseline: {acc:.2f}% margin={margin:.3f} (N={n})", flush=True)
        base_acc, base_margin, base_n = baseline_cache[rel_key]

        # Load SAE feature direction
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "relations": relations, "n_samples": len(indices),
            "baseline_vsr_acc": base_acc,
            "baseline_margin": base_margin,
            "strategies": {}
        }

        for strategy in STRATEGIES:
            layer_scales = _get_injection_layers(strategy, layer_idx, N_LAYERS)
            print(f"\n[STRAT] {key} strategy={strategy} layers={[l for l,_ in layer_scales]}", flush=True)
            strat_result = {"layer_scales": [(l, round(s, 4)) for l, s in layer_scales], "alphas": {}}

            for alpha in INJECTION_ALPHAS:
                print(f"  [INJECT] {key} strategy={strategy} α={alpha:+g} (N={len(indices)})...", flush=True)
                acc, margin, n = run_vsr_injected(indices, fv, layer_scales, alpha)
                delta_acc    = acc - base_acc
                delta_margin = margin - base_margin
                strat_result["alphas"][str(alpha)] = {
                    "acc": acc, "delta_acc": delta_acc,
                    "margin": margin, "delta_margin": delta_margin,
                    "n": n
                }
                print(f"    α={alpha:+g}: acc={acc:.2f}% (Δ={delta_acc:+.2f}%) "
                      f"margin={margin:.3f} (Δ={delta_margin:+.3f})", flush=True)

            result["strategies"][strategy] = strat_result

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    # Summary table
    print(f"\n{'='*130}")
    print("pt-448 Multi-Layer Injection — Best delta per strategy (accuracy %, Δ vs baseline)")
    print(f"{'='*130}")
    header = f"{'L/F':<12} {'Relations':<32} {'N':>5} {'Base':>7}"
    for s in STRATEGIES:
        header += f"  {s:>12}"
    print(header)
    print("-" * 130)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rels = "; ".join(r["relations"])[:31]
        row = f"{key:<12} {rels:<32} {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}%"
        for s in STRATEGIES:
            best = max(
                (v["delta_acc"] for v in r["strategies"].get(s, {}).get("alphas", {}).values()),
                default=None
            )
            row += f"  {best:>+12.2f}" if best is not None else f"  {'--':>12}"
        print(row)

    import csv
    csv_path = OUT_DIR / "multilayer_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer", "feature", "relations", "n_samples", "baseline_vsr_acc", "baseline_margin"]
        for s in STRATEGIES:
            for a in INJECTION_ALPHAS:
                fieldnames.append(f"{s}_alpha_{a}_delta_acc")
                fieldnames.append(f"{s}_alpha_{a}_delta_margin")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {
                "layer": r["layer"], "feature": r["feature"],
                "relations": "; ".join(r["relations"]),
                "n_samples": r["n_samples"],
                "baseline_vsr_acc": f"{r['baseline_vsr_acc']:.2f}",
                "baseline_margin": f"{r.get('baseline_margin', 0):.4f}",
            }
            for s in STRATEGIES:
                for a in INJECTION_ALPHAS:
                    v = r["strategies"].get(s, {}).get("alphas", {}).get(str(a), {})
                    row[f"{s}_alpha_{a}_delta_acc"]    = f"{v.get('delta_acc', ''):}" if v else ""
                    row[f"{s}_alpha_{a}_delta_margin"] = f"{v.get('delta_margin', ''):}" if v else ""
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
