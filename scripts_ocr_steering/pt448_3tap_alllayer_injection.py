#!/usr/bin/env python3
"""
3-tap all-layer injection for pt-448 — exact mirror of the ablation.

The ablation script removes the feature direction at 3 taps × 26 layers = 78
intervention points per forward pass:
    for l in range(26):
        attn_out -= (attn_out @ fv.T) * fv   # project out at attn output
        mlp_out  -= (mlp_out  @ fv.T) * fv   # project out at MLP output
        layer_out -= (layer_out @ fv.T) * fv  # project out at residual stream

This script does the reverse — ADDS the feature direction at the same 78 points:
    for l in range(26):
        attn_out  += alpha * ones * fv        # inject at attn output
        mlp_out   += alpha * ones * fv        # inject at MLP output
        layer_out += alpha * ones * fv        # inject at residual stream

The ablation produces -6% to -31% drops. By symmetry, injection at the same
scale should produce the maximum possible positive signal in pt-448.

Why this matters: our single-layer injection (pt448_feature_injection.py) only
touches layer_out at one layer — 1/78 of the intervention points. This script
tests all 78, which should give much stronger effects even at smaller alpha.

Alpha range is smaller since 78 simultaneous injections accumulate:
    alpha=1.0 here ≈ alpha=78 in single-layer injection (rough upper bound)

Adds log-odds margin as a continuous metric alongside binary accuracy.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_3tap_alllayer_injection/

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 pt448_3tap_alllayer_injection.py
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
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_3tap_alllayer_injection")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Small alphas because 78 taps accumulate
# alpha=0.1 at 78 taps ≈ alpha=7.8 effective; alpha=1.0 ≈ alpha=78 effective
INJECTION_ALPHAS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

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
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n
    p_yes = max(y / d if d > 0 else 0.5, 1e-7)
    p_no  = max(1.0 - p_yes, 1e-7)
    margin = math.log(p_yes / p_no)  # positive = model predicts Yes
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
                correct_margin = margin if label == 1 else -margin
                margins.append(correct_margin)
            except Exception:
                pred = 0; margins.append(0.0)
            total += 1
            correct += (pred == label)
        acc = correct / max(total, 1) * 100
        mean_margin = sum(margins) / max(len(margins), 1)
        return acc, mean_margin, total

    def run_vsr_3tap_alllayer(indices, fv, alpha):
        """
        Inject alpha * fv at attn_out, mlp_out, AND layer_out for ALL 26 layers.
        Exact mirror of the 3-point projection ablation.
        Text tokens only (img_end: onwards).
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
                    for l in range(N_LAYERS):
                        # Tap 1: attention output
                        attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                        ones_a   = (attn_out @ fv_col) * 0.0 + 1.0
                        attn_out += alpha * ones_a * fv

                        # Tap 2: MLP output
                        mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                        ones_m  = (mlp_out @ fv_col) * 0.0 + 1.0
                        mlp_out += alpha * ones_m * fv

                        # Tap 3: residual stream (layer output)
                        layer_out = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        ones_r    = (layer_out @ fv_col) * 0.0 + 1.0
                        layer_out += alpha * ones_r * fv

                    logits_s = nns_model.output.logits.save()

                pred, margin = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
                correct_margin = margin if label == 1 else -margin
                margins.append(correct_margin)
            except Exception as e:
                pred = 0; margins.append(0.0)
            total += 1
            correct += (pred == label)

        acc = correct / max(total, 1) * 100
        mean_margin = sum(margins) / max(len(margins), 1)
        return acc, mean_margin, total

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations in TOP10:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"3tap_{key}.json"
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

        # Load SAE feature direction from mix-448 checkpoint
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
            "method": "3tap_alllayer_injection",
            "n_taps": 3, "n_layers": N_LAYERS,
            "total_intervention_points": 3 * N_LAYERS,
            "alphas": {}
        }

        for alpha in INJECTION_ALPHAS:
            print(f"[INJECT] {key} α={alpha:+g} (3tap×26layers, N={len(indices)})...", flush=True)
            acc, margin, n = run_vsr_3tap_alllayer(indices, fv, alpha)
            delta_acc    = acc    - base_acc
            delta_margin = margin - base_margin
            result["alphas"][str(alpha)] = {
                "acc": acc, "delta_acc": delta_acc,
                "margin": margin, "delta_margin": delta_margin, "n": n
            }
            print(f"  α={alpha:+g}: {acc:.2f}% (Δ={delta_acc:+.2f}%)  "
                  f"margin={margin:.3f} (Δ={delta_margin:+.3f})", flush=True)

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    # Summary table
    print(f"\n{'='*110}")
    print("pt-448 3-Tap All-Layer Injection (mirror of ablation: 3×26=78 points per forward pass)")
    print(f"{'='*110}")
    header = f"{'L/F':<12} {'Relations':<32} {'N':>5} {'Base':>7} {'Xfer':>6}"
    for a in INJECTION_ALPHAS:
        header += f"  {a:>+6}"
    print(header)
    print("-" * 110)

    xfer_map = {
        (9,  387):   0.07, (14, 10561): 0.41, (11, 12278): 0.40,
        (9,  7540):  0.25, (4,  14233): 1.00, (6,  7539):  0.48,
        (11, 9639):  0.24, (13, 15219): 0.00, (15, 220):   0.27,
        (12, 2257):  0.00,
    }

    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rels = "; ".join(r["relations"])[:31]
        xfer = xfer_map.get((r["layer"], r["feature"]), -1)
        row = f"{key:<12} {rels:<32} {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}% {xfer:>5.2f}×"
        for a in INJECTION_ALPHAS:
            d = r["alphas"].get(str(a), {}).get("delta_acc")
            row += f"  {d:>+6.2f}" if d is not None else f"  {'--':>6}"
        print(row)

    import csv
    csv_path = OUT_DIR / "3tap_alllayer_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer", "feature", "relations", "n_samples",
                      "transfer_ratio", "baseline_vsr_acc", "baseline_margin"]
        for a in INJECTION_ALPHAS:
            fieldnames += [f"delta_acc_alpha_{a}", f"delta_margin_alpha_{a}"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            xfer = xfer_map.get((r["layer"], r["feature"]), -1)
            row = {
                "layer": r["layer"], "feature": r["feature"],
                "relations": "; ".join(r["relations"]),
                "n_samples": r["n_samples"],
                "transfer_ratio": xfer,
                "baseline_vsr_acc": f"{r['baseline_vsr_acc']:.2f}",
                "baseline_margin": f"{r.get('baseline_margin', 0):.4f}",
            }
            for a in INJECTION_ALPHAS:
                v = r["alphas"].get(str(a), {})
                row[f"delta_acc_alpha_{a}"]    = f"{v.get('delta_acc', ''):}" if v else ""
                row[f"delta_margin_alpha_{a}"] = f"{v.get('delta_margin', ''):}" if v else ""
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
