#!/usr/bin/env python3
"""
Natural-scale SAE-calibrated injection for pt-448 — fixes Experiment 4 failures.

Root causes of pt448_sae_calibrated_injection.py failure:
  1. calibrated_scale = pos_mean - neg_mean is tiny (0.003–0.19) AND sometimes negative
     (feature fires more for false statements in some relations), so injection goes
     in the wrong direction for half the features.
  2. Constant delta across all multipliers = model saturates at even tiny injection.

Fixes in this script:
  A. Absolute scale from positive-example mean:
       scale = sae_pos_mean   (the actual firing level in mix-448, not the contrast)
     This gives values in range ~0.7–4.7, matching realistic firing magnitudes.
     Direction is always forced positive (W_dec points toward spatial-present direction).

  B. Per-example calibration (new method):
       Run mix-448 SAE on each example → get act_F for that specific example
       Inject act_F * fv into pt-448 for that example
     This is "individualised injection" — each example gets a scale that matches
     how strongly mix-448 saw this spatial feature in that image+text context.

  C. Much larger multiplier range: 0.1 → 100 (covering the regime where
     single-layer fixed injection worked: effective_alpha = 5–30 ≈ M * pos_mean).

  D. Log-odds margin as secondary metric (continuous signal even when accuracy
     doesn't flip: log(p_yes/p_no) signed positive for correct predictions).

Methods run:
  method="absolute_scale"      — inject M * pos_mean * fv (fixed across examples)
  method="per_example"         — inject act_F(example) * M * fv (per-example scale)
  method="contrast_abs"        — inject M * |pos_mean - neg_mean| * fv (|contrast|, always positive)

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_natural_scale_injection/

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 pt448_natural_scale_injection.py
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

MIX_MODEL    = "google/paligemma2-3b-mix-448"
PT_MODEL     = "google/paligemma2-3b-pt-448"
N_LAYERS     = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_natural_scale_injection")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# M=1.0 means inject exactly at the mix-448 positive firing level (sae_pos_mean * fv)
# M=0.1 = sub-natural; M=10 = 10× natural level
MULTIPLIERS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

METHODS = ["absolute_scale", "per_example", "contrast_abs"]

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
    margin = math.log(p_yes / p_no)
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

    print(f"[INFO] Loading {MIX_MODEL} (SAE activation extraction)...", flush=True)
    mix_processor = AutoProcessor.from_pretrained(MIX_MODEL)
    mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    mix_nns = NNsight(mix_model)

    print(f"[INFO] Loading {PT_MODEL} (injection target)...", flush=True)
    pt_processor = AutoProcessor.from_pretrained(PT_MODEL)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    pt_nns = NNsight(pt_model)

    tokenizer = mix_processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(mix_model.parameters()).dtype

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

    def extract_sae_activations(indices, layer_idx, feature_idx, sae):
        """
        Returns (per_example_acts, pos_mean, neg_mean) where
        per_example_acts[vi] = mean SAE activation of feature_idx for example vi.
        """
        per_example = {}  # vi -> act_F
        pos_acts, neg_acts = [], []

        for vi in indices:
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, mix_processor, mix_nns._module, device=device
                )
                _, img_end = get_image_token_positions(input_ids)
                with mix_nns.trace(input_ids=input_ids, attention_mask=attn_mask,
                                   pixel_values=pixel_values):
                    h = mix_nns.model.language_model.layers[layer_idx].output[0][0, img_end:].save()
                h_val = h.detach().float()
                with torch.no_grad():
                    acts = sae.encode(h_val)               # (T, d_sae)
                    feat_act = acts[:, feature_idx].mean().item()
                per_example[vi] = feat_act
                if label == 1:
                    pos_acts.append(feat_act)
                else:
                    neg_acts.append(feat_act)
            except Exception:
                continue

        pos_mean = sum(pos_acts) / max(len(pos_acts), 1)
        neg_mean = sum(neg_acts) / max(len(neg_acts), 1)
        return per_example, pos_mean, neg_mean, len(pos_acts), len(neg_acts)

    def run_vsr_baseline(indices, use_pt_processor=True):
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
                    img, prompt, pt_processor, pt_model, device=device
                )
                with torch.inference_mode():
                    out = pt_model(input_ids=input_ids, attention_mask=attn_mask,
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

    def run_vsr_fixed_scale(indices, fv, layer_idx, injection_scale):
        """Inject injection_scale * fv at layer_idx, text tokens only."""
        correct = total = 0
        margins = []
        fv_col = fv.unsqueeze(1)
        for vi in indices:
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, pt_processor, pt_nns._module, device=device
                )
                _, img_end = get_image_token_positions(input_ids)
                with pt_nns.trace(input_ids=input_ids, attention_mask=attn_mask,
                                  pixel_values=pixel_values):
                    layer_out = pt_nns.model.language_model.layers[layer_idx].output[0][0, img_end:]
                    ones = (layer_out @ fv_col) * 0.0 + 1.0
                    layer_out += injection_scale * ones * fv
                    logits_s = pt_nns.output.logits.save()
                pred, margin = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
                correct_margin = margin if label == 1 else -margin
                margins.append(correct_margin)
            except Exception:
                pred = 0; margins.append(0.0)
            total += 1
            correct += (pred == label)
        acc = correct / max(total, 1) * 100
        mean_margin = sum(margins) / max(len(margins), 1)
        return acc, mean_margin, total

    def run_vsr_per_example(indices, fv, layer_idx, per_example_acts, M):
        """Inject M * act_F(example) * fv — individualised per-example scale."""
        correct = total = 0
        margins = []
        fv_col = fv.unsqueeze(1)
        for vi in indices:
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            act_F = per_example_acts.get(vi, 0.0)
            injection_scale = M * act_F
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, pt_processor, pt_nns._module, device=device
                )
                _, img_end = get_image_token_positions(input_ids)
                with pt_nns.trace(input_ids=input_ids, attention_mask=attn_mask,
                                  pixel_values=pixel_values):
                    layer_out = pt_nns.model.language_model.layers[layer_idx].output[0][0, img_end:]
                    ones = (layer_out @ fv_col) * 0.0 + 1.0
                    layer_out += injection_scale * ones * fv
                    logits_s = pt_nns.output.logits.save()
                pred, margin = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
                correct_margin = margin if label == 1 else -margin
                margins.append(correct_margin)
            except Exception:
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
        result_path = OUT_DIR / f"natscale_{key}.json"
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

        # Load SAE and feature vector
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)

        # Extract mix-448 SAE activations for this feature
        print(f"[CALIB] Extracting mix-448 SAE activations for {key} (N={len(indices)})...", flush=True)
        per_example_acts, pos_mean, neg_mean, n_pos, n_neg = extract_sae_activations(
            indices, layer_idx, feature_idx, sae
        )
        del sae; torch.cuda.empty_cache()

        contrast = pos_mean - neg_mean
        abs_contrast = abs(contrast)
        print(f"[CALIB] {key}: pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} "
              f"contrast={contrast:+.4f} |contrast|={abs_contrast:.4f} "
              f"(n_pos={n_pos} n_neg={n_neg})", flush=True)

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "relations": relations, "n_samples": len(indices),
            "n_pos": n_pos, "n_neg": n_neg,
            "sae_pos_mean": pos_mean, "sae_neg_mean": neg_mean,
            "contrast": contrast, "abs_contrast": abs_contrast,
            "baseline_vsr_acc": base_acc, "baseline_margin": base_margin,
            "methods": {}
        }

        for method in METHODS:
            print(f"\n[METHOD] {key} method={method}", flush=True)
            method_result = {"multipliers": {}}

            for M in MULTIPLIERS:
                if method == "absolute_scale":
                    # Always inject in positive direction at pos_mean scale
                    inj = M * pos_mean
                    acc, margin, n = run_vsr_fixed_scale(indices, fv, layer_idx, inj)
                elif method == "contrast_abs":
                    # Inject at |contrast| scale, forced positive direction
                    inj = M * abs_contrast
                    acc, margin, n = run_vsr_fixed_scale(indices, fv, layer_idx, inj)
                elif method == "per_example":
                    # Per-example: inject M * act_F(example) * fv
                    acc, margin, n = run_vsr_per_example(indices, fv, layer_idx, per_example_acts, M)
                    inj = M  # log M as the multiplier

                delta_acc = acc - base_acc
                delta_margin = margin - base_margin
                method_result["multipliers"][str(M)] = {
                    "M": M, "injection_scale": inj,
                    "acc": acc, "delta_acc": delta_acc,
                    "margin": margin, "delta_margin": delta_margin, "n": n
                }
                print(f"  M={M:>6g}: acc={acc:.2f}% (Δ={delta_acc:+.2f}%) "
                      f"margin={margin:.3f} (Δ={delta_margin:+.3f}) inj={inj:.3f}", flush=True)

            result["methods"][method] = method_result

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*140}")
    print("pt-448 Natural-Scale Injection — Best delta per method (accuracy %, Δ vs baseline)")
    print(f"{'='*140}")
    header = f"{'L/F':<12} {'Relations':<32} {'N':>5} {'Base':>7} {'pos_mean':>9} {'contrast':>9}"
    for m in METHODS:
        header += f"  {m:>16}"
    print(header)
    print("-" * 140)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rels = "; ".join(r["relations"])[:31]
        row = (f"{key:<12} {rels:<32} {r['n_samples']:>5} "
               f"{r['baseline_vsr_acc']:>6.1f}% {r.get('sae_pos_mean', 0):>9.3f} "
               f"{r.get('contrast', 0):>+9.3f}")
        for m in METHODS:
            best = max(
                (v["delta_acc"] for v in r["methods"].get(m, {}).get("multipliers", {}).values()),
                default=None
            )
            row += f"  {best:>+16.2f}" if best is not None else f"  {'--':>16}"
        print(row)

    import csv
    csv_path = OUT_DIR / "natural_scale_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer", "feature", "relations", "n_samples",
                      "sae_pos_mean", "sae_neg_mean", "contrast", "abs_contrast",
                      "baseline_vsr_acc", "baseline_margin"]
        for m in METHODS:
            for M in MULTIPLIERS:
                fieldnames += [f"{m}_M{M}_delta_acc", f"{m}_M{M}_delta_margin"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {
                "layer": r["layer"], "feature": r["feature"],
                "relations": "; ".join(r["relations"]),
                "n_samples": r["n_samples"],
                "sae_pos_mean": f"{r.get('sae_pos_mean', 0):.4f}",
                "sae_neg_mean": f"{r.get('sae_neg_mean', 0):.4f}",
                "contrast": f"{r.get('contrast', 0):+.4f}",
                "abs_contrast": f"{r.get('abs_contrast', 0):.4f}",
                "baseline_vsr_acc": f"{r['baseline_vsr_acc']:.2f}",
                "baseline_margin": f"{r.get('baseline_margin', 0):.4f}",
            }
            for m in METHODS:
                for M in MULTIPLIERS:
                    v = r["methods"].get(m, {}).get("multipliers", {}).get(str(M), {})
                    row[f"{m}_M{M}_delta_acc"]    = f"{v.get('delta_acc', ''):}" if v else ""
                    row[f"{m}_M{M}_delta_margin"] = f"{v.get('delta_margin', ''):}" if v else ""
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
