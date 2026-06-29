#!/usr/bin/env python3
"""
SAE-calibrated feature injection for pt-448.

For each top-10 feature (layer_L, feature_F, relations_R):
  1. Filter VSR → subset S_R
  2. Run mix-448 SAE.encode(h[layer_L]) for each example → get feature F activation
  3. Compute calibrated scale:
       scale = mean(act_F | label==1) - mean(act_F | label==0)
     This is the empirical firing contrast of feature F in mix-448.
  4. Inject into pt-448 at layer_L:
       layer_out += scale * ones_proxy * W_dec[F]
     Direction = W_dec[F] (same as pt448_feature_injection.py)
     Magnitude = empirically calibrated to mix-448's actual activation difference

This combines the best of both approaches:
  - W_dec[F] direction preserves feature identity (vs. DIM's full hidden-state contrast)
  - scale is empirical (vs. manual alpha in pt448_feature_injection.py)

Also sweeps a multiplier M over the calibrated scale:
    layer_out += M * scale * ones_proxy * W_dec[F]
so M=1.0 is the "natural" mix-448 injection magnitude.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_calibrated_injection/

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 pt448_sae_calibrated_injection.py
"""

import os, sys, json, gc, hashlib, warnings
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
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_calibrated_injection")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# M=1.0 = inject exactly the empirical mix-448 contrast magnitude
# M<1 = softer, M>1 = stronger
SCALE_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

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


def _predict(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


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

    print(f"[INFO] Loading {MIX_MODEL} (for SAE activation extraction)...", flush=True)
    processor = AutoProcessor.from_pretrained(MIX_MODEL)
    mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    mix_nns = NNsight(mix_model)

    print(f"[INFO] Loading {PT_MODEL} (for injection steering)...", flush=True)
    pt_processor = AutoProcessor.from_pretrained(PT_MODEL)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    pt_nns = NNsight(pt_model)

    tokenizer = processor.tokenizer
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

    def extract_sae_calibrated_scale(indices, layer_idx, feature_idx, sae):
        """
        Collect SAE feature_F activations from mix-448 on the relation subset.
        Returns scale = mean(act_F | label==1) - mean(act_F | label==0).
        """
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
                    img, prompt, processor, mix_nns._module, device=device
                )
                _, img_end = get_image_token_positions(input_ids)
                with mix_nns.trace(input_ids=input_ids, attention_mask=attn_mask,
                                   pixel_values=pixel_values):
                    # Grab layer output hidden states for text tokens
                    h = mix_nns.model.language_model.layers[layer_idx].output[0][0, img_end:].save()
                h_val = h.detach().float()
                # Run SAE encode on the mean text-token hidden state
                with torch.no_grad():
                    acts = sae.encode(h_val)          # (T, d_sae)
                    feat_act = acts[:, feature_idx].mean().item()   # mean over text tokens
                if label == 1:
                    pos_acts.append(feat_act)
                else:
                    neg_acts.append(feat_act)
            except Exception:
                continue

        if not pos_acts or not neg_acts:
            return None, 0, 0, 0.0, 0.0

        pos_mean = sum(pos_acts) / len(pos_acts)
        neg_mean = sum(neg_acts) / len(neg_acts)
        scale    = pos_mean - neg_mean
        return scale, len(pos_acts), len(neg_acts), pos_mean, neg_mean

    def run_vsr(indices, fv=None, layer_idx=None, injection_scale=0.0):
        """Run pt-448 VSR; if fv given inject injection_scale * fv at layer_idx."""
        correct = total = 0
        for vi in indices:
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, pt_processor, pt_model if fv is None else pt_nns._module,
                    device=device
                )
                if fv is None:
                    with torch.inference_mode():
                        out = pt_model(input_ids=input_ids, attention_mask=attn_mask,
                                       pixel_values=pixel_values, use_cache=False)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                else:
                    _, img_end = get_image_token_positions(input_ids)
                    fv_col = fv.unsqueeze(1)
                    with pt_nns.trace(input_ids=input_ids, attention_mask=attn_mask,
                                      pixel_values=pixel_values):
                        layer_out = pt_nns.model.language_model.layers[layer_idx].output[0][0, img_end:]
                        ones = (layer_out @ fv_col) * 0.0 + 1.0
                        layer_out += injection_scale * ones * fv
                        logits_s = pt_nns.output.logits.save()
                    pred = _predict(logits_s[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            total += 1
            correct += (pred == label)
        return correct / max(total, 1) * 100, total

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations in TOP10:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"sae_calib_{key}.json"
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
            acc, n = run_vsr(indices)
            baseline_cache[rel_key] = (acc, n)
            print(f"[BASE] Baseline: {acc:.2f}% (N={n})", flush=True)
        base_acc, base_n = baseline_cache[rel_key]

        # Load mix-448 SAE for this layer
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)

        # Extract calibrated scale from mix-448
        print(f"[CALIB] Extracting SAE activation contrast for {key} (N={len(indices)})...", flush=True)
        scale, n_pos, n_neg, pos_mean, neg_mean = extract_sae_calibrated_scale(
            indices, layer_idx, feature_idx, sae
        )
        del sae; torch.cuda.empty_cache()

        if scale is None:
            print(f"[WARN] {key}: could not compute scale", flush=True)
            continue

        print(f"[CALIB] {key}: pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} scale={scale:.4f} "
              f"(n_pos={n_pos} n_neg={n_neg})", flush=True)

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "relations": relations, "n_samples": len(indices),
            "n_pos": n_pos, "n_neg": n_neg,
            "sae_pos_mean": pos_mean, "sae_neg_mean": neg_mean,
            "calibrated_scale": scale,
            "baseline_vsr_acc": base_acc,
            "method": "SAE_calibrated_W_dec",
            "multipliers": {}
        }

        for M in SCALE_MULTIPLIERS:
            injection = M * scale
            print(f"[INJECT] {key} M={M:+g} (injection={injection:.4f}) (N={len(indices)})...", flush=True)
            acc, n = run_vsr(indices, fv, layer_idx, injection)
            delta = acc - base_acc
            result["multipliers"][str(M)] = {
                "multiplier": M, "injection_scale": injection,
                "acc": acc, "delta": delta, "n": n
            }
            print(f"  M={M:+g} (scale={injection:.3f}): {acc:.2f}% (Δ={delta:+.2f}%)", flush=True)

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*120}")
    print("pt-448 SAE-Calibrated Injection — W_dec[F] direction, empirical mix-448 scale")
    print(f"{'='*120}")
    header = f"{'L/F':<12} {'Relations':<32} {'N':>5} {'Base':>7} {'Scale':>7}"
    for M in SCALE_MULTIPLIERS:
        header += f"  {M:>+5}x"
    print(header)
    print("-" * 120)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rels = "; ".join(r["relations"])[:31]
        row = (f"{key:<12} {rels:<32} {r['n_samples']:>5} "
               f"{r['baseline_vsr_acc']:>6.1f}% {r.get('calibrated_scale', 0):>7.3f}")
        for M in SCALE_MULTIPLIERS:
            d = r["multipliers"].get(str(M), {}).get("delta")
            row += f"  {d:>+5.1f}" if d is not None else f"  {'--':>5}"
        print(row)

    import csv
    csv_path = OUT_DIR / "sae_calibrated_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer", "feature", "relations", "n_samples",
                      "n_pos", "n_neg", "sae_pos_mean", "sae_neg_mean",
                      "calibrated_scale", "baseline_vsr_acc"]
        for M in SCALE_MULTIPLIERS:
            fieldnames.append(f"delta_M_{M}")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {
                "layer": r["layer"], "feature": r["feature"],
                "relations": "; ".join(r["relations"]),
                "n_samples": r["n_samples"],
                "n_pos": r.get("n_pos", ""), "n_neg": r.get("n_neg", ""),
                "sae_pos_mean": f"{r.get('sae_pos_mean', 0):.4f}",
                "sae_neg_mean": f"{r.get('sae_neg_mean', 0):.4f}",
                "calibrated_scale": f"{r.get('calibrated_scale', 0):.4f}",
                "baseline_vsr_acc": f"{r['baseline_vsr_acc']:.2f}",
            }
            for M in SCALE_MULTIPLIERS:
                d = r["multipliers"].get(str(M), {}).get("delta", "")
                row[f"delta_M_{M}"] = f"{d:.2f}" if d != "" else ""
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
