#!/usr/bin/env python3
"""
Approach 2: W_dec injection scaled by mix-448 mean feature activation.

For each feature in the canonical top-8 spatial feature set:
  1. Load mix-448, compute mean SAE feature activation over the home-relation
     VSR subset (all splits), then unload mix-448.
  2. Load pt-448, inject alpha * mean_mix_act * W_dec[feature_id] at the
     home layer only, sweep alpha over ALPHAS.

Injection formula (text tokens only):
    lo += alpha * mean_mix_act * ones * wdec_vec
where ones = (lo @ wdec_vec.unsqueeze(1)) * 0.0 + 1.0  (nnsight proxy constant)

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_wdec_natural_scale/
  One JSON per feature: wdec_natural_L{layer}_F{feature}.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 pt448_wdec_natural_scale.py
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

sys.path.insert(0, str(Path(__file__).parent))

MIX_MODEL      = "google/paligemma2-3b-mix-448"
PT_MODEL       = "google/paligemma2-3b-pt-448"
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_wdec_natural_scale")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

FEATURES = [
    {"layer": 4,  "feature": 14233, "relations": ["ahead of", "behind"]},
    {"layer": 6,  "feature": 7539,  "relations": ["left of", "right of", "across from", "alongside", "at the back of", "below", "facing away from"]},
    {"layer": 9,  "feature": 387,   "relations": ["at the right side of", "adjacent to", "far from", "attached to"]},
    {"layer": 9,  "feature": 7540,  "relations": ["on", "next to", "parallel to", "in the middle of", "opposite to", "away from", "consists of"]},
    {"layer": 11, "feature": 12278, "relations": ["touching", "on top of", "surrounding", "under"]},
    {"layer": 12, "feature": 2257,  "relations": ["facing", "beneath", "near", "off", "enclosed by", "inside", "within", "beyond", "at the side of"]},
    {"layer": 14, "feature": 10561, "relations": ["close to", "by", "connected to"]},
    {"layer": 15, "feature": 220,   "relations": ["above", "at the left side of", "beside", "contains", "over", "part of", "right of", "outside", "toward"]},
]

ALPHAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_vsr_prompt(statement):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\nAnswer:"
    )


def _parse_relation(caption):
    """Return the VSR relation string from caption (not used directly; kept for symmetry)."""
    return caption


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids  -= overlap
    return yes_ids, no_ids


def _predict(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item()  if no_ids  else 1e-9
    d = y + n
    p_yes = max(y / d if d > 0 else 0.5, 1e-7)
    return 1 if p_yes > 0.5 else 0


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    h  = hashlib.md5(url.encode()).hexdigest()
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


def _load_sae_weights(layer_idx, device):
    """Load SAE checkpoint, return (W_enc, W_dec, b_enc, b_dec, threshold) tensors on CPU."""
    ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    W_enc    = state["W_enc"]    # [2304, 16384]
    W_dec    = state["W_dec"]    # [16384, 2304]
    b_enc    = state["b_enc"]    # [16384]
    b_dec    = state["b_dec"]    # [2304]
    threshold = state["threshold"]  # [16384]
    return W_enc, W_dec, b_enc, b_dec, threshold


# ---------------------------------------------------------------------------
# Phase 1: compute mean feature activation from mix-448
# ---------------------------------------------------------------------------

def compute_mean_mix_activation(
    vsr_all, relation_indices, relations,
    layer_idx, feature_idx,
    mix_nns, mix_proc, device,
    W_enc, b_enc, threshold,
):
    """
    For each sample in the home-relation VSR subset, run mix-448, extract hidden
    states at home layer (text tokens only), apply JumpReLU, get mean activation
    of feature_idx.  Returns (mean_mix_act, pct_zero, n_processed).
    """
    from utils import process_vlm_inputs, get_image_token_positions

    # Gather indices for all relations in this feature's group
    indices = []
    for r in relations:
        indices.extend(relation_indices.get(r, []))

    W_enc_col = W_enc[:, feature_idx].float().to(device)   # [2304]
    b_enc_f   = b_enc[feature_idx].float().to(device)       # scalar
    thr_f     = threshold[feature_idx].float().to(device)   # scalar

    act_list  = []
    n_zero    = 0

    for vi in indices:
        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            continue
        prompt = _build_vsr_prompt(str(ex.get("caption", "")))
        try:
            iids, attn, pv = process_vlm_inputs(
                img, prompt, mix_proc, mix_nns._module, device=device
            )
            _, img_end = get_image_token_positions(iids)

            with mix_nns.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                hidden = mix_nns.model.language_model.layers[layer_idx].output[0].save()

            h = hidden[0, img_end:, :].float()          # [T, 2304]
            pre_act = h @ W_enc_col + b_enc_f           # [T]
            act     = torch.relu(pre_act - thr_f)       # JumpReLU
            feat_act = act.mean().item()
            act_list.append(feat_act)
            if feat_act == 0.0:
                n_zero += 1
        except Exception:
            continue

    n = len(act_list)
    if n == 0:
        return 0.0, 1.0, 0, indices

    mean_mix_act  = sum(act_list) / n
    pct_zero      = n_zero / n
    return mean_mix_act, pct_zero, n, indices


# ---------------------------------------------------------------------------
# Phase 2: run pt-448 baseline and steered inference
# ---------------------------------------------------------------------------

def run_pt_baseline(indices, vsr_all, pt_nns, pt_proc, yes_ids, no_ids, device):
    from utils import process_vlm_inputs

    correct = total = 0
    for vi in indices:
        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            continue
        label  = int(ex.get("label", 0))
        prompt = _build_vsr_prompt(str(ex.get("caption", "")))
        try:
            iids, attn, pv = process_vlm_inputs(
                img, prompt, pt_proc, pt_nns._module, device=device
            )
            with torch.inference_mode():
                out = pt_nns._module(
                    input_ids=iids, attention_mask=attn,
                    pixel_values=pv, use_cache=False
                )
            pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
        except Exception:
            pred = 0
        total   += 1
        correct += (pred == label)

    return correct / max(total, 1) * 100, total


def run_pt_steered(
    indices, vsr_all, pt_nns, pt_proc, yes_ids, no_ids, device,
    layer_idx, wdec_vec, alpha, mean_mix_act,
):
    """Inject alpha * mean_mix_act * wdec_vec at home layer (text tokens only)."""
    from utils import process_vlm_inputs, get_image_token_positions

    correct = total = 0
    v_col = wdec_vec.unsqueeze(1)      # [2304, 1]

    for vi in indices:
        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            continue
        label  = int(ex.get("label", 0))
        prompt = _build_vsr_prompt(str(ex.get("caption", "")))
        try:
            iids, attn, pv = process_vlm_inputs(
                img, prompt, pt_proc, pt_nns._module, device=device
            )
            _, img_end = get_image_token_positions(iids)

            with pt_nns.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                lo   = pt_nns.model.language_model.layers[layer_idx].output[0][0, img_end:]
                ones = (lo @ v_col) * 0.0 + 1.0
                lo  += alpha * mean_mix_act * ones * wdec_vec
                logits_s = pt_nns.output.logits.save()

            pred = _predict(logits_s[0, -1, :], yes_ids, no_ids)
        except Exception:
            pred = 0
        total   += 1
        correct += (pred == label)

    return correct / max(total, 1) * 100, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Check which features are already done
    # -----------------------------------------------------------------------
    pending = []
    for feat in FEATURES:
        L, F = feat["layer"], feat["feature"]
        out_path = OUT_DIR / f"wdec_natural_L{L}_F{F}.json"
        if out_path.exists():
            print(f"[SKIP] L{L}_F{F} — output exists", flush=True)
        else:
            pending.append(feat)

    if not pending:
        print("[INFO] All features already processed. Exiting.", flush=True)
        return

    # -----------------------------------------------------------------------
    # Load VSR (all splits) once — shared across both phases
    # -----------------------------------------------------------------------
    print("[INFO] Loading VSR dataset (all splits)...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    # -----------------------------------------------------------------------
    # Phase 1: mix-448 — compute mean activations for all pending features
    # -----------------------------------------------------------------------
    print(f"\n[PHASE 1] Loading {MIX_MODEL} for mean activation extraction...", flush=True)
    mix_proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    mix_nns = NNsight(mix_model)

    # Store extraction results keyed by (layer, feature)
    extraction_results = {}   # (L, F) -> {mean_mix_act, pct_zero, n, indices}

    for feat in pending:
        L, F, relations = feat["layer"], feat["feature"], feat["relations"]
        key = f"L{L}_F{F}"
        print(f"[EXTRACT] {key} ({relations})...", flush=True)

        W_enc, W_dec, b_enc, b_dec, threshold = _load_sae_weights(L, device)

        mean_mix_act, pct_zero, n, indices = compute_mean_mix_activation(
            vsr_all, relation_indices, relations,
            L, F,
            mix_nns, mix_proc, device,
            W_enc, b_enc, threshold,
        )
        extraction_results[(L, F)] = {
            "mean_mix_act": mean_mix_act,
            "pct_zero":     pct_zero,
            "n":            n,
            "indices":      indices,
            "W_dec":        W_dec,   # keep on CPU until needed
        }
        print(
            f"[EXTRACT] {key}: mean_mix_act={mean_mix_act:.4f}  "
            f"pct_zero={pct_zero:.1%}  n={n}  n_subset={len(indices)}",
            flush=True,
        )

        del W_enc, b_enc, b_dec, threshold
        gc.collect()

    # Unload mix-448 to free VRAM
    print("[INFO] Unloading mix-448...", flush=True)
    del mix_nns, mix_model, mix_proc
    torch.cuda.empty_cache()
    gc.collect()

    # -----------------------------------------------------------------------
    # Phase 2: pt-448 — baseline + steered inference
    # -----------------------------------------------------------------------
    print(f"\n[PHASE 2] Loading {PT_MODEL} for steered inference...", flush=True)
    pt_proc  = AutoProcessor.from_pretrained(PT_MODEL)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    pt_nns = NNsight(pt_model)

    tokenizer  = pt_proc.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(pt_model.parameters()).dtype

    baseline_cache = {}  # rel_key -> base_acc

    for feat in pending:
        L, F, relations = feat["layer"], feat["feature"], feat["relations"]
        key      = f"L{L}_F{F}"
        out_path = OUT_DIR / f"wdec_natural_L{L}_F{F}.json"

        er       = extraction_results[(L, F)]
        mean_mix_act = er["mean_mix_act"]
        pct_zero     = er["pct_zero"]
        n_extracted  = er["n"]
        indices      = er["indices"]
        W_dec        = er["W_dec"]   # [16384, 2304] on CPU

        # W_dec row for this feature (NOT normalized — use raw decoder direction)
        wdec_vec = W_dec[F].to(model_dtype).to(device)    # [2304]

        # Baseline
        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            print(f"[BASE] {key} pt-448 baseline (N={len(indices)})...", flush=True)
            base_acc, base_n = run_pt_baseline(
                indices, vsr_all, pt_nns, pt_proc, yes_ids, no_ids, device
            )
            baseline_cache[rel_key] = (base_acc, base_n)
            print(f"[BASE] {key}: {base_acc:.2f}%  N={base_n}", flush=True)
        base_acc, base_n = baseline_cache[rel_key]

        # Alpha sweep
        alpha_results = {}
        for alpha in ALPHAS:
            print(f"[STEER] {key}  alpha={alpha}  inj={alpha * mean_mix_act:.4f}...", flush=True)
            acc, n_run = run_pt_steered(
                indices, vsr_all, pt_nns, pt_proc, yes_ids, no_ids, device,
                L, wdec_vec, alpha, mean_mix_act,
            )
            delta_acc = acc - base_acc
            alpha_results[str(alpha)] = {"acc": acc, "delta_acc": delta_acc}
            print(
                f"  alpha={alpha}: acc={acc:.2f}%  Δ={delta_acc:+.2f}%",
                flush=True,
            )

        # Best alpha
        best_alpha = max(alpha_results, key=lambda a: alpha_results[a]["delta_acc"])
        best_delta = alpha_results[best_alpha]["delta_acc"]

        result = {
            "layer":               L,
            "feature":             F,
            "relations":           relations,
            "n":                   base_n,
            "mean_mix_activation": mean_mix_act,
            "pct_zero_activations": pct_zero,
            "base_acc":            base_acc,
            "alphas":              alpha_results,
            "best_alpha":          float(best_alpha),
            "best_delta":          best_delta,
        }

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[SAVE] {out_path}", flush=True)

        del wdec_vec, W_dec
        torch.cuda.empty_cache()
        gc.collect()

    print("\n[DONE] All features processed.", flush=True)


if __name__ == "__main__":
    main()
