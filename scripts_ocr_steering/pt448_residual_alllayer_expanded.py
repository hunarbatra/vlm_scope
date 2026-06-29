#!/usr/bin/env python3
"""
Residual-stream all-26-layer injection — EXPANDED relation sets.

For each of the 8 top SAE features, tests injection across the FULL oracle-assigned
relation subset (not just the 1-2 narrow relations used in the original sweep).

For each feature:
  - Combines all oracle-assigned relations into a single pool
  - Runs baseline once on that combined pool
  - Sweeps 5 strategies × 7 alphas
  - Saves best (strategy, alpha) by delta_acc to JSON

Strategies: single, flat_all, sae_only_down, sae_only_up, decay_fwd
Alphas:     [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_residual_alllayer_expanded/

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 pt448_residual_alllayer_expanded.py
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
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_residual_alllayer_expanded")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

INJECTION_ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
STRATEGIES = ["single", "flat_all", "sae_only_down", "sae_only_up", "decay_fwd"]
DECAY = 0.85

# Full oracle-assigned relation sets per feature (BEST_FEATURE_MAP)
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


# ---------------------------------------------------------------------------
# Layer-weight schedules
# ---------------------------------------------------------------------------

def _layer_weights(strategy, home_layer, n_layers=N_LAYERS, decay=DECAY):
    if strategy == "single":
        return {home_layer: 1.0}
    elif strategy == "flat_all":
        return {l: 1.0 for l in range(n_layers)}
    elif strategy == "sae_only_down":
        return {l: 1.0 for l in range(home_layer, n_layers)}
    elif strategy == "sae_only_up":
        return {l: 1.0 for l in range(0, home_layer + 1)}
    elif strategy == "decay_fwd":
        return {l: decay ** max(l - home_layer, 0) for l in range(n_layers)}
    return {l: 1.0 for l in range(n_layers)}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _build_vsr_prompt(s):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {s.strip()}\nAnswer:"
    )


def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    ov = yes_ids & no_ids; yes_ids -= ov; no_ids -= ov
    return yes_ids, no_ids


def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n; p = max(y / d if d > 0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1 - p, 1e-7))


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists():
            return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"[INFO] Loading {MODEL_NAME}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # ------------------------------------------------------------------
    # Load VSR and build relation index
    # ------------------------------------------------------------------
    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    def get_combined_indices(relations):
        idxs = []
        for r in relations:
            idxs.extend(relation_indices.get(r, []))
        return idxs

    # ------------------------------------------------------------------
    # Baseline runner
    # ------------------------------------------------------------------
    def run_baseline(indices):
        correct = total = 0
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, _build_vsr_prompt(str(ex.get("caption", ""))),
                    processor, model_raw, device=device
                )
                with torch.inference_mode():
                    out = model_raw(input_ids=iids, attention_mask=attn,
                                   pixel_values=pv, use_cache=False)
                pred, _ = _pm(out.logits[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            total += 1; correct += (pred == label)
        return correct / max(total, 1) * 100, total

    # ------------------------------------------------------------------
    # Injection runner
    # ------------------------------------------------------------------
    def run_injected(indices, fv, layer_weights, alpha):
        correct = total = 0
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, _build_vsr_prompt(str(ex.get("caption", ""))),
                    processor, nns_model._module, device=device
                )
                _, img_end = get_image_token_positions(iids)
                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l, w in layer_weights.items():
                        lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        v_col = fv.unsqueeze(1)
                        ones = (lo @ v_col) * 0.0 + 1.0
                        lo += (alpha * w) * ones * fv
                    logits_s = nns_model.output.logits.save()
                pred, _ = _pm(logits_s[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            total += 1; correct += (pred == label)
        return correct / max(total, 1) * 100, total

    # ------------------------------------------------------------------
    # Per-feature sweep
    # ------------------------------------------------------------------
    all_results = []

    for feat_cfg in FEATURES:
        layer_idx   = feat_cfg["layer"]
        feature_idx = feat_cfg["feature"]
        relations   = feat_cfg["relations"]
        key = f"L{layer_idx}_F{feature_idx}"

        result_path = OUT_DIR / f"expanded_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key} — already done", flush=True)
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        indices = get_combined_indices(relations)
        if not indices:
            print(f"[WARN] {key} — no samples found, skipping", flush=True)
            continue

        # ---- Baseline ----
        print(f"\n[FEATURE] {key}  relations={relations}  N={len(indices)}", flush=True)
        print(f"  [BASE] running baseline on {len(indices)} samples...", flush=True)
        base_acc, n_valid = run_baseline(indices)
        print(f"  [BASE] acc={base_acc:.2f}%  N_valid={n_valid}", flush=True)

        # ---- Load W_dec feature vector ----
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir="/data1/hf_cache/hub")
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        # ---- Strategy × alpha sweep ----
        result = {
            "layer": layer_idx,
            "feature": feature_idx,
            "relations": relations,
            "n": n_valid,
            "base_acc": base_acc,
            "strategies": {},
        }

        global_best_delta = -1e9
        global_best_strategy = None
        global_best_alpha = None

        for strategy in STRATEGIES:
            lw = _layer_weights(strategy, layer_idx)
            print(f"\n  [STRAT] {key} strategy={strategy}  n_active_layers={len(lw)}", flush=True)

            strat_best_delta = -1e9
            strat_best_alpha = None
            alphas_res = {}

            for alpha in INJECTION_ALPHAS:
                print(f"    [INJECT] {key} {strategy} α={alpha}...", flush=True)
                acc, _ = run_injected(indices, fv, lw, alpha)
                da = acc - base_acc
                alphas_res[str(alpha)] = {"acc": round(acc, 4), "delta_acc": round(da, 4)}
                print(f"      acc={acc:.2f}%  Δacc={da:+.2f}%", flush=True)

                if da > strat_best_delta:
                    strat_best_delta = da
                    strat_best_alpha = alpha
                if da > global_best_delta:
                    global_best_delta = da
                    global_best_strategy = strategy
                    global_best_alpha = alpha

            result["strategies"][strategy] = {
                "alphas": alphas_res,
                "best_alpha": strat_best_alpha,
                "best_delta": round(strat_best_delta, 4),
            }
            print(f"  [STRAT BEST] {strategy}: α={strat_best_alpha}  Δacc={strat_best_delta:+.2f}%", flush=True)

        result["best_strategy"] = global_best_strategy
        result["best_alpha"]    = global_best_alpha
        result["best_delta"]    = round(global_best_delta, 4)

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  [SAVED] {result_path}", flush=True)

        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*110}", flush=True)
    print("pt-448 Residual All-Layer — Expanded Relation Sets — Best (strategy, alpha) per feature", flush=True)
    print(f"{'='*110}", flush=True)

    hdr = (f"{'Feature':<14} {'N':>5} {'BaseAcc':>8}  "
           f"{'single':>10} {'flat_all':>10} {'sae_dn':>10} {'sae_up':>10} {'decay_fwd':>10}  "
           f"{'BestStrat':<12} {'BestAlpha':>9} {'BestDelta':>10}")
    print(hdr, flush=True)
    print("-" * 110, flush=True)

    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        row = f"{key:<14} {r['n']:>5} {r['base_acc']:>7.2f}%  "
        for s in STRATEGIES:
            bd = r["strategies"].get(s, {}).get("best_delta")
            row += f"  {bd:>+9.2f}" if bd is not None else f"  {'--':>9}"
        bs = r.get("best_strategy", "--")
        ba = r.get("best_alpha", "--")
        bd = r.get("best_delta")
        row += f"  {bs:<12} {ba!s:>9} {bd:>+9.2f}%" if bd is not None else f"  {'--':<12} {'--':>9} {'--':>10}"
        print(row, flush=True)

    print(f"\n[DONE] Results in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
