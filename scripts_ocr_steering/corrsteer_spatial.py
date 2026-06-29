#!/usr/bin/env python3
"""
CorrSteer for spatial SAE features.

Phase 1 — Collect: For each layer (distributed across GPUs), run VSR train+dev
  through model+SAE, record per-sample mean/last-token feature activations and
  whether the baseline prediction was correct.  Saves activations_layer_{L}.npz.

Phase 2 — Correlate (CPU): Pearson r between each (layer, feature) activation
  and correctness across all samples.  Select top-K features by positive r.

Phase 3 — Steer: On VSR test split, inject constant alpha × W_dec[F] at layer L
  for each selected (L, F).  Sweep alphas.  Compare VSR + ctrl accuracy.

Key difference from ablation-selectivity steering: features here are selected
by co-occurrence with CORRECT predictions, not by accuracy drop on removal.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

sys.path.insert(0, str(Path(__file__).parent))
from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

MODEL_NAME  = "google/paligemma2-3b-mix-448"
N_LAYERS    = 26
D_SAE       = 16384
D_MODEL     = 2304
VSR_DATASET = "cambridgeltl/vsr_random"

CHECKPOINT_DIR = Path(os.environ.get("VLMSCOPE_CKPT_DIR",
                                      "/data1/vlm_scope_sae_mix448_textonly/checkpoints"))
HF_CACHE       = os.environ.get("HF_HOME", "/data1/hf_cache")
IMAGE_CACHE_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")

CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}

# Alphas for phase 3 sweep (small — avoid Hydra collapse)
STEER_ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
TOP_K_VALUES = [5, 10, 20]


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_vsr_prompt(statement: str) -> str:
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\n"
        "Answer:"
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


def _predict_yesno(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


def _load_vsr_image(ex):
    import requests
    from PIL import Image
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = IMAGE_CACHE_DIR / f"{url_hash}.jpg"
    try:
        if cache_path.exists():
            return Image.open(cache_path).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        img.save(cache_path, "JPEG")
        return img
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: activation collection worker
# ──────────────────────────────────────────────────────────────────────────────

def _collect_worker(gpu_id, layer_assignments, out_dir, splits):
    """
    For each assigned layer: run VSR samples (given splits) through model+SAE,
    record per-sample (mean_acts, last_act, correct, label, relation).
    Saves activations_layer_{L}.npz per layer.
    """
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    out_dir = Path(out_dir)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer  = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)

    print(f"[Collect GPU{gpu_id}] Loading VSR splits: {splits}", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s) for s in splits
    ])
    print(f"[Collect GPU{gpu_id}] VSR n={len(vsr)}", flush=True)

    for layer_idx in layer_assignments:
        save_path = out_dir / f"activations_layer_{layer_idx}.npz"
        if save_path.exists():
            print(f"[Collect GPU{gpu_id}] Layer {layer_idx}: cached, skip", flush=True)
            continue

        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae  = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()
        print(f"[Collect GPU{gpu_id}] Layer {layer_idx}: collecting {len(vsr)} samples", flush=True)

        mean_acts_list = []  # [n_valid, D_SAE]  float16
        last_acts_list = []  # [n_valid, D_SAE]  float16
        correct_list   = []  # [n_valid]          int8
        label_list     = []  # [n_valid]          int8
        relation_list  = []  # [n_valid]          str

        for i in range(len(vsr)):
            if i % 1000 == 0:
                print(f"  [Collect GPU{gpu_id}] L{layer_idx}: {i}/{len(vsr)}", flush=True)
            ex  = vsr[i]
            img = _load_vsr_image(ex)
            if img is None:
                continue
            caption  = str(ex.get("caption", "")).strip()
            label    = int(ex.get("label", 0))
            relation = ex.get("relation", "")
            prompt   = _build_vsr_prompt(caption)

            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model_raw, device=device)
                _, img_end = get_image_token_positions(input_ids)

                # Single forward pass: save residual at layer + logits
                with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                     pixel_values=pixel_values):
                    residual_saved = nns_model.model.language_model.layers[
                        layer_idx].output[0][0, img_end:].save()
                    logits_saved = nns_model.output.logits.save()

                logits   = logits_saved
                residual = residual_saved  # [n_text_tokens, d_model]

                pred    = _predict_yesno(logits[0, -1, :], yes_ids, no_ids)
                correct = int(pred == label)

                with torch.no_grad():
                    acts = sae.encode(residual.to(sae.W_enc.dtype))  # [n_text, d_sae]
                    mean_act = acts.mean(dim=0)   # [d_sae]
                    last_act = acts[-1]           # [d_sae] — last text token

                mean_acts_list.append(mean_act.cpu().to(torch.float16).numpy())
                last_acts_list.append(last_act.cpu().to(torch.float16).numpy())
                correct_list.append(correct)
                label_list.append(label)
                relation_list.append(relation)

                del residual, acts, mean_act, last_act, logits
            except Exception as e:
                continue

        n = len(correct_list)
        print(f"[Collect GPU{gpu_id}] Layer {layer_idx}: saved {n} samples", flush=True)
        np.savez_compressed(
            save_path,
            mean_acts=np.array(mean_acts_list, dtype=np.float16),
            last_acts=np.array(last_acts_list, dtype=np.float16),
            correct=np.array(correct_list, dtype=np.int8),
            label=np.array(label_list, dtype=np.int8),
            relation=np.array(relation_list, dtype=object),
        )

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[Collect GPU{gpu_id}] Done.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: correlation computation (CPU)
# ──────────────────────────────────────────────────────────────────────────────

def run_correlation(out_dir, top_k=20, agg="mean"):
    """
    Load per-layer activation files, compute Pearson r(activation, correct),
    return list of (r, layer, feature_idx) sorted descending.
    """
    from scipy.stats import pearsonr

    out_dir = Path(out_dir)
    results = []

    for layer_idx in range(N_LAYERS):
        path = out_dir / f"activations_layer_{layer_idx}.npz"
        if not path.exists():
            print(f"  [Corr] Layer {layer_idx}: missing, skip")
            continue
        data    = np.load(path, allow_pickle=True)
        acts    = data[f"{agg}_acts"].astype(np.float32)  # [n, D_SAE]
        correct = data["correct"].astype(np.float32)       # [n]

        # Only compute on features that ever fire (>0 mean)
        firing_mask = acts.mean(axis=0) > 0
        firing_idx  = np.where(firing_mask)[0]

        print(f"  [Corr] Layer {layer_idx}: {acts.shape[0]} samples, "
              f"{firing_idx.size} firing features", flush=True)

        for fi in firing_idx:
            x = acts[:, fi]
            if x.std() < 1e-8:
                continue
            r, _ = pearsonr(x, correct)
            if not np.isfinite(r):
                continue
            results.append((float(r), layer_idx, int(fi)))

    results.sort(key=lambda t: t[0], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: steering evaluation worker
# ──────────────────────────────────────────────────────────────────────────────

def _steer_worker(gpu_id, sample_assignments, selected_features, out_dir, alphas, top_k_values):
    """
    For each (alpha, top_k) combo: run VSR test samples with constant steering
    injection: for each (layer, feature) in top-k, add alpha × W_dec[F] to
    layer output at text token positions.
    """
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    out_dir = Path(out_dir)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer  = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    print(f"[Steer GPU{gpu_id}] Loading VSR test...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_test = load_dataset(VSR_DATASET, data_files=data_files, split="test")
    print(f"[Steer GPU{gpu_id}] VSR test n={len(vsr_test)}", flush=True)

    # Pre-load decoder vectors for all relevant features (cache by layer)
    # selected_features: list of (r, layer, feat_idx), sorted by r desc
    max_k    = max(top_k_values)
    top_feats = selected_features[:max_k]  # [(r, layer, feat_idx), ...]

    dec_cache = {}  # layer -> W_dec tensor [D_SAE, D_MODEL] on device
    for _, layer_idx, _ in top_feats:
        if layer_idx not in dec_cache:
            ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
            sae  = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                           device=device, cache_dir=HF_CACHE)
            dec_cache[layer_idx] = sae.W_dec.to(model_dtype)  # [D_SAE, D_MODEL]
            del sae
            torch.cuda.empty_cache()

    # For each (alpha, top_k), precompute per-layer steering deltas
    # delta_per_layer[layer] = sum of alpha × W_dec[F] for all selected (L,F) at that layer
    def build_steering(alpha, top_k):
        feats = top_feats[:top_k]
        by_layer = defaultdict(list)
        for _, layer_idx, feat_idx in feats:
            by_layer[layer_idx].append(feat_idx)
        per_layer = {}
        for layer_idx, fidxs in by_layer.items():
            # Sum decoder vectors for all selected features at this layer
            vecs = dec_cache[layer_idx][fidxs]  # [K, D_MODEL]
            per_layer[layer_idx] = (alpha * vecs.sum(dim=0)).to(model_dtype)  # [D_MODEL]
        return per_layer

    # Evaluate
    indices = sample_assignments
    results = {}  # key: (alpha, top_k) -> {vsr_acc, ctrl_acc, ...}

    # Baseline first
    print(f"[Steer GPU{gpu_id}] Computing baseline...", flush=True)
    base_correct = base_total = ctrl_correct = ctrl_total = 0
    for vi in indices:
        ex  = vsr_test[vi]
        img = _load_vsr_image(ex)
        if img is None: continue
        caption  = str(ex.get("caption", "")).strip()
        label    = int(ex.get("label", 0))
        relation = ex.get("relation", "")
        prompt   = _build_vsr_prompt(caption)
        try:
            input_ids, attn_mask, pixel_values = process_vlm_inputs(
                img, prompt, processor, model_raw, device=device)
            with torch.inference_mode():
                out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                pixel_values=pixel_values, use_cache=False)
            pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
        except Exception:
            pred = 0
        base_total += 1
        if pred == label: base_correct += 1
        if relation in CONTROL_RELATIONS:
            ctrl_total += 1
            if pred == label: ctrl_correct += 1

    base_vsr  = 100 * base_correct  / max(base_total, 1)
    base_ctrl = 100 * ctrl_correct / max(ctrl_total, 1)
    print(f"[Steer GPU{gpu_id}] Baseline VSR={base_vsr:.1f}%  Ctrl={base_ctrl:.1f}%", flush=True)
    results["baseline"] = {"vsr_acc": base_vsr, "ctrl_acc": base_ctrl,
                           "vsr_total": base_total, "ctrl_total": ctrl_total}

    # Steered sweep
    for top_k in top_k_values:
        for alpha in alphas:
            per_layer = build_steering(alpha, top_k)
            key = f"k{top_k}_a{alpha}"

            vsr_c = vsr_t = ctrl_c = ctrl_t = 0
            for vi in indices:
                ex  = vsr_test[vi]
                img = _load_vsr_image(ex)
                if img is None: continue
                caption  = str(ex.get("caption", "")).strip()
                label    = int(ex.get("label", 0))
                relation = ex.get("relation", "")
                prompt   = _build_vsr_prompt(caption)
                try:
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        img, prompt, processor, model_raw, device=device)
                    _, img_end = get_image_token_positions(input_ids)

                    with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                         pixel_values=pixel_values):
                        for layer_idx, delta in per_layer.items():
                            nns_model.model.language_model.layers[
                                layer_idx].output[0][0, img_end:] += delta.unsqueeze(0)
                        logits_saved = nns_model.output.logits.save()

                    pred = _predict_yesno(logits_saved[0, -1, :], yes_ids, no_ids)
                except Exception:
                    pred = 0
                vsr_t += 1
                if pred == label: vsr_c += 1
                if relation in CONTROL_RELATIONS:
                    ctrl_t += 1
                    if pred == label: ctrl_c += 1

            vsr_acc  = 100 * vsr_c  / max(vsr_t, 1)
            ctrl_acc = 100 * ctrl_c / max(ctrl_t, 1)
            delta_v  = vsr_acc  - base_vsr
            delta_c  = ctrl_acc - base_ctrl
            results[key] = {"vsr_acc": vsr_acc, "ctrl_acc": ctrl_acc,
                            "delta_vsr": delta_v, "delta_ctrl": delta_c}
            print(f"  [k={top_k}, α={alpha}] VSR={vsr_acc:.1f}% ({delta_v:+.2f})  "
                  f"Ctrl={ctrl_acc:.1f}% ({delta_c:+.2f})", flush=True)

    save_path = out_dir / f"steer_results_gpu{gpu_id}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Steer GPU{gpu_id}] Done → {save_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True,
                        choices=[1, 2, 3, 23],
                        help="1=collect, 2=correlate, 3=steer, 23=correlate+steer")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0,1,2,3,4,5,6,7])
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--collect-splits", type=str, nargs="+",
                        default=["train", "dev"],
                        help="VSR splits for Phase 1 activation collection")
    parser.add_argument("--top-k", type=int, nargs="+", default=TOP_K_VALUES)
    parser.add_argument("--alphas", type=float, nargs="+", default=STEER_ALPHAS)
    parser.add_argument("--agg", type=str, default="mean",
                        choices=["mean", "last"],
                        help="Activation aggregation for correlation")
    args = parser.parse_args()

    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    os.environ["VLMSCOPE_CKPT_DIR"] = args.checkpoint_dir

    from pathlib import Path as _P
    out_dir = _P(args.out_dir) if args.out_dir else \
              Path(os.environ.get("VLMSCOPE_ANALYSIS_DIR",
                                  "/data1/vlm_scope_sae_mix448_textonly/analysis")) \
              / "corrsteer_spatial"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ── Phase 1: collect activations ──────────────────────────────────────────
    if args.phase in (1,):
        n_gpus = len(args.gpus)
        # Distribute layers across GPUs
        layer_chunks = [[] for _ in range(n_gpus)]
        for l in range(N_LAYERS):
            layer_chunks[l % n_gpus].append(l)

        print(f"Phase 1: collecting activations for splits={args.collect_splits}")
        for i, gpu_id in enumerate(args.gpus):
            print(f"  GPU {gpu_id}: layers {layer_chunks[i]}")

        processes = []
        for i, gpu_id in enumerate(args.gpus):
            if layer_chunks[i]:
                p = mp.Process(target=_collect_worker,
                               args=(gpu_id, layer_chunks[i], str(out_dir),
                                     args.collect_splits))
                p.start()
                processes.append(p)
        for p in processes:
            p.join()
        print("Phase 1 complete.")

    # ── Phase 2: correlation ──────────────────────────────────────────────────
    if args.phase in (2, 23):
        print(f"\nPhase 2: computing correlations (agg={args.agg})...")
        results = run_correlation(out_dir, top_k=max(args.top_k), agg=args.agg)

        corr_path = out_dir / f"feature_correlations_{args.agg}.json"
        with open(corr_path, "w") as f:
            json.dump(results[:500], f, indent=2)  # save top-500
        print(f"Saved top-500 correlations → {corr_path}")

        print(f"\nTop-20 features by positive r ({args.agg}):")
        for r, layer, feat in results[:20]:
            print(f"  r={r:+.4f}  L{layer:>2}/F{feat:>6}")

        print(f"\nBottom-5 (negative correlation, for reference):")
        for r, layer, feat in results[-5:]:
            print(f"  r={r:+.4f}  L{layer:>2}/F{feat:>6}")

    # ── Phase 3: steer ────────────────────────────────────────────────────────
    if args.phase in (3, 23):
        corr_path = out_dir / f"feature_correlations_{args.agg}.json"
        if not corr_path.exists():
            print(f"ERROR: run Phase 2 first (missing {corr_path})")
            return

        with open(corr_path) as f:
            selected = json.load(f)  # list of [r, layer, feat_idx]
        selected = [tuple(x) for x in selected]

        print(f"\nPhase 3: steering on VSR test split")
        print(f"  top_k={args.top_k}  alphas={args.alphas}")
        print(f"  Using top-{max(args.top_k)} features from correlation")
        for r, layer, feat in selected[:10]:
            print(f"    r={r:+.4f}  L{layer:>2}/F{feat:>6}")

        # Load test split to get indices
        from datasets import load_dataset
        data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        vsr_test = load_dataset(VSR_DATASET, data_files=data_files, split="test")
        n_test = len(vsr_test)
        print(f"  VSR test: {n_test} samples")

        # Distribute test samples across GPUs
        n_gpus = len(args.gpus)
        all_idx = list(range(n_test))
        chunks = [all_idx[i::n_gpus] for i in range(n_gpus)]

        processes = []
        for i, gpu_id in enumerate(args.gpus):
            if chunks[i]:
                p = mp.Process(target=_steer_worker,
                               args=(gpu_id, chunks[i], selected,
                                     str(out_dir), args.alphas, args.top_k))
                p.start()
                processes.append(p)
        for p in processes:
            p.join()

        # Aggregate
        all_gpu_results = []
        for i, gpu_id in enumerate(args.gpus):
            path = out_dir / f"steer_results_gpu{gpu_id}.json"
            if path.exists():
                with open(path) as f:
                    all_gpu_results.append(json.load(f))

        if all_gpu_results:
            # Merge: average over GPUs (each GPU ran same keys on different samples)
            keys = [k for k in all_gpu_results[0].keys() if k != "baseline"]
            merged = {}

            # Baseline: weighted average
            base_total = sum(r["baseline"]["vsr_total"] for r in all_gpu_results)
            base_correct = sum(r["baseline"]["vsr_acc"] / 100 * r["baseline"]["vsr_total"]
                               for r in all_gpu_results)
            ctrl_total  = sum(r["baseline"]["ctrl_total"] for r in all_gpu_results)
            ctrl_correct = sum(r["baseline"]["ctrl_acc"] / 100 * r["baseline"]["ctrl_total"]
                               for r in all_gpu_results)
            merged["baseline"] = {
                "vsr_acc":  100 * base_correct / max(base_total, 1),
                "ctrl_acc": 100 * ctrl_correct / max(ctrl_total, 1),
            }

            for key in keys:
                # Simple average (each GPU has equal samples)
                vsr_accs  = [r[key]["vsr_acc"]  for r in all_gpu_results if key in r]
                ctrl_accs = [r[key]["ctrl_acc"] for r in all_gpu_results if key in r]
                merged[key] = {
                    "vsr_acc":    float(np.mean(vsr_accs)),
                    "ctrl_acc":   float(np.mean(ctrl_accs)),
                    "delta_vsr":  float(np.mean(vsr_accs))  - merged["baseline"]["vsr_acc"],
                    "delta_ctrl": float(np.mean(ctrl_accs)) - merged["baseline"]["ctrl_acc"],
                }

            summary_path = out_dir / "steer_summary.json"
            with open(summary_path, "w") as f:
                json.dump(merged, f, indent=2)

            print(f"\n=== CorrSteer Results (agg={args.agg}) ===")
            print(f"Baseline: VSR={merged['baseline']['vsr_acc']:.1f}%  "
                  f"Ctrl={merged['baseline']['ctrl_acc']:.1f}%")
            for key in sorted(keys):
                m = merged[key]
                print(f"  {key:20s}  VSR={m['vsr_acc']:.1f}% ({m['delta_vsr']:+.2f})  "
                      f"Ctrl={m['ctrl_acc']:.1f}% ({m['delta_ctrl']:+.2f})")
            print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
