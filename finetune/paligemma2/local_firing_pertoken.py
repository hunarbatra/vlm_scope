#!/usr/bin/env python3
"""
Compute per-TOKEN firing frequencies for VQA and VSR datasets.
Matches the original LLaVA pipeline methodology exactly.

For each layer:
  - Process VQA/VSR samples through PaliGemma2
  - Extract residual stream activations
  - Run through JumpReLU SAE
  - Count per-token firing: for each feature, how many tokens activated > 0
  - Save firing_count and total_tokens per dataset

Then run Fisher exact test (one-sided, greater) to find features that fire
significantly more on VSR than VQA at the per-token level.

Usage:
    # Full pipeline (8 GPUs):
    HF_TOKEN=hf_... python local_firing_pertoken.py \
        --n-vqa 10000 --n-vsr 10000 \
        --gpus 0,1,2,3,4,5,6,7

    # Fisher test only (reuse existing firing data):
    python local_firing_pertoken.py --fisher-only

    # Single GPU worker (called internally):
    python local_firing_pertoken.py --worker --gpu 0 --layers 0,8,16,24
"""

import argparse
import os
import sys
import json
import math
import gc
import csv
import ast
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

N_LAYERS = 26
D_SAE = 16384
MODEL_NAME = "google/paligemma2-3b-pt-224"


def predownload_data(n_vqa, n_vsr, cache_dir):
    """Pre-download VQA and VSR data, save images to disk for workers."""
    import pickle
    from datasets import load_dataset
    from PIL import Image
    import requests
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vqa_cache = cache_dir / "vqa_samples.pkl"
    vsr_cache = cache_dir / "vsr_samples.pkl"

    # VQA
    if vqa_cache.exists():
        print(f"Loading cached VQA from {vqa_cache}")
        with open(vqa_cache, "rb") as f:
            vqa_samples = pickle.load(f)
        print(f"  {len(vqa_samples)} VQA samples")
    else:
        print(f"Downloading VQA (target: {n_vqa} yes/no samples)...")
        ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        vqa_samples = []
        for item in tqdm(ds, desc="Scanning VQA", total=n_vqa * 3):
            ans = item.get("multiple_choice_answer", "")
            if ans.lower() in ("yes", "no"):
                vqa_samples.append({
                    "image": item["image"],
                    "question": item["question"],
                    "answer": ans.lower(),
                })
                if len(vqa_samples) >= n_vqa:
                    break
        print(f"  Loaded {len(vqa_samples)} VQA samples, caching...")
        with open(vqa_cache, "wb") as f:
            pickle.dump(vqa_samples, f)

    # VSR
    if vsr_cache.exists():
        print(f"Loading cached VSR from {vsr_cache}")
        with open(vsr_cache, "rb") as f:
            vsr_samples = pickle.load(f)
        print(f"  {len(vsr_samples)} VSR samples")
    else:
        print(f"Downloading VSR (target: {n_vsr} samples)...")
        raw_items = []
        for split in ["train", "validation", "test"]:
            ds = load_dataset("cambridgeltl/vsr_random", split=split)
            for item in ds:
                raw_items.append(item)
                if len(raw_items) >= n_vsr:
                    break
            if len(raw_items) >= n_vsr:
                break

        print(f"  Collected {len(raw_items)} VSR items, downloading images...")

        def download_one(item):
            try:
                url = item.get("image_link", item.get("image", ""))
                if not isinstance(url, str) or not url.startswith("http"):
                    return None
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return {
                    "image": img,
                    "caption": item.get("caption", ""),
                    "label": item.get("label", 0),
                    "relation": item.get("relation", ""),
                }
            except Exception:
                return None

        vsr_samples = []
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {executor.submit(download_one, item): i for i, item in enumerate(raw_items)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading VSR"):
                result = future.result()
                if result is not None:
                    vsr_samples.append(result)

        print(f"  Downloaded {len(vsr_samples)} VSR images, caching...")
        with open(vsr_cache, "wb") as f:
            pickle.dump(vsr_samples, f)

    return vqa_samples, vsr_samples


def load_cached_data(cache_dir):
    """Load pre-downloaded data from cache."""
    import pickle
    cache_dir = Path(cache_dir)
    with open(cache_dir / "vqa_samples.pkl", "rb") as f:
        vqa = pickle.load(f)
    with open(cache_dir / "vsr_samples.pkl", "rb") as f:
        vsr = pickle.load(f)
    return vqa, vsr


def worker_main(gpu_id, layer_indices, n_vqa, n_vsr, results_dir, cache_dir="analysis_results/data_cache"):
    """Single-GPU worker: load model, process VQA+VSR, save per-token firing."""
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from utils import initialize_jumprelu_sae

    # CUDA_VISIBLE_DEVICES remaps, so always use cuda:0
    device = torch.device("cuda:0")
    results_dir = Path(results_dir)

    # Check which layers still need processing
    todo_layers = []
    for l in layer_indices:
        out_path = results_dir / f"firing_pertoken_layer_{l}.json"
        if out_path.exists():
            print(f"[GPU {gpu_id}] L{l}: SKIP (already done)")
        else:
            todo_layers.append(l)

    if not todo_layers:
        print(f"[GPU {gpu_id}] All layers done, exiting")
        return

    # Load pre-downloaded data from cache
    vqa_samples, vsr_samples = load_cached_data(cache_dir)

    # Load model
    print(f"[GPU {gpu_id}] Loading PaliGemma2...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    for layer_idx in todo_layers:
        out_path = results_dir / f"firing_pertoken_layer_{layer_idx}.json"
        print(f"\n[GPU {gpu_id}] Layer {layer_idx}")

        # Load SAE
        sae = initialize_jumprelu_sae(layer_idx, device=device)
        sae.eval()

        # Hook to capture residual stream
        hook_layer = model.language_model.layers[layer_idx]
        captured = {}

        def hook_fn(module, input, output):
            captured["hidden"] = output[0].detach()

        handle = hook_layer.register_forward_hook(hook_fn)

        # --- VQA per-token firing ---
        vqa_fire_count = np.zeros(D_SAE, dtype=np.int64)
        vqa_total_tokens = 0

        for i, s in enumerate(tqdm(vqa_samples, desc=f"GPU{gpu_id} L{layer_idx} VQA")):
            try:
                inputs = processor(
                    text=s["question"], images=s["image"],
                    return_tensors="pt", padding=True,
                ).to(device)

                with torch.no_grad():
                    model(**inputs)

                hidden = captured["hidden"].float()  # (1, seq, 2304)
                seq_len = hidden.shape[1]

                with torch.no_grad():
                    codes = sae.encode(hidden[0])  # (seq, d_sae)

                # PER-TOKEN: count tokens where each feature fires > 0
                fired_tokens = (codes > 0).sum(dim=0).cpu().numpy()  # (d_sae,)
                vqa_fire_count += fired_tokens.astype(np.int64)
                vqa_total_tokens += seq_len

            except Exception as e:
                if i < 3:
                    print(f"  [WARN] VQA sample {i}: {e}")

        print(f"  VQA: {len(vqa_samples)} samples, {vqa_total_tokens:,} tokens")

        # --- VSR per-token firing ---
        vsr_fire_count = np.zeros(D_SAE, dtype=np.int64)
        vsr_total_tokens = 0

        for i, s in enumerate(tqdm(vsr_samples, desc=f"GPU{gpu_id} L{layer_idx} VSR")):
            try:
                prompt = f"Is this true: {s['caption']}?"
                inputs = processor(
                    text=prompt, images=s["image"],
                    return_tensors="pt", padding=True,
                ).to(device)

                with torch.no_grad():
                    model(**inputs)

                hidden = captured["hidden"].float()
                seq_len = hidden.shape[1]

                with torch.no_grad():
                    codes = sae.encode(hidden[0])

                fired_tokens = (codes > 0).sum(dim=0).cpu().numpy()
                vsr_fire_count += fired_tokens.astype(np.int64)
                vsr_total_tokens += seq_len

            except Exception as e:
                if i < 3:
                    print(f"  [WARN] VSR sample {i}: {e}")

        print(f"  VSR: {len(vsr_samples)} samples, {vsr_total_tokens:,} tokens")

        handle.remove()
        del sae
        torch.cuda.empty_cache()

        # Save
        result = {
            "layer": layer_idx,
            "n_vqa_samples": len(vqa_samples),
            "n_vsr_samples": len(vsr_samples),
            "n_vqa_tokens": int(vqa_total_tokens),
            "n_vsr_tokens": int(vsr_total_tokens),
            "vqa_fire_count": vqa_fire_count.tolist(),
            "vsr_fire_count": vsr_fire_count.tolist(),
        }
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"  Saved: {out_path}")


def run_fisher_test(results_dir, adapted_csv, p_thr=0.01, odds_thr=3.0, min_diff=0.05):
    """Run Fisher exact test on per-token firing counts."""
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    print(f"\n{'='*60}")
    print(f"Fisher exact test (per-token): OR>={odds_thr}, min_diff>={min_diff}")
    print(f"{'='*60}")

    rows = []
    for layer_idx in range(N_LAYERS):
        fpath = results_dir / f"firing_pertoken_layer_{layer_idx}.json"
        if not fpath.exists():
            print(f"  L{layer_idx}: MISSING")
            continue
        with open(fpath) as f:
            data = json.load(f)

        n_vqa = data["n_vqa_tokens"]
        n_vsr = data["n_vsr_tokens"]
        vqa_fire = data["vqa_fire_count"]
        vsr_fire = data["vsr_fire_count"]

        for fi in range(D_SAE):
            c_vqa = int(vqa_fire[fi])
            c_vsr = int(vsr_fire[fi])
            if c_vqa == 0 and c_vsr == 0:
                continue
            rows.append({
                "layer": layer_idx, "feature": fi,
                "c_vqa": c_vqa, "n_vqa": n_vqa,
                "c_vsr": c_vsr, "n_vsr": n_vsr,
            })

    if not rows:
        print("No firing data found!")
        return None

    df = pd.DataFrame(rows)
    print(f"Total features with any firing: {len(df)}")

    # Compute stats
    pvals, odds = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Fisher tests"):
        c_vsr = min(int(r.c_vsr), int(r.n_vsr))
        c_vqa = min(int(r.c_vqa), int(r.n_vqa))
        table = [[c_vsr, max(0, int(r.n_vsr) - c_vsr)],
                 [c_vqa, max(0, int(r.n_vqa) - c_vqa)]]
        try:
            o, p = fisher_exact(table, alternative="greater")
            odds.append(o if not math.isinf(o) else 1e9)
            pvals.append(p)
        except ValueError:
            odds.append(1.0)
            pvals.append(1.0)

    df["odds_ratio"] = odds
    df["p_raw"] = pvals
    df["freq_vsr"] = df.c_vsr / df.n_vsr
    df["freq_vqa"] = df.c_vqa / df.n_vqa
    df["freq_diff"] = df.freq_vsr - df.freq_vqa

    # FDR correction
    df["p_adj"] = multipletests(df.p_raw, method="fdr_bh")[1]

    # Filter: matching original thresholds
    keep = (df.odds_ratio >= odds_thr) & (df.freq_diff >= min_diff)
    spatial = df.loc[keep].sort_values("odds_ratio", ascending=False)

    print(f"\nSpatial features: {len(spatial)} / {len(df)} ({100*len(spatial)/len(df):.1f}%)")

    # Save
    df.to_csv(results_dir / "fisher_all_pertoken.csv", index=False)
    spatial.to_csv(results_dir / "spatial_features_pertoken.csv", index=False)

    for l in sorted(spatial.layer.unique()):
        n = len(spatial[spatial.layer == l])
        print(f"  L{l:2d}: {n} spatial features")

    # Intersect with adapted features
    if adapted_csv and Path(adapted_csv).exists():
        adapted_df = pd.read_csv(adapted_csv)
        adapted_set = set()
        for _, r in adapted_df.iterrows():
            indices = ast.literal_eval(r["adapted_indices"])
            for idx in indices:
                adapted_set.add((int(r["layer"]), int(idx)))

        spatial_adapted = spatial[
            spatial.apply(lambda r: (int(r.layer), int(r.feature)) in adapted_set, axis=1)
        ]
        print(f"\nAdapted ∩ Spatial: {len(spatial_adapted)} features")
        spatial_adapted.to_csv(results_dir / "spatial_adapted_pertoken.csv", index=False)

        for l in sorted(spatial_adapted.layer.unique()):
            n = len(spatial_adapted[spatial_adapted.layer == l])
            print(f"  L{l:2d}: {n} features")

    return spatial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vqa", type=int, default=10000)
    parser.add_argument("--n-vsr", type=int, default=10000)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--results-dir", default="analysis_results/firing_pertoken")
    parser.add_argument("--adapted-csv", default="analysis_results/adapted_features_results.csv")
    parser.add_argument("--p-thr", type=float, default=0.01)
    parser.add_argument("--odds-thr", type=float, default=3.0)
    parser.add_argument("--min-diff", type=float, default=0.05)
    parser.add_argument("--fisher-only", action="store_true")
    parser.add_argument("--cache-dir", default="analysis_results/data_cache")
    # Worker mode args
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layers", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Worker mode: called as subprocess per GPU
    if args.worker:
        layer_indices = [int(x) for x in args.layers.split(",") if x]
        worker_main(args.gpu, layer_indices, args.n_vqa, args.n_vsr,
                     args.results_dir, args.cache_dir)
        return

    if not args.fisher_only:
        gpu_ids = [int(g) for g in args.gpus.split(",")]

        # Pre-download data once (parent process, no GPU needed)
        print("Pre-downloading datasets...")
        predownload_data(args.n_vqa, args.n_vsr, args.cache_dir)
        print("Data cached, launching GPU workers...\n")

        # Distribute layers round-robin
        layers_per_gpu = [[] for _ in gpu_ids]
        for i in range(N_LAYERS):
            layers_per_gpu[i % len(gpu_ids)].append(i)

        print(f"GPU assignments:")
        for i, gpu_id in enumerate(gpu_ids):
            print(f"  GPU {gpu_id}: layers {layers_per_gpu[i]}")

        # Launch one subprocess per GPU
        procs = []
        for i, gpu_id in enumerate(gpu_ids):
            if not layers_per_gpu[i]:
                continue
            layers_str = ",".join(str(l) for l in layers_per_gpu[i])
            cmd = [
                sys.executable, __file__,
                "--worker", "--gpu", str(gpu_id),
                "--layers", layers_str,
                "--n-vqa", str(args.n_vqa),
                "--n-vsr", str(args.n_vsr),
                "--results-dir", str(results_dir),
                "--cache-dir", str(args.cache_dir),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = subprocess.Popen(cmd, env=env)
            procs.append((gpu_id, p))
            print(f"  Launched GPU {gpu_id} (PID {p.pid})")

        # Wait for all
        for gpu_id, p in procs:
            p.wait()
            print(f"  GPU {gpu_id} finished (exit code {p.returncode})")

        print("\nAll GPU workers done.")

    # Run Fisher test
    run_fisher_test(results_dir, args.adapted_csv,
                    p_thr=args.p_thr, odds_thr=args.odds_thr, min_diff=args.min_diff)


if __name__ == "__main__":
    main()
