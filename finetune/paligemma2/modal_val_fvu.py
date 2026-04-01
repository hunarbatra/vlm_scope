"""
Compute validation FVU (Fraction of Variance Unexplained) for PaliGemma2 SAEs.

Breaks down by token type (Full / Image / Text) to match paper's Table 1.
Uses cached validation activations from H5 files + trained SAE checkpoints.
Parallelized across 8 GPUs (each handles ~3-4 layers).

Usage:
    cd finetune/paligemma2
    MODAL_PROFILE=hunar-oxford modal run modal_val_fvu.py
"""

import os
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
GPU_TYPE = "A100"
TIMEOUT = 86400

app = modal.App("vlm-scope-val-fvu")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.44",
        "sae-lens>=4.0",
        "h5py",
        "tqdm",
        "huggingface-hub",
        "numpy",
        "pandas",
        "accelerate",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

N_LAYERS = 26
D_IN = 2304
D_SAE = 16384
N_TRAINING_SAMPLES = 50_000
N_VAL_SAMPLES = 5_000
CHUNK_SIZE = 1_000
RESULTS_BASE = "/vol/results/paligemma2"
METHODS = ["pretrained", "random"]
N_GPUS = 8


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def fvu_worker(worker_id: int, layer_indices: list):
    """Compute validation FVU per layer, broken down by Full/Image/Text tokens."""
    import sys
    import gc
    import json
    import torch
    import numpy as np
    import h5py
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_sae

    out_dir = Path(RESULTS_BASE) / "analysis" / "val_fvu"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(RESULTS_BASE) / "run" / "checkpoints"
    act_dir = Path(RESULTS_BASE) / "run" / "activations"

    # Validation chunks: samples 50000-55000
    val_chunks = []
    for ci in range(N_TRAINING_SAMPLES // CHUNK_SIZE,
                     (N_TRAINING_SAMPLES + N_VAL_SAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE):
        cs = ci * CHUNK_SIZE
        ce = min(cs + CHUNK_SIZE, N_TRAINING_SAMPLES + N_VAL_SAMPLES)
        h5_path = act_dir / f"chunk_{cs}_{ce}.h5"
        if h5_path.exists():
            val_chunks.append((cs, ce, h5_path))

    print(f"[FVU W{worker_id}] Layers: {layer_indices}, val chunks: {len(val_chunks)}")

    results = []

    for layer_idx in tqdm(layer_indices, desc=f"W{worker_id} layers"):
        for method in METHODS:
            ckpt_path = ckpt_dir / f"{method}_layer_{layer_idx}.pt"
            if not ckpt_path.exists():
                print(f"[FVU W{worker_id}] SKIP {method} L{layer_idx} — no checkpoint")
                continue

            sae = initialize_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                 device="cuda", cache_dir="/vol/cache/huggingface")
            sae.eval()

            # Accumulators for online variance computation
            # Full tokens
            sum_sq_err_full = 0.0
            sum_sq_var_full = 0.0
            sum_acts_full = np.zeros(D_IN, dtype=np.float64)
            n_full = 0

            # Image tokens
            sum_sq_err_img = 0.0
            sum_sq_var_img = 0.0
            sum_acts_img = np.zeros(D_IN, dtype=np.float64)
            n_img = 0

            # Text tokens
            sum_sq_err_txt = 0.0
            sum_sq_var_txt = 0.0
            sum_acts_txt = np.zeros(D_IN, dtype=np.float64)
            n_txt = 0

            # First pass: compute means
            for cs, ce, h5_path in tqdm(val_chunks, desc=f"W{worker_id} {method} L{layer_idx} mean",
                                         leave=False):
                with h5py.File(h5_path, "r") as f:
                    grp = f.get(f"layer_{layer_idx}")
                    if grp is None:
                        continue

                    for si in range(cs, ce):
                        key = f"sample_{si}"
                        if key not in grp:
                            continue

                        ds = grp[key]
                        act = ds[:]  # (seq, d_in) as numpy
                        img_s = int(ds.attrs.get("img_start", 0))
                        img_e = int(ds.attrs.get("img_end", 0))

                        # Full
                        sum_acts_full += act.sum(axis=0).astype(np.float64)
                        n_full += act.shape[0]

                        # Image
                        if img_e > img_s:
                            img_act = act[img_s:img_e]
                            sum_acts_img += img_act.sum(axis=0).astype(np.float64)
                            n_img += img_act.shape[0]

                        # Text
                        txt_parts = []
                        if img_s > 0:
                            txt_parts.append(act[:img_s])
                        if img_e < act.shape[0]:
                            txt_parts.append(act[img_e:])
                        if txt_parts:
                            txt_act = np.concatenate(txt_parts, axis=0)
                            sum_acts_txt += txt_act.sum(axis=0).astype(np.float64)
                            n_txt += txt_act.shape[0]

            mean_full = sum_acts_full / max(n_full, 1)
            mean_img = sum_acts_img / max(n_img, 1)
            mean_txt = sum_acts_txt / max(n_txt, 1)

            # Second pass: compute FVU = MSE(recon) / Var(original)
            for cs, ce, h5_path in tqdm(val_chunks, desc=f"W{worker_id} {method} L{layer_idx} fvu",
                                         leave=False):
                with h5py.File(h5_path, "r") as f:
                    grp = f.get(f"layer_{layer_idx}")
                    if grp is None:
                        continue

                    for si in range(cs, ce):
                        key = f"sample_{si}"
                        if key not in grp:
                            continue

                        ds = grp[key]
                        act_np = ds[:]  # (seq, d_in)
                        img_s = int(ds.attrs.get("img_start", 0))
                        img_e = int(ds.attrs.get("img_end", 0))

                        act_t = torch.from_numpy(act_np).float().to("cuda")

                        # Reconstruct through SAE
                        with torch.no_grad():
                            recon = sae(act_t.unsqueeze(0)).squeeze(0)  # (seq, d_in)

                        err = (recon - act_t).cpu().numpy().astype(np.float64)
                        act_cpu = act_np.astype(np.float64)

                        # Full tokens
                        sum_sq_err_full += (err ** 2).sum()
                        sum_sq_var_full += ((act_cpu - mean_full) ** 2).sum()

                        # Image tokens
                        if img_e > img_s:
                            sum_sq_err_img += (err[img_s:img_e] ** 2).sum()
                            sum_sq_var_img += ((act_cpu[img_s:img_e] - mean_img) ** 2).sum()

                        # Text tokens
                        txt_err_parts = []
                        txt_var_parts = []
                        if img_s > 0:
                            txt_err_parts.append(err[:img_s])
                            txt_var_parts.append(act_cpu[:img_s] - mean_txt)
                        if img_e < act_cpu.shape[0]:
                            txt_err_parts.append(err[img_e:])
                            txt_var_parts.append(act_cpu[img_e:] - mean_txt)
                        if txt_err_parts:
                            txt_err = np.concatenate(txt_err_parts, axis=0)
                            txt_var = np.concatenate(txt_var_parts, axis=0)
                            sum_sq_err_txt += (txt_err ** 2).sum()
                            sum_sq_var_txt += (txt_var ** 2).sum()

            fvu_full = sum_sq_err_full / max(sum_sq_var_full, 1e-12)
            fvu_img = sum_sq_err_img / max(sum_sq_var_img, 1e-12)
            fvu_txt = sum_sq_err_txt / max(sum_sq_var_txt, 1e-12)

            result = {
                "layer": layer_idx,
                "method": method,
                "fvu_full": float(fvu_full),
                "fvu_image": float(fvu_img),
                "fvu_text": float(fvu_txt),
                "n_full_tokens": int(n_full),
                "n_image_tokens": int(n_img),
                "n_text_tokens": int(n_txt),
            }
            results.append(result)

            print(f"[FVU W{worker_id}] {method} L{layer_idx}: "
                  f"full={fvu_full:.4f}, img={fvu_img:.4f}, txt={fvu_txt:.4f} "
                  f"(tokens: {n_full:,} full, {n_img:,} img, {n_txt:,} txt)")

            del sae
            torch.cuda.empty_cache()
            gc.collect()

    # Save per-worker results
    out_path = out_dir / f"fvu_worker_{worker_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()
    return f"FVU W{worker_id}: layers {layer_indices} done"


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def merge_fvu_results():
    """Merge per-worker FVU results into a single table."""
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    fvu_dir = Path(RESULTS_BASE) / "analysis" / "val_fvu"

    all_results = []
    for f_path in sorted(fvu_dir.glob("fvu_worker_*.json")):
        with open(f_path) as f:
            all_results.extend(json.load(f))

    if not all_results:
        print("[WARN] No FVU results found")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(["method", "layer"])

    # Save full table
    df.to_csv(fvu_dir / "val_fvu_table.csv", index=False)

    # Print summary table matching paper format
    print("\n" + "=" * 80)
    print("Validation FVU Table (PaliGemma2-3B SAE)")
    print("=" * 80)

    for method in METHODS:
        mdf = df[df["method"] == method]
        if mdf.empty:
            continue

        print(f"\n--- {method.upper()} ---")
        print(f"{'Layer':<8} {'Full':<10} {'Image':<10} {'Text':<10}")
        for _, row in mdf.iterrows():
            print(f"{int(row['layer']):<8} {row['fvu_full']:.4f}     "
                  f"{row['fvu_image']:.4f}     {row['fvu_text']:.4f}")

        # Summary stats
        print(f"\n{'Metric':<10} {'Full':<10} {'Image':<10} {'Text':<10}")
        for stat, fn in [("Mean", np.mean), ("Std", np.std), ("Min", np.min), ("Max", np.max)]:
            print(f"{stat:<10} {fn(mdf['fvu_full']):.4f}     "
                  f"{fn(mdf['fvu_image']):.4f}     {fn(mdf['fvu_text']):.4f}")

        if not mdf.empty:
            row0 = mdf.iloc[0]
            n_full_M = row0["n_full_tokens"] / 1e6
            n_img_M = row0["n_image_tokens"] / 1e6
            n_txt_M = row0["n_text_tokens"] / 1e6
            print(f"{'Tokens(M)':<10} {n_full_M:.1f}       {n_img_M:.1f}       {n_txt_M:.1f}")

    # Save summary JSON
    summary = {}
    for method in METHODS:
        mdf = df[df["method"] == method]
        if mdf.empty:
            continue
        summary[method] = {
            "full": {"mean": float(mdf["fvu_full"].mean()), "std": float(mdf["fvu_full"].std()),
                     "min": float(mdf["fvu_full"].min()), "max": float(mdf["fvu_full"].max())},
            "image": {"mean": float(mdf["fvu_image"].mean()), "std": float(mdf["fvu_image"].std()),
                      "min": float(mdf["fvu_image"].min()), "max": float(mdf["fvu_image"].max())},
            "text": {"mean": float(mdf["fvu_text"].mean()), "std": float(mdf["fvu_text"].std()),
                     "min": float(mdf["fvu_text"].min()), "max": float(mdf["fvu_text"].max())},
        }

    with open(fvu_dir / "val_fvu_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    print(f"\n[FVU] Results saved to {fvu_dir}")
    return summary


@app.local_entrypoint()
def main():
    import math

    # Distribute 26 layers across 8 GPUs
    layers_per_worker = math.ceil(N_LAYERS / N_GPUS)
    assignments = []
    for w in range(N_GPUS):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, N_LAYERS)
        worker_layers = list(range(start, end))
        if worker_layers:
            assignments.append((w, worker_layers))

    print(f"{'=' * 60}")
    print(f"[Validation FVU] {N_LAYERS} layers x {len(METHODS)} methods across {len(assignments)} GPUs")
    print(f"{'=' * 60}")
    for w, layers in assignments:
        print(f"  GPU {w}: layers {layers}")

    # Run FVU computation across GPUs
    results = list(fvu_worker.starmap(assignments))
    for r in results:
        print(r)

    # Merge and display results
    print(f"\n{'=' * 60}")
    print("[Merging results...]")
    print(f"{'=' * 60}")
    summary = merge_fvu_results.remote()
    print(f"\nSummary: {summary}")

    print(f"\n{'=' * 60}")
    print("[SUCCESS] Validation FVU computation complete!")
    print(f"{'=' * 60}")
