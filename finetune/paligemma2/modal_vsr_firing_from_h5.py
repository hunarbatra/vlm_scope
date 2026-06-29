"""
Compute VSR firing frequencies from cached H5 activations.

Much simpler than modal_vsr_firing.py — no model loading, no image downloads.
Just reads cached activations → encodes through SAE → tracks firings.

Then compares VQA vs VSR using Fisher exact test (global FDR, matching old pipeline).

Usage:
    MODAL_PROFILE=hunar-oxford modal run modal_vsr_firing_from_h5.py
"""

import math
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("vlm-scope-vsr-firing-h5")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers>=4.44", "sae-lens>=4.0",
        "nnsight>=0.3", "h5py", "tqdm", "huggingface-hub",
        "numpy", "scipy", "statsmodels", "pandas", "accelerate",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR",
        "HUGGING_FACE_HUB_TOKEN": "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR",
        "WANDB_MODE": "disabled",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

N_LAYERS = 26
D_SAE = 16384
D_IN = 2304
N_GPUS = 8
RESULTS_BASE = "/vol/results/paligemma2"
SAE_TYPE = "jumprelu"


# ================================================================
#  Step 1: Compute VSR firing from cached H5 activations (GPU)
# ================================================================

@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=7200,
)
def vsr_firing_h5_worker(worker_id: int, layer_indices: list):
    """Compute per-feature firing stats from cached VSR H5 activations."""
    import sys
    import gc
    import json
    import torch
    import numpy as np
    import h5py
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_jumprelu_sae

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    out_dir = Path(RESULTS_BASE) / "analysis" / f"firing_vsr{sae_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(RESULTS_BASE) / f"run{sae_suffix}" / "checkpoints"
    act_dir = Path(RESULTS_BASE) / "run" / "activations_vsr"

    print(f"[VSR-H5 W{worker_id}] Layers: {layer_indices}")

    # Find all VSR H5 chunks
    h5_files = sorted(act_dir.glob("vsr_chunk_*.h5"))
    print(f"[VSR-H5 W{worker_id}] Found {len(h5_files)} H5 chunks")
    for f in h5_files:
        print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")

    for layer_idx in layer_indices:
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"[VSR-H5 W{worker_id}] SKIP L{layer_idx} — no checkpoint")
            continue

        # Check if already computed
        out_path = out_dir / f"firing_vsr_layer_{layer_idx}.json"
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
                if existing.get("n_samples", 0) > 0:
                    print(f"[VSR-H5 W{worker_id}] SKIP L{layer_idx} — already has {existing['n_samples']} samples")
                    continue

        sae = initialize_jumprelu_sae(
            layer_idx, checkpoint_path=str(ckpt_path),
            device="cuda", cache_dir="/vol/cache/huggingface"
        )
        sae.eval()
        print(f"[VSR-H5 W{worker_id}] Processing L{layer_idx}")

        # Sample-level counters
        fire_count_all = np.zeros(D_SAE, dtype=np.int64)
        fire_count_img = np.zeros(D_SAE, dtype=np.int64)
        # Token-level counters
        token_firing_count = np.zeros(D_SAE, dtype=np.int64)
        total_tokens = 0
        n_samples = 0
        n_failed = 0

        for h5_path in tqdm(h5_files, desc=f"W{worker_id} L{layer_idx} chunks"):
            try:
                with h5py.File(str(h5_path), "r") as hf:
                    grp = hf.get(f"layer_{layer_idx}")
                    if grp is None:
                        print(f"  [WARN] No layer_{layer_idx} in {h5_path.name}")
                        continue

                    sample_keys = [k for k in grp.keys() if k.startswith("sample_")]
                    for key in sample_keys:
                        try:
                            ds = grp[key]
                            act = torch.from_numpy(ds[:]).to("cuda").float()  # (seq, d_in)
                            img_s = int(ds.attrs.get("img_start", 0))
                            img_e = int(ds.attrs.get("img_end", 0))

                            with torch.no_grad():
                                codes = sae.encode(act)  # (seq, d_sae)

                            seq_len = codes.shape[0]
                            total_tokens += seq_len

                            # Sample-level
                            fired = (codes != 0).any(dim=0).cpu().numpy()
                            fire_count_all += fired.astype(np.int64)

                            # Token-level
                            token_firing_count += (codes != 0).sum(dim=0).cpu().numpy().astype(np.int64)

                            # Image token firings (sample-level)
                            if img_e > img_s:
                                img_fired = (codes[img_s:img_e] != 0).any(dim=0).cpu().numpy()
                                fire_count_img += img_fired.astype(np.int64)

                            n_samples += 1
                        except Exception as e:
                            n_failed += 1
                            if n_failed <= 3:
                                print(f"  [WARN] {key}: {e}")
            except Exception as e:
                print(f"  [ERROR] {h5_path.name}: {e}")

        # Save
        layer_data = {
            "layer": layer_idx,
            "n_samples": n_samples,
            "n_failed": n_failed,
            "total_tokens": int(total_tokens),
            "dataset": "vsr",
            "fire_count_all": fire_count_all.tolist(),
            "fire_count_img": fire_count_img.tolist(),
            "token_firing_count": token_firing_count.tolist(),
        }
        with open(out_path, "w") as f:
            json.dump(layer_data, f)

        print(f"[W{worker_id}] L{layer_idx}: {n_samples} samples, {n_failed} failed, "
              f"features >50%: {(fire_count_all > n_samples * 0.5).sum()}")

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    volume.commit()
    return f"VSR-H5 W{worker_id}: layers {layer_indices} done"


# ================================================================
#  Step 2: Fisher test — VQA vs VSR (CPU, global FDR)
# ================================================================

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=3600,
)
def find_spatial_features_vsr():
    """Compare VQA vs VSR firing using Fisher exact test.

    Matches old LLaVA-MORE pipeline:
    - Global FDR across all layers/features
    - Filter by odds_ratio + freq_diff (not p_adj)
    - Test ALL features (adapted intersection done separately)
    """
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    vqa_firing_dir = Path(RESULTS_BASE) / "analysis" / f"firing{sae_suffix}"
    vsr_firing_dir = Path(RESULTS_BASE) / "analysis" / f"firing_vsr{sae_suffix}"
    out_dir = Path(RESULTS_BASE) / "analysis" / f"spatial_vsr{sae_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load adapted features for post-hoc intersection
    adapted_path = Path(RESULTS_BASE) / "analysis" / f"adapted{sae_suffix}" / "adapted_features_results.csv"
    adapted_by_layer = {}
    if adapted_path.exists():
        import ast, csv
        with open(adapted_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer"])
                indices = ast.literal_eval(row["adapted_indices"])
                adapted_by_layer[layer] = set(indices)
        print(f"[INFO] Adapted features: {sum(len(v) for v in adapted_by_layer.values())} total")

    ODDS_THR = 2.0
    MIN_FREQ_DIFF = 0.005

    # Collect all rows across all layers
    all_rows = []
    for layer_idx in tqdm(range(N_LAYERS), desc="Loading firing stats"):
        vqa_path = vqa_firing_dir / f"firing_layer_{layer_idx}.json"
        vsr_path = vsr_firing_dir / f"firing_vsr_layer_{layer_idx}.json"

        if not vqa_path.exists() or not vsr_path.exists():
            print(f"  L{layer_idx}: SKIP (missing)")
            continue

        with open(vqa_path) as f:
            vqa_data = json.load(f)
        with open(vsr_path) as f:
            vsr_data = json.load(f)

        n_vqa = vqa_data["n_all"]
        n_vsr = vsr_data["n_samples"]
        fire_vqa = np.array(vqa_data["fire_count_all"])
        fire_vsr = np.array(vsr_data["fire_count_all"])

        if n_vqa == 0 or n_vsr == 0:
            print(f"  L{layer_idx}: SKIP (n_vqa={n_vqa}, n_vsr={n_vsr})")
            continue

        print(f"  L{layer_idx}: n_vqa={n_vqa}, n_vsr={n_vsr}")

        for fi in range(D_SAE):
            all_rows.append({
                "layer": layer_idx,
                "feature": fi,
                "c_vqa": int(fire_vqa[fi]),
                "n_vqa": n_vqa,
                "c_vsr": int(fire_vsr[fi]),
                "n_vsr": n_vsr,
            })

    if not all_rows:
        print("[ERROR] No data collected!")
        volume.commit()
        return {"total_spatial": 0, "error": "no data"}

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df)}, layers: {df['layer'].nunique()}")

    # Compute frequencies
    df["freq_vqa"] = df["c_vqa"] / df["n_vqa"]
    df["freq_vsr"] = df["c_vsr"] / df["n_vsr"]
    df["freq_diff"] = df["freq_vsr"] - df["freq_vqa"]

    # Fisher exact test for all features
    p_values = []
    odds_ratios = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Fisher tests"):
        c_vsr = min(int(r["c_vsr"]), int(r["n_vsr"]))
        c_vqa = min(int(r["c_vqa"]), int(r["n_vqa"]))
        table = [[c_vsr, max(0, int(r["n_vsr"]) - c_vsr)],
                 [c_vqa, max(0, int(r["n_vqa"]) - c_vqa)]]
        try:
            o, p = fisher_exact(table, alternative="greater")
            odds_ratios.append(o if not np.isinf(o) else 1e9)
            p_values.append(p)
        except Exception:
            odds_ratios.append(1.0)
            p_values.append(1.0)

    df["odds_ratio"] = odds_ratios
    df["p_raw"] = p_values

    # Global FDR correction
    df["p_adj"] = multipletests(df["p_raw"].values, method="fdr_bh")[1]

    # Filter: odds_ratio + freq_diff (matching old pipeline)
    spatial = df[(df["odds_ratio"] >= ODDS_THR) & (df["freq_diff"] >= MIN_FREQ_DIFF)].copy()
    spatial = spatial.sort_values("odds_ratio", ascending=False)

    print(f"\n[RESULT] {len(spatial)} spatial features (before adapted intersection)")
    spatial.to_csv(out_dir / "spatial_features_vsr_all.csv", index=False)

    # Intersect with adapted
    if adapted_by_layer:
        mask = spatial.apply(
            lambda r: int(r["feature"]) in adapted_by_layer.get(int(r["layer"]), set()),
            axis=1
        )
        final = spatial[mask].copy()
        print(f"[RESULT] {len(final)} spatial+adapted features")
    else:
        final = spatial
        print("[WARN] No adapted features — using all spatial")

    final_dir = Path(RESULTS_BASE) / "analysis" / f"final_features_vsr{sae_suffix}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(final_dir / "final_spatial_visual_features.csv", index=False)
    spatial.to_csv(out_dir / "spatial_features_vsr.csv", index=False)

    # Print top features
    print(f"\nTop 30 (adapted ∩ spatial):")
    print(f"{'Layer':<6} {'Feat':<8} {'OR':>8} {'f_vqa':>8} {'f_vsr':>8} {'diff':>8} {'p_adj':>10}")
    for _, r in final.head(30).iterrows():
        print(f"  L{int(r['layer']):<4} F{int(r['feature']):<6} "
              f"{r['odds_ratio']:>8.2f} {r['freq_vqa']:>8.4f} {r['freq_vsr']:>8.4f} "
              f"{r['freq_diff']:>8.4f} {r['p_adj']:>10.2e}")

    summary = {
        "total_spatial_all": len(spatial),
        "total_spatial_adapted": len(final),
        "odds_threshold": ODDS_THR,
        "min_freq_diff": MIN_FREQ_DIFF,
        "fdr_method": "global_bh",
        "method": "VQA vs VSR sample-level (from H5 cache)",
    }
    with open(out_dir / "spatial_vsr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    return summary


# ================================================================
#  Entrypoint
# ================================================================

@app.local_entrypoint()
def main():
    # Distribute 26 layers across 8 GPUs
    layers_per_worker = math.ceil(N_LAYERS / N_GPUS)
    assignments = []
    for w in range(N_GPUS):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, N_LAYERS)
        worker_layers = list(range(start, end))
        if worker_layers:
            assignments.append((w, worker_layers))

    print(f"\n{'='*60}")
    print(f"[Step 1] VSR Firing from H5 — {len(assignments)} GPU workers")
    print(f"{'='*60}")
    for w, layers in assignments:
        print(f"  GPU {w}: layers {layers}")

    results = list(vsr_firing_h5_worker.starmap(assignments))
    for r in results:
        print(r)

    # Step 2: Fisher test
    print(f"\n{'='*60}")
    print(f"[Step 2] Spatial Feature Selection (VQA vs VSR)")
    print(f"{'='*60}")

    summary = find_spatial_features_vsr.remote()
    print(f"\nResult: {summary}")

    print(f"\n{'='*60}")
    print("[DONE] Spatial feature selection complete!")
    print(f"{'='*60}")
