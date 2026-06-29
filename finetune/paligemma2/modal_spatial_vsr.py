"""
Redo spatial feature selection: compare VQA vs VSR firing frequencies.

Step 1: Compute firing frequencies on VQA activations (already cached)
Step 2: Compute firing frequencies on VSR activations (from modal_cache_vsr_activations.py)
Step 3: Fisher exact test — features that fire significantly MORE on VSR than VQA
Step 4: Intersect with adapted features → final spatial visual features

Usage:
    MODAL_PROFILE=hunar-oxford modal run modal_spatial_vsr.py
"""
import os
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
GPU_TYPE = "A100"
TIMEOUT = 86400

app = modal.App("vlm-scope-spatial-vsr")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers>=4.44", "sae-lens>=4.0",
        "nnsight>=0.3", "datasets", "h5py", "tqdm", "huggingface-hub",
        "Pillow", "numpy", "scipy", "statsmodels", "pandas", "accelerate",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_MODE": "disabled",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

RESULTS_BASE = "/vol/results/paligemma2"
N_LAYERS = 26
D_SAE = 16384
SAE_TYPE = "jumprelu"
N_GPUS = 8

# Spatial feature identification parameters
P_THR = 0.01
ODDS_THR = 2.0  # Slightly relaxed from 3.0 since VSR is a cleaner signal
MIN_FREQ_DIFF = 0.003


@app.function(image=image, gpu=GPU_TYPE, volumes={"/vol": volume}, timeout=TIMEOUT)
def firing_worker(worker_id: int, layer_indices: list):
    """Compute per-feature firing frequencies on VQA and VSR activations."""
    import sys
    import json
    import torch
    import h5py
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    from collections import defaultdict

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_jumprelu_sae

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    ckpt_dir = Path(RESULTS_BASE) / f"run{sae_suffix}" / "checkpoints"
    vqa_act_dir = Path(RESULTS_BASE) / "run" / "activations"
    vsr_act_dir = Path(RESULTS_BASE) / "run" / "activations_vsr"
    out_dir = Path(RESULTS_BASE) / "analysis" / f"firing_vsr{sae_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Firing W{worker_id}] Layers: {layer_indices}")

    # Check what activation files exist
    vqa_chunks = sorted(vqa_act_dir.glob("chunk_*.h5"))
    vsr_chunks = sorted(vsr_act_dir.glob("vsr_chunk_*.h5"))
    print(f"[Firing W{worker_id}] VQA chunks: {len(vqa_chunks)}, VSR chunks: {len(vsr_chunks)}")

    if not vsr_chunks:
        return f"W{worker_id}: NO VSR activations found!"

    for layer_idx in layer_indices:
        print(f"\n[Firing W{worker_id}] Layer {layer_idx}")

        # Skip if already computed with data
        out_path_check = out_dir / f"firing_vsr_layer_{layer_idx}.json"
        if out_path_check.exists():
            import json as json_check
            with open(str(out_path_check)) as fc:
                existing = json_check.load(fc)
            if existing.get("n_vqa", 0) > 0 and existing.get("n_vsr", 0) > 0:
                print(f"  SKIP — already computed (n_vqa={existing['n_vqa']}, n_vsr={existing['n_vsr']})")
                continue

        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP — no checkpoint")
            continue

        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device="cuda", cache_dir="/vol/cache/huggingface")
        sae.eval()

        layer_key = f"layer_{layer_idx}"

        # --- VQA firing frequencies ---
        # Use first 10 VQA chunks (10K samples) for balanced comparison with VSR
        vqa_fire_count = np.zeros(D_SAE, dtype=np.int64)
        vqa_fire_count_img = np.zeros(D_SAE, dtype=np.int64)
        vqa_n_samples = 0

        for cf in tqdm(vqa_chunks[:10], desc=f"W{worker_id} L{layer_idx} VQA"):
            try:
                with h5py.File(str(cf), "r") as hf:
                    if layer_key not in hf:
                        continue
                    layer_grp = hf[layer_key]
                    for sample_key in layer_grp.keys():
                        if not sample_key.startswith("sample_"):
                            continue
                        ds = layer_grp[sample_key]
                        act = torch.from_numpy(ds[:]).float().to("cuda")
                        img_start = int(ds.attrs.get("img_start", 0))
                        img_end = int(ds.attrs.get("img_end", 256))

                        with torch.no_grad():
                            codes = sae.encode(act)  # (seq, d_sae)

                        # Feature fires if max activation > 0 anywhere in sequence
                        fired = (codes.max(dim=0).values > 0).cpu().numpy()
                        vqa_fire_count += fired.astype(np.int64)

                        # Image-token firing
                        if img_end > img_start:
                            img_fired = (codes[img_start:img_end].max(dim=0).values > 0).cpu().numpy()
                            vqa_fire_count_img += img_fired.astype(np.int64)

                        vqa_n_samples += 1
            except Exception as e:
                print(f"  [WARN] VQA {cf.name}: {e}")

        print(f"  VQA: {vqa_n_samples} samples processed")

        # --- VSR firing frequencies ---
        vsr_fire_count = np.zeros(D_SAE, dtype=np.int64)
        vsr_fire_count_img = np.zeros(D_SAE, dtype=np.int64)
        vsr_n_samples = 0
        vsr_spatial_fire = np.zeros(D_SAE, dtype=np.int64)  # spatial relations only
        vsr_n_spatial = 0

        SPATIAL_RELATIONS = {
            "above", "below", "left", "right", "behind", "in front of",
            "on", "under", "beside", "near", "next to", "between",
            "inside", "outside", "on top of", "beneath", "over",
            "across", "along", "around", "through", "toward", "away from",
        }
        NON_SPATIAL_RELATIONS = {
            "has", "wears", "holds", "made of", "part of", "contains",
            "wearing", "holding", "carrying", "eating", "playing",
        }

        for cf in tqdm(vsr_chunks, desc=f"W{worker_id} L{layer_idx} VSR"):
            try:
                with h5py.File(str(cf), "r") as hf:
                    if layer_key not in hf:
                        continue
                    layer_grp = hf[layer_key]
                    for sample_key in layer_grp.keys():
                        if not sample_key.startswith("sample_"):
                            continue
                        ds = layer_grp[sample_key]
                        act = torch.from_numpy(ds[:]).float().to("cuda")
                        img_start = int(ds.attrs.get("img_start", 0))
                        img_end = int(ds.attrs.get("img_end", 256))
                        relation = ds.attrs.get("relation", "")

                        with torch.no_grad():
                            codes = sae.encode(act)

                        fired = (codes.max(dim=0).values > 0).cpu().numpy()
                        vsr_fire_count += fired.astype(np.int64)

                        if img_end > img_start:
                            img_fired = (codes[img_start:img_end].max(dim=0).values > 0).cpu().numpy()
                            vsr_fire_count_img += img_fired.astype(np.int64)

                        # Track spatial vs non-spatial
                        rel_lower = relation.lower().strip() if relation else ""
                        is_spatial = any(s in rel_lower for s in SPATIAL_RELATIONS) and \
                                     not any(s in rel_lower for s in NON_SPATIAL_RELATIONS)
                        if is_spatial:
                            vsr_spatial_fire += fired.astype(np.int64)
                            vsr_n_spatial += 1

                        vsr_n_samples += 1
            except Exception as e:
                print(f"  [WARN] VSR {cf.name}: {e}")

        print(f"  VSR: {vsr_n_samples} samples ({vsr_n_spatial} spatial)")

        # Save per-layer results
        result = {
            "n_vqa": vqa_n_samples,
            "n_vsr": vsr_n_samples,
            "n_vsr_spatial": vsr_n_spatial,
            "fire_count_vqa": vqa_fire_count.tolist(),
            "fire_count_vqa_img": vqa_fire_count_img.tolist(),
            "fire_count_vsr": vsr_fire_count.tolist(),
            "fire_count_vsr_img": vsr_fire_count_img.tolist(),
            "fire_count_vsr_spatial": vsr_spatial_fire.tolist(),
        }
        out_path = out_dir / f"firing_vsr_layer_{layer_idx}.json"
        with open(str(out_path), "w") as f:
            json.dump(result, f)

        del sae
        torch.cuda.empty_cache()
        print(f"  Saved to {out_path}")

    volume.commit()
    return f"Firing W{worker_id}: {len(layer_indices)} layers done"


@app.function(image=image, volumes={"/vol": volume}, timeout=600)
def find_spatial_features_vsr():
    """Fisher exact test: features firing more on VSR than VQA."""
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    firing_dir = Path(RESULTS_BASE) / "analysis" / f"firing_vsr{sae_suffix}"
    out_dir = Path(RESULTS_BASE) / "analysis" / f"spatial_vsr{sae_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for layer_idx in range(N_LAYERS):
        firing_path = firing_dir / f"firing_vsr_layer_{layer_idx}.json"
        if not firing_path.exists():
            continue

        with open(str(firing_path)) as f:
            data = json.load(f)

        n_vqa = data["n_vqa"]
        n_vsr = data["n_vsr"]
        fire_vqa = np.array(data["fire_count_vqa"])
        fire_vsr = np.array(data["fire_count_vsr"])
        fire_vqa_img = np.array(data["fire_count_vqa_img"])
        fire_vsr_img = np.array(data["fire_count_vsr_img"])

        if n_vqa == 0 or n_vsr == 0:
            continue

        freq_vqa = fire_vqa / n_vqa
        freq_vsr = fire_vsr / n_vsr
        freq_diff = freq_vsr - freq_vqa

        # Also compute image-token frequency difference
        freq_vqa_img = fire_vqa_img / n_vqa
        freq_vsr_img = fire_vsr_img / n_vsr
        freq_diff_img = freq_vsr_img - freq_vqa_img

        # Candidates: fire more on VSR than VQA
        candidates = np.where(freq_diff > MIN_FREQ_DIFF)[0]
        if len(candidates) == 0:
            print(f"  L{layer_idx}: 0 candidates (freq_diff)")
            continue

        p_values = []
        odds_ratios = []
        for fi in candidates:
            a = int(fire_vsr[fi])
            b = int(n_vsr - fire_vsr[fi])
            c = int(fire_vqa[fi])
            d = int(n_vqa - fire_vqa[fi])

            table = [[a, b], [c, d]]
            odds, p = fisher_exact(table, alternative="greater")
            p_values.append(p)
            odds_ratios.append(odds)

        # Multiple testing correction
        if len(p_values) > 0:
            reject, p_adj, _, _ = multipletests(p_values, alpha=P_THR, method="fdr_bh")
        else:
            p_adj = []

        n_sig = 0
        for i, fi in enumerate(candidates):
            if p_adj[i] < P_THR and odds_ratios[i] > ODDS_THR:
                all_results.append({
                    "layer": layer_idx,
                    "feature": int(fi),
                    "freq_vqa": float(freq_vqa[fi]),
                    "freq_vsr": float(freq_vsr[fi]),
                    "freq_diff": float(freq_diff[fi]),
                    "freq_vqa_img": float(freq_vqa_img[fi]),
                    "freq_vsr_img": float(freq_vsr_img[fi]),
                    "freq_diff_img": float(freq_diff_img[fi]),
                    "odds_ratio": float(odds_ratios[i]),
                    "p_adj": float(p_adj[i]),
                })
                n_sig += 1

        print(f"  L{layer_idx}: {len(candidates)} candidates, {n_sig} significant "
              f"(n_vqa={n_vqa}, n_vsr={n_vsr})")

    # Save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(str(out_dir / "spatial_features_vsr.csv"), index=False)
        print(f"\n[Spatial VSR] Total: {len(all_results)} features across "
              f"{df['layer'].nunique()} layers")

        # Top features by odds ratio
        df_sorted = df.sort_values("odds_ratio", ascending=False)
        print("\nTop 20 by odds ratio:")
        for _, row in df_sorted.head(20).iterrows():
            print(f"  L{int(row['layer'])}/F{int(row['feature'])}: "
                  f"OR={row['odds_ratio']:.2f}, freq_diff={row['freq_diff']:.4f}, "
                  f"img_diff={row['freq_diff_img']:.4f}")
    else:
        print("[Spatial VSR] No significant features found")

    summary = {
        "total_spatial_vsr": len(all_results),
        "p_threshold": P_THR,
        "odds_threshold": ODDS_THR,
        "min_freq_diff": MIN_FREQ_DIFF,
    }
    with open(str(out_dir / "spatial_vsr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    return summary


@app.function(image=image, volumes={"/vol": volume}, timeout=600)
def intersect_adapted_spatial_vsr():
    """Intersect adapted features with VSR-based spatial features."""
    import ast
    import csv
    import json
    import pandas as pd
    from pathlib import Path

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    analysis_dir = Path(RESULTS_BASE) / "analysis"
    out_dir = analysis_dir / f"final_features_vsr{sae_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load adapted features
    adapted_path = analysis_dir / f"adapted{sae_suffix}" / "adapted_features_results.csv"
    adapted_by_layer = {}
    if adapted_path.exists():
        with open(str(adapted_path)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer"])
                indices = ast.literal_eval(row["adapted_indices"])
                adapted_by_layer[layer] = set(indices)
        print(f"[Intersect] Adapted: {sum(len(v) for v in adapted_by_layer.values())} features")
    else:
        print(f"[ERROR] No adapted features at {adapted_path}")
        return {}

    # Load VSR spatial features
    spatial_path = analysis_dir / f"spatial_vsr{sae_suffix}" / "spatial_features_vsr.csv"
    spatial_by_layer = {}
    if spatial_path.exists():
        df = pd.read_csv(str(spatial_path))
        for layer in df["layer"].unique():
            spatial_by_layer[int(layer)] = set(df[df["layer"] == layer]["feature"].tolist())
        print(f"[Intersect] Spatial VSR: {sum(len(v) for v in spatial_by_layer.values())} features")
    else:
        print(f"[ERROR] No spatial VSR features at {spatial_path}")
        return {}

    # Intersect
    final_features = []
    all_layers = sorted(set(adapted_by_layer.keys()) | set(spatial_by_layer.keys()))

    for layer in all_layers:
        adapted = adapted_by_layer.get(layer, set())
        spatial = spatial_by_layer.get(layer, set())
        common = adapted & spatial

        for fi in sorted(common):
            final_features.append({"layer": layer, "feature": fi})

        if common:
            print(f"  L{layer}: adapted={len(adapted)}, spatial_vsr={len(spatial)}, "
                  f"intersection={len(common)}")

    if final_features:
        df_final = pd.DataFrame(final_features)
        df_final.to_csv(str(out_dir / "final_spatial_visual_features.csv"), index=False)

    summary = {
        "total_final": len(final_features),
        "total_adapted": sum(len(v) for v in adapted_by_layer.values()),
        "total_spatial_vsr": sum(len(v) for v in spatial_by_layer.values()),
    }
    with open(str(out_dir / "intersection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    print(f"\n[Intersect] Final: {len(final_features)} features (adapted ∩ spatial_vsr)")
    return summary


@app.local_entrypoint()
def main():
    import math

    # Step 1+2: Compute firing frequencies on both VQA and VSR
    print("=" * 60)
    print("Step 1+2: Firing frequencies (VQA vs VSR)")
    print("=" * 60)

    layers_per_worker = math.ceil(N_LAYERS / N_GPUS)
    assignments = []
    for w in range(N_GPUS):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, N_LAYERS)
        worker_layers = list(range(start, end))
        if worker_layers:
            assignments.append((w, worker_layers))

    print(f"Distributing {N_LAYERS} layers across {len(assignments)} GPUs")
    for w, layers in assignments:
        print(f"  GPU {w}: layers {layers}")

    results = list(firing_worker.starmap(assignments))
    for r in results:
        print(r)

    # Step 3: Fisher exact test
    print("\n" + "=" * 60)
    print("Step 3: Identify spatial features (VQA vs VSR)")
    print("=" * 60)

    spatial_summary = find_spatial_features_vsr.remote()
    print(f"Spatial VSR: {spatial_summary}")

    # Step 4: Intersect with adapted
    print("\n" + "=" * 60)
    print("Step 4: Intersect adapted ∩ spatial_vsr")
    print("=" * 60)

    intersection = intersect_adapted_spatial_vsr.remote()
    print(f"Final: {intersection}")

    print("\n[DONE] New spatial feature selection complete!")
