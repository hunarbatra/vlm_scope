"""
Modal deployment for PaliGemma2 SAE analysis pipeline (8-GPU parallel).

Implements Steps 1-8 of the analysis plan:
  Phase A (fast):  Step 1 FVU table, Step 2 cosine similarities
  Phase B (heavy): Step 3 visual energy Ev, Step 5 firing frequencies (8 GPUs each)
  Phase C (CPU):   Step 4 adapted features, Step 6 spatial features, Step 8 intersection
  Phase D (GPU):   Step 7 lexical artifact filtering

Usage:
    cd finetune/paligemma2
    MODAL_PROFILE=hunar-oxford modal run modal_analysis.py
"""

import os
import modal
from pathlib import Path

# --------------- Modal configuration ---------------

VOLUME_NAME = "vlm-scope-data-v2"
GPU_TYPE = "A100"
TIMEOUT = 86400  # 24h

app = modal.App("vlm-scope-analysis")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.44",
        "sae-lens>=4.0",
        "nnsight>=0.3",
        "datasets",
        "h5py",
        "tqdm",
        "huggingface-hub",
        "Pillow",
        "numpy",
        "scipy",
        "statsmodels",
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

# --------------- Constants ---------------

N_LAYERS = 26
D_SAE = 16384
N_TRAINING_SAMPLES = 50_000
N_VAL_SAMPLES = 5_000
CHUNK_SIZE = 1_000
RESULTS_BASE = "/vol/results/paligemma2"
MODEL_NAME = "google/paligemma2-3b-pt-224"
METHODS = ["pretrained", "random"]
N_GPUS = 8

# Spatial keywords for filtering VQA questions
SPATIAL_KEYWORDS = [
    "above", "below", "left", "right", "between", "next to", "near",
    "behind", "in front", "on top", "under", "beside", "adjacent",
    "opposite", "across", "toward", "away from", "inside", "outside",
    "surrounding", "closer", "farther", "top", "bottom", "middle",
    "center", "corner", "edge", "side",
]

# Generic prompts for lexical artifact filtering
GENERIC_PROMPTS = [
    "Describe how the items are arranged.",
    "Comment on the overall layout and organization of the scene.",
    "Summarize the structure in terms of grouping or separation.",
    "Explain the relative positioning of objects without naming directions.",
    "Describe patterns of arrangement, such as order or symmetry.",
]

# Adapted feature selection parameters
EPSILON = 0.01
COSINE_PERCENTILE = 25.0

# Spatial feature identification parameters
P_THR = 0.01
ODDS_THR = 3.0
MIN_FREQ_DIFF = 0.005


# ================================================================
#  Step 1: FVU Table (no GPU needed)
# ================================================================

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def generate_fvu_table():
    """Read training log CSVs and extract final FVU per layer per method."""
    import csv
    from pathlib import Path

    log_dir = Path(RESULTS_BASE) / "run" / "logs"
    out_dir = Path(RESULTS_BASE) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for method in METHODS:
        results[method] = {}
        for layer_idx in range(N_LAYERS):
            csv_path = log_dir / f"metrics_{method}_layer_{layer_idx}.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing log: {csv_path}")
                results[method][layer_idx] = None
                continue

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                last = rows[-1]
                fvu = float(last["fvu"])
                tokens = int(last["total_tokens"])
                results[method][layer_idx] = {"fvu": fvu, "total_tokens": tokens}
                print(f"  {method} L{layer_idx}: FVU={fvu:.4f} ({tokens:,} tokens)")
            else:
                results[method][layer_idx] = None

    # Save as CSV
    out_path = out_dir / "fvu_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + METHODS)
        for layer_idx in range(N_LAYERS):
            row = [layer_idx]
            for method in METHODS:
                entry = results[method].get(layer_idx)
                row.append(f"{entry['fvu']:.6f}" if entry else "N/A")
            writer.writerow(row)

    print(f"\n[FVU] Saved table to {out_path}")

    # Summary
    print("\n=== FVU Summary ===")
    print(f"{'Layer':<8}", end="")
    for method in METHODS:
        print(f"{method:<15}", end="")
    print()
    for layer_idx in range(N_LAYERS):
        print(f"{layer_idx:<8}", end="")
        for method in METHODS:
            entry = results[method].get(layer_idx)
            if entry:
                print(f"{entry['fvu']:.4f}         ", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()

    volume.commit()
    return results


# ================================================================
#  Step 2: Cosine Similarity — 8 GPU workers
# ================================================================

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def cosine_worker(worker_id: int, layer_indices: list):
    """Compute cosine similarity between Gemma Scope base and finetuned W_dec."""
    import sys
    import torch
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_sae, _download_gemma_scope_params, _load_gemma_scope_weights

    out_dir = Path(RESULTS_BASE) / "analysis" / "cosines"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(RESULTS_BASE) / "run" / "checkpoints"

    print(f"[Cosine W{worker_id}] Layers: {layer_indices}")

    for layer_idx in tqdm(layer_indices, desc=f"W{worker_id} cosines"):
        # Load base Gemma Scope W_dec
        params_path = _download_gemma_scope_params(layer_idx, cache_dir="/vol/cache/huggingface")
        base_weights = _load_gemma_scope_weights(params_path)
        base_W_dec = base_weights["W_dec"]  # (d_sae, d_in)

        # Load finetuned W_dec (pretrained method only — random has no base to compare)
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"[Cosine W{worker_id}] SKIP L{layer_idx} — no checkpoint")
            continue

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        finetuned_W_dec = state["W_dec"]  # (d_sae, d_in)

        # Cosine similarity per feature (row-wise)
        base_norm = base_W_dec / base_W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        ft_norm = finetuned_W_dec / finetuned_W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cosines = (base_norm * ft_norm).sum(dim=1).numpy()  # (d_sae,)

        np.save(out_dir / f"cosines_layer_{layer_idx}.npy", cosines)
        print(f"[Cosine W{worker_id}] L{layer_idx}: mean={cosines.mean():.4f}, "
              f"min={cosines.min():.4f}, max={cosines.max():.4f}")

    volume.commit()
    return f"Cosine W{worker_id}: layers {layer_indices} done"


# ================================================================
#  Step 3: Visual Energy Ev — 8 GPU workers
# ================================================================

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def energy_worker(worker_id: int, layer_indices: list):
    """Compute per-feature visual energy Ev from cached activations."""
    import sys
    import gc
    import torch
    import numpy as np
    import h5py
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_sae

    out_dir = Path(RESULTS_BASE) / "analysis" / "energy"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(RESULTS_BASE) / "run" / "checkpoints"
    act_dir = Path(RESULTS_BASE) / "run" / "activations"

    # Use validation chunks (50k-55k) for energy computation
    val_chunks = []
    for ci in range(N_TRAINING_SAMPLES // CHUNK_SIZE, (N_TRAINING_SAMPLES + N_VAL_SAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE):
        cs = ci * CHUNK_SIZE
        ce = min(cs + CHUNK_SIZE, N_TRAINING_SAMPLES + N_VAL_SAMPLES)
        h5_path = act_dir / f"chunk_{cs}_{ce}.h5"
        if h5_path.exists():
            val_chunks.append((cs, ce, h5_path))
    print(f"[Energy W{worker_id}] Layers: {layer_indices}, val chunks: {len(val_chunks)}")

    for layer_idx in tqdm(layer_indices, desc=f"W{worker_id} energy layers"):
        # Load finetuned SAE (pretrained method)
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"[Energy W{worker_id}] SKIP L{layer_idx} — no checkpoint")
            continue

        sae = initialize_sae(layer_idx, checkpoint_path=str(ckpt_path),
                             device="cuda", cache_dir="/vol/cache/huggingface")
        sae.eval()

        # Accumulate squared feature activations for image and text tokens separately
        sum_sq_img = np.zeros(D_SAE, dtype=np.float64)
        sum_sq_txt = np.zeros(D_SAE, dtype=np.float64)
        n_img = 0
        n_txt = 0

        for cs, ce, h5_path in tqdm(val_chunks, desc=f"W{worker_id} L{layer_idx} chunks", leave=False):
            with h5py.File(h5_path, "r") as f:
                grp = f.get(f"layer_{layer_idx}")
                if grp is None:
                    continue

                for si in tqdm(range(cs, ce), desc=f"samples", leave=False):
                    key = f"sample_{si}"
                    if key not in grp:
                        continue

                    ds = grp[key]
                    act = torch.from_numpy(ds[:]).to("cuda")  # (seq, d_in)
                    img_s = int(ds.attrs.get("img_start", 0))
                    img_e = int(ds.attrs.get("img_end", 0))

                    # Encode through SAE
                    with torch.no_grad():
                        feature_acts = sae.encode(act.unsqueeze(0).float()).squeeze(0)  # (seq, d_sae)

                    fa_cpu = feature_acts.cpu().numpy().astype(np.float64)

                    # Image tokens
                    if img_e > img_s:
                        img_fa = fa_cpu[img_s:img_e]
                        sum_sq_img += (img_fa ** 2).sum(axis=0)
                        n_img += img_fa.shape[0]

                    # Text tokens (before and after image)
                    txt_parts = []
                    if img_s > 0:
                        txt_parts.append(fa_cpu[:img_s])
                    if img_e < fa_cpu.shape[0]:
                        txt_parts.append(fa_cpu[img_e:])
                    if txt_parts:
                        txt_fa = np.concatenate(txt_parts, axis=0)
                        sum_sq_txt += (txt_fa ** 2).sum(axis=0)
                        n_txt += txt_fa.shape[0]

        # Ev = image_energy / (image_energy + text_energy)
        total_energy = sum_sq_img + sum_sq_txt
        Ev = np.divide(sum_sq_img, total_energy, out=np.zeros_like(sum_sq_img),
                        where=total_energy > 1e-12)
        Et = np.divide(sum_sq_txt, total_energy, out=np.zeros_like(sum_sq_txt),
                        where=total_energy > 1e-12)

        np.save(out_dir / f"Ev_layer_{layer_idx}.npy", Ev.astype(np.float32))
        np.save(out_dir / f"Et_layer_{layer_idx}.npy", Et.astype(np.float32))

        print(f"[Energy W{worker_id}] L{layer_idx}: mean Ev={Ev.mean():.4f}, "
              f"n_img_tokens={n_img}, n_txt_tokens={n_txt}, "
              f"features with Ev>{EPSILON}: {(Ev > EPSILON).sum()}")

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    volume.commit()
    return f"Energy W{worker_id}: layers {layer_indices} done"


# ================================================================
#  Step 4: Select Adapted Features (no GPU)
# ================================================================

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def select_adapted_features():
    """Select adapted features: Ev > epsilon AND cosine in bottom percentile."""
    import numpy as np
    import csv
    import json
    from pathlib import Path
    from tqdm import tqdm

    analysis_dir = Path(RESULTS_BASE) / "analysis"
    cos_dir = analysis_dir / "cosines"
    ev_dir = analysis_dir / "energy"
    out_dir = analysis_dir / "adapted"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate all layers
    all_Ev = []
    all_cosines = []
    layer_indices = []

    for layer_idx in tqdm(range(N_LAYERS), desc="Loading Ev + cosines"):
        ev_path = ev_dir / f"Ev_layer_{layer_idx}.npy"
        cos_path = cos_dir / f"cosines_layer_{layer_idx}.npy"

        if not ev_path.exists() or not cos_path.exists():
            print(f"[WARN] Missing data for layer {layer_idx}")
            continue

        ev = np.load(ev_path)
        cos = np.load(cos_path)
        all_Ev.append(ev)
        all_cosines.append(cos)
        layer_indices.extend([layer_idx] * len(ev))

    global_Ev = np.concatenate(all_Ev)
    global_cosines = np.concatenate(all_cosines)
    layer_arr = np.array(layer_indices)

    # Save global arrays
    np.save(analysis_dir / "global_Ev.npy", global_Ev)
    np.save(analysis_dir / "global_cosines.npy", global_cosines)
    np.save(analysis_dir / "layer_indices.npy", layer_arr)

    # Select: Ev > epsilon AND cosine in bottom percentile
    ev_mask = global_Ev > EPSILON
    cos_threshold = float(np.percentile(global_cosines, COSINE_PERCENTILE))
    cos_mask = global_cosines <= cos_threshold
    adapted_mask = ev_mask & cos_mask
    adapted_global_indices = set(np.where(adapted_mask)[0].tolist())

    print(f"\n[Adapted] Ev > {EPSILON}: {ev_mask.sum()}")
    print(f"[Adapted] Cosine <= {cos_threshold:.4f} ({COSINE_PERCENTILE}th pctile): {cos_mask.sum()}")
    print(f"[Adapted] Intersection: {len(adapted_global_indices)}")

    # Group by layer
    results = []
    for layer_idx in range(N_LAYERS):
        layer_mask = layer_arr == layer_idx
        layer_adapted = adapted_mask & layer_mask
        feature_indices = np.where(layer_adapted)[0] - (layer_arr < layer_idx).sum()
        feature_indices = feature_indices.tolist()

        layer_ev = global_Ev[layer_mask]
        layer_cos = global_cosines[layer_mask]

        results.append({
            "layer": layer_idx,
            "n_adapted": len(feature_indices),
            "adapted_indices": feature_indices,
            "mean_cosine": float(layer_cos.mean()) if len(layer_cos) > 0 else 0,
            "mean_Ev": float(layer_ev.mean()) if len(layer_ev) > 0 else 0,
        })
        if feature_indices:
            print(f"  L{layer_idx}: {len(feature_indices)} adapted features")

    # Save CSV
    out_path = out_dir / "adapted_features_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "n_adapted", "adapted_indices", "mean_cosine", "mean_Ev"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "layer": r["layer"],
                "n_adapted": r["n_adapted"],
                "adapted_indices": str(r["adapted_indices"]),
                "mean_cosine": f"{r['mean_cosine']:.6f}",
                "mean_Ev": f"{r['mean_Ev']:.6f}",
            })

    # Save summary JSON
    summary = {
        "total_adapted": len(adapted_global_indices),
        "epsilon": EPSILON,
        "cosine_percentile": COSINE_PERCENTILE,
        "cosine_threshold": cos_threshold,
        "per_layer": {r["layer"]: r["n_adapted"] for r in results},
    }
    with open(out_dir / "adapted_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    print(f"\n[Adapted] Total: {len(adapted_global_indices)} features across {N_LAYERS} layers")
    return summary


# ================================================================
#  Step 5: Track Firing Frequencies — 8 GPU workers
# ================================================================

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def firing_worker(worker_id: int, layer_indices: list):
    """Track per-feature firing frequencies from cached activations."""
    import sys
    import gc
    import json
    import torch
    import numpy as np
    import h5py
    from pathlib import Path
    from tqdm import tqdm
    from datasets import load_dataset

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_sae

    out_dir = Path(RESULTS_BASE) / "analysis" / "firing"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(RESULTS_BASE) / "run" / "checkpoints"
    act_dir = Path(RESULTS_BASE) / "run" / "activations"

    # Identify spatial samples by filtering VQA questions
    print(f"[Firing W{worker_id}] Loading VQAv2 to identify spatial samples...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")

    spatial_indices = set()
    for i in tqdm(range(min(N_TRAINING_SAMPLES, len(vqa))), desc=f"W{worker_id} spatial filter"):
        q = vqa[i]["question"].lower()
        if any(kw in q for kw in SPATIAL_KEYWORDS):
            spatial_indices.add(i)
    print(f"[Firing W{worker_id}] {len(spatial_indices)} spatial samples out of {N_TRAINING_SAMPLES}")

    # Training chunks
    train_chunks = []
    for ci in range(N_TRAINING_SAMPLES // CHUNK_SIZE):
        cs = ci * CHUNK_SIZE
        ce = min(cs + CHUNK_SIZE, N_TRAINING_SAMPLES)
        h5_path = act_dir / f"chunk_{cs}_{ce}.h5"
        if h5_path.exists():
            train_chunks.append((cs, ce, h5_path))

    print(f"[Firing W{worker_id}] Layers: {layer_indices}, train chunks: {len(train_chunks)}")

    for layer_idx in tqdm(layer_indices, desc=f"W{worker_id} firing layers"):
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"[Firing W{worker_id}] SKIP L{layer_idx}")
            continue

        sae = initialize_sae(layer_idx, checkpoint_path=str(ckpt_path),
                             device="cuda", cache_dir="/vol/cache/huggingface")
        sae.eval()

        # Per-feature counts
        fire_count_all = np.zeros(D_SAE, dtype=np.int64)      # samples where feature fires (all VQA)
        fire_count_spatial = np.zeros(D_SAE, dtype=np.int64)   # samples where feature fires (spatial)
        fire_count_img_all = np.zeros(D_SAE, dtype=np.int64)   # image-token firings (all)
        fire_count_img_spatial = np.zeros(D_SAE, dtype=np.int64)
        n_all = 0
        n_spatial = 0

        for cs, ce, h5_path in tqdm(train_chunks, desc=f"W{worker_id} L{layer_idx} chunks", leave=False):
            with h5py.File(h5_path, "r") as f:
                grp = f.get(f"layer_{layer_idx}")
                if grp is None:
                    continue

                for si in range(cs, ce):
                    key = f"sample_{si}"
                    if key not in grp:
                        continue

                    ds = grp[key]
                    act = torch.from_numpy(ds[:]).to("cuda").float()  # (seq, d_in)
                    img_s = int(ds.attrs.get("img_start", 0))
                    img_e = int(ds.attrs.get("img_end", 0))

                    with torch.no_grad():
                        codes = sae.encode(act.unsqueeze(0)).squeeze(0)  # (seq, d_sae)

                    # Feature fired if any token has nonzero activation
                    fired = (codes != 0).any(dim=0).cpu().numpy()  # (d_sae,)
                    fire_count_all += fired.astype(np.int64)
                    n_all += 1

                    # Image-token firings
                    if img_e > img_s:
                        img_fired = (codes[img_s:img_e] != 0).any(dim=0).cpu().numpy()
                        fire_count_img_all += img_fired.astype(np.int64)

                    if si in spatial_indices:
                        fire_count_spatial += fired.astype(np.int64)
                        n_spatial += 1
                        if img_e > img_s:
                            img_fired_s = (codes[img_s:img_e] != 0).any(dim=0).cpu().numpy()
                            fire_count_img_spatial += img_fired_s.astype(np.int64)

        # Save per-layer results
        layer_data = {
            "layer": layer_idx,
            "n_all": int(n_all),
            "n_spatial": int(n_spatial),
            "fire_count_all": fire_count_all.tolist(),
            "fire_count_spatial": fire_count_spatial.tolist(),
            "fire_count_img_all": fire_count_img_all.tolist(),
            "fire_count_img_spatial": fire_count_img_spatial.tolist(),
        }

        out_path = out_dir / f"firing_layer_{layer_idx}.json"
        with open(out_path, "w") as f:
            json.dump(layer_data, f)

        print(f"[Firing W{worker_id}] L{layer_idx}: n_all={n_all}, n_spatial={n_spatial}, "
              f"features firing in >50% all: {(fire_count_all > n_all * 0.5).sum()}")

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    volume.commit()
    return f"Firing W{worker_id}: layers {layer_indices} done"


# ================================================================
#  Step 6: Identify Spatial Features (no GPU)
# ================================================================

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def find_spatial_features():
    """Fisher exact test + odds ratio to identify spatial features."""
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    firing_dir = Path(RESULTS_BASE) / "analysis" / "firing"
    out_dir = Path(RESULTS_BASE) / "analysis" / "spatial"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for layer_idx in tqdm(range(N_LAYERS), desc="Spatial features"):
        firing_path = firing_dir / f"firing_layer_{layer_idx}.json"
        if not firing_path.exists():
            continue

        with open(firing_path) as f:
            data = json.load(f)

        n_all = data["n_all"]
        n_spatial = data["n_spatial"]
        fire_all = np.array(data["fire_count_all"])
        fire_spatial = np.array(data["fire_count_spatial"])

        if n_all == 0 or n_spatial == 0:
            continue

        freq_all = fire_all / n_all
        freq_spatial = fire_spatial / n_spatial
        freq_diff = freq_spatial - freq_all

        # Only test features with positive frequency difference above threshold
        candidates = np.where(freq_diff > MIN_FREQ_DIFF)[0]
        if len(candidates) == 0:
            print(f"  L{layer_idx}: 0 candidates")
            continue

        p_values = []
        odds_ratios = []
        for fi in candidates:
            # 2x2 contingency table
            a = int(fire_spatial[fi])           # spatial + fired
            b = int(n_spatial - fire_spatial[fi])  # spatial + not fired
            c = int(fire_all[fi] - fire_spatial[fi])  # non-spatial + fired
            d = int((n_all - n_spatial) - c)       # non-spatial + not fired
            c = max(c, 0)
            d = max(d, 0)

            table = [[a, b], [c, d]]
            odds, p = fisher_exact(table, alternative="greater")
            p_values.append(p)
            odds_ratios.append(odds)

        # Multiple testing correction
        if len(p_values) > 0:
            reject, p_adj, _, _ = multipletests(p_values, alpha=P_THR, method="fdr_bh")
        else:
            p_adj = []
            reject = []

        n_sig = 0
        for i, fi in enumerate(candidates):
            if p_adj[i] < P_THR and odds_ratios[i] > ODDS_THR:
                all_results.append({
                    "layer": layer_idx,
                    "feature": int(fi),
                    "freq_all": float(freq_all[fi]),
                    "freq_spatial": float(freq_spatial[fi]),
                    "freq_diff": float(freq_diff[fi]),
                    "odds_ratio": float(odds_ratios[i]),
                    "p_adj": float(p_adj[i]),
                    "c_all": int(fire_all[fi]),
                    "c_spatial": int(fire_spatial[fi]),
                    "n_all": n_all,
                    "n_spatial": n_spatial,
                })
                n_sig += 1

        print(f"  L{layer_idx}: {len(candidates)} candidates, {n_sig} significant")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_dir / "spatial_features.csv", index=False)
        print(f"\n[Spatial] Total: {len(all_results)} features across {df['layer'].nunique()} layers")
    else:
        print("[Spatial] No significant spatial features found")

    # Save summary
    summary = {
        "total_spatial": len(all_results),
        "p_threshold": P_THR,
        "odds_threshold": ODDS_THR,
        "min_freq_diff": MIN_FREQ_DIFF,
        "per_layer": {},
    }
    if all_results:
        df = pd.DataFrame(all_results)
        for layer_idx in df["layer"].unique():
            summary["per_layer"][int(layer_idx)] = int((df["layer"] == layer_idx).sum())

    with open(out_dir / "spatial_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    return summary


# ================================================================
#  Step 7: Lexical Artifact Filtering — 8 GPU workers
# ================================================================

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def lexical_worker(worker_id: int, feature_assignments: list):
    """Test candidate features with generic prompts to filter lexical artifacts.

    feature_assignments: list of (layer_idx, feature_idx) tuples
    """
    import sys
    import json
    import torch
    from pathlib import Path
    from tqdm import tqdm
    from datasets import load_dataset
    from nnsight import NNsight

    sys.path.insert(0, "/root/paligemma2")
    from utils import (
        initialize_vlm_model, process_vlm_inputs,
        get_image_token_positions, initialize_sae,
    )

    out_dir = Path(RESULTS_BASE) / "analysis" / "lexical"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(RESULTS_BASE) / "run" / "checkpoints"

    print(f"[Lexical W{worker_id}] {len(feature_assignments)} features to test")

    if not feature_assignments:
        return f"Lexical W{worker_id}: no features assigned"

    # Load model once
    print(f"[Lexical W{worker_id}] Loading PaliGemma2...")
    processor, model_raw = initialize_vlm_model(MODEL_NAME, device="cpu")
    model_raw = model_raw.to("cuda")
    nns_model = NNsight(model_raw)

    # Load VQA dataset
    print(f"[Lexical W{worker_id}] Loading VQAv2...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")

    # Group features by layer to reuse SAEs
    from collections import defaultdict
    layer_features = defaultdict(list)
    for (layer_idx, feature_idx) in feature_assignments:
        layer_features[layer_idx].append(feature_idx)

    results = []
    samples_per_feature = 10
    activation_threshold = 0.01

    for layer_idx in tqdm(sorted(layer_features.keys()), desc=f"W{worker_id} layers"):
        features = layer_features[layer_idx]

        # Load SAE for this layer
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        sae = initialize_sae(layer_idx, checkpoint_path=str(ckpt_path),
                             device="cuda", cache_dir="/vol/cache/huggingface")
        sae.eval()

        # Load firing data to find top samples per feature
        firing_path = Path(RESULTS_BASE) / "analysis" / "firing" / f"firing_layer_{layer_idx}.json"
        if firing_path.exists():
            with open(firing_path) as f:
                firing_data = json.load(f)
            # Use spatial samples for testing
            spatial_sample_count = firing_data["n_spatial"]
        else:
            spatial_sample_count = 0

        for feature_idx in tqdm(features, desc=f"W{worker_id} L{layer_idx} features", leave=False):
            passed = True
            n_tested = 0

            # Test on random samples (first N from dataset)
            for si in range(min(samples_per_feature, len(vqa))):
                try:
                    sample = vqa[si]
                    image = sample["image"].convert("RGB")

                    # Test with generic prompts — check if feature fires on image tokens
                    best_generic_max = 0.0
                    for prompt in GENERIC_PROMPTS:
                        input_ids, attention_mask, pixel_values = process_vlm_inputs(
                            image, prompt, processor, model_raw, device="cuda"
                        )
                        img_start, img_end = get_image_token_positions(input_ids)

                        with torch.no_grad():
                            with nns_model.trace(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                            ) as tr:
                                layer_out = nns_model.model.language_model.layers[layer_idx].output
                                if isinstance(layer_out, tuple):
                                    layer_act = layer_out[0].detach().cpu().save()
                                else:
                                    layer_act = layer_out.detach().cpu().save()

                        act_tensor = layer_act
                        if isinstance(act_tensor, tuple):
                            act_tensor = act_tensor[0]
                        act_tensor = act_tensor.squeeze(0).float()  # (seq, d_in)

                        # Encode through SAE
                        with torch.no_grad():
                            codes = sae.encode(act_tensor.unsqueeze(0).to("cuda")).squeeze(0).cpu()

                        # Check feature activation on image tokens
                        if img_end > img_start:
                            img_act = codes[img_start:img_end, feature_idx]
                            max_act = float(img_act.max().item())
                            best_generic_max = max(best_generic_max, max_act)

                    if best_generic_max <= activation_threshold:
                        passed = False
                        break

                    n_tested += 1

                except Exception as e:
                    print(f"[Lexical W{worker_id}] Error L{layer_idx}/F{feature_idx} sample {si}: {e}")
                    continue

            results.append({
                "layer": layer_idx,
                "feature": feature_idx,
                "passed": passed,
                "n_tested": n_tested,
            })

            if passed:
                print(f"  L{layer_idx}/F{feature_idx}: PASSED")
            else:
                print(f"  L{layer_idx}/F{feature_idx}: FILTERED OUT")

        del sae
        torch.cuda.empty_cache()

    # Save results
    out_path = out_dir / f"lexical_results_w{worker_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()
    passed_count = sum(1 for r in results if r["passed"])
    return f"Lexical W{worker_id}: {passed_count}/{len(results)} passed"


# ================================================================
#  Step 8: Feature Intersection (no GPU)
# ================================================================

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def compute_intersection():
    """Compute adapted ∩ spatial ∩ lexical-filtered feature sets."""
    import json
    import ast
    import csv
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm

    analysis_dir = Path(RESULTS_BASE) / "analysis"
    out_dir = analysis_dir / "final_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load adapted features
    adapted_path = analysis_dir / "adapted" / "adapted_features_results.csv"
    adapted_by_layer = {}
    if adapted_path.exists():
        with open(adapted_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer"])
                indices = ast.literal_eval(row["adapted_indices"])
                adapted_by_layer[layer] = set(indices)
    print(f"[Intersection] Adapted: {sum(len(v) for v in adapted_by_layer.values())} features")

    # Load spatial features
    spatial_path = analysis_dir / "spatial" / "spatial_features.csv"
    spatial_by_layer = {}
    if spatial_path.exists():
        df = pd.read_csv(spatial_path)
        for layer in df["layer"].unique():
            spatial_by_layer[int(layer)] = set(df[df["layer"] == layer]["feature"].tolist())
    print(f"[Intersection] Spatial: {sum(len(v) for v in spatial_by_layer.values())} features")

    # Load lexical filtering results (merge all worker outputs)
    lexical_dir = analysis_dir / "lexical"
    lexical_passed = {}
    if lexical_dir.exists():
        for f_path in sorted(lexical_dir.glob("lexical_results_w*.json")):
            with open(f_path) as f:
                worker_results = json.load(f)
            for r in worker_results:
                if r["passed"]:
                    layer = r["layer"]
                    if layer not in lexical_passed:
                        lexical_passed[layer] = set()
                    lexical_passed[layer].add(r["feature"])
    print(f"[Intersection] Lexical passed: {sum(len(v) for v in lexical_passed.values())} features")

    # Compute intersection
    all_layers = sorted(set(adapted_by_layer.keys()) | set(spatial_by_layer.keys()))
    final_features = []

    for layer in tqdm(all_layers, desc="Intersection"):
        adapted = adapted_by_layer.get(layer, set())
        spatial = spatial_by_layer.get(layer, set())

        # First: adapted ∩ spatial
        common = adapted & spatial

        # Then filter by lexical (if lexical filtering was run)
        if lexical_passed:
            lex = lexical_passed.get(layer, set())
            common = common & lex

        for fi in sorted(common):
            final_features.append({"layer": layer, "feature": fi})

        if common:
            print(f"  L{layer}: adapted={len(adapted)}, spatial={len(spatial)}, "
                  f"lexical={len(lexical_passed.get(layer, set()))}, final={len(common)}")

    # Save results
    if final_features:
        df_final = pd.DataFrame(final_features)
        df_final.to_csv(out_dir / "final_spatial_visual_features.csv", index=False)

    summary = {
        "total_final": len(final_features),
        "total_adapted": sum(len(v) for v in adapted_by_layer.values()),
        "total_spatial": sum(len(v) for v in spatial_by_layer.values()),
        "total_lexical_passed": sum(len(v) for v in lexical_passed.values()),
        "per_layer": {},
    }
    for layer in all_layers:
        layer_features = [f for f in final_features if f["layer"] == layer]
        summary["per_layer"][layer] = len(layer_features)

    with open(out_dir / "intersection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    print(f"\n[Intersection] Final: {len(final_features)} features")
    return summary


# ================================================================
#  Entrypoint: Orchestrate all phases
# ================================================================

@app.local_entrypoint()
def main():
    import math

    # ========== Phase A: FVU table + Cosine similarities ==========
    print(f"\n{'='*60}")
    print("[Phase A] Step 1: FVU Table + Step 2: Cosine Similarities")
    print(f"{'='*60}")

    # Step 1: FVU table (no GPU)
    fvu_result = generate_fvu_table.remote()

    # Step 2: Cosine — distribute 26 layers across 8 GPUs
    layers_per_worker = math.ceil(N_LAYERS / N_GPUS)
    cosine_assignments = []
    for w in range(N_GPUS):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, N_LAYERS)
        worker_layers = list(range(start, end))
        if worker_layers:
            cosine_assignments.append((w, worker_layers))

    print(f"\n[Step 2] Cosine: {len(cosine_assignments)} workers")
    for w, layers in cosine_assignments:
        print(f"  GPU {w}: layers {layers}")

    cosine_results = list(cosine_worker.starmap(cosine_assignments))
    for r in cosine_results:
        print(r)

    # ========== Phase B: Visual Energy + Firing Frequencies ==========
    print(f"\n{'='*60}")
    print("[Phase B] Step 3: Visual Energy Ev + Step 5: Firing Frequencies")
    print(f"{'='*60}")

    # Same layer distribution for both
    energy_assignments = []
    firing_assignments = []
    for w in range(N_GPUS):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, N_LAYERS)
        worker_layers = list(range(start, end))
        if worker_layers:
            energy_assignments.append((w, worker_layers))
            firing_assignments.append((w, worker_layers))

    print(f"\n[Step 3] Energy Ev: {len(energy_assignments)} workers")
    print(f"[Step 5] Firing: {len(firing_assignments)} workers")

    # Run both in parallel (16 GPUs total)
    energy_futures = list(energy_worker.starmap(energy_assignments))
    for r in energy_futures:
        print(r)

    firing_futures = list(firing_worker.starmap(firing_assignments))
    for r in firing_futures:
        print(r)

    # ========== Phase C: Feature Selection (CPU) ==========
    print(f"\n{'='*60}")
    print("[Phase C] Step 4: Adapted + Step 6: Spatial + Step 8: Intersection")
    print(f"{'='*60}")

    # Step 4: Select adapted features
    adapted_summary = select_adapted_features.remote()
    print(f"Adapted: {adapted_summary}")

    # Step 6: Identify spatial features
    spatial_summary = find_spatial_features.remote()
    print(f"Spatial: {spatial_summary}")

    # Step 7: Lexical filtering — need candidate features first
    # Load adapted ∩ spatial intersection for lexical testing
    print(f"\n{'='*60}")
    print("[Phase D] Step 7: Lexical Artifact Filtering")
    print(f"{'='*60}")

    # We need to compute the intersection first to know which features to test
    # For now, run intersection without lexical filtering, then filter, then re-intersect
    pre_intersection = compute_intersection.remote()
    print(f"Pre-lexical intersection: {pre_intersection}")

    # Load candidate features for lexical testing
    # Read the pre-intersection results
    if pre_intersection and pre_intersection.get("total_final", 0) > 0:
        import json
        # We'll read the features from the volume in the lexical workers
        # For now, collect them from the summary
        # Need to read the CSV — but we're in local entrypoint
        # Instead, let's run a helper to get the feature list
        candidate_features = get_candidate_features.remote()

        if candidate_features:
            # Distribute across 8 GPUs
            n_features = len(candidate_features)
            features_per_worker = math.ceil(n_features / N_GPUS)
            lexical_assignments = []
            for w in range(N_GPUS):
                start = w * features_per_worker
                end = min(start + features_per_worker, n_features)
                worker_features = candidate_features[start:end]
                if worker_features:
                    lexical_assignments.append((w, worker_features))

            print(f"\n[Step 7] Lexical: {len(lexical_assignments)} workers, {n_features} features")

            lexical_results = list(lexical_worker.starmap(lexical_assignments))
            for r in lexical_results:
                print(r)

            # Re-compute intersection with lexical filtering
            final_summary = compute_intersection.remote()
            print(f"\nFinal intersection: {final_summary}")
        else:
            print("[WARN] No candidate features for lexical filtering")
    else:
        print("[WARN] No features in pre-lexical intersection, skipping lexical filtering")

    print(f"\n{'='*60}")
    print("[SUCCESS] Analysis pipeline complete!")
    print(f"  Results: {RESULTS_BASE}/analysis/")
    print(f"{'='*60}")


# Helper function to read candidate features from volume
@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=120,
)
def get_candidate_features():
    """Read pre-intersection features for lexical filtering."""
    import pandas as pd
    from pathlib import Path

    csv_path = Path(RESULTS_BASE) / "analysis" / "final_features" / "final_spatial_visual_features.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    features = [(int(row["layer"]), int(row["feature"])) for _, row in df.iterrows()]
    print(f"[INFO] {len(features)} candidate features for lexical filtering")
    return features
