"""
Compute per-feature firing frequencies on VSR dataset.

This is the CORRECT approach to identify spatial features:
compare VQA firing (already computed) vs VSR firing (this script).

The old (broken) approach compared VQA-all vs VQA-spatial-keyword-subset,
which only finds features correlating with spatial LANGUAGE, not spatial REASONING.

8 GPU workers, each handles ~3 layers. Each worker:
1. Loads PaliGemma2 + SAEs for assigned layers
2. Runs all VSR samples through model
3. Encodes activations through SAE, tracks per-feature firings
4. Saves firing stats in same format as VQA firing stats

Then a CPU step compares VQA vs VSR firing using Fisher exact test.

Usage:
    export HF_TOKEN=hf_...
    MODAL_PROFILE=hunar-oxford modal run modal_vsr_firing.py
"""

import os
import math
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("vlm-scope-vsr-firing")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers>=4.44", "sae-lens>=4.0",
        "nnsight>=0.3", "datasets", "h5py", "tqdm", "huggingface-hub",
        "Pillow", "numpy", "scipy", "statsmodels", "pandas",
        "accelerate", "requests",
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
N_GPUS = 8
RESULTS_BASE = "/vol/results/paligemma2"
MODEL_NAME = "google/paligemma2-3b-pt-224"
SAE_TYPE = "jumprelu"
BATCH_SIZE = 1  # PaliGemma2 with NNsight, process one at a time for reliability


# ================================================================
#  Step 1: Compute VSR firing frequencies (GPU workers)
# ================================================================

@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=14400,  # 4 hours
)
def vsr_firing_worker(worker_id: int, layer_indices: list):
    """Compute per-feature firing stats on VSR dataset for assigned layers."""
    import sys
    import gc
    import json
    import io
    import torch
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    from datasets import load_dataset, concatenate_datasets
    from nnsight import NNsight
    from PIL import Image
    import requests as req

    sys.path.insert(0, "/root/paligemma2")
    from utils import (
        process_vlm_inputs,
        get_image_token_positions, initialize_jumprelu_sae,
    )
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    out_dir = Path(RESULTS_BASE) / "analysis" / f"firing_vsr{sae_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(RESULTS_BASE) / f"run{sae_suffix}" / "checkpoints"

    print(f"[VSR-Firing W{worker_id}] Layers: {layer_indices}")

    # Load PaliGemma2 from cached volume (local_files_only to avoid gated repo issues)
    print(f"[VSR-Firing W{worker_id}] Loading PaliGemma2 from cache...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model_raw = model_raw.to("cuda")
    model_raw.eval()
    nns_model = NNsight(model_raw)

    # Load VSR dataset
    print(f"[VSR-Firing W{worker_id}] Loading VSR dataset...")
    vsr_splits = []
    for split in ["train", "validation", "test"]:
        try:
            ds = load_dataset("cambridgeltl/vsr_random", split=split)
            vsr_splits.append(ds)
        except Exception as e:
            print(f"  [WARN] Failed to load VSR {split}: {e}")
    vsr = concatenate_datasets(vsr_splits)
    print(f"[VSR-Firing W{worker_id}] VSR samples: {len(vsr)}")

    # Load SAEs for assigned layers
    saes = {}
    for layer_idx in layer_indices:
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"[VSR-Firing W{worker_id}] SKIP L{layer_idx} — no checkpoint")
            continue
        saes[layer_idx] = initialize_jumprelu_sae(
            layer_idx, checkpoint_path=str(ckpt_path),
            device="cuda", cache_dir="/vol/cache/huggingface"
        )
        saes[layer_idx].eval()
        print(f"[VSR-Firing W{worker_id}] Loaded SAE for L{layer_idx}")

    if not saes:
        return f"VSR-Firing W{worker_id}: no SAEs loaded"

    # Sample-level counters (for Fisher test comparison with VQA data)
    fire_count_all = {l: np.zeros(D_SAE, dtype=np.int64) for l in saes}  # samples where feature fires
    fire_count_img = {l: np.zeros(D_SAE, dtype=np.int64) for l in saes}  # samples where feature fires on img tokens
    # Token-level counters (for future use / richer analysis)
    token_firing_count = {l: np.zeros(D_SAE, dtype=np.int64) for l in saes}
    total_tokens = {l: 0 for l in saes}
    n_samples = 0
    n_failed = 0

    # Process VSR samples
    for si in tqdm(range(len(vsr)), desc=f"W{worker_id} VSR samples"):
        sample = vsr[si]
        caption = str(sample.get("caption", ""))

        # Load image from URL
        url = sample.get("image_link", "")
        if not url:
            n_failed += 1
            continue

        try:
            resp = req.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            if n_failed < 5:
                print(f"  [IMG FAIL] sample {si}: {type(e).__name__}: {e}")
            n_failed += 1
            continue

        # Process through model
        try:
            input_ids, attention_mask, pixel_values = process_vlm_inputs(
                img, caption, processor, model_raw, device="cuda"
            )
            img_start, img_end = get_image_token_positions(input_ids)
        except Exception as e:
            if n_failed < 5:
                print(f"  [VLM FAIL] sample {si}: {type(e).__name__}: {e}")
            n_failed += 1
            continue

        # Forward pass, extract activations at assigned layers
        try:
            with torch.no_grad():
                with nns_model.trace(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                ) as tr:
                    saved_acts = {}
                    for layer_idx in saes:
                        layer_out = nns_model.model.language_model.layers[layer_idx].output[0]
                        saved_acts[layer_idx] = layer_out[0].detach().cpu().save()

            # Encode through SAEs and track firings
            for layer_idx, sae in saes.items():
                act = saved_acts[layer_idx]
                if isinstance(act, tuple):
                    act = act[0]
                act = act.float().to("cuda")  # (seq, d_in)

                with torch.no_grad():
                    codes = sae.encode(act)  # (seq, d_sae)

                seq_len = codes.shape[0]
                total_tokens[layer_idx] += seq_len

                # Sample-level: feature fires if ANY token has nonzero activation
                # (matches VQA firing data format for consistent Fisher test)
                fired = (codes != 0).any(dim=0).cpu().numpy()  # (d_sae,)
                fire_count_all[layer_idx] += fired.astype(np.int64)

                # Token-level: count tokens per feature (saved for future analysis)
                token_firing_count[layer_idx] += (codes != 0).sum(dim=0).cpu().numpy().astype(np.int64)

                # Image token firings (sample-level)
                if img_end > img_start:
                    img_fired = (codes[img_start:img_end] != 0).any(dim=0).cpu().numpy()
                    fire_count_img[layer_idx] += img_fired.astype(np.int64)

            n_samples += 1

        except Exception as e:
            if si < 5:
                print(f"  [WARN] Sample {si}: {e}")
            n_failed += 1
            continue

        # Progress logging
        if (si + 1) % 500 == 0:
            print(f"[W{worker_id}] {si+1}/{len(vsr)} samples ({n_failed} failed)")
            volume.commit()

    # Save results per layer
    for layer_idx in saes:
        layer_data = {
            "layer": layer_idx,
            "n_samples": n_samples,
            "n_failed": n_failed,
            "total_tokens": int(total_tokens[layer_idx]),
            "dataset": "vsr",
            # Sample-level counts (for Fisher test — matches VQA format)
            "fire_count_all": fire_count_all[layer_idx].tolist(),
            "fire_count_img": fire_count_img[layer_idx].tolist(),
            # Token-level counts (for future richer analysis)
            "token_firing_count": token_firing_count[layer_idx].tolist(),
        }
        out_path = out_dir / f"firing_vsr_layer_{layer_idx}.json"
        with open(out_path, "w") as f:
            json.dump(layer_data, f)
        print(f"[W{worker_id}] Saved L{layer_idx}: {n_samples} samples, {total_tokens[layer_idx]} tokens, "
              f"features firing >50% samples: {(fire_count_all[layer_idx] > n_samples * 0.5).sum()}")

    volume.commit()
    return f"VSR-Firing W{worker_id}: {n_samples} samples, {n_failed} failed, layers {layer_indices}"


# ================================================================
#  Step 2: Compare VQA vs VSR firing — Fisher exact test (CPU)
# ================================================================

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=3600,  # 1 hour — Fisher test on 26*16384 features takes time
)
def find_spatial_features_vsr():
    """Compare VQA firing (existing) vs VSR firing (new) to find spatial features.

    This is the CORRECT approach — identical to the original LLaVA-MORE pipeline.
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

    # Load adapted features for intersection (done AFTER selection, not as pre-filter)
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
        print(f"[INFO] Loaded adapted features: {sum(len(v) for v in adapted_by_layer.values())} total")

    ODDS_THR = 2.0
    MIN_FREQ_DIFF = 0.005

    # ---- Collect all (layer, feature) rows across all layers (like old pipeline) ----
    all_rows = []

    for layer_idx in tqdm(range(N_LAYERS), desc="Collecting firing stats"):
        vqa_path = vqa_firing_dir / f"firing_layer_{layer_idx}.json"
        vsr_path = vsr_firing_dir / f"firing_vsr_layer_{layer_idx}.json"

        if not vqa_path.exists() or not vsr_path.exists():
            print(f"  L{layer_idx}: SKIP (missing data)")
            continue

        with open(vqa_path) as f:
            vqa_data = json.load(f)
        with open(vsr_path) as f:
            vsr_data = json.load(f)

        # Both use sample-level counts for consistent Fisher test
        n_vqa = vqa_data["n_all"]
        fire_vqa = np.array(vqa_data["fire_count_all"])

        n_vsr = vsr_data["n_samples"]
        fire_vsr = np.array(vsr_data["fire_count_all"])

        if n_vqa == 0 or n_vsr == 0:
            continue

        # Test ALL 16384 features — no pre-filtering to adapted
        for fi in range(D_SAE):
            c_vsr = int(fire_vsr[fi])
            c_vqa = int(fire_vqa[fi])
            all_rows.append({
                "layer": layer_idx,
                "feature": fi,
                "c_vqa": c_vqa,
                "n_vqa": n_vqa,
                "c_vsr": c_vsr,
                "n_vsr": n_vsr,
            })

        print(f"  L{layer_idx}: collected {D_SAE} features (n_vqa={n_vqa}, n_vsr_tokens={n_vsr})")

    if not all_rows:
        print("[WARN] No data collected!")
        volume.commit()
        return {"total_spatial": 0}

    df = pd.DataFrame(all_rows)
    print(f"\n[INFO] Total feature rows: {len(df)} across {df['layer'].nunique()} layers")

    # ---- Compute stats: freq, odds ratio, Fisher p-value (like old pipeline) ----
    df["freq_vqa"] = df["c_vqa"] / df["n_vqa"]
    df["freq_vsr"] = df["c_vsr"] / df["n_vsr"]
    df["freq_diff"] = df["freq_vsr"] - df["freq_vqa"]

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

    # ---- Global FDR correction across ALL features and ALL layers ----
    df["p_adj"] = multipletests(df["p_raw"].values, method="fdr_bh")[1]

    # ---- Filter: odds_ratio + freq_diff (matching old pipeline, not p_adj) ----
    spatial = df[(df["odds_ratio"] >= ODDS_THR) & (df["freq_diff"] >= MIN_FREQ_DIFF)].copy()
    spatial = spatial.sort_values("odds_ratio", ascending=False)

    print(f"\n[RESULT] {len(spatial)} spatial features (all, before adapted intersection) "
          f"across {spatial['layer'].nunique()} layers")

    # Save all spatial features (before adapted intersection)
    spatial.to_csv(out_dir / "spatial_features_vsr_all.csv", index=False)

    # ---- Intersect with adapted features ----
    if adapted_by_layer:
        adapted_mask = spatial.apply(
            lambda r: int(r["feature"]) in adapted_by_layer.get(int(r["layer"]), set()),
            axis=1
        )
        final = spatial[adapted_mask].copy()
        print(f"[RESULT] {len(final)} spatial+adapted features across {final['layer'].nunique()} layers")
    else:
        final = spatial
        print("[WARN] No adapted features found — using all spatial features")

    # Save final (adapted ∩ spatial)
    final_dir = Path(RESULTS_BASE) / "analysis" / f"final_features_vsr{sae_suffix}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(final_dir / "final_spatial_visual_features.csv", index=False)

    # Also save just spatial (before intersection) with full stats
    spatial.to_csv(out_dir / "spatial_features_vsr.csv", index=False)

    # Print top features
    print("\nTop 30 spatial features (by odds ratio, adapted ∩ spatial):")
    print(f"{'Layer':<8} {'Feature':<10} {'OR':>8} {'freq_vqa':>10} {'freq_vsr':>10} {'diff':>8} {'p_adj':>10}")
    for _, r in final.head(30).iterrows():
        print(f"  L{int(r['layer']):<5} F{int(r['feature']):<8} "
              f"{r['odds_ratio']:>8.2f} {r['freq_vqa']:>10.4f} {r['freq_vsr']:>10.4f} "
              f"{r['freq_diff']:>8.4f} {r['p_adj']:>10.2e}")

    summary = {
        "total_spatial_all": len(spatial),
        "total_spatial_adapted": len(final),
        "odds_threshold": ODDS_THR,
        "min_freq_diff": MIN_FREQ_DIFF,
        "fdr_method": "global_bh",
        "method": "VQA vs VSR token-level (matching old pipeline)",
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
    """Run VSR firing computation then spatial feature selection."""

    # Step 1: Distribute 26 layers across 8 GPUs
    layers_per_worker = math.ceil(N_LAYERS / N_GPUS)
    assignments = []
    for w in range(N_GPUS):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, N_LAYERS)
        worker_layers = list(range(start, end))
        if worker_layers:
            assignments.append((w, worker_layers))

    print(f"\n{'='*60}")
    print(f"[Step 1] VSR Firing Frequencies — {len(assignments)} GPU workers")
    print(f"{'='*60}")
    for w, layers in assignments:
        print(f"  GPU {w}: layers {layers}")

    results = list(vsr_firing_worker.starmap(assignments))
    for r in results:
        print(r)

    # Step 2: Compare VQA vs VSR firing
    print(f"\n{'='*60}")
    print(f"[Step 2] Spatial Feature Selection (VQA vs VSR)")
    print(f"{'='*60}")

    summary = find_spatial_features_vsr.remote()
    print(f"\nResult: {summary}")

    print(f"\n{'='*60}")
    print("[DONE] Spatial feature selection complete!")
    print(f"  Results: {RESULTS_BASE}/analysis/spatial_vsr_jumprelu/")
    print(f"{'='*60}")
