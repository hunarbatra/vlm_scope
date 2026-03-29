#!/usr/bin/env python3
"""
Compute per-feature metrics for adapted-feature analysis and save them for later selection/plotting.

This script computes, per SAE feature across all layers:
  - Raw energies per feature: E_v, E_t
  - Layer index for each global feature position

Masking behavior matches select_adapted_features_v2.py:
  - If --text_only_mask is provided, both text and VLM activations are filtered to text tokens only
  - Otherwise, both use their full sequences

Outputs (saved under --output_dir):
  - plotting_data_global_Ev.npy            # concatenated E_v per feature
  - plotting_data_global_Et.npy            # concatenated E_t per feature
  - plotting_data_layer_indices.npy        # per-global-index layer id
  - plotting_data_metadata.json            # stats + provenance

Usage (example):
  CUDA_VISIBLE_DEVICES=1 python features/adapted/compute_feature_metrics.py \
  --sae_dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
  --text_data /scratch/local/ssd/lachin/activations/validation_50k/text_chunk_50000_54096.h5 \
  --vlm_data /scratch/local/ssd/lachin/activations/validation_50k/vlm_chunk_50000_54096.h5 \
  --output_dir results/stage_2/metrics_run_no_mask \
  --num_layers 32 \
  --text_only_mask
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Project imports
import sys
ROOT_DIR = Path(__file__).parent.parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Also try to add current working directory to path in case script is run from project root
current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from utils.sae_utils import load_sae_models, load_activations_from_h5


def compute_energies(text_acts: np.ndarray, vlm_acts: np.ndarray,
                     sae, device: str = "cuda", batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Ev and Et per feature using streaming."""
    sae.eval()

    sum_sq_t: np.ndarray | None = None
    sum_sq_v: np.ndarray | None = None
    n_t = 0
    n_v = 0

    with torch.no_grad():
        # Text stream
        for i in range(0, len(text_acts), batch_size):
            batch_np = text_acts[i:i+batch_size]
            batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
            enc = sae.encode(batch).float().cpu().numpy()
            if sum_sq_t is None:
                d = enc.shape[1]
                sum_sq_t = np.zeros(d, dtype=np.float64)
            sum_sq_t += np.sum(enc * enc, axis=0)
            n_t += enc.shape[0]
            del batch
        torch.cuda.empty_cache()

        # VLM stream
        for i in range(0, len(vlm_acts), batch_size):
            batch_np = vlm_acts[i:i+batch_size]
            batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
            enc = sae.encode(batch).float().cpu().numpy()
            if sum_sq_v is None:
                d = enc.shape[1]
                sum_sq_v = np.zeros(d, dtype=np.float64)
            sum_sq_v += np.sum(enc * enc, axis=0)
            n_v += enc.shape[0]
            del batch
        torch.cuda.empty_cache()

    if n_t == 0 or n_v == 0:
        raise ValueError("Empty activations encountered while computing energies.")

    # Raw energies per feature
    Et = (sum_sq_t / float(n_t))
    Ev = (sum_sq_v / float(n_v))

    return Ev.astype(np.float32), Et.astype(np.float32)


def load_layer_metrics(layer_idx: int, sae_dir: str, text_data_path: str, vlm_data_path: str,
                       device: str = "cuda", text_only_mask: bool = False,
                       batch_size: int = 512) -> Dict[str, np.ndarray]:
    """Load per-layer activations and compute Ev and Et.

    Masking matches select_adapted_features_v2.py:
      - If text_only_mask is True, filter image tokens for BOTH text and VLM streams.
      - Otherwise, use full sequences for BOTH streams.
    """
    # Load SAE model (VLM side) for the layer
    _, sae_vlm = load_sae_models(layer_idx, sae_dir)
    sae_vlm.to(device).eval()

    # Load activations (symmetric masking controlled by text_only_mask)
    text_acts = load_activations_from_h5(text_data_path, layer_idx, filter_image_tokens=text_only_mask)
    vlm_acts = load_activations_from_h5(vlm_data_path, layer_idx, filter_image_tokens=text_only_mask)

    # Compute energies (streaming, memory-friendly)
    E_v, E_t = compute_energies(text_acts, vlm_acts, sae_vlm, device=device, batch_size=batch_size)

    return {
        "Et": E_t,
        "Ev": E_v,
    }


def collect_metrics(num_layers: int, sae_dir: str, text_data_path: str, vlm_data_path: str,
                    device: str = "cuda", text_only_mask: bool = False,
                    batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Collect concatenated Ev, Et, and per-feature layer indices across all layers."""
    all_Ev: List[np.ndarray] = []
    all_Et: List[np.ndarray] = []
    layer_indices: List[int] = []

    print("Collecting per-feature metrics from all layers...")
    for layer_idx in tqdm(range(num_layers), desc="Loading layer metrics"):
        try:
            data = load_layer_metrics(
                layer_idx, sae_dir, text_data_path, vlm_data_path,
                device, text_only_mask, batch_size
            )
            Ev = data["Ev"]
            Et = data["Et"]

            all_Ev.append(Ev)
            all_Et.append(Et)
            layer_indices.extend([layer_idx] * len(Ev))
        except Exception as e:
            print(f"Warning: Failed to load layer {layer_idx}: {e}")
            continue

    global_Ev = np.concatenate(all_Ev, axis=0)
    global_Et = np.concatenate(all_Et, axis=0)
    print(f"Collected {len(global_Ev)} total features across {len(set(layer_indices))} layers")
    return global_Ev, global_Et, layer_indices


def save_metrics(global_Ev: np.ndarray, global_Et: np.ndarray,
                 layer_indices: List[int], out_dir: Path):
    """Save arrays and metadata (energies only)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save energies
    np.save(out_dir / "plotting_data_global_Ev.npy", global_Ev)
    np.save(out_dir / "plotting_data_global_Et.npy", global_Et)
    np.save(out_dir / "plotting_data_layer_indices.npy", np.array(layer_indices, dtype=np.int32))

    metadata = {
        "total_features": int(len(global_Ev)),
        "Ev_stats": {
            "min": float(np.min(global_Ev)),
            "max": float(np.max(global_Ev)),
            "mean": float(np.mean(global_Ev)),
            "std": float(np.std(global_Ev)),
        },
        "Et_stats": {
            "min": float(np.min(global_Et)),
            "max": float(np.max(global_Et)),
            "mean": float(np.mean(global_Et)),
            "std": float(np.std(global_Et)),
        },
    }

    with open(out_dir / "plotting_data_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metrics to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute Ev and Et and save")
    parser.add_argument("--sae_dir", required=True, help="Directory with VLM SAE checkpoints")
    parser.add_argument("--text_data", required=True, help="Path to text-only activations H5 file")
    parser.add_argument("--vlm_data", required=True, help="Path to VLM activations H5 file")
    parser.add_argument("--output_dir", required=True, help="Output directory to save metrics")
    parser.add_argument("--num_layers", type=int, default=32, help="Number of layers to process")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument("--text_only_mask", action="store_true", help="Filter image tokens for BOTH streams (symmetric masking)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for encode()")
    args = parser.parse_args()

    print("Computing energies Ev and Et...")
    print(f"SAE dir: {args.sae_dir}")
    print(f"Text acts: {args.text_data}")
    print(f"VLM acts: {args.vlm_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Text-only mask: {args.text_only_mask}")

    global_Ev, global_Et, layer_indices = collect_metrics(
        num_layers=args.num_layers,
        sae_dir=args.sae_dir,
        text_data_path=args.text_data,
        vlm_data_path=args.vlm_data,
        device=args.device,
        text_only_mask=args.text_only_mask,
        batch_size=args.batch_size,
    )

    save_metrics(global_Ev, global_Et, layer_indices, Path(args.output_dir))


if __name__ == "__main__":
    main()


