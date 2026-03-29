#!/usr/bin/env python3
"""
Reconstruction Gap Analysis: Measure how well VLM fine-tuned SAEs reconstruct 
activations from both text-only and vision-language data across all layers.

This script works with H5 activation files created by vqa/cache_activations.py.

Usage:
CUDA_VISIBLE_DEVICES=4 && python finetune/experiments/reconstruction_gap.py \
  --sae_dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
  --text_data /scratch/local/ssd/lachin/activations/validation_50k/text_chunk_50000_54096.h5 \
  --vlm_data /scratch/local/ssd/lachin/activations/validation_50k/vlm_chunk_50000_54096.h5 \
  --output_dir results/reconstruction_gap/vqa_val_text_only_50k

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from utils.sae_utils import load_sae_models


# ======================== ACTIVATION LOADING ========================

def load_activations_from_h5(h5_file_path: str, layer_idx: int) -> np.ndarray:
    """Load activations for a specific layer from H5 file created by cache_activations.py."""
    activations = []
    
    with h5py.File(h5_file_path, 'r') as f:
        layer_group_name = f'layer_{layer_idx}'
        if layer_group_name not in f:
            raise ValueError(f"Layer {layer_idx} not found in {h5_file_path}")
        
        layer_group = f[layer_group_name]
        
        sample_keys = sorted(layer_group.keys(), key=lambda x: int(x.split('_')[1]))
        
        for sample_key in sample_keys:
            sample_data = layer_group[sample_key][:]
            activations.append(sample_data)
    
    if activations:
        return np.concatenate(activations, axis=0)
    else:
        return np.array([])


# ======================== RECONSTRUCTION ANALYSIS ========================

def compute_reconstruction_mse(sae, activations: np.ndarray, device: str = "cuda", 
                             batch_size: int = 256, debug_layer: int = None) -> float:
    """Compute normalized MSE between activations and their SAE reconstructions."""
    sae.eval()
    total_nmse = 0.0
    total_tokens = 0
    
    # Keep activations on CPU, only move batches to GPU as needed
    acts_tensor = torch.tensor(activations, dtype=torch.float32, device="cpu")
    
    # Debug info for specific layer
    debug_info = None
    if debug_layer is not None:
        debug_info = {
            'activation_stats': {
                'mean': float(torch.mean(acts_tensor)),
                'std': float(torch.std(acts_tensor)),
                'min': float(torch.min(acts_tensor)),
                'max': float(torch.max(acts_tensor)),
                'shape': acts_tensor.shape
            }
        }
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(acts_tensor), batch_size):
            batch_cpu = acts_tensor[i:i+batch_size]
            # Move only this batch to GPU
            batch = batch_cpu.to(device)
            recon = sae(batch)
            
            # Compute normalized MSE: ||x - x̂||² / ||x||²
            # This gives us the fraction of variance not explained by the SAE
            batch_norm = torch.sum(batch**2, dim=-1)
            recon_error = torch.sum((batch - recon)**2, dim=-1)
            
            # Filter out tokens with near-zero norm to prevent inflated NMSE
            valid = batch_norm > 1e-5
            if valid.sum() > 0:
                recon_error = recon_error[valid]
                batch_norm = batch_norm[valid]
                nmse = torch.mean(recon_error / (batch_norm + 1e-8))
            else:
                nmse = torch.tensor(0.0, device=device)
            
            # Debug info for specific layer
            if debug_layer is not None and i == 0:  # Only for first batch
                debug_info['reconstruction_stats'] = {
                    'recon_mean': float(torch.mean(recon)),
                    'recon_std': float(torch.std(recon)),
                    'recon_min': float(torch.min(recon)),
                    'recon_max': float(torch.max(recon)),
                    'batch_norm_mean': float(torch.mean(batch_norm)),
                    'batch_norm_min': float(torch.min(batch_norm)),
                    'batch_norm_max': float(torch.max(batch_norm)),
                    'recon_error_mean': float(torch.mean(recon_error)),
                    'recon_error_max': float(torch.max(recon_error)),
                    'nmse': float(nmse),
                    'valid_tokens': int(valid.sum()),
                    'total_tokens': len(batch)
                }
            
            total_nmse += nmse.item() * len(batch)
            total_tokens += len(batch)
            
            # Clean up GPU memory
            del batch, recon, batch_norm, recon_error, nmse
            if device != "cpu":
                torch.cuda.empty_cache()
    
    final_nmse = total_nmse / total_tokens if total_tokens > 0 else float('nan')
    
    # Print debug info for specific layer
    if debug_layer is not None and debug_info is not None:
        print(f"\n=== DEBUG INFO FOR LAYER {debug_layer} ===")
        print(f"Activation stats: {debug_info['activation_stats']}")
        print(f"Reconstruction stats: {debug_info['reconstruction_stats']}")
        print(f"Final NMSE: {final_nmse}")
        print("=" * 50)
    
    return final_nmse


def analyze_layer_reconstruction(layer_idx: int, sae_dir: str, text_data_path: str, 
                               vlm_data_path: str, device: str = "cuda", batch_size: int = 256) -> Dict:
    """Analyze reconstruction performance for a single layer."""
    # Load both LM and VLM SAEs
    sae_lm, sae_vlm = load_sae_models(layer_idx, sae_dir)
    sae_lm.to(device).eval()
    sae_vlm.to(device).eval()
    
    # Load activations from H5 files
    text_acts = load_activations_from_h5(text_data_path, layer_idx)
    vlm_acts = load_activations_from_h5(vlm_data_path, layer_idx)
    
    # Debug layer 30 specifically
    debug_layer = 30 if layer_idx == 30 else None
    
    # Compute reconstruction MSE for all 4 combinations
    # VLM SAE reconstructions
    vlm_sae_on_text = compute_reconstruction_mse(sae_vlm, text_acts, device, batch_size)
    vlm_sae_on_vlm = compute_reconstruction_mse(sae_vlm, vlm_acts, device, batch_size)
    
    # LM SAE reconstructions  
    lm_sae_on_text = compute_reconstruction_mse(sae_lm, text_acts, device, batch_size, debug_layer)
    lm_sae_on_vlm = compute_reconstruction_mse(sae_lm, vlm_acts, device, batch_size)
    
    # Compute gaps
    vlm_gap = vlm_sae_on_text - vlm_sae_on_vlm  # VLM SAE: text vs vlm performance
    lm_gap = lm_sae_on_text - lm_sae_on_vlm     # LM SAE: text vs vlm performance
    text_gap = vlm_sae_on_text - lm_sae_on_text # Text data: VLM vs LM SAE performance
    vlm_data_gap = vlm_sae_on_vlm - lm_sae_on_vlm # VLM data: VLM vs LM SAE performance
    
    return {
        "layer": layer_idx,
        # VLM SAE results
        "vlm_sae_on_text": vlm_sae_on_text,
        "vlm_sae_on_vlm": vlm_sae_on_vlm,
        # LM SAE results
        "lm_sae_on_text": lm_sae_on_text, 
        "lm_sae_on_vlm": lm_sae_on_vlm,
        # Gaps
        "vlm_gap": vlm_gap,           # VLM SAE cross-domain gap
        "lm_gap": lm_gap,             # LM SAE cross-domain gap  
        "text_gap": text_gap,         # SAE comparison on text data
        "vlm_data_gap": vlm_data_gap, # SAE comparison on VLM data
        # Metadata
        "text_acts_shape": text_acts.shape,
        "vlm_acts_shape": vlm_acts.shape
    }


# ======================== PLOTTING FUNCTIONS ========================

def plot_reconstruction_analysis(results: List[Dict], out_dir: Path):
    """Create focused visualization of reconstruction results with improved styling."""
    # Figure style inspired by features/adapted/select_and_plot_adapted_features.py
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
    valid_results = [r for r in results if not np.isnan(r['vlm_sae_on_text']) and r['layer'] <= 30]
    if not valid_results:
        print("No valid data for plotting")
        return
    
    df = pd.DataFrame(valid_results)
    
    # Create plot 1: All 4 MSE combinations (log scale)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(df['layer'], df['vlm_sae_on_text'], color='#1f77b4', marker='o', label='VLM SAE on Text', linewidth=2.5, markersize=6)
    ax.plot(df['layer'], df['vlm_sae_on_vlm'], color='#d62728', marker='o', label='VLM SAE on VLM', linewidth=2.5, markersize=6)
    ax.plot(df['layer'], df['lm_sae_on_text'], color='#2ca02c', marker='s', label='LM SAE on Text', linewidth=2.5, markersize=6)
    ax.plot(df['layer'], df['lm_sae_on_vlm'], color='#9467bd', marker='s', label='LM SAE on VLM', linewidth=2.5, markersize=6)
    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Mean Squared Error (log scale)', fontsize=14, fontweight='bold')
    ax.set_title('SAE Reconstruction Error: All Combinations', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "reconstruction_analysis_full.png", bbox_inches='tight')
    plt.close()
    
    # Create plot 2: Two focused comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Direct MSE comparison (linear scale)
    ax1.plot(df['layer'], df['lm_sae_on_text'], color='#2ca02c', marker='s', label='LM SAE on Text', linewidth=2.5, markersize=6)
    ax1.plot(df['layer'], df['vlm_sae_on_text'], color='#1f77b4', marker='o', label='VLM SAE on Text', linewidth=2.5, markersize=6)
    ax1.plot(df['layer'], df['lm_sae_on_vlm'], color='#9467bd', marker='s', label='LM SAE on VLM', linewidth=2.5, markersize=6)
    ax1.plot(df['layer'], df['vlm_sae_on_vlm'], color='#d62728', marker='o', label='VLM SAE on VLM', linewidth=2.5, markersize=6)
    ax1.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Normalized Mean Squared Error', fontsize=13, fontweight='bold')
    ax1.set_title('SAE Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: VLM SAE cross-domain performance
    ax2.plot(df['layer'], df['vlm_sae_on_text'], color='#1f77b4', marker='o', label='VLM SAE on Text', linewidth=2.5, markersize=6)
    ax2.plot(df['layer'], df['vlm_sae_on_vlm'], color='#d62728', marker='o', label='VLM SAE on VLM', linewidth=2.5, markersize=6)
    ax2.fill_between(df['layer'], df['vlm_sae_on_text'], df['vlm_sae_on_vlm'], 
                     color='#1f77b4', alpha=0.12, label='Cross-domain Gap')
    ax2.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Normalized Mean Squared Error', fontsize=13, fontweight='bold')
    ax2.set_title('VLM SAE: Cross-Domain Performance', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "reconstruction_detailed_comparison.png", bbox_inches='tight')
    plt.close()


# ======================== SAVE FUNCTIONS ========================

def save_results(results: List[Dict], out_dir: Path):
    """Save results to CSV and JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "reconstruction_gap_results.csv", index=False)
    
    # Save as JSON
    with open(out_dir / "reconstruction_gap_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return df


# ======================== MAIN FUNCTION ========================

def main():
    parser = argparse.ArgumentParser(description="Reconstruction Gap Analysis")
    parser.add_argument("--sae_dir", required=True, help="Directory with VLM SAE checkpoints")
    parser.add_argument("--text_data", required=True, help="Path to text-only activations")
    parser.add_argument("--vlm_data", required=True, help="Path to VLM activations")
    parser.add_argument("--output_dir", default="reconstruction_gap_results", help="Output directory")
    parser.add_argument("--num_layers", type=int, default=32, help="Number of layers to analyze")
    parser.add_argument("--device", default="cuda", help="Device for computation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing activations")
    args = parser.parse_args()

    # Setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing reconstruction gap across {args.num_layers} layers")
    print(f"SAE directory: {args.sae_dir}")
    print(f"Text data: {args.text_data}")
    print(f"VLM data: {args.vlm_data}")
    print(f"Output directory: {out_dir}")
    
    # Process all layers
    results = []
    for layer_idx in tqdm(range(args.num_layers), desc="Processing layers"):
        result = analyze_layer_reconstruction(
            layer_idx, args.sae_dir, args.text_data, args.vlm_data, args.device, args.batch_size
        )
        results.append(result)
        
        # Print layer summary
        print(f"Layer {layer_idx:2d}: "
              f"VLM_SAE(Text)={result['vlm_sae_on_text']:.6f}, "
              f"VLM_SAE(VLM)={result['vlm_sae_on_vlm']:.6f}, "
              f"LM_SAE(Text)={result['lm_sae_on_text']:.6f}, "
              f"LM_SAE(VLM)={result['lm_sae_on_vlm']:.6f}")
    
    # Save results and create plots
    df = save_results(results, out_dir)
    plot_reconstruction_analysis(results, out_dir)
    
    # Print summary statistics
    valid_results = [r for r in results if not np.isnan(r['vlm_sae_on_text'])]
    if valid_results:
        # Average MSEs for all 4 combinations
        avg_vlm_sae_text = np.mean([r['vlm_sae_on_text'] for r in valid_results])
        avg_vlm_sae_vlm = np.mean([r['vlm_sae_on_vlm'] for r in valid_results])
        avg_lm_sae_text = np.mean([r['lm_sae_on_text'] for r in valid_results])
        avg_lm_sae_vlm = np.mean([r['lm_sae_on_vlm'] for r in valid_results])
        
        # Average gaps
        avg_vlm_gap = np.mean([r['vlm_gap'] for r in valid_results])
        avg_lm_gap = np.mean([r['lm_gap'] for r in valid_results])
        avg_text_gap = np.mean([r['text_gap'] for r in valid_results])
        avg_vlm_data_gap = np.mean([r['vlm_data_gap'] for r in valid_results])
        
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Processed {len(valid_results)} layers successfully")
        print(f"\nAverage MSE by SAE and Data Type:")
        print(f"  VLM SAE on Text: {avg_vlm_sae_text:.6f}")
        print(f"  VLM SAE on VLM:  {avg_vlm_sae_vlm:.6f}")
        print(f"  LM SAE on Text:  {avg_lm_sae_text:.6f}")
        print(f"  LM SAE on VLM:   {avg_lm_sae_vlm:.6f}")
        
        print(f"\nAverage Cross-Domain Gaps (Text MSE - VLM MSE):")
        print(f"  VLM SAE Gap: {avg_vlm_gap:.6f}")
        print(f"  LM SAE Gap:  {avg_lm_gap:.6f}")
        
        print(f"\nAverage SAE Comparison Gaps (VLM SAE - LM SAE):")
        print(f"  On Text Data: {avg_text_gap:.6f}")
        print(f"  On VLM Data:  {avg_vlm_data_gap:.6f}")
        
        print(f"\nCross-Domain Performance:")
        print(f"  VLM SAE better on text than VLM: {(np.array([r['vlm_gap'] for r in valid_results]) < 0).sum()} layers")
        print(f"  LM SAE better on text than VLM:  {(np.array([r['lm_gap'] for r in valid_results]) < 0).sum()} layers")
        
        print(f"\nSAE Comparison:")
        print(f"  VLM SAE better than LM SAE on text: {(np.array([r['text_gap'] for r in valid_results]) < 0).sum()} layers")
        print(f"  VLM SAE better than LM SAE on VLM:  {(np.array([r['vlm_data_gap'] for r in valid_results]) < 0).sum()} layers")
    
    print(f"\nAnalysis complete! Results saved to: {out_dir}")


if __name__ == "__main__":
    main() 