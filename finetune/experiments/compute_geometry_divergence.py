#!/usr/bin/env python3
"""
Compute cosine similarity between LLM and VLM SAE features to analyze geometry divergence.

Usage:
# Single checkpoint analysis
python experiments/compute_geometry_divergence.py \
  --vlm_ckpt_dir /scratch/local/ssd/lachin/checkpoints_50k/pretrained \
  --num_layers 32 \
  --out_dir results/cosines/pretrained_50k

# Multi-checkpoint comparison across all layers
python experiments/compute_geometry_divergence.py \
  --vlm_ckpt_dir /scratch/local/ssd/lachin/checkpoints_50k/pretrained \
  --multi_checkpoint \
  --checkpoint_dirs \
    "pretrained:/scratch/local/ssd/lachin/checkpoints_50k/pretrained" \
    "image-only:/scratch/local/ssd/lachin/checkpoints_50k/image-only" \
    "text-only:/scratch/local/ssd/lachin/checkpoints_50k/text-only" \
    "random:/scratch/local/ssd/lachin/checkpoints_50k/random" \
  --num_layers 32 \
  --out_dir results/stage_1/cosines/multi_checkpoint_comparison
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

from sae_lens import SAE
from utils.sae_utils import load_sae_models


def get_encoder_weights(sae: SAE) -> torch.Tensor:
    """Extract encoder weights as (d_model, d_sae) tensor."""
    if hasattr(sae, "W_enc"):
        return sae.W_enc.detach().cpu().float()
    else:
        return sae.encoder.weight.T.detach().cpu().float()


def get_decoder_weights(sae: SAE) -> torch.Tensor:
    """Extract decoder weights as (d_sae, d_model) tensor."""
    if hasattr(sae, "W_dec"):
        return sae.W_dec.detach().cpu().float()
    else:
        return sae.decoder.weight.T.detach().cpu().float()


# ======================== SIMILARITY METRICS ========================

def compute_cosine_similarity(w1: torch.Tensor, w2: torch.Tensor) -> np.ndarray:
    """Compute feature-wise cosine similarity between weight matrices."""
    cosines = torch.nn.functional.cosine_similarity(w1, w2, dim=1)
    return cosines.numpy()


def compute_weight_changes(w1: torch.Tensor, w2: torch.Tensor) -> np.ndarray:
    """Compute relative L2 norm of weight changes per feature."""
    delta_abs = (w2 - w1).norm(dim=1)
    delta_rel = delta_abs / (w1.norm(dim=1) + 1e-9)
    return delta_rel.numpy()


# ======================== ANALYSIS FUNCTIONS ========================

def analyze_layer(layer_idx: int, vlm_ckpt_dir: str, cosine_threshold: float = 0.8) -> dict:
    """Analyze geometry divergence for a single layer."""
    sae_llm, sae_vlm = load_sae_models(layer_idx, vlm_ckpt_dir)
    
    W_enc_llm = get_encoder_weights(sae_llm)
    W_dec_llm = get_decoder_weights(sae_llm)
    W_enc_vlm = get_encoder_weights(sae_vlm)
    W_dec_vlm = get_decoder_weights(sae_vlm)
    
    cos_encoder = compute_cosine_similarity(W_enc_llm.T, W_enc_vlm.T)
    cos_decoder = compute_cosine_similarity(W_dec_llm, W_dec_vlm)
    
    delta_encoder = compute_weight_changes(W_enc_llm.T, W_enc_vlm.T)
    delta_decoder = compute_weight_changes(W_dec_llm, W_dec_vlm)
    
    return {
        'layer': layer_idx,
        'encoder': {
            'cosines': cos_encoder,
            'deltas': delta_encoder,
            'mean_cosine': float(np.mean(cos_encoder)),
            'std_cosine': float(np.std(cos_encoder)),
            'mean_delta': float(np.mean(delta_encoder)),
            'low_cosine_count': int(np.sum(cos_encoder < cosine_threshold))
        },
        'decoder': {
            'cosines': cos_decoder,
            'deltas': delta_decoder,
            'mean_cosine': float(np.mean(cos_decoder)),
            'std_cosine': float(np.std(cos_decoder)),
            'mean_delta': float(np.mean(delta_decoder)),
            'low_cosine_count': int(np.sum(cos_decoder < cosine_threshold))
        }
    }


def analyze_multiple_checkpoints(layer_idx: int, checkpoint_dirs: dict, cosine_threshold: float = 0.8) -> dict:
    """Analyze geometry divergence for multiple checkpoint types for a single layer."""
    results = {}
    
    sae_llm, _ = load_sae_models(layer_idx, list(checkpoint_dirs.values())[0])
    W_dec_llm = get_decoder_weights(sae_llm)
    
    for name, ckpt_dir in checkpoint_dirs.items():
        print(f"Processing {name} checkpoint...")
        
        _, sae_vlm = load_sae_models(layer_idx, ckpt_dir)
        W_dec_vlm = get_decoder_weights(sae_vlm)
        
        cos_decoder = compute_cosine_similarity(W_dec_llm, W_dec_vlm)
        delta_decoder = compute_weight_changes(W_dec_llm, W_dec_vlm)
        
        results[name] = {
            'layer': layer_idx,
            'decoder': {
                'cosines': cos_decoder,
                'deltas': delta_decoder,
                'mean_cosine': float(np.mean(cos_decoder)),
                'std_cosine': float(np.std(cos_decoder)),
                'mean_delta': float(np.mean(delta_decoder)),
                'low_cosine_count': int(np.sum(cos_decoder < cosine_threshold))
            }
        }
    
    return results


# ======================== PLOTTING FUNCTIONS ========================

def plot_layer_comparison(results: dict, out_dir: Path):
    """Plot encoder vs decoder cosine similarity for a single layer."""
    layer_idx = results['layer']
    cos_encoder = results['encoder']['cosines']
    cos_decoder = results['decoder']['cosines']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Use the same color palette as FVU curves
    colors = {'encoder': '#7F58AF', 'decoder': '#64C5EB'}
    
    ax1.hist(cos_encoder, bins=50, alpha=0.7, color=colors['encoder'], edgecolor='black', linewidth=0.5)
    ax1.axvline(np.mean(cos_encoder), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(cos_encoder):.3f}')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Layer {layer_idx}: Encoder Similarity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    ax2.hist(cos_decoder, bins=50, alpha=0.7, color=colors['decoder'], edgecolor='black', linewidth=0.5)
    ax2.axvline(np.mean(cos_decoder), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(cos_decoder):.3f}')
    ax2.set_xlabel('Cosine Similarity', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Layer {layer_idx}: Decoder Similarity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Set style for publication quality
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"layer_{layer_idx:02d}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_multi_checkpoint_comparison(checkpoint_results: dict, out_dir: Path, layer_idx: int = 0):
    """Plot decoder similarities across multiple checkpoint types for comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use the same color palette as FVU curves
    colors = {'full': '#7F58AF', 'random': '#64C5EB', 'image': '#E84D8A', 'text': '#FEB326'}
    
    # Filter out 'base' and map names to match FVU curves
    name_mapping = {'pretrained': 'full', 'image-only': 'image', 'text-only': 'text', 'random': 'random'}
    filtered_results = {name_mapping.get(k, k): v for k, v in checkpoint_results.items() if k != 'base'}
    
    for name, results in filtered_results.items():
        cos_decoder = results['decoder']['cosines']
        color = colors.get(name, '#666666')
        
        ax.hist(cos_decoder, bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5,
                label=f'{name.capitalize()} (mean: {np.mean(cos_decoder):.3f})')
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Layer {layer_idx}: Decoder Similarity Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Set style for publication quality
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"layer_{layer_idx:02d}_multi_checkpoint_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_multi_checkpoint_trends(all_multi_results: list, out_dir: Path, checkpoint_names: list):
    """Plot trends across all layers for multiple checkpoint types."""
    layers = list(range(len(all_multi_results)))
    
    # Filter out 'base' and map names to match FVU curves
    name_mapping = {'pretrained': 'full', 'image-only': 'image', 'text-only': 'text', 'random': 'random'}
    filtered_names = [name_mapping.get(name, name) for name in checkpoint_names if name != 'base']
    
    checkpoint_means = {name: [] for name in filtered_names}
    
    for layer_results in all_multi_results:
        for name in checkpoint_names:
            if name != 'base':
                mapped_name = name_mapping.get(name, name)
                mean_cos = layer_results[name]['decoder']['mean_cosine']
                checkpoint_means[mapped_name].append(mean_cos)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Use the same color palette as FVU curves
    colors = {'full': '#7F58AF', 'random': '#64C5EB', 'image': '#E84D8A', 'text': '#FEB326'}
    
    for name in filtered_names:
        color = colors.get(name, '#666666')
        ax.plot(layers, checkpoint_means[name], 'o-', label=name.capitalize(), color=color, linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Decoder Cosine Similarity', fontsize=12)
    ax.set_title('Decoder Similarity Trends Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Set style for publication quality
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / "multi_checkpoint_trends.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_layer_trends(all_results: list, out_dir: Path, cosine_threshold: float = 0.8):
    """Plot trends across all layers."""
    layers = [r['layer'] for r in all_results]
    
    enc_mean_cos = [r['encoder']['mean_cosine'] for r in all_results]
    dec_mean_cos = [r['decoder']['mean_cosine'] for r in all_results]
    enc_low_cos = [r['encoder']['low_cosine_count'] for r in all_results]
    dec_low_cos = [r['decoder']['low_cosine_count'] for r in all_results]
    enc_mean_delta = [r['encoder']['mean_delta'] for r in all_results]
    dec_mean_delta = [r['decoder']['mean_delta'] for r in all_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Use the same color palette as FVU curves
    colors = {'encoder': '#7F58AF', 'decoder': '#64C5EB'}
    
    ax1.plot(layers, enc_mean_cos, 'o-', label='Encoder', color=colors['encoder'], linewidth=2.5, markersize=6)
    ax1.plot(layers, dec_mean_cos, 'o-', label='Decoder', color=colors['decoder'], linewidth=2.5, markersize=6)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Mean Cosine Similarity', fontsize=12)
    ax1.set_title('Mean Cosine Similarity Across Layers', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    ax2.plot(layers, enc_low_cos, 'o-', label='Encoder', color=colors['encoder'], linewidth=2.5, markersize=6)
    ax2.plot(layers, dec_low_cos, 'o-', label='Decoder', color=colors['decoder'], linewidth=2.5, markersize=6)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel(f'Features with Cosine < {cosine_threshold}', fontsize=12)
    ax2.set_title('Low Similarity Features Across Layers', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(layers, enc_mean_delta, 'o-', label='Encoder', color=colors['encoder'], linewidth=2.5, markersize=6)
    ax3.plot(layers, dec_mean_delta, 'o-', label='Decoder', color=colors['decoder'], linewidth=2.5, markersize=6)
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Mean Weight Change (Relative)', fontsize=12)
    ax3.set_title('Weight Change Magnitude Across Layers (Relative)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    import seaborn as sns
    import pandas as pd
    
    all_cosines = []
    layer_labels = []
    for result in all_results:
        cosines = result['decoder']['cosines']
        if len(cosines) > 1000:
            cosines = np.random.choice(cosines, 1000, replace=False)
        all_cosines.extend(cosines)
        layer_labels.extend([f"L{result['layer']}"] * len(cosines))
    
    df = pd.DataFrame({'Layer': layer_labels, 'Cosine Similarity': all_cosines})
    sns.violinplot(data=df, x='Layer', y='Cosine Similarity', ax=ax4, color=colors['decoder'])
    ax4.set_title('Decoder Cosine Distribution by Layer', fontsize=14, fontweight='bold')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Set style for publication quality
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / "trends_across_layers.png", dpi=300, bbox_inches='tight')
    plt.close()


# ======================== SAVE/LOAD FUNCTIONS ========================

def save_layer_results(results: dict, out_dir: Path):
    """Save all results for a single layer."""
    layer_idx = results['layer']
    
    np.save(out_dir / f"layer_{layer_idx:02d}_encoder_cosines.npy", results['encoder']['cosines'])
    np.save(out_dir / f"layer_{layer_idx:02d}_decoder_cosines.npy", results['decoder']['cosines'])
    np.save(out_dir / f"layer_{layer_idx:02d}_encoder_deltas.npy", results['encoder']['deltas'])
    np.save(out_dir / f"layer_{layer_idx:02d}_decoder_deltas.npy", results['decoder']['deltas'])
    
    np.savetxt(out_dir / f"layer_{layer_idx:02d}_encoder_cosines.csv", results['encoder']['cosines'])
    np.savetxt(out_dir / f"layer_{layer_idx:02d}_decoder_cosines.csv", results['decoder']['cosines'])


def save_summary(all_results: list, out_dir: Path):
    """Save comprehensive summary of all results."""
    summary = {
        'total_layers': len(all_results),
        'layers_processed': [r['layer'] for r in all_results],
        'encoder_summary': {
            'mean_cosine_per_layer': [r['encoder']['mean_cosine'] for r in all_results],
            'std_cosine_per_layer': [r['encoder']['std_cosine'] for r in all_results],
            'low_cosine_count_per_layer': [r['encoder']['low_cosine_count'] for r in all_results],
            'mean_delta_per_layer': [r['encoder']['mean_delta'] for r in all_results]
        },
        'decoder_summary': {
            'mean_cosine_per_layer': [r['decoder']['mean_cosine'] for r in all_results],
            'std_cosine_per_layer': [r['decoder']['std_cosine'] for r in all_results],
            'low_cosine_count_per_layer': [r['decoder']['low_cosine_count'] for r in all_results],
            'mean_delta_per_layer': [r['decoder']['mean_delta'] for r in all_results]
        }
    }
    
    with open(out_dir / 'geometry_divergence_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)



def main():
    parser = argparse.ArgumentParser(description="Compute geometry divergence between LLM and VLM SAEs")
    parser.add_argument("--vlm_ckpt_dir", required=True, help="Directory with VLM checkpoint files")
    parser.add_argument("--num_layers", type=int, default=32, help="Number of layers to analyze")
    parser.add_argument("--out_dir", default="geometry_results", help="Output directory")
    parser.add_argument("--cosine_threshold", type=float, default=0.8, help="Cosine similarity threshold for low similarity features")
    parser.add_argument("--multi_checkpoint", action="store_true", help="Compare multiple checkpoint types")
    parser.add_argument("--checkpoint_dirs", nargs="+", help="Additional checkpoint directories for comparison (format: name:path)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing geometry divergence across {args.num_layers} layers")
    print(f"VLM checkpoints: {args.vlm_ckpt_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Cosine threshold: {args.cosine_threshold}")
    
    if args.multi_checkpoint and args.checkpoint_dirs:
        checkpoint_dirs = {"base": args.vlm_ckpt_dir}
        for ckpt_spec in args.checkpoint_dirs:
            name, path = ckpt_spec.split(":", 1)
            checkpoint_dirs[name] = path
        
        print(f"Comparing {len(checkpoint_dirs)} checkpoint types: {list(checkpoint_dirs.keys())}")
        
        all_multi_results = []
        for layer_idx in tqdm(range(args.num_layers), desc="Processing layers"):
            multi_results = analyze_multiple_checkpoints(layer_idx, checkpoint_dirs, args.cosine_threshold)
            all_multi_results.append(multi_results)
            
            print(f"Layer {layer_idx:2d}: ", end="")
            for name, results in multi_results.items():
                dec = results['decoder']
                print(f"{name}[cos={dec['mean_cosine']:.3f}] ", end="")
            print()
        
        plot_multi_checkpoint_trends(all_multi_results, out_dir, list(checkpoint_dirs.keys()))
        
        print(f"\nMulti-checkpoint analysis complete! Results saved to: {out_dir}")
        print(f"Processed {len(all_multi_results)} layers for {len(checkpoint_dirs)} checkpoint types")
        return
    
    all_results = []
    for layer_idx in tqdm(range(args.num_layers), desc="Processing layers"):
        results = analyze_layer(layer_idx, args.vlm_ckpt_dir, args.cosine_threshold)
        all_results.append(results)
        
        save_layer_results(results, out_dir)
        plot_layer_comparison(results, out_dir)
        
        enc = results['encoder']
        dec = results['decoder']
        print(f"Layer {layer_idx:2d}: "
              f"Enc[cos={enc['mean_cosine']:.3f}, low={enc['low_cosine_count']:4d}] "
              f"Dec[cos={dec['mean_cosine']:.3f}, low={dec['low_cosine_count']:4d}]")
    
    plot_layer_trends(all_results, out_dir, args.cosine_threshold)
    save_summary(all_results, out_dir)
    
    print(f"\nAnalysis complete! Results saved to: {out_dir}")
    print(f"Processed {len(all_results)} layers successfully")


if __name__ == "__main__":
    main() 