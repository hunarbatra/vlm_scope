#!/usr/bin/env python3
"""
activation analysis script for SAE feature analysis using H5 files.

This script analyzes the activation patterns of rotated SAE features on image vs text tokens

The script requires geometry divergence results from compute_geometry_divergence.py for analysis and runs
over all 32 layers by default.

Usage:
CUDA_VISIBLE_DEVICES=0 python finetune/experiments/image_text_ratio.py \
    --data_path /scratch/local/ssd/lachin/activations/validation_50k/vlm_chunk_50000_54096.h5 \
    --geometry_dir results/cosines/text-only_50k \
    --vlm_ckpt_dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
    --output_dir results/stage_1/image_text_ratio/text-only_v2_95 \
    --cosine_threshold 0.95 \
    --max_samples 100
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import json
from typing import List, Dict, Optional, Tuple
from scipy import stats

import h5py

# Add project root to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

# Import SAE-related modules directly
try:
    from sae_lens import SAE, SAEConfig
except ImportError:
    print("Warning: sae_lens not available. SAE functionality will be limited.")
    SAE = None
    SAEConfig = None

# Simple AnalysisConfig class to replace the missing one
class AnalysisConfig:
    def __init__(self, debug=False, verbose=True, max_samples=50):
        self.debug = debug
        self.verbose = verbose
        self.max_samples = max_samples
    
    def log(self, message, level="INFO"):
        if self.verbose or level in ["ERROR", "WARN"]:
            print(f"[{level}] {message}")

# Function to load SAE models using sae_lens directly
def load_sae_models(layer_idx, vlm_ckpt_dir):
    """Load SAE models using sae_lens directly"""
    if SAE is None or SAEConfig is None:
        raise ImportError("sae_lens is required but not available")
    
    # Try to load from the checkpoint directory
    vlm_checkpoint_path = Path(vlm_ckpt_dir) / f"layer_{layer_idx}" / "sae_weights.safetensors"
    
    if not vlm_checkpoint_path.exists():
        # Try alternative path structure
        vlm_checkpoint_path = Path(vlm_ckpt_dir) / f"l{layer_idx}r_8x" / "sae_weights.safetensors"
    
    if not vlm_checkpoint_path.exists():
        # Try to load from pretrained
        try:
            sae, cfg, sparsity = SAE.from_pretrained(
                release="llama_scope_lxr_8x",
                sae_id=f"l{layer_idx}r_8x",
                device="cpu"
            )
            
            # Create topk config
            topk_cfg = dict(cfg)
            del topk_cfg['architecture']
            del topk_cfg['jump_relu_threshold']
            del topk_cfg['neuronpedia_id']
            del topk_cfg['activation_fn_str']
            
            new_topk_cfg = SAEConfig(
                architecture="topk",
                activation_fn_kwargs={"k": 50},
                activation_fn_str='topk',
                **topk_cfg
            )
            
            sae_vlm = SAE(new_topk_cfg).to("cpu")
            
            # Load checkpoint if available
            if vlm_checkpoint_path.exists():
                sae_vlm.load_state_dict(torch.load(vlm_checkpoint_path, weights_only=True))
            else:
                # Use original weights
                og_weights = sae.state_dict().copy()
                del og_weights['threshold']
                sae_vlm.load_state_dict(og_weights)
            
            del sae  # Clean up
            
        except Exception as e:
            print(f"Warning: Could not load SAE for layer {layer_idx}: {e}")
            return None, None
    else:
        # Load from checkpoint
        try:
            sae, cfg, sparsity = SAE.from_pretrained(
                release="llama_scope_lxr_8x",
                sae_id=f"l{layer_idx}r_8x",
                device="cpu"
            )
            
            # Create topk config
            topk_cfg = dict(cfg)
            del topk_cfg['architecture']
            del topk_cfg['jump_relu_threshold']
            del topk_cfg['neuronpedia_id']
            del topk_cfg['activation_fn_str']
            
            new_topk_cfg = SAEConfig(
                architecture="topk",
                activation_fn_kwargs={"k": 50},
                activation_fn_str='topk',
                **topk_cfg
            )
            
            sae_vlm = SAE(new_topk_cfg).to("cpu")
            sae_vlm.load_state_dict(torch.load(vlm_checkpoint_path, weights_only=True))
            del sae  # Clean up
            
        except Exception as e:
            print(f"Warning: Could not load SAE for layer {layer_idx}: {e}")
            return None, None
    
    # Return None for sae_llm since the script doesn't seem to use it much
    return None, sae_vlm


def load_h5_activations_with_positions(h5_file_path: str, layer_idx: int, max_samples: Optional[int] = None) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Load activations and image token positions from H5 file.
    
    Args:
        h5_file_path: Path to H5 file
        layer_idx: Layer index to load
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (activations_list, image_positions_list)
    """
    activations = []
    image_positions = []
    
    with h5py.File(h5_file_path, 'r') as f:
        layer_group_name = f'layer_{layer_idx}'
        if layer_group_name not in f:
            raise ValueError(f"Layer {layer_idx} not found in {h5_file_path}")
        
        layer_group = f[layer_group_name]
        sample_keys = sorted(layer_group.keys(), key=lambda x: int(x.split('_')[1]))
        
        if max_samples:
            sample_keys = sample_keys[:max_samples]
        
        for sample_key in sample_keys:
            sample_data = layer_group[sample_key][:]
            activations.append(sample_data)
            
            # Get image token positions from metadata
            img_start = layer_group[sample_key].attrs.get('img_start', -1)
            img_end = layer_group[sample_key].attrs.get('img_end', -1)
            image_positions.append((img_start, img_end))
    
    return activations, image_positions


def get_token_masks(seq_len: int, img_start: int, img_end: int) -> Tuple[List[int], List[int]]:
    """
    Create token position masks for image and text tokens.
    
    Args:
        seq_len: Total sequence length
        img_start: Start position of image tokens
        img_end: End position of image tokens
        
    Returns:
        Tuple of (text_positions, image_positions)
    """
    # Validate positions
    if img_start < 0 or img_end < 0 or img_start >= img_end:
        # If no valid image positions, treat all as text
        return list(range(seq_len)), []
    
    # Text tokens are everything except image tokens
    text_positions = list(range(0, img_start)) + list(range(img_end, seq_len))
    image_positions = list(range(img_start, min(img_end, seq_len)))
    
    return text_positions, image_positions


def build_token_metadata(data_path: str, layer_idx: int, max_samples: int = None) -> Tuple[np.ndarray, List[Tuple[int, int, str]]]:
    """Build token-level activations with metadata for mapping back to samples."""
    token_acts = []
    metadata = []  # (sample_idx, local_token_idx, token_type)
    
    with h5py.File(data_path, 'r') as f:
        layer_group = f[f'layer_{layer_idx}']
        sample_keys = sorted(layer_group.keys(), key=lambda x: int(x.split('_')[1]))
        
        if max_samples is not None:
            sample_keys = sample_keys[:max_samples]
        
        for sample_key in sample_keys:
            sample_idx = int(sample_key.split('_')[1])
            sample_data = layer_group[sample_key][:]
            
            img_start = layer_group[sample_key].attrs.get('img_start', -1)
            img_end = layer_group[sample_key].attrs.get('img_end', -1)
            
            for token_idx in range(sample_data.shape[0]):
                token_acts.append(sample_data[token_idx])
                
                if img_start != -1 and img_end != -1 and img_start <= token_idx < img_end:
                    token_type = "image"
                else:
                    token_type = "text"
                
                metadata.append((sample_idx, token_idx, token_type))
    
    return np.array(token_acts), metadata


def load_rotated_features(layer_idx: int, geometry_dir: str, cosine_threshold: float = 0.8) -> List[int]:
    """
    Load rotated feature indices for a given layer from geometry divergence results.
    
    Args:
        layer_idx: Layer index
        geometry_dir: Directory containing geometry divergence results
        cosine_threshold: Threshold for identifying rotated features (default: 0.8)
        
    Returns:
        List of rotated feature indices
    """
    # Try decoder cosines first (more commonly used)
    decoder_cosines_file = Path(geometry_dir) / f"layer_{layer_idx:02d}_decoder_cosines.npy"
    encoder_cosines_file = Path(geometry_dir) / f"layer_{layer_idx:02d}_encoder_cosines.npy"
    
    if decoder_cosines_file.exists():
        cosines = np.load(decoder_cosines_file)
        print(f"Using decoder cosines for layer {layer_idx}")
    elif encoder_cosines_file.exists():
        cosines = np.load(encoder_cosines_file)
        print(f"Using encoder cosines for layer {layer_idx}")
    else:
        raise FileNotFoundError(f"No cosine similarity files found for layer {layer_idx} in {geometry_dir}")
    
    # Find features with cosine similarity below threshold (rotated features)
    rotated_indices = np.where(cosines < cosine_threshold)[0].tolist()
    
    print(f"Found {len(rotated_indices)} rotated features (cosine < {cosine_threshold}) out of {len(cosines)} total features")
    
    return rotated_indices


def analyze_activation_patterns(h5_file_path: str, layer_idx: int, sae_llm, sae_vlm, config: AnalysisConfig, geometry_dir: str, cosine_threshold: float = 0.8):
    """Analyze activation patterns of rotated SAE features on image vs text tokens."""
    
    # Load rotated features
    try:
        rotated_indices = load_rotated_features(layer_idx, geometry_dir, cosine_threshold)
        if len(rotated_indices) == 0:
            config.log("No rotated features found.", "WARN")
            return None
    except Exception as e:
        config.log(f"Error loading rotated features: {e}", "ERROR")
        return None
    
    # Load and process activations
    activations, image_positions = load_h5_activations_with_positions(h5_file_path, layer_idx, config.max_samples)
    if not activations:
        config.log("No activations loaded.", "WARN")
        return None
    
    # Setup SAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sae_vlm = sae_vlm.to(device, dtype=torch.float32).eval()
    except Exception as e:
        config.log(f"GPU failed, using CPU: {e}", "WARN")
        device = "cpu"
        sae_vlm = sae_vlm.to(device, dtype=torch.float32).eval()
    
    # Process samples
    all_image_acts = []
    all_text_acts = []
    
    for sample_idx, (layer_acts, (img_start, img_end)) in enumerate(zip(activations, image_positions)):
        if layer_acts.shape[0] == 0:
            continue
        
        # Encode through SAE
        with torch.no_grad():
            acts_tensor = torch.tensor(layer_acts, dtype=torch.float32)
            batch = acts_tensor.to(device, dtype=torch.float32)
            try:
                feature_acts = sae_vlm.encode(batch)
                feature_acts = feature_acts.cpu().numpy()
            except Exception as e:
                # Fallback to CPU
                batch_cpu = acts_tensor.to("cpu", dtype=torch.float32)
                sae_cpu = sae_vlm.to("cpu", dtype=torch.float32)
                feature_acts = sae_cpu.encode(batch_cpu)
                feature_acts = feature_acts.numpy()
                sae_vlm = sae_vlm.to(device, dtype=torch.float32)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Separate tokens
        seq_len = feature_acts.shape[0]
        text_positions, image_positions_sample = get_token_masks(seq_len, img_start, img_end)
        
        if text_positions:
            all_text_acts.append(feature_acts[text_positions, :])
        if image_positions_sample:
            all_image_acts.append(feature_acts[image_positions_sample, :])
    
    if len(all_image_acts) == 0 or len(all_text_acts) == 0:
        config.log("No valid activations found.", "WARN")
        return None
    
    # Analyze results
    image_acts_flat = np.concatenate(all_image_acts, axis=0)
    text_acts_flat = np.concatenate(all_text_acts, axis=0)
    
    # Filter rotated features
    max_idx = min(image_acts_flat.shape[1], text_acts_flat.shape[1]) - 1
    rotated_indices_safe = [idx for idx in rotated_indices if idx <= max_idx]
    
    if len(rotated_indices_safe) == 0:
        config.log("No valid rotated indices.", "WARN")
        return None
    
    # Calculate statistics
    rotated_image_acts = image_acts_flat[:, rotated_indices_safe]
    rotated_text_acts = text_acts_flat[:, rotated_indices_safe]
    
    image_mean = np.mean(np.abs(rotated_image_acts))
    text_mean = np.mean(np.abs(rotated_text_acts))
    ratio = image_mean / text_mean if text_mean > 0 else float('inf')
    
    # Log results
    config.log(f"Layer {layer_idx}: Ratio={ratio:.3f}, {len(rotated_indices_safe)} features")
    if ratio > 1.2:
        config.log("SUPPORT: Image preference")
    elif ratio < 0.8:
        config.log("CONTRADICTION: Text preference")
    else:
        config.log("NEUTRAL")
    
    return {
        'layer_idx': layer_idx,
        'rotated_image_activation': image_mean,
        'rotated_text_activation': text_mean,
        'ratio_image_text': ratio,
        'num_rotated_features': len(rotated_indices_safe),
        'total_features': len(rotated_indices)
    }


def create_cross_layer_comparison_plots(all_results: List[Dict], output_dir: str = ".", output_filename: str = "cross_layer_analysis.png"):
    """Create cross-layer plots with improved styling; save only ratio and count figures."""
    if not all_results:
        print("No results to plot")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    
    # Extract data for plotting
    layers = [r['layer_idx'] for r in all_results]
    ratios = [r['ratio_image_text'] for r in all_results]
    num_rotated = [r['num_rotated_features'] for r in all_results]
    # rotated_ratios kept for completeness but not plotted
    rotated_ratios = [r['num_rotated_features'] / r['total_features'] * 100 for r in all_results]
    
    # Derive filenames from provided base name
    base = output_filename.replace('.png', '')
    ratio_file = output_path / f"{base}_ratio.png"
    count_file = output_path / f"{base}_count.png"

    # Plot 1: Image/Text Activation Ratio across layers
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1.plot(layers, ratios, color='#1f77b4', marker='o', linewidth=2.5, markersize=6, label='Image/Text Ratio')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Activation (1.0)')
    ax1.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Image/Text Activation Ratio', fontsize=14, fontweight='bold')
    ax1.set_title('Rotated Features: Image vs Text Activation Ratio', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    plt.tight_layout()
    plt.savefig(ratio_file, bbox_inches='tight')
    plt.close(fig1)

    # Plot 3: Number of rotated features (skip percentage plot)
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    ax3.plot(layers, num_rotated, color='#7E6BFF', marker='o', linewidth=2.5, markersize=6, label='Rotated Feature Count')
    ax3.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Rotated Features', fontsize=14, fontweight='bold')
    ax3.set_title('Count of Rotated Features per Layer', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    plt.tight_layout()
    plt.savefig(count_file, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"Saved plots: {ratio_file} and {count_file}")
    
    # Save the data
    summary_data = {
        'layers': [int(x) for x in layers],
        'ratios': [float(x) for x in ratios],
        'num_rotated': [int(x) for x in num_rotated],
        'rotated_ratios': [float(x) for x in rotated_ratios]
    }
    
    data_file = output_path / f"{base}_data.json"
    with open(data_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary data saved as {data_file}")


def analyze_multiple_layers(h5_file_path: str, layers: List[int], config: AnalysisConfig, output_dir: str = ".", vlm_ckpt_dir: str = "checkpoints_15000", geometry_dir: str = None, cosine_threshold: float = 0.8):
    """Analyze multiple layers and create comparison plots."""
    if not Path(h5_file_path).exists():
        raise FileNotFoundError(f"H5 file not found: {h5_file_path}")
    
    all_results = []
    
    for layer_idx in layers:
        try:
            sae_llm, sae_vlm = load_sae_models(layer_idx, vlm_ckpt_dir)
            results = analyze_activation_patterns(h5_file_path, layer_idx, sae_llm, sae_vlm, config, geometry_dir, cosine_threshold)
            
            if results:
                all_results.append(results)
                
        except Exception as e:
            config.log(f"ERROR Layer {layer_idx}: {e}", "ERROR")
            continue
    
    # Create plots and summary
    if all_results:
        create_cross_layer_comparison_plots(all_results, output_dir)
        
        # Summary
        ratios = [r['ratio_image_text'] for r in all_results]
        
        config.log(f"\nSUMMARY: {len(all_results)} layers analyzed")
        config.log(f"  Mean ratio: {np.mean(ratios):.2f}")
        config.log(f"  Image preference: {sum(1 for r in ratios if r > 1.2)}")
        config.log(f"  Text preference: {sum(1 for r in ratios if r < 0.8)}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE feature activations on image vs text tokens")
    parser.add_argument("--data_path", type=str, required=True, help="H5 file with cached activations")
    parser.add_argument("--output_dir", type=str, default="activation_analysis_results", help="Output directory")
    parser.add_argument("--vlm_ckpt_dir", type=str, default="checkpoints_15000", help="VLM SAE checkpoints")
    parser.add_argument("--geometry_dir", type=str, required=True, help="Geometry divergence results")
    parser.add_argument("--cosine_threshold", type=float, default=0.8, help="Cosine threshold for rotated features")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples to analyze")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    args = parser.parse_args()
    
    # Validate inputs
    if not args.data_path.endswith('.h5'):
        print(f"Error: Expected H5 file, got: {args.data_path}")
        return
    
    if not Path(args.data_path).exists():
        print(f"Error: H5 file not found: {args.data_path}")
        return
    
    if not Path(args.geometry_dir).exists():
        print(f"Error: Geometry divergence directory not found: {args.geometry_dir}")
        return
    
    # Setup and run
    config = AnalysisConfig(debug=args.debug, verbose=not args.quiet, max_samples=args.max_samples)
    layers = list(range(32))
    
    try:
        analyze_multiple_layers(args.data_path, layers, config, args.output_dir, args.vlm_ckpt_dir, args.geometry_dir, args.cosine_threshold)
    except Exception as e:
        config.log(f"ERROR Analysis failed: {e}", "ERROR")


if __name__ == "__main__":
    main() 