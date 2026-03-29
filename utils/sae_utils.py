#!/usr/bin/env python3
"""
Shared utilities for SAE analysis across different scripts.


"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple
from sae_lens import SAE
from sae_lens.sae import SAEConfig
import h5py


def load_sae_models(layer_idx: int, vlm_ckpt_dir: str = "checkpoints_15000") -> Tuple[SAE, SAE]:
    """
    Load SAE models for a given layer.
    
    Args:
        layer_idx: Layer index to load
        vlm_ckpt_dir: Directory containing VLM checkpoints
        
    Returns:
        Tuple of (sae_llm, sae_vlm)
    """
    # Load LLM SAE
    sae_llm, cfg, _ = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{layer_idx}r_8x",
        device="cpu",
    )
    
    # Try multiple checkpoint locations and formats
    possible_checkpoints = [
        Path(vlm_ckpt_dir) / f"layer{layer_idx}.pt",
        Path(vlm_ckpt_dir) / f"pretrained_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir) / f"text-only_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir) / f"image-only_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir) / f"random_layer_{layer_idx}.pt",
        # Also check subdirectories
        Path(vlm_ckpt_dir) / "pretrained" / f"pretrained_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir) / "text-only" / f"text-only_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir) / "image-only" / f"image-only_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir) / "random" / f"random_layer_{layer_idx}.pt",
        # Also check if vlm_ckpt_dir is already a subdirectory
        Path(vlm_ckpt_dir).parent / "pretrained" / f"pretrained_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir).parent / "text-only" / f"text-only_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir).parent / "image-only" / f"image-only_layer_{layer_idx}.pt",
        Path(vlm_ckpt_dir).parent / "random" / f"random_layer_{layer_idx}.pt",
    ]
    
    vlm_checkpoint = None
    for checkpoint_path in possible_checkpoints:
        if checkpoint_path.exists():
            vlm_checkpoint = checkpoint_path
            break
    
    if vlm_checkpoint is None:
        raise FileNotFoundError(f"VLM checkpoint not found for layer {layer_idx}")
    
    # Create VLM SAE configuration
    cfg_clean = dict(cfg)
    for key in ['architecture', 'jump_relu_threshold', 'neuronpedia_id', 'activation_fn_str', 'activation_fn_kwargs']:
        cfg_clean.pop(key, None)
    
    new_topk_cfg = SAEConfig(
        architecture="topk",
        activation_fn_kwargs={"k": 50},
        activation_fn_str="topk",
        **cfg_clean
    )
    sae_vlm = SAE(new_topk_cfg)
    
    # Load the checkpoint
    checkpoint = torch.load(vlm_checkpoint, map_location="cpu")
    sae_vlm.load_state_dict(checkpoint)
    
    return sae_llm, sae_vlm


def load_activations_from_h5(h5_file_path: str, layer_idx: int, filter_image_tokens: bool = False) -> np.ndarray:
    """
    Load activations for a specific layer from H5 file created by cache_activations.py.
    
    Args:
        h5_file_path: Path to H5 file
        layer_idx: Layer index to load
        filter_image_tokens: If True, exclude image tokens from VLM data
        
    Returns:
        Concatenated activations for the specified layer
    """
    activations = []
    
    with h5py.File(h5_file_path, 'r') as f:
        layer_group_name = f'layer_{layer_idx}'
        if layer_group_name not in f:
            raise ValueError(f"Layer {layer_idx} not found in {h5_file_path}")
        
        layer_group = f[layer_group_name]
        sample_keys = sorted(layer_group.keys(), key=lambda x: int(x.split('_')[1]))
        
        for sample_key in sample_keys:
            sample_data = layer_group[sample_key][:]
            
            # Filter out image tokens if requested and available
            if filter_image_tokens:
                img_start = layer_group[sample_key].attrs.get('img_start', -1)
                img_end = layer_group[sample_key].attrs.get('img_end', -1)
                
                if img_start != -1 and img_end != -1 and img_start < img_end:
                    # Extract only language tokens (before and after image tokens)
                    lang_tokens_before = sample_data[:img_start]
                    lang_tokens_after = sample_data[img_end:]
                    sample_data = np.concatenate([lang_tokens_before, lang_tokens_after], axis=0)
            
            activations.append(sample_data)
    
    if activations:
        return np.concatenate(activations, axis=0)
    else:
        return np.array([])


def load_activations(data_path: str, layer_idx: int) -> np.ndarray:
    """
    Load activations for a specific layer from pickle file.
    
    Args:
        data_path: Path to pickle file containing activations
        layer_idx: Layer index to extract
        
    Returns:
        Concatenated activations for the specified layer
    """
    with open(data_path, 'rb') as f:
        all_samples = pickle.load(f)
    
    # Extract layer activations and concatenate
    layer_acts = []
    for sample in all_samples:
        arr = sample[layer_idx]
        if arr.ndim == 3:  # (1, T, D)
            arr = arr.squeeze(0)
        layer_acts.append(arr)
    
    return np.concatenate(layer_acts, axis=0) 