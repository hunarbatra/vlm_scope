#!/usr/bin/env python3
"""
Attention visualization for spatial features using organized feature folders.

This script processes all organized spatial feature folders and creates attention visualizations
using the DLA summary and feature samples from each folder.

Example:
CUDA_VISIBLE_DEVICES=2 python features/spatial/attn_viz_spatial_features.py \
  --organized-features-dir results/organized_spatial_features \
  --top-k 5 \
  --bottom-k 5 \
  --method both \
  --feature-filter layer_26_feature_807 \
  --query-mode avg_answer \
  --rope-aware \
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import math
import torch
from nnsight import NNsight
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import seaborn as sns
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import sys
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    get_image_token_positions,
)
from datasets import load_dataset


def load_dataset_vqa():
    """Load the VQAv2 validation split dataset object (lazy loading)."""
    return load_dataset("lmms-lab/VQAv2", split="validation")


def get_vqa_samples_from_feature(feature_samples_path: Path) -> List[int]:
    """Extract VQA indices from feature samples JSON file."""
    try:
        with open(feature_samples_path, 'r') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        vqa_indices = []
        
        for sample in samples:
            vqa_idx = sample.get('vqa_idx')
            if vqa_idx is not None:
                vqa_indices.append(vqa_idx)
        
        return vqa_indices
    except Exception as e:
        print(f"[WARN] Failed to load feature samples from {feature_samples_path}: {e}")
        return []


def get_vqa_samples_from_organized_folder(feature_dir: Path) -> List[int]:
    """Get VQA samples from an organized feature folder using the same method as attribution_patching.py."""
    # Extract layer and feature from directory name
    layer = None
    feature = None
    dir_name = feature_dir.name
    if dir_name.startswith("layer_") and "_feature_" in dir_name:
        parts = dir_name.split("_")
        try:
            layer = int(parts[1])
            feature = int(parts[3])
        except (IndexError, ValueError):
            pass
    
    if layer is None or feature is None:
        return []
    
    # Load top-K VQA sample indices from pre-extracted feature samples (JSON files)
    # Same logic as attribution_patching.py
    vqa_feature_data_dir = "results/stage_4/feature_samples/vqa_spatial_all_spatial"
    feature_file = Path(vqa_feature_data_dir) / f"text-only_layer_{layer}_feature_{feature}" / "sample_info.json"
    
    if not feature_file.exists():
        print(f"[WARN] Feature file not found: {feature_file}")
        return []
    
    try:
        with open(feature_file, "r") as f:
            samples = json.load(f)
        
        if not samples:
            print(f"[WARN] No samples found in {feature_file}")
            return []
        
        # Extract VQA indices from the top samples by magnitude
        # Samples are already sorted by magnitude (rank field)
        # Use the same number as attribution_patching.py (100 by default)
        top_k = 10
        top_indices = [sample["sample_idx"] for sample in samples[:top_k] if "sample_idx" in sample]
        
        if not top_indices:
            print(f"[WARN] No VQA indices found in top {len(samples)} samples from {feature_file}")
            return []
        
        print(f"[INFO] Feature {feature} (layer {layer}) has {len(samples)} total samples")
        print(f"[INFO] Using top {len(top_indices)} samples by magnitude from {feature_file}")
        
        # Map spatial subset indices to base VQA indices using cached mapping
        # Same logic as attribution_patching.py
        spatial_map = load_vqa_spatial_indices()
        if spatial_map is None:
            print("[WARN] Spatial VQA indices cache not found. Assuming sample_idx are base VQAv2 indices.")
            return top_indices
        else:
            base_indices = []
            for sidx in top_indices:
                if sidx < 0 or sidx >= len(spatial_map):
                    print(f"[WARN] Spatial index {sidx} out of range {len(spatial_map)}; skipping")
                    continue
                base_indices.append(int(spatial_map[sidx]))
            
            if len(base_indices) == 0:
                print("[WARN] After mapping spatial indices, no valid base VQAv2 indices remained")
                return []
            
            print(f"[INFO] Mapped {len(top_indices)} spatial indices to {len(base_indices)} base VQAv2 indices")
            return base_indices
            
    except Exception as e:
        print(f"[ERROR] Failed to load/parse feature data: {e}")
        return []


def load_vqa_spatial_indices(cache_dir: str = ".cache/vqa_spatial_filter") -> List[int]:
    """Load cached mapping from spatial subset indices to base VQA indices.
    
    Same logic as attribution_patching.py
    """
    candidates = []
    try:
        # Fallback to search in cache_dir and default known dir
        search_dirs = []
        if cache_dir:
            search_dirs.append(Path(cache_dir))
        search_dirs.append(Path(".cache/vqa_spatial_filter"))
        
        for d in search_dirs:
            if d.exists():
                for f in sorted(d.glob("indices_validation_*.json")):
                    candidates.append(f)
        
        for f in candidates:
            try:
                payload = json.loads(Path(f).read_text())
                indices = payload.get("indices") or payload.get("filtered_indices")
                if indices and isinstance(indices, list):
                    return [int(x) for x in indices]
            except Exception:
                continue
    except Exception:
        pass
    return []


def get_head_lists(summary: Dict, args, method_key: str) -> Tuple[List, List]:
    """Get top-k and bottom-k heads from the DLA summary."""
    top_heads = summary.get("top_heads", [])
    bottom_heads = summary.get("bottom_heads", [])
    
    if not top_heads or not bottom_heads:
        # Compute head lists from head scores if available
        head_scores = summary.get(method_key)
        if head_scores:
            # Find all head scores across all layers
            all_head_scores = []
            for layer_idx, layer_scores in enumerate(head_scores):
                for head_idx, score in enumerate(layer_scores):
                    all_head_scores.append((layer_idx, head_idx, score))
            
            # Sort by score
            all_head_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Get top-k and bottom-k
            top_heads = [(layer, head) for layer, head, score in all_head_scores[:args.top_k]]
            bottom_heads = [(layer, head) for layer, head, score in all_head_scores[-args.bottom_k:]]
            
            print(f"[INFO] Computed {len(top_heads)} top and {len(bottom_heads)} bottom heads from {method_key}")
        else:
            print(f"[WARN] No top_heads, bottom_heads, or {method_key} found in DLA summary")
            return [], []
    
    # Limit to specified k values
    top_heads = top_heads[:args.top_k]
    bottom_heads = bottom_heads[:args.bottom_k]
    
    return top_heads, bottom_heads


def visualize_feature(feature_dir: Path, args, ds, tokenizer, hf_model, image_processor, model):
    """Visualize attention for a single feature."""
    print(f"[INFO] Processing feature: {feature_dir.name}")
    
    # Load DLA summary
    dla_summary_path = feature_dir / "dla_summary.json"
    if not dla_summary_path.exists():
        print(f"[WARN] No DLA summary found in {feature_dir}")
        return
    
    try:
        with open(dla_summary_path, 'r') as f:
            summary = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load DLA summary: {e}")
        return
    
    # Get top and bottom heads
    method_key = f"head_scores_method_{args.method}"
    top_heads, bottom_heads = get_head_lists(summary, args, method_key)
    
    if not top_heads and not bottom_heads:
        print(f"[WARN] No heads found for {feature_dir.name}")
        return
    
    # Robust fallbacks from model config if JSON is missing
    num_heads = int(summary.get("num_heads", getattr(hf_model.config, "num_attention_heads", 32)))
    _fallback_hidden = int(getattr(hf_model.config, "hidden_size", max(1, num_heads) * 128))
    head_dim = int(summary.get("head_dim", _fallback_hidden // max(1, num_heads)))
    
    # Get VQA samples for this feature
    vqa_indices = get_vqa_samples_from_organized_folder(feature_dir)
    if not vqa_indices:
        print(f"[WARN] No VQA samples found for {feature_dir.name}")
        return
    
    print(f"[INFO] Found {len(vqa_indices)} VQA samples for {feature_dir.name}")
    
    # Generate random sample indices for control group
    import random
    dataset_size = len(ds)
    random_indices = random.sample(range(dataset_size), min(args.num_random_samples, dataset_size))
    print(f"[INFO] Generated {len(random_indices)} random control samples")
    
    # Combine target and random samples
    all_sample_indices = vqa_indices + random_indices
    sample_types = ["target"] * len(vqa_indices) + ["random"] * len(random_indices)
    
    print(f"[INFO] Total samples to process: {len(all_sample_indices)} ({len(vqa_indices)} target + {len(random_indices)} random)")
    
    # Output dirs - separate by method and head type (shorter names)
    out_dir_top = feature_dir / f"attn_top_{args.method}"
    out_dir_bottom = feature_dir / f"attn_bottom_{args.method}"
    ex_dir = feature_dir / "examples_vqa"
    
    out_dir_top.mkdir(parents=True, exist_ok=True)
    if args.bottom_k > 0:
        out_dir_bottom.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)
    
    # GQA config (grouped-query attention)
    try:
        num_kv_heads: int = int(getattr(model.model.config, "num_key_value_heads", 0))
    except Exception:
        num_kv_heads = 0
    if not num_kv_heads:
        num_kv_heads = num_heads
    group_size = max(1, num_heads // max(1, num_kv_heads))
    
    # Get all unique layers needed for tracing
    all_layers = set()
    for layer, _ in top_heads + bottom_heads:
        all_layers.add(layer)
    
    # Process each VQA sample
    for sample_idx, (vqa_idx, sample_type) in enumerate(zip(all_sample_indices, sample_types)):
            
        # Load only the specific dataset sample we need
        try:
            sample = ds[vqa_idx]
            image = sample["image"].convert("RGB")
            question = str(sample.get("question", "")).strip()
            answer = sample.get("answer", None)
            prompt = question if question else "Answer the question."
        except Exception as e:
            print(f"[WARN] Failed to load VQA sample {vqa_idx}: {e}")
            continue
        
        # Inputs (and reconstruct the exact CLIP-processed, center-cropped image for display)
        input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
            image, prompt, image_processor, model._module, tokenizer
        )
        proc = image_processor(images=image, return_tensors="pt")
        pix = proc["pixel_values"][0]  # (3, H, W), normalized
        mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(image_processor.image_std).view(3, 1, 1)
        disp = (pix * std + mean).clamp(0, 1)
        disp_np = (disp.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        display_image = PILImage.fromarray(disp_np)
        
        img_start, img_end = get_image_token_positions(input_ids)
        
        # Trace (register saves first, forward-only; no gradients needed)
        with torch.no_grad():
            with model.trace(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=image_sizes,
            ):
                q_by_layer = {}
                k_by_layer = {}
                for layer in all_layers:
                    # If rope-aware, prefer the tensors after RoPE where available; fall back to pre-RoPE
                    try:
                        if args.rope_aware and hasattr(model.model.layers[layer].self_attn, 'q_proj'):
                            q_by_layer[layer] = (
                                model.model.layers[layer].self_attn.q_proj.output[0]
                                .save()
                            )
                            k_by_layer[layer] = (
                                model.model.layers[layer].self_attn.k_proj.output[0]
                                .save()
                            )
                        else:
                            q_by_layer[layer] = (
                                model.model.layers[layer].self_attn.q_proj.output[0]
                                .save()
                            )
                            k_by_layer[layer] = (
                                model.model.layers[layer].self_attn.k_proj.output[0]
                                .save()
                            )
                    except Exception:
                        # Fallback to pre-RoPE projections
                        q_by_layer[layer] = (
                            model.model.layers[layer].self_attn.q_proj.output[0]
                            .save()
                        )
                        k_by_layer[layer] = (
                            model.model.layers[layer].self_attn.k_proj.output[0]
                            .save()
                        )

                # Materialize up to max layer via a cheap scalar
                max_layer = max(all_layers) if all_layers else 0
                _ = model.model.layers[max_layer].output[0].sum()
        
        # Save raw image
        try:
            image.save(ex_dir / f"vqa_sample{vqa_idx}.png")
        except Exception:
            pass
        
        # Process top heads
        for layer, head in top_heads:
            try:
                q = q_by_layer[layer].value.detach().cpu()
                k = k_by_layer[layer].value.detach().cpu()
            except Exception:
                continue
            seq_len, hidden_dim = q.shape
            
            try:
                q_h = q.view(seq_len, num_heads, head_dim)
                k_h = k.view(seq_len, num_kv_heads, head_dim)
            except Exception:
                continue
            # Option A: apply RoPE ourselves to get post-RoPE Q/K when requested
            if args.rope_aware:
                try:
                    rope = model.model.layers[layer].self_attn.rotary_emb
                    rope_device = getattr(getattr(rope, 'inv_freq', None), 'device', q_h.device)
                    pos = torch.arange(seq_len, device=rope_device, dtype=torch.long)
                    cos, sin = rope(pos)
                    # Move to q_h device for arithmetic
                    cos = cos.to(q_h.device)
                    sin = sin.to(q_h.device)
                    # Ensure shape [seq, 1, dim] for broadcasting over heads
                    while cos.dim() < 3:
                        cos = cos.unsqueeze(1)
                        sin = sin.unsqueeze(1)
                    cos_q = cos.expand(seq_len, num_heads, -1)
                    sin_q = sin.expand(seq_len, num_heads, -1)
                    cos_k = cos.expand(seq_len, num_kv_heads, -1)
                    sin_k = sin.expand(seq_len, num_kv_heads, -1)
                    q_h = apply_rotary_pos_emb(q_h, cos_q, sin_q)
                    k_h = apply_rotary_pos_emb(k_h, cos_k, sin_k)
                except Exception:
                    pass
            
            # Choose query token per mode
            if args.query_mode == "last":
                q_vec = q_h[seq_len - 1, head]
            elif args.query_mode == "first_answer":
                # Heuristic: first text token after image span
                first_tok = int(img_end)
                first_tok = min(max(0, first_tok), seq_len - 1)
                q_vec = q_h[first_tok, head]
            else:  # avg_answer
                start = int(img_end)
                end = seq_len
                if end - start <= 0:
                    q_vec = q_h[seq_len - 1, head]
                else:
                    q_vec = q_h[start:end, head].mean(dim=0)
            kv_head = min(num_kv_heads - 1, max(0, head // group_size))
            # Normalize over full prefix, then slice image tokens
            scores_all = (k_h[:, kv_head] @ q_vec) / math.sqrt(head_dim)  # [seq_len]
            mask = torch.full_like(scores_all, float('-inf'))
            mask[img_start:img_end] = 0.0
            attn_all = torch.softmax(scores_all + mask, dim=0)
            attn_weights = attn_all[img_start:img_end].detach().cpu().numpy()
            
            num_img_tokens = attn_weights.shape[0]
            side = int(math.sqrt(num_img_tokens))
            if side * side != num_img_tokens:
                # simple fix-up: drop/pad by 1 token to nearest square if off by one (e.g., 575)
                if (side + 1) * (side + 1) - num_img_tokens == 1:
                    # pad with the min value
                    pad_val = float(attn_weights.min()) if hasattr(attn_weights, 'min') else float(attn_weights.min())
                    attn_weights = np.pad(attn_weights, (0, 1), mode='constant', constant_values=pad_val)
                    side = side + 1
                elif num_img_tokens - side * side == 1:
                    # drop last token
                    attn_weights = attn_weights[: side * side]
                else:
                    # skip if far from square
                    continue
            attn_map = attn_weights.reshape(side, side)
            
            # Normalize and optionally enhance contrast
            attn_map = (attn_map - attn_map.min()) / (attn_map.ptp() + 1e-8)
            if args.gamma and args.gamma != 1.0:
                attn_map = np.power(attn_map, args.gamma)
            
            # Resize heatmap to image resolution for a clean overlay
            attn_img = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
                display_image.size, resample=PILImage.BILINEAR
            )
            attn_arr = np.asarray(attn_img).astype(np.float32) / 255.0
            
            # Build figure with question and head rank/score
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
            
            fig, ax = plt.subplots(figsize=(8, 9))
            ax.imshow(display_image)
            ax.imshow(attn_arr, cmap=args.cmap, alpha=args.alpha, interpolation="bilinear", vmin=0.0, vmax=1.0)
            ax.axis("off")
            
            # Rank/score if available from summary
            head_scores_mat = None
            try:
                head_scores_mat = np.array(summary.get(f"head_scores_method_{args.method}", []))
            except Exception:
                head_scores_mat = None
            if args.rope_aware is False:
                # annotate proxy nature when not rope-aware
                pass
            rank = None
            try:
                # top_heads may be list of lists or tuples
                if len(top_heads) > 0 and isinstance(top_heads[0], list):
                    rank = top_heads.index([layer, head]) + 1
                else:
                    rank = top_heads.index((layer, head)) + 1
            except ValueError:
                rank = None
            
            title_bits = [f"VQA Sample {vqa_idx} ({sample_type.upper()}) – L{layer} H{head}"]
            if rank is not None:
                title_bits.append(f"Top Rank {rank}")
            
            # Question at the top (larger font)
            if question:
                question_text = "Q: " + (question[:200] + ("…" if len(question) > 200 else ""))
                ax.set_title(question_text, fontsize=22, fontweight='bold', pad=20, wrap=True)
            else:
                ax.set_title(" | ".join(title_bits), fontsize=14, fontweight='bold', pad=20)
            
            # Sample info at the bottom (cleaner, without post-RoPE and score)
            clean_title_bits = [f"VQA Sample {vqa_idx} ({sample_type.upper()}) – L{layer} H{head}"]
            if rank is not None:
                clean_title_bits.append(f"Top Rank {rank}")
            
            fig.subplots_adjust(bottom=0.12, top=0.90)
            fig.text(0.5, 0.12, " | ".join(clean_title_bits), ha="center", va="bottom", 
                    fontsize=11, fontweight='bold', wrap=True)
            
            out_path = out_dir_top / f"vqa_sample{vqa_idx}_{sample_type}_L{layer}_H{head}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Process bottom heads
        for layer, head in bottom_heads:
            try:
                q = q_by_layer[layer].value.detach().cpu()
                k = k_by_layer[layer].value.detach().cpu()
            except Exception:
                continue
            seq_len, hidden_dim = q.shape
            
            try:
                q_h = q.view(seq_len, num_heads, head_dim)
                k_h = k.view(seq_len, num_kv_heads, head_dim)
            except Exception:
                continue
            if args.rope_aware:
                try:
                    rope = model.model.layers[layer].self_attn.rotary_emb
                    rope_device = getattr(getattr(rope, 'inv_freq', None), 'device', q_h.device)
                    pos = torch.arange(seq_len, device=rope_device, dtype=torch.long)
                    cos, sin = rope(pos)
                    cos = cos.to(q_h.device)
                    sin = sin.to(q_h.device)
                    while cos.dim() < 3:
                        cos = cos.unsqueeze(1)
                        sin = sin.unsqueeze(1)
                    cos_q = cos.expand(seq_len, num_heads, -1)
                    sin_q = sin.expand(seq_len, num_heads, -1)
                    cos_k = cos.expand(seq_len, num_kv_heads, -1)
                    sin_k = sin.expand(seq_len, num_kv_heads, -1)
                    q_h = apply_rotary_pos_emb(q_h, cos_q, sin_q)
                    k_h = apply_rotary_pos_emb(k_h, cos_k, sin_k)
                except Exception:
                    pass
            
            # Choose query token per mode
            if args.query_mode == "last":
                q_vec = q_h[seq_len - 1, head]
            elif args.query_mode == "first_answer":
                first_tok = int(img_end)
                first_tok = min(max(0, first_tok), seq_len - 1)
                q_vec = q_h[first_tok, head]
            else:  # avg_answer
                start = int(img_end)
                end = seq_len
                if end - start <= 0:
                    q_vec = q_h[seq_len - 1, head]
                else:
                    q_vec = q_h[start:end, head].mean(dim=0)
            kv_head = min(num_kv_heads - 1, max(0, head // group_size))
            # Normalize over full prefix, then slice image tokens
            scores_all = (k_h[:, kv_head] @ q_vec) / math.sqrt(head_dim)  # [seq_len]
            mask = torch.full_like(scores_all, float('-inf'))
            mask[img_start:img_end] = 0.0
            attn_all = torch.softmax(scores_all + mask, dim=0)
            attn_weights = attn_all[img_start:img_end].detach().cpu().numpy()
            
            num_img_tokens = attn_weights.shape[0]
            side = int(math.sqrt(num_img_tokens))
            if side * side != num_img_tokens:
                if (side + 1) * (side + 1) - num_img_tokens == 1:
                    pad_val = float(attn_weights.min()) if hasattr(attn_weights, 'min') else float(attn_weights.min())
                    attn_weights = np.pad(attn_weights, (0, 1), mode='constant', constant_values=pad_val)
                    side = side + 1
                elif num_img_tokens - side * side == 1:
                    attn_weights = attn_weights[: side * side]
                else:
                    continue
            attn_map = attn_weights.reshape(side, side)
            
            # Normalize and optionally enhance contrast
            attn_map = (attn_map - attn_map.min()) / (attn_map.ptp() + 1e-8)
            if args.gamma and args.gamma != 1.0:
                attn_map = np.power(attn_map, args.gamma)
            
            # Resize heatmap to image resolution for a clean overlay
            attn_img = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
                display_image.size, resample=PILImage.BILINEAR
            )
            attn_arr = np.asarray(attn_img).astype(np.float32) / 255.0
            
            # Build figure with question and head rank/score
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
            
            fig, ax = plt.subplots(figsize=(8, 9))
            ax.imshow(display_image)
            ax.imshow(attn_arr, cmap=args.cmap, alpha=args.alpha, interpolation="bilinear", vmin=0.0, vmax=1.0)
            ax.axis("off")
            
            # Rank/score if available from summary
            head_scores_mat = None
            try:
                head_scores_mat = np.array(summary.get(f"head_scores_method_{args.method}", []))
            except Exception:
                head_scores_mat = None
            rank = None
            try:
                # bottom_heads may be list of lists or tuples
                if len(bottom_heads) > 0 and isinstance(bottom_heads[0], list):
                    rank = bottom_heads.index([layer, head]) + 1
                else:
                    rank = bottom_heads.index((layer, head)) + 1
            except ValueError:
                rank = None
            
            title_bits = [f"VQA Sample {vqa_idx} ({sample_type.upper()}) – L{layer} H{head}"]
            if rank is not None:
                title_bits.append(f"Bottom Rank {rank}")
            if head_scores_mat is not None and head_scores_mat.size > 0:
                if 0 <= layer < head_scores_mat.shape[0] and 0 <= head < head_scores_mat.shape[1]:
                    title_bits.append(f"Score {float(head_scores_mat[layer, head]):.4f}")

            
            # Question at the top
            if question:
                question_text = "Q: " + (question[:200] + ("…" if len(question) > 200 else ""))
                ax.set_title(question_text, fontsize=18, fontweight='bold', pad=20, wrap=True)
            else:
                ax.set_title(" | ".join(title_bits), fontsize=14, fontweight='bold', pad=20)
            
            # Sample info at the bottom
            fig.subplots_adjust(bottom=0.12, top=0.90)
            fig.text(0.5, 0.12, " | ".join(title_bits), ha="center", va="bottom", 
                    fontsize=11, fontweight='bold', wrap=True)
            
            out_path = out_dir_bottom / f"vqa_sample{vqa_idx}_{sample_type}_L{layer}_H{head}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"  - Processed VQA sample {vqa_idx} ({sample_type})")
    
    print(f"[INFO] Saved attention visualizations for {feature_dir.name}")
    print(f"  → Top heads: {out_dir_top}")
    if args.bottom_k > 0:
        print(f"  → Bottom heads: {out_dir_bottom}")


def main():
    parser = argparse.ArgumentParser(description="Attention visualization for organized spatial features")
    parser.add_argument("--organized-features-dir", type=str, required=True,
                        help="Directory containing organized spatial feature folders")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top heads to visualize (default: 5)")
    parser.add_argument("--bottom-k", type=int, default=5,
                        help="Number of bottom heads to visualize (default: 5)")
    parser.add_argument("--num-random-samples", type=int, default=5,
                        help="Number of random control samples to process")
    parser.add_argument("--method", type=str, choices=["A", "B", "both"], default="A",
                        help="DLA method to use for head scores (A, B, or both)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay transparency (0..1)")
    parser.add_argument("--cmap", type=str, default="magma",
                        help="Matplotlib colormap for the heatmap")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma for attention contrast (1.0 disables)")
    parser.add_argument("--feature-filter", type=str, default="",
                        help="Optional filter for specific feature folders (e.g., 'layer_2_feature_27525')")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--query-mode", type=str, choices=["last", "first_answer", "avg_answer"], default="last",
                        help="Which query token(s) to use: last token, first answer token, or average over answer tokens")
    parser.add_argument("--rope-aware", action="store_true",
                        help="Attempt to capture post-RoPE Q/K if available; otherwise use pre-RoPE proxy and note in titles")
    args = parser.parse_args()
    
    organized_features_dir = Path(args.organized_features_dir)
    if not organized_features_dir.exists():
        print(f"[ERROR] Organized features directory not found: {organized_features_dir}")
        return
    
    # Find all feature folders
    feature_dirs = []
    for item in organized_features_dir.iterdir():
        if item.is_dir() and item.name.startswith("layer_") and "_feature_" in item.name:
            if args.feature_filter and args.feature_filter not in item.name:
                continue
            feature_dirs.append(item)
    
    if not feature_dirs:
        print(f"[ERROR] No feature folders found in {organized_features_dir}")
        return
    
    print(f"[INFO] Found {len(feature_dirs)} feature folders to process")
    
    # Initialize model and dataset once
    print("[INFO] Initializing model and dataset...")
    # Determinism
    try:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    tokenizer, hf_model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    model = NNsight(hf_model)
    ds = load_dataset_vqa()  # Lazy loading - only loads when accessed
    
    # Process each feature
    for i, feature_dir in enumerate(sorted(feature_dirs)):
        print(f"\n[{i+1}/{len(feature_dirs)}] Processing {feature_dir.name}")
        
        # Process for each method
        methods_to_process = ["A", "B"] if args.method == "both" else [args.method]
        
        for method in methods_to_process:
            print(f"  Processing method {method}...")
            # Create a copy of args with the current method
            import copy
            method_args = copy.deepcopy(args)
            method_args.method = method
            
            try:
                visualize_feature(feature_dir, method_args, ds, tokenizer, hf_model, image_processor, model)
            except Exception as e:
                print(f"[ERROR] Failed to process {feature_dir.name} with method {method}: {e}")
                continue
    
    print(f"\n[INFO] Completed processing {len(feature_dirs)} feature folders")


if __name__ == "__main__":
    main()
