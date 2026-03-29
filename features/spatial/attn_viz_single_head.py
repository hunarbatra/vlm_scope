#!/usr/bin/env python3
"""
Single head attention visualization for specific samples with customizable questions.

This script allows you to visualize attention maps for a specific attention head
on a specific VQA sample, with the ability to change the question.

Example:
CUDA_VISIBLE_DEVICES=2 python features/spatial/attn_viz_single_head.py \
  --sample-idx 449 \
  --dataset vsr \
  --vsr-split train \
  --layer 15 \
  --head 10 \
  --output-dir results/single_head_viz \
  --questions "The food is on top of the tray.", "The man is next to the oven.", "The food is in the oven.", "The man is holding the tray." \
  --wrap-vsr-prompt



"""

from __future__ import annotations

import argparse
import math
import json
from io import BytesIO
from pathlib import Path
from typing import Optional, List

import torch
from nnsight import NNsight
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import requests
import seaborn as sns
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import sys
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    get_image_token_positions,
)
from datasets import load_dataset, concatenate_datasets


def load_dataset_vqa():
    """Load the VQAv2 validation split dataset object (lazy loading)."""
    return load_dataset("lmms-lab/VQAv2", split="validation")


def load_dataset_vsr(split: str = "train"):
    """Load VSR dataset split (cambridgeltl/vsr_random)."""
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    if split == "train+dev+test":
        train_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="train")
        dev_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="dev")
        test_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="test")
        return concatenate_datasets([train_ds, dev_ds, test_ds])
    return load_dataset("cambridgeltl/vsr_random", data_files=data_files, split=split)


def _build_vsr_prompt(statement: str) -> str:
    s = statement.strip()
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {s}\n"
        "Answer:"
    )


def _lookup_vsr_baseline(jsonl_path: str, sample_idx: int) -> Optional[dict]:
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if int(rec.get("dataset_index", -1)) == int(sample_idx):
                        return rec
                except Exception:
                    continue
    except Exception:
        return None
    return None


def visualize_single_head_attention(
    sample_idx: int,
    layer: int,
    head: int,
    questions: List[str],
    output_dir: Path,
    ds,
    tokenizer,
    hf_model,
    image_processor,
    model,
    alpha: float = 0.6,
    cmap: str = "magma",
    gamma: float = 1.0,
    save_raw_image: bool = True,
    show_question: bool = True,
    figsize: tuple = (10, 8),
    query_mode: str = "last",
    rope_aware: bool = False,
    dataset: str = "vqa",
    vsr_baseline_jsonl: Optional[str] = None,
    wrap_vsr_prompt: bool = False,
):
    """Visualize attention for a single head on a specific sample."""
    
    print(f"[INFO] Processing sample {sample_idx}, layer {layer}, head {head}")
    
    # Load the specific dataset sample
    try:
        sample = ds[sample_idx]
        if dataset == "vqa":
            image = sample["image"].convert("RGB")
            original_question = str(sample.get("question", "")).strip()
            answer = sample.get("answer", None)
        else:
            # VSR: download by URL
            image_url = sample.get("image_link") or sample.get("image") or ""
            try:
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
                image = PILImage.open(BytesIO(resp.content)).convert("RGB")
            except Exception:
                image = PILImage.new("RGB", (224, 224), (128, 128, 128))
            original_question = str(sample.get("caption", "")).strip()
            answer = sample.get("label", None)
        
        # Use custom questions if provided, otherwise default per dataset
        if questions:
            if dataset == "vsr" and wrap_vsr_prompt:
                prompts = [_build_vsr_prompt(q) for q in questions]
            else:
                prompts = questions
            questions_display = questions
        else:
            if dataset == "vqa":
                prompts = [original_question if original_question else "Answer the question."]
                questions_display = [original_question]
            else:
                prompts = [_build_vsr_prompt(original_question if original_question else "")]  # yes/no prompt
                questions_display = [original_question]
        
    except Exception as e:
        print(f"[ERROR] Failed to load VQA sample {sample_idx}: {e}")
        return
    
    # Reconstruct the exact CLIP-processed, center-cropped image for display
    proc = image_processor(images=image, return_tensors="pt")
    pix = proc["pixel_values"][0]  # (3, H, W), normalized
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    disp = (pix * std + mean).clamp(0, 1)
    disp_np = (disp.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    display_image = PILImage.fromarray(disp_np)
    
    # Get model configuration first (needed for tensor processing)
    num_heads = int(getattr(model.model.config, "num_attention_heads", 32))
    head_dim = int(getattr(model.model.config, "hidden_size", 4096) // num_heads)
    
    # GQA config (grouped-query attention)
    try:
        num_kv_heads: int = int(getattr(model.model.config, "num_key_value_heads", 0))
    except Exception:
        num_kv_heads = 0
    if not num_kv_heads:
        num_kv_heads = num_heads
    group_size = max(1, num_heads // max(1, num_kv_heads))
    
    print(f"[INFO] Model config: {num_heads} heads, {head_dim} dim, {num_kv_heads} KV heads")
    
    # Store attention maps for each question
    attention_maps = []
    attention_weights_list = []
    
    # Process each question to get attention maps
    for i, prompt in enumerate(prompts):
        print(f"  Processing question {i+1}: {prompt[:50]}...")
        
        # Process inputs for this question
        input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
            image, prompt, image_processor, model._module, tokenizer
        )
        
        img_start, img_end = get_image_token_positions(input_ids)
        
        # Trace the model to get attention weights for this question
        with torch.no_grad():
            with model.trace(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=image_sizes,
            ):
                # Get q and k for the specific layer
                q = (
                    model.model.layers[layer].self_attn.q_proj.output[0]
                    .save()
                )
                k = (
                    model.model.layers[layer].self_attn.k_proj.output[0]
                    .save()
                )
                
                # Materialize up to the specific layer
                _ = model.model.layers[layer].output[0].sum()
        
        # Extract q and k values
        try:
            # Handle both NNsight wrapped tensors and direct tensors
            if hasattr(q, 'value'):
                q = q.value.detach().cpu()
            elif hasattr(q, 'tensor'):
                q = q.tensor.detach().cpu()
            elif hasattr(q, 'data'):
                q = q.data.detach().cpu()
            else:
                # Direct access if it's already a tensor
                q = q.detach().cpu()
                
            if hasattr(k, 'value'):
                k = k.value.detach().cpu()
            elif hasattr(k, 'tensor'):
                k = k.tensor.detach().cpu()
            elif hasattr(k, 'data'):
                k = k.data.detach().cpu()
            else:
                # Direct access if it's already a tensor
                k = k.detach().cpu()
                
        except Exception as e:
            print(f"[ERROR] Failed to extract q/k values for question {i+1}: {e}")
            continue
        
        seq_len, hidden_dim = q.shape
        
        # Reshape to separate heads
        try:
            q_h = q.view(seq_len, num_heads, head_dim)
            k_h = k.view(seq_len, num_kv_heads, head_dim)
        except Exception as e:
            print(f"[ERROR] Failed to reshape q/k to heads for question {i+1}: {e}")
            continue
        
        # Apply RoPE if requested
        if rope_aware:
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
        if query_mode == "last":
            q_vec = q_h[seq_len - 1, head]
        elif query_mode == "first_answer":
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
            # Simple fix-up: drop/pad by 1 token to nearest square if off by one
            if (side + 1) * (side + 1) - num_img_tokens == 1:
                pad_val = float(attn_weights.min()) if hasattr(attn_weights, 'min') else float(attn_weights.min())
                attn_weights = np.pad(attn_weights, (0, 1), mode='constant', constant_values=pad_val)
                side = side + 1
            elif num_img_tokens - side * side == 1:
                attn_weights = attn_weights[: side * side]
            else:
                print(f"[WARN] Image tokens ({num_img_tokens}) don't form a perfect square for question {i+1}, using 1D visualization")
                attn_map = attn_weights.reshape(1, -1)
                attention_maps.append(attn_map)
                attention_weights_list.append(attn_weights)
                continue
        
        attn_map = attn_weights.reshape(side, side)
        
        # Normalize and optionally enhance contrast
        attn_map = (attn_map - attn_map.min()) / (attn_map.ptp() + 1e-8)
        if gamma != 1.0:
            attn_map = np.power(attn_map, gamma)
        
        # Resize heatmap to image resolution for overlay
        attn_img = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
            display_image.size, resample=PILImage.BILINEAR
        )
        attn_arr = np.asarray(attn_img).astype(np.float32) / 255.0
        
        attention_maps.append(attn_arr)
        attention_weights_list.append(attn_weights)
    
    if not attention_maps:
        print(f"[ERROR] No attention maps generated for any question")
        return
    
    print(f"[INFO] Generated {len(attention_maps)} attention maps")
    
    # Create side-by-side visualization with improved styling
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
    
    # Calculate figure size based on number of questions
    num_plots = len(attention_maps) + 1  # +1 for original image
    plot_width = 8
    plot_height = 9
    total_width = num_plots * plot_width
    
    fig, axes = plt.subplots(1, num_plots, figsize=(total_width, plot_height))
    if num_plots == 1:
        axes = [axes]  # Make it iterable for single plot case
    
    # Plot 1: Original image
    axes[0].imshow(display_image)
    axes[0].set_title(f"Original Image\nSample {sample_idx}", fontsize=18, fontweight='bold', pad=20)
    axes[0].axis("off")
    
    # Plot 2 onwards: Attention maps for each question
    for i, (attn_arr, question_text) in enumerate(zip(attention_maps, questions_display)):
        ax = axes[i + 1]
        
        # Show the image with attention overlay
        ax.imshow(display_image)
        im = ax.imshow(attn_arr, cmap=cmap, alpha=alpha, interpolation="bilinear", vmin=0.0, vmax=1.0)
        ax.axis("off")
        
        # Add question as title (truncated if too long)
        question_text_clean = "Q: " + (question_text[:200] + ("…" if len(question_text) > 200 else ""))
        ax.set_title(question_text_clean, fontsize=22, fontweight='bold', pad=20, wrap=True)
    
    # Add overall title
    title_bits = [f"Sample {sample_idx} | Layer {layer} | Head {head}"]
    if rope_aware:
        title_bits.append("(RoPE-aware)")
    if query_mode != "last":
        title_bits.append(f"Query: {query_mode}")
    
    fig.suptitle(" | ".join(title_bits), fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.12, top=0.90, wspace=0.3)
    
    # Add sample info at the bottom
    info_bits = [f"VQA Sample {sample_idx} – L{layer} H{head}"]
    if query_mode != "last":
        info_bits.append(f"Query Mode: {query_mode}")
    if rope_aware:
        info_bits.append("RoPE-aware")
    
    fig.text(0.5, 0.12, " | ".join(info_bits), ha="center", va="bottom", 
            fontsize=11, fontweight='bold', wrap=True)
    
    # Add original answer if available (centered below all plots)
    if answer:
        answer_text = f"Original A: {str(answer)[:80]}{'…' if len(str(answer)) > 80 else ''}"
        fig.text(0.5, 0.05, answer_text, ha="center", va="bottom", 
                fontsize=9, style='italic', color='gray',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

    # Optionally add VSR baseline metrics (label/pred/correct/p_yes/p_no)
    if dataset == "vsr" and vsr_baseline_jsonl:
        rec = _lookup_vsr_baseline(vsr_baseline_jsonl, sample_idx)
        if rec:
            try:
                lbl = int(rec.get("label"))
                pred = int(rec.get("pred"))
                corr = int(rec.get("correct"))
                py = float(rec.get("p_yes", 0.0))
                pn = float(rec.get("p_no", 0.0))
                base_text = f"Baseline – label={lbl} pred={pred} correct={corr}  p_yes={py:.3f}  p_no={pn:.3f}"
                fig.text(0.5, 0.02, base_text, ha="center", va="bottom", fontsize=10, color='black')
            except Exception:
                pass
    
    # Save the visualization with questions in filename
    if questions:
        # Create a safe filename from the first question
        first_question = questions[0]
        safe_question = "".join(c for c in first_question if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_question = safe_question.replace(' ', '_')[:30]  # Limit length
        if len(questions) > 1:
            filename = f"sample_{sample_idx}_L{layer}_H{head}_Q{len(questions)}questions_{safe_question}.png"
        else:
            filename = f"sample_{sample_idx}_L{layer}_H{head}_Q_{safe_question}.png"
    else:
        filename = f"sample_{sample_idx}_L{layer}_H{head}.png"
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[INFO] Saved visualization to {output_path}")
    print(f"[INFO] Filename: {filename}")
    
    # Print some statistics for each question
    print(f"[INFO] Attention statistics for {len(attention_maps)} questions:")
    for i, attn_weights in enumerate(attention_weights_list):
        print(f"  Question {i+1}:")
        print(f"    - Min weight: {attn_weights.min():.6f}")
        print(f"    - Max weight: {attn_weights.max():.6f}")
        print(f"    - Mean weight: {attn_weights.mean():.6f}")
        print(f"    - Std weight: {attn_weights.std():.6f}")
        
        # Show top attended tokens
        top_indices = np.argsort(attn_weights)[-5:][::-1]
        print(f"    - Top 5 attended image tokens: {top_indices.tolist()}")
        print(f"    - Top 5 attention weights: {attn_weights[top_indices].tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Single head attention visualization for specific samples")
    parser.add_argument("--sample-idx", type=int, required=True,
                        help="Sample index to visualize (VQA or VSR)")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer number for the attention head")
    parser.add_argument("--head", type=int, required=True,
                        help="Head number within the layer")
    parser.add_argument("--questions", type=str, nargs='+', default=[],
                        help="Custom questions to ask (if empty, uses original question)")
    parser.add_argument("--dataset", type=str, choices=["vqa", "vsr"], default="vqa",
                        help="Dataset to visualize from (vqa or vsr)")
    parser.add_argument("--vsr-split", type=str, choices=["train", "dev", "test", "train+dev+test"], default="train",
                        help="VSR split to load when --dataset vsr")
    parser.add_argument("--vsr-baseline-jsonl", type=str, default="",
                        help="Optional path to baseline JSONL to annotate label/pred/correct under the figure (VSR only)")
    parser.add_argument("--output-dir", type=str, default="results/single_head_viz",
                        help="Output directory for visualizations")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay transparency (0..1)")
    parser.add_argument("--cmap", type=str, default="magma",
                        help="Matplotlib colormap for the heatmap")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma for attention contrast (1.0 disables)")
    parser.add_argument("--figsize", type=str, default="10,8",
                        help="Figure size as 'width,height'")
    parser.add_argument("--no-raw-image", action="store_true",
                        help="Don't save the raw image")
    parser.add_argument("--no-question", action="store_true",
                        help="Don't show the question text")
    parser.add_argument("--query-mode", type=str, choices=["last", "first_answer", "avg_answer"], default="last",
                        help="Which query token(s) to use: last token, first answer token, or average over answer tokens")
    parser.add_argument("--rope-aware", action="store_true",
                        help="Attempt to capture post-RoPE Q/K if available; otherwise use pre-RoPE proxy and note in titles")
    parser.add_argument("--wrap-vsr-prompt", action="store_true",
                        help="When --dataset vsr and custom --questions are provided, wrap each question with the standard VSR yes/no prompt")
    
    args = parser.parse_args()
    
    # Parse figure size
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        figsize = (10, 8)
        print(f"[WARN] Invalid figsize format, using default: {figsize}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Single head attention visualization")
    print(f"[INFO] Sample: {args.sample_idx}")
    print(f"[INFO] Layer: {args.layer}, Head: {args.head}")
    print(f"[INFO] Query mode: {args.query_mode}")
    if args.rope_aware:
        print(f"[INFO] RoPE-aware processing enabled")
    if args.questions:
        print(f"[INFO] Questions: {len(args.questions)} custom questions")
        for i, q in enumerate(args.questions):
            print(f"  Q{i+1}: {q}")
    else:
        print(f"[INFO] Question: (original)")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Initialize model and dataset
    print("[INFO] Initializing model and dataset...")
    tokenizer, hf_model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    model = NNsight(hf_model)
    ds = load_dataset_vqa() if args.dataset == "vqa" else load_dataset_vsr(args.vsr_split)
    
    # Check if sample exists
    if args.sample_idx >= len(ds):
        print(f"[ERROR] Sample index {args.sample_idx} is out of range. Dataset has {len(ds)} samples.")
        return
    
    # Visualize the attention
    try:
        visualize_single_head_attention(
            sample_idx=args.sample_idx,
            layer=args.layer,
            head=args.head,
            questions=args.questions if args.questions else [],
            output_dir=output_dir,
            ds=ds,
            tokenizer=tokenizer,
            hf_model=hf_model,
            image_processor=image_processor,
            model=model,
            alpha=args.alpha,
            cmap=args.cmap,
            gamma=args.gamma,
            save_raw_image=not args.no_raw_image,
            show_question=not args.no_question,
            figsize=figsize,
            query_mode=args.query_mode,
            rope_aware=args.rope_aware,
            dataset=args.dataset,
            vsr_baseline_jsonl=(args.vsr_baseline_jsonl if args.vsr_baseline_jsonl else None),
            wrap_vsr_prompt=bool(args.wrap_vsr_prompt),
        )
        print(f"[SUCCESS] Visualization completed!")
        
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
