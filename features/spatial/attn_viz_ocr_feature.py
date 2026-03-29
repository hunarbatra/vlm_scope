#!/usr/bin/env python3
"""
Visualize attention overlays for a single OCR feature using top samples from a
given dataset directory (e.g., results/stage_4/feature_samples/full/vqa_ocr).

This script is a focused variant of attn_viz_common_features.py tailored for
OCR experiments where you already have per-feature sample directories with a
sample_info.json.

It will:
- Locate the feature directory inside --dataset-dir for --layer/--feature
- Read top samples from sample_info.json
- Load top heads from an aggregated attribution JSON (e.g., results/ocr_attribution_summary.json)
  or fall back to the per-feature results/organized_spatial_features/.../dla_summary.json
- Render attention overlays for the selected heads

Example:
CUDA_VISIBLE_DEVICES=1 python features/spatial/attn_viz_ocr_feature.py \
  --dataset-dir results/stage_4/feature_samples/full/vqa_ocr \
  --layer 22 --feature 7291 \
  --combined-attrib-path results/ocr_attribution_summary.json \
  --output-dir results/ocr_attention_viz_single \
  --top-k 3 --samples-per-feature 10 --rope-aware --method A
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image as PILImage

from nnsight import NNsight

import sys
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    get_image_token_positions,
)
from datasets import load_dataset


def find_feature_directory(dataset_dir: Path, layer: int, feature: int) -> Optional[Path]:
    """Return the path to the feature directory inside dataset_dir, if present."""
    candidates = [
        dataset_dir / f"text-only_layer_{layer}_feature_{feature}",
        dataset_dir / f"layer_{layer}_feature_{feature}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_sample_info(feature_dir: Path) -> List[Dict]:
    """Load sample_info.json from the feature directory."""
    info_path = feature_dir / "sample_info.json"
    if not info_path.exists():
        return []
    try:
        return json.loads(info_path.read_text())
    except Exception:
        return []


def select_top_samples(samples_info: List[Dict], top_k: int) -> List[Dict]:
    """Return top_k sample entries. Assumes samples_info already sorted by rank."""
    if top_k <= 0:
        return samples_info
    return samples_info[:top_k]


def parse_heads_from_agg_for_feature(
    agg_path: Path, layer: int, feature: int, method: str, top_k: int, bottom: bool
) -> List[Tuple[int, int]]:
    """Parse top heads for one feature from aggregated OCR attribution summary.

    Expected structure (results/ocr_attribution_summary.json):
      {
        "features": [
          {"layer": L, "feature": F, "top_heads_method_A": [...], "top_heads_method_B": [...], ...},
          ...
        ]
      }
    """
    try:
        payload = json.loads(agg_path.read_text())
    except Exception:
        return []

    feats = payload.get("features")
    if not isinstance(feats, list):
        return []

    key = f"top_heads_method_{method.upper()}"
    selected: List[Tuple[int, int]] = []
    for item in feats:
        try:
            if int(item.get("layer")) == int(layer) and int(item.get("feature")) == int(feature):
                heads = item.get(key) or []
                pairs: List[Tuple[int, int]] = []
                for h in heads:
                    try:
                        pairs.append((int(h.get("layer")), int(h.get("head"))))
                    except Exception:
                        continue
                if not pairs:
                    return []
                if top_k > 0:
                    pairs = (pairs[-top_k:] if bottom else pairs[:top_k])
                selected = pairs
                break
        except Exception:
            continue
    return selected


def parse_heads_from_dla_summary(
    base_dir: Path, layer: int, feature: int, method: str, top_k: int, bottom: bool
) -> List[Tuple[int, int]]:
    """Fallback: read per-feature dla_summary.json for top/bottom heads."""
    feat_dir = base_dir / f"layer_{layer}_feature_{feature}"
    path = feat_dir / "dla_summary.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []

    key_top = f"top_heads_method_{method.upper()}"
    key_bottom = f"bottom_heads_method_{method.upper()}"
    chosen_key = key_bottom if bottom else key_top
    heads = payload.get(chosen_key) or []
    pairs: List[Tuple[int, int]] = []
    for h in heads:
        try:
            pairs.append((int(h.get("layer")), int(h.get("head"))))
        except Exception:
            continue
    if not pairs:
        return []
    if top_k > 0:
        pairs = (pairs[-top_k:] if bottom else pairs[:top_k])
    return pairs


def load_vqa_validation() -> any:
    """Load the VQAv2 validation split dataset object."""
    return load_dataset("lmms-lab/VQAv2", split="validation")


def get_prompt_for_vqa(sample: Dict, fallback: str = "Answer the question.") -> str:
    q = str(sample.get("question", "")).strip()
    return q if q else fallback


def compute_attention_overlay(
    model: NNsight,
    hf_model,
    tokenizer,
    image_processor,
    image: PILImage.Image,
    prompt_text: str,
    layer_head_pairs: List[Tuple[int, int]],
    rope_aware: bool,
    alpha: float,
    cmap: str,
    gamma: float,
    query_mode: str,
) -> List[Tuple[int, int, np.ndarray, PILImage.Image]]:
    """Compute attention overlays for requested (layer, head) pairs.

    Returns list of tuples: (layer, head, attn_arr, display_image)
    """
    input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
        image, prompt_text, image_processor, model._module, tokenizer
    )

    proc = image_processor(images=image, return_tensors="pt")
    pix = proc["pixel_values"][0]
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    disp = (pix * std + mean).clamp(0, 1)
    disp_np = (disp.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    display_image = PILImage.fromarray(disp_np)

    img_start, img_end = get_image_token_positions(input_ids)

    num_heads = int(getattr(hf_model.config, "num_attention_heads", 32))
    _fallback_hidden = int(getattr(hf_model.config, "hidden_size", max(1, num_heads) * 128))
    head_dim = int(_fallback_hidden // max(1, num_heads))

    try:
        num_kv_heads: int = int(getattr(model.model.config, "num_key_value_heads", 0))
    except Exception:
        num_kv_heads = 0
    if not num_kv_heads:
        num_kv_heads = num_heads
    group_size = max(1, num_heads // max(1, num_kv_heads))

    needed_layers = sorted({ly for ly, _ in layer_head_pairs})

    overlays: List[Tuple[int, int, np.ndarray, PILImage.Image]] = []

    with torch.no_grad():
        with model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=image_sizes,
        ):
            q_by_layer = {}
            k_by_layer = {}
            for ly in needed_layers:
                try:
                    q_by_layer[ly] = (
                        model.model.layers[ly].self_attn.q_proj.output[0]
                        .save()
                    )
                    k_by_layer[ly] = (
                        model.model.layers[ly].self_attn.k_proj.output[0]
                        .save()
                    )
                except Exception:
                    continue
            if needed_layers:
                _ = model.model.layers[max(needed_layers)].output[0].sum()

    seq_len = int(q_by_layer[needed_layers[0]].value.shape[0]) if needed_layers else 0

    for ly, hd in layer_head_pairs:
        try:
            q = q_by_layer[ly].value.detach().cpu()
            k = k_by_layer[ly].value.detach().cpu()
        except Exception:
            continue

        try:
            q_h = q.view(seq_len, num_heads, head_dim)
            k_h = k.view(seq_len, num_kv_heads, head_dim)
        except Exception:
            continue

        if rope_aware:
            try:
                rope = model.model.layers[ly].self_attn.rotary_emb
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
                q_h = apply_rotary_pos_emb(q_h, cos_q, sin_q)  # type: ignore[name-defined]
                k_h = apply_rotary_pos_emb(k_h, cos_k, sin_k)  # type: ignore[name-defined]
            except Exception:
                pass

        if query_mode == "last":
            q_vec = q_h[seq_len - 1, hd]
        elif query_mode == "first_answer":
            first_tok = int(img_end)
            first_tok = min(max(0, first_tok), seq_len - 1)
            q_vec = q_h[first_tok, hd]
        else:
            start = int(img_end)
            end = seq_len
            if end - start <= 0:
                q_vec = q_h[seq_len - 1, hd]
            else:
                q_vec = q_h[start:end, hd].mean(dim=0)

        kv_head = min(num_kv_heads - 1, max(0, hd // group_size))
        scores_all = (k_h[:, kv_head] @ q_vec) / math.sqrt(head_dim)
        mask = torch.full_like(scores_all, float('-inf'))
        mask[img_start:img_end] = 0.0
        attn_all = torch.softmax(scores_all + mask, dim=0)
        attn_weights = attn_all[img_start:img_end].detach().cpu().numpy()

        num_img_tokens = attn_weights.shape[0]
        side = int(math.sqrt(num_img_tokens))
        if side * side != num_img_tokens:
            if (side + 1) * (side + 1) - num_img_tokens == 1:
                pad_val = float(attn_weights.min()) if hasattr(attn_weights, 'min') else float(np.min(attn_weights))
                attn_weights = np.pad(attn_weights, (0, 1), mode='constant', constant_values=pad_val)
                side = side + 1
            elif num_img_tokens - side * side == 1:
                attn_weights = attn_weights[: side * side]
            else:
                continue
        attn_map = attn_weights.reshape(side, side)

        attn_map = (attn_map - attn_map.min()) / (attn_map.ptp() + 1e-8)
        if gamma and gamma != 1.0:
            attn_map = np.power(attn_map, gamma)

        attn_img = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
            display_image.size, resample=PILImage.BILINEAR
        )
        attn_arr = np.asarray(attn_img).astype(np.float32) / 255.0

        overlays.append((ly, hd, attn_arr, display_image))

    return overlays


def save_attention_figure(
    display_image: PILImage.Image,
    attn_arr: np.ndarray,
    out_path: Path,
    title: str,
    question_text: str,
    cmap: str,
    alpha: float,
) -> None:
    plt.style.use('default')
    sns.set_palette("husl")
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
    ax.imshow(attn_arr, cmap=cmap, alpha=alpha, interpolation="bilinear", vmin=0.0, vmax=1.0)
    ax.axis("off")

    # Create title with question text (similar to attn_viz_common_features.py)
    title_bits = [
        f"VQA Sample {title.split(' – ')[0].split(' ')[-1]} – {title.split(' – ')[1]} – {title.split(' – ')[2]}",
        f"{'Bottom' if 'bottom' in title.lower() else 'Top'} (Method {title.split('(')[-1].split(')')[0].split()[-1]})"
    ]
    
    if question_text:
        question_display = ("Q: " + (question_text[:200] + ("…" if len(question_text) > 200 else "")))
        ax.set_title(question_display, fontsize=22, fontweight='bold', pad=20, wrap=True)
    else:
        ax.set_title(" | ".join(title_bits), fontsize=14, fontweight='bold', pad=20)

    fig.subplots_adjust(bottom=0.12, top=0.90)
    fig.text(0.5, 0.12, " | ".join(title_bits), ha="center", va="bottom",
            fontsize=11, fontweight='bold', wrap=True)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Attention visualization for a single OCR feature using dataset-dir samples")
    ap.add_argument("--dataset-dir", required=True, help="Directory with per-feature subfolders and sample_info.json (e.g., results/stage_4/feature_samples/full/vqa_ocr)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--combined-attrib-path", default="results/ocr_attribution_summary.json", help="Aggregated OCR attribution summary JSON; fallback to per-feature dla_summary.json if missing")
    ap.add_argument("--per-feature-results-dir", default="results/organized_spatial_features", help="Base dir containing layer_*/dla_summary.json for fallback")
    ap.add_argument("--method", choices=["A", "B"], default="B", help="Attribution method to use for head selection")
    ap.add_argument("--top-k", type=int, default=2, help="Number of heads to visualize")
    ap.add_argument("--bottom-heads", action="store_true", help="Use bottom-k heads instead of top-k")
    ap.add_argument("--samples-per-feature", type=int, default=5, help="Number of samples to visualize")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cmap", type=str, default="magma")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--query-mode", type=str, choices=["last", "first_answer", "avg_answer"], default="last")
    ap.add_argument("--rope-aware", action="store_true")
    ap.add_argument("--output-dir", default="results/ocr_attention_viz_single")

    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    feat_dir = find_feature_directory(dataset_dir, args.layer, args.feature)
    if not feat_dir:
        raise FileNotFoundError(f"Feature dir not found under {dataset_dir} for layer {args.layer}, feature {args.feature}")

    samples_info = load_sample_info(feat_dir)
    if not samples_info:
        raise FileNotFoundError(f"sample_info.json missing or empty in {feat_dir}")

    samples = select_top_samples(samples_info, args.samples_per_feature)

    # Load heads from aggregated OCR attribution first, then fallback to per-feature dla_summary
    heads = []
    agg_path = Path(args.combined_attrib_path)
    if agg_path.exists():
        heads = parse_heads_from_agg_for_feature(
            agg_path, args.layer, args.feature, args.method, args.top_k, args.bottom_heads
        )
    if not heads:
        heads = parse_heads_from_dla_summary(
            Path(args.per_feature_results_dir), args.layer, args.feature, args.method, args.top_k, args.bottom_heads
        )
    if not heads:
        raise FileNotFoundError("Could not determine heads to visualize from attribution JSONs")

    # Initialize model and dataset
    print("[INFO] Initializing model and dataset...")
    tokenizer, hf_model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    model = NNsight(hf_model)
    ds_vqa = load_vqa_validation()

    # Prepare output directories
    base_out_dir = Path(args.output_dir) / f"layer_{args.layer}_feature_{args.feature}"
    out_dir = base_out_dir / f"attn_top_{'bottom' if args.bottom_heads else 'top'}_{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process samples
    for rec in samples:
        try:
            idx = int(rec.get("sample_idx"))
        except Exception:
            continue
        try:
            sample = ds_vqa[idx]
            image = sample["image"].convert("RGB")
            prompt = get_prompt_for_vqa(sample)
        except Exception as e:
            print(f"[WARN] Failed to load VQA sample {idx}: {e}")
            continue

        overlays = compute_attention_overlay(
            model=model,
            hf_model=hf_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image=image,
            prompt_text=prompt,
            layer_head_pairs=heads,
            rope_aware=bool(args.rope_aware),
            alpha=float(args.alpha),
            cmap=str(args.cmap),
            gamma=float(args.gamma),
            query_mode=str(args.query_mode),
        )

        for ly, hd, attn_arr, disp_img in overlays:
            title = f"VQA sample {idx} – layer_{args.layer}_feature_{args.feature} – L{ly} H{hd} (Method {args.method})"
            out_path = out_dir / f"vqa_sample{idx}_L{ly}_H{hd}.png"
            save_attention_figure(
                display_image=disp_img,
                attn_arr=attn_arr,
                out_path=out_path,
                title=title,
                question_text=prompt,
                cmap=args.cmap,
                alpha=args.alpha,
            )
        print(f"  - Processed VQA sample {idx}")

    print(f"[INFO] Saved overlays to {out_dir}")


if __name__ == "__main__":
    # Local import to avoid circular type checker complaints
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: F401
    main()


