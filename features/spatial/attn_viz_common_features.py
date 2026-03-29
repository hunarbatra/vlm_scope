#!/usr/bin/env python3
"""
Attention visualization for common features using VQA spatial samples.

This script mirrors attn_viz_spatial_features.py but sources:
- feature list and VQA spatial top samples from a common features summary JSON
- top heads from either:
  - A combined attribution JSON (per-feature attribution.method_B_top_heads)
  - An aggregated attribution JSON (global method B top heads for all features)

Example with per-feature attribution:
CUDA_VISIBLE_DEVICES=2 python features/spatial/attn_viz_common_features.py \
  --common-summary-path results/stage_4/common_features_summary_35_2_with_passed_with_relations.json \
  --combined-attrib-path results/stage6/combined_ablation_attribution.json \
  --top-k 1 \
  --samples-per-feature 5 \
  --rope-aware

Example with aggregated attribution:
CUDA_VISIBLE_DEVICES=1 python features/spatial/attn_viz_common_features.py \
  --common-summary-path results/experiments/dataset_all_features.json \
  --combined-attrib-path results/experiments/aggregated_attribution_from_all_features.json \
  --output-dir results/experiments/attention_viz_all_features \
  --top-k 2 \
  --samples-per-feature 5 \
  --rope-aware  

CUDA_VISIBLE_DEVICES=4 python features/spatial/attn_viz_common_features.py \
  --common-summary-path results/experiments/dataset_all_features.json \
  --combined-attrib-path results/experiments/attribution_summary.json \
  --output-dir results/experiments/attention_viz_all_features \
  --top-k 2 \
  --samples-per-feature 5 \
  --rope-aware  

CUDA_VISIBLE_DEVICES=4 python features/spatial/attn_viz_common_features.py \
  --common-summary-path results/experiments/dataset_all_features.json \
  --combined-attrib-path results/experiments/attribution_summary.json \
  --output-dir results/experiments/attention_viz_all_features_one_head_10 \
  --top-k 2 \
  --samples-per-feature 10 \
  --rope-aware  \
    --feature-filter layer_18_feature_29948



CUDA_VISIBLE_DEVICES=4 python features/spatial/attn_viz_common_features.py \
  --common-summary-path results/experiments/dataset_all_features.json \
  --combined-attrib-path results/experiments/attribution_summary_with_bottom_heads.json \
  --output-dir results/experiments/attention_viz_true_bottom_heads \
  --top-k 3 \
  --samples-per-feature 10 \
  --rope-aware \
  --bottom-heads \
  --feature-filter layer_20_feature_22247


CUDA_VISIBLE_DEVICES=2 python features/spatial/attn_viz_common_features.py \
  --common-summary-path results/experiments/dataset_all_features.json \
  --combined-attrib-path results/experiments/attribution_summary_with_bottom_heads.json \
  --output-dir results/experiments/attention_viz_vsr_only \
  --top-k 1 \
  --samples-per-feature 10 \
  --rope-aware \
  --dataset-only vsr \
  --feature-filter layer_15_feature_10748

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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
import requests
from io import BytesIO


def load_dataset_vqa():
    """Load the VQAv2 validation split dataset object (lazy loading)."""
    return load_dataset("lmms-lab/VQAv2", split="validation")


def load_vqa_spatial_indices(cache_dir: str = ".cache/vqa_spatial_filter") -> List[int]:
    """Load cached mapping from spatial subset indices to base VQA indices.

    Reused logic from attn_viz_spatial_features.py
    """
    candidates = []
    try:
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


def parse_common_features(common_summary_path: Path,
                         feature_filter: str,
                         samples_per_feature: int,
                         dataset_only: str = "") -> Dict[str, Dict]:
    """Parse common features JSON and extract per-feature top samples across datasets.

    Returns mapping: feature_key -> {
        'layer': int,
        'feature': int,
        'top_samples': List[{ 'dataset': str, 'sample_idx': int, 'magnitude': float, 'text': str }]
    }
    """
    with open(common_summary_path, "r") as f:
        payload = json.load(f)

    features = payload.get("features", {})
    result: Dict[str, Dict] = {}
    # Load mapping from VQA spatial subset indices to base VQA indices for deduplication
    spatial_index_map = load_vqa_spatial_indices()
    for feature_key, info in features.items():
        if feature_filter and feature_filter not in feature_key:
            continue
        layer = int(info.get("layer"))
        feat_id = int(info.get("feature"))
        ds_info = info.get("datasets", {})
        # gather from all datasets with potential base VQA index for dedup
        merged_with_base: List[Dict] = []
        for ds_name in ("vqa", "vqa_spatial", "vsr"):
            # If a specific dataset is requested, skip others
            if dataset_only and ds_name != dataset_only:
                continue
            ds_block = ds_info.get(ds_name) or {}
            for s in ds_block.get("top_samples", []) or []:
                try:
                    idx = int(s.get("sample_idx"))
                    mag = float(s.get("magnitude", 0.0))
                    text = s.get("question") if ds_name != "vsr" else s.get("caption")
                    base_vqa_idx = None
                    if ds_name == "vqa":
                        base_vqa_idx = idx
                    elif ds_name == "vqa_spatial":
                        try:
                            if spatial_index_map and 0 <= idx < len(spatial_index_map):
                                base_vqa_idx = int(spatial_index_map[idx])
                        except Exception:
                            base_vqa_idx = None
                    merged_with_base.append({
                        "dataset": ds_name,
                        "sample_idx": idx,
                        "magnitude": mag,
                        "text": text or "",
                        "_base_vqa_idx": base_vqa_idx,
                    })
                except Exception:
                    continue
        if not merged_with_base:
            continue
        # sort by magnitude across all datasets
        merged_with_base.sort(key=lambda x: x.get("magnitude", 0.0), reverse=True)
        # deduplicate by base VQA index (so vqa and vqa_spatial duplicates collapse),
        # while also avoiding duplicates within the same dataset/sample_idx
        seen_base: set = set()
        seen_pair: set = set()
        deduped: List[Dict] = []
        for rec in merged_with_base:
            ds = rec["dataset"]
            idx = rec["sample_idx"]
            base = rec.get("_base_vqa_idx")
            pair = (ds, idx)
            if base is not None:
                if base in seen_base:
                    continue
                seen_base.add(base)
            else:
                if pair in seen_pair:
                    continue
                seen_pair.add(pair)
            deduped.append({
                "dataset": ds,
                "sample_idx": idx,
                "magnitude": rec["magnitude"],
                "text": rec.get("text", ""),
            })
        # keep top N after dedup
        if samples_per_feature > 0:
            deduped = deduped[:samples_per_feature]
        result[feature_key] = {
            "layer": layer,
            "feature": feat_id,
            "top_samples": deduped,
        }
    return result


def parse_method_b_top_heads(combined_attrib_path: Path,
                             desired_top_k: int,
                             bottom: bool = False) -> Dict[str, List[Tuple[int, int]]]:
    """Parse combined attribution JSON and extract method_B_top_heads per feature key.
    
    Returns mapping: feature_key -> List[(layer, head)] limited to desired_top_k.
    If bottom=True, selects the bottom-k heads instead of the top-k.
    """
    with open(combined_attrib_path, "r") as f:
        payload = json.load(f)

    features = payload.get("features", {})
    result: Dict[str, List[Tuple[int, int]]] = {}
    for feature_key, info in features.items():
        attrib = info.get("attribution", {})
        # Support both key styles:
        # - "method_B_top_heads" (older per-feature format)
        # - "top_heads_method_B" (attribution_summary.json format)
        if bottom:
            heads = attrib.get("method_B_bottom_heads") or attrib.get("bottom_heads_method_B") or []
        else:
            heads = attrib.get("method_B_top_heads") or attrib.get("top_heads_method_B") or []
        if not heads:
            continue
        pairs: List[Tuple[int, int]] = []
        for h in heads:
            try:
                pairs.append((int(h.get("layer")), int(h.get("head"))))
            except Exception:
                continue
        if not pairs:
            continue
        if desired_top_k > 0:
            pairs = (pairs[-desired_top_k:] if bottom else pairs[:desired_top_k])
        result[feature_key] = pairs
    return result


def parse_method_a_top_heads(combined_attrib_path: Path,
                             desired_top_k: int,
                             bottom: bool = False) -> Dict[str, List[Tuple[int, int]]]:
    """Parse combined attribution JSON and extract method_A top heads per feature key.

    Supports keys "method_A_top_heads" and "top_heads_method_A".
    Returns mapping: feature_key -> List[(layer, head)]. If bottom=True, selects bottom-k.
    """
    with open(combined_attrib_path, "r") as f:
        payload = json.load(f)

    features = payload.get("features", {})
    result: Dict[str, List[Tuple[int, int]]] = {}
    for feature_key, info in features.items():
        attrib = info.get("attribution", {})
        if bottom:
            heads = attrib.get("method_A_bottom_heads") or attrib.get("bottom_heads_method_A") or []
        else:
            heads = attrib.get("method_A_top_heads") or attrib.get("top_heads_method_A") or []
        if not heads:
            continue
        pairs: List[Tuple[int, int]] = []
        for h in heads:
            try:
                pairs.append((int(h.get("layer")), int(h.get("head"))))
            except Exception:
                continue
        if not pairs:
            continue
        if desired_top_k > 0:
            pairs = (pairs[-desired_top_k:] if bottom else pairs[:desired_top_k])
        result[feature_key] = pairs
    return result


def parse_aggregated_method_b_top_heads(aggregated_attrib_path: Path,
                                       desired_top_k: int,
                                       bottom: bool = False) -> Dict[str, List[Tuple[int, int]]]:
    """Parse aggregated attribution JSON and extract method B top heads for all features.
    
    This function works with the aggregated_attribution_from_all_features.json structure
    where method B top heads are provided globally, not per individual feature.
    
    Returns mapping: feature_key -> List[(layer, head)] limited to desired_top_k.
    """
    with open(aggregated_attrib_path, "r") as f:
        payload = json.load(f)

    # Get method B top heads from the aggregated structure
    methods = payload.get("methods", {})
    method_b = methods.get("B", {})
    aggregated_heads = method_b.get("aggregated_top_heads", [])
    
    if not aggregated_heads:
        print("[WARN] No method B aggregated top heads found in attribution file")
        return {}
    
    # Parse the aggregated heads into (layer, head) tuples
    parsed_heads: List[Tuple[int, int]] = []
    for head_info in aggregated_heads:
        try:
            name = head_info.get("name", "")
            if not name.startswith("L") or "H" not in name:
                continue
            
            # Parse format like "L13H1" -> (13, 1)
            parts = name[1:].split("H")  # Remove "L" and split on "H"
            if len(parts) == 2:
                layer = int(parts[0])
                head = int(parts[1])
                parsed_heads.append((layer, head))
        except (ValueError, IndexError) as e:
            print(f"[WARN] Failed to parse head name '{head_info.get('name', '')}': {e}")
            continue
    
    # Limit to desired top_k or bottom_k
    if desired_top_k > 0:
        parsed_heads = (parsed_heads[-desired_top_k:] if bottom else parsed_heads[:desired_top_k])
    
    print(f"[INFO] Parsed {len(parsed_heads)} method B top heads from aggregated attribution")
    
    # For aggregated attribution, we return the same heads for all features
    # since we don't have per-feature attribution data
    result: Dict[str, List[Tuple[int, int]]] = {}
    
    # We'll populate this with the common heads for all features
    # The actual feature list will come from the common features summary
    return result, parsed_heads


def parse_aggregated_method_a_top_heads(aggregated_attrib_path: Path,
                                       desired_top_k: int,
                                       bottom: bool = False) -> Dict[str, List[Tuple[int, int]]]:
    """Parse aggregated attribution JSON and extract method A top heads for all features.

    Returns (empty map to fill, common_heads_list).
    """
    with open(aggregated_attrib_path, "r") as f:
        payload = json.load(f)

    methods = payload.get("methods", {})
    method_a = methods.get("A", {})
    aggregated_heads = method_a.get("aggregated_top_heads", [])

    if not aggregated_heads:
        print("[WARN] No method A aggregated top heads found in attribution file")
        return {}, []

    parsed_heads: List[Tuple[int, int]] = []
    for head_info in aggregated_heads:
        try:
            name = head_info.get("name", "")
            if not name.startswith("L") or "H" not in name:
                continue
            parts = name[1:].split("H")
            if len(parts) == 2:
                layer = int(parts[0])
                head = int(parts[1])
                parsed_heads.append((layer, head))
        except (ValueError, IndexError) as e:
            print(f"[WARN] Failed to parse head name '{head_info.get('name', '')}': {e}")
            continue

    if desired_top_k > 0:
        parsed_heads = (parsed_heads[-desired_top_k:] if bottom else parsed_heads[:desired_top_k])

    print(f"[INFO] Parsed {len(parsed_heads)} method A top heads from aggregated attribution")
    return {}, parsed_heads


def visualize_for_feature(feature_key: str,
                         per_feature: Dict,
                         top_heads: List[Tuple[int, int]],
                         method_label: str,
                         args,
                         ds_vqa,
                         ds_vsr,
                         tokenizer,
                         hf_model,
                         image_processor,
                         model,
                         manifest_pairs: Optional[Dict[str, List[Tuple[str, int]]]] = None):
    """Visualize attention overlays for one feature using top heads and selected top samples across datasets."""
    layer = per_feature["layer"]
    feat_id = per_feature["feature"]
    top_samples = per_feature.get("top_samples") or []
    # If a manifest is provided, restrict to listed (dataset, sample_idx) pairs per feature
    if manifest_pairs and feature_key in manifest_pairs:
        allowed = set((ds, int(idx)) for ds, idx in manifest_pairs[feature_key])
        filtered = []
        for rec in top_samples:
            try:
                ds = str(rec.get("dataset"))
                idx = int(rec.get("sample_idx"))
                if (ds, idx) in allowed:
                    filtered.append(rec)
            except Exception:
                continue
        top_samples = filtered

    if not top_heads:
        print(f"[WARN] No method_B_top_heads for {feature_key}; skipping")
        return

    # Output dirs
    base_out_dir = Path(args.output_dir) / feature_key
    subdir = f"attn_top_{method_label}"
    out_dir_top = base_out_dir / subdir
    ex_dir = base_out_dir / "examples"
    out_dir_top.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)

    # Model head config
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

    # Collect layers to trace
    needed_layers = sorted({ly for ly, _ in top_heads})

    # Iterate samples
    for sample_rec in top_samples:
        ds_name = str(sample_rec.get("dataset"))
        sample_idx = int(sample_rec.get("sample_idx"))
        prompt_text = str(sample_rec.get("text", "")).strip()

        # Load dataset sample and image/prompt per dataset
        try:
            if ds_name in ("vqa", "vqa_spatial"):
                if ds_vqa is None:
                    print(f"[WARN] Dataset '{ds_name}' not loaded due to --dataset-only; skipping sample {sample_idx}")
                    continue
                sample = ds_vqa[sample_idx]
                image = sample["image"].convert("RGB")
                question = str(sample.get("question", "")).strip()
                prompt = prompt_text if prompt_text else (question if question else "Answer the question.")
            elif ds_name == "vsr":
                if ds_vsr is None:
                    print(f"[WARN] Dataset 'vsr' not loaded due to --dataset-only; skipping sample {sample_idx}")
                    continue
                sample = ds_vsr[sample_idx]
                image_url = sample.get("image_link")
                caption = str(sample.get("caption", "")).strip()
                prompt = prompt_text if prompt_text else (caption if caption else "Describe the image.")
                # download the image
                try:
                    resp = requests.get(image_url, timeout=10)
                    resp.raise_for_status()
                    image = PILImage.open(BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    print(f"[WARN] Failed to fetch VSR image {image_url}: {e}")
                    image = PILImage.new("RGB", (224, 224), (128, 128, 128))
            else:
                print(f"[WARN] Unknown dataset '{ds_name}' for sample {sample_idx}; skipping")
                continue
        except Exception as e:
            print(f"[WARN] Failed to load sample {sample_idx} from {ds_name}: {e}")
            continue

        # Inputs and display image reconstruction
        input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
            image, prompt, image_processor, model._module, tokenizer
        )
        proc = image_processor(images=image, return_tensors="pt")
        pix = proc["pixel_values"][0]
        mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(image_processor.image_std).view(3, 1, 1)
        disp = (pix * std + mean).clamp(0, 1)
        disp_np = (disp.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        display_image = PILImage.fromarray(disp_np)

        img_start, img_end = get_image_token_positions(input_ids)

        # Trace and capture q/k per needed layer
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
                        # capture pre-projection outputs (same as in reference script)
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
                # materialize up to max layer
                if needed_layers:
                    _ = model.model.layers[max(needed_layers)].output[0].sum()


        # Visualize top heads (method A or B)
        for ly, hd in top_heads:
            try:
                q = q_by_layer[ly].value.detach().cpu()
                k = k_by_layer[ly].value.detach().cpu()
            except Exception:
                continue
            seq_len, hidden_dim = q.shape

            try:
                q_h = q.view(seq_len, num_heads, head_dim)
                k_h = k.view(seq_len, num_kv_heads, head_dim)
            except Exception:
                continue

            # Apply RoPE if requested
            if args.rope_aware:
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
                    q_h = apply_rotary_pos_emb(q_h, cos_q, sin_q)
                    k_h = apply_rotary_pos_emb(k_h, cos_k, sin_k)
                except Exception:
                    pass

            # Choose query representation
            if args.query_mode == "last":
                q_vec = q_h[seq_len - 1, hd]
            elif args.query_mode == "first_answer":
                first_tok = int(img_end)
                first_tok = min(max(0, first_tok), seq_len - 1)
                q_vec = q_h[first_tok, hd]
            else:  # avg_answer
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

            # Normalize and gamma
            attn_map = (attn_map - attn_map.min()) / (attn_map.ptp() + 1e-8)
            if args.gamma and args.gamma != 1.0:
                attn_map = np.power(attn_map, args.gamma)

            # Resize to image
            attn_img = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
                display_image.size, resample=PILImage.BILINEAR
            )
            attn_arr = np.asarray(attn_img).astype(np.float32) / 255.0

            # Style
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
            ax.imshow(attn_arr, cmap=args.cmap, alpha=args.alpha, interpolation="bilinear", vmin=0.0, vmax=1.0)
            ax.axis("off")

            # Titles
            title_bits = [
                f"{ds_name.upper()} Sample {sample_idx} – {feature_key} – L{ly} H{hd}",
                f"{'Bottom' if getattr(args, 'bottom_heads', False) else 'Top'} (Method {method_label})"
            ]
            if prompt:
                question_text = ("Q: " if ds_name != "vsr" else "Text: ") + (prompt[:200] + ("…" if len(prompt) > 200 else ""))
                ax.set_title(question_text, fontsize=22, fontweight='bold', pad=20, wrap=True)
            else:
                ax.set_title(" | ".join(title_bits), fontsize=14, fontweight='bold', pad=20)

            fig.subplots_adjust(bottom=0.12, top=0.90)
            fig.text(0.5, 0.12, " | ".join(title_bits), ha="center", va="bottom",
                    fontsize=11, fontweight='bold', wrap=True)

            out_path = out_dir_top / f"{ds_name}_sample{sample_idx}_L{ly}_H{hd}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            
            # Also save clean version in examples folder (no text, just attention overlay)
            fig_clean, ax_clean = plt.subplots(figsize=(8, 8))
            ax_clean.imshow(display_image)
            ax_clean.imshow(attn_arr, cmap=args.cmap, alpha=args.alpha, interpolation="bilinear", vmin=0.0, vmax=1.0)
            ax_clean.axis("off")
            
            ex_path = ex_dir / f"{ds_name}_sample{sample_idx}_L{ly}_H{hd}.png"
            plt.savefig(ex_path, dpi=300, bbox_inches='tight')
            plt.close(fig_clean)
            
            plt.close(fig)

        print(f"  - Processed {ds_name} sample {sample_idx} for {feature_key}")


def main():
    parser = argparse.ArgumentParser(description="Attention visualization for common features using VQA spatial/VSR samples")
    parser.add_argument("--common-summary-path", type=str, required=True,
                        help="Path to common_features_summary_*.json")
    parser.add_argument("--combined-attrib-path", type=str, required=True,
                        help="Path to attribution JSON file (either combined_ablation_attribution.json or aggregated_attribution_from_all_features.json)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of heads to visualize per feature (top-k by default)")
    parser.add_argument("--bottom-heads", action="store_true",
                        help="Select bottom-k heads instead of top-k")
    parser.add_argument("--samples-per-feature", type=int, default=10,
                        help="Number of VQA spatial top samples per feature to visualize (0 = all available)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay transparency (0..1)")
    parser.add_argument("--cmap", type=str, default="magma",
                        help="Matplotlib colormap for the heatmap")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma for attention contrast (1.0 disables)")
    parser.add_argument("--feature-filter", type=str, default="",
                        help="Optional filter for specific feature keys (e.g., 'layer_26_feature_807')")
    parser.add_argument("--dataset-only", type=str, default="",
                        choices=["", "vqa", "vqa_spatial", "vsr"],
                        help="Restrict visualization to a single dataset (vqa, vqa_spatial, or vsr)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--query-mode", type=str, choices=["last", "first_answer", "avg_answer"], default="last",
                        help="Which query token(s) to use")
    parser.add_argument("--rope-aware", action="store_true",
                        help="Attempt to capture post-RoPE Q/K if available; otherwise use pre-RoPE proxy")
    parser.add_argument("--output-dir", type=str, default="results/common_attn_viz",
                        help="Base output directory for visualizations")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Optional JSON manifest mapping feature_key -> list of {dataset, sample_idx} to restrict overlays")
    args = parser.parse_args()

    common_summary_path = Path(args.common_summary_path)
    combined_attrib_path = Path(args.combined_attrib_path)
    if not common_summary_path.exists():
        print(f"[ERROR] Common summary not found: {common_summary_path}")
        return
    if not combined_attrib_path.exists():
        print(f"[ERROR] Combined attribution not found: {combined_attrib_path}")
        return

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

    # Parse inputs
    per_feature_map = parse_common_features(
        common_summary_path,
        args.feature_filter,
        args.samples_per_feature,
        dataset_only=args.dataset_only,
    )
    # Load optional manifest
    manifest_pairs: Optional[Dict[str, List[Tuple[str, int]]]] = None
    if args.manifest_path:
        try:
            mp = json.loads(Path(args.manifest_path).read_text())
            manifest_pairs = {}
            for fkey, items in (mp or {}).items():
                pairs: List[Tuple[str, int]] = []
                for it in (items or []):
                    try:
                        pairs.append((str(it.get("dataset")), int(it.get("sample_idx"))))
                    except Exception:
                        continue
                if pairs:
                    manifest_pairs[fkey] = pairs
        except Exception as e:
            print(f"[WARN] Failed to read manifest {args.manifest_path}: {e}")
            manifest_pairs = None
    if not per_feature_map:
        print(f"[ERROR] No features found after filtering in {common_summary_path}")
        return
    
    # Check if this is an aggregated attribution file or per-feature attribution file
    with open(combined_attrib_path, "r") as f:
        test_payload = json.load(f)
    
    if "methods" in test_payload and "B" in test_payload["methods"]:
        # This is an aggregated attribution file
        print("[INFO] Using aggregated attribution file structure")
        heads_map_B, common_heads_B = parse_aggregated_method_b_top_heads(combined_attrib_path, args.top_k, bottom=args.bottom_heads)
        heads_map_A, common_heads_A = parse_aggregated_method_a_top_heads(combined_attrib_path, args.top_k, bottom=args.bottom_heads)
        # Assign the same common heads to all features
        for feature_key in per_feature_map.keys():
            heads_map_B[feature_key] = common_heads_B
            heads_map_A[feature_key] = common_heads_A
    else:
        # This is a per-feature attribution file
        print("[INFO] Using per-feature attribution file structure")
        heads_map_B = parse_method_b_top_heads(combined_attrib_path, args.top_k, bottom=args.bottom_heads)
        heads_map_A = parse_method_a_top_heads(combined_attrib_path, args.top_k, bottom=args.bottom_heads)

    # Load mapping from spatial subset to base indices
    spatial_index_map = load_vqa_spatial_indices()
    if not spatial_index_map:
        print("[WARN] Could not find VQA spatial index mapping; samples may be misaligned if indices are subset-relative")

    # Initialize model and dataset once
    print("[INFO] Initializing model and dataset...")
    tokenizer, hf_model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    model = NNsight(hf_model)
    # Conditionally load datasets based on filter (load fewer resources when possible)
    ds_vqa = load_dataset_vqa() if (not args.dataset_only or args.dataset_only in ("vqa", "vqa_spatial")) else None
    # Load VSR dataset (all samples, not just label=1) only if needed
    try:
        if not args.dataset_only or args.dataset_only == "vsr":
            data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
            ds_vsr = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="train")
        else:
            ds_vsr = None
    except Exception:
        ds_vsr = None

    # Iterate features
    feature_keys = sorted(per_feature_map.keys())
    print(f"[INFO] Found {len(feature_keys)} features to process")

    for i, fkey in enumerate(feature_keys):
        print(f"\n[{i+1}/{len(feature_keys)}] Processing {fkey}")
        top_heads_B = heads_map_B.get(fkey) or []
        top_heads_A = heads_map_A.get(fkey) or []
        if not top_heads_B and not top_heads_A:
            # Silently skip features that don't exist in attribution file
            continue
        try:
            if top_heads_B:
                visualize_for_feature(
                    feature_key=fkey,
                    per_feature=per_feature_map[fkey],
                    top_heads=top_heads_B,
                    method_label="B",
                    args=args,
                    ds_vqa=ds_vqa,
                    ds_vsr=ds_vsr,
                    tokenizer=tokenizer,
                    hf_model=hf_model,
                    image_processor=image_processor,
                    model=model,
                    manifest_pairs=manifest_pairs,
                )
            if top_heads_A:
                visualize_for_feature(
                    feature_key=fkey,
                    per_feature=per_feature_map[fkey],
                    top_heads=top_heads_A,
                    method_label="A",
                    args=args,
                    ds_vqa=ds_vqa,
                    ds_vsr=ds_vsr,
                    tokenizer=tokenizer,
                    hf_model=hf_model,
                    image_processor=image_processor,
                    model=model,
                    manifest_pairs=manifest_pairs,
                )
        except Exception as e:
            print(f"[ERROR] Failed to process {fkey}: {e}")
            continue

    print(f"\n[INFO] Completed processing {len(feature_keys)} features")


if __name__ == "__main__":
    main()


