#!/usr/bin/env python3
"""
Quick probe to test whether SAE features are visual (image-driven) vs text-driven on VQA spatial samples.

This script reads feature samples from JSON files (like auto_interp_spatial_features.py) instead of PT files.

Example:
CUDA_VISIBLE_DEVICES=1 python features/spatial/probe_vqa_spatial_features.py \
  --vqa-samples-dir results/stage_4/feature_samples/vqa_all_spatial \
  --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k \
  --method text-only \
  --layer 6 \
  --feature 26958 \
  --samples-per-feature 10
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import re

import torch
from datasets import load_dataset
from PIL import Image

import sys
sys.path.append(".")
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    get_image_token_positions,
)
from utils.datasets import load_vqa
from finetune.vqa.utils import initialize_sae
from nnsight import NNsight


# GENERIC_PROMPTS = [

#     "Describe containment and overlap: inside, surrounding, overlapping.",
#     "Describe alignment and ordering: row, column, stacked.",
#     "Explain occlusion and depth order: which items are in front or behind.",
#     "Describe object orientations and facing directions.",
#     "Describe nearest neighbors and relative distances between objects.",
#     "Identify adjacency: which items touch or share a boundary.",
#     "Describe spacing and gaps between objects.",
#     "Describe clusters and separated groups of objects.",
#     "Identify the central object and those on the periphery.",
#     "Use compass directions to describe positions.",
#     "Describe relative sizes and scale with respect to nearby objects.",
#     "Describe the layout of paths or lines connecting objects.",
#     "Summarize symmetry or asymmetry in the arrangement.",
#     "Outline the overall scene structure in terms of object placement.",
#     "Describe how objects are distributed across the scene.",
#     "Identify clusters versus isolated items and comment on their spacing.",
#     "Characterize the arrangement pattern.",
# ]



GENERIC_PROMPTS = [
    "Describe how the items are arranged.",
    "Comment on the overall layout and organization of the scene.",
    "Summarize the structure in terms of grouping or separation.",
    "Explain the relative positioning of objects without naming directions.",
    "Describe patterns of arrangement, such as order or symmetry."
]



def read_feature_samples_json(samples_dir: Path, layer: int, feature: int, method: str = "text-only") -> List[Dict]:
    """Read feature samples from JSON file in the new directory structure."""
    feature_dir = samples_dir / f"{method}_layer_{layer}_feature_{feature}"
    if not feature_dir.exists():
        print(f"[WARN] Feature directory not found: {feature_dir}")
        return []
    
    sample_file = feature_dir / "sample_info.json"
    if not sample_file.exists():
        print(f"[WARN] Sample info file not found: {sample_file}")
        return []
    
    try:
        with open(sample_file, "r") as f:
            samples = json.load(f)
        print(f"[INFO] Loaded {len(samples)} samples for {method}_L{layer}/F{feature}")
        return samples
    except Exception as e:
        print(f"[ERROR] Failed to load {sample_file}: {e}")
        return []


@torch.no_grad()
def feature_activation_for_prompt(
    vlm_tokenizer,
    vlm_model: NNsight,
    vlm_image_processor,
    sae,
    image: Image.Image,
    prompt: str,
    layer_idx: int,
    target_feature: int,
) -> Tuple[float, float, float, bool, bool]:
    """Returns (max_all, max_img, max_txt, fired_any, fired_in_img)."""
    input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
        image, prompt, vlm_image_processor, vlm_model, vlm_tokenizer
    )

    with vlm_model.trace(
        input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        image_sizes=image_sizes,
    ) as tr:
        layer_out = vlm_model.model.layers[layer_idx].output[0][:, 1:].detach().cpu().save()

    acts = layer_out.squeeze(0)  # (seq, hidden)
    img_start, img_end = get_image_token_positions(input_ids)
    # Adjust for removed BOS (we sliced [:, 1:]) to align with activations
    if img_start is not None and img_end is not None:
        img_start = int(img_start) - 1
        img_end = int(img_end) - 1
    # Skip samples with unknown image span
    if img_start is None or img_end is None:
        print("[WARN] No image tokens found; skipping sample.")
        return 0.0, 0.0, 0.0, False, False

    feature_acts = sae.encode(acts.unsqueeze(0).to(sae.device)).squeeze(0).cpu()  # (seq, feats)
    f = target_feature
    seq_vals = feature_acts[:, f]
    max_all = float(torch.max(seq_vals).item())

    img_slice = seq_vals[img_start : img_end + 1] if img_end >= img_start else torch.tensor([])
    txt_prefix = seq_vals[:img_start] if img_start > 0 else torch.tensor([])
    txt_suffix = seq_vals[img_end + 1 :] if img_end + 1 < seq_vals.shape[0] else torch.tensor([])
    txt_slice = torch.cat([txt_prefix, txt_suffix]) if txt_prefix.numel() + txt_suffix.numel() > 0 else torch.tensor([])

    max_img = float(img_slice.max().item()) if img_slice.numel() > 0 else 0.0
    max_txt = float(txt_slice.max().item()) if txt_slice.numel() > 0 else 0.0

    fired_any = max_all > 0
    fired_in_img = max_img > 0
    return max_all, max_img, max_txt, fired_any, fired_in_img


def main():
    parser = argparse.ArgumentParser(description="Probe whether SAE features are image-driven on VQA spatial samples")
    parser.add_argument("--vqa-samples-dir", type=str, required=True, 
                        help="Directory containing feature samples (e.g., results/stage_4/feature_samples/vqa_all_spatial)")
    parser.add_argument("--sae-checkpoint-dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="pretrained")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature", type=int, default=None)
    parser.add_argument("--samples-per-feature", type=int, default=5)
    parser.add_argument("--num-random-features", type=int, default=5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vqa-split", type=str, default="validation", choices=["train", "validation", "test"])
    
    args = parser.parse_args()
    random.seed(args.seed)

    # Load VQA dataset using unified loader
    vqa_dataset = load_vqa(args.vqa_split)
    
    # Load feature samples directory
    samples_dir = Path(args.vqa_samples_dir)
    if not samples_dir.exists():
        raise RuntimeError(f"VQA samples directory not found: {samples_dir}")

    # Choose features to probe
    if args.feature is not None:
        feature_ids = [args.feature]
    else:
        # Find all available features in the layer
        feature_dirs = list(samples_dir.glob(f"{args.method}_layer_{args.layer}_feature_*"))
        if not feature_dirs:
            raise RuntimeError(f"No feature directories found for {args.method}_layer_{args.layer}")
        
        feature_ids = []
        for d in feature_dirs:
            try:
                # Extract feature ID from directory name like "text-only_layer_0_feature_454"
                feature_id = int(d.name.split("_feature_")[1])
                feature_ids.append(feature_id)
            except:
                continue
        
        if len(feature_ids) == 0:
            raise RuntimeError("No valid feature IDs found")
        
        random.shuffle(feature_ids)
        feature_ids = feature_ids[:args.num_random_features]
        print(f"[INFO] Randomly selected features: {feature_ids}")

    # Init model + SAE
    vlm_tokenizer, base_model, vlm_image_processor = initialize_vlm_model("llava-more")
    vlm_model = NNsight(base_model)

    ckpt_path = Path(args.sae_checkpoint_dir)
    sae_path = ckpt_path / args.method / f"{args.method}_layer_{args.layer}.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")
    sae = initialize_sae(layer_idx=args.layer, checkpoint_path=sae_path, device="cuda")

    # Track features that pass the generic test
    passing_features = []
    
    for feat_id in feature_ids:
        # Load samples for this feature
        samples = read_feature_samples_json(samples_dir, args.layer, feat_id, args.method)
        if not samples:
            continue
        
        # Sort by magnitude and take top samples
        samples.sort(key=lambda x: x.get("magnitude", 0), reverse=True)
        top_samples = samples[:args.samples_per_feature]

        # Track if this feature passes the generic test
        feature_passed_generic = True  # Start assuming it passes, will be set to False if any sample fails
        feature_results = []
        activation_threshold = 0.01  # Minimum activation threshold

        for sample_info in top_samples:
            vqa_idx = sample_info["sample_idx"]
            magnitude = sample_info.get("magnitude", 0)
            rank = sample_info.get("rank", 0)
            
            try:
                # Load VQA sample
                sample = vqa_dataset[vqa_idx]
                image = sample["image"].convert("RGB")
                original_prompt = sample["question"].strip()
                
                print(f"[INFO] Processing L{args.layer} F{feat_id} sample {vqa_idx} (rank {rank}, mag {magnitude:.3f})")
                
                # Test with original VQA prompt
                max_all_a, max_img_a, max_txt_a, fired_a, fired_img_a = feature_activation_for_prompt(
                    vlm_tokenizer, vlm_model, vlm_image_processor, sae, 
                    image, original_prompt, args.layer, feat_id
                )

                # Test with multiple generic prompts and take the best result
                best_generic_max_all = 0.0
                best_generic_max_img = 0.0
                best_generic_max_txt = 0.0
                best_generic_fired = False
                best_generic_fired_img = False
                
                for generic_prompt in GENERIC_PROMPTS:
                    max_all_b, max_img_b, max_txt_b, fired_b, fired_img_b = feature_activation_for_prompt(
                        vlm_tokenizer, vlm_model, vlm_image_processor, sae, 
                        image, generic_prompt, args.layer, feat_id
                    )
                    
                    # Keep the best generic prompt result
                    if max_all_b > best_generic_max_all:
                        best_generic_max_all = max_all_b
                        best_generic_max_img = max_img_b
                        best_generic_max_txt = max_txt_b
                        best_generic_fired = fired_b
                        best_generic_fired_img = fired_img_b

                # Check if this sample passes the threshold - if not, mark feature as failed
                if best_generic_max_all <= activation_threshold:
                    feature_passed_generic = False

                feature_results.append({
                    "layer": args.layer,
                    "feature": feat_id,
                    "sample_idx": int(vqa_idx),
                    "magnitude": magnitude,
                    "rank": rank,
                    "orig_max_all": max_all_a,
                    "orig_max_img": max_img_a,
                    "orig_max_txt": max_txt_a,
                    "orig_fired_any": fired_a,
                    "orig_fired_in_img": fired_img_a,
                    "generic_max_all": best_generic_max_all,
                    "generic_max_img": best_generic_max_img,
                    "generic_max_txt": best_generic_max_txt,
                    "generic_fired_any": best_generic_fired,
                    "generic_fired_in_img": best_generic_fired_img,
                    "question": original_prompt,
                })

                print(
                    f"[L{args.layer} F{feat_id}] sample {vqa_idx} | "
                    f"orig(max_all={max_all_a:.3f}, img={max_img_a:.3f}, txt={max_txt_a:.3f}) | "
                    f"best_generic(max_all={best_generic_max_all:.3f}, img={best_generic_max_img:.3f}, txt={best_generic_max_txt:.3f})"
                )
                
            except Exception as e:
                print(f"[ERROR] Failed to process sample {vqa_idx}: {e}")
                continue

        # Only add features that pass the generic test
        if feature_passed_generic:
            passing_features.extend(feature_results)
            print(f"[INFO] Feature {feat_id} PASSED generic test - will be included in output")
        else:
            print(f"[INFO] Feature {feat_id} FAILED generic test - will be excluded from output")

    # Save results as CSV (only features that passed generic test)
    if passing_features:
        out_path = samples_dir / f"visual_probe_layer{args.layer}_{args.method}_generic_passed.csv"
        
        # Define CSV headers
        fieldnames = [
            "layer", "feature", "sample_idx", "magnitude", "rank",
            "orig_max_all", "orig_max_img", "orig_max_txt", "orig_fired_any", "orig_fired_in_img",
            "generic_max_all", "generic_max_img", "generic_max_txt", "generic_fired_any", "generic_fired_in_img",
            "question"
        ]
        
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(passing_features)
        
        print(f"[INFO] Saved {len(passing_features)} results to {out_path}")
        print(f"[INFO] {len(set(r['feature'] for r in passing_features))} features passed the generic test")
    else:
        print("[INFO] No features passed the generic test - no output file created")


if __name__ == "__main__":
    main()
