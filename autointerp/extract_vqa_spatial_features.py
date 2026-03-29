#!/usr/bin/env python3
"""
Extract top samples for VQA SAE features from spatial questions only.
This script filters VQAv2 to spatial questions and extracts top samples for features.

Usage:
    CUDA_VISIBLE_DEVICES=6 python autointerp/extract_vqa_spatial_features.py \
        --features-csv results/stage_3/spatial/spatial_features_vqa.csv \
        --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/ \
        --output-dir results/stage_4/feature_samples/vqa_spatial_all_spatial \
        --max-samples-per-feature None  # Use None to extract all samples

CUDA_VISIBLE_DEVICES=7 python autointerp/extract_vqa_spatial_features.py \
    --features-csv results/stage_3/spatial/spatial_features_vqa.csv \
    --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/ \
    --output-dir results/stage_4/feature_samples/full/vqa_spatial_all_spatial \
    --methods pretrained \
    --max-samples-per-feature None
"""

import os
import csv
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import gc
import re
import hashlib

# Add to path for imports
import sys
sys.path.append("finetune/vqa")
from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions, initialize_sae

# Constants
NUM_VIS_TOKENS = 575

def _default_spatial_keywords() -> List[str]:
    return [
        # Basic directions and movement
        "left", "right", "front", "back", "ahead", "behind", "forward", "backward",
        "forwards", "backwards", "up", "down", "upward", "downward",

        # Corners, sides, and extremes
        "top", "bottom", "upper", "lower", "leftmost", "rightmost", "topmost", "bottommost",
        "uppermost", "lowermost", "corner", "edge", "border", "side",
        "left side", "right side", "top side", "bottom side",

        # Multi-axis quadrant phrases (hyphen or space variants handled by regex)
        "top left", "top right", "bottom left", "bottom right",
        "upper left", "upper right", "lower left", "lower right",
        "middle left", "middle right", "center left", "center right",

        # Relative spatial relations
        "above", "over", "overhead", "atop", "on top", "on top of",
        "below", "under", "underneath", "beneath",
        "in front", "in front of", "at the front", "at the back",
        "next to", "beside", "alongside", "near", "nearby", "close to",
        "adjacent", "adjacent to", "across from", "opposite", "opposite to", "facing",
        "around", "surrounding", "encircling", "between", "in between", "among", "amid",
        "inside", "inside of", "outside", "outside of", "within",
        "to the left", "to the right", "to the left of", "to the right of",

        # Distance and extent
        "distance", "closer", "closest", "nearest", "nearer",
        "far", "farther", "farthest", "further", "furthest",
        "height", "width",

        # Orientation and axes
        "vertical", "horizontal", "diagonal", "direction", "oriented", "orientation",
        "rotated", "rotation",

        # Compass directions
        "north", "south", "east", "west",
        "north east", "north west", "south east", "south west",
        "northeast", "northwest", "southeast", "southwest",

        # Locative cues
        "position", "positioned", "located", "location", "placement", "placed",

        # Foreground/background
        "foreground", "background", "frontmost", "backmost", "background of",
    ]

def _compile_keywords_regex(keywords: List[str]) -> re.Pattern:
    """Compile a case-insensitive regex that matches any keyword as a word-ish token."""
    escaped_variants = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        # allow flexible whitespace or hyphen between words in multi-word phrases
        parts = [re.escape(p) for p in kw.split()]
        if len(parts) == 1:
            pattern = parts[0]
        else:
            joiner = r"(?:\s+|-)"
            pattern = joiner.join(parts)
        # use word boundaries on both ends when feasible
        escaped_variants.append(rf"\b{pattern}\b")
    combined = "|".join(escaped_variants)
    if not combined:
        # fallback to a pattern that matches nothing
        combined = r"a^"
    return re.compile(combined, flags=re.IGNORECASE)

def _normalize_keywords(keywords: List[str]) -> List[str]:
    return [k.strip().lower() for k in keywords if k and k.strip()]

def _keywords_hash(split: str, keywords: List[str]) -> str:
    norm = _normalize_keywords(keywords)
    key = f"{split}::" + "||".join(sorted(norm))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

class SpatialVQADataset:
    """VQAv2 validation subset filtered to spatial questions by keyword/regex."""

    def __init__(
        self,
        split: str = "validation",
        keywords_file: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        from datasets import load_dataset
        self._base = load_dataset("lmms-lab/VQAv2", split=split)

        if keywords_file is not None and os.path.exists(keywords_file):
            with open(keywords_file, "r") as f:
                file_keywords = [line.strip() for line in f.readlines()]
            keywords_list = file_keywords
        elif keywords is not None and len(keywords) > 0:
            keywords_list = keywords
        else:
            keywords_list = _default_spatial_keywords()

        keywords_norm = _normalize_keywords(keywords_list)

        # Try cache first
        filtered_indices: List[int] | None = None
        cache_used = False
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            fname = f"indices_{split}_{_keywords_hash(split, keywords_norm)}.json"
            cache_file = cache_path / fname
            if cache_file.exists():
                try:
                    payload = json.loads(cache_file.read_text())
                    if payload.get("split") == split:
                        filtered_indices = list(map(int, payload.get("indices", [])))
                        cache_used = True
                        print(f"[INFO] Loaded VQA filter indices from cache → {cache_file}")
                except Exception as e:
                    print(f"[WARN] Failed to read cache at {cache_file}: {e}")

        if filtered_indices is None:
            pattern = _compile_keywords_regex(keywords_norm)
            tmp_indices: List[int] = []
            for idx in range(len(self._base)):
                q = str(self._base[idx]["question"])  # robust to unexpected types
                if pattern.search(q):
                    tmp_indices.append(idx)
            filtered_indices = tmp_indices

            if cache_dir:
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                fname = f"indices_{split}_{_keywords_hash(split, keywords_norm)}.json"
                cache_file = cache_path / fname
                payload = {
                    "split": split,
                    "keywords": keywords_norm,
                    "count": len(filtered_indices),
                    "indices": filtered_indices,
                }
                try:
                    cache_file.write_text(json.dumps(payload, indent=2))
                    print(f"[INFO] Saved VQA filter indices to cache → {cache_file}")
                except Exception as e:
                    print(f"[WARN] Failed to write cache at {cache_file}: {e}")

        self._indices = filtered_indices
        src = "cache" if cache_used else "fresh filter"
        print(
            f"[INFO] SpatialVQADataset: kept {len(self._indices)} of {len(self._base)} questions ({src})"
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        base_idx = self._indices[idx]
        sample = self._base[base_idx]
        image = sample["image"].convert("RGB")
        prompt = sample["question"]
        return image, prompt

class VQASpatialFeatureExtractor:
    def __init__(self, sae_checkpoint_dir: str, methods: List[str], device: str = "cuda"):
        self.sae_checkpoint_dir = Path(sae_checkpoint_dir)
        self.methods = methods
        self.device = device
        
        # Load SAEs
        self.saes = self._load_saes()
        
        # Initialize VLM model
        print("[INFO] Initializing VLM model...")
        self.tokenizer, self.model, self.image_processor = initialize_vlm_model("llava-more")
        self.model = self.model.to(device)
        
        # Wrap with NNsight for tracing
        from nnsight import NNsight
        self.model = NNsight(self.model)
        
        # Ensure model is on the correct device after NNsight wrapping
        self.model = self.model.to(device)
        
    def _load_saes(self) -> Dict[str, Dict[int, any]]:
        """Load SAEs for specified methods and layers."""
        saes = {}
        for method in self.methods:
            saes[method] = {}
            method_dir = self.sae_checkpoint_dir / method
            if method_dir.exists():
                for checkpoint_path in method_dir.glob(f"{method}_layer_*.pt"):
                    # Extract layer number from filename
                    layer_idx = int(checkpoint_path.stem.split("_")[-1])
                    saes[method][layer_idx] = initialize_sae(
                        layer_idx=layer_idx, 
                        checkpoint_path=checkpoint_path, 
                        device=self.device
                    )
                    print(f"[INFO] Loaded {method} SAE for layer {layer_idx}")
            else:
                print(f"[WARN] Method directory not found: {method_dir}")
        return saes
    
    def load_vqa_dataset(self, keywords_file: Optional[str] = None, cache_dir: Optional[str] = None):
        """Load spatial VQA validation dataset."""
        self.dataset = SpatialVQADataset(
            split="validation",
            keywords_file=keywords_file,
            cache_dir=cache_dir
        )
        print(f"[INFO] Loaded {len(self.dataset)} spatial VQA validation samples")
    
    def get_sample_data(self, idx: int) -> Tuple[any, str]:
        """Get image and question for a VQA sample."""
        image, question = self.dataset[idx]
        return image, question
    
    def extract_feature_samples(self, target_features: List[Tuple[str, int, int]], 
                              max_samples_per_feature: int = 10,
                              batch_size: int = 8) -> Dict:
        """
        Extract top samples for target features with correct text-only magnitude calculations.
        Uses same efficient batching as extract_common_features.py
        
        Args:
            target_features: List of (method, layer, feature_idx) tuples
            max_samples_per_feature: Maximum samples to extract per feature
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with feature samples and their magnitudes
        """
        results = {}
        
        # Group features by method and layer for efficient processing
        feature_groups = {}
        for method, layer, feat_idx in target_features:
            if method not in feature_groups:
                feature_groups[method] = {}
            if layer not in feature_groups[method]:
                feature_groups[method][layer] = []
            feature_groups[method][layer].append(feat_idx)
        
        # Process all layers together in single forward passes (like extract_common_features.py)
        for method in feature_groups.keys():
            if method not in self.saes:
                continue
                
            results[method] = {}
            layers_to_do = [l for l in feature_groups[method].keys() if l in self.saes[method]]
            
            # Process dataset in batches (limit to first 50k samples)
            max_samples = 50000
            for start_idx in tqdm(range(0, min(len(self.dataset), max_samples), batch_size), 
                                desc=f"Processing {method} batches"):
                end_idx = min(start_idx + batch_size, min(len(self.dataset), max_samples))
                
                # Process batch for all layers at once
                batch_results = self._process_batch_all_layers(
                    start_idx, end_idx, method, layers_to_do, feature_groups[method]
                )
                
                # Merge results
                for layer in layers_to_do:
                    if layer not in results[method]:
                        results[method][layer] = {}
                    
                    for feat_idx in feature_groups[method][layer]:
                        if feat_idx not in results[method][layer]:
                            results[method][layer][feat_idx] = []
                        
                        if layer in batch_results and feat_idx in batch_results[layer]:
                            results[method][layer][feat_idx].extend(batch_results[layer][feat_idx])
            
            # Keep only top samples per feature (or all if max_samples_per_feature is None)
            for layer in results[method]:
                for feat_idx in results[method][layer]:
                    if feat_idx in results[method][layer]:
                        # Sort by magnitude
                        sorted_samples = sorted(
                            results[method][layer][feat_idx], 
                            key=lambda x: x['magnitude'], 
                            reverse=True
                        )
                        # If max_samples_per_feature is None, keep all samples
                        if max_samples_per_feature is not None:
                            results[method][layer][feat_idx] = sorted_samples[:max_samples_per_feature]
                        else:
                            results[method][layer][feat_idx] = sorted_samples
        
        return results
    
    def _process_batch_all_layers(self, start_idx: int, end_idx: int, method: str, 
                                 layers_to_do: List[int], feature_groups: Dict) -> Dict:
        """Process a batch of samples for all layers in a single forward pass."""
        batch_results = {layer: {feat_idx: [] for feat_idx in feature_groups[layer]} for layer in layers_to_do}
        
        # Prepare batch inputs
        batch_input_ids = []
        batch_attention_mask = []
        batch_image_tensors = []
        batch_image_sizes = []
        batch_sample_indices = []
        img_positions = []
        
        for j in range(start_idx, end_idx):
            image, question = self.get_sample_data(j)
            input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
                image, question, self.image_processor, self.model, self.tokenizer
            )
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_image_tensors.append(image_tensor)
            batch_image_sizes.append(image_sizes[0])
            batch_sample_indices.append(j)
            
            img_start, img_end = get_image_token_positions(input_ids)
            img_positions.append((img_start - 1, img_end - 1))
            
            del input_ids, attention_mask, image_tensor
        
        # Pad and move to device
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            [ids.squeeze(0) for ids in batch_input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [mask.squeeze(0) for mask in batch_attention_mask],
            batch_first=True,
            padding_value=0,
        )
        batch_image_tensors = torch.cat(batch_image_tensors, dim=0)
        
        # Ensure all tensors are on the same device as the model
        model_device = next(self.model.parameters()).device
        batch_input_ids = batch_input_ids.to(model_device)
        batch_attention_mask = batch_attention_mask.to(model_device)
        batch_image_tensors = batch_image_tensors.to(model_device)
        
        # Get activations for all layers in single forward pass (like extract_common_features.py)
        with torch.no_grad():
            with self.model.trace(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                images=batch_image_tensors,
                image_sizes=batch_image_sizes,
            ):
                # Save activations for all layers (deferred computation)
                layer_outputs = []
                for layer in layers_to_do:
                    layer_outputs.append(self.model.model.layers[layer].output[0][:, 1:].detach().save())
        
        # Process each layer's activations (outside trace context)
        for i, layer in enumerate(layers_to_do):
            layer_activations = layer_outputs[i]
            sae = self.saes[method][layer]
            target_feats = feature_groups[layer]
            
            # Process each sample
            for sample_idx in range(layer_activations.shape[0]):
                img_start, img_end = img_positions[sample_idx]
                # Sum text tokens (excluding BOS) and dynamic image token length
                text_token_count = int(batch_attention_mask[sample_idx, 1:].sum().item())
                img_len = 0
                if img_start is not None and img_end is not None and img_end >= img_start:
                    img_len = int(img_end - img_start + 1)
                actual_seq_len = text_token_count + img_len
                # Clamp to available activation width
                avail_len = int(layer_activations.shape[1])
                actual_seq_len = max(0, min(actual_seq_len, avail_len))

                activations = layer_activations[sample_idx, :actual_seq_len]
                
                # Create method-specific mask
                if method.lower() == "text-only":
                    mask = torch.ones(actual_seq_len, dtype=torch.bool, device=self.device)
                    if img_start is not None and img_end is not None:
                        clamped_start = max(0, int(img_start))
                        clamped_end = min(actual_seq_len - 1, int(img_end)) if actual_seq_len > 0 else -1
                        if clamped_end >= clamped_start and clamped_end >= 0:
                            mask[clamped_start:clamped_end + 1] = False  # Zero out image tokens
                elif method.lower() == "image-only":
                    mask = torch.zeros(actual_seq_len, dtype=torch.bool, device=self.device)
                    if img_start is not None and img_end is not None:
                        clamped_start = max(0, int(img_start))
                        clamped_end = min(actual_seq_len - 1, int(img_end)) if actual_seq_len > 0 else -1
                        if clamped_end >= clamped_start and clamped_end >= 0:
                            mask[clamped_start:clamped_end + 1] = True  # Only image tokens
                else:
                    mask = torch.ones(actual_seq_len, dtype=torch.bool, device=self.device)
                
                if mask.sum() == 0:
                    continue
                
                # Apply SAE to masked activations
                masked_activations = activations * mask.unsqueeze(-1)
                feature_acts = sae.encode(masked_activations.unsqueeze(0)).squeeze(0)  # (seq, feats)
                
                # Get magnitudes for target features (ONLY over masked tokens)
                for feat_idx in target_feats:
                    if feat_idx < feature_acts.shape[1]:
                        # STRICT text-only max (safer than multiply-zero approach)
                        masked = feature_acts[:, feat_idx].masked_fill(~mask, float("-inf"))
                        max_magnitude = masked.max().item()
                        if np.isneginf(max_magnitude) or max_magnitude <= 0:
                            continue
                        
                        batch_results[layer][feat_idx].append({
                            'sample_idx': batch_sample_indices[sample_idx],
                            'magnitude': max_magnitude,
                            'masked_token_count': mask.sum().item(),
                            'total_token_count': actual_seq_len
                        })
        
        # Clean up
        del batch_input_ids, batch_attention_mask, batch_image_tensors, layer_activations
        torch.cuda.empty_cache()
        
        return batch_results


def load_target_features(csv_path: str, method: str = "text-only") -> List[Tuple[str, int, int]]:
    """Load target features from CSV file."""
    features = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row['layer'])
            feature_idx = int(row['feature'])  # Use 'feature' column, not 'feature_idx'
            features.append((method, layer, feature_idx))
    
    print(f"[INFO] Loaded {len(features)} features from CSV")
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract top samples for VQA SAE features from spatial questions")
    parser.add_argument("--features-csv", type=str, required=True, 
                       help="CSV file with columns: method,layer,feature_idx")
    parser.add_argument("--sae-checkpoint-dir", type=str, required=True,
                       help="Directory containing SAE checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/vqa_spatial_feature_samples",
                       help="Output directory for results")
    parser.add_argument("--methods", nargs='+', default=["pretrained"],
                       help="SAE methods to analyze")
    parser.add_argument("--max-samples-per-feature", type=str, default="10",
                       help="Maximum samples to extract per feature (default: 10, use 'None' to extract all)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--keywords-file", type=str, default=None,
                       help="Optional path to newline-separated spatial keywords/phrases")
    parser.add_argument("--vqa-cache-dir", type=str, default=".cache/vqa_spatial_filter",
                       help="Directory to read/write cached VQAv2 filtered indices")
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading target features from {args.features_csv}")
    target_features = load_target_features(args.features_csv, args.methods[0])
    print(f"[INFO] Found {len(target_features)} target features")
    
    # Initialize extractor
    extractor = VQASpatialFeatureExtractor(args.sae_checkpoint_dir, args.methods)
    
    # Load spatial VQA dataset
    extractor.load_vqa_dataset(
        keywords_file=args.keywords_file,
        cache_dir=args.vqa_cache_dir
    )
    
    # Parse max_samples_per_feature argument
    if args.max_samples_per_feature.lower() == "none":
        max_samples_per_feature = None
    else:
        max_samples_per_feature = int(args.max_samples_per_feature)
    
    # Extract feature samples
    print(f"[INFO] Extracting samples for {len(target_features)} features...")
    results = extractor.extract_feature_samples(
        target_features, 
        max_samples_per_feature=max_samples_per_feature,
        batch_size=args.batch_size
    )
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_path / "vqa_spatial_feature_samples.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary CSV
    summary_rows = []
    for method in results:
        for layer in results[method]:
            for feat_idx in results[method][layer]:
                samples = results[method][layer][feat_idx]
                for sample in samples:
                    summary_rows.append({
                        'method': method,
                        'layer': layer,
                        'feature_idx': feat_idx,
                        'sample_idx': sample['sample_idx'],
                        'magnitude': sample['magnitude'],
                        'masked_token_count': sample['masked_token_count'],
                        'total_token_count': sample['total_token_count']
                    })
    
    summary_file = output_path / "vqa_spatial_feature_samples_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    
    # Create feature directories with sample_info.json files for later visualization
    print(f"\n[INFO] Creating feature directories with sample_info.json files...")
    for method in tqdm(results.keys(), desc="Creating feature directories"):
        for layer in results[method]:
            for feat_idx in results[method][layer]:
                samples = results[method][layer][feat_idx]
                if samples:
                    # Create feature folder
                    feature_folder = output_path / f"{method}_layer_{layer}_feature_{feat_idx}"
                    feature_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Save sample info for later visualization with extract_top_samples.py
                    sample_info = []
                    for i, sample in enumerate(samples):
                        sample_info.append({
                            'rank': i + 1,
                            'sample_idx': sample['sample_idx'],
                            'magnitude': sample['magnitude'],
                            'masked_token_count': sample['masked_token_count'],
                            'total_token_count': sample['total_token_count']
                        })
                    
                    # Save sample info as JSON
                    with open(feature_folder / "sample_info.json", 'w') as f:
                        json.dump(sample_info, f, indent=2)
                    
                    print(f"  Created {feature_folder.name} with {len(samples)} samples")
    
    print(f"\n[INFO] Results saved to {output_path}")
    print(f"[INFO] Detailed results: {results_file}")
    print(f"[INFO] Summary CSV: {summary_file}")
    print(f"[INFO] Feature directories created with sample_info.json files")
    print(f"[INFO] Use extract_top_samples.py to visualize individual features later")
    
    # Print summary statistics
    for method in results:
        print(f"\n=== {method.upper()} ===")
        for layer in sorted(results[method].keys()):
            print(f"  Layer {layer}:")
            for feat_idx in sorted(results[method][layer].keys()):
                samples = results[method][layer][feat_idx]
                if samples:
                    magnitudes = [s['magnitude'] for s in samples]
                    print(f"    Feature {feat_idx}: {len(samples)} samples, "
                          f"max mag: {max(magnitudes):.4f}, "
                          f"mean mag: {np.mean(magnitudes):.4f}")


if __name__ == "__main__":
    main()



