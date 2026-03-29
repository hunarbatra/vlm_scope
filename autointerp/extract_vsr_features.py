#!/usr/bin/env python3
"""
Extract top samples for common SAE features from VSR dataset with correct text-only magnitude calculations.
This script uses the exact same structure as track_firing_vsr.py but fixes the max magnitude bug.

Usage:
    CUDA_VISIBLE_DEVICES=7 python autointerp/extract_vsr_features.py \
        --features-csv results/stage_3/spatial/spatial_features_vqa.csv \
        --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/ \
        --output-dir results/stage_4/feature_samples/vsr_all_spatial_fixed \
        --max-samples-per-feature None \
        --cache-dir /scratch/local/ssd/lachin/vsr_image_cache
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

# Add to path for imports
import sys
sys.path.append("finetune/vqa")
from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions, initialize_sae

# Constants
NUM_VIS_TOKENS = 575

_MAP = {
    r"\bon (the )?right side of\b": "to the right of",
    r"\bon (the )?left side of\b":  "to the left of",
    r"\bon top of\b":               "on top of",
    r"\babove\b":                   "above",
    r"\bbelow\b":                   "below",
    r"\bbehind\b":                  "behind",
    r"\bin front of\b":             "in front of",
    r"\bbetween (.+?) and (.+?)$":  r"between \\1 and \\2",
    r"\binside\b":                  "inside",
    r"\boutside\b":                 "outside",
}

def canonicalize_vsr_statement_to_question(s: str) -> str:
    stmt = re.sub(r"[.?!\s]+$", "", s.strip())
    for pat, repl in _MAP.items():
        stmt = re.sub(pat, repl, stmt, flags=re.IGNORECASE)
    m = re.search(
        r"^(the |a |an )?(?P<subj>.+?)\s+(?P<rel>"
        r"to the right of|to the left of|on top of|above|below|behind|in front of|between .+? and .+?|inside|outside)"
        r"\s+(?P<obj>.+)$",
        stmt,
        flags=re.IGNORECASE,
    )
    if m:
        subj, rel, obj = m.group("subj"), m.group("rel"), m.group("obj")
        return f"Is {subj} {rel} {obj}?"
    return f"Is it true that {stmt}?"

class VSRFeatureExtractor:
    def __init__(self, sae_checkpoint_dir: str, device: str = "cuda"):
        self.sae_checkpoint_dir = Path(sae_checkpoint_dir)
        self.device = device
        
        # Load text-only SAEs
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
        
    def _load_saes(self) -> Dict[int, any]:
        """Load text-only SAEs for all layers."""
        saes = {}
        method_dir = self.sae_checkpoint_dir / "text-only"
        if method_dir.exists():
            for checkpoint_path in method_dir.glob("text-only_layer_*.pt"):
                # Extract layer number from filename
                layer_idx = int(checkpoint_path.stem.split("_")[-1])
                saes[layer_idx] = initialize_sae(
                    layer_idx=layer_idx, 
                    checkpoint_path=checkpoint_path, 
                    device=self.device
                )
                print(f"[INFO] Loaded text-only SAE for layer {layer_idx}")
        else:
            print(f"[WARN] Method directory not found: {method_dir}")
        return saes
    
    def load_vsr_dataset(self, cache_dir: Optional[str] = None):
        """Load VSR dataset with image caching."""
        from datasets import load_dataset
        
        data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="train")
        
        # Filter to only true statements
        # dataset = dataset.filter(lambda x: x["label"] == 1)
        # print(f"[INFO] Loaded {len(dataset)} true statements from VSR")
        
        self.dataset = dataset
        self.cache_dir = cache_dir
        if cache_dir:
            print(f"[INFO] Using image cache: {cache_dir}")

    def load_image_with_cache(self, image_url: str, timeout: int = 10):
        """Load image with disk caching (same behavior as extract_top_samples.py)."""
        from PIL import Image
        import requests
        from io import BytesIO
        import hashlib
        import os

        cache_dir = self.cache_dir or "/scratch/local/ssd/lachin/vsr_image_cache"
        os.makedirs(cache_dir, exist_ok=True)
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{url_hash}.jpg")

        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to load cached image {cache_path}: {e}")

        try:
            resp = requests.get(image_url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            try:
                img.save(cache_path, "JPEG")
            except Exception as e:
                print(f"[WARN] Failed to cache image: {e}")
            return img
        except Exception as e:
            print(f"[WARN] Failed to download image {image_url}: {e}")
            return Image.new("RGB", (224, 224), (128, 128, 128))
    
    def load_image_with_fallback(self, image_url: str, max_retries: int = 2) -> any:
        """Load image from URL with robust error handling."""
        from PIL import Image
        import requests
        from io import BytesIO
        import time
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            import hashlib
            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{url_hash}.jpg")
            
            if os.path.exists(cache_path):
                try:
                    return Image.open(cache_path).convert("RGB")
                except Exception as e:
                    print(f"[WARN] Failed to load cached image {cache_path}: {e}")
        
        for attempt in range(max_retries):
            try:
                session = requests.Session()
                response = session.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                if self.cache_dir:
                    try:
                        image.save(cache_path, "JPEG")
                    except Exception as e:
                        print(f"[WARN] Failed to cache image: {e}")
                
                return image
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[WARN] Failed to load image {image_url}: {e}")
                    return Image.new("RGB", (224, 224), (128, 128, 128))
                else:
                    time.sleep(2 ** attempt)
        
        return Image.new("RGB", (224, 224), (128, 128, 128))
    
    def get_sample_data(self, idx: int) -> Tuple[any, str]:
        """Get image and prompt for a sample."""
        sample = self.dataset[idx]
        image = self.load_image_with_cache(sample["image_link"])
        statement = sample["caption"].strip()
        prompt = canonicalize_vsr_statement_to_question(statement)
        return image, prompt
    
    def extract_feature_samples(self, target_features: List[Tuple[int, int]], 
                              max_samples_per_feature: int = 20,
                              batch_size: int = 8) -> Dict:
        """
        Extract top samples for target features with correct text-only magnitude calculations.
        Uses same efficient batching as extract_vqa_spatial_features.py
        
        Args:
            target_features: List of (layer, feature_idx) tuples
            max_samples_per_feature: Maximum samples to extract per feature
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with feature samples and their magnitudes
        """
        results = {}
        
        # Group features by layer for efficient processing
        feature_groups = {}
        for layer, feat_idx in target_features:
            if layer not in feature_groups:
                feature_groups[layer] = []
            feature_groups[layer].append(feat_idx)
        
        # Process all layers together in single forward passes (like extract_vqa_spatial_features.py)
        layers_to_do = [l for l in feature_groups.keys() if l in self.saes]
        
        # Process dataset in batches (limit to first 50k samples)
        max_samples = 50000
        for start_idx in tqdm(range(0, min(len(self.dataset), max_samples), batch_size), 
                            desc="Processing batches"):
            end_idx = min(start_idx + batch_size, min(len(self.dataset), max_samples))
            
            # Process batch for all layers at once
            batch_results = self._process_batch_all_layers(
                start_idx, end_idx, layers_to_do, feature_groups
            )
            
            # Merge results
            for layer in layers_to_do:
                if layer not in results:
                    results[layer] = {}
                
                for feat_idx in feature_groups[layer]:
                    if feat_idx not in results[layer]:
                        results[layer][feat_idx] = []
                    
                    if layer in batch_results and feat_idx in batch_results[layer]:
                        results[layer][feat_idx].extend(batch_results[layer][feat_idx])
        
        # Keep only top samples per feature (or all if max_samples_per_feature is None)
        for layer in results:
            for feat_idx in results[layer]:
                if feat_idx in results[layer]:
                    # Sort by magnitude
                    sorted_samples = sorted(
                        results[layer][feat_idx], 
                        key=lambda x: x['magnitude'], 
                        reverse=True
                    )
                    # If max_samples_per_feature is None, keep all samples
                    if max_samples_per_feature is not None:
                        results[layer][feat_idx] = sorted_samples[:max_samples_per_feature]
                    else:
                        results[layer][feat_idx] = sorted_samples
        
        return results
    
    def _process_batch_all_layers(self, start_idx: int, end_idx: int, 
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
            image, prompt = self.get_sample_data(j)
            input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
                image, prompt, self.image_processor, self.model, self.tokenizer
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
        
        # Get activations for all layers in single forward pass (like extract_vqa_spatial_features.py)
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
            sae = self.saes[layer]
            target_feats = feature_groups[layer]
            
            # Process each sample
            for sample_idx in range(layer_activations.shape[0]):
                # Compute text token count (excluding BOS) and dynamic image token length
                img_start, img_end = img_positions[sample_idx]
                text_token_count = int(batch_attention_mask[sample_idx, 1:].sum().item())
                img_len = 0
                if img_start is not None and img_end is not None and img_end >= img_start:
                    img_len = int(img_end - img_start + 1)
                actual_seq_len = text_token_count + img_len
                # Clamp to available activation width for safety
                avail_len = int(layer_activations.shape[1])
                actual_seq_len = max(0, min(actual_seq_len, avail_len))

                activations = layer_activations[sample_idx, :actual_seq_len]
                
                # Create text-only mask on the same device as activations
                device_for_mask = activations.device
                mask = torch.ones(actual_seq_len, dtype=torch.bool, device=device_for_mask)
                if img_start is not None and img_end is not None:
                    clamped_start = max(0, int(img_start))
                    clamped_end = min(actual_seq_len - 1, int(img_end)) if actual_seq_len > 0 else -1
                    if clamped_end >= clamped_start and clamped_end >= 0:
                        mask[clamped_start:clamped_end + 1] = False  # Zero out image tokens
                
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


def load_target_features(csv_path: str) -> List[Tuple[int, int]]:
    """Load target features from common_features_detailed.csv file."""
    features = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row['layer'])
            feature_idx = int(row['feature'])
            features.append((layer, feature_idx))
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract top samples for common SAE features from VSR dataset")
    parser.add_argument("--features-csv", type=str, required=True, 
                       help="Path to common_features_detailed.csv")
    parser.add_argument("--sae-checkpoint-dir", type=str, required=True,
                       help="Directory containing SAE checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/vsr_feature_samples",
                       help="Output directory for results")
    parser.add_argument("--max-samples-per-feature", type=str, default="20",
                       help="Maximum samples to extract per feature (default: 20, use 'None' to extract all)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--cache-dir", type=str, default="/scratch/local/ssd/lachin/vsr_image_cache",
                       help="Image cache directory")
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading target features from {args.features_csv}")
    target_features = load_target_features(args.features_csv)
    print(f"[INFO] Found {len(target_features)} target features")
    
    # Initialize extractor
    extractor = VSRFeatureExtractor(args.sae_checkpoint_dir)
    
    # Load dataset
    extractor.load_vsr_dataset(cache_dir=args.cache_dir)
    
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
    results_file = output_path / "vsr_feature_samples.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary CSV
    summary_rows = []
    for layer in results:
        for feat_idx in results[layer]:
            samples = results[layer][feat_idx]
            for sample in samples:
                summary_rows.append({
                    'layer': layer,
                    'feature_idx': feat_idx,
                    'sample_idx': sample['sample_idx'],
                    'magnitude': sample['magnitude'],
                    'masked_token_count': sample['masked_token_count'],
                    'total_token_count': sample['total_token_count']
                })
    
    summary_file = output_path / "vsr_feature_samples_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    
    # Create feature directories with sample_info.json files for later visualization
    print(f"\n[INFO] Creating feature directories with sample_info.json files...")
    for layer in tqdm(results.keys(), desc="Creating feature directories"):
        for feat_idx in results[layer]:
            samples = results[layer][feat_idx]
            if samples:
                # Create feature folder
                feature_folder = output_path / f"text-only_layer_{layer}_feature_{feat_idx}"
                feature_folder.mkdir(parents=True, exist_ok=True)
                
                # Save sample info for later visualization with extract_top_samples.py
                sample_info = []
                for i, sample in enumerate(samples):
                    # Attach stable identifiers to avoid dependency on dataset ordering later
                    try:
                        base_sample = extractor.dataset[sample['sample_idx']]
                        image_link = base_sample.get('image_link', None)
                        caption = base_sample.get('caption', None)
                    except Exception:
                        image_link = None
                        caption = None

                    sample_info.append({
                        'rank': i + 1,
                        'sample_idx': sample['sample_idx'],
                        'magnitude': sample['magnitude'],
                        'masked_token_count': sample['masked_token_count'],
                        'total_token_count': sample['total_token_count'],
                        'image_link': image_link,
                        'caption': caption,
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
    print(f"\n=== SUMMARY ===")
    for layer in sorted(results.keys()):
        print(f"  Layer {layer}:")
        for feat_idx in sorted(results[layer].keys()):
            samples = results[layer][feat_idx]
            if samples:
                magnitudes = [s['magnitude'] for s in samples]
                print(f"    Feature {feat_idx}: {len(samples)} samples, "
                      f"max mag: {max(magnitudes):.4f}, "
                      f"mean mag: {np.mean(magnitudes):.4f}")


if __name__ == "__main__":
    main()
