#!/usr/bin/env python3
"""
Extract top samples for specific SAE features with correct text-only magnitude calculations.
This script fixes the max magnitude bug by computing magnitudes only over text tokens.

Usage:
    python extract_feature_samples.py \
        --features-csv features_to_extract.csv \
        --sae-checkpoint-dir /path/to/sae/checkpoints \
        --output-dir results/feature_samples \
        --methods text-only \
        --dataset vsr \
        --max-samples-per-feature 20
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

# Add to path for imports
import sys
sys.path.append("finetune/vqa")
from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions, initialize_sae

# Constants
NUM_VIS_TOKENS = 575

class FeatureExtractor:
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
    
    def load_vsr_dataset(self, cache_dir: Optional[str] = None):
        """Load VSR dataset with image caching."""
        from datasets import load_dataset
        
        data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="train")
        
        # Filter to only true statements
        dataset = dataset.filter(lambda x: x["label"] == 1)
        print(f"[INFO] Loaded {len(dataset)} true statements from VSR")
        
        self.dataset = dataset
        self.cache_dir = cache_dir
        if cache_dir:
            print(f"[INFO] Using image cache: {cache_dir}")
    
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
        image = self.load_image_with_fallback(sample["image_link"])
        statement = sample["caption"].strip()
        prompt = f"Is the following statement correct? : '{statement}'"
        return image, prompt
    
    def extract_feature_samples(self, target_features: List[Tuple[str, int, int]], 
                              max_samples_per_feature: int = 20,
                              batch_size: int = 8) -> Dict:
        """
        Extract top samples for target features with correct text-only magnitudes.
        
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
        
        # Process each method and layer
        for method in tqdm(feature_groups.keys(), desc="Processing methods"):
            results[method] = {}
            
            for layer in tqdm(feature_groups[method].keys(), desc=f"Processing {method} layers"):
                if layer not in self.saes[method]:
                    print(f"[WARN] No SAE found for {method} layer {layer}")
                    continue
                
                sae = self.saes[method][layer]
                target_feats = feature_groups[method][layer]
                results[method][layer] = {}
                
                # Process dataset in batches
                for start_idx in tqdm(range(0, len(self.dataset), batch_size), 
                                    desc=f"Processing {method} layer {layer}"):
                    end_idx = min(start_idx + batch_size, len(self.dataset))
                    
                    # Process batch
                    batch_results = self._process_batch(
                        start_idx, end_idx, method, layer, sae, target_feats
                    )
                    
                    # Merge results
                    for feat_idx in target_feats:
                        if feat_idx not in results[method][layer]:
                            results[method][layer][feat_idx] = []
                        
                        if feat_idx in batch_results:
                            results[method][layer][feat_idx].extend(batch_results[feat_idx])
                
                # Keep only top samples per feature
                for feat_idx in target_feats:
                    if feat_idx in results[method][layer]:
                        # Sort by magnitude and keep top samples
                        sorted_samples = sorted(
                            results[method][layer][feat_idx], 
                            key=lambda x: x['magnitude'], 
                            reverse=True
                        )
                        results[method][layer][feat_idx] = sorted_samples[:max_samples_per_feature]
        
        return results
    
    def _process_batch(self, start_idx: int, end_idx: int, method: str, 
                      layer: int, sae: any, target_feats: List[int]) -> Dict:
        """Process a batch of samples for specific features."""
        batch_results = {feat_idx: [] for feat_idx in target_feats}
        
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
        
        batch_input_ids = batch_input_ids.to(self.device)
        batch_attention_mask = batch_attention_mask.to(self.device)
        batch_image_tensors = batch_image_tensors.to(self.device)
        
        # Get activations
        with torch.no_grad():
            with self.model.trace(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                images=batch_image_tensors,
                image_sizes=batch_image_sizes,
            ) as tr:
                layer_activations = self.model.model.layers[layer].output[0][:, 1:].detach()
        
        # Process each sample
        for sample_idx in range(layer_activations.shape[0]):
            actual_seq_len = batch_attention_mask[sample_idx, 1:].sum().item() + NUM_VIS_TOKENS
            activations = layer_activations[sample_idx, :actual_seq_len]
            img_start, img_end = img_positions[sample_idx]
            
            # Create text-only mask
            if method.lower() == "text-only":
                mask = torch.ones(actual_seq_len, dtype=torch.bool, device=self.device)
                if img_start is not None and img_end is not None:
                    mask[img_start:img_end + 1] = False  # Zero out image tokens
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
                    # Get activations for this feature over masked tokens only
                    masked_feature_acts = feature_acts[:, feat_idx] * mask
                    
                    # Find maximum magnitude over masked tokens only
                    if masked_feature_acts.sum() > 0:
                        max_magnitude = masked_feature_acts.max().item()
                        
                        if max_magnitude > 0:
                            batch_results[feat_idx].append({
                                'sample_idx': batch_sample_indices[sample_idx],
                                'magnitude': max_magnitude,
                                'masked_token_count': mask.sum().item(),
                                'total_token_count': actual_seq_len
                            })
        
        # Clean up
        del batch_input_ids, batch_attention_mask, batch_image_tensors, layer_activations
        torch.cuda.empty_cache()
        
        return batch_results


def load_target_features(csv_path: str) -> List[Tuple[str, int, int]]:
    """Load target features from CSV file."""
    features = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            layer = int(row['layer'])
            feature_idx = int(row['feature_idx'])
            features.append((method, layer, feature_idx))
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract top samples for specific SAE features")
    parser.add_argument("--features-csv", type=str, required=True, 
                       help="CSV file with columns: method,layer,feature_idx")
    parser.add_argument("--sae-checkpoint-dir", type=str, required=True,
                       help="Directory containing SAE checkpoints")
    parser.add_argument("--output-dir", type=str, default="results/feature_samples",
                       help="Output directory for results")
    parser.add_argument("--methods", nargs='+', default=["text-only"],
                       help="SAE methods to analyze")
    parser.add_argument("--dataset", type=str, default="vsr", choices=["vsr", "vqa"],
                       help="Dataset to use")
    parser.add_argument("--max-samples-per-feature", type=int, default=20,
                       help="Maximum samples to extract per feature")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--cache-dir", type=str, default="/scratch/local/ssd/lachin/vsr_image_cache",
                       help="Image cache directory")
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading target features from {args.features_csv}")
    target_features = load_target_features(args.features_csv)
    print(f"[INFO] Found {len(target_features)} target features")
    
    # Initialize extractor
    extractor = FeatureExtractor(args.sae_checkpoint_dir, args.methods)
    
    # Load dataset
    if args.dataset == "vsr":
        extractor.load_vsr_dataset(cache_dir=args.cache_dir)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
    
    # Extract feature samples
    print(f"[INFO] Extracting samples for {len(target_features)} features...")
    results = extractor.extract_feature_samples(
        target_features, 
        max_samples_per_feature=args.max_samples_per_feature,
        batch_size=args.batch_size
    )
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_path / f"feature_samples_{args.dataset}.json"
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
    
    summary_file = output_path / f"feature_samples_{args.dataset}_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    
    print(f"[INFO] Results saved to {output_path}")
    print(f"[INFO] Detailed results: {results_file}")
    print(f"[INFO] Summary CSV: {summary_file}")
    
    # Print summary statistics
    for method in results:
        print(f"\n=== {method.upper()} ===")
        for layer in results[method]:
            print(f"  Layer {layer}:")
            for feat_idx in results[method][layer]:
                samples = results[method][layer][feat_idx]
                if samples:
                    magnitudes = [s['magnitude'] for s in samples]
                    print(f"    Feature {feat_idx}: {len(samples)} samples, "
                          f"max mag: {max(magnitudes):.4f}, "
                          f"mean mag: {np.mean(magnitudes):.4f}")


if __name__ == "__main__":
    main()
