"""
Extract top samples for common SAE features from VSR dataset with correct text-only magnitude calculations.
This script uses the exact same structure as track_firing_vsr.py but fixes the max magnitude bug.

Usage:
    CUDA_VISIBLE_DEVICES=0 python extract_vsr_features.py \
        --features-csv results/stage_3/adapted_spatial_features_text-only/common_features_detailed.csv \
        --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/ \
        --output-dir results/vsr_feature_samples \
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

import sys
sys.path.append("finetune/vqa")
from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions, initialize_sae

NUM_VIS_TOKENS = 575

class VSRFeatureExtractor:
    def __init__(self, sae_checkpoint_dir: str, device: str = "cuda"):
        self.sae_checkpoint_dir = Path(sae_checkpoint_dir)
        self.device = device
        
        self.saes = self._load_saes()
        
        print("[INFO] Initializing VLM model...")
        self.tokenizer, self.model, self.image_processor = initialize_vlm_model("llava-more")
        self.model = self.model.to(device)
        
        from nnsight import NNsight
        self.model = NNsight(self.model)
        
    def _load_saes(self) -> Dict[int, any]:
        """Load text-only SAEs for all layers."""
        saes = {}
        method_dir = self.sae_checkpoint_dir / "text-only"
        if method_dir.exists():
            for checkpoint_path in method_dir.glob("text-only_layer_*.pt"):
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
    
    def extract_feature_samples(self, target_features: List[Tuple[int, int]], 
                              max_samples_per_feature: int = 20,
                              batch_size: int = 16) -> Dict:
        """
        Extract top samples for target features with correct text-only magnitude calculations.
        Uses exact same structure as track_firing_vsr.py but fixes max magnitude bug.
        """
        results = {}
        
        feature_groups = {}
        for layer, feat_idx in target_features:
            if layer not in feature_groups:
                feature_groups[layer] = []
            feature_groups[layer].append(feat_idx)
        
        for i in tqdm(range(0, len(self.dataset), batch_size), desc="Processing batches"):
            end_idx = min(i + batch_size, len(self.dataset))
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_image_tensors = []
            batch_image_sizes = []
            batch_sample_indices = []
            img_positions = []

            for j in range(i, end_idx):
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
            
            with torch.no_grad():
                with self.model.trace(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    images=batch_image_tensors,
                    image_sizes=batch_image_sizes,
                ):
                    layer_outputs = []
                    layers_to_do = [l for l in feature_groups.keys() if l in self.saes]
                    for layer in layers_to_do:
                        layer_outputs.append(self.model.model.layers[layer].output[0][:, 1:].detach().cpu().save())

            for i, layer in enumerate(layers_to_do):
                layer_activations = layer_outputs[i]
                sae = self.saes[layer]
                target_feats = feature_groups[layer]
                
                if layer not in results:
                    results[layer] = {}
                
                batch_entries = []
                for sample_idx in range(layer_activations.shape[0]):
                    actual_seq_len = batch_attention_mask[sample_idx, 1:].sum().item() + NUM_VIS_TOKENS
                    act = layer_activations[sample_idx, :actual_seq_len]
                    img_start, img_end = img_positions[sample_idx]
                    batch_entries.append((act, img_start, img_end))

                batch_max_seq_len = max(act.shape[0] for act, _, _ in batch_entries)

                padded_acts_list = []
                base_masks_list = []
                for (act, _, _) in batch_entries:
                    original_seq_len = act.shape[0]
                    pad_len = batch_max_seq_len - original_seq_len
                    if pad_len > 0:
                        act_padded = torch.nn.functional.pad(act, (0, 0, 0, pad_len))
                        mask = torch.ones(batch_max_seq_len, dtype=torch.bool, device=act.device)
                        mask[original_seq_len:] = False
                    else:
                        act_padded = act
                        mask = torch.ones(batch_max_seq_len, dtype=torch.bool, device=act.device)
                    padded_acts_list.append(act_padded)
                    base_masks_list.append(mask)

                batch_activations_cpu = torch.stack(padded_acts_list)
                base_masks_cpu = torch.stack(base_masks_list)

                batch_image_masks_cpu = torch.zeros_like(base_masks_cpu)
                for idx_s, (_, img_s, img_e) in enumerate(batch_entries):
                    if img_s is not None and img_e is not None:
                        batch_image_masks_cpu[idx_s, img_s:img_e+1] = True
                image_token_masks_cpu = base_masks_cpu & batch_image_masks_cpu
                text_token_masks_cpu = base_masks_cpu & (~batch_image_masks_cpu)

                batch_activations = batch_activations_cpu.to(sae.device)
                batch_masks = text_token_masks_cpu.to(sae.device)  # text-only

                if batch_masks.sum() == 0:
                    continue

                feature_acts = sae.encode(batch_activations)
                masked_feature_acts = feature_acts * batch_masks.unsqueeze(-1)

                for s_idx in range(feature_acts.shape[0]):
                    sample_acts = feature_acts[s_idx]
                    sample_mask = batch_masks[s_idx]
                    total_token_count = int(base_masks_cpu[s_idx].sum().item())

                    for feat_idx in target_feats:
                        if feat_idx >= sample_acts.shape[1]:
                            continue

                        masked = sample_acts[:, feat_idx].masked_fill(~sample_mask, float("-inf"))
                        max_magnitude = masked.max().item()
                        if np.isneginf(max_magnitude) or max_magnitude <= 0:
                            continue

                        results.setdefault(layer, {}).setdefault(feat_idx, []).append({
                            'sample_idx': batch_sample_indices[s_idx],
                            'magnitude': max_magnitude,
                            'masked_token_count': int(sample_mask.sum().item()),
                            'total_token_count': total_token_count,
                        })

                del batch_activations, batch_masks, feature_acts, masked_feature_acts
                torch.cuda.empty_cache()

        for layer in results:
            for feat_idx in results[layer]:
                if feat_idx in results[layer]:
                    sorted_samples = sorted(
                        results[layer][feat_idx], 
                        key=lambda x: x['magnitude'], 
                        reverse=True
                    )
                    results[layer][feat_idx] = sorted_samples[:max_samples_per_feature]
        
        return results


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
    parser.add_argument("--max-samples-per-feature", type=int, default=20,
                       help="Maximum samples to extract per feature (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--cache-dir", type=str, default="/scratch/local/ssd/lachin/vsr_image_cache",
                       help="Image cache directory")
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading target features from {args.features_csv}")
    target_features = load_target_features(args.features_csv)
    print(f"[INFO] Found {len(target_features)} target features")
    
    extractor = VSRFeatureExtractor(args.sae_checkpoint_dir)
    
    extractor.load_vsr_dataset(cache_dir=args.cache_dir)
    
    print(f"[INFO] Extracting samples for {len(target_features)} features...")
    results = extractor.extract_feature_samples(
        target_features, 
        max_samples_per_feature=args.max_samples_per_feature,
        batch_size=args.batch_size
    )
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "vsr_feature_samples.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
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
    
    print(f"\n[INFO] Creating feature directories with sample_info.json files...")
    for layer in tqdm(results.keys(), desc="Creating feature directories"):
        for feat_idx in results[layer]:
            samples = results[layer][feat_idx]
            if samples:
                feature_folder = output_path / f"text-only_layer_{layer}_feature_{feat_idx}"
                feature_folder.mkdir(parents=True, exist_ok=True)
                
                sample_info = []
                for i, sample in enumerate(samples):
                    sample_info.append({
                        'rank': i + 1,
                        'sample_idx': sample['sample_idx'],
                        'magnitude': sample['magnitude'],
                        'masked_token_count': sample['masked_token_count'],
                        'total_token_count': sample['total_token_count']
                    })
                
                with open(feature_folder / "sample_info.json", 'w') as f:
                    json.dump(sample_info, f, indent=2)
                
                print(f"  Created {feature_folder.name} with {len(samples)} samples")
    
    print(f"\n[INFO] Results saved to {output_path}")
    print(f"[INFO] Detailed results: {results_file}")
    print(f"[INFO] Summary CSV: {summary_file}")
    print(f"[INFO] Feature directories created with sample_info.json files")
    print(f"[INFO] Use extract_top_samples.py to visualize individual features later")
    
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
