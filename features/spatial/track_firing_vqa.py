#!/usr/bin/env python3
"""
Track SAE feature firing frequencies across VQA samples.
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import dotenv
dotenv.load_dotenv(".env")

import os
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from utils.datasets import load_vqa, VQAPairDataset
from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions, initialize_sae
import torch
from nnsight import NNsight
import gc
from tqdm import tqdm
import argparse
from collections import defaultdict
import wandb

NUM_VIS_TOKENS = 575

class DatasetWrapper:
    def __init__(self):
        self.dataset = load_dataset("lmms-lab/VQAv2", split="validation")

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        prompt = sample["question"]
        return image, prompt

    def __len__(self):
        return len(self.dataset)



def track_feature_firing_chunk(
    vlm_tokenizer, vlm_model, vlm_image_processor, dataset, 
    start_idx, end_idx, from_layer, to_layer, caching_batch_size,
    methods, sae_checkpoint_dir, output_dir="feature_analysis"
):
    """
    Track SAE feature firing
    """
    
    saes = {}
    for method in methods:
        saes[method] = {}
        if sae_checkpoint_dir:
            sae_checkpoint_dir_path = Path(sae_checkpoint_dir)
            for layer_idx in range(from_layer, to_layer):
                checkpoint_path = sae_checkpoint_dir_path / method / f"{method}_layer_{layer_idx}.pt"
                if checkpoint_path.exists():
                    saes[method][layer_idx] = initialize_sae(layer_idx=layer_idx, checkpoint_path=checkpoint_path, device="cuda")
                    print(f"[INFO] Loaded {method} SAE for layer {layer_idx} on GPU")
                else:
                    print(f"[WARN] No {method} SAE checkpoint found for layer {layer_idx}")
    
    counters = {}
    for method in methods:
        counters[method] = {
            'feature_firing': {},
            'image_firing': {},
            'text_firing': {},
            'activation_sum': {},
            'activation_count': {},
            'total_tokens': {},
            'sample_features': {},
            'feature_samples': {}  
        }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Tracking feature firing for samples {start_idx} → {end_idx}")
    print(f"[INFO] Layers {from_layer} → {to_layer}")
    print(f"[INFO] Methods: {methods}")
    
    for i in tqdm(range(start_idx, end_idx, caching_batch_size), desc="Processing chunks"):
        batch_input_ids = []
        batch_attention_mask = []
        batch_image_tensors = []
        batch_image_sizes = []  # fix for image_sizes per sample
        batch_sample_indices = []  # capture sample indices
        img_positions = []

        for j in range(i, min(i + caching_batch_size, end_idx)):
            image, prompt = dataset[j]
            input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
                image, prompt, vlm_image_processor, vlm_model, vlm_tokenizer
            )
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_image_tensors.append(image_tensor)
            batch_image_sizes.append(image_sizes[0])  # collect size per sample
            batch_sample_indices.append(j)

            img_start, img_end = get_image_token_positions(input_ids)
            img_positions.append((img_start - 1, img_end - 1))
            del input_ids, attention_mask, image_tensor

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            [ids.squeeze(0) for ids in batch_input_ids],
            batch_first=True,
            padding_value=vlm_tokenizer.pad_token_id,
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [mask.squeeze(0) for mask in batch_attention_mask],
            batch_first=True,
            padding_value=0,
        )
        batch_image_tensors = torch.cat(batch_image_tensors, dim=0)
        batch_input_ids = batch_input_ids.to(vlm_model.device)
        batch_attention_mask = batch_attention_mask.to(vlm_model.device)
        batch_image_tensors = batch_image_tensors.to(vlm_model.device)
        
        with torch.no_grad():
            with vlm_model.trace(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                images=batch_image_tensors,
                image_sizes=batch_image_sizes,
            ) as tr:
                layer_outputs = []
                for layer_idx in range(from_layer, to_layer):
                    layer_outputs.append(vlm_model.model.layers[layer_idx].output[0][:, 1:].detach().cpu().save())

            # Process each layer's activations (outside trace context)
            for layer_idx in range(from_layer, to_layer):
                layer_activations = layer_outputs[layer_idx - from_layer]
                
                # --------------------------------------------------
                # Prepare padded activations & base mask ONCE per layer
                # --------------------------------------------------
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

                # Stack once on CPU to save GPU mem – will move to GPU when needed
                batch_activations_cpu = torch.stack(padded_acts_list)          # (batch, seq, d_model)
                base_masks_cpu = torch.stack(base_masks_list)                  # (batch, seq)

                # Pre-compute image token mask for this batch (CPU)
                batch_image_masks_cpu = torch.zeros_like(base_masks_cpu)
                for idx_s, (_, img_s, img_e) in enumerate(batch_entries):
                    if img_s is not None and img_e is not None:
                        batch_image_masks_cpu[idx_s, img_s:img_e+1] = True
                image_token_masks_cpu = base_masks_cpu & batch_image_masks_cpu
                text_token_masks_cpu = base_masks_cpu & (~batch_image_masks_cpu)

                # --------------------------------------------------
                # Process each method's SAE (re-use tensors above)
                # --------------------------------------------------
                for method in methods:
                    if layer_idx not in saes[method]:
                        continue
                    sae = saes[method][layer_idx]
                    # SAE already on GPU from initialization

                    # Move activations to GPU only once per method
                    batch_activations = batch_activations_cpu.to(sae.device)

                    # Select appropriate mask
                    lower = method.lower()
                    if lower == "image-only":
                        batch_masks = image_token_masks_cpu.to(sae.device)
                    elif lower == "text-only":
                        batch_masks = text_token_masks_cpu.to(sae.device)
                    else:
                        batch_masks = base_masks_cpu.to(sae.device)

                    if batch_masks.sum() == 0:
                        continue

                    feature_acts = sae.encode(batch_activations)  # (batch, seq, num_features)

                    masked_feature_acts = feature_acts * batch_masks.unsqueeze(-1)
                    # --- extra metrics (token split, activ mags, sample-level) ---
                    if lower == "image-only":
                        image_token_masks = batch_masks
                        text_token_masks = torch.zeros_like(batch_masks)
                    elif lower == "text-only":
                        image_token_masks = torch.zeros_like(batch_masks)
                        text_token_masks = batch_masks
                    else:
                        image_token_masks = image_token_masks_cpu.to(sae.device)
                        text_token_masks = text_token_masks_cpu.to(sae.device)
                    
                    # Sample-level fired features with magnitudes (vectorized for speed)
                    for s_idx in range(feature_acts.shape[0]):
                            sample_acts = feature_acts[s_idx]          # (seq, feats)
                            sample_mask = batch_masks[s_idx]            # (seq,)
                            fired_mask = (sample_acts > 0) & sample_mask.unsqueeze(-1)

                            # Identify indices of fired features
                            fired_idxs = torch.nonzero(fired_mask.any(dim=0), as_tuple=False).squeeze(-1)
                            sample_idx = batch_sample_indices[s_idx]
                            fired_features = {}

                            if fired_idxs.numel() > 0:
                                    # Max magnitude per fired feature (vectorised)
                                    max_magnitudes = sample_acts[:, fired_idxs].max(dim=0).values

                                    # Move to CPU for cheaper dict construction
                                    fired_ids = fired_idxs.cpu().tolist()
                                    fired_mags = max_magnitudes.cpu().tolist()
                                    fired_features = dict(zip(fired_ids, fired_mags))

                                    # Store feature → sample mapping
                                    if layer_idx not in counters[method]['feature_samples']:
                                        counters[method]['feature_samples'][layer_idx] = {}
                                    for f_id, mag in zip(fired_ids, fired_mags):
                                        if f_id not in counters[method]['feature_samples'][layer_idx]:
                                            counters[method]['feature_samples'][layer_idx][f_id] = {}
                                        counters[method]['feature_samples'][layer_idx][f_id][sample_idx] = mag

                            # Store sample → features mapping (back-compat)
                            if layer_idx not in counters[method]['sample_features']:
                                counters[method]['sample_features'][layer_idx] = {}
                                counters[method]['sample_features'][layer_idx][sample_idx] = fired_features

                    
                    # Simple tensor operations
                    with torch.no_grad():
                                # Count fired features per type
                                fired = (feature_acts > 0)
                                img_fired = fired & image_token_masks.unsqueeze(-1)
                                txt_fired = fired & text_token_masks.unsqueeze(-1)

                                # Sum across batch and sequence
                                img_counts = img_fired.sum(dim=(0,1)).cpu()
                                txt_counts = txt_fired.sum(dim=(0,1)).cpu()

                                # IMPORTANT: respect the selected mask
                                masked_fired = fired & batch_masks.unsqueeze(-1)
                                feature_counts = masked_fired.sum(dim=(0,1)).cpu()

                                # Activation magnitudes (already masked to selected tokens)
                                pos_acts = torch.where(masked_feature_acts > 0, masked_feature_acts, 0)
                                pos_sum = pos_acts.sum(dim=(0,1)).cpu()
                                pos_cnt = (pos_acts > 0).sum(dim=(0,1)).cpu()

                    
                    # Update counters
                    for f_idx in range(feature_counts.shape[0]):
                                if feature_counts[f_idx].item() > 0:
                                    if layer_idx not in counters[method]['feature_firing']:
                                        counters[method]['feature_firing'][layer_idx] = {}
                                    if f_idx not in counters[method]['feature_firing'][layer_idx]:
                                        counters[method]['feature_firing'][layer_idx][f_idx] = 0
                                    counters[method]['feature_firing'][layer_idx][f_idx] += int(feature_counts[f_idx])
                                    
                                    # Image/text specific counts
                                    if img_counts[f_idx].item() > 0:
                                        if layer_idx not in counters[method]['image_firing']:
                                            counters[method]['image_firing'][layer_idx] = {}
                                        if f_idx not in counters[method]['image_firing'][layer_idx]:
                                            counters[method]['image_firing'][layer_idx][f_idx] = 0
                                        counters[method]['image_firing'][layer_idx][f_idx] += int(img_counts[f_idx])
                                        
                                    if txt_counts[f_idx].item() > 0:
                                        if layer_idx not in counters[method]['text_firing']:
                                            counters[method]['text_firing'][layer_idx] = {}
                                        if f_idx not in counters[method]['text_firing'][layer_idx]:
                                            counters[method]['text_firing'][layer_idx][f_idx] = 0
                                        counters[method]['text_firing'][layer_idx][f_idx] += int(txt_counts[f_idx])
                                    
                                    # Activation magnitudes
                                    if pos_cnt[f_idx].item() > 0:
                                        if layer_idx not in counters[method]['activation_sum']:
                                            counters[method]['activation_sum'][layer_idx] = {}
                                        if f_idx not in counters[method]['activation_sum'][layer_idx]:
                                            counters[method]['activation_sum'][layer_idx][f_idx] = 0.0
                                        counters[method]['activation_sum'][layer_idx][f_idx] += float(pos_sum[f_idx])
                                        
                                        if layer_idx not in counters[method]['activation_count']:
                                            counters[method]['activation_count'][layer_idx] = {}
                                        if f_idx not in counters[method]['activation_count'][layer_idx]:
                                            counters[method]['activation_count'][layer_idx][f_idx] = 0
                                        counters[method]['activation_count'][layer_idx][f_idx] += int(pos_cnt[f_idx])
                            
                    
                    # Total tokens
                    if layer_idx not in counters[method]['total_tokens']:
                        counters[method]['total_tokens'][layer_idx] = 0
                    counters[method]['total_tokens'][layer_idx] += batch_masks.sum().item()
                    
                    del batch_activations, batch_masks, feature_acts, masked_feature_acts, img_counts, txt_counts, pos_sum, pos_cnt
                    torch.cuda.empty_cache()  # Re-enabled to prevent OOM
                
                del layer_activations

        # Clean up batch tensors
        del batch_input_ids, batch_attention_mask, batch_image_tensors
        torch.cuda.empty_cache()
    
    # Calculate feature firing frequencies
    results = {}
    for method in methods:
        method_results = {}
        for layer_idx in range(from_layer, to_layer):
            if layer_idx in saes[method]:
                num_features = saes[method][layer_idx].cfg.d_sae
                layer_frequencies = {}
                
                for feature_idx in range(num_features):
                    firing_count = counters[method]['feature_firing'].get(layer_idx, {}).get(feature_idx, 0)
                    total_tokens = counters[method]['total_tokens'].get(layer_idx, 0)
                    frequency = firing_count / total_tokens if total_tokens > 0 else 0
                    layer_frequencies[feature_idx] = {
                        'firing_count': firing_count,
                        'total_tokens': total_tokens,
                        'frequency': frequency,
                        'log_frequency': np.log10(frequency + 1e-10)
                    }
                
                method_results[layer_idx] = layer_frequencies
        results[method] = method_results
    
    # Save results
    analysis_results = {
        'feature_firing_frequencies': results,
        'total_tokens_per_layer': {method: counters[method]['total_tokens'] for method in methods},
        'analysis_params': {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'from_layer': from_layer,
            'to_layer': to_layer,
            'caching_batch_size': caching_batch_size,
            'num_samples_analyzed': end_idx - start_idx,
            'methods': methods
        }
    }
    
    results_path = output_path / f"feature_firing_analysis_{start_idx}_{end_idx}.json"
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Save extra metrics - separate files per method AND layer for efficiency
    for method in methods:
        # Save basic stats (small file)
        basic_metrics = {
            'image_firing_counts': counters[method]['image_firing'],
            'text_firing_counts': counters[method]['text_firing'],
            'activation_sum': counters[method]['activation_sum'],
            'activation_count': counters[method]['activation_count']
        }
        torch.save(basic_metrics, output_path / f"basic_metrics_{method}_{start_idx}_{end_idx}.pt")
        
        # NOTE: Skipping sample -> feature save to reduce size as requested
        # for layer_idx in counters[method]['sample_features']:
        #     layer_sample_data = {
        #         'sample_feature_firing': {layer_idx: counters[method]['sample_features'][layer_idx]}
        #     }
        #     torch.save(layer_sample_data, output_path / f"sample_data_{method}_layer{layer_idx}_{start_idx}_{end_idx}.pt")
        
        # Save feature->sample mappings per layer (trim to top-20 samples per feature)
        for layer_idx in counters[method]['feature_samples']:
            trimmed_layer = {}
            for feat_id, sample_map in counters[method]['feature_samples'][layer_idx].items():
                # sort samples by magnitude desc and keep top 20
                top_items = sorted(sample_map.items(), key=lambda kv: kv[1], reverse=True)[:10]
                trimmed_layer[feat_id] = {s_idx: mag for s_idx, mag in top_items}

            layer_feature_data = {
                'feature_sample_firing': {layer_idx: trimmed_layer}
            }
            torch.save(layer_feature_data, output_path / f"feature_data_{method}_layer{layer_idx}_{start_idx}_{end_idx}.pt")
        
        print(f"[INFO] Saved {method} metrics split by layer")
    
    # Also save combined file for backward compatibility
    # extra_metrics = {
    #     'image_firing_counts': {method: counters[method]['image_firing'] for method in methods},
    #     'text_firing_counts': {method: counters[method]['text_firing'] for method in methods},
    #     'activation_sum': {method: counters[method]['activation_sum'] for method in methods},
    #     'activation_count': {method: counters[method]['activation_count'] for method in methods},
    #     'sample_feature_firing': {method: counters[method]['sample_features'] for method in methods},
    #     'feature_sample_firing': {method: counters[method]['feature_samples'] for method in methods}
    # }
    # torch.save(extra_metrics, output_path / f"extra_metrics_{start_idx}_{end_idx}.pt")
    
    # Log to wandb
    try:
        wandb_run = wandb.init(project="sae-feature-firing", name=f"feature_firing_{start_idx}_{end_idx}")
        
        for method in methods:
            for layer_idx, layer_frequencies in results[method].items():
                frequencies = [freq['frequency'] for freq in layer_frequencies.values()]
                log_frequencies = [freq['log_frequency'] for freq in layer_frequencies.values()]
                
                wandb.log({
                    f"{method}/layer_{layer_idx}/mean_firing_frequency": np.mean(frequencies),
                    f"{method}/layer_{layer_idx}/median_firing_frequency": np.median(frequencies),
                    f"{method}/layer_{layer_idx}/std_firing_frequency": np.std(frequencies),
                    f"{method}/layer_{layer_idx}/dead_features": sum(1 for f in frequencies if f < 1e-6),
                    f"{method}/layer_{layer_idx}/dense_features": sum(1 for f in frequencies if f > 0.01),
                    f"{method}/layer_{layer_idx}/feature_frequency_histogram": wandb.Histogram(log_frequencies)
                })
        
        wandb.finish()
    except Exception as e:
        print(f"[WARN] Could not log to wandb: {e}")
    
    vlm_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Track SAE feature firing frequencies")
    parser.add_argument("--from-layer", type=int, default=0, help="First layer index (inclusive)")
    parser.add_argument("--to-layer", type=int, default=32, help="Last layer index (exclusive)")
    parser.add_argument("--start-sample", type=int, default=0, help="Starting sample index")
    parser.add_argument("--end-sample", type=int, default=50000, help="Ending sample index")
    parser.add_argument("--caching-batch-size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--output-dir", type=str, default="feature_analysis", help="Output directory")
    parser.add_argument("--sae-checkpoint-dir", type=str, required=True, help="Directory containing SAE checkpoints organized by method")
    parser.add_argument("--methods", nargs='+', default=["pretrained", "random", "text-only", "image-only"], 
                       help="SAE methods to analyze")
    parser.add_argument("--vqa-split", type=str, default="validation", choices=["train", "validation", "test"],
                       help="VQAv2 split to use")
    
    args = parser.parse_args()
    
    print(f"[INFO] Feature firing analysis for samples {args.start_sample} → {args.end_sample}")
    print(f"[INFO] Layers {args.from_layer} → {args.to_layer}")
    print(f"[INFO] Methods: {args.methods}")
    print(f"[INFO] SAE checkpoints: {args.sae_checkpoint_dir}")
    print(f"[INFO] VQA split: {args.vqa_split}")
    
    # Initialize model and dataset
    vlm_tokenizer, vlm_model, vlm_image_processor = initialize_vlm_model("llava-more", device="cuda")
    vlm_model = NNsight(vlm_model)
    
    # Load VQA dataset using unified loader
    vqa_dataset = load_vqa(split=args.vqa_split)
    dataset = VQAPairDataset(vqa_dataset)
    
    # Run analysis
    results = track_feature_firing_chunk(
        vlm_tokenizer,
        vlm_model,
        vlm_image_processor,
        dataset,
        args.start_sample,
        args.end_sample,
        args.from_layer,
        args.to_layer,
        args.caching_batch_size,
        args.methods,
        args.sae_checkpoint_dir,
        args.output_dir
    )
    
    print(f"[INFO] Analysis complete! Results saved to {args.output_dir}")
    
    # Print summary statistics
    for method in args.methods:
        print(f"\n=== {method.upper()} SAE ===")
        for layer_idx in range(args.from_layer, args.to_layer):
            if layer_idx in results['feature_firing_frequencies'][method]:
                layer_freqs = results['feature_firing_frequencies'][method][layer_idx]
                frequencies = [freq['frequency'] for freq in layer_freqs.values()]
                
                print(f"Layer {layer_idx}:")
                print(f"  Mean firing frequency: {np.mean(frequencies):.6f}")
                print(f"  Dead features (< 1e-6): {sum(1 for f in frequencies if f < 1e-6)}")
                print(f"  Dense features (> 0.01): {sum(1 for f in frequencies if f > 0.01)}")

if __name__ == "__main__":
    main() 