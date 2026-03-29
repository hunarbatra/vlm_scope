#!/usr/bin/env python3
"""
Track SAE feature firing frequencies on a spatial-only subset of VQAv2.

This mirrors finetune/vqa/track_firing_vqa.py but filters the VQAv2 validation
set to questions that contain spatial cues (e.g., "left", "right", "behind").

Usage example
-------------
CUDA_VISIBLE_DEVICES=7 python finetune/vqa/track_firing_vqa_spatial.py \
  --from-layer 0 \
  --to-layer 32 \
  --start-sample 0 \
  --end-sample 20000 \
  --caching-batch-size 8 \
  --output-dir "results/feature_firing_analysis/vqa_spatial_text_only" \
  --sae-checkpoint-dir "/scratch/local/ssd/lachin/checkpoints_50k/" \
  --methods text-only
"""

import os
import json
from pathlib import Path
from typing import List, Optional

import dotenv
dotenv.load_dotenv(".env")

import numpy as np
import torch
from datasets import load_dataset
from nnsight import NNsight
from tqdm import tqdm
import argparse
import gc
import re
import wandb
import hashlib

from utils.datasets import load_vqa_spatial
from utils import (
    process_vlm_inputs,
    get_image_token_positions,
)
from finetune.vqa.utils import initialize_sae, initialize_vlm_model


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
    """Compile a case-insensitive regex that matches any keyword as a word-ish token.

    Multi-word keywords (e.g., "in front") will be converted to allow flexible spacing
    or a hyphen (e.g., "top left" or "top-left").
    """
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
    """VQAv2 validation subset filtered to spatial questions by keyword/regex.

    If a keyword file is provided, it should be a text file with one keyword or
    phrase per line. Otherwise a default list is used.
    """

    def __init__(
        self,
        split: str = "validation",
        keywords_file: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
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


def track_feature_firing_chunk(
    vlm_tokenizer, vlm_model, vlm_image_processor, dataset,
    start_idx, end_idx, from_layer, to_layer, caching_batch_size,
    methods, sae_checkpoint_dir, output_dir="feature_analysis",
):
    """Track SAE feature firing on a dataset slice [start_idx, end_idx).

    Mirrors finetune/vqa/track_firing_vqa.py for consistency so results are
    directly comparable across datasets.
    """

    saes = {}
    for method in methods:
        saes[method] = {}
        if sae_checkpoint_dir:
            sae_checkpoint_dir_path = Path(sae_checkpoint_dir)
            for layer_idx in range(from_layer, to_layer):
                checkpoint_path = sae_checkpoint_dir_path / method / f"{method}_layer_{layer_idx}.pt"
                if checkpoint_path.exists():
                    saes[method][layer_idx] = initialize_sae(
                        layer_idx=layer_idx, checkpoint_path=checkpoint_path, device="cuda"
                    )
                    print(f"[INFO] Loaded {method} SAE for layer {layer_idx} on GPU")
                else:
                    print(f"[WARN] No {method} SAE checkpoint found for layer {layer_idx}")

    counters = {}
    for method in methods:
        counters[method] = {
            "feature_firing": {},
            "image_firing": {},
            "text_firing": {},
            "activation_sum": {},
            "activation_count": {},
            "total_tokens": {},
            "sample_features": {},
            "feature_samples": {},
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
        batch_image_sizes = []
        batch_sample_indices = []
        img_positions = []

        for j in range(i, min(i + caching_batch_size, end_idx)):
            image, prompt = dataset[j]
            input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
                image, prompt, vlm_image_processor, vlm_model, vlm_tokenizer
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
                    layer_outputs.append(
                        vlm_model.model.layers[layer_idx].output[0][:, 1:].detach().cpu().save()
                    )

            for layer_idx in range(from_layer, to_layer):
                layer_activations = layer_outputs[layer_idx - from_layer]

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
                        batch_image_masks_cpu[idx_s, img_s : img_e + 1] = True
                image_token_masks_cpu = base_masks_cpu & batch_image_masks_cpu
                text_token_masks_cpu = base_masks_cpu & (~batch_image_masks_cpu)

                for method in methods:
                    if layer_idx not in saes[method]:
                        continue
                    sae = saes[method][layer_idx]

                    batch_activations = batch_activations_cpu.to(sae.device)

                    lower = method.lower()
                    if lower == "image-only":
                        batch_masks = image_token_masks_cpu.to(sae.device)
                    elif lower == "text-only":
                        batch_masks = text_token_masks_cpu.to(sae.device)
                    else:
                        batch_masks = base_masks_cpu.to(sae.device)

                    if batch_masks.sum() == 0:
                        continue

                    feature_acts = sae.encode(batch_activations)

                    masked_feature_acts = feature_acts * batch_masks.unsqueeze(-1)

                    if lower == "image-only":
                        image_token_masks = batch_masks
                        text_token_masks = torch.zeros_like(batch_masks)
                    elif lower == "text-only":
                        image_token_masks = torch.zeros_like(batch_masks)
                        text_token_masks = batch_masks
                    else:
                        image_token_masks = image_token_masks_cpu.to(sae.device)
                        text_token_masks = text_token_masks_cpu.to(sae.device)

                    for s_idx in range(feature_acts.shape[0]):
                        sample_acts = feature_acts[s_idx]
                        sample_mask = batch_masks[s_idx]
                        fired_mask = (sample_acts > 0) & sample_mask.unsqueeze(-1)

                        fired_idxs = torch.nonzero(fired_mask.any(dim=0), as_tuple=False).squeeze(-1)
                        sample_idx = batch_sample_indices[s_idx]
                        fired_features = {}

                        if fired_idxs.numel() > 0:
                            max_magnitudes = sample_acts[:, fired_idxs].max(dim=0).values
                            fired_ids = fired_idxs.cpu().tolist()
                            fired_mags = max_magnitudes.cpu().tolist()
                            fired_features = dict(zip(fired_ids, fired_mags))

                            if layer_idx not in counters[method]["feature_samples"]:
                                counters[method]["feature_samples"][layer_idx] = {}
                            for f_id, mag in zip(fired_ids, fired_mags):
                                if f_id not in counters[method]["feature_samples"][layer_idx]:
                                    counters[method]["feature_samples"][layer_idx][f_id] = {}
                                counters[method]["feature_samples"][layer_idx][f_id][sample_idx] = mag

                        if layer_idx not in counters[method]["sample_features"]:
                            counters[method]["sample_features"][layer_idx] = {}
                            counters[method]["sample_features"][layer_idx][sample_idx] = fired_features

                    with torch.no_grad():
                        fired = feature_acts > 0
                        img_fired = fired & image_token_masks.unsqueeze(-1)
                        txt_fired = fired & text_token_masks.unsqueeze(-1)

                        img_counts = img_fired.sum(dim=(0, 1)).cpu()
                        txt_counts = txt_fired.sum(dim=(0, 1)).cpu()

                        masked_fired = fired & batch_masks.unsqueeze(-1)
                        feature_counts = masked_fired.sum(dim=(0, 1)).cpu()

                        pos_acts = torch.where(masked_feature_acts > 0, masked_feature_acts, 0)
                        pos_sum = pos_acts.sum(dim=(0, 1)).cpu()
                        pos_cnt = (pos_acts > 0).sum(dim=(0, 1)).cpu()

                    for f_idx in range(feature_counts.shape[0]):
                        if feature_counts[f_idx].item() > 0:
                            if layer_idx not in counters[method]["feature_firing"]:
                                counters[method]["feature_firing"][layer_idx] = {}
                            if f_idx not in counters[method]["feature_firing"][layer_idx]:
                                counters[method]["feature_firing"][layer_idx][f_idx] = 0
                            counters[method]["feature_firing"][layer_idx][f_idx] += int(feature_counts[f_idx])

                            if img_counts[f_idx].item() > 0:
                                if layer_idx not in counters[method]["image_firing"]:
                                    counters[method]["image_firing"][layer_idx] = {}
                                if f_idx not in counters[method]["image_firing"][layer_idx]:
                                    counters[method]["image_firing"][layer_idx][f_idx] = 0
                                counters[method]["image_firing"][layer_idx][f_idx] += int(img_counts[f_idx])

                            if txt_counts[f_idx].item() > 0:
                                if layer_idx not in counters[method]["text_firing"]:
                                    counters[method]["text_firing"][layer_idx] = {}
                                if f_idx not in counters[method]["text_firing"][layer_idx]:
                                    counters[method]["text_firing"][layer_idx][f_idx] = 0
                                counters[method]["text_firing"][layer_idx][f_idx] += int(txt_counts[f_idx])

                            if pos_cnt[f_idx].item() > 0:
                                if layer_idx not in counters[method]["activation_sum"]:
                                    counters[method]["activation_sum"][layer_idx] = {}
                                if f_idx not in counters[method]["activation_sum"][layer_idx]:
                                    counters[method]["activation_sum"][layer_idx][f_idx] = 0.0
                                counters[method]["activation_sum"][layer_idx][f_idx] += float(pos_sum[f_idx])

                                if layer_idx not in counters[method]["activation_count"]:
                                    counters[method]["activation_count"][layer_idx] = {}
                                if f_idx not in counters[method]["activation_count"][layer_idx]:
                                    counters[method]["activation_count"][layer_idx][f_idx] = 0
                                counters[method]["activation_count"][layer_idx][f_idx] += int(pos_cnt[f_idx])

                    if layer_idx not in counters[method]["total_tokens"]:
                        counters[method]["total_tokens"][layer_idx] = 0
                    counters[method]["total_tokens"][layer_idx] += batch_masks.sum().item()

                    del (
                        batch_activations,
                        batch_masks,
                        feature_acts,
                        masked_feature_acts,
                        img_counts,
                        txt_counts,
                        pos_sum,
                        pos_cnt,
                    )
                    torch.cuda.empty_cache()

                del layer_activations

        del batch_input_ids, batch_attention_mask, batch_image_tensors
        torch.cuda.empty_cache()

    results = {}
    for method in methods:
        method_results = {}
        for layer_idx in range(from_layer, to_layer):
            if layer_idx in saes[method]:
                num_features = saes[method][layer_idx].cfg.d_sae
                layer_frequencies = {}

                for feature_idx in range(num_features):
                    firing_count = counters[method]["feature_firing"].get(layer_idx, {}).get(feature_idx, 0)
                    total_tokens = counters[method]["total_tokens"].get(layer_idx, 0)
                    frequency = firing_count / total_tokens if total_tokens > 0 else 0
                    layer_frequencies[feature_idx] = {
                        "firing_count": firing_count,
                        "total_tokens": total_tokens,
                        "frequency": frequency,
                        "log_frequency": np.log10(frequency + 1e-10),
                    }

                method_results[layer_idx] = layer_frequencies
        results[method] = method_results

    analysis_results = {
        "feature_firing_frequencies": results,
        "total_tokens_per_layer": {method: counters[method]["total_tokens"] for method in methods},
        "analysis_params": {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "from_layer": from_layer,
            "to_layer": to_layer,
            "caching_batch_size": caching_batch_size,
            "num_samples_analyzed": end_idx - start_idx,
            "methods": methods,
        },
    }

    output_path = Path(output_dir)
    results_path = output_path / f"feature_firing_analysis_{start_idx}_{end_idx}.json"
    with open(results_path, "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)

    for method in methods:
        basic_metrics = {
            "image_firing_counts": counters[method]["image_firing"],
            "text_firing_counts": counters[method]["text_firing"],
            "activation_sum": counters[method]["activation_sum"],
            "activation_count": counters[method]["activation_count"],
        }
        torch.save(basic_metrics, output_path / f"basic_metrics_{method}_{start_idx}_{end_idx}.pt")

        for layer_idx in counters[method]["feature_samples"]:
            trimmed_layer = {}
            for feat_id, sample_map in counters[method]["feature_samples"][layer_idx].items():
                top_items = sorted(sample_map.items(), key=lambda kv: kv[1], reverse=True)
                trimmed_layer[feat_id] = {s_idx: mag for s_idx, mag in top_items}

            layer_feature_data = {"feature_sample_firing": {layer_idx: trimmed_layer}}
            torch.save(
                layer_feature_data,
                output_path / f"feature_data_{method}_layer{layer_idx}_{start_idx}_{end_idx}.pt",
            )

        print(f"[INFO] Saved {method} metrics split by layer")

    try:
        wandb_run = wandb.init(project="sae-feature-firing", name=f"vqa_spatial_feature_firing_{start_idx}_{end_idx}")

        for method in methods:
            for layer_idx, layer_frequencies in results[method].items():
                frequencies = [freq["frequency"] for freq in layer_frequencies.values()]
                log_frequencies = [freq["log_frequency"] for freq in layer_frequencies.values()]

                wandb.log(
                    {
                        f"{method}/layer_{layer_idx}/mean_firing_frequency": np.mean(frequencies),
                        f"{method}/layer_{layer_idx}/median_firing_frequency": np.median(frequencies),
                        f"{method}/layer_{layer_idx}/std_firing_frequency": np.std(frequencies),
                        f"{method}/layer_{layer_idx}/dead_features": sum(1 for f in frequencies if f < 1e-6),
                        f"{method}/layer_{layer_idx}/dense_features": sum(1 for f in frequencies if f > 0.01),
                        f"{method}/layer_{layer_idx}/feature_frequency_histogram": wandb.Histogram(log_frequencies),
                    }
                )

        wandb.finish()
    except Exception as e:
        print(f"[WARN] Could not log to wandb: {e}")

    vlm_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    return analysis_results


def main():
    parser = argparse.ArgumentParser(description="Track SAE feature firing on spatial-only VQAv2 subset")
    parser.add_argument("--from-layer", type=int, default=0, help="First layer index (inclusive)")
    parser.add_argument("--to-layer", type=int, default=32, help="Last layer index (exclusive)")
    parser.add_argument("--start-sample", type=int, default=0, help="Starting sample index (in spatial subset)")
    parser.add_argument("--end-sample", type=int, default=50000, help="Ending sample index (in spatial subset)")
    parser.add_argument("--caching-batch-size", type=int, default=16, help="Batch size for processing")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/feature_firing_analysis/vqa_spatial",
        help="Output directory",
    )
    parser.add_argument(
        "--sae-checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing SAE checkpoints organized by method",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pretrained", "random", "text-only", "image-only"],
        help="SAE methods to analyze",
    )
    parser.add_argument(
        "--keywords-file",
        type=str,
        default=None,
        help="Optional path to newline-separated spatial keywords/phrases",
    )
    parser.add_argument(
        "--vqa-cache-dir",
        type=str,
        default=".cache/vqa_spatial_filter",
        help="Directory to read/write cached VQAv2 filtered indices",
    )
    parser.add_argument(
        "--vqa-split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="VQAv2 split to use",
    )
    args = parser.parse_args()

    print(f"[INFO] Spatial VQA analysis for samples {args.start_sample} → {args.end_sample}")
    print(f"[INFO] Layers {args.from_layer} → {args.to_layer}")
    print(f"[INFO] Methods: {args.methods}")
    print(f"[INFO] SAE checkpoints: {args.sae_checkpoint_dir}")

    vlm_tokenizer, vlm_model, vlm_image_processor = initialize_vlm_model("llava-more", device="cuda")
    vlm_model = NNsight(vlm_model)

    # Load spatial VQA dataset using unified loader
    dataset = load_vqa_spatial(
        split=args.vqa_split,
        keywords_file=args.keywords_file,
        keywords=None,
        cache_dir=args.vqa_cache_dir,
    )

    end_idx = min(args.end_sample, len(dataset))
    if args.start_sample >= end_idx:
        raise ValueError(
            f"start-sample ({args.start_sample}) must be < end-sample ({end_idx}) within filtered dataset of size {len(dataset)}"
        )

    print(
        f"[INFO] Spatial filter kept {len(dataset)} samples. Processing range: "
        f"{args.start_sample} → {end_idx} (count={end_idx - args.start_sample})."
    )

    results = track_feature_firing_chunk(
        vlm_tokenizer,
        vlm_model,
        vlm_image_processor,
        dataset,
        args.start_sample,
        end_idx,
        args.from_layer,
        args.to_layer,
        args.caching_batch_size,
        args.methods,
        args.sae_checkpoint_dir,
        args.output_dir,
    )

    print(f"[INFO] Spatial VQA analysis complete! Results saved to {args.output_dir}")

    for method in args.methods:
        print(f"\n=== {method.upper()} SAE ===")
        for layer_idx in range(args.from_layer, args.to_layer):
            if layer_idx in results["feature_firing_frequencies"].get(method, {}):
                layer_freqs = results["feature_firing_frequencies"][method][layer_idx]
                frequencies = [freq["frequency"] for freq in layer_freqs.values()]

                print(f"Layer {layer_idx}:")
                print(f"  Mean firing frequency: {np.mean(frequencies):.6f}")
                print(f"  Dead features (< 1e-6): {sum(1 for f in frequencies if f < 1e-6)}")
                print(f"  Dense features (> 0.01): {sum(1 for f in frequencies if f > 0.01)}")


if __name__ == "__main__":
    main()

