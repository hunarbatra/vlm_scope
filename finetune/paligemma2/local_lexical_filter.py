#!/usr/bin/env python3
"""
Lexical Artifact Filter for PaliGemma2 spatial features.

Tests whether candidate spatial features fire due to IMAGE content (visual)
or TEXT content (lexical artifact). For each candidate feature:
1. Run random VQA images through PaliGemma2 with GENERIC (non-spatial) prompts
2. Encode activations through SAE
3. Check if feature fires on image tokens
4. If it fires → feature is visually grounded → KEEP
5. If it never fires → feature was text-driven → REMOVE

Usage:
    HF_TOKEN=hf_... python local_lexical_filter.py \
        --features-csv analysis_results/final_spatial_visual_features.csv \
        --n-images 200 --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import os
import sys
import json
import gc
import csv
import ast
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    process_vlm_inputs,
    get_image_token_positions,
    initialize_jumprelu_sae,
)

GENERIC_PROMPTS = [
    "Describe how the items are arranged.",
    "Comment on the overall layout and organization of the scene.",
    "Summarize the structure in terms of grouping or separation.",
    "Explain the relative positioning of objects without naming directions.",
    "Describe patterns of arrangement, such as order or symmetry.",
]

SAE_REPO = "hunarbatra/vlm_scope_paligemma2_sae"
MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "google/paligemma2-3b-ft-docci-448")
CHECKPOINT_DIR = Path(os.environ.get("SAE_CHECKPOINT_DIR", "/data1/vlm_scope_sae_docci/checkpoints"))
D_SAE = 16384
N_LAYERS = 26


def get_sae_checkpoint_path(layer_idx: int) -> str:
    """Get SAE checkpoint path — prefer local checkpoints, fall back to HF."""
    local_path = CHECKPOINT_DIR / f"pretrained_layer_{layer_idx}.pt"
    if local_path.exists():
        return str(local_path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(SAE_REPO, f"jumprelu/pretrained/pretrained_layer_{layer_idx}.pt")


def process_layer_batch(
    gpu_id: int,
    layer_indices: list,
    candidate_features: dict,  # layer -> set of feature indices
    images: list,  # PIL images
    results_dir: Path,
    min_fire_fraction: float = 0.01,
):
    """Process assigned layers on one GPU.

    For each layer:
    1. Load SAE checkpoint
    2. For each image × generic prompt combo:
       - Run through PaliGemma2, extract activations at this layer
       - Encode through SAE
       - Check which candidate features fire on image tokens
    3. Feature passes if it fires on image tokens in >= min_fire_fraction of tests
    """
    from nnsight import NNsight
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading PaliGemma2...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    nns_model = NNsight(model_raw)

    results = {}

    for layer_idx in layer_indices:
        if layer_idx not in candidate_features or not candidate_features[layer_idx]:
            continue

        feat_set = candidate_features[layer_idx]
        feat_list = sorted(feat_set)
        print(f"[GPU {gpu_id}] Layer {layer_idx}: {len(feat_list)} candidate features")

        # Load SAE
        ckpt_path = get_sae_checkpoint_path(layer_idx)
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=ckpt_path, device=device)
        sae.eval()

        # Track: for each feature, how many image×prompt combos it fires on image tokens
        fire_count = defaultdict(int)
        total_tests = 0

        for img_idx, image in enumerate(tqdm(images, desc=f"GPU{gpu_id} L{layer_idx}")):
            for prompt in GENERIC_PROMPTS:
                try:
                    input_ids, attention_mask, pixel_values = process_vlm_inputs(
                        image, prompt, processor, model_raw, device=device
                    )
                    img_start, img_end = get_image_token_positions(input_ids)

                    with torch.no_grad():
                        with nns_model.trace(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                        ) as tr:
                            layer_out = nns_model.model.language_model.layers[layer_idx].output[0]
                            saved_act = layer_out[0].detach().cpu().save()

                    act = saved_act
                    if isinstance(act, tuple):
                        act = act[0]
                    act = act.float().to(device)

                    with torch.no_grad():
                        codes = sae.encode(act)  # (seq, d_sae)

                    # Check which candidate features fire on image tokens
                    if img_end > img_start:
                        img_codes = codes[img_start:img_end]  # (n_img_tokens, d_sae)
                        # For each candidate feature, check if any image token fires
                        for fi in feat_list:
                            if (img_codes[:, fi] != 0).any():
                                fire_count[fi] += 1

                    total_tests += 1

                except Exception as e:
                    if total_tests < 3:
                        print(f"  [WARN] img {img_idx}: {type(e).__name__}: {e}")
                    continue

        # Determine which features pass
        layer_results = []
        for fi in feat_list:
            n_fires = fire_count.get(fi, 0)
            fraction = n_fires / max(total_tests, 1)
            passed = fraction >= min_fire_fraction
            layer_results.append({
                "layer": layer_idx,
                "feature": fi,
                "generic_fire_count": n_fires,
                "total_tests": total_tests,
                "fire_fraction": fraction,
                "passed_lexical_filter": passed,
            })

        results[layer_idx] = layer_results
        n_passed = sum(1 for r in layer_results if r["passed_lexical_filter"])
        print(f"[GPU {gpu_id}] L{layer_idx}: {n_passed}/{len(feat_list)} features passed "
              f"({total_tests} tests, min_frac={min_fire_fraction})")

        # Save per-layer results
        out_path = results_dir / f"lexical_filter_layer_{layer_idx}.json"
        with open(out_path, "w") as f:
            json.dump(layer_results, f, indent=2)

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, required=True,
                        help="CSV of candidate features (from Fisher test)")
    parser.add_argument("--n-images", type=int, default=200,
                        help="Number of VQA images to test")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--min-fire-fraction", type=float, default=0.01,
                        help="Min fraction of tests where feature must fire on image tokens")
    parser.add_argument("--results-dir", type=str,
                        default="analysis_results/lexical_filter",
                        help="Output directory")
    parser.add_argument("--vqa-split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load candidate features
    print(f"Loading candidate features from {args.features_csv}...")
    df = pd.read_csv(args.features_csv)
    candidate_features = defaultdict(set)
    for _, row in df.iterrows():
        candidate_features[int(row["layer"])].add(int(row["feature"]))
    total_candidates = sum(len(v) for v in candidate_features.values())
    print(f"  {total_candidates} features across {len(candidate_features)} layers")

    # Load VQA images via streaming (avoids downloading full dataset)
    print(f"Loading VQA {args.vqa_split} images via streaming...")
    vqa = load_dataset("lmms-lab/VQAv2", split=args.vqa_split, streaming=True)
    images = []
    for i, sample in enumerate(tqdm(vqa, desc="Loading images", total=args.n_images)):
        if i >= args.n_images:
            break
        try:
            img = sample["image"].convert("RGB")
            images.append(img)
        except Exception:
            continue
    print(f"  Loaded {len(images)} images")

    # Distribute layers across GPUs
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    layers_with_features = sorted(candidate_features.keys())

    if len(gpu_ids) == 1:
        # Single GPU — process all layers sequentially
        results = process_layer_batch(
            gpu_ids[0], layers_with_features, candidate_features,
            images, results_dir, args.min_fire_fraction
        )
    else:
        # Multi-GPU — use multiprocessing
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        # Distribute layers round-robin
        assignments = {g: [] for g in gpu_ids}
        for i, layer in enumerate(layers_with_features):
            assignments[gpu_ids[i % len(gpu_ids)]].append(layer)

        print("\nGPU assignments:")
        for g, layers in assignments.items():
            n_feats = sum(len(candidate_features[l]) for l in layers)
            print(f"  GPU {g}: layers {layers} ({n_feats} features)")

        # Run in parallel
        processes = []
        for gpu_id, layers in assignments.items():
            if not layers:
                continue
            p = mp.Process(
                target=process_layer_batch,
                args=(gpu_id, layers, candidate_features, images,
                      results_dir, args.min_fire_fraction),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # Aggregate results
    print("\n" + "=" * 60)
    print("Aggregating lexical filter results...")
    all_results = []
    for layer_idx in range(N_LAYERS):
        path = results_dir / f"lexical_filter_layer_{layer_idx}.json"
        if path.exists():
            with open(path) as f:
                all_results.extend(json.load(f))

    if all_results:
        results_df = pd.DataFrame(all_results)
        passed = results_df[results_df["passed_lexical_filter"]]
        failed = results_df[~results_df["passed_lexical_filter"]]

        print(f"\nTotal tested: {len(results_df)}")
        print(f"Passed lexical filter: {len(passed)}")
        print(f"Filtered out: {len(failed)}")

        # Save combined results
        results_df.to_csv(results_dir / "lexical_filter_all.csv", index=False)
        passed.to_csv(results_dir / "features_passed_lexical.csv", index=False)

        # Merge back with original spatial features to get full info
        spatial_df = pd.read_csv(args.features_csv)
        final = spatial_df.merge(
            passed[["layer", "feature", "fire_fraction"]],
            on=["layer", "feature"],
            how="inner"
        )
        final.to_csv(results_dir / "final_features_lexical_filtered.csv", index=False)
        print(f"\nFinal features (spatial ∩ adapted ∩ lexical): {len(final)}")
        print(f"Results saved to {results_dir}/")

        # Per-layer summary
        print(f"\n{'Layer':<8} {'Tested':>8} {'Passed':>8} {'Rate':>8}")
        for l in sorted(results_df["layer"].unique()):
            tested = len(results_df[results_df["layer"] == l])
            p = len(passed[passed["layer"] == l])
            print(f"  L{l:<5} {tested:>8} {p:>8} {p/tested*100:>7.1f}%")
    else:
        print("No results found!")


if __name__ == "__main__":
    main()
