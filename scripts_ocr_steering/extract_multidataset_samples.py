#!/usr/bin/env python3
"""
Extract top-K samples per feature across VQA, VQA-spatial, and VSR datasets.
Matches the original LLaVA-MORE multi-dataset approach, adapted for PaliGemma2.

For each feature and each dataset, runs all samples through the model,
encodes text-only activations with the SAE, and saves top-K samples by magnitude.

Usage:
    python3 extract_multidataset_samples.py \
        --features /path/to/final_spatial_visual_features.csv \
        --gpus 0 1 2 3 4 5 6 7 \
        --top-k 20
"""

import os
import sys
import csv
import json
import re
import hashlib
import argparse
import warnings
import gc
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import numpy as np

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# ---- Config ----
MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384
D_IN = 2304
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
ANALYSIS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis")
HF_CACHE = "/data1/vlm_scope_sae_docci/hf_cache/hub"
IMAGE_CACHE_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/vlm_scope_sae_docci/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

VSR_DATASET = "cambridgeltl/vsr_random"
VQA_DATASET = "lmms-lab/VQAv2"

# Spatial keywords for VQA-spatial filtering (matches original)
SPATIAL_KEYWORDS = [
    "left", "right", "front", "back", "ahead", "behind", "forward", "backward",
    "forwards", "backwards", "up", "down", "upward", "downward",
    "top", "bottom", "upper", "lower", "leftmost", "rightmost", "topmost", "bottommost",
    "uppermost", "lowermost", "corner", "edge", "border", "side",
    "left side", "right side", "top side", "bottom side",
    "top left", "top right", "bottom left", "bottom right",
    "upper left", "upper right", "lower left", "lower right",
    "middle left", "middle right", "center left", "center right",
    "above", "over", "overhead", "atop", "on top", "on top of",
    "below", "under", "underneath", "beneath",
    "in front", "in front of", "at the front", "at the back",
    "next to", "beside", "alongside", "near", "nearby", "close to",
    "adjacent", "adjacent to", "across from", "opposite", "opposite to", "facing",
    "around", "surrounding", "encircling", "between", "in between", "among", "amid",
    "inside", "inside of", "outside", "outside of", "within",
    "to the left", "to the right", "to the left of", "to the right of",
    "distance", "closer", "closest", "nearest", "nearer",
    "far", "farther", "farthest", "further", "furthest",
    "height", "width",
    "vertical", "horizontal", "diagonal", "direction", "oriented", "orientation",
    "rotated", "rotation",
    "north", "south", "east", "west",
    "north east", "north west", "south east", "south west",
    "northeast", "northwest", "southeast", "southwest",
    "position", "positioned", "located", "location", "placement", "placed",
    "foreground", "background", "frontmost", "backmost", "background of",
]


def _compile_spatial_regex():
    escaped = []
    for kw in SPATIAL_KEYWORDS:
        kw = kw.strip()
        if not kw:
            continue
        parts = [re.escape(p) for p in kw.split()]
        if len(parts) == 1:
            pattern = parts[0]
        else:
            pattern = r"(?:\s+|-)".join(parts)
        escaped.append(rf"\b{pattern}\b")
    return re.compile("|".join(escaped), flags=re.IGNORECASE)


SPATIAL_RE = _compile_spatial_regex()


def _build_vsr_prompt(statement: str) -> str:
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\n"
        "Answer:"
    )


def _build_vqa_prompt(question: str) -> str:
    return f"{question.strip()}\n"


def _extraction_worker(gpu_id, layer_assignments, output_dir, top_k, datasets_to_use):
    """Worker: for each assigned layer, run all dataset samples through SAE."""
    import requests
    from PIL import Image
    from datasets import load_dataset, concatenate_datasets

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, "/home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2")
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    output_dir = Path(output_dir)

    if not layer_assignments:
        return

    # Load model
    print(f"[GPU{gpu_id}] Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE, local_files_only=True)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, local_files_only=True
    ).to(device).eval()
    nns_model = NNsight(model_raw)

    # Load datasets
    loaded_datasets = {}

    if "vsr" in datasets_to_use:
        print(f"[GPU{gpu_id}] Loading VSR (all splits)...", flush=True)
        data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        train_ds = load_dataset(VSR_DATASET, data_files=data_files, split="train")
        dev_ds = load_dataset(VSR_DATASET, data_files=data_files, split="dev")
        test_ds = load_dataset(VSR_DATASET, data_files=data_files, split="test")
        vsr = concatenate_datasets([train_ds, dev_ds, test_ds])
        print(f"[GPU{gpu_id}] VSR total: {len(vsr)} samples", flush=True)
        loaded_datasets["vsr"] = vsr

    if "vqa" in datasets_to_use or "vqa_spatial" in datasets_to_use:
        print(f"[GPU{gpu_id}] Loading VQA validation...", flush=True)
        vqa = load_dataset(VQA_DATASET, split="validation", cache_dir="/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache")
        print(f"[GPU{gpu_id}] VQA total: {len(vqa)} samples", flush=True)
        if "vqa" in datasets_to_use:
            loaded_datasets["vqa"] = vqa

        if "vqa_spatial" in datasets_to_use:
            # Filter to spatial subset
            spatial_indices = []
            for i in range(len(vqa)):
                q = str(vqa[i].get("question", ""))
                if SPATIAL_RE.search(q):
                    spatial_indices.append(i)
            print(f"[GPU{gpu_id}] VQA-spatial: {len(spatial_indices)} samples (from {len(vqa)})", flush=True)
            loaded_datasets["vqa_spatial"] = (vqa, spatial_indices)

    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_image(source, sample):
        """Load image from different dataset types."""
        if source == "vsr":
            url = sample.get("image_link", "")
            if not url.startswith("http"):
                return None
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_path = IMAGE_CACHE_DIR / f"{url_hash}.jpg"
            try:
                if cache_path.exists():
                    return Image.open(cache_path).convert("RGB")
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img.save(cache_path, "JPEG")
                return img
            except Exception:
                return None
        elif source in ("vqa", "vqa_spatial"):
            try:
                return sample["image"].convert("RGB")
            except Exception:
                return None
        return None

    def _get_prompt_and_text(source, sample):
        """Get prompt and text for a sample."""
        if source == "vsr":
            caption = str(sample.get("caption", "")).strip()
            return _build_vsr_prompt(caption), caption
        elif source in ("vqa", "vqa_spatial"):
            question = str(sample.get("question", "")).strip()
            return _build_vqa_prompt(question), question
        return "", ""

    # Process each layer
    for layer_idx, feature_indices in layer_assignments:
        print(f"[GPU{gpu_id}] Layer {layer_idx}: {len(feature_indices)} features", flush=True)

        # Load SAE
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()

        # Process each dataset
        for ds_name in datasets_to_use:
            if ds_name not in loaded_datasets:
                continue

            if ds_name == "vqa_spatial":
                base_ds, spatial_idx = loaded_datasets["vqa_spatial"]
                ds_len = len(spatial_idx)
            else:
                base_ds = loaded_datasets[ds_name]
                ds_len = len(base_ds)

            # Track top-K per feature for this dataset
            top_samples = {fi: [] for fi in feature_indices}

            print(f"[GPU{gpu_id}] Layer {layer_idx} / {ds_name}: {ds_len} samples", flush=True)

            for vi in range(ds_len):
                # Get actual index into dataset
                if ds_name == "vqa_spatial":
                    actual_idx = spatial_idx[vi]
                    ex = base_ds[actual_idx]
                else:
                    actual_idx = vi
                    ex = base_ds[vi]

                img = _load_image(ds_name, ex)
                if img is None:
                    continue

                prompt, text = _get_prompt_and_text(ds_name, ex)

                try:
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        img, prompt, processor, model_raw, device=device)
                    _, img_end = get_image_token_positions(input_ids)

                    with nns_model.trace(
                        input_ids=input_ids, attention_mask=attn_mask,
                        pixel_values=pixel_values,
                    ) as tr:
                        layer_out = nns_model.model.language_model.layers[layer_idx].output[0][0].save()

                    text_acts = layer_out[img_end:].detach().float()
                    if text_acts.shape[0] == 0:
                        continue

                    with torch.no_grad():
                        codes = sae.encode(text_acts)

                    for fi in feature_indices:
                        feat_acts = codes[:, fi]
                        max_mag = feat_acts.max().item()

                        if max_mag > 0:
                            heap = top_samples[fi]
                            entry = {
                                "dataset": ds_name,
                                "sample_idx": vi,  # Index into this dataset (or spatial subset)
                                "base_idx": actual_idx,  # Index into base dataset
                                "magnitude": max_mag,
                                "text": text,
                            }
                            # VSR-specific fields
                            if ds_name == "vsr":
                                entry["caption"] = text
                                entry["relation"] = str(ex.get("relation", "")).strip()
                                entry["label"] = int(ex.get("label", 0))

                            if len(heap) < top_k:
                                heap.append(entry)
                            elif max_mag > min(h["magnitude"] for h in heap):
                                min_idx = min(range(len(heap)), key=lambda i: heap[i]["magnitude"])
                                heap[min_idx] = entry

                except Exception as e:
                    if vi < 3:
                        print(f"  [ERROR] GPU{gpu_id} L{layer_idx}/{ds_name} sample {vi}: {e}", flush=True)
                    continue

                if (vi + 1) % 2000 == 0:
                    print(f"  [GPU{gpu_id}] L{layer_idx}/{ds_name}: {vi+1}/{ds_len}", flush=True)
                    torch.cuda.empty_cache()

            # Save per-feature per-dataset
            ds_dir = output_dir / f"{ds_name}" / f"layer_{layer_idx}"
            ds_dir.mkdir(parents=True, exist_ok=True)
            for fi in feature_indices:
                samples = sorted(top_samples[fi], key=lambda x: -x["magnitude"])
                feat_dir = ds_dir / f"text-only_layer_{layer_idx}_feature_{fi}"
                feat_dir.mkdir(parents=True, exist_ok=True)
                with open(feat_dir / "sample_info.json", "w") as f:
                    json.dump(samples, f, indent=2)

            print(f"[GPU{gpu_id}] L{layer_idx}/{ds_name}: DONE.", flush=True)
            del top_samples
            torch.cuda.empty_cache()

        del sae
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[GPU{gpu_id}] Layer {layer_idx}: ALL DATASETS DONE.", flush=True)

    print(f"[GPU{gpu_id}] All layers done.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV with layer,feature columns")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["vqa", "vqa_spatial", "vsr"],
                        choices=["vqa", "vqa_spatial", "vsr"])
    args = parser.parse_args()

    # Load features grouped by layer
    layer_features = defaultdict(list)
    with open(args.features) as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_features[int(row["layer"])].append(int(row["feature"]))

    total_features = sum(len(v) for v in layer_features.values())
    print(f"Loaded {total_features} features across {len(layer_features)} layers")
    print(f"Datasets: {args.datasets}")

    out_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR / "multidataset_feature_samples"

    # Distribute layers across GPUs (balance by feature count)
    layers_sorted = sorted(layer_features.items(), key=lambda x: -len(x[1]))
    gpu_assignments = [[] for _ in args.gpus]
    gpu_loads = [0] * len(args.gpus)

    for layer_idx, feats in layers_sorted:
        min_gpu = min(range(len(args.gpus)), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append((layer_idx, feats))
        gpu_loads[min_gpu] += len(feats)

    for i, gpu_id in enumerate(args.gpus):
        n_layers = len(gpu_assignments[i])
        n_feats = gpu_loads[i]
        print(f"  GPU {gpu_id}: {n_layers} layers, {n_feats} features")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for i, gpu_id in enumerate(args.gpus):
        p = mp.Process(target=_extraction_worker,
                       args=(gpu_id, gpu_assignments[i], str(out_dir), args.top_k, args.datasets))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Summary
    for ds_name in args.datasets:
        ds_dir = out_dir / ds_name
        if ds_dir.exists():
            total_saved = sum(1 for _ in ds_dir.rglob("sample_info.json"))
            print(f"  {ds_name}: {total_saved} feature files")
    print(f"\nExtraction complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
