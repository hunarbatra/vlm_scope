#!/usr/bin/env python3
"""
Ablate one or multiple attention heads in the VLM and evaluate accuracy on VSR.

Example:

#   --relations "left of,right of,on top of" \

CUDA_VISIBLE_DEVICES=2 python ablation/ablate_head_vsr.py \
  --pairs L5H17,L13H0,L13H1,L13H18,L8H29 \
  --split train --max-samples 1000 --seed 0 --relations "at the back of"\
  --cache-dir /scratch/local/ssd/lachin/vsr_image_cache

CUDA_VISIBLE_DEVICES=4 python ablation/ablate_head_vsr.py \
  --pairs L13H1,L13H0,L15H10,L13H18 \
  --split train \
  --mean-split dev \
  --max-samples 1000 \
  --num-mean-examples 128 \
  --cache-dir /scratch/local/ssd/lachin/vsr_image_cache \
  --relations "above,below,beneath,under,on top of,in front of,behind,at the back of,left of,right of,at the left side of,at the right side of,touching" \

CUDA_VISIBLE_DEVICES=6 python ablation/ablate_head_vsr.py \
  --pairs L15H10,L13H0,L13H1,L13H18,L12H12,L15H21,L18H22,L11H3,L7H6,L9H17 \
  --split train \
  --mean-split dev \
  --max-samples 1000 \
  --num-mean-examples 128 \
  --cache-dir /scratch/local/ssd/lachin/vsr_image_cache

CUDA_VISIBLE_DEVICES=6 python ablation/ablate_head_vsr.py \
  --pairs L13H0,L13H1,L13H18,L13H28,L13H27,L13H26,L13H25,L13H24,L13H23,L13H22,L13H21,L13H20,L13H19,L13H17,L13H16,L13H15,L13H14,L13H13,L13H12,L13H11,L13H10,L13H9,L13H8,L13H7,L13H6,L13H5,L13H4,L13H3,L13H2,L13H1 \
  --split train \
  --mean-split dev \
  --max-samples 1000 \
  --num-mean-examples 128 \
  --cache-dir /scratch/local/ssd/lachin/vsr_image_cache

CUDA_VISIBLE_DEVICES=1 python ablation/ablate_head_vsr.py \
  --pairs L13H1,L13H0,L18H22,L15H10,L13H18,L12H12 \
  --split train \
  --mean-split dev \
  --max-samples 1000 \
  --num-mean-examples 128 \
  --cache-dir /scratch/local/ssd/lachin/vsr_image_cache \
 --relations "below,under,on top of,behind,beneath,next to"
  


  
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, List, Set
import re
import json
import hashlib
from pathlib import Path

import dotenv
dotenv.load_dotenv(".env")

import torch
from PIL import Image
import requests
from io import BytesIO

import sys
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    get_image_token_positions,
)
sys.path.pop()  # Remove the finetune/vqa path
from datasets import load_dataset
from nnsight import NNsight

PROGRESS_INTERVAL = 100


def load_vsr(split: str = "train", only_true: bool = False):
    """Load VSR dataset split via huggingface datasets."""
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    if split == "train+dev+test":
        # Load all three splits
        train_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="train")
        dev_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="dev")
        test_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="test")
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([train_ds, dev_ds, test_ds])
    else:
        dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split=split)
    if only_true:
        dataset = dataset.filter(lambda x: x["label"] == 1)
    return dataset


def load_image_with_cache(url: str, cache_dir: str = "/scratch/local/ssd/lachin/vsr_image_cache", timeout: int = 10) -> Image.Image:
    """Load image with disk caching to avoid re-downloading."""
    os.makedirs(cache_dir, exist_ok=True)
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{url_hash}.jpg")
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load cached image {cache_path}: {e}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        try:
            img.save(cache_path, "JPEG")
        except Exception as e:
            print(f"[WARN] Failed to cache image: {e}")
        return img
    except Exception as e:
        print(f"[WARN] Failed to download image {url}: {e}")
        return Image.new("RGB", (224, 224), (128, 128, 128))




def build_prompt(statement: str) -> str:
    s = statement.strip()
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {s}\n"
        "Answer:"
    )


def _parse_lh_pairs_arg(pairs_arg: str) -> List[Tuple[int, int]]:
    """Parse comma-separated tokens like 'L13H1,L7H2' into [(13,1),(7,2)]."""
    if not pairs_arg:
        return []
    tokens = [t.strip() for t in pairs_arg.split(',') if t.strip()]
    pairs: List[Tuple[int, int]] = []
    for tok in tokens:
        m = re.fullmatch(r"[Ll](\d+)[Hh](\d+)", tok)
        if not m:
            raise ValueError(f"Invalid layer-head token: {tok}. Expected form like L13H1")
        layer_val = int(m.group(1))
        head_val = int(m.group(2))
        pairs.append((layer_val, head_val))
    return pairs


def _first_token_ids(tokenizer, texts):
    """Get first token IDs for a list of texts."""
    ids = set()
    for t in texts:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            ids.add(toks[0])  # only the first token
    return ids

def _build_yes_no_token_sets(tokenizer) -> Tuple[Set[int], Set[int]]:
    """Build disjoint sets of first-token IDs for yes/no responses."""
    # leading-space and no-space variants to be safe
    yes_first = _first_token_ids(tokenizer, [" Yes", "Yes", " yes", "YES"])
    no_first = _first_token_ids(tokenizer, [" No", "No", " no", "NO"])
    # ensure disjointness
    overlap = yes_first & no_first
    if overlap:
        yes_first -= overlap
        no_first -= overlap
    return yes_first, no_first


@torch.inference_mode()
def _next_token_yes_no_probs(image, prompt, image_processor, vlm_model, vlm_tokenizer, yes_ids: Set[int], no_ids: Set[int]) -> Tuple[float, float]:
    """Compute P(next token is a Yes/No variant) from logits at last input position."""
    input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
        image, prompt, image_processor, vlm_model, vlm_tokenizer, json_mode=False
    )
    out = vlm_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        image_sizes=image_sizes,
        use_cache=False,
    )
    logits = out.logits[:, -1, :]  # (1, vocab)
    probs = torch.softmax(logits, dim=-1)[0]

    yes_mass = probs[list(yes_ids)].sum() if yes_ids else torch.tensor(0.0, device=probs.device)
    no_mass = probs[list(no_ids)].sum() if no_ids else torch.tensor(0.0, device=probs.device)
    denom = yes_mass + no_mass
    if float(denom) == 0.0:
        # fall back to raw mass (keeps old behavior) rather than crash
        p_yes = float(yes_mass)
        p_no = float(no_mass)
    else:
        p_yes = float(yes_mass / denom)
        p_no = float(no_mass / denom)
    return p_yes, p_no


def _evaluate_pass(
    ds,
    indices: List[int],
    cache_dir: str,
    tokenizer,
    model,
    image_processor,
) -> Tuple[float, int, int, float, int]:
    """Evaluate one pass: returns (acc, correct, total, mean_p_correct, count_p)."""
    total = 0
    correct = 0
    sum_p_correct = 0.0
    count_p = 0

    yes_ids, no_ids = _build_yes_no_token_sets(tokenizer)

    for i, idx in enumerate(indices):
        if i % PROGRESS_INTERVAL == 0:
            print(f"[INFO] Processing sample {i}/{len(indices)}...")

        ex = ds[idx]
        image_url = ex.get("image_link") or ex.get("image") or ""
        statement = str(ex.get("caption") or ex.get("text") or "").strip()
        label = int(ex.get("label", 0))

        if not statement:
            continue

        image = load_image_with_cache(image_url, cache_dir=cache_dir)
        prompt = build_prompt(statement)

        # Compute yes/no probabilities
        try:
            p_yes, p_no = _next_token_yes_no_probs(image, prompt, image_processor, model, tokenizer, yes_ids, no_ids)
        except Exception:
            p_yes, p_no = 0.0, 0.0

        # Probability-based decision
        pred = 1 if float(p_yes) > float(p_no) else 0
        total += 1
        if pred == label:
            correct += 1

        # Always aggregate probability metric for reporting
        try:
            p_correct = p_yes if label == 1 else p_no
            sum_p_correct += float(p_correct)
            count_p += 1
        except Exception:
            pass

        if (i + 1) % PROGRESS_INTERVAL == 0:
            torch.cuda.empty_cache()

    acc = (correct / total) if total > 0 else 0.0
    mean_p = (sum_p_correct / count_p) if count_p > 0 else 0.0
    return acc, correct, total, mean_p, count_p


def evaluate_with_ablation(
    split: str,
    max_samples: int | None,
    cache_dir: str,
    layer_head_pairs: List[Tuple[int, int]],
    scale: float,
    seed: int | None,
    relations: List[str] | None,
    num_mean_examples: int,
    mean_split: str,
) -> Tuple[Tuple[float, int, int, float, int], Tuple[float, int, int, float, int]]:
    """Run paired baseline (clean) and ablated evaluations on the same samples."""
    disp_tokens = [f"L{l}H{h}" for (l, h) in layer_head_pairs]
    print(f"[INFO] Starting VSR evaluation with paired baseline and ablation ({','.join(disp_tokens)} target=o_proj, decision=prob)")

    # Model
    print("[INFO] Initializing VLM model...")
    tokenizer, base_model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    model = NNsight(base_model)
    model.eval()
    print("[INFO] VLM model loaded and ready!")

    # Data
    print("[INFO] Loading VSR dataset...")
    ds = load_vsr(split=split, only_true=False)
    print(f"[INFO] Loaded {len(ds)} VSR samples")
    if relations:
        allowed = set([r.strip() for r in relations if r.strip()])
        if len(allowed) > 0:
            print(f"[INFO] Filtering to relations: {sorted(list(allowed))}")
            ds = ds.filter(lambda x: x["relation"] in allowed)
            print(f"[INFO] After filtering: {len(ds)} samples")

    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    indices = list(range(len(ds)))
    if seed is not None:
        import random
        rnd = random.Random(seed)
        rnd.shuffle(indices)
    indices = indices[:n]
    print(f"[INFO] Evaluating {n} paired samples...")

    # Baseline cache key (model, split, n, seed, relations, decision)
    model_id = str(getattr(base_model.config, "_name_or_path", "unknown")).lower()
    key_obj = {
        "model": model_id,
        "split": split,
        "n": n,
        "seed": seed,
        "relations": sorted(relations) if relations else [],
        "decision": "prob",
    }
    key_hash = hashlib.md5(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()
    baseline_cache_dir = Path("results/stage6/vsr_baseline_cache")
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache_path = baseline_cache_dir / f"{key_hash}.json"

    # Clean pass with caching
    if baseline_cache_path.exists():
        try:
            with open(baseline_cache_path, "r") as f:
                cached = json.load(f)
            cached_indices = cached.get("indices", [])
            cached_metrics = cached.get("metrics", None)
            if isinstance(cached_indices, list) and cached_metrics:
                indices = cached_indices
                clean_metrics = tuple(cached_metrics)
                print(f"[INFO] Loaded cached baseline (key={key_hash}, N={len(indices)})")
            else:
                raise ValueError("Malformed cache; will recompute baseline")
        except Exception as e:
            print(f"[WARN] Failed to load baseline cache: {e}; recomputing baseline")
            print("[INFO] Running clean (no hook) pass...")
            clean_metrics = _evaluate_pass(ds, indices, cache_dir, tokenizer, base_model, image_processor)
            try:
                with open(baseline_cache_path, "w") as f:
                    json.dump({"key": key_obj, "indices": indices, "metrics": list(clean_metrics)}, f)
                print(f"[INFO] Saved baseline cache → {baseline_cache_path}")
            except Exception as e2:
                print(f"[WARN] Failed to write baseline cache: {e2}")
    else:
        print("[INFO] Running clean (no hook) pass...")
        clean_metrics = _evaluate_pass(ds, indices, cache_dir, tokenizer, base_model, image_processor)
        try:
            with open(baseline_cache_path, "w") as f:
                json.dump({"key": key_obj, "indices": indices, "metrics": list(clean_metrics)}, f)
            print(f"[INFO] Saved baseline cache → {baseline_cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to write baseline cache: {e}")

    # Precompute per-layer mean activations of o_proj.input across a few examples (Method 4)
    print(f"[INFO] Computing per-layer mean o_proj.input across {num_mean_examples} examples from {mean_split} split for head patching...")
    from collections import defaultdict
    mean_attn_acts: dict[int, torch.Tensor] = {}
    for l in range(base_model.config.num_hidden_layers):
        mean_attn_acts[l] = torch.zeros(
            base_model.config.hidden_size,
            dtype=torch.float16,
            device=base_model.device,
        )

    # Load mean dataset (separate from eval dataset)
    mean_ds = load_vsr(split=mean_split, only_true=False)
    if relations:
        allowed = set([r.strip() for r in relations if r.strip()])
        if len(allowed) > 0:
            mean_ds = mean_ds.filter(lambda x: x["relation"] in allowed)

    # Collect means over first num_mean_examples samples (or fewer if dataset is small)
    collect_n = min(num_mean_examples, len(mean_ds))
    for i in range(collect_n):
        ex_i = mean_ds[i]
        image_url_i = ex_i.get("image_link") or ex_i.get("image") or ""
        statement_i = str(ex_i.get("caption") or ex_i.get("text") or "").strip()
        if not statement_i:
            continue
        image_i = load_image_with_cache(image_url_i, cache_dir=cache_dir)
        prompt_i = build_prompt(statement_i)
        input_ids_i, attention_mask_i, image_tensor_i, image_sizes_i = process_vlm_inputs(
            image_i, prompt_i, image_processor, base_model, tokenizer
        )
        img_start_i, img_end_i = get_image_token_positions(input_ids_i)
        img_start_i = int(img_start_i)
        img_end_i = int(img_end_i)

        local_means: dict[int, any] = {}
        with torch.no_grad():
            with model.trace(
                input_ids=input_ids_i,
                attention_mask=attention_mask_i,
                images=image_tensor_i,
                image_sizes=image_sizes_i,
            ) as tr_mean:
                for l in range(base_model.config.num_hidden_layers):
                    attn_input = model.model.layers[l].self_attn.o_proj.input[0, :, :]
                    local_means[l] = attn_input.mean(dim=0).save()

        for l in range(base_model.config.num_hidden_layers):
            mean_attn_acts[l] += local_means[l]

        # Cleanup tensors to be safe
        del input_ids_i, attention_mask_i, image_tensor_i, image_sizes_i, local_means

    # Normalize
    for l in range(base_model.config.num_hidden_layers):
        if collect_n > 0:
            mean_attn_acts[l] /= collect_n
    
    print(f"[DEBUG] Computed means from {collect_n} examples. Mean norm for layer 15: {mean_attn_acts[15].norm().item():.4f}")

    # Ablated pass via NNsight trace using mean head patching (Method 4)
    print("[INFO] Running ablated pass with mean head patching (no hooks)...")
    layer_to_heads: dict[int, List[int]] = defaultdict(list)
    for l, h in layer_head_pairs:
        layer_to_heads[int(l)].append(int(h))

    total = 0
    correct = 0
    sum_p_correct = 0.0
    count_p = 0
    yes_ids, no_ids = _build_yes_no_token_sets(tokenizer)

    for i, idx in enumerate(indices):
        if i % PROGRESS_INTERVAL == 0:
            print(f"[INFO] (ablated) Processing sample {i}/{len(indices)}...")

        ex = ds[idx]
        image_url = ex.get("image_link") or ex.get("image") or ""
        statement = str(ex.get("caption") or ex.get("text") or "").strip()
        label = int(ex.get("label", 0))
        if not statement:
            continue

        image = load_image_with_cache(image_url, cache_dir=cache_dir)
        prompt = build_prompt(statement)
        input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
            image, prompt, image_processor, base_model, tokenizer
        )
        image_start, image_end = get_image_token_positions(input_ids)
        image_start = int(image_start)
        image_end = int(image_end)

        logits_saved = None
        try:
            with model.trace(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=image_sizes,
            ) as tr:
                for layer_idx, heads in layer_to_heads.items():
                    heads_sorted = sorted(set(heads))
                    for h in heads_sorted:
                        head_dim = int(base_model.config.head_dim)
                        head_start = h * head_dim
                        head_end = (h + 1) * head_dim

                        mean_act = mean_attn_acts[layer_idx][head_start:head_end]
                        zero_act = torch.zeros_like(mean_act)
                        model.model.layers[layer_idx].self_attn.o_proj.input[0, :, head_start:head_end] = zero_act

                logits_saved = model.output.logits.save()

            output_logits = logits_saved
            logits_last = output_logits[:, -1, :]
            probs_last = torch.softmax(logits_last, dim=-1)

            yes_mass = probs_last[0, list(yes_ids)].sum() if yes_ids else torch.tensor(0.0, device=probs_last.device)
            no_mass = probs_last[0, list(no_ids)].sum() if no_ids else torch.tensor(0.0, device=probs_last.device)
            denom = yes_mass + no_mass
            if float(denom) == 0.0:
                # fall back to raw mass (keeps old behavior) rather than crash
                p_yes = float(yes_mass)
                p_no = float(no_mass)
            else:
                p_yes = float(yes_mass / denom)
                p_no = float(no_mass / denom)

        except Exception as e:
            print(f"[ERROR] (ablated) Sample {i}: Trace/forward failed: {e}")
            p_yes, p_no = 0.0, 0.0

        # Probability-based decision
        pred = 1 if float(p_yes) > float(p_no) else 0
        total += 1
        if pred == label:
            correct += 1

        # Always aggregate probability metric for reporting
        try:
            p_correct = p_yes if label == 1 else p_no
            sum_p_correct += float(p_correct)
            count_p += 1
        except Exception:
            pass

        if (i + 1) % PROGRESS_INTERVAL == 0:
            torch.cuda.empty_cache()

    ablated_metrics = (
        (correct / total) if total > 0 else 0.0,
        correct,
        total,
        (sum_p_correct / count_p) if count_p > 0 else 0.0,
        count_p,
    )

    return clean_metrics, ablated_metrics


def main():
    print("[INFO] Starting VSR head-ablation evaluation script...")

    ap = argparse.ArgumentParser(description="Ablate a specific attention head and evaluate on VSR")
    ap.add_argument("--pairs", type=str, required=True, help="Comma-separated layer-head tokens, e.g., L13H1,L7H2")
    ap.add_argument("--scale", type=float, default=0.0, help="Scale factor for the head subspace (0.0=ablate)")

    ap.add_argument("--split", type=str, default="train", choices=["train", "dev", "test", "train+dev+test"], help="VSR split")
    ap.add_argument("--max-samples", type=int, default=1000, help="Max samples to evaluate (0 for all)")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed to shuffle paired indices deterministically")
    ap.add_argument("--relations", type=str, default="", help="Comma-separated VSR relations to include (optional)")
    ap.add_argument("--cache-dir", type=str, default="/scratch/local/ssd/lachin/vsr_image_cache", help="Directory to cache downloaded images")
    ap.add_argument("--num-mean-examples", type=int, default=5, help="Number of samples to compute mean head activations for Method 4 patching")
    ap.add_argument("--mean-split", type=str, default=None, choices=["train", "dev", "test", "train+dev+test"], help="Split to compute mean activations from (default: same as --split)")
    args = ap.parse_args()

    max_samples = None if args.max_samples == 0 else args.max_samples
    relations = [r.strip() for r in args.relations.split(",") if r.strip()] if args.relations else None
    # Build layer-head pairs strictly from --pairs
    layer_head_pairs: List[Tuple[int, int]] = _parse_lh_pairs_arg(args.pairs)
    # De-duplicate pairs and sort for stable display
    layer_head_pairs = sorted(set(layer_head_pairs))
    if len(layer_head_pairs) == 0:
        raise ValueError("Must specify --pairs with tokens like L13H1")

    mean_split = args.mean_split if args.mean_split else args.split
    print(f"[INFO] Configuration: pairs={[f'L{l}H{h}' for (l,h) in layer_head_pairs]}, target=o_proj, split={args.split}, mean_split={mean_split}, max_samples={max_samples}, seed={args.seed}, relations={relations}, cache_dir={args.cache_dir}, num_mean_examples={args.num_mean_examples}")

    clean, ablated = evaluate_with_ablation(args.split, max_samples, args.cache_dir, layer_head_pairs, args.scale, args.seed, relations, args.num_mean_examples, mean_split)
    (acc_c, correct_c, total_c, meanp_c, countp_c) = clean
    (acc_a, correct_a, total_a, meanp_a, countp_a) = ablated

    print("\n=== Paired Results ===")
    print(f"Baseline accuracy: {acc_c*100:.2f}%  ({correct_c}/{total_c})")
    print(f"Ablated  accuracy: {acc_a*100:.2f}%  ({correct_a}/{total_a})")
    print(f"ΔAcc: {(acc_a-acc_c)*100:.2f} pp")
    if countp_c > 0 and countp_a > 0:
        print(f"Baseline mean P(correct): {meanp_c:.4f}  (N={countp_c})")
        print(f"Ablated  mean P(correct): {meanp_a:.4f}  (N={countp_a})")
        print(f"ΔP(correct): {(meanp_a-meanp_c):.4f}")
    else:
        print("P(correct) not available for this model/template.")


if __name__ == "__main__":
    main()


