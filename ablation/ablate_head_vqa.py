#!/usr/bin/env python3
"""
Ablate one or multiple attention heads in the VLM and evaluate accuracy on VQA yes/no.

- Hooks into layer[L].self_attn.{o,q,k,v}_proj to scale selected head subspaces
- Uses same evaluation protocol as VSR head ablation but on VQAv2 yes/no

Example:

CUDA_VISIBLE_DEVICES=3 python ablation/ablate_head_vqa.py \
  --pairs L13H0,L14H31 --scale 0.0 --target o_proj \
  --split validation --max-samples 1000 --seed 0
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

# Import VLM utilities directly
import sys
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    generate_vlm_response,
)

# Reuse VQA yes/no loader from the VQA SAE script
from ablation.ablate_sae_feature_vqa import load_vqa_yesno  # type: ignore


def normalize_answer(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("yes") or "\nyes" in t or " yes" in t:
        return "yes"
    if t.startswith("no") or "\nno" in t or " no" in t:
        return "no"
    if '"answer": "yes"' in t or "answer: yes" in t:
        return "yes"
    if '"answer": "no"' in t or "answer: no" in t:
        return "no"
    return "unknown"


def build_prompt(question: str) -> str:
    q = question.strip()
    return (
        "Answer the following question with only 'Yes' or 'No':\n"
        f"Question: {q}\n"
        "Answer:"
    )


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
    logits = out.logits[:, -1, :]
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


def register_head_ablation_hook(model, layer_idx: int, head_idx, scale: float = 0.0, target: str = "o_proj"):
    hidden_size = int(model.model.config.hidden_size)
    num_heads = int(model.model.config.num_attention_heads)
    head_dim = int(model.model.config.head_dim)
    if head_dim * num_heads != hidden_size:
        head_dim = hidden_size // num_heads
    if not (0 <= layer_idx < len(model.model.layers)):
        raise ValueError(f"layer_idx out of range: {layer_idx}")
    if isinstance(head_idx, int):
        head_indices = [head_idx]
    else:
        head_indices = list(head_idx)
    if len(head_indices) == 0:
        raise ValueError("No head indices provided")
    bad = [h for h in head_indices if not (0 <= h < num_heads)]
    if bad:
        raise ValueError(f"head_idx out of range: {bad} (num_heads={num_heads})")

    attn = model.model.layers[layer_idx].self_attn

    if target == "o_proj":
        tgt_linear = attn.o_proj
        def pre_hook(_module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            try:
                x = x.clone()
                for h in head_indices:
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    x[..., start:end] = x[..., start:end] * scale
                return (x,)
            except Exception:
                return inputs
        handle = tgt_linear.register_forward_pre_hook(pre_hook, with_kwargs=False)

    elif target in ("q_proj", "k_proj", "v_proj"):
        proj = getattr(attn, target)
        def post_hook(_module, inputs, output):
            try:
                y = output
                for h in head_indices:
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    y[..., start:end] = y[..., start:end] * scale
                return y
            except Exception:
                return output
        handle = proj.register_forward_hook(post_hook, with_kwargs=False)
    else:
        raise ValueError(f"Invalid target: {target}")

    return handle


def _parse_lh_pairs_arg(pairs_arg: str) -> List[Tuple[int, int]]:
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


def _evaluate_pass(
    ds,
    indices: List[int],
    tokenizer,
    model,
    image_processor,
) -> Tuple[float, int, int, float, int]:
    total = 0
    correct = 0
    sum_p_correct = 0.0
    count_p = 0

    yes_ids, no_ids = _build_yes_no_token_sets(tokenizer)

    for i, idx in enumerate(indices):
        if i % 100 == 0:
            print(f"[INFO] Processing sample {i}/{len(indices)}...")

        ex = ds[idx]
        image = ex["image"].convert("RGB")
        question = str(ex.get("question", "").strip())
        label_str = str(ex.get("multiple_choice_answer", "").strip().lower())
        if label_str not in {"yes", "no"}:
            continue

        prompt = build_prompt(question)

        try:
            resp = generate_vlm_response(
                image=image,
                prompt=prompt,
                image_processor=image_processor,
                vlm_model=model,
                vlm_tokenizer=tokenizer,
                max_new_tokens=4,
                json_mode=False,
            )
        except Exception as e:
            print(f"[ERROR] Sample {i}: Generation failed: {e}")
            resp = ""

        pred_norm = normalize_answer(resp)
        if pred_norm != "unknown":
            pred = "yes" if pred_norm == "yes" else "no"
            total += 1
            if pred == label_str:
                correct += 1

        try:
            p_yes, p_no = _next_token_yes_no_probs(image, prompt, image_processor, model, tokenizer, yes_ids, no_ids)
            p_correct = p_yes if label_str == "yes" else p_no
            sum_p_correct += float(p_correct)
            count_p += 1
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    acc = (correct / total) if total > 0 else 0.0
    mean_p = (sum_p_correct / count_p) if count_p > 0 else 0.0
    return acc, correct, total, mean_p, count_p


def evaluate_with_ablation(
    split: str,
    max_samples: int | None,
    layer_head_pairs: List[Tuple[int, int]],
    scale: float,
    target: str,
    seed: int | None,
    index_cache_dir: str | None,
):
    disp_tokens = [f"L{l}H{h}" for (l, h) in layer_head_pairs]
    print(f"[INFO] Starting VQA yes/no head ablation ({','.join(disp_tokens)} scale={scale}, target={target})")

    print("[INFO] Initializing VLM model...")
    tokenizer, model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    model.eval()
    print("[INFO] VLM model loaded and ready!")

    print("[INFO] Loading VQAv2 dataset (yes/no only)...")
    ds_filtered, _indices_full = load_vqa_yesno(split=split, max_samples=max_samples, seed=seed, index_cache_dir=index_cache_dir)
    indices = list(range(len(ds_filtered)))
    n = len(indices)
    print(f"[INFO] Evaluating {n} yes/no samples...")

    # Baseline cache key (model, split, n, seed)
    model_id = str(getattr(model.config, "_name_or_path", "unknown")).lower()
    key_obj = {
        "model": model_id,
        "split": split,
        "n": n,
        "seed": seed,
        "subset": "vqa_yesno",
        "target": target,
        "pairs": disp_tokens,
    }
    key_hash = hashlib.md5(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()
    baseline_cache_dir = Path("results/stage6/vqa_yesno_head_baseline_cache")
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
            clean_metrics = _evaluate_pass(ds_filtered, indices, tokenizer, model, image_processor)
            try:
                with open(baseline_cache_path, "w") as f:
                    json.dump({"key": key_obj, "indices": indices, "metrics": list(clean_metrics)}, f)
                print(f"[INFO] Saved baseline cache → {baseline_cache_path}")
            except Exception as e2:
                print(f"[WARN] Failed to write baseline cache: {e2}")
    else:
        print("[INFO] Running clean (no hook) pass...")
        clean_metrics = _evaluate_pass(ds_filtered, indices, tokenizer, model, image_processor)
        try:
            with open(baseline_cache_path, "w") as f:
                json.dump({"key": key_obj, "indices": indices, "metrics": list(clean_metrics)}, f)
            print(f"[INFO] Saved baseline cache → {baseline_cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to write baseline cache: {e}")

    # Ablated pass
    print("[INFO] Registering head ablation hooks and running ablated pass...")
    from collections import defaultdict
    layer_to_heads: dict[int, List[int]] = defaultdict(list)
    for l, h in layer_head_pairs:
        layer_to_heads[l].append(h)
    handles = []
    try:
        for l, hs in layer_to_heads.items():
            hs_sorted = sorted(set(hs))
            handles.append(register_head_ablation_hook(model, layer_idx=l, head_idx=hs_sorted, scale=scale, target=target))
        ablated_metrics = _evaluate_pass(ds_filtered, indices, tokenizer, model, image_processor)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    return clean_metrics, ablated_metrics


def main():
    print("[INFO] Starting VQA yes/no head-ablation evaluation script...")

    ap = argparse.ArgumentParser(description="Ablate a specific attention head and evaluate on VQA yes/no")
    ap.add_argument("--pairs", type=str, required=True, help="Comma-separated layer-head tokens, e.g., L13H1,L7H2")
    ap.add_argument("--scale", type=float, default=0.0, help="Scale factor for the head subspace (0.0=ablate)")
    ap.add_argument("--target", type=str, default="o_proj", choices=["o_proj", "q_proj", "k_proj", "v_proj"], help="Where to apply the head scaling")

    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation"], help="VQAv2 split")
    ap.add_argument("--max-samples", type=int, default=1000, help="Max samples to evaluate (0 for all)")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed to shuffle indices deterministically")
    args = ap.parse_args()

    max_samples = None if args.max_samples == 0 else args.max_samples

    layer_head_pairs: List[Tuple[int, int]] = _parse_lh_pairs_arg(args.pairs)
    layer_head_pairs = sorted(set(layer_head_pairs))
    if len(layer_head_pairs) == 0:
        raise ValueError("Must specify --pairs with tokens like L13H1")

    print(f"[INFO] Configuration: pairs={[f'L{l}H{h}' for (l,h) in layer_head_pairs]}, scale={args.scale}, target={args.target}, split={args.split}, max_samples={max_samples}, seed={args.seed}")

    clean, ablated = evaluate_with_ablation(args.split, max_samples, layer_head_pairs, args.scale, args.target, args.seed, None)
    (acc_c, correct_c, total_c, meanp_c, countp_c) = clean
    (acc_a, correct_a, total_a, meanp_a, countp_a) = ablated

    print("\n=== Paired Results (VQA Yes/No) ===")
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

















