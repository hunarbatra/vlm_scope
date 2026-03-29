#!/usr/bin/env python3
"""
Ablate one SAE feature (by layer and feature idx) during VQA yes/no evaluation.

- Uses the same VLM model and NNsight pipeline as ablate_sae_feature_vsr.py (LLaVA-MORE)
- Loads SAEs via initialize_sae
- Applies feature ablation by projecting out selected SAE feature subspaces
  at the specified transformer layer(s) using NNsight traces.

Example:


CUDA_VISIBLE_DEVICES=3 python ablation/ablate_sae_feature_vqa.py \
  --pairs L4F246 \
  --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
  --apply-to-all-layers \
  --split validation --max-samples 1000
"""

from __future__ import annotations

import argparse
import os
import re
import json
import hashlib
from typing import Tuple, List, Set
from pathlib import Path

import dotenv
dotenv.load_dotenv(".env")

import torch

import sys
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    generate_vlm_response,
    initialize_sae,
    get_image_token_positions,
)
sys.path.pop()

from datasets import load_dataset
from nnsight import NNsight


def load_vqa_yesno(
    split: str = "validation",
    max_samples: int | None = None,
    seed: int | None = None,
    index_cache_dir: str | None = None,
):
    split_norm = split
    if split.lower() in {"dev", "val"}:
        split_norm = "validation"
    if split_norm not in {"train", "validation"}:
        raise ValueError("VQAv2 supported splits are 'train' or 'validation'")

    print(f"[INFO] Loading VQAv2 {split_norm} split...")
    dataset = load_dataset("lmms-lab/VQAv2", split=split_norm)

    cached_indices: List[int] | None = None
    cache_path: Path | None = None
    if index_cache_dir:
        cache_dir = Path(index_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"indices_{split_norm}_yesno.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                if payload.get("split") == split_norm and isinstance(payload.get("indices"), list):
                    cached_indices = list(map(int, payload["indices"]))
                    print(f"[INFO] Loaded yes/no indices from cache → {cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to read yes/no index cache: {e}")

    if cached_indices is None:
        print("[INFO] Building yes/no index list (metadata-only scan)...")
        try:
            meta = dataset.remove_columns(["image"])
        except Exception:
            meta = dataset

        yesno_indices: List[int] = []
        early_stop_target = max_samples if (max_samples is not None and seed is None) else None
        for i in range(len(meta)):
            ex = meta[i]
            at = str(ex.get("answer_type", "")).lower()
            mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
            if at == "yes/no" and mc in {"yes", "no"}:
                yesno_indices.append(i)
                if early_stop_target is not None and len(yesno_indices) >= early_stop_target:
                    break

        if cache_path is not None and early_stop_target is None:
            try:
                cache_path.write_text(json.dumps({"split": split_norm, "indices": yesno_indices}))
                print(f"[INFO] Saved yes/no indices cache → {cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to write yes/no index cache: {e}")
        indices = yesno_indices
    else:
        indices = cached_indices

    if seed is not None:
        import random
        rnd = random.Random(seed)
        rnd.shuffle(indices)

    if max_samples is not None:
        indices = indices[:max_samples]

    filtered = dataset.select(indices)
    print(f"[INFO] Filtered to {len(filtered)} yes/no questions (from {len(dataset)} total)")
    return filtered, indices




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
            ids.add(toks[0])
    return ids

def _build_yes_no_token_sets(tokenizer) -> Tuple[Set[int], Set[int]]:
    """Build disjoint sets of first-token IDs for yes/no responses."""
    yes_first = _first_token_ids(tokenizer, [" Yes", "Yes", " yes", "YES"])
    no_first = _first_token_ids(tokenizer, [" No", "No", " no", "NO"])
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
        p_yes = float(yes_mass)
        p_no = float(no_mass)
    else:
        p_yes = float(yes_mass / denom)
        p_no = float(no_mass / denom)
    return p_yes, p_no


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
            p_yes, p_no = _next_token_yes_no_probs(image, prompt, image_processor, model, tokenizer, yes_ids, no_ids)
        except Exception:
            p_yes, p_no = 0.0, 0.0

        pred = 1 if float(p_yes) > float(p_no) else 0
        total += 1
        if pred == (1 if label_str == "yes" else 0):
            correct += 1

        try:
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


def evaluate_with_feature_ablation(
    split: str,
    max_samples: int | None,
    layer_feature_pair: Tuple[int, int],
    sae_checkpoint_dir: str | None,
    index_cache_dir: str | None,
    apply_to_all_layers: bool,
):
    layer_idx, feature_idx = layer_feature_pair
    print(f"[INFO] Starting VQA yes/no evaluation with SAE feature ablation (L{layer_idx}F{feature_idx} apply_to_all_layers={apply_to_all_layers}, decision=prob)")

    print("[INFO] Initializing VLM model...")
    tokenizer, model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    model = NNsight(model)
    model.eval()
    print("[INFO] VLM model loaded and ready!")

    print("[INFO] Loading VQAv2 dataset (yes/no only)...")
    ds_filtered, yesno_indices = load_vqa_yesno(split=split, max_samples=max_samples, seed=None, index_cache_dir=index_cache_dir)
    indices = list(range(len(ds_filtered)))
    n = len(indices)
    print(f"[INFO] Evaluating {n} yes/no samples...")

    model_id = str(getattr(model._module.config, "_name_or_path", "unknown")).lower()
    key_obj = {
        "model": model_id,
        "split": split,
        "n": n,
        "subset": "vqa_yesno",
        "decision": "prob",
    }
    key_hash = hashlib.md5(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()
    baseline_cache_dir = Path("results/stage6/vqa_yesno_baseline_cache")
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache_path = baseline_cache_dir / f"{key_hash}.json"

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
            clean_metrics = _evaluate_pass(ds_filtered, indices, tokenizer, model._module, image_processor)
            try:
                with open(baseline_cache_path, "w") as f:
                    json.dump({"key": key_obj, "indices": indices, "metrics": list(clean_metrics)}, f)
                print(f"[INFO] Saved baseline cache → {baseline_cache_path}")
            except Exception as e2:
                print(f"[WARN] Failed to write baseline cache: {e2}")
    else:
        print("[INFO] Running clean (no hook) pass...")
        clean_metrics = _evaluate_pass(ds_filtered, indices, tokenizer, model._module, image_processor)
        try:
            with open(baseline_cache_path, "w") as f:
                json.dump({"key": key_obj, "indices": indices, "metrics": list(clean_metrics)}, f)
            print(f"[INFO] Saved baseline cache → {baseline_cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to write baseline cache: {e}")

    print("[INFO] Running ablated pass with NNsight QR projection (no hooks)...")
    
    checkpoint_path = None
    if sae_checkpoint_dir:
        candidates = list(Path(sae_checkpoint_dir).glob(f"*layer_{layer_idx}*.pt"))
        if candidates:
            checkpoint_path = str(candidates[0])
    
    sae = initialize_sae(layer_idx=layer_idx, checkpoint_path=checkpoint_path, initialize_random=(checkpoint_path is None), device="cuda")
    
    try:
        sae_num_features = int(getattr(sae, "num_features", sae.W_dec.shape[0]))
        if 0 <= feature_idx < sae_num_features:
            feature_vec = sae.W_dec[feature_idx].detach().to(torch.float16).to(model._module.device)
            feature_vec = feature_vec / feature_vec.norm()
        else:
            print(f"[WARN] Feature index {feature_idx} out of range (num_features={sae_num_features})")
            return clean_metrics, clean_metrics
    except Exception as e:
        print(f"[WARN] Failed to load feature {feature_idx} from layer {layer_idx}: {e}")
        return clean_metrics, clean_metrics

    total = 0
    correct = 0
    sum_p_correct = 0.0
    count_p = 0
    yes_ids, no_ids = _build_yes_no_token_sets(tokenizer)
    
    for i, idx in enumerate(indices):
        if i % 100 == 0:
            print(f"[INFO] (ablated) Processing sample {i}/{len(indices)}...")
        
        ex = ds_filtered[idx]
        image = ex["image"].convert("RGB")
        question = str(ex.get("question", "").strip())
        label_str = str(ex.get("multiple_choice_answer", "").strip().lower())
        if label_str not in {"yes", "no"}:
            continue
        
        prompt = build_prompt(question)
        
        input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
            image, prompt, image_processor, model._module, tokenizer
        )
        
        image_start, img_end = get_image_token_positions(input_ids)
        img_end = int(img_end)
        
        logits_saved = None
        try:
            with model.trace(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=image_sizes,
                use_cache=False,
            ) as tr:
                if apply_to_all_layers:
                    for l in range(model._module.config.num_hidden_layers):
                        attn_output = model.model.layers[l].self_attn.output[0][0, img_end:]
                        attn_feature_proj = (attn_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                        model.model.layers[l].self_attn.output[0][0, img_end:] -= attn_feature_proj
                        
                        mlp_output = model.model.layers[l].mlp.output[0, img_end:]
                        mlp_feature_proj = (mlp_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                        model.model.layers[l].mlp.output[0, img_end:] -= mlp_feature_proj
                        
                        layer_output = model.model.layers[l].output[0][0, img_end:]
                        layer_feature_proj = (layer_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                        model.model.layers[l].output[0][0, img_end:] -= layer_feature_proj
                else:
                    attn_output = model.model.layers[layer_idx].self_attn.output[0][0, img_end:]
                    attn_feature_proj = (attn_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                    model.model.layers[layer_idx].self_attn.output[0][0, img_end:] -= attn_feature_proj
                    
                    mlp_output = model.model.layers[layer_idx].mlp.output[0, img_end:]
                    mlp_feature_proj = (mlp_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                    model.model.layers[layer_idx].mlp.output[0, img_end:] -= mlp_feature_proj
                    
                    layer_output = model.model.layers[layer_idx].output[0][0, img_end:]
                    layer_feature_proj = (layer_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                    model.model.layers[layer_idx].output[0][0, img_end:] -= layer_feature_proj

                logits_saved = model.output.logits.save()

            output_logits = logits_saved
            logits_last = output_logits[:, -1, :]
            probs_last = torch.softmax(logits_last, dim=-1)

            yes_mass = probs_last[0, list(yes_ids)].sum() if yes_ids else torch.tensor(0.0, device=probs_last.device)
            no_mass = probs_last[0, list(no_ids)].sum() if no_ids else torch.tensor(0.0, device=probs_last.device)
            denom = yes_mass + no_mass
            if float(denom) == 0.0:
                p_yes = float(yes_mass)
                p_no = float(no_mass)
            else:
                p_yes = float(yes_mass / denom)
                p_no = float(no_mass / denom)

        except Exception as e:
            print(f"[ERROR] (ablated) Sample {i}: Trace/forward failed: {e}")
            p_yes, p_no = 0.0, 0.0

        pred = 1 if float(p_yes) > float(p_no) else 0
        total += 1
        if pred == (1 if label_str == "yes" else 0):
            correct += 1

        try:
            p_correct = p_yes if label_str == "yes" else p_no
            sum_p_correct += float(p_correct)
            count_p += 1
        except Exception:
            pass
        
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    ablated_metrics = (
        (correct / total) if total > 0 else 0.0,
        correct,
        total,
        (sum_p_correct / count_p) if count_p > 0 else 0.0,
        count_p,
    )

    return clean_metrics, ablated_metrics


def _parse_lf_pairs_arg(pairs_arg: str) -> Tuple[int, int]:
    """Parse single layer-feature token like 'L13F16873' -> (13,16873)."""
    if not pairs_arg:
        raise ValueError("Must specify --pairs with a single layer-feature token")
    
    m = re.fullmatch(r"[Ll](\d+)[Ff](\d+)", pairs_arg.strip())
    if not m:
        raise ValueError(f"Invalid layer-feature token: {pairs_arg}. Expected form like L13F16873")
    
    layer_val = int(m.group(1))
    feat_val = int(m.group(2))
    return (layer_val, feat_val)


def test_yes_no_tokenization():
    """Test function to verify yes/no tokenization is working correctly."""
    print("[INFO] Testing yes/no tokenization...")
    tokenizer, _, _ = initialize_vlm_model("llava-more", device="cpu")  # Use CPU for testing
    yes_ids, no_ids = _build_yes_no_token_sets(tokenizer)
    
    test_responses = ["Yes", "No", " Yes", " No", "yes", "no", "YES", "NO"]
    print("\nTesting sample responses:")
    for resp in test_responses:
        ids = tokenizer.encode(resp, add_special_tokens=False)
        decoded = [tokenizer.decode([id]) for id in ids]
        yes_match = any(id in yes_ids for id in ids)
        no_match = any(id in no_ids for id in ids)
        print(f"  '{resp}' -> {ids} -> {decoded} -> Yes: {yes_match}, No: {no_match}")


def main():
    print("[INFO] Starting VQA yes/no SAE feature-ablation evaluation script...")

    ap = argparse.ArgumentParser(description="Ablate a specific SAE feature and evaluate on VQAv2 yes/no")
    ap.add_argument("--pairs", type=str, required=True, help="Single layer-feature token, e.g., L13F16873")
    ap.add_argument("--apply-to-all-layers", action="store_true", help="If set, remove projection across ALL transformer layers using a shared subspace built from specified features")

    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation"], help="VQAv2 split")
    ap.add_argument("--max-samples", type=int, default=1000, help="Max samples to evaluate (0 for all)")
    ap.add_argument("--sae-checkpoint-dir", type=str, required=True, help="Directory containing per-layer SAE checkpoints (text-only)")
    ap.add_argument("--index-cache-dir", type=str, default=None, help="Directory to cache yes/no indices (optional)")
    ap.add_argument("--test-tokenization", action="store_true", help="Test yes/no tokenization and exit")
    args = ap.parse_args()
    
    if args.test_tokenization:
        test_yes_no_tokenization()
        return

    max_samples = None if args.max_samples == 0 else args.max_samples
    layer_feature_pair = _parse_lf_pairs_arg(args.pairs)

    print(f"[INFO] Configuration: feature=L{layer_feature_pair[0]}F{layer_feature_pair[1]}, apply_to_all_layers={args.apply_to_all_layers}, split={args.split}, max_samples={max_samples}, pairs_csv={'(none)' if not args.index_cache_dir else args.index_cache_dir}")

    clean, ablated = evaluate_with_feature_ablation(
        args.split, max_samples, layer_feature_pair, args.sae_checkpoint_dir, args.index_cache_dir, args.apply_to_all_layers
    )
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
