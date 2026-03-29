"""
Ablate one SAE feature (by layer and feature idx) during VSR evaluation.

- Uses the same VLM model and VSR pipeline as ablate_head_vsr.py (LLaVA-MORE)
- Loads SAEs similarly to experiments/attribution_patching.py via initialize_sae
- Applies feature ablation by projecting out selected SAE feature subspaces
  at the specified transformer layer(s) using NNsight traces
- Uses probability-based decisions (p_yes vs p_no) for robust evaluation

Example:

CUDA_VISIBLE_DEVICES=4 python ablation/generate_vsr_ablation_cmds.py   --csv-file results/experiments/filtered_top.csv   --vsr-samples-dir results/stage_4/feature_samples/vsr_all_spatial_fixed   --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/text-only   --split train+dev+test   --top-k-samples 10   --min-relation-count 1   --execute   --save-results   --results-json results/stage6/generated_ablation_results_filtered_top.json   --apply-to-all-layers   --cache-dir /scratch/local/ssd/lachin/vsr_image_cache_double_check   --save-samples /scratch/local/ssd/lachin/vsr_runs/per_sample_results.jsonl 



CUDA_VISIBLE_DEVICES=4 \
python ablation/ablate_sae_feature_vsr.py \
  --pairs L14F17873 \
  --split train+dev+test \
  --max-samples 0 \
  --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
  --cache-dir /scratch/local/ssd/lachin/vsr_image_cache \
  --baseline-cache-dir results/stage6/vsr_baseline_cache \
  --steering-coef -1.0 \
  --apply-to-all-layers \
    --relations 'at the right side of'

CUDA_VISIBLE_DEVICES=0 \
python ablation/ablate_sae_feature_vsr.py \
  --pairs L14F17873 \
  --split train+dev+test \
  --max-samples 0 \
  --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
  --cache-dir /scratch/local/ssd/lachin/vsr_image_cache \
  --baseline-cache-dir results/stage6/vsr_baseline_cache \
  --coef-steering 10 \
  --proj-eps 1e-6 \
  --delta-clip 1.0 \
    --relations 'at the right side of'


"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Set, Dict, Optional

import dotenv
import requests
import torch
import numpy as np
from datasets import load_dataset
from nnsight import NNsight
from PIL import Image

import sys
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    initialize_sae,
    get_image_token_positions,
)
sys.path.pop()

dotenv.load_dotenv(".env")

PROGRESS_INTERVAL = 100
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CACHE_DIR = "/scratch/local/ssd/lachin/vsr_image_cache"
DEFAULT_TIMEOUT = 10


def load_vsr(split: str = "train", only_true: bool = False):
    """Load VSR dataset with optional filtering."""
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    if split == "train+dev+test":
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


def load_image_with_cache(url: str, cache_dir: str = DEFAULT_CACHE_DIR, timeout: int = DEFAULT_TIMEOUT) -> Image.Image:
    """Load image from URL with caching support."""
    os.makedirs(cache_dir, exist_ok=True)
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
        return Image.new("RGB", DEFAULT_IMAGE_SIZE, (128, 128, 128))


def build_prompt(statement: str) -> str:
    """Build a VSR evaluation prompt from a statement."""
    s = statement.strip()
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {s}\n"
        "Answer:"
    )


def _first_token_ids(tokenizer, texts: List[str]) -> Set[int]:
    """Get first token IDs for a list of texts."""
    ids = set()
    for t in texts:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            ids.add(toks[0])  # only the first token
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




def _parse_lf_pairs_arg(pairs_arg: str) -> List[Tuple[int, int]]:
    """Parse comma-separated layer-feature tokens like 'L13F16873,L7F42' -> [(13,16873),(7,42)]."""
    if not pairs_arg:
        raise ValueError("Must specify --pairs with layer-feature tokens")
    
    tokens = [t.strip() for t in pairs_arg.split(',') if t.strip()]
    pairs: List[Tuple[int, int]] = []
    for tok in tokens:
        m = re.fullmatch(r"[Ll](\d+)[Ff](\d+)", tok)
        if not m:
            raise ValueError(f"Invalid layer-feature token: {tok}. Expected form like L13F16873")
        layer_val = int(m.group(1))
        feat_val = int(m.group(2))
        pairs.append((layer_val, feat_val))
    return pairs


def _evaluate_pass(
    ds,
    indices: List[int],
    cache_dir: str,
    tokenizer,
    model,
    image_processor,
    save_samples_path: Optional[str] = None,
    pass_name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Tuple[float, int, int, float, int]:
    total = 0
    correct = 0
    sum_p_correct = 0.0
    count_p = 0

    yes_ids, no_ids = _build_yes_no_token_sets(tokenizer)

    if save_samples_path:
        try:
            Path(save_samples_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[WARN] Could not create directory for save path {save_samples_path}: {e}")

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

        try:
            p_yes, p_no = _next_token_yes_no_probs(image, prompt, image_processor, model, tokenizer, yes_ids, no_ids)
        except Exception:
            p_yes, p_no = 0.0, 0.0

        pred = 1 if float(p_yes) > float(p_no) else 0
        total += 1
        if pred == label:
            correct += 1

        try:
            p_correct = p_yes if label == 1 else p_no
            sum_p_correct += float(p_correct)
            count_p += 1
        except Exception:
            pass

        if save_samples_path:
            try:
                record = {
                    "pass": pass_name or "baseline",
                    "index_in_eval": int(i),
                    "dataset_index": int(idx),
                    "label": int(label),
                    "pred": int(pred),
                    "correct": int(pred == label),
                    "p_yes": float(p_yes),
                    "p_no": float(p_no),
                }
                try:
                    record["statement"] = statement
                    record["image_url"] = image_url
                except Exception:
                    pass
                with open(save_samples_path, "a") as f:
                    json.dump(record, f)
                    f.write("\n")
            except Exception as e:
                print(f"[WARN] Failed to save sample record: {e}")

        if (i + 1) % PROGRESS_INTERVAL == 0:
            torch.cuda.empty_cache()

    acc = (correct / total) if total > 0 else 0.0
    mean_p = (sum_p_correct / count_p) if count_p > 0 else 0.0
    return acc, correct, total, mean_p, count_p


def evaluate_with_feature_ablation(
    split: str,
    max_samples: Optional[int],
    cache_dir: str,
    layer_feature_pairs: List[Tuple[int, int]],
    sae_checkpoint_dir: Optional[str],
    relations: Optional[List[str]],
    apply_to_all_layers: bool,
    steering_coeff: float,
    negative_steering: bool = False,
    baseline_cache_dir: Optional[str] = None,
    save_samples_path: Optional[str] = None,
    coef_steering: float = 0.0,
) -> Tuple[Tuple[float, int, int, float, int], Tuple[float, int, int, float, int]]:
    """Evaluate VSR with SAE feature ablation."""
    pairs_str = ",".join([f"L{l}F{f}" for l, f in layer_feature_pairs])
    print(f"[INFO] Starting VSR evaluation with SAE feature ablation ({pairs_str} apply_to_all_layers={apply_to_all_layers}, decision=prob)")

    print("[INFO] Initializing VLM model...")
    tokenizer, hf_model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    nns_model = NNsight(hf_model)
    nns_model.eval()
    print("[INFO] VLM model loaded and ready!")

    print("[INFO] Loading VSR dataset...")
    ds = load_vsr(split=split, only_true=False)
    print(f"[INFO] Loaded {len(ds)} VSR samples")
    if relations:
        allowed = set([r.strip() for r in relations if r.strip()])
        if len(allowed) > 0:
            print(f"[INFO] Filtering to relations (without mutating dataset): {sorted(list(allowed))}")
            indices = [i for i in range(len(ds)) if (ds[i].get("relation") in allowed)]
            print(f"[INFO] After filtering: {len(indices)} samples")
        else:
            indices = list(range(len(ds)))
    else:
        indices = list(range(len(ds)))

    n = len(indices) if max_samples is None else min(len(indices), max_samples)
    indices = indices[:n]
    print(f"[INFO] Evaluating {n} paired samples...")

    model_id = str(getattr(hf_model.config, "_name_or_path", "unknown")).lower()
    key_obj = {
        "model": model_id,
        "split": split,
        "n": n,
        "relations": sorted(relations) if relations else [],
        "decision": "prob",
    }
    key_hash = hashlib.md5(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()
    baseline_cache_dir_path = Path(baseline_cache_dir or "results/stage6/vsr_baseline_cache")
    baseline_cache_dir_path.mkdir(parents=True, exist_ok=True)
    baseline_cache_path = baseline_cache_dir_path / f"{key_hash}.json"

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
                if save_samples_path:
                    try:
                        common_meta = {
                            "pairs": pairs_str,
                            "apply_to_all_layers": bool(apply_to_all_layers),
                            "steering_coeff": float(steering_coeff),
                            "negative_steering": bool(negative_steering),
                            "split": split,
                        }
                        _ = _evaluate_pass(
                            ds,
                            indices,
                            cache_dir,
                            tokenizer,
                            hf_model,
                            image_processor,
                            save_samples_path=save_samples_path,
                            pass_name="baseline",
                            extra_metadata=common_meta,
                        )
                    except Exception as e:
                        print(f"[WARN] Failed to produce per-sample baseline logs from cache: {e}")
            else:
                raise ValueError("Malformed cache; will recompute baseline")
        except Exception as e:
            print(f"[WARN] Failed to load baseline cache: {e}; recomputing baseline")
            print("[INFO] Running clean (no hook) pass...")
            common_meta = {
                "pairs": pairs_str,
                "apply_to_all_layers": bool(apply_to_all_layers),
                "steering_coeff": float(steering_coeff),
                "negative_steering": bool(negative_steering),
                "split": split,
            }
            clean_metrics = _evaluate_pass(
                ds,
                indices,
                cache_dir,
                tokenizer,
                hf_model,
                image_processor,
                save_samples_path=save_samples_path,
                pass_name="baseline",
                extra_metadata=common_meta,
            )
            try:
                with open(baseline_cache_path, "w") as f:
                    json.dump({"key": key_obj, "indices": indices, "metrics": list(clean_metrics)}, f)
                print(f"[INFO] Saved baseline cache → {baseline_cache_path}")
            except Exception as e2:
                print(f"[WARN] Failed to write baseline cache: {e2}")
    else:
        print("[INFO] Running clean (no hook) pass...")
        common_meta = {
            "pairs": pairs_str,
            "apply_to_all_layers": bool(apply_to_all_layers),
            "steering_coeff": float(steering_coeff),
            "negative_steering": bool(negative_steering),
            "split": split,
        }
        clean_metrics = _evaluate_pass(
            ds,
            indices,
            cache_dir,
            tokenizer,
            hf_model,
            image_processor,
            save_samples_path=save_samples_path,
            pass_name="baseline",
            extra_metadata=common_meta,
        )
        try:
            with open(baseline_cache_path, "w") as f:
                json.dump({"key": key_obj, "indices": indices, "metrics": list(clean_metrics)}, f)
            print(f"[INFO] Saved baseline cache → {baseline_cache_path}")
        except Exception as e:
            print(f"[WARN] Failed to write baseline cache: {e}")

    print("[INFO] Running ablated pass with NNsight QR projection (no hooks)...")
    
    feature_vectors = {}
    layer_saes = {}
    for layer_idx, feature_idx in layer_feature_pairs:
        checkpoint_path = None
        if sae_checkpoint_dir:
            candidates = list(Path(sae_checkpoint_dir).glob(f"*layer_{layer_idx}.pt"))
            if candidates:
                checkpoint_path = str(candidates[0])
        
        sae = initialize_sae(layer_idx=layer_idx, checkpoint_path=checkpoint_path, initialize_random=(checkpoint_path is None), device="cuda")
        layer_saes[layer_idx] = sae
        
        try:
            sae_num_features = int(getattr(sae, "num_features", sae.W_dec.shape[0]))
            if 0 <= feature_idx < sae_num_features:
                model_dtype = next(nns_model._module.parameters()).dtype
                feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to(nns_model._module.device)
                feature_vec = feature_vec / feature_vec.norm()
                feature_vectors[(layer_idx, feature_idx)] = feature_vec
            else:
                print(f"[WARN] Feature index {feature_idx} out of range for layer {layer_idx} (num_features={sae_num_features})")
        except Exception as e:
            print(f"[WARN] Failed to load feature {feature_idx} from layer {layer_idx}: {e}")
    
    if not feature_vectors and not layer_saes:
        print("[ERROR] No valid features loaded, returning clean metrics")
        return clean_metrics, clean_metrics


    total = 0
    correct = 0
    sum_p_correct = 0.0
    count_p = 0
    yes_ids, no_ids = _build_yes_no_token_sets(tokenizer)
    
    if save_samples_path:
        try:
            Path(save_samples_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[WARN] Could not create directory for save path {save_samples_path}: {e}")

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
            image, prompt, image_processor, nns_model._module, tokenizer
        )
        
        _, img_end = get_image_token_positions(input_ids)
        img_end = int(img_end)
        
        logits_saved = None
        try:
            with nns_model.trace(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=image_sizes,
            ) as tr:
                for layer_idx, feature_idx in layer_feature_pairs:
                    if negative_steering:
                        if layer_idx in layer_saes:
                            sae = layer_saes[layer_idx]
                            
                            layer_output = nns_model.model.layers[layer_idx].output[0][0, img_end:]
                            codes = sae.encode(layer_output)
                            coefficients = codes[:, feature_idx]

                            feature_vec = sae.W_dec[feature_idx]
                            model_dtype = next(nns_model._module.parameters()).dtype
                            feature_vec = feature_vec.detach().to(model_dtype).to(nns_model._module.device)

                            coefficients = coefficients.to(feature_vec.dtype)
                            delta = -coefficients.unsqueeze(1) * feature_vec.unsqueeze(0)
                            nns_model.model.layers[layer_idx].output[0][0, img_end:] += delta
                    elif float(coef_steering) != 0.0:
                        if layer_idx in layer_saes:
                            sae = layer_saes[layer_idx]
                            target_tokens = nns_model.model.layers[layer_idx].output[0][0, img_end:]
                            codes = sae.encode(target_tokens)
                            coefficients = codes[:, feature_idx]
                            feature_vec = sae.W_dec[feature_idx].detach().to(nns_model._module.device)
                            delta = float(coef_steering) * (coefficients.unsqueeze(1) * feature_vec.unsqueeze(0))
                            nns_model.model.layers[layer_idx].output[0][0, img_end:] += delta
                    else:
                        combined_coef = steering_coeff if float(steering_coeff) != 0.0 else -1.0
                        if (layer_idx, feature_idx) in feature_vectors:
                            feature_vec = feature_vectors[(layer_idx, feature_idx)]

                            if apply_to_all_layers:
                                for l in range(nns_model._module.config.num_hidden_layers):
                                    attn_output = nns_model.model.layers[l].self_attn.output[0][0, img_end:]
                                    attn_feature_proj = (attn_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                                    nns_model.model.layers[l].self_attn.output[0][0, img_end:] += combined_coef * attn_feature_proj

                                    mlp_output = nns_model.model.layers[l].mlp.output[0, img_end:]
                                    mlp_feature_proj = (mlp_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                                    nns_model.model.layers[l].mlp.output[0, img_end:] += combined_coef * mlp_feature_proj 
                                    
                                    layer_output = nns_model.model.layers[l].output[0][0, img_end:]
                                    layer_feature_proj = (layer_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                                    nns_model.model.layers[l].output[0][0, img_end:] += combined_coef * layer_feature_proj            

                            else:
                                attn_output = nns_model.model.layers[layer_idx].self_attn.output[0][0, img_end:]
                                attn_feature_proj = (attn_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                                nns_model.model.layers[layer_idx].self_attn.output[0][0, img_end:] += combined_coef * attn_feature_proj

                                mlp_output = nns_model.model.layers[layer_idx].mlp.output[0, img_end:]
                                mlp_feature_proj = (mlp_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                                nns_model.model.layers[layer_idx].mlp.output[0, img_end:] += combined_coef * mlp_feature_proj 
                                    
                                layer_output = nns_model.model.layers[layer_idx].output[0][0, img_end:]
                                layer_feature_proj = (layer_output @ feature_vec.unsqueeze(0).T) * feature_vec.unsqueeze(0)
                                nns_model.model.layers[layer_idx].output[0][0, img_end:] += combined_coef * layer_feature_proj            


                logits_saved = nns_model.output.logits.save()

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
        if pred == label:
            correct += 1
        
        try:
            p_correct = p_yes if label == 1 else p_no
            sum_p_correct += float(p_correct)
            count_p += 1
        except Exception:
            pass

        if save_samples_path:
            try:
                record = {
                    "pass": "ablated",
                    "index_in_eval": int(i),
                    "dataset_index": int(idx),
                    "label": int(label),
                    "pred": int(pred),
                    "correct": int(pred == label),
                    "p_yes": float(p_yes),
                    "p_no": float(p_no),
                }
                try:
                    record["statement"] = statement
                    record["image_url"] = image_url
                except Exception:
                    pass
                with open(save_samples_path, "a") as f:
                    json.dump(record, f)
                    f.write("\n")
            except Exception as e:
                print(f"[WARN] Failed to save ablated sample record: {e}")
        
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
    """Main function to run SAE feature ablation evaluation."""
    print("[INFO] Starting VSR SAE feature-ablation evaluation script...")

    ap = argparse.ArgumentParser(description="Ablate a specific SAE feature and evaluate on VSR")
    ap.add_argument("--pairs", type=str, required=True, help="Comma-separated layer-feature tokens, e.g., L13F16873,L7F42")
    ap.add_argument("--split", type=str, default="train", choices=["train", "dev", "test", "train+dev+test"], help="VSR split")
    ap.add_argument("--max-samples", type=int, default=1000, help="Max samples to evaluate (0 for all)")
    ap.add_argument("--relations", type=str, default="", help="Comma-separated VSR relations to include (optional)")
    ap.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Directory to cache downloaded images")
    ap.add_argument("--sae-checkpoint-dir", type=str, required=True, help="Directory containing per-layer SAE checkpoints (text-only)")
    ap.add_argument("--apply-to-all-layers", action="store_true", help="If set, remove projection across ALL transformer layers using a shared subspace built from specified features")
    ap.add_argument("--steering-coef", type=float, default=0.0, help="If non-zero, add this multiple of the feature projection (negative to steer away, positive to steer towards). When zero, perform removal (ablation)")
    ap.add_argument("--negative-steering", action="store_true", help="Apply negative steering: get SAE coefficient for each text token and subtract coefficient * feature")
    ap.add_argument("--baseline-cache-dir", type=str, default="results/stage6/vsr_baseline_cache", help="Directory to cache/load baseline metrics (default: results/stage6/vsr_baseline_cache)")
    ap.add_argument("--save-samples", type=str, default="", help="If set, append per-sample JSONL records (baseline and ablated) to this file")
    ap.add_argument("--coef-steering", type=float, default=0.0, help="If non-zero, use per-token SAE coefficient times this value to steer along feature (positive for amplify, negative for suppress)")

    args = ap.parse_args()

    max_samples = None if args.max_samples == 0 else args.max_samples
    relations = [r.strip() for r in args.relations.split(",") if r.strip()] if args.relations else None
    layer_feature_pairs = _parse_lf_pairs_arg(args.pairs)
    
    layer_feature_pairs = sorted(set(layer_feature_pairs))
    if len(layer_feature_pairs) == 0:
        raise ValueError("Must specify --pairs with tokens like L13F16873")

    pairs_str = ",".join([f"L{l}F{f}" for l, f in layer_feature_pairs])
    print(f"[INFO] Configuration: features={pairs_str}, apply_to_all_layers={args.apply_to_all_layers}, split={args.split}, max_samples={max_samples}, relations={relations}, cache_dir={args.cache_dir}, baseline_cache_dir={args.baseline_cache_dir}, save_samples={bool(args.save_samples)}")

    save_samples_path = args.save_samples if args.save_samples else None
    clean, ablated = evaluate_with_feature_ablation(
        args.split,
        max_samples,
        args.cache_dir,
        layer_feature_pairs,
        args.sae_checkpoint_dir,
        relations,
        args.apply_to_all_layers,
        args.steering_coef,
        args.negative_steering,
        args.baseline_cache_dir,
        save_samples_path,
        coef_steering=args.coef_steering,
    )
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
