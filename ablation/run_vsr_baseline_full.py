#!/usr/bin/env python3
"""
Run baseline VSR evaluation (no ablation, no filtering) over a chosen split and
save per-sample results to a JSONL file.

Outputs one JSON line per sample with:
- dataset_index, label, pred, correct, p_yes, p_no, statement, image_url

Example:
CUDA_VISIBLE_DEVICES=1 python ablation/run_vsr_baseline_full.py \
  --split train \
  --max-samples 0 \
  --cache-dir /scratch/local/ssd/lachin/vsr_image_cache \
  --out-jsonl results/vsr_baseline_train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import List, Set, Tuple

import dotenv
import requests
import torch
from datasets import load_dataset, concatenate_datasets
from PIL import Image

import sys
sys.path.append("finetune/vqa")
from utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    get_image_token_positions,
)
sys.path.pop()

dotenv.load_dotenv(".env")

PROGRESS_INTERVAL = 100
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CACHE_DIR = "/scratch/local/ssd/lachin/vsr_image_cache"
DEFAULT_TIMEOUT = 10


def load_vsr(split: str = "train"):
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    if split == "train+dev+test":
        train_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="train")
        dev_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="dev")
        test_ds = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="test")
        return concatenate_datasets([train_ds, dev_ds, test_ds])
    return load_dataset("cambridgeltl/vsr_random", data_files=data_files, split=split)


def load_image_with_cache(url: str, cache_dir: str = DEFAULT_CACHE_DIR, timeout: int = DEFAULT_TIMEOUT) -> Image.Image:
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{url_hash}.jpg")
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        try:
            img.save(cache_path, "JPEG")
        except Exception:
            pass
        return img
    except Exception:
        return Image.new("RGB", DEFAULT_IMAGE_SIZE, (128, 128, 128))


def _first_token_ids(tokenizer, texts: List[str]) -> Set[int]:
    ids: Set[int] = set()
    for t in texts:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            ids.add(toks[0])
    return ids


def _build_yes_no_token_sets(tokenizer) -> Tuple[Set[int], Set[int]]:
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


def build_prompt(statement: str) -> str:
    s = statement.strip()
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {s}\n"
        "Answer:"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline VSR evaluation over full split and save per-sample results")
    ap.add_argument("--split", type=str, default="train+dev+test", choices=["train", "dev", "test", "train+dev+test"])
    ap.add_argument("--max-samples", type=int, default=0, help="0 for all samples")
    ap.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out-jsonl", type=str, default="results/vsr_baseline_full.jsonl")

    args = ap.parse_args()

    print(f"[INFO] Loading VSR dataset split={args.split} ...")
    ds = load_vsr(args.split)
    print(f"[INFO] Loaded {len(ds)} VSR samples")

    # Model
    print("[INFO] Initializing VLM model...")
    tokenizer, hf_model, image_processor = initialize_vlm_model("llava-more", device="cuda")
    hf_model.eval()
    print("[INFO] VLM model ready!")

    n_total = len(ds)
    n = n_total if args.max_samples == 0 else min(n_total, int(args.max_samples))
    indices = list(range(n))
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    yes_ids, no_ids = _build_yes_no_token_sets(tokenizer)

    with open(out_path, "w") as f:
        for i, idx in enumerate(indices):
            if i % PROGRESS_INTERVAL == 0:
                print(f"[INFO] Baseline processing sample {i}/{n}...")

            ex = ds[idx]
            image_url = ex.get("image_link") or ex.get("image") or ""
            statement = str(ex.get("caption") or ex.get("text") or "").strip()
            label = int(ex.get("label", 0))
            if not statement:
                continue

            image = load_image_with_cache(image_url, cache_dir=args.cache_dir)
            prompt = build_prompt(statement)

            try:
                p_yes, p_no = _next_token_yes_no_probs(image, prompt, image_processor, hf_model, tokenizer, yes_ids, no_ids)
            except Exception:
                p_yes, p_no = 0.0, 0.0

            pred = 1 if float(p_yes) > float(p_no) else 0

            record = {
                "dataset_index": int(idx),
                "label": int(label),
                "pred": int(pred),
                "correct": int(pred == label),
                "p_yes": float(p_yes),
                "p_no": float(p_no),
                "statement": statement,
                "image_url": image_url,
            }
            json.dump(record, f)
            f.write("\n")

            if (i + 1) % PROGRESS_INTERVAL == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[INFO] Wrote baseline per-sample JSONL → {out_path}")


if __name__ == "__main__":
    main()



