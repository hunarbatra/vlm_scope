#!/usr/bin/env python3
"""
Auto-interpret features using multi-dataset samples (VQA + VQA-spatial + VSR).
Matches the original auto_interp_common_features_raw_samples.py exactly.

For each feature:
- Gathers top-K samples across all datasets (sorted by magnitude)
- Sends to GPT API for one-sentence interpretation
- Validates with held-out positive + random negative samples
- Computes F1 score

Usage:
    python3 auto_interp_multidataset.py \
        --common-summary /path/to/dataset_all_features.json \
        --output-dir /path/to/auto_interp_multi \
        --api-key sk-... \
        --samples-per-feature 5
"""

import argparse
import base64
import io
import json
import os
import random
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image as PILImage

# ---- Config ----
ANALYSIS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis")
IMAGE_CACHE_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VQA_DATASET_NAME = "lmms-lab/VQAv2"
VSR_DATASET_NAME = "cambridgeltl/vsr_random"

API_MODEL = "gpt-5.4-mini"
API_URL = "https://api.openai.com/v1/chat/completions"


def _encode_pil_to_base64(img: PILImage.Image) -> str:
    """Encode image to base64 PNG (matching original)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class DatasetManager:
    """Lazy-load and cache datasets."""
    def __init__(self):
        self._vqa = None
        self._vsr = None
        self._vqa_spatial_indices = None

    def get_vqa(self):
        if self._vqa is None:
            from datasets import load_dataset
            print("Loading VQA validation...")
            self._vqa = load_dataset(VQA_DATASET_NAME, split="validation",
                                      cache_dir="/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache")
            print(f"VQA: {len(self._vqa)} samples")
        return self._vqa

    def get_vsr(self):
        if self._vsr is None:
            from datasets import load_dataset, concatenate_datasets
            print("Loading VSR (all splits)...")
            data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
            train = load_dataset(VSR_DATASET_NAME, data_files=data_files, split="train")
            dev = load_dataset(VSR_DATASET_NAME, data_files=data_files, split="dev")
            test = load_dataset(VSR_DATASET_NAME, data_files=data_files, split="test")
            self._vsr = concatenate_datasets([train, dev, test])
            print(f"VSR: {len(self._vsr)} samples")
        return self._vsr

    def get_vqa_spatial_indices(self):
        """Get indices mapping vqa_spatial subset idx -> base VQA idx."""
        if self._vqa_spatial_indices is None:
            cache_path = Path("/data1/vlm_scope_sae_mix448_textonly/vqa_spatial_filter")
            for f in sorted(cache_path.glob("indices_validation_*.json")) if cache_path.exists() else []:
                try:
                    payload = json.loads(f.read_text())
                    indices = payload.get("indices") or payload.get("filtered_indices")
                    if indices:
                        self._vqa_spatial_indices = [int(x) for x in indices]
                        print(f"VQA-spatial indices: {len(self._vqa_spatial_indices)} (from {f.name})")
                        return self._vqa_spatial_indices
                except Exception:
                    pass
            # Fallback: recompute
            print("Recomputing VQA-spatial filter...")
            import re
            from utils_datasets import SPATIAL_KEYWORDS, _compile_spatial_regex
            vqa = self.get_vqa()
            spatial_re = _compile_spatial_regex()
            self._vqa_spatial_indices = [i for i in range(len(vqa)) if spatial_re.search(str(vqa[i].get("question", "")))]
            print(f"VQA-spatial: {len(self._vqa_spatial_indices)} samples")
        return self._vqa_spatial_indices

    def load_image(self, dataset_name: str, sample_idx: int) -> Optional[PILImage.Image]:
        """Load image for a sample from any dataset."""
        try:
            if dataset_name in ("vqa", "vqa_spatial"):
                vqa = self.get_vqa()
                if dataset_name == "vqa_spatial":
                    indices = self.get_vqa_spatial_indices()
                    if indices and 0 <= sample_idx < len(indices):
                        base_idx = indices[sample_idx]
                    else:
                        base_idx = sample_idx
                else:
                    base_idx = sample_idx
                if 0 <= base_idx < len(vqa):
                    return vqa[base_idx]["image"].convert("RGB")
            elif dataset_name == "vsr":
                vsr = self.get_vsr()
                if 0 <= sample_idx < len(vsr):
                    url = vsr[sample_idx].get("image_link", "")
                    return self._load_vsr_image(url)
        except Exception:
            pass
        return None

    def _load_vsr_image(self, url: str) -> Optional[PILImage.Image]:
        if not url or not url.startswith("http"):
            return None
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = IMAGE_CACHE_DIR / f"{url_hash}.jpg"
        try:
            if cache_path.exists():
                return PILImage.open(cache_path).convert("RGB")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
            img.save(cache_path, "JPEG")
            return img
        except Exception:
            return None

    def get_random_negative(self, used_keys: set) -> Optional[Dict]:
        """Get a random negative sample from VQA (matching original)."""
        vqa = self.get_vqa()
        vqa_len = len(vqa)
        tried = set()
        while len(tried) < vqa_len * 3:
            ridx = random.randint(0, vqa_len - 1)
            if ridx in tried:
                continue
            tried.add(ridx)
            if ("vqa", ridx) in used_keys:
                continue
            try:
                img = vqa[ridx]["image"].convert("RGB")
                b64 = _encode_pil_to_base64(img)
                text = str(vqa[ridx].get("question", "")).strip()
                return {
                    "dataset": "vqa",
                    "sample_idx": ridx,
                    "text": text if text else "Answer the question.",
                    "image_b64": b64,
                    "label": 0,
                }
            except Exception:
                continue
        return None


def _call_api(messages, api_key, timeout_s=120, max_retries=3):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": API_MODEL,
        "messages": messages,
        "max_completion_tokens": 400,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] API call failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def _create_interp_prompt(feature_key: str, samples: List[Dict]) -> List[Dict]:
    """Matching original exactly."""
    system_message = (
        "You are analyzing individual neurons using their top activating samples, each with an image and a question.\n\n"
        "Task: Produce a concise ONE SENTENCE description that completes the phrase: 'this neuron activates for ...'.\n\n"
        "Guidelines:\n"
        "- Base your description on pattens supported by both image and text.\n"
        "- Focus on consistent visual-spatial patterns (objects, parts, relations, or configurations).\n"
        "- Be specific and concrete; avoid vague or generic phrases.\n"
        "- Keep the output to one short, lower-case sentence with no hedging.\n"
        "- Return strict JSON only with fields.\n"
    )

    content = []
    content.append({"type": "text", "text": f"Feature: {feature_key}. Analyze the following samples."})
    for i, s in enumerate(samples):
        header = (
            f"Sample {i+1}: dataset={s['dataset']}, sample_idx={s['sample_idx']}, "
            f"magnitude={s.get('magnitude', 0):.4f}.\n"
            f"Text: {s.get('text', '')}"
        )
        content.append({"type": "text", "text": header})
        if s.get("image_b64"):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{s['image_b64']}"}
            })

    schema = (
        "Return strict JSON matching exactly:\n"
        "{\n"
        '  "description": "one concise sentence"\n'
        "}"
    )
    content.append({"type": "text", "text": schema})

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content},
    ]


def _create_eval_prompt(description: str, samples: List[Dict]) -> List[Dict]:
    """Matching original exactly."""
    system_message = (
        "You are validating a neuron description by reviewing short examples (each has an image and a brief text).\n\n"
        "Task: For each sample, decide if it reasonably matches the neuron description. Output 1 if the description is supported; otherwise 0.\n\n"
        "Guidelines:\n"
        "- Use both the image and the text; let the text clarify ambiguous visuals when helpful.\n"
        "- Be tolerant of minor mismatches; look for the main idea rather than exact wording.\n"
        "- Prefer consistency across similar cases.\n\n"
        "Output format:\n"
        "- Return JSON only.\n"
        '- Use exactly: {"classifications": [<0 or 1 per sample, in order>]}.\n'
    )

    content = []
    content.append({"type": "text", "text": f"Neuron description: {description}"})
    for i, s in enumerate(samples):
        header = (
            f"Sample {i+1}: dataset={s['dataset']}, sample_idx={s['sample_idx']}\n"
            f"Text: {s.get('text', '')}"
        )
        content.append({"type": "text", "text": header})
        if s.get("image_b64"):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{s['image_b64']}"}
            })

    schema = (
        "Return strict JSON matching exactly:\n"
        "{\n"
        '  "classifications": [0, 1, 0]\n'
        "}"
    )
    content.append({"type": "text", "text": schema})

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content},
    ]


def _compute_f1(preds, labels):
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    denom = 2 * tp + fp + fn
    return (2.0 * tp) / denom if denom > 0 else 0.0


def process_feature(feature_key, feature_data, dm, api_key,
                     samples_per_feature=5, output_dir=None):
    """Process one feature: interpret + validate using multi-dataset samples."""

    # Check if done
    if output_dir:
        result_path = Path(output_dir) / f"{feature_key}.json"
        if result_path.exists():
            return None

    layer = feature_data["layer"]
    feat = feature_data["feature"]
    datasets_info = feature_data.get("datasets", {})

    if not datasets_info:
        return None

    # Gather all samples across datasets, sort by magnitude
    all_samples_raw = []
    for ds_name, ds_data in datasets_info.items():
        for s in ds_data.get("top_samples", []):
            all_samples_raw.append({
                "dataset": ds_name,
                "sample_idx": s["sample_idx"],
                "magnitude": s.get("magnitude", 0),
                "text": s.get("question", ""),
            })
    all_samples_raw.sort(key=lambda x: x["magnitude"], reverse=True)

    if not all_samples_raw:
        return None

    # Load images for top samples (dedup by text)
    interp_samples = []
    seen_texts = set()
    for s in all_samples_raw:
        if len(interp_samples) >= samples_per_feature:
            break
        text_norm = s["text"].strip().lower()
        if text_norm in seen_texts:
            continue
        img = dm.load_image(s["dataset"], s["sample_idx"])
        if img is None:
            continue
        seen_texts.add(text_norm)
        b64 = _encode_pil_to_base64(img)
        interp_samples.append({
            **s,
            "image_b64": b64,
            "label": 1,
        })

    if not interp_samples:
        return None

    # Step 1: Interpret
    messages = _create_interp_prompt(feature_key, interp_samples)
    interp_resp = _call_api(messages, api_key)
    if not interp_resp:
        return None

    description = str(interp_resp.get("description", "")).strip()
    if not description:
        return None

    # Step 2: Validate - held-out positives
    used_keys = {(s["dataset"], s["sample_idx"]) for s in interp_samples}
    eval_pos = []
    for s in all_samples_raw:
        if len(eval_pos) >= 5:
            break
        key = (s["dataset"], s["sample_idx"])
        if key in used_keys:
            continue
        img = dm.load_image(s["dataset"], s["sample_idx"])
        if img is None:
            continue
        b64 = _encode_pil_to_base64(img)
        eval_pos.append({
            **s,
            "image_b64": b64,
            "label": 1,
        })
        used_keys.add(key)

    # Random negatives from VQA (matching original)
    eval_neg = []
    for _ in range(5):
        neg = dm.get_random_negative(used_keys)
        if neg:
            eval_neg.append(neg)
            used_keys.add((neg["dataset"], neg["sample_idx"]))

    # Two-round eval (matching original)
    r1_pos_n = min(3, len(eval_pos))
    r1_neg_n = min(2, len(eval_neg))
    r2_pos_n = min(2, max(0, len(eval_pos) - r1_pos_n))
    r2_neg_n = min(3, max(0, len(eval_neg) - r1_neg_n))

    r1_payload = eval_pos[:r1_pos_n] + eval_neg[:r1_neg_n]
    r2_payload = eval_pos[r1_pos_n:r1_pos_n + r2_pos_n] + eval_neg[r1_neg_n:r1_neg_n + r2_neg_n]

    all_preds = []
    all_labels = []
    for payload in (r1_payload, r2_payload):
        if not payload:
            continue
        random.shuffle(payload)
        eval_messages = _create_eval_prompt(description, payload)
        eval_resp = _call_api(eval_messages, api_key)
        if eval_resp:
            try:
                preds = [int(x) for x in (eval_resp.get("classifications") or [])]
                labels = [int(s.get("label", 0)) for s in payload][:len(preds)]
                all_preds.extend(preds)
                all_labels.extend(labels)
            except Exception:
                continue

    f1 = _compute_f1(all_preds, all_labels) if all_preds and all_labels else None

    # Save
    samples_for_saving = [{
        "dataset": s["dataset"],
        "sample_idx": s["sample_idx"],
        "magnitude": s["magnitude"],
        "text": s.get("text", ""),
    } for s in interp_samples]

    record = {
        "feature_key": feature_key,
        "layer": layer,
        "feature": feat,
        "samples_used": samples_for_saving,
        "interpretation": {"description": description},
        "validation": {"f1": f1, "n_preds": len(all_preds)},
    }

    if output_dir:
        result_path = Path(output_dir) / f"{feature_key}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(record, f, indent=2)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-summary", required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--samples-per-feature", type=int, default=5)
    parser.add_argument("--delay-s", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] No API key.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR / "auto_interp_multi"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load common features summary
    print(f"Loading: {args.common_summary}")
    with open(args.common_summary) as f:
        data = json.load(f)
    features = data.get("features", {})
    feature_keys = sorted(features.keys())
    print(f"Features: {len(feature_keys)}")

    if args.limit > 0:
        feature_keys = feature_keys[:args.limit]

    # Init dataset manager
    dm = DatasetManager()
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for i, fkey in enumerate(feature_keys):
        result = process_feature(fkey, features[fkey], dm, api_key,
                                  samples_per_feature=args.samples_per_feature,
                                  output_dir=str(output_dir))
        if result is None:
            skipped += 1
        else:
            processed += 1
            desc = result["interpretation"]["description"]
            f1 = result["validation"].get("f1")
            f1_str = f"{f1:.2f}" if f1 is not None else "N/A"
            print(f"[{i+1}/{len(feature_keys)}] {fkey}: F1={f1_str} | {desc[:80]}", flush=True)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(feature_keys)} ({processed} done, {skipped} skipped)", flush=True)

        time.sleep(args.delay_s)

    summary = {"total": len(feature_keys), "processed": processed, "skipped": skipped}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Processed {processed}, skipped {skipped}. Results in {output_dir}")


if __name__ == "__main__":
    main()
