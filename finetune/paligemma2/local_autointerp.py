"""
Local auto-interpretability: call GPT-4o-mini on top-activating samples.

Reads top_samples.json (from modal_find_top_samples.py), loads images from
VQA/VSR datasets, sends to GPT-4o-mini for interpretation + validation.

No GPU needed — just dataset loading + API calls.

Usage:
    export OPENAI_API_KEY=sk-...
    python local_autointerp.py [--top-samples top_samples.json] [--output-dir results/autointerp]
"""
import argparse
import base64
import io
import json
import os
import random
import time
from pathlib import Path

import requests
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def encode_pil_b64(img, max_size=224, quality=80):
    img = img.copy()
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def call_gpt4o(api_key, messages, max_tokens=400):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions",
                         headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def load_datasets():
    print("[INFO] Loading VQA validation...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")

    print("[INFO] Loading VSR (all splits)...")
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_splits = [load_dataset("cambridgeltl/vsr_random", data_files=data_files, split=s)
                  for s in ["train", "dev", "test"]]
    vsr = concatenate_datasets(vsr_splits)

    return vqa, vsr


def load_sample_image(sample_info, vqa, vsr, img_cache_dir=None):
    """Load image and text for a sample. Returns (img, text) or (None, None)."""
    ds = sample_info["dataset"]
    idx = sample_info["sample_idx"]

    try:
        if ds == "vqa":
            ex = vqa[idx]
            img = ex.get("image")
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            elif isinstance(img, str):
                img = Image.open(img).convert("RGB")
            else:
                return None, None
            return img, str(ex.get("question", ""))

        elif ds == "vsr":
            ex = vsr[idx]
            caption = str(ex.get("caption", ""))
            url = ex.get("image_link", "")

            # Try cache first
            if img_cache_dir:
                import hashlib
                url_hash = hashlib.md5(url.encode()).hexdigest()
                cache_path = os.path.join(img_cache_dir, f"{url_hash}.jpg")
                if os.path.exists(cache_path):
                    return Image.open(cache_path).convert("RGB"), caption

            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return img, caption

    except Exception as e:
        print(f"  [WARN] Failed to load {ds}[{idx}]: {e}")

    return None, None


def interpret_feature(api_key, fkey, interp_samples, vqa, vsr):
    """Send top samples to GPT-4o-mini, get interpretation."""
    content = [{
        "type": "text",
        "text": (f"Feature: {fkey}. Analyze the following top-activating samples "
                 f"from a vision-language model's SAE (Sparse Autoencoder) feature.")
    }]

    loaded = []
    for i, s in enumerate(interp_samples):
        img, text = load_sample_image(s, vqa, vsr)
        if img is None:
            continue
        b64 = encode_pil_b64(img)
        content.append({
            "type": "text",
            "text": f"Sample {i+1} (dataset={s['dataset']}, activation={s['activation']:.3f}):\nText: {text}"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        loaded.append({"ds": s["dataset"], "idx": s["sample_idx"], "act": s["activation"], "text": text, "b64": b64})

    if len(loaded) < 2:
        return None, loaded

    content.append({
        "type": "text",
        "text": (
            "Based on the images and text above, produce a concise ONE SENTENCE description "
            "completing: 'this neuron activates for ...'\n"
            "Focus on consistent visual-spatial patterns (objects, relations, positions, configurations).\n"
            "Be specific and concrete. Return strict JSON: {\"description\": \"...\"}"
        )
    })

    messages = [
        {"role": "system", "content": (
            "You are analyzing individual neurons in a vision-language model using their "
            "top activating samples. Each sample has an image and associated text (question or caption). "
            "Identify what visual or spatial pattern consistently triggers this neuron."
        )},
        {"role": "user", "content": content},
    ]

    result = call_gpt4o(api_key, messages)
    return result, loaded


def validate_description(api_key, description, valid_pos, valid_neg, vqa, vsr):
    """Validate interpretation with held-out samples. Returns F1 score."""
    eval_batch = []

    for s in valid_pos:
        img, text = load_sample_image(s, vqa, vsr)
        if img is not None:
            eval_batch.append({"text": text, "b64": encode_pil_b64(img), "label": 1})

    for idx in valid_neg:
        try:
            ex = vqa[idx]
            img = ex.get("image")
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            elif isinstance(img, str):
                img = Image.open(img).convert("RGB")
            else:
                continue
            eval_batch.append({"text": str(ex.get("question", "")), "b64": encode_pil_b64(img), "label": 0})
        except Exception:
            continue

    if not eval_batch:
        return None

    random.shuffle(eval_batch)

    eval_content = [{"type": "text", "text": f"Neuron description: {description}\n\nFor each sample, output 1 if it matches the description, 0 if not."}]
    for i, s in enumerate(eval_batch):
        eval_content.append({"type": "text", "text": f"Sample {i+1}: {s['text']}"})
        eval_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s['b64']}"}})
    eval_content.append({"type": "text", "text": "Return JSON: {\"classifications\": [0 or 1 per sample]}"})

    messages = [
        {"role": "system", "content": "Validate a neuron description against sample images. Output 1 if sample matches, 0 otherwise."},
        {"role": "user", "content": eval_content},
    ]

    try:
        resp = call_gpt4o(api_key, messages)
        if resp and "classifications" in resp:
            preds = [int(x) for x in resp["classifications"]]
            labels = [s["label"] for s in eval_batch[:len(preds)]]
            tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
            fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
            fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
            denom = 2 * tp + fp + fn
            return (2 * tp / denom) if denom > 0 else 0.0
    except Exception as e:
        print(f"  [WARN] Validation API error: {e}")

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-samples", type=str, default="top_samples.json")
    parser.add_argument("--output-dir", type=str, default="results/autointerp")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] Set OPENAI_API_KEY or pass --api-key")
        return

    # Load top samples
    top_samples_path = Path(args.top_samples)
    if not top_samples_path.exists():
        print(f"[ERROR] {top_samples_path} not found. Run modal_find_top_samples.py first.")
        return

    with open(top_samples_path) as f:
        top_samples = json.load(f)

    print(f"[INFO] {len(top_samples)} features in top_samples.json")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    vqa, vsr = load_datasets()

    all_results = []
    feature_keys = sorted(top_samples.keys())

    for fi, fkey in enumerate(tqdm(feature_keys, desc="Interpreting")):
        samples = top_samples[fkey]
        if not samples:
            continue

        # Check if already done
        out_file = output_dir / f"{fkey}.json"
        if out_file.exists():
            existing = json.loads(out_file.read_text())
            all_results.append(existing)
            continue

        interp_samples = samples[:5]
        valid_samples = samples[5:10]

        print(f"\n[{fi+1}/{len(feature_keys)}] {fkey}")

        # Interpret
        try:
            result, loaded = interpret_feature(api_key, fkey, interp_samples, vqa, vsr)
        except Exception as e:
            print(f"  [ERROR] Interpretation failed: {e}")
            time.sleep(1)
            continue

        if not result or "description" not in result:
            print(f"  [WARN] No description returned")
            continue

        description = result["description"]
        print(f"  -> {description}")

        # Validate
        used_indices = set(s["sample_idx"] for s in samples)
        neg_indices = []
        tried = set()
        while len(neg_indices) < 5:
            ridx = random.randint(0, len(vqa) - 1)
            if ridx in tried or ridx in used_indices:
                continue
            tried.add(ridx)
            neg_indices.append(ridx)

        f1 = None
        if valid_samples:
            try:
                f1 = validate_description(api_key, description, valid_samples, neg_indices, vqa, vsr)
                if f1 is not None:
                    print(f"  F1: {f1:.3f}")
            except Exception as e:
                print(f"  [WARN] Validation failed: {e}")

        record = {
            "feature_key": fkey,
            "layer": int(fkey.split("_")[0][1:]),
            "feature": int(fkey.split("_")[1][1:]),
            "description": description,
            "validation_f1": f1,
            "n_interp_samples": len(loaded) if loaded else 0,
            "top_activation": samples[0]["activation"] if samples else 0,
            "interp_samples": [{"ds": s["ds"], "idx": s["idx"], "act": s["act"], "text": s["text"]}
                               for s in (loaded or [])],
        }
        all_results.append(record)

        with open(out_file, "w") as f:
            json.dump(record, f, indent=2)

        time.sleep(args.delay)

    # Summary
    summary = {
        "total_features": len(feature_keys),
        "interpreted": len(all_results),
        "results": all_results,
    }
    with open(output_dir / "autointerp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print("AUTO-INTERP RESULTS")
    print("=" * 90)
    results_sorted = sorted([r for r in all_results if r.get("validation_f1") is not None],
                            key=lambda x: -x["validation_f1"])

    print(f"\n{'Feature':<15} {'F1':>5} {'Description'}")
    print("-" * 90)
    for r in results_sorted:
        desc = r["description"][:65]
        print(f"{r['feature_key']:<15} {r['validation_f1']:>4.2f}  {desc}")

    spatial_kw = ["left", "right", "above", "below", "behind", "front", "near",
                  "between", "beside", "under", "over", "next to", "spatial",
                  "position", "location", "arrangement", "top", "bottom"]
    n_spatial = sum(1 for r in all_results
                    if any(kw in r.get("description", "").lower() for kw in spatial_kw))
    print(f"\nSpatial descriptions: {n_spatial}/{len(all_results)}")
    print(f"[DONE] Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
