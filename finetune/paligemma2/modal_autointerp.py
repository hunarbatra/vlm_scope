"""
Auto-interpretability pipeline for PaliGemma2 SAE features on Modal.

For each feature in our final features list:
1. Load cached activations from the volume
2. Encode through the SAE to find top-activating samples
3. Load the corresponding images + text from VQA/VSR datasets
4. Send top-5 samples (image + text) to GPT-4o-mini
5. Get back a one-sentence description: "this neuron activates for ..."
6. Validate with held-out positive + random negative samples (F1 score)

Runs on 2 GPUs (for SAE encoding) + CPU workers for API calls.

Usage:
    cd finetune/paligemma2
    export OPENAI_API_KEY=sk-...
    MODAL_PROFILE=hunar-oxford modal run modal_autointerp.py
"""

import os
import json
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("vlm-scope-autointerp")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers>=4.44", "sae-lens>=4.0",
        "nnsight>=0.3", "datasets", "h5py", "tqdm", "huggingface-hub",
        "Pillow", "numpy", "accelerate", "requests",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_MODE": "disabled",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

# Use the same gpu_image for CPU tasks too — avoids building two images
cpu_image = gpu_image

# Constants
RESULTS_BASE = "/vol/results/paligemma2"
SAE_TYPE = "jumprelu"
N_TRAINING_SAMPLES = 5000  # VQA samples used for activation collection
TOP_K_SAMPLES = 10  # top samples per feature (5 for interp, 5 for validation)
N_LAYERS = 26


# --------------- Step 1: Find top-activating samples per feature (GPU) ---------------

@app.function(image=gpu_image, gpu="A100", volumes={"/vol": volume}, timeout=7200)
def find_top_samples(feature_list: list):
    """For each (layer, feature), find top-activating VQA sample indices.

    Uses cached activations from the volume (h5 files from training pipeline).
    Returns: {f"L{layer}_F{feat}": [(sample_idx, max_activation), ...], ...}
    """
    import sys
    import torch
    import numpy as np
    import h5py
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_jumprelu_sae

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    ckpt_dir = Path(RESULTS_BASE) / f"run{sae_suffix}" / "checkpoints"
    act_dir = Path(RESULTS_BASE) / "run" / "activations"

    # H5 files are: chunk_0_1000.h5, chunk_1000_2000.h5, ...
    # Structure: layer_X/sample_Y -> (seq, 2304)
    chunk_files = sorted(act_dir.glob("chunk_*.h5"))
    if not chunk_files:
        print("[TopSamples] No activation chunks found!")
        return {}
    # Limit to first 10 chunks (10K samples) to avoid scanning 55 × 63GB
    chunk_files = chunk_files[:10]
    print(f"[TopSamples] Found {len(chunk_files)} chunks to scan")

    # Group features by layer
    from collections import defaultdict
    layer_features = defaultdict(list)
    for layer, feat in feature_list:
        layer_features[layer].append(feat)

    # Load all needed SAEs upfront
    saes = {}
    for layer_idx in sorted(layer_features.keys()):
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP L{layer_idx} — no checkpoint")
            continue
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device="cuda", cache_dir="/vol/cache/huggingface")
        sae.eval()
        saes[layer_idx] = sae
        print(f"[TopSamples] Loaded SAE for layer {layer_idx}")

    results = {}
    # Initialize per-feature top-k trackers
    feat_top = {}  # (layer, feat) -> [(activation, sample_idx)]
    for layer, feat in feature_list:
        feat_top[(layer, feat)] = []

    # Scan chunks — each chunk has all layers
    for ci, cf in enumerate(chunk_files):
        print(f"[TopSamples] Scanning chunk {ci+1}/{len(chunk_files)}: {cf.name}")
        try:
            with h5py.File(str(cf), "r") as hf:
                for layer_idx in sorted(layer_features.keys()):
                    if layer_idx not in saes:
                        continue
                    layer_key = f"layer_{layer_idx}"
                    if layer_key not in hf:
                        continue
                    layer_grp = hf[layer_key]
                    sae = saes[layer_idx]
                    features = layer_features[layer_idx]

                    sample_keys = [k for k in layer_grp.keys() if k.startswith("sample_")]
                    for sk in sample_keys:
                        sample_idx = int(sk.split("_")[1])
                        act = torch.from_numpy(layer_grp[sk][:]).float().to("cuda")

                        with torch.no_grad():
                            codes = sae.encode(act)  # (seq, d_sae)

                        for feat_idx in features:
                            max_act = codes[:, feat_idx].max().item()
                            if max_act > 0:
                                heap = feat_top[(layer_idx, feat_idx)]
                                if len(heap) < TOP_K_SAMPLES:
                                    heap.append((max_act, sample_idx))
                                    heap.sort(key=lambda x: x[0])
                                elif max_act > heap[0][0]:
                                    heap[0] = (max_act, sample_idx)
                                    heap.sort(key=lambda x: x[0])
        except Exception as e:
            print(f"  [WARN] Error reading {cf.name}: {e}")
            continue

    # Cleanup SAEs
    for sae in saes.values():
        del sae
    torch.cuda.empty_cache()

    for (layer_idx, feat_idx), heap in feat_top.items():
        key = f"L{layer_idx}_F{feat_idx}"
        top = sorted(heap, key=lambda x: -x[0])
        results[key] = [(act, sidx) for act, sidx in top]
        if top:
            print(f"  {key}: top_act={top[0][0]:.4f}, n_samples={len(top)}")
        else:
            print(f"  {key}: NO activating samples found")

    # Save to volume
    out_path = Path(RESULTS_BASE) / "analysis" / "autointerp" / "top_samples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"[TopSamples] Saved {len(results)} features to {out_path}")

    return results


# --------------- Step 2: Call GPT-4o-mini for interpretation (CPU) ---------------

@app.function(image=cpu_image, volumes={"/vol": volume}, timeout=7200,
              secrets=[modal.Secret.from_dict({"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")})])
def interpret_features(top_samples: dict):
    """For each feature, load images for top samples, send to GPT-4o-mini."""
    import base64
    import io
    import time
    import random
    import requests as req
    from PIL import Image
    from datasets import load_dataset
    from pathlib import Path

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "No OPENAI_API_KEY"}

    print("[Interp] Loading VQA dataset...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")

    print("[Interp] Loading VSR dataset...")
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    try:
        from datasets import concatenate_datasets
        vsr_splits = [load_dataset("cambridgeltl/vsr_random", data_files=data_files, split=s)
                      for s in ["train", "dev", "test"]]
        vsr = concatenate_datasets(vsr_splits)
    except Exception:
        vsr = None

    IMAGE_CACHE_DIR = "/vol/cache/vsr_images"

    def encode_pil_b64(img):
        buf = io.BytesIO()
        img.thumbnail((224, 224))
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()

    def load_vqa_sample(idx):
        ex = vqa[idx]
        img = ex.get("image")
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        elif isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            return None, ""
        question = str(ex.get("question", ""))
        return img, question

    def load_vsr_sample(idx):
        if vsr is None:
            return None, ""
        ex = vsr[idx]
        caption = str(ex.get("caption", ""))
        url = ex.get("image_link", "")
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(IMAGE_CACHE_DIR, f"{url_hash}.jpg")
        try:
            if os.path.exists(cache_path):
                img = Image.open(cache_path).convert("RGB")
            else:
                resp = req.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return img, caption

    def call_gpt4o(messages, max_tokens=400):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = req.post("https://api.openai.com/v1/chat/completions",
                           headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"  [API ERROR] {e}")
            return None

    all_results = []
    feature_keys = sorted(top_samples.keys())
    print(f"[Interp] Processing {len(feature_keys)} features...")

    for fi, fkey in enumerate(feature_keys):
        samples = top_samples[fkey]
        if not samples:
            print(f"[{fi+1}/{len(feature_keys)}] {fkey}: no samples, skip")
            continue

        # Load top 5 for interpretation, next 5 for validation
        interp_samples = samples[:5]
        valid_samples = samples[5:10]

        # Build interpretation prompt
        content = [{"type": "text", "text": f"Feature: {fkey}. Analyze the following top-activating samples from a vision-language model's SAE feature."}]

        loaded_interp = []
        for i, (act, sidx) in enumerate(interp_samples):
            if sidx < 0:
                # VSR sample
                real_idx = -(sidx + 1)
                img, text = load_vsr_sample(real_idx)
                ds = "vsr"
            else:
                img, text = load_vqa_sample(sidx)
                ds = "vqa"

            if img is None:
                continue

            b64 = encode_pil_b64(img)
            content.append({"type": "text", "text": f"Sample {i+1} (dataset={ds}, activation={act:.3f}):\nText: {text}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            loaded_interp.append({"ds": ds, "idx": sidx, "act": act, "text": text, "b64": b64})

        if len(loaded_interp) < 2:
            print(f"[{fi+1}/{len(feature_keys)}] {fkey}: only {len(loaded_interp)} valid samples, skip")
            continue

        content.append({"type": "text", "text": (
            "Based on the images and text above, produce a concise ONE SENTENCE description "
            "completing: 'this neuron activates for ...'\n"
            "Focus on consistent visual-spatial patterns (objects, relations, positions, configurations).\n"
            "Be specific and concrete. Return strict JSON: {\"description\": \"...\"}"
        )})

        messages = [
            {"role": "system", "content": "You are analyzing individual neurons in a vision-language model using their top activating samples. Each sample has an image and associated text (question or caption). Identify what visual or spatial pattern consistently triggers this neuron."},
            {"role": "user", "content": content},
        ]

        print(f"[{fi+1}/{len(feature_keys)}] {fkey}: calling GPT-4o-mini for interpretation...")
        interp_resp = call_gpt4o(messages)
        if not interp_resp or "description" not in interp_resp:
            print(f"  FAILED to get interpretation")
            continue

        description = interp_resp["description"]
        print(f"  Description: {description}")

        # Validation: test on held-out positives + random negatives
        val_pos = []
        for act, sidx in valid_samples:
            if sidx < 0:
                real_idx = -(sidx + 1)
                img, text = load_vsr_sample(real_idx)
                ds = "vsr"
            else:
                img, text = load_vqa_sample(sidx)
                ds = "vqa"
            if img is not None:
                val_pos.append({"ds": ds, "text": text, "b64": encode_pil_b64(img), "label": 1})

        # Random negatives from VQA
        val_neg = []
        tried = set()
        used_indices = set(abs(s) for _, s in samples)
        while len(val_neg) < min(5, len(val_pos)):
            ridx = random.randint(0, len(vqa) - 1)
            if ridx in tried or ridx in used_indices:
                continue
            tried.add(ridx)
            img, text = load_vqa_sample(ridx)
            if img is not None:
                val_neg.append({"ds": "vqa_random", "text": text, "b64": encode_pil_b64(img), "label": 0})

        eval_batch = val_pos + val_neg
        random.shuffle(eval_batch)

        if eval_batch:
            eval_content = [{"type": "text", "text": f"Neuron description: {description}\n\nFor each sample, output 1 if it matches the description, 0 if not."}]
            for i, s in enumerate(eval_batch):
                eval_content.append({"type": "text", "text": f"Sample {i+1}: {s['text']}"})
                eval_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s['b64']}"}})
            eval_content.append({"type": "text", "text": "Return JSON: {\"classifications\": [0 or 1 per sample]}"})

            eval_messages = [
                {"role": "system", "content": "Validate a neuron description against sample images. Output 1 if sample matches, 0 otherwise."},
                {"role": "user", "content": eval_content},
            ]

            eval_resp = call_gpt4o(eval_messages)
            f1 = None
            if eval_resp and "classifications" in eval_resp:
                preds = [int(x) for x in eval_resp["classifications"]]
                labels = [s["label"] for s in eval_batch[:len(preds)]]
                tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
                fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
                fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
                denom = 2 * tp + fp + fn
                f1 = (2 * tp / denom) if denom > 0 else 0.0
                print(f"  Validation F1: {f1:.3f} (tp={tp}, fp={fp}, fn={fn})")
        else:
            f1 = None

        record = {
            "feature_key": fkey,
            "layer": int(fkey.split("_")[0][1:]),
            "feature": int(fkey.split("_")[1][1:]),
            "description": description,
            "validation_f1": f1,
            "n_interp_samples": len(loaded_interp),
            "top_activation": samples[0][0] if samples else 0,
            "interp_samples": [{"ds": s["ds"], "idx": s["idx"], "act": s["act"], "text": s["text"]} for s in loaded_interp],
        }
        all_results.append(record)

        # Save incrementally
        out_dir = Path(RESULTS_BASE) / "analysis" / "autointerp"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{fkey}.json", "w") as f:
            json.dump(record, f, indent=2)

        time.sleep(0.5)  # Rate limiting

    # Save summary
    out_dir = Path(RESULTS_BASE) / "analysis" / "autointerp"
    summary = {
        "total_features": len(feature_keys),
        "interpreted": len(all_results),
        "results": all_results,
    }
    with open(out_dir / "autointerp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()

    print(f"\n[Interp] Done! {len(all_results)}/{len(feature_keys)} features interpreted.")
    return summary


# --------------- Step 3: Load feature list ---------------

@app.function(image=gpu_image, volumes={"/vol": volume}, timeout=120)
def load_features():
    """Load final features list from volume."""
    import pandas as pd
    from pathlib import Path

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    csv_path = Path(RESULTS_BASE) / "analysis" / f"final_features{sae_suffix}" / "final_spatial_visual_features.csv"
    if not csv_path.exists():
        print(f"[WARN] No CSV at {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    features = [(int(row["layer"]), int(row["feature"])) for _, row in df.iterrows()]
    print(f"[INFO] Loaded {len(features)} features from {csv_path}")
    return features


# --------------- Entrypoint ---------------

@app.local_entrypoint()
def main():
    import argparse
    print("=" * 60)
    print("AUTO-INTERPRETABILITY PIPELINE — Step 1: Find Top Samples")
    print("PaliGemma2 JumpReLU SAE Features")
    print("=" * 60)

    # Step 1: Load feature list
    print("\n[Step 1] Loading features...")
    features = load_features.remote()
    print(f"  {len(features)} features to interpret")

    if not features:
        print("[ERROR] No features found")
        return

    # Step 2: Find top-activating samples (GPU)
    print("\n[Step 2] Finding top-activating samples per feature (GPU)...")
    top_samples = find_top_samples.remote(features)

    n_with_samples = sum(1 for v in top_samples.values() if v)
    print(f"\n  Found samples for {n_with_samples}/{len(top_samples)} features")
    print(f"  Results saved to volume: analysis/autointerp/top_samples.json")
    print(f"\n  To run interpretation locally:")
    print(f"    1. Download top_samples.json from Modal volume")
    print(f"    2. export OPENAI_API_KEY=sk-...")
    print(f"    3. python local_autointerp.py --top-samples top_samples.json")
