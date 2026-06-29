#!/usr/bin/env python3
"""
Step 1 of two-step transfer: extract mix-448 SAE activations to disk.

For each top-10 feature (layer_L, feature_F) and each VSR example in the
relation subset, runs mix-448 forward, encodes hidden state through mix-448 SAE,
and records the per-example mean activation of feature F over text tokens.

Saves: /data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts/acts_L{L}_F{F}.json
  {vi: act_F_mean, ...}  for each example index vi in the subset.

This runs mix-448 on GPU 2 ONLY — no pt-448, no interference.
The companion script pt448_precomputed_transfer.py reads these and runs pt-448 on GPU 4.

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 extract_mix_sae_acts.py
"""

import os, sys, json, hashlib, warnings
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MIX_MODEL      = "google/paligemma2-3b-mix-448"
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TOP10 = [
    (9,  387,   ["at the right side of"]),
    (14, 10561, ["close to"]),
    (11, 12278, ["touching"]),
    (9,  7540,  ["consists of"]),
    (4,  14233, ["ahead of"]),
    (6,  7539,  ["left of", "right of"]),
    (11, 9639,  ["in", "inside", "on"]),
    (13, 15219, ["behind"]),
    (15, 220,   ["across from", "at the left side of", "at the right side of", "right of"]),
    (12, 2257,  ["facing"]),
]


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB"); img.save(cp, "JPEG")
        return img
    except Exception: return None


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {MIX_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    nns = NNsight(model)
    model_dtype = next(model.parameters()).dtype

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    for layer_idx, feature_idx, relations in TOP10:
        key = f"L{layer_idx}_F{feature_idx}"
        out_path = OUT_DIR / f"acts_{key}.json"
        if out_path.exists():
            print(f"[SKIP] {key}", flush=True); continue

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue

        print(f"[EXTRACT] {key} {relations} N={len(indices)}...", flush=True)

        # Load SAE
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()

        acts = {}
        pos_acts, neg_acts = [], []
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: acts[vi] = 0.0; continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, nns._module, device=device)
                _, img_end = get_image_token_positions(iids)
                with nns.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    h = nns.model.language_model.layers[layer_idx].output[0][0, img_end:].save()
                h_val = h.detach().float()
                with torch.no_grad():
                    feat_acts = sae.encode(h_val)[:, feature_idx]
                    a = feat_acts.mean().item()
                acts[vi] = a
                (pos_acts if label == 1 else neg_acts).append(a)
            except Exception: acts[vi] = 0.0

        del sae; torch.cuda.empty_cache()

        pos_mean = sum(pos_acts) / max(len(pos_acts), 1)
        neg_mean = sum(neg_acts) / max(len(neg_acts), 1)
        print(f"[DONE] {key}: n={len(acts)} pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} "
              f"contrast={pos_mean-neg_mean:+.4f}", flush=True)

        meta = {
            "layer": layer_idx, "feature": feature_idx, "relations": relations,
            "n_samples": len(indices), "n_pos": len(pos_acts), "n_neg": len(neg_acts),
            "pos_mean": pos_mean, "neg_mean": neg_mean, "contrast": pos_mean - neg_mean,
            "acts": {str(vi): v for vi, v in acts.items()}
        }
        with open(out_path, "w") as f: json.dump(meta, f)
        print(f"[SAVED] {out_path}", flush=True)

    print("\n[DONE] All features extracted.", flush=True)


if __name__ == "__main__":
    main()
