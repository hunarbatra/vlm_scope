#!/usr/bin/env python3
"""
Build last-token mix-448 hidden cache for CAA.

Captures h[0, -1, :] at each target layer — the "Answer:" colon position,
which is the decision point before the model predicts Yes/No. This matches
Rimsky et al. 2023 last-prompt-token extraction convention (adapted to our
prompt format).

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_lasttoken/vi_{vi:05d}.pt
        dict {layer: tensor[2304] bfloat16}

Layers cached: {4, 6, 9, 11, 12, 13, 14, 15}
  - 13 is MIDDLE
  - 4, 6, 9, 11, 12, 14, 15 are the per-feature spatial layers

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -B build_mix_hidden_lasttoken.py
"""
import os, sys, json, hashlib, warnings
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

MIX_MODEL   = "google/paligemma2-3b-mix-448"
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_lasttoken")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

LAYERS = [4, 6, 9, 11, 12, 13, 14, 15]


def _build_vsr_prompt(statement):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\nAnswer:"
    )

def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h  = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {MIX_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    print(f"[INFO] VSR total: {N}  Target layers: {LAYERS}", flush=True)

    n_done = n_skipped = n_failed = 0
    for vi in range(N):
        out_path = OUT_DIR / f"vi_{vi:05d}.pt"
        if out_path.exists():
            n_skipped += 1
            continue

        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            n_failed += 1; continue
        prompt = _build_vsr_prompt(str(ex.get("caption", "")))

        captures = {}
        hooks = []
        def make_hook(l):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captures[l] = h[0, -1, :].detach().to(torch.bfloat16).cpu()
            return hook_fn
        for l in LAYERS:
            hooks.append(
                model.model.language_model.layers[l].register_forward_hook(make_hook(l))
            )

        try:
            iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
            with torch.no_grad():
                model(input_ids=iids, attention_mask=attn, pixel_values=pv)
            h_dict = {l: captures[l] for l in LAYERS if l in captures}
            if len(h_dict) == len(LAYERS):
                torch.save(h_dict, out_path)
                n_done += 1
            else:
                n_failed += 1
        except Exception as e:
            n_failed += 1
            if n_failed < 5:
                print(f"  [WARN] vi={vi}: {e}", flush=True)
        finally:
            for hh in hooks:
                try: hh.remove()
                except Exception: pass
            captures.clear()

        if (vi + 1) % 500 == 0:
            print(f"  {vi+1}/{N}  done={n_done} skip={n_skipped} fail={n_failed}", flush=True)

    print(f"[DONE] total={N}  done={n_done}  skipped={n_skipped}  failed={n_failed}", flush=True)


if __name__ == "__main__":
    main()
