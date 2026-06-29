#!/usr/bin/env python3
"""
Extract mix-448 SAE activations for DocVQA features → R(F) subsets.

For each (layer, feature) in FEATURES, runs mix-448 + SAE on ALL DocVQA samples,
records mean-over-text-tokens activation of feature F per sample.
R(F) = samples where activation > 0 (JumpReLU threshold already baked in via SAE encode).

Output per feature:
  analysis_docvqa/sae_acts/acts_L{L}_F{F}.json
  {
    "layer": L, "feature": F,
    "acts": {"si": activation_float, ...},   # all samples
    "train_end": 4279,
    "n_firing_train": int, "n_firing_test": int
  }

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 extract_sae_acts_docvqa.py
"""
import os, sys, json, warnings
from pathlib import Path

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL   = "google/paligemma2-3b-mix-448"
SAE_ROOT    = Path("/data1/vlm_scope_sae_mix448_textonly")
SPLITS_JSON = SAE_ROOT / "analysis_docvqa/splits.json"
OUT_DIR     = SAE_ROOT / "analysis_docvqa/sae_acts"
CKPT_DIR    = SAE_ROOT / "checkpoints"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# DocVQA-specific features from MMDiff pipeline (top300_docvqa_features.csv, layers 15-21)
FEATURES = [
    {"layer": 16, "feature": 2825},
    {"layer": 21, "feature": 12658},
    {"layer": 19, "feature": 6922},
    {"layer": 18, "feature": 8613},
    {"layer": 17, "feature": 3672},
]


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions, initialize_jumprelu_sae

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading DocVQA...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    splits = json.load(open(SPLITS_JSON))
    train_indices = set(splits["train"])   # 0..4278
    test_indices  = set(splits["test"])    # 4279..5348
    all_indices   = sorted(train_indices | test_indices)
    TRAIN_END     = max(train_indices) + 1
    print(f"  {len(all_indices)} samples (train≤{TRAIN_END-1}, test={min(test_indices)}..{max(test_indices)})", flush=True)

    print("[INFO] Loading mix-448...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    print("  mix-448 loaded", flush=True)

    for feat in FEATURES:
        layer, feature = feat["layer"], feat["feature"]
        key = f"L{layer}_F{feature}"
        out_path = OUT_DIR / f"acts_{key}.json"
        if out_path.exists():
            print(f"[SKIP] {key}", flush=True)
            continue

        print(f"\n[EXTRACT] {key} across {len(all_indices)} samples...", flush=True)

        # Load SAE for this layer
        ckpt = CKPT_DIR / f"text-only_layer_{layer}.pt"
        sae = initialize_jumprelu_sae(layer, checkpoint_path=str(ckpt), device=device)
        sae.eval()

        # Hook hidden state at this layer
        captured = {}
        def make_hook(lid):
            def _hook(mod, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                if x.shape[1] > 1:
                    captured["h"] = x.detach()
            return _hook
        handle = model.model.language_model.layers[layer].register_forward_hook(make_hook(layer))

        acts = {}
        try:
            for i, si in enumerate(all_indices):
                ex = ds[si]
                img = ex.get("image")
                q   = str(ex.get("question", "")).strip()
                if img is None or not q:
                    acts[si] = 0.0
                    continue
                try:
                    img = img.convert("RGB")
                    iids, attn, pv = process_vlm_inputs(
                        img, f"answer en {q}", proc, model, device=device)
                    _, img_end = get_image_token_positions(iids)

                    captured.clear()
                    with torch.inference_mode():
                        model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                       max_new_tokens=1, do_sample=False, use_cache=False)

                    if "h" not in captured:
                        acts[si] = 0.0
                        continue

                    h_text = captured["h"][0, img_end:, :].float()
                    feat_acts = sae.encode(h_text)[:, feature]   # [n_text_tokens]
                    acts[si] = feat_acts.mean().item()
                except Exception:
                    acts[si] = 0.0

                if (i + 1) % 500 == 0:
                    n_fire = sum(1 for v in acts.values() if v > 0)
                    print(f"  {i+1}/{len(all_indices)}  firing={n_fire}", flush=True)
        finally:
            handle.remove()
            del sae
            torch.cuda.empty_cache()

        n_fire_train = sum(1 for si, v in acts.items() if si in train_indices and v > 0)
        n_fire_test  = sum(1 for si, v in acts.items() if si in test_indices  and v > 0)
        print(f"  {key}: train_firing={n_fire_train}/{len(train_indices)}  test_firing={n_fire_test}/{len(test_indices)}", flush=True)

        out = {
            "layer": layer, "feature": feature,
            "train_end": TRAIN_END,
            "n_firing_train": n_fire_train,
            "n_firing_test":  n_fire_test,
            "acts": {str(si): v for si, v in acts.items()},
        }
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"  Saved → {out_path}", flush=True)

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
