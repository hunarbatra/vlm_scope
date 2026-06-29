#!/usr/bin/env python3
"""
MathVerse Step 4b: Extract SAE activations for positive-drop (suppress) features.
These are features where ablation IMPROVES accuracy → suppress with negative alpha.
Targets: L17_F12318 (drop=+3.70%), L19_F12800 (drop=+2.47%), L1_F11461 (drop=+2.47%)

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 -u mathverse_step4b_suppress_acts.py
"""
import os, sys, json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL = "google/paligemma2-3b-mix-448"
SAE_ROOT  = Path("/data1/vlm_scope_sae_mix448_textonly")
CKPT_DIR  = SAE_ROOT / "checkpoints"
OUT_DIR   = SAE_ROOT / "analysis_mathverse/sae_acts"
TRAIN_END = 344

# Features where ablating IMPROVES accuracy → suppress with negative steering
SUPPRESS_FEATURES = [
    {"layer": 17, "feature": 12318, "key": "L17_F12318"},  # drop=+3.70%
    {"layer": 19, "feature": 12800, "key": "L19_F12800"},  # drop=+2.47%
    {"layer": 1,  "feature": 11461, "key": "L1_F11461"},   # drop=+2.47%
]

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions, initialize_jumprelu_sae

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading MathVerse...", flush=True)
    ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")
    N  = len(ds)

    print("[INFO] Loading mix-448...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()

    for feat in SUPPRESS_FEATURES:
        layer, feature, key = feat["layer"], feat["feature"], feat["key"]
        out_path = OUT_DIR / f"acts_{key}.json"
        if out_path.exists():
            print(f"  [SKIP] {key}", flush=True); continue

        print(f"\n[EXTRACT] {key} across {N} samples...", flush=True)
        ckpt = CKPT_DIR / f"text-only_layer_{layer}.pt"
        sae  = initialize_jumprelu_sae(layer, checkpoint_path=str(ckpt), device=device)
        sae.eval()

        captured = {}
        def make_hook(lid):
            def _hook(mod, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                if x.shape[1] > 1: captured["h"] = x.detach()
            return _hook
        handle = model.model.language_model.layers[layer].register_forward_hook(make_hook(layer))

        acts = {}
        try:
            for si in range(N):
                ex = ds[si]; img = ex.get("image")
                if img is None: acts[si] = 0.0; continue
                try:
                    img = img.convert("RGB")
                    iids, attn, pv = process_vlm_inputs(
                        img, f"answer en {ex['prompt']}", proc, model, device=device)
                    _, img_end = get_image_token_positions(iids)
                    captured.clear()
                    with torch.inference_mode():
                        model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                       max_new_tokens=1, do_sample=False, use_cache=False)
                    if "h" not in captured: acts[si] = 0.0; continue
                    h_text = captured["h"][0, img_end:, :].float()
                    acts[si] = sae.encode(h_text)[:, feature].mean().item()
                except Exception: acts[si] = 0.0
                if (si + 1) % 100 == 0:
                    n_fire = sum(1 for v in acts.values() if v > 0)
                    print(f"  {si+1}/{N}  firing={n_fire}", flush=True)
        finally:
            handle.remove(); del sae; torch.cuda.empty_cache()

        n_tr = sum(1 for si, v in acts.items() if si < TRAIN_END and v > 0)
        n_te = sum(1 for si, v in acts.items() if si >= TRAIN_END and v > 0)
        print(f"  {key}: train_firing={n_tr}/{TRAIN_END}  test_firing={n_te}/{N-TRAIN_END}", flush=True)
        with open(out_path, "w") as f:
            json.dump({"layer": layer, "feature": feature, "train_end": TRAIN_END,
                       "n_firing_train": n_tr, "n_firing_test": n_te,
                       "acts": {str(si): v for si, v in acts.items()}}, f)
        print(f"  Saved → {out_path}", flush=True)

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
