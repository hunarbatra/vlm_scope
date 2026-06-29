#!/usr/bin/env python3
"""
Extract SAE acts under 'ocr' prompt for the WINNING candidates from firing scan:
features that fire under ocr AND have ablation impact.
"""
import os, sys, json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

MIX_MODEL = "google/paligemma2-3b-mix-448"
SAE_ROOT  = Path("/data1/vlm_scope_sae_mix448_textonly")
OUT_DIR   = SAE_ROOT / "analysis_ocr/sae_acts_ocrprompt"
CKPT_DIR  = SAE_ROOT / "checkpoints"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# WINNERS: fire under ocr AND have ablation impact
FEATURES = [
    {"layer": 21, "feature": 13072},  # 44% firing, -8.3% abl  ← BEST
    {"layer": 19, "feature": 9893},   # 38% firing, -3.1% abl
    {"layer": 21, "feature": 677},    # 22% firing, -1.7% abl
    {"layer": 19, "feature": 8866},   # 58% firing, -0.4% abl  ← high firing
    {"layer": 17, "feature": 9368},   # 58% firing (no abl data)
    {"layer": 17, "feature": 12336},  # 58% firing (no abl data)
    {"layer": 21, "feature": 10675},  # 57% firing high act
    {"layer": 19, "feature": 89},     # 57% firing
]


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions, initialize_jumprelu_sae

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")
    print("[INFO] Loading mix-448...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()

    for feat in FEATURES:
        layer, feature = feat["layer"], feat["feature"]
        key = f"L{layer}_F{feature}"
        out_path = OUT_DIR / f"acts_{key}.json"
        if out_path.exists():
            print(f"[SKIP] {key}", flush=True); continue

        print(f"\n[EXTRACT/winner] {key} across {len(ds)} samples (ocr prompt)...", flush=True)
        ckpt = CKPT_DIR / f"text-only_layer_{layer}.pt"
        sae = initialize_jumprelu_sae(layer, checkpoint_path=str(ckpt), device=device)
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
            for si in range(len(ds)):
                ex = ds[si]; img = ex.get("image")
                if img is None: acts[si] = 0.0; continue
                try:
                    img = img.convert("RGB")
                    iids, attn, pv = process_vlm_inputs(img, "ocr", proc, model, device=device)
                    _, img_end = get_image_token_positions(iids)
                    captured.clear()
                    with torch.inference_mode():
                        model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                       max_new_tokens=1, do_sample=False, use_cache=False)
                    if "h" not in captured: acts[si] = 0.0; continue
                    h_text = captured["h"][0, img_end:, :].float()
                    feat_acts = sae.encode(h_text)[:, feature]
                    acts[si] = feat_acts.mean().item()
                except Exception:
                    acts[si] = 0.0
                if (si + 1) % 250 == 0:
                    n = sum(1 for v in acts.values() if v > 0)
                    print(f"  {si+1}/{len(ds)}  firing={n}", flush=True)
        finally:
            handle.remove(); del sae; torch.cuda.empty_cache()

        n_fire = sum(1 for v in acts.values() if v > 0)
        print(f"  {key}: firing={n_fire}/{len(ds)}", flush=True)
        out = {"layer": layer, "feature": feature,
               "n_firing_total": n_fire,
               "acts": {str(si): v for si, v in acts.items()}}
        with open(out_path, "w") as f: json.dump(out, f)
        print(f"  Saved → {out_path}", flush=True)

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
