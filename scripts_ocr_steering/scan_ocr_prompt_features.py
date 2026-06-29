#!/usr/bin/env python3
"""
Scan firing rate of MANY SAE features under the 'ocr' prompt to find features
that activate during ocr-mode inference.

For each layer in TARGET_LAYERS, encodes all 16384 SAE features on a stride-sample
of OCR-Bench under "ocr" prompt. Saves per-layer firing counts.

Output: analysis_ocr/firing_ocr_prompt/firing_L{L}.json

Usage:
  CUDA_VISIBLE_DEVICES=X python3 -B scripts/scan_ocr_prompt_features.py --layer L
"""
import os, sys, json, argparse, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

MIX_MODEL = "google/paligemma2-3b-mix-448"
SAE_ROOT  = Path("/data1/vlm_scope_sae_mix448_textonly")
OUT_DIR   = SAE_ROOT / "analysis_ocr/firing_ocr_prompt"
CKPT_DIR  = SAE_ROOT / "checkpoints"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Subsample: 200 samples evenly across 1000 to find which features fire
N_SCAN = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    layer = args.layer

    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions, initialize_jumprelu_sae

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"firing_L{layer}.json"
    if out_path.exists():
        print(f"[SKIP] L{layer} firing already scanned"); return

    print(f"[INFO] Loading OCR-Bench, scanning {N_SCAN}/1000 samples", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")
    indices = list(range(0, len(ds), max(1, len(ds) // N_SCAN)))[:N_SCAN]

    print(f"[INFO] Loading mix-448 + SAE L{layer}...", flush=True)
    proc = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    sae = initialize_jumprelu_sae(layer,
        checkpoint_path=str(CKPT_DIR / f"text-only_layer_{layer}.pt"),
        device=device); sae.eval()

    captured = {}
    def make_hook(lid):
        def _hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if x.shape[1] > 1: captured["h"] = x.detach()
        return _hook
    handle = model.model.language_model.layers[layer].register_forward_hook(make_hook(layer))

    # Track: count how many samples each feature fires on (>0 mean)
    n_features = sae.W_dec.shape[0] if hasattr(sae, "W_dec") else 16384
    fire_count = torch.zeros(n_features, dtype=torch.int32)
    activations_sum = torch.zeros(n_features, dtype=torch.float32)

    n_processed = 0
    try:
        for si in indices:
            ex = ds[si]
            img = ex.get("image")
            if img is None: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(img, "ocr", proc, model, device=device)
                _, img_end = get_image_token_positions(iids)
                captured.clear()
                with torch.inference_mode():
                    model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                   max_new_tokens=1, do_sample=False, use_cache=False)
                if "h" not in captured: continue
                h_text = captured["h"][0, img_end:, :].float()
                feat_acts = sae.encode(h_text).mean(dim=0)  # [n_features]
                fired = (feat_acts > 0).cpu()
                fire_count += fired.to(torch.int32)
                activations_sum += feat_acts.cpu()
                n_processed += 1
            except Exception:
                continue
            if (n_processed % 50) == 0:
                print(f"  L{layer}: {n_processed}/{len(indices)}", flush=True)
    finally:
        handle.remove(); del sae; torch.cuda.empty_cache()

    print(f"[L{layer}] processed {n_processed} samples", flush=True)

    # Find features with moderate firing rate (5%-50% of samples)
    fire_rate = fire_count.float() / max(n_processed, 1)
    selective = ((fire_rate > 0.05) & (fire_rate < 0.5)).nonzero(as_tuple=True)[0].tolist()
    print(f"[L{layer}] features with 5-50% firing rate under ocr: {len(selective)}", flush=True)

    # Top by firing count among selective features
    if selective:
        feat_data = [(int(i), int(fire_count[i].item()), float(activations_sum[i].item())) for i in selective]
        feat_data.sort(key=lambda x: -x[1])
        print(f"  top-10:")
        for f, n, s in feat_data[:10]:
            print(f"    L{layer}_F{f}: fires {n}/{n_processed} ({n/n_processed*100:.1f}%), sum_act={s:.1f}", flush=True)

    out = {
        "layer": layer,
        "n_processed": n_processed,
        "indices_scanned": indices[:n_processed],
        "fire_count": fire_count.tolist(),
        "fire_rate": fire_rate.tolist(),
        "activations_sum": activations_sum.tolist(),
    }
    with open(out_path, "w") as f: json.dump(out, f)
    print(f"[L{layer}] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
