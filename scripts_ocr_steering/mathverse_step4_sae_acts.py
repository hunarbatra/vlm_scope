#!/usr/bin/env python3
"""
MathVerse Step 7: Extract SAE activations for top-5 steering features → R(F) subsets.

Reads TOP_FEATURES from ablation_results.json (biggest performance drop),
extracts per-sample mean SAE activation for each.

Output per feature:
  analysis_mathverse/sae_acts/acts_L{L}_F{F}.json
    {"layer": L, "feature": F, "acts": {str(si): float}, "train_end": 344, ...}

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -u mathverse_step4_sae_acts.py
    (Run after mathverse_step3_ablation.py has completed.)
"""
import os, sys, json
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL  = "google/paligemma2-3b-mix-448"
SAE_ROOT   = Path("/data1/vlm_scope_sae_mix448_textonly")
ABL_PATH   = SAE_ROOT / "analysis_mathverse/ablation_results.json"
CKPT_DIR   = SAE_ROOT / "checkpoints"
OUT_DIR    = SAE_ROOT / "analysis_mathverse/sae_acts"

TRAIN_END  = 344
TOP_N      = 8   # take 8 features with biggest ablation drop (more candidates in case drops are small)

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

    abl = json.load(open(ABL_PATH))
    drops = [(k, v) for k, v in abl.items() if k != "base" and "drop" in v]
    drops.sort(key=lambda x: x[1]["drop"])
    # Take both ends: most-negative (amplify) and most-positive (suppress)
    n_each = TOP_N // 2
    bottom = drops[:n_each]
    top    = drops[-n_each:]
    seen, combined = set(), []
    for k, v in (bottom + top):
        if k not in seen:
            seen.add(k); combined.append((k, v))
    top_feats = [{"layer": v["layer"], "feature": v["feature"], "key": k}
                 for k, v in combined]

    print(f"[INFO] Steering targets (both ends of ablation):")
    for f in top_feats:
        r = abl[f["key"]]
        print(f"  {f['key']}: drop={r['drop']:+.2f}%  fisher={r['fisher_score']:.2f}", flush=True)

    print("[INFO] Loading MathVerse...", flush=True)
    ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")
    N  = len(ds)
    all_idx = list(range(N))

    print("[INFO] Loading mix-448...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()

    for feat in top_feats:
        layer, feature, key = feat["layer"], feat["feature"], feat["key"]
        out_path = OUT_DIR / f"acts_{key}.json"
        if out_path.exists():
            print(f"  [SKIP] {key}", flush=True)
            continue

        print(f"\n[EXTRACT] {key} across {N} samples...", flush=True)
        ckpt = CKPT_DIR / f"text-only_layer_{layer}.pt"
        sae  = initialize_jumprelu_sae(layer, checkpoint_path=str(ckpt), device=device)
        sae.eval()

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
            for si in all_idx:
                ex = ds[si]
                img = ex.get("image")
                if img is None:
                    acts[si] = 0.0; continue
                try:
                    img = img.convert("RGB")
                    iids, attn, pv = process_vlm_inputs(
                        img, f"answer en {ex['prompt']}", proc, model, device=device)
                    _, img_end = get_image_token_positions(iids)
                    captured.clear()
                    with torch.inference_mode():
                        model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                       max_new_tokens=1, do_sample=False, use_cache=False)
                    if "h" not in captured:
                        acts[si] = 0.0; continue
                    h_text = captured["h"][0, img_end:, :].float()
                    feat_acts = sae.encode(h_text)[:, feature]
                    acts[si] = feat_acts.mean().item()
                except Exception:
                    acts[si] = 0.0

                if (si + 1) % 100 == 0:
                    n_fire = sum(1 for v in acts.values() if v > 0)
                    print(f"  {si+1}/{N}  firing={n_fire}", flush=True)
        finally:
            handle.remove()
            del sae; torch.cuda.empty_cache()

        train_idx = [si for si in all_idx if si < TRAIN_END]
        test_idx  = [si for si in all_idx if si >= TRAIN_END]
        n_fire_tr = sum(1 for si in train_idx if acts.get(si, 0) > 0)
        n_fire_te = sum(1 for si in test_idx  if acts.get(si, 0) > 0)
        print(f"  {key}: train_firing={n_fire_tr}/{len(train_idx)}  test_firing={n_fire_te}/{len(test_idx)}", flush=True)

        out = {
            "layer": layer, "feature": feature, "train_end": TRAIN_END,
            "n_firing_train": n_fire_tr, "n_firing_test": n_fire_te,
            "acts": {str(si): v for si, v in acts.items()},
        }
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"  Saved → {out_path}", flush=True)

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
