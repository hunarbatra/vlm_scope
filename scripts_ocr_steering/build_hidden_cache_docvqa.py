#!/usr/bin/env python3
"""
Build per-sample mix-448 mean-over-text-token hidden state cache + correctness labels for DocVQA.

Single pass per sample: hooks capture mean-over-text-tokens hidden states at all 26 layers
during the prefill of model.generate(); the generated response gives the correctness label.

Output per sample (train + test indices from splits.json):
  analysis_docvqa/mix_hidden_cache/vi_{si:05d}.pt
    → {layer_int: tensor([2304], bfloat16), "correct": bool}

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 build_hidden_cache_docvqa.py
"""
import os, sys, json, warnings
from pathlib import Path

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL   = "google/paligemma2-3b-mix-448"
SAE_ROOT    = Path("/data1/vlm_scope_sae_mix448_textonly")
SPLITS_JSON = SAE_ROOT / "analysis_docvqa/splits.json"
OUT_DIR     = SAE_ROOT / "analysis_docvqa/mix_hidden_cache"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

LAYERS         = list(range(26))
MAX_NEW_TOKENS = 64


def _correct_docvqa(resp, gt_list):
    if resp is None: return False
    if isinstance(gt_list, str): gt_list = [gt_list]
    r = resp.strip().lower()
    for gt in gt_list:
        g = str(gt).strip().lower()
        if g and (g in r or r in g): return True
    return False


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading DocVQA...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    splits = json.load(open(SPLITS_JSON))
    all_indices = sorted(set(splits["train"]) | set(splits["test"]))
    print(f"  {len(all_indices)} samples (train={len(splits['train'])}, test={len(splits['test'])})", flush=True)

    print("[INFO] Loading mix-448...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok   = proc.tokenizer
    print(f"  mix-448 loaded on {device}", flush=True)

    hiddens = {}
    handles = []
    for l in LAYERS:
        def make_hook(lid):
            def _hook(mod, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                if x.shape[1] > 1:
                    hiddens[lid] = x.detach()
            return _hook
        handles.append(model.model.language_model.layers[l].register_forward_hook(make_hook(l)))

    done = skipped = errors = 0
    try:
        for i, si in enumerate(all_indices):
            out_path = OUT_DIR / f"vi_{si:05d}.pt"
            if out_path.exists():
                skipped += 1
                continue

            ex = ds[si]
            img = ex.get("image")
            q   = str(ex.get("question", "")).strip()
            gt  = ex.get("answers") or []
            if isinstance(gt, str): gt = [gt]
            if img is None or not q:
                errors += 1
                continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(
                    img, f"answer en {q}", proc, model, device=device)
                _, img_end = get_image_token_positions(iids)

                hiddens.clear()
                with torch.inference_mode():
                    out_ids = model.generate(
                        input_ids=iids, attention_mask=attn, pixel_values=pv,
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)

                if len(hiddens) != len(LAYERS):
                    errors += 1
                    continue

                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                correct = _correct_docvqa(resp, gt)

                h_dict = {l: hiddens[l][0, img_end:, :].mean(0).to(torch.bfloat16).cpu()
                          for l in LAYERS}
                h_dict["correct"] = correct
                torch.save(h_dict, out_path)
                done += 1
            except Exception:
                errors += 1
                continue

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(all_indices)}  done={done}  skip={skipped}  err={errors}", flush=True)
    finally:
        for h in handles:
            h.remove()

    print(f"[DONE] done={done}  skipped={skipped}  errors={errors}", flush=True)


if __name__ == "__main__":
    main()
