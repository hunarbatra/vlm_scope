#!/usr/bin/env python3
"""
Build correct/incorrect CAA vectors for a dataset using mix-448.

Single pass per sample: hooks capture mean-over-text-tokens hidden states at all
target layers during the prefill of model.generate(). The generated response is
then used to determine correct/incorrect label. No separate forward pass needed.

Extraction protocol matches the working VSR implementation (pt448_hidden_delta.py):
  v[L] = h[0, img_end:, :].mean(0)  — mean over all text token positions after image.

Usage:
  # DocVQA vectors (train split val[0:4278])
  CUDA_VISIBLE_DEVICES=0 python3 build_caa_vectors.py --dataset docvqa --layers 13 15 17 19 21 --gpu 0

  # OCR-Bench vectors (full test set)
  CUDA_VISIBLE_DEVICES=1 python3 build_caa_vectors.py --dataset ocr --layers 13 15 17 19 20 21 --gpu 0
"""
import os, sys, json, warnings, argparse
from pathlib import Path

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL   = "google/paligemma2-3b-mix-448"
SAE_ROOT    = Path("/data1/vlm_scope_sae_mix448_textonly")
SPLITS_JSON = SAE_ROOT / "analysis_docvqa/splits.json"
VEC_DIR     = SAE_ROOT / "analysis/caa_vectors"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MAX_NEW_TOKENS = 64


def _correct_docvqa(resp, gt_list):
    if resp is None: return False
    if isinstance(gt_list, str): gt_list = [gt_list]
    r = resp.strip().lower()
    for gt in gt_list:
        g = str(gt).strip().lower()
        if g and (g in r or r in g): return True
    return False


def _correct_ocr(resp, gt):
    if resp is None: return False
    r = resp.strip().lower()
    g = str(gt).strip().lower()
    return g in r or r in g


def build_vectors(model, processor, tokenizer, samples, layers, device, tag, correct_fn):
    """Single pass per sample: capture hidden states during generate() prefill.

    generate() runs a full prefill forward pass internally before decoding.
    Hooks installed before generate() fire during that prefill, capturing
    mean-over-text-token hiddens at all target layers simultaneously.
    One call per sample — no separate model() forward pass.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    VEC_DIR.mkdir(parents=True, exist_ok=True)

    needed = [l for l in layers if not (VEC_DIR / f"{tag}_caa_L{l}.pt").exists()]
    if not needed:
        print(f"[{tag}] All vectors already cached.", flush=True)
        return

    print(f"[{tag}] Building vectors at layers {needed} over {len(samples)} samples", flush=True)

    # Hooks fire during generate()'s prefill pass; we store the full activation
    # and extract mean-over-text-tokens after the call using img_end
    hiddens = {l: None for l in needed}
    handles = []
    for l in needed:
        def make_hook(layer_id):
            def _hook(mod, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                # Only capture during prefill (sequence length > 1)
                if x.shape[1] > 1:
                    hiddens[layer_id] = x.detach().cpu()
                return out
            return _hook
        handles.append(
            model.model.language_model.layers[l].register_forward_hook(make_hook(l))
        )

    pos = {l: [] for l in needed}
    neg = {l: [] for l in needed}

    try:
        for i, sample in enumerate(samples):
            try:
                img = sample.get("image")
                q   = str(sample.get("question", "")).strip()
                gt  = sample.get("answers") or sample.get("answer") or []
                if isinstance(gt, str): gt = [gt]
                if img is None or not q or not gt: continue
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(
                    img, f"answer en {q}", processor, model, device=device)

                _, img_end = get_image_token_positions(iids)

                # Single call: hooks fire during prefill, output gives response
                with torch.inference_mode():
                    out_ids = model.generate(
                        input_ids=iids, attention_mask=attn, pixel_values=pv,
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)

                # Extract mean-over-text-tokens from prefill capture
                if any(hiddens[l] is None for l in needed):
                    continue
                snap = {l: hiddens[l][0, img_end:, :].float().mean(0) for l in needed}

                resp = tokenizer.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                ok = correct_fn(resp, gt)
                for l in needed:
                    (pos[l] if ok else neg[l]).append(snap[l])

            except Exception:
                continue

            if (i + 1) % 200 == 0:
                p0 = len(pos[needed[0]]); n0 = len(neg[needed[0]])
                print(f"  [{tag}] {i+1}/{len(samples)} pos={p0} neg={n0}", flush=True)

    finally:
        for h in handles: h.remove()

    for l in needed:
        p = torch.stack(pos[l]) if pos[l] else torch.zeros(0, 2304)
        n = torch.stack(neg[l]) if neg[l] else torch.zeros(0, 2304)
        if p.shape[0] == 0 or n.shape[0] == 0:
            print(f"  [{tag}] L{l}: empty split, skipping", flush=True)
            continue
        v = p.mean(0) - n.mean(0)
        v_unit = v / v.norm().clamp(min=1e-8)
        torch.save({"v": v, "v_unit": v_unit, "n_pos": p.shape[0], "n_neg": n.shape[0], "layer": l},
                   VEC_DIR / f"{tag}_caa_L{l}.pt")
        print(f"  [{tag}] L{l}: ||v||={v.norm():.3f} n+={p.shape[0]} n-={n.shape[0]} → saved", flush=True)


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["docvqa", "ocr"], required=True)
    ap.add_argument("--layers",  type=int, nargs="+", required=True)
    ap.add_argument("--gpu",     type=str, default="0")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"

    print(f"[INFO] Loading mix-448...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok   = proc.tokenizer
    print(f"  mix-448 loaded on {device}", flush=True)

    if args.dataset == "docvqa":
        print("[INFO] Loading DocVQA train split...", flush=True)
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        splits = json.load(open(SPLITS_JSON))
        samples = [ds[i] for i in splits["train"]]
        print(f"  {len(samples)} samples (val[0:4278])", flush=True)
        build_vectors(model, proc, tok, samples, args.layers, device, "docvqa", _correct_docvqa)

    else:  # ocr
        print("[INFO] Loading OCR-Bench...", flush=True)
        ds = load_dataset("echo840/OCRBench", split="test")
        samples = [{"image": ex["image"], "question": ex["question"],
                    "answers": [str(ex["answer"])]} for ex in ds]
        print(f"  {len(samples)} samples", flush=True)
        build_vectors(model, proc, tok, samples, args.layers, device, "ocr", _correct_ocr)

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
