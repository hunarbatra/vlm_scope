#!/usr/bin/env python3
"""Diagnose pt-448 failures on R(L17_F13602) ∩ all-1000 OCR-Bench."""
import os, sys, json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL  = "google/paligemma2-3b-pt-448"
SAE_ROOT  = Path("/data1/vlm_scope_sae_mix448_textonly")
PAIR_CACHE = SAE_ROOT / "analysis_ocr/paired_contrast_cache"
SAE_ACTS  = SAE_ROOT / "analysis_ocr/sae_acts"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def _correct(resp, gt):
    if not resp: return False
    r = resp.strip().lower(); g = str(gt).strip().lower()
    return bool(g) and (g in r or r in g)


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    feature_key = "L17_F13602"
    device = "cuda:0"   # CUDA_VISIBLE_DEVICES restricts to one GPU; ordinal is 0

    ds = load_dataset("echo840/OCRBench", split="test")

    # R(F) ∩ all-1000
    ad = json.load(open(SAE_ACTS / f"acts_{feature_key}.json"))
    rF = sorted([int(k) for k, v in ad["acts"].items() if v > 0])
    print(f"# R({feature_key}) over all 1000: n={len(rF)}", flush=True)

    # Load pt-448
    print(f"# Loading {PT_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl  = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok  = proc.tokenizer

    # Per-category breakdown setup
    cat_counts  = {}
    cat_correct = {}
    print()
    print(f"{'idx':>4} {'cat':<20} {'GT':<32}  {'pt':<32}  {'mix':<32}  {'pt_ok':>5} {'mix_ok':>6}")
    print("-" * 150)

    n_pt_ok = n_mix_ok = 0
    for si in rF:
        ex = ds[si]
        img = ex.get("image"); q = str(ex.get("question","")).strip()
        gt  = _parse_gt(ex.get("answer"))
        cat = str(ex.get("question_type", "?"))[:20]
        if img is None or not q or not gt: continue

        # pt-448 response
        pt_resp = ""
        try:
            img = img.convert("RGB")
            iids, attn, pv = process_vlm_inputs(
                img, f"answer en {q}", proc, mdl, device=device)
            with torch.no_grad():
                out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                       pixel_values=pv, max_new_tokens=64,
                                       do_sample=False, use_cache=True)
            pt_resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            pt_resp = f"<ERR:{type(e).__name__}>"

        # mix-448 response from cache
        mix_resp = "?"
        cache_path = PAIR_CACHE / f"vi_{si:05d}.pt"
        if cache_path.exists():
            try:
                d = torch.load(cache_path, map_location="cpu", weights_only=False)
                mix_resp = d.get("resp", "?")
            except Exception: pass

        pt_ok  = _correct(pt_resp, gt)
        mix_ok = _correct(mix_resp, gt)
        n_pt_ok  += pt_ok
        n_mix_ok += mix_ok
        cat_counts[cat]  = cat_counts.get(cat, 0) + 1
        cat_correct.setdefault(cat, [0, 0])
        cat_correct[cat][0] += pt_ok
        cat_correct[cat][1] += mix_ok

        gt_s  = (gt[:30]   + '..') if len(gt)   > 30 else gt
        pt_s  = (pt_resp[:30]  + '..') if len(pt_resp)  > 30 else pt_resp
        mix_s = (mix_resp[:30] + '..') if len(mix_resp) > 30 else mix_resp
        print(f"{si:>4} {cat:<20} {gt_s:<32}  {pt_s:<32}  {mix_s:<32}  "
              f"{'✓' if pt_ok else '✗':>5} {'✓' if mix_ok else '✗':>6}", flush=True)

    print()
    print(f"# TOTAL: pt={n_pt_ok}/{len(rF)} = {n_pt_ok/len(rF)*100:.2f}%   "
          f"mix={n_mix_ok}/{len(rF)} = {n_mix_ok/len(rF)*100:.2f}%")
    print("\n# Per-category:")
    for cat, n in sorted(cat_counts.items()):
        pt_c, mix_c = cat_correct[cat]
        print(f"  {cat:<25} n={n:>3}  pt={pt_c}/{n}={pt_c/n*100:5.1f}%  "
              f"mix={mix_c}/{n}={mix_c/n*100:5.1f}%")

if __name__ == "__main__":
    main()
