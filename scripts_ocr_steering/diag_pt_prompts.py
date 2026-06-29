#!/usr/bin/env python3
"""Try different prompt formats with pt-448 on R(L17_F13602) OCR samples."""
import os, sys, json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL  = "google/paligemma2-3b-pt-448"
SAE_ROOT  = Path("/data1/vlm_scope_sae_mix448_textonly")
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


PROMPTS = [
    ("answer_en", lambda q: f"answer en {q}"),
    ("just_q",    lambda q: q),
    ("q_blank",   lambda q: f"{q}\n"),
    ("ocr",       lambda q: "ocr"),
    ("caption_en",lambda q: "caption en"),
    ("answer_q",  lambda q: f"answer {q}"),
]


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    feature_key = "L17_F13602"
    device = "cuda:0"

    ds = load_dataset("echo840/OCRBench", split="test")
    ad = json.load(open(SAE_ACTS / f"acts_{feature_key}.json"))
    rF = sorted([int(k) for k, v in ad["acts"].items() if v > 0])
    print(f"# R({feature_key}) over all 1000: n={len(rF)}", flush=True)

    print(f"# Loading {PT_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl  = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok  = proc.tokenizer

    # Per-prompt accuracy + sample outputs
    results = {name: {"correct": 0, "total": 0, "samples": []} for name, _ in PROMPTS}

    for i, si in enumerate(rF):
        ex = ds[si]
        img = ex.get("image"); q = str(ex.get("question","")).strip()
        gt  = _parse_gt(ex.get("answer"))
        if img is None or not q or not gt: continue
        img = img.convert("RGB")

        for name, builder in PROMPTS:
            prompt = builder(q)
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, prompt, proc, mdl, device=device)
                with torch.no_grad():
                    out_ids = mdl.generate(
                        input_ids=iids, attention_mask=attn, pixel_values=pv,
                        max_new_tokens=64, do_sample=False, use_cache=True)
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True).strip()
            except Exception as e:
                resp = f"<ERR:{type(e).__name__}>"
            ok = _correct(resp, gt)
            results[name]["correct"] += ok
            results[name]["total"]   += 1
            if i < 8:
                results[name]["samples"].append((si, q[:60], gt[:30], resp[:30], ok))

        if (i+1) % 25 == 0:
            print(f"# Progress {i+1}/{len(rF)}", flush=True)
            for name, r in results.items():
                if r["total"]:
                    print(f"#   {name:<12} {r['correct']}/{r['total']} = {r['correct']/r['total']*100:.2f}%", flush=True)

    print("\n# === FINAL ===")
    for name, r in results.items():
        if r["total"]:
            print(f"# {name:<12} {r['correct']}/{r['total']} = {r['correct']/r['total']*100:.2f}%")

    print("\n# === First 8 samples per prompt ===")
    for name, r in results.items():
        print(f"\n--- {name} ---")
        for si, q, gt, resp, ok in r["samples"]:
            mk = "✓" if ok else "✗"
            print(f"  {mk} si={si:>4} Q='{q}' GT='{gt}' resp='{resp}'")


if __name__ == "__main__":
    main()
