#!/usr/bin/env python3
"""
DocVQA baseline evaluation for pt-448 and mix-448.
Runs on DocVQA validation (5349 samples, all have answers).
Reports overall accuracy + per question_type breakdown.

Usage:
    python3 -u docvqa_baseline_eval.py --model pt  --device cuda:6 2>&1 | tee /tmp/docvqa_baseline_pt.log
    python3 -u docvqa_baseline_eval.py --model mix --device cuda:7 2>&1 | tee /tmp/docvqa_baseline_mix.log
"""
import os, sys, json, argparse, warnings
from pathlib import Path
from collections import defaultdict

os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
warnings.filterwarnings("ignore")

import torch
from PIL import Image as PILImage

PT_MODEL  = "google/paligemma2-3b-pt-448"
MIX_MODEL = "google/paligemma2-3b-mix-448"
OUT_BASE  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_docvqa")
MAX_NEW_TOKENS = 64


def _correct(response, gt_list):
    if response is None: return False
    if isinstance(gt_list, str): gt_list = [gt_list]
    resp = response.strip().lower()
    if not resp: return False
    for gt in gt_list:
        gt_l = str(gt).strip().lower()
        if not gt_l: continue
        if gt_l in resp or resp in gt_l: return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pt", "mix"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="Cap samples for quick test")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import process_vlm_inputs
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    model_id = PT_MODEL if args.model == "pt" else MIX_MODEL
    device = args.device

    print(f"[INFO] Model: {model_id}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)

    print("[INFO] Loading DocVQA validation...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    print(f"[INFO] {len(ds)} validation samples", flush=True)

    print(f"[INFO] Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = processor.tokenizer

    results = []
    n = args.limit or len(ds)
    correct_total = 0
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for si in range(n):
        ex = ds[si]
        question = str(ex.get("question", "")).strip()
        img = ex.get("image")
        gt_list = ex.get("answers", [])
        qtypes = ex.get("question_types", ["unknown"])
        if img is None or not question or not gt_list:
            continue
        try:
            img = img.convert("RGB")
            prompt = f"answer en {question}"
            input_ids, attn_mask, pixel_values = process_vlm_inputs(
                img, prompt, processor, model, device=device)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            resp = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            ok = _correct(resp, gt_list)
        except Exception as e:
            resp, ok = "", False

        correct_total += int(ok)
        for qt in qtypes:
            by_type[qt]["correct"] += int(ok)
            by_type[qt]["total"] += 1
        results.append({"si": si, "correct": bool(ok), "response": resp,
                        "question": question, "question_types": qtypes})

        if (si + 1) % 200 == 0:
            print(f"  {si+1}/{n}  acc={100*correct_total/(si+1):.1f}%", flush=True)

    overall_acc = 100 * correct_total / max(len(results), 1)
    print(f"\n=== {args.model.upper()} DocVQA Results ===", flush=True)
    print(f"Overall: {overall_acc:.2f}%  ({correct_total}/{len(results)})", flush=True)
    print(f"{'Question Type':<30} {'Acc':>6}  (n)", flush=True)
    print("-" * 50, flush=True)
    for qt, d in sorted(by_type.items(), key=lambda x: -x[1]["total"]):
        acc = 100 * d["correct"] / max(d["total"], 1)
        print(f"  {qt:<28} {acc:>5.1f}%  ({d['correct']}/{d['total']})", flush=True)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model,
        "model_id": model_id,
        "overall_acc": overall_acc,
        "correct": correct_total,
        "total": len(results),
        "by_type": {qt: {"acc": 100*d["correct"]/max(d["total"],1),
                         "correct": d["correct"], "total": d["total"]}
                    for qt, d in by_type.items()},
        "results": results,
    }
    out_path = OUT_BASE / f"baseline_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
