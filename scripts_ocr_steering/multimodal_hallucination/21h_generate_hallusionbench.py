#!/usr/bin/env python3
"""
Phase A — Generate mix-448 responses on HallusionBench (image split, 951 samples).
Dataset: lmms-lab/HallusionBench.

Prompt: mix-448 VQA format: "answer en <question>".

Output: analysis_hallucination/responses/mix448_hallusionbench_responses.jsonl
  {uid, category, subcategory, set_id, figure_id, question_id, question,
   gt_answer, gt_answer_details, response, status}

Resumable. Usage: CUDA_VISIBLE_DEVICES=X python3 -B 21h_generate_hallusionbench.py
"""
import json, os, warnings
from pathlib import Path
import torch

warnings.filterwarnings("ignore")

MODEL_NAME = "google/paligemma2-3b-mix-448"
OUT_FILE = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_hallucination/responses/mix448_hallusionbench_responses.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def _uid(s):
    return f"{s['category']}_{s['subcategory']}_set{s['set_id']}_fig{s['figure_id']}_q{s['question_id']}"


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    device = "cuda:0"
    print(f"[INFO] Loading {MODEL_NAME}...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_NAME)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()

    print(f"[INFO] Loading HallusionBench ...", flush=True)
    ds = load_dataset("lmms-lab/HallusionBench", split="image")
    print(f"[INFO] {len(ds)} samples. Categories: {sorted(set(ds['category']))}  Subcats: {sorted(set(ds['subcategory']))}", flush=True)

    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try: done_ids.add(json.loads(line).get("uid"))
                except Exception: pass
        print(f"[INFO] resuming — {len(done_ids)} already done", flush=True)

    out = open(OUT_FILE, "a")
    n_ok = n_err = n_skip = 0
    try:
        for i in range(len(ds)):
            s = ds[i]
            uid = _uid(s)
            if uid in done_ids:
                n_skip += 1; continue
            img = s["image"]
            if not hasattr(img, "size"):
                n_err += 1; continue
            prompt = f"answer en {s['question']}"
            try:
                inputs = proc(text=prompt, images=img.convert("RGB"),
                              return_tensors="pt").to(device, torch.bfloat16)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out_ids = mdl.generate(
                        **inputs, max_new_tokens=30, do_sample=False,
                        use_cache=True, pad_token_id=proc.tokenizer.pad_token_id,
                    )
                text = proc.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()
                rec = {
                    "uid": uid,
                    "category": s["category"], "subcategory": s["subcategory"],
                    "set_id": s["set_id"], "figure_id": s["figure_id"],
                    "question_id": s["question_id"],
                    "question": s["question"],
                    "gt_answer": s["gt_answer"],
                    "gt_answer_details": s.get("gt_answer_details", ""),
                    "filename": s.get("filename", ""),
                    "response": text, "status": "ok",
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                n_ok += 1
                if n_ok <= 6 or n_ok % 100 == 0:
                    print(f"  [{i+1}/{len(ds)}] {uid}  gt={s['gt_answer']}\n"
                          f"    Q: {s['question'][:120]!r}\n"
                          f"    A: {text[:100]!r}", flush=True)
            except Exception as e:
                n_err += 1
                if n_err <= 10: print(f"  [ERR] {uid}: {e}", flush=True)
                rec = {"uid": uid, "category": s["category"], "subcategory": s["subcategory"],
                       "response": None, "status": f"error: {e}"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    finally:
        out.close()
    print(f"[DONE] ok={n_ok} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
