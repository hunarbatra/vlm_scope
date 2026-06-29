#!/usr/bin/env python3
"""
Phase A — Generate mix-448 responses on all VLSBench samples.
Dataset: Foreshhh/vlsbench (2,241 samples, images bundled in parquet).

Prompt: mix-448 VQA format: "answer en <instruction>".

Output: analysis_safety/responses/mix448_vlsbench_responses.jsonl
  {instruction_id, category, sub_category, instruction, response, status, ...}

Resumable. Usage: CUDA_VISIBLE_DEVICES=X python3 -B 21_generate_vlsbench.py
"""
import json, os, warnings
from pathlib import Path
import torch

warnings.filterwarnings("ignore")

MODEL_NAME = "google/paligemma2-3b-mix-448"
OUT_FILE = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_safety/responses/mix448_vlsbench_responses.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    device = "cuda:0"
    print(f"[INFO] Loading {MODEL_NAME}...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_NAME)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()

    print(f"[INFO] Loading VLSBench ...", flush=True)
    ds = load_dataset("Foreshhh/vlsbench", split="train")
    print(f"[INFO] {len(ds)} samples. Categories: {sorted(set(ds['category']))}", flush=True)

    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try: done_ids.add(json.loads(line).get("instruction_id"))
                except Exception: pass
        print(f"[INFO] resuming — {len(done_ids)} already done", flush=True)

    out = open(OUT_FILE, "a")
    n_ok = n_err = n_skip = 0
    try:
        for i in range(len(ds)):
            s = ds[i]
            iid = str(s["instruction_id"])
            if iid in done_ids:
                n_skip += 1; continue
            img = s["image"]
            if not hasattr(img, "size"):
                n_err += 1; continue
            prompt = f"answer en {s['instruction']}"
            try:
                inputs = proc(text=prompt, images=img.convert("RGB"),
                              return_tensors="pt").to(device, torch.bfloat16)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out_ids = mdl.generate(
                        **inputs, max_new_tokens=200, do_sample=False,
                        use_cache=True, pad_token_id=proc.tokenizer.pad_token_id,
                    )
                text = proc.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()
                rec = {
                    "instruction_id": iid,
                    "category": s["category"], "sub_category": s["sub_category"],
                    "instruction": s["instruction"], "image_description": s["image_description"],
                    "safety_reason": s["safety_reason"], "source": s["source"],
                    "response": text, "status": "ok",
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                n_ok += 1
                if n_ok <= 6 or n_ok % 100 == 0:
                    print(f"  [{i+1}/{len(ds)}] id={iid} cat={s['category']}/{s['sub_category']}\n"
                          f"    Q: {s['instruction'][:120]!r}\n"
                          f"    A: {text[:220]!r}", flush=True)
            except Exception as e:
                n_err += 1
                if n_err <= 10: print(f"  [ERR] id={iid}: {e}", flush=True)
                rec = {"instruction_id": iid, "category": s["category"], "response": None,
                       "status": f"error: {e}"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    finally:
        out.close()
    print(f"[DONE] ok={n_ok} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
