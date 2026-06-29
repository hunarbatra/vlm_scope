#!/usr/bin/env python3
"""
Phase A — Generate mix-448 responses on all MM-SafetyBench samples
(13 categories × SD_TYPO variant). Resumable.

Output: analysis_safety/responses/mix448_mmsafety_responses.jsonl
  {"category", "variant", "idx", "question_id", "question", "response", "status"}

Usage: CUDA_VISIBLE_DEVICES=X python3 -B 11_generate_mm_safetybench.py
"""
import json, os, warnings
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch

warnings.filterwarnings("ignore")

MODEL_NAME = "google/paligemma2-3b-mix-448"
OUT_FILE   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_safety/responses/mix448_mmsafety_responses.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

CATEGORIES = [
    "EconomicHarm", "Financial_Advice", "Fraud", "Gov_Decision", "HateSpeech",
    "Health_Consultation", "Illegal_Activitiy", "Legal_Opinion",
    "Malware_Generation", "Physical_Harm", "Political_Lobbying",
    "Privacy_Violence", "Sex",
]
VARIANT = "SD_TYPO"   # strongest attack variant per MM-SafetyBench paper


def decode_img(obj):
    if isinstance(obj, dict) and "bytes" in obj:
        return Image.open(BytesIO(obj["bytes"])).convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(BytesIO(obj)).convert("RGB")
    if hasattr(obj, "size"):
        return obj.convert("RGB")
    return None


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    device = "cuda:0"
    print(f"[INFO] Loading {MODEL_NAME}...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_NAME)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()

    done_keys = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["category"], r["variant"], r["idx"]))
                except Exception: pass
        print(f"[INFO] resuming — {len(done_keys)} records already done", flush=True)

    out = open(OUT_FILE, "a")
    n_ok = n_err = n_skip = 0
    try:
        for cat in CATEGORIES:
            print(f"\n[INFO] {cat} / {VARIANT} ...", flush=True)
            ds = load_dataset(
                "PKU-Alignment/MM-SafetyBench",
                data_files=f"data/{cat}/{VARIANT}.parquet",
                split="train",
            )
            for i in range(len(ds)):
                key = (cat, VARIANT, i)
                if key in done_keys:
                    n_skip += 1; continue
                s = ds[i]
                img = decode_img(s["image"])
                if img is None:
                    n_err += 1; continue
                # MM-SafetyBench SD_TYPO prompt: the question itself is the instruction
                prompt = f"answer en {s['question']}"
                try:
                    inputs = proc(text=prompt, images=img, return_tensors="pt").to(device, torch.bfloat16)
                    input_len = inputs["input_ids"].shape[-1]
                    with torch.no_grad():
                        out_ids = mdl.generate(
                            **inputs, max_new_tokens=200, do_sample=False,
                            use_cache=True, pad_token_id=proc.tokenizer.pad_token_id,
                        )
                    text = proc.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()
                    rec = {"category": cat, "variant": VARIANT, "idx": i,
                           "question_id": str(s.get("id", i)),
                           "question": s["question"], "response": text, "status": "ok"}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                    n_ok += 1
                    if n_ok <= 5 or n_ok % 100 == 0:
                        print(f"  [{cat} #{i}] {text[:140]!r}", flush=True)
                except Exception as e:
                    n_err += 1
                    if n_err <= 10:
                        print(f"  [ERR] {cat} #{i}: {e}", flush=True)
                    rec = {"category": cat, "variant": VARIANT, "idx": i,
                           "question_id": str(s.get("id", i)),
                           "question": s["question"], "response": None,
                           "status": f"error: {e}"}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    finally:
        out.close()
    print(f"[DONE] ok={n_ok} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
