#!/usr/bin/env python3
"""
Phase A, Step 2 — Generate mix-448 responses on all 400 MSTS prompts.

Uses the VLM to produce free-form answers, saved as JSONL:
  {"prompt_id", "case_id", "unsafe_image_id", "hazard_subcategory",
   "prompt_text", "prompt_type", "response"}

Image is loaded from analysis_safety/msts/images/<unsafe_image_id>.jpg;
any prompt whose image failed to download is skipped (tracked in log).

Uses PaliGemma2 mix-448 VQA-style prompt format: "answer en <prompt_text>".

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B 02_generate_msts_responses.py
"""
import csv, json, os, sys, warnings
from pathlib import Path
from PIL import Image
import torch

warnings.filterwarnings("ignore")

MODEL_NAME = "google/paligemma2-3b-mix-448"
MSTS_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_safety/msts")
PROMPTS    = MSTS_DIR / "english_multimodal.csv"
IMG_DIR    = MSTS_DIR / "images"
OUT_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_safety/responses")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE   = OUT_DIR / "mix448_msts_responses.jsonl"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def main():
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    device = "cuda:0"
    print(f"[INFO] Loading {MODEL_NAME}...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_NAME)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()

    with open(PROMPTS) as f:
        rows = list(csv.DictReader(f))
    print(f"[INFO] {len(rows)} prompts, {len(list(IMG_DIR.glob('*.jpg')))} images on disk", flush=True)

    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["prompt_id"])
                except Exception: pass
        print(f"[INFO] resuming — {len(done_ids)} prompts already done", flush=True)

    n_ok = n_skip = n_missing = n_err = 0
    out = open(OUT_FILE, "a")
    try:
        for i, r in enumerate(rows):
            pid  = r["prompt_id"]
            iid  = r["unsafe_image_id"]
            if pid in done_ids:
                n_skip += 1; continue
            img_p = IMG_DIR / f"{iid}.jpg"
            if not img_p.exists():
                n_missing += 1
                # Still write a record so we can filter downstream
                rec = dict(r); rec["response"] = None; rec["status"] = "image_missing"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                continue
            try:
                img = Image.open(img_p).convert("RGB")
                # caption-style elicit full response rather than yes/no
                # PaliGemma 2 mix-448 is trained on caption/VQA/detection — "answer en" gives short answers.
                # We want full reasoning: use the `describe_and_answer` pattern that prompts for longer generation.
                prompt = f"answer en {r['prompt_text']}"
                inputs = proc(text=prompt, images=img, return_tensors="pt").to(device, torch.bfloat16)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out_ids = mdl.generate(
                        **inputs, max_new_tokens=200, do_sample=False,
                        use_cache=True, pad_token_id=proc.tokenizer.pad_token_id,
                    )
                gen = out_ids[0, input_len:]
                text = proc.decode(gen, skip_special_tokens=True).strip()
                rec = dict(r); rec["response"] = text; rec["status"] = "ok"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                n_ok += 1
                if (n_ok + n_skip) <= 5 or (n_ok + n_skip) % 25 == 0:
                    print(f"  [{i+1}/{len(rows)}] {pid} ({r['hazard_subcategory']}): {text[:100]!r}", flush=True)
            except Exception as e:
                n_err += 1
                if n_err <= 10:
                    print(f"  [ERR] {pid}: {e}", flush=True)
                rec = dict(r); rec["response"] = None; rec["status"] = f"error: {e}"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    finally:
        out.close()
    print(f"[DONE] ok={n_ok} skip={n_skip} missing={n_missing} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
