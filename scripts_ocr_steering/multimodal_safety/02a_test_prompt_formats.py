#!/usr/bin/env python3
"""
Compare 4 prompt formats on 5 MSTS samples to see which elicits rich responses
from mix-448. Prints all 20 (format × sample) outputs for hand-inspection.
"""
import csv, json, os
from pathlib import Path
from PIL import Image
import torch

MODEL_NAME = "google/paligemma2-3b-mix-448"
MSTS_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_safety/msts")
IMG_DIR    = MSTS_DIR / "images"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

FORMATS = {
    "vqa_answer_en":   lambda q: f"answer en {q}",
    "question_only":   lambda q: q,
    "caption_chat":    lambda q: f"caption en\nUser: {q}\nPlease reply as a helpful assistant with a complete answer.\nAssistant:",
    "cot":             lambda q: f"answer en {q} Explain your reasoning.",
    "ocr_trick":       lambda q: f"describe the image and then answer: {q}",
}

def main():
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(MODEL_NAME)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()

    with open(MSTS_DIR / "english_multimodal.csv") as f:
        rows = list(csv.DictReader(f))
    # Pick 5 samples with downloaded images, diverse hazards
    picks = []
    seen_cat = set()
    for r in rows:
        if (IMG_DIR / f"{r['unsafe_image_id']}.jpg").exists() and r["hazard_subcategory"] not in seen_cat:
            picks.append(r); seen_cat.add(r["hazard_subcategory"])
        if len(picks) >= 5: break

    for r in picks:
        img = Image.open(IMG_DIR / f"{r['unsafe_image_id']}.jpg").convert("RGB")
        print("="*90)
        print(f"{r['prompt_id']} [{r['hazard_subcategory']}]  image: {r['unsafe_image_description']}")
        print(f"  USER: {r['prompt_text']}")
        for name, fmt in FORMATS.items():
            prompt = fmt(r["prompt_text"])
            inputs = proc(text=prompt, images=img, return_tensors="pt").to(device, torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out_ids = mdl.generate(
                    **inputs, max_new_tokens=150, do_sample=False,
                    use_cache=True, pad_token_id=proc.tokenizer.pad_token_id,
                )
            gen = out_ids[0, input_len:]
            text = proc.decode(gen, skip_special_tokens=True).strip()
            print(f"  [{name:<18}] {text!r}")


if __name__ == "__main__":
    main()
