#!/usr/bin/env python3
"""
Build VQA-clean-of-OCR control set for OCR-Bench ablation.

Filters VQAv2 validation to ~500 yes/no samples whose:
  1. answer_type == 'yes/no'
  2. question does NOT mention text/numbers/signs/time
  3. image has NO detected text regions (per EasyOCR detector)

Output:
  /data1/vlm_scope_sae_mix448_textonly/analysis_ocr/vqa_clean_yesno/indices.json
    list of {vqa_index: int, question: str, label: int(0/1), n_text_regions: 0}

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -B build_vqa_clean_control.py [--target-n 500]
"""
import os, sys, json, re, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"

OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/vqa_clean_yesno")

# Drop questions about text content
TEXT_Q_PATTERNS = re.compile(
    r"\b(sign|signs|signage|text|letter|letters|word|words|written|writing|read|reads|reading|"
    r"label|labeled|labelled|caption|title|name|named|number|numbers|numbered|digit|digits|"
    r"time|clock|watch|date|year|brand|logo|printed|print|menu|page|book|paper|note|notice|"
    r"poster|billboard|graffiti|street name|license plate|plate|message|inscription|spelling|"
    r"says?|spelled?|font|character|symbol)\b",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-n", type=int, default=500)
    ap.add_argument("--scan-max", type=int, default=20000,
                    help="Max VQAv2 samples to scan before giving up")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "indices.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        print(f"[INFO] {out_path} exists with {len(existing)} samples; remove to rebuild.")
        return

    print("[INFO] Loading EasyOCR detector...", flush=True)
    import easyocr
    import torch
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False,
                              model_storage_directory="/data1/hf_cache/easyocr_models",
                              download_enabled=True)
    print(f"[INFO] EasyOCR loaded (gpu={use_gpu})", flush=True)

    print("[INFO] Loading VQAv2 validation...", flush=True)
    from datasets import load_dataset
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    print(f"[INFO] VQAv2 validation: {len(vqa)} samples", flush=True)

    import numpy as np

    kept = []
    n_seen = n_yesno = n_qfilter = n_imgfilter = 0
    scan_n = min(args.scan_max, len(vqa))

    for i in range(scan_n):
        n_seen += 1
        ex = vqa[i]
        # 1. yes/no answer type
        if str(ex.get("answer_type", "")).lower() != "yes/no":
            continue
        mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
        if mc not in {"yes", "no"}:
            continue
        n_yesno += 1

        # 2. text-question filter
        question = str(ex.get("question", "")).strip()
        if TEXT_Q_PATTERNS.search(question):
            n_qfilter += 1
            continue

        # 3. image-text-detection filter
        try:
            img = ex.get("image")
            if img is None: continue
            img = img.convert("RGB")
            arr = np.array(img)
            # Use detect() not readtext() — only bounding boxes, faster
            horizontal_list, free_list = reader.detect(arr, text_threshold=0.5,
                                                       low_text=0.4, link_threshold=0.4)
            n_regions = len(horizontal_list[0]) + len(free_list[0])
            if n_regions > 0:
                n_imgfilter += 1
                continue
        except Exception as e:
            continue

        kept.append({
            "vqa_index": i,
            "question": question,
            "label": 1 if mc == "yes" else 0,
            "n_text_regions": 0,
        })

        if len(kept) % 50 == 0:
            print(f"  [{i+1}/{scan_n}]  kept={len(kept):>4}  "
                  f"y/n={n_yesno}  qfilt={n_qfilter}  imgfilt={n_imgfilter}", flush=True)

        if len(kept) >= args.target_n:
            break

    print(f"\n[DONE] scanned={n_seen}, yes/no={n_yesno}, "
          f"q-text-filter={n_qfilter}, img-text-filter={n_imgfilter}, kept={len(kept)}")
    n_yes = sum(1 for k in kept if k["label"] == 1)
    n_no = len(kept) - n_yes
    print(f"  label balance: yes={n_yes} no={n_no}")

    with open(out_path, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
