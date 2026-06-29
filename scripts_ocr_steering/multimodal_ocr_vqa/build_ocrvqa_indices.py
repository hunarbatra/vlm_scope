#!/usr/bin/env python3
"""
Generate reproducible 10K-sample indices for OCR-VQA train (firing) and test (ablation).

For each sampled row, also pick one of the 5 (question, answer) pairs.
Writes:
  analysis_ocrvqa/ocrvqa_indices_train.json — 10K samples for firing
  analysis_ocrvqa/ocrvqa_indices_test.json  — 10K samples for ablation

Each entry: {row_idx, q_idx, question, answer} (image fetched live from dataset).
"""
import os, json, random
from pathlib import Path
os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"

from datasets import load_dataset

OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocrvqa")
N_TRAIN = 10_000
N_TEST = 10_000
SEED = 42

def build(split, out_path, n, seed):
    if out_path.exists():
        d = json.load(open(out_path))
        print(f"[CACHED] {out_path.name}: {len(d)} samples")
        return
    print(f"[INFO] Loading OCR-VQA/{split}...", flush=True)
    ds = load_dataset("howard-hou/OCR-VQA", split=split)
    print(f"[INFO] {split}: {len(ds)} rows total", flush=True)
    rng = random.Random(seed)
    n = min(n, len(ds))
    rows = rng.sample(range(len(ds)), n)
    out = []
    for r in rows:
        ex = ds[r]
        qs = ex.get("questions", []) or []
        ans = ex.get("answers", []) or []
        if not qs or not ans:
            continue
        k = min(len(qs), len(ans))
        qi = rng.randrange(k)
        out.append({"row_idx": r, "q_idx": qi,
                    "question": str(qs[qi]).strip(),
                    "answer":  str(ans[qi]).strip()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: json.dump(out, f)
    print(f"[INFO] wrote {out_path.name}: {len(out)} samples")

if __name__ == "__main__":
    build("train", OUT_DIR / "ocrvqa_indices_train.json", N_TRAIN, SEED)
    build("test",  OUT_DIR / "ocrvqa_indices_test.json",  N_TEST,  SEED + 1)
