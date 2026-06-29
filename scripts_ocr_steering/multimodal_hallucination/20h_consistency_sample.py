#!/usr/bin/env python3
"""
Step 0 — Multi-run consistency sampling on HallusionBench.

Runs 5 SAMPLED (temperature > 0) generations of mix-448 on all 951 HallusionBench
image-split samples, across 8 GPUs in parallel. Each sample is processed on one
GPU for all 5 runs (same sample → same GPU) to avoid CUDA issues.

Output:
  analysis_hallucination/consistency/runs/run_{0..4}_mix448_hb.jsonl

Later, 20h_build_consistency_split.py aggregates these to build:
  - robustly-CORRECT samples (correct on all 5 runs)
  - robustly-INCORRECT samples (incorrect on all 5 runs)
  - discarded borderline samples

Usage: python3 -B 20h_consistency_sample.py
"""
import json, os, sys, warnings
from pathlib import Path
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

MODEL_NAME = "google/paligemma2-3b-mix-448"
N_RUNS = 5
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 20
N_GPUS = 8

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
OUT_DIR = ROOT / "analysis_hallucination" / "consistency" / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = str(ROOT / "hf_datasets_cache")
os.environ["HF_HOME"] = "/data1/hbatra/mmdiff/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def _uid(s):
    return f"{s['category']}_{s['subcategory']}_set{s['set_id']}_fig{s['figure_id']}_q{s['question_id']}"


def _worker(gpu_id, sample_indices):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from datasets import load_dataset

    print(f"[GPU{gpu_id}] loading model...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL_NAME)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()

    print(f"[GPU{gpu_id}] loading HallusionBench...", flush=True)
    ds = load_dataset("lmms-lab/HallusionBench", split="image")
    print(f"[GPU{gpu_id}] {len(sample_indices)} samples × {N_RUNS} runs", flush=True)

    # Each run is a separate file, so resumability is easy
    out_files = []
    done_per_run = [set() for _ in range(N_RUNS)]
    for run_idx in range(N_RUNS):
        p = OUT_DIR / f"run_{run_idx}_mix448_hb_gpu{gpu_id}.jsonl"
        out_files.append(p)
        if p.exists():
            for line in open(p):
                try: done_per_run[run_idx].add(json.loads(line).get("uid"))
                except Exception: pass
    for run_idx in range(N_RUNS):
        if done_per_run[run_idx]:
            print(f"[GPU{gpu_id}] run {run_idx}: resuming — {len(done_per_run[run_idx])} done", flush=True)

    handles = [open(p, "a") for p in out_files]

    try:
        # Per-GPU seed so each GPU's 5 runs are deterministic within-GPU
        # but independent across runs
        for n_done, idx in enumerate(sample_indices):
            s = ds[idx]
            uid = _uid(s)
            img = s["image"]
            if not hasattr(img, "size"): continue
            prompt = f"answer en {s['question']}"

            for run_idx in range(N_RUNS):
                if uid in done_per_run[run_idx]: continue
                try:
                    torch.manual_seed(run_idx * 100_003 + idx * 17 + gpu_id * 1009)
                    inputs = proc(text=prompt, images=img.convert("RGB"),
                                  return_tensors="pt").to(device, torch.bfloat16)
                    input_len = inputs["input_ids"].shape[-1]
                    with torch.no_grad():
                        out_ids = mdl.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            use_cache=True,
                            pad_token_id=proc.tokenizer.pad_token_id,
                        )
                    text = proc.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()
                    rec = {
                        "uid": uid,
                        "category": s["category"],
                        "subcategory": s["subcategory"],
                        "gt_answer": s["gt_answer"],
                        "question": s["question"],
                        "run_idx": run_idx,
                        "response": text,
                        "status": "ok",
                    }
                except Exception as e:
                    rec = {"uid": uid, "run_idx": run_idx, "response": None, "status": f"error: {e}"}
                handles[run_idx].write(json.dumps(rec, ensure_ascii=False) + "\n")
                handles[run_idx].flush()

            if (n_done + 1) % 20 == 0 and gpu_id == 0:
                print(f"  [GPU0] {n_done+1}/{len(sample_indices)}", flush=True)
    finally:
        for h in handles: h.close()
    print(f"[GPU{gpu_id}] done", flush=True)


def main():
    # Distribute 951 samples across 8 GPUs by modulo
    mp.set_start_method("spawn", force=True)
    total = 951
    shards = [[] for _ in range(N_GPUS)]
    for i in range(total):
        shards[i % N_GPUS].append(i)
    print(f"[MAIN] {total} samples × {N_RUNS} runs across {N_GPUS} GPUs")
    for g, s in enumerate(shards):
        print(f"  GPU{g}: {len(s)} samples")

    procs = []
    for g in range(N_GPUS):
        p = mp.Process(target=_worker, args=(g, shards[g]))
        p.start(); procs.append(p)
    for p in procs: p.join()
    print("[MAIN] all done")


if __name__ == "__main__":
    main()
