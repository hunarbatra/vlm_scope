#!/usr/bin/env python3
"""
Step 10b (safety) — SAFE-control ablation.

Same as 35_ablate_safety_features.py but ablates on 100 samples that were
originally judged SAFE on baseline mix-448 (stratified across categories).

Purpose: confirm the ΔASR signal for unsafe features is specific to
unsafe-compliant contexts. If the feature is compliance-dormant on safe
prompts, the "ASR on safe subset" should stay near zero under ablation
(i.e. ablating the feature doesn't turn safe responses into unsafe ones).

Outputs live in a dedicated subdir so the UNSAFE ablation isn't overwritten:
  analysis_safety/ablation_results_safe/responses_L{L}_F{F}.jsonl
  analysis_safety/ablation_results_safe/vqa_L{L}_F{F}.json

Step 38 judges & summarizes.

Usage: python3 -B 37_ablate_safe_control.py
"""
import os, sys, json, gc, warnings, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

# reuse the heavy worker from 35_ablate_safety_features.py
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ab35", Path(__file__).parent / "35_ablate_safety_features.py")
ab35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab35)

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
ANALYSIS_DIR = ROOT / "analysis_safety"
FEATURES_CSV = ANALYSIS_DIR / "ablation_input" / "features_to_ablate.csv"
JUDGE_FILE   = ANALYSIS_DIR / "judgments" / "mix448_vlsbench_qwen_judgments.jsonl"
OUT_DIR      = ANALYSIS_DIR / "ablation_results_safe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAFE_EVAL = 100
N_VQA_EVAL  = 200
N_GPUS      = 8


def load_safe_eval_set():
    per_cat = defaultdict(list)
    with open(JUDGE_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("judge_label") == "SAFE":
                per_cat[r["category"]].append(r["instruction_id"])
    total = sum(len(v) for v in per_cat.values())
    stratified = []
    for cat, ids in per_cat.items():
        n = max(1, round(N_SAFE_EVAL * len(ids) / total))
        stride = max(1, len(ids) // n)
        picks = ids[::stride][:n]
        for iid in picks:
            stratified.append((iid, cat))
    return stratified[:N_SAFE_EVAL]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(N_GPUS)))
    args = parser.parse_args()

    # Hot-swap the UNSAFE loader + OUT_DIR inside ab35 before spawning workers
    # (spawn will re-import, so we also pass via a small shim)
    if not FEATURES_CSV.exists():
        print(f"[FATAL] {FEATURES_CSV} missing"); return

    df = pd.read_csv(FEATURES_CSV)
    print(f"[MAIN] {len(df)} features, SAFE-control mode")

    shards = [[] for _ in args.gpus]
    for i, row in df.iterrows():
        shards[i % len(args.gpus)].append(row.to_dict())
    for g, s in zip(args.gpus, shards):
        print(f"  GPU{g}: {len(s)} features")

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id, shard in zip(args.gpus, shards):
        if not shard: continue
        p = mp.Process(target=_safe_worker, args=(gpu_id, shard))
        p.start(); procs.append(p)
    for p in procs: p.join()
    print("[MAIN] all workers done")


def _safe_worker(gpu_id, feature_rows):
    # Monkey-patch ab35 inside this spawned process
    ab35.load_eval_set = load_safe_eval_set
    ab35.OUT_DIR = OUT_DIR
    ab35.N_VLSBENCH_EVAL = N_SAFE_EVAL
    ab35._ablation_worker(gpu_id, feature_rows)


if __name__ == "__main__":
    main()
