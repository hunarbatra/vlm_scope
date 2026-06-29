#!/usr/bin/env python3
"""
Step 7 (safety variant) — lexical-artifact filter.

For each unsafe candidate feature F from Step 6 (overall pass):
  1. Scan UP TO 500 HallusionBench-unsafe samples → find top-5 most-activating samples
  2. Keep their IMAGE, replace the original unsafe-sounding prompt with a
     GENERIC benign prompt ("Describe this image.", etc.)
  3. Re-run VLM + SAE, check whether F still activates strongly (> threshold)
  4. Feature PASSES iff it fires on ALL top-5 samples with the benign prompt
     (meaning F responds to the visual unsafety in the image, not to lexical
      cues in the prompt text).
  5. Features that drop below threshold on any top sample are filtered out
     (they're lexical artifacts tied to specific words in the unsafe prompts).

Output: analysis_hallucination/_greedy/lexical/lexical_results_w<GPU>.json

Mirrors steps 5/7 in local_analysis_textonly.py. Runs 8-GPU parallel.

Usage: python3 -B 32_lexical_filter_safety.py
"""
import json, os, sys, gc, warnings
from pathlib import Path
from collections import defaultdict
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_NAME = "google/paligemma2-3b-mix-448"
HF_CACHE = "/data1/vlm_scope_sae_docci/hf_cache/hub"
N_GPUS = 8
D_SAE = 16384

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
CHECKPOINT_DIR = ROOT / "checkpoints"
ANALYSIS_DIR = ROOT / "analysis_hallucination/_greedy"
JUDGE_FILE   = ROOT / "analysis_hallucination" / "judgments" / "mix448_hallusionbench_judgments.jsonl.greedy_backup"
HALLUC_CSV   = ANALYSIS_DIR / "halluc_pertoken" / "halluc_features_pertoken.csv"
OUT_DIR       = ANALYSIS_DIR / "lexical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]  = "/data1/hbatra/mmdiff/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

# Benign prompts — generic "describe this image" style, no unsafe cues
GENERIC_PROMPTS = [
    "Describe this image.",
    "What do you see in this picture?",
    "Summarize the contents of the image.",
    "Describe the objects and scene in this image.",
    "What is happening in this image?",
]

TOP_K = 5
SCAN_SAMPLES = 500
ACT_THR = 0.01


def _uid(s):
    return f"{s['category']}_{s['subcategory']}_set{s['set_id']}_fig{s['figure_id']}_q{s['question_id']}"


def load_incorrect_ids():
    out = []
    for line in open(JUDGE_FILE):
        r = json.loads(line)
        if r.get("pred_label") == "INCORRECT":
            out.append(r["uid"])
    return set(out)


def _lexical_worker(gpu_id, feature_assignments):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, "/home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2")
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset

    out_path = OUT_DIR / f"lexical_results_w{gpu_id}.json"
    if out_path.exists() and not feature_assignments:
        return
    if not feature_assignments:
        return

    print(f"[Lex GPU{gpu_id}] {len(feature_assignments)} features", flush=True)
    print(f"[Lex GPU{gpu_id}] loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE, local_files_only=True)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, local_files_only=True
    ).to(device).eval()
    nns_model = NNsight(model_raw)

    ds = load_dataset("lmms-lab/HallusionBench", split="image")
    incorrect_set = load_incorrect_ids()
    keep_idx = [i for i in range(len(ds)) if _uid(ds[i]) in incorrect_set]
    scan_n = min(SCAN_SAMPLES, len(keep_idx))
    print(f"[Lex GPU{gpu_id}] scan pool: {scan_n} samples", flush=True)

    layer_features = defaultdict(list)
    for (l, f) in feature_assignments:
        layer_features[l].append(f)

    results = []
    for layer_idx in sorted(layer_features.keys()):
        features = layer_features[layer_idx]
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        feature_top = {f: [] for f in features}

        # --- Phase 1: find top-activating samples ---
        print(f"[Lex GPU{gpu_id}] L{layer_idx}: phase 1 scan ({len(features)} feats, {scan_n} samples)", flush=True)
        for s_i, idx in enumerate(keep_idx[:scan_n]):
            try:
                sample = ds[idx]
                img = sample["image"].convert("RGB")
                q = sample["question"]
                prompt = f"answer en {q}"
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                with torch.no_grad():
                    with nns_model.trace(input_ids=iids, attention_mask=attn,
                                         pixel_values=pv, use_cache=False) as tr:
                        layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()
                act = layer_out.detach().squeeze(0).float()
                with torch.no_grad():
                    codes = sae.encode(act.to(device)).detach().cpu()
                for f in features:
                    m = float(codes[:, f].max().item())
                    if m > 0:
                        feature_top[f].append((m, idx))
            except Exception as e:
                if s_i < 3: print(f"  L{layer_idx} scan err idx={idx}: {e}", flush=True)

        # --- Phase 2: re-test top-k with generic prompts ---
        print(f"[Lex GPU{gpu_id}] L{layer_idx}: phase 2 lexical test", flush=True)
        for f in features:
            cands = sorted(feature_top[f], key=lambda x: -x[0])[:TOP_K]
            if not cands:
                results.append({"layer": layer_idx, "feature": f, "passed": False,
                                "n_tested": 0, "note": "never_fired_in_scan"})
                continue
            passed = True
            details = []
            for (mag, idx) in cands:
                sample = ds[idx]
                img = sample["image"].convert("RGB")
                best_generic_max = 0.0
                for raw in GENERIC_PROMPTS:
                    prompt = f"answer en {raw}"
                    try:
                        iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                        with torch.no_grad():
                            with nns_model.trace(input_ids=iids, attention_mask=attn,
                                                 pixel_values=pv, use_cache=False) as tr:
                                lo = nns_model.model.language_model.layers[layer_idx].output[0].save()
                        act = lo.detach().squeeze(0).float()
                        with torch.no_grad():
                            codes = sae.encode(act.to(device)).detach().cpu()
                        m = float(codes[:, f].max().item())
                        best_generic_max = max(best_generic_max, m)
                    except Exception: continue
                details.append({"vqa_idx": int(idx), "orig_mag": mag, "generic_max": best_generic_max})
                if best_generic_max < ACT_THR:
                    passed = False
            results.append({"layer": layer_idx, "feature": f, "passed": passed,
                            "n_tested": len(cands), "details": details})

        del sae; torch.cuda.empty_cache(); gc.collect()

    with open(out_path, "w") as ff:
        json.dump({"results": results}, ff)
    n_pass = sum(1 for r in results if r["passed"])
    print(f"[Lex GPU{gpu_id}] done. {n_pass}/{len(results)} passed", flush=True)


def main():
    mp.set_start_method("spawn", force=True)
    if not HALLUC_CSV.exists():
        print("[FATAL] halluc_features_pertoken.csv missing. Run 31_fisher first."); return
    df = pd.read_csv(HALLUC_CSV)
    print(f"[MAIN] {len(df)} candidate unsafe features to lexical-test")
    feats = list(zip(df["layer"].astype(int), df["feature"].astype(int)))
    shards = [[] for _ in range(N_GPUS)]
    for i, f in enumerate(feats):
        shards[i % N_GPUS].append(f)
    for g, s in enumerate(shards):
        print(f"  GPU{g}: {len(s)} features")
    procs = []
    for g in range(N_GPUS):
        p = mp.Process(target=_lexical_worker, args=(g, shards[g]))
        p.start(); procs.append(p)
    for p in procs: p.join()
    print("[MAIN] all workers done")


if __name__ == "__main__":
    main()
