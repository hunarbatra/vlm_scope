#!/usr/bin/env python3
"""
DocVQA Steps 5-8 pipeline: feature firing, Fisher test, lexical filtering, intersection.

Mirrors VSR pipeline design:
  - ALL labeled DocVQA samples (validation, 5349) used for firing + Fisher (like VSR uses all 3 splits)
  - Split: first 4279 = "train" (CAA vector construction), last 1070 = "test" (steering eval)
  - lmms-lab/DocVQA has only 'validation' (labeled) and 'test' (no answers)
  - VQAv2 validation 50K = baseline contrast set

Steps 5-8:
  5: Firing frequencies (VQAv2 50K + DocVQA val all 5349)
  6: Fisher exact test (VQA vs DocVQA) → DocVQA-enriched features
  7: Lexical filtering (generic prompts on DocVQA images)
  8: Intersection (adapted ∩ DocVQA-enriched ∩ lexical-passed)

Also writes split indices: analysis_docvqa/splits.json
  {"train": [0..4278], "test": [4279..5348]}

Usage:
    cd /data1/vlm_scope_sae_mix448_textonly/scripts/multimodal_docvqa
    python3 -u local_analysis_textonly_docvqa.py --step 5 6 7 8 --gpus 8 2>&1 | tee /tmp/docvqa_pipeline_5678.log
"""
import os, sys, gc, re, ast, csv, json, math, argparse, time, warnings
from pathlib import Path
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import numpy as np

os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
warnings.filterwarnings("ignore")

# ─────────────────────────── Config ───────────────────────────
MODEL_NAME  = "google/paligemma2-3b-mix-448"
N_LAYERS    = 26
D_SAE       = 16384
N_GPUS      = 8

CHECKPOINT_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
ANALYSIS_OCR_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr")
ANALYSIS_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_docvqa")
HF_CACHE         = "/data1/hf_cache/hub"

N_FIRING_SAMPLES = 50_000  # VQAv2 baseline
DOCVQA_TRAIN_N   = 4279    # first N val samples → CAA train split
# remaining 1070 → test split

ODDS_THR      = 3.0
MIN_FREQ_DIFF = 0.05

GENERIC_PROMPTS = [
    "Describe this image.",
    "What do you see in this picture?",
    "Summarize the contents of the image.",
    "Describe the objects and scene in this image.",
    "What is happening in this image?",
]


def write_splits(dvqa_len):
    """Write train/test split indices mirroring VSR design."""
    splits = {
        "train": list(range(DOCVQA_TRAIN_N)),
        "test":  list(range(DOCVQA_TRAIN_N, dvqa_len)),
        "all":   list(range(dvqa_len)),
        "note":  "train used for CAA vector; test used for steering eval; all used for firing+Fisher"
    }
    out_path = ANALYSIS_DIR / "splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Splits written: train={len(splits['train'])}, test={len(splits['test'])} → {out_path}")
    return splits


# ─────────────────────────── Step 5 Firing ───────────────────────────

def _firing_worker(gpu_id, layer_indices):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset

    vqa_dir  = ANALYSIS_DIR / "firing_vqa"
    dvqa_dir = ANALYSIS_DIR / "firing_docvqa"
    vqa_dir.mkdir(parents=True, exist_ok=True)
    dvqa_dir.mkdir(parents=True, exist_ok=True)

    remaining_vqa  = [l for l in layer_indices
                      if not (vqa_dir  / f"firing_vqa_layer_{l}.json").exists()]
    remaining_dvqa = [l for l in layer_indices
                      if not (dvqa_dir / f"firing_docvqa_layer_{l}.json").exists()]

    if not remaining_vqa and not remaining_dvqa:
        print(f"[Firing GPU{gpu_id}] All layers cached, skipping", flush=True)
        return

    print(f"[Firing GPU{gpu_id}] Loading model {MODEL_NAME}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_raw)

    # ── Pass 1: VQAv2 baseline ──
    if remaining_vqa:
        print(f"[Firing GPU{gpu_id}] Loading VQAv2...", flush=True)
        vqa = load_dataset("lmms-lab/VQAv2", split="validation")
        n_baseline = min(N_FIRING_SAMPLES, len(vqa))
        print(f"[Firing GPU{gpu_id}] VQA: {n_baseline} samples, layers: {remaining_vqa}", flush=True)

        for layer_idx in remaining_vqa:
            ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
            sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                          device=device, cache_dir=HF_CACHE)
            sae.eval()
            fire_count = np.zeros(D_SAE, dtype=np.int64)
            n_samples = n_errors = 0

            for si in range(n_baseline):
                try:
                    sample = vqa[si]
                    image  = sample["image"].convert("RGB")
                    prompt = f"answer en {sample['question']}"
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        image, prompt, processor, model_raw, device=device)
                    img_s, img_e = get_image_token_positions(input_ids)

                    with torch.no_grad():
                        with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                              pixel_values=pixel_values, use_cache=False) as _:
                            layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()
                    act = layer_out.detach().squeeze(0).float()
                    with torch.no_grad():
                        codes = sae.encode(act).detach()

                    seq_len  = codes.shape[0]
                    txt_mask = torch.ones(seq_len, dtype=torch.bool, device=codes.device)
                    if img_e > img_s:
                        txt_mask[img_s:img_e] = False
                    txt_codes = codes[txt_mask]
                    if txt_codes.shape[0] > 0:
                        fired = (txt_codes > 0).any(dim=0).cpu().numpy()
                        fire_count += fired.astype(np.int64)
                    n_samples += 1
                except Exception as e:
                    n_errors += 1
                    if n_errors <= 3:
                        print(f"[Firing GPU{gpu_id}] VQA err si={si}: {e}", flush=True)
                    continue
                if si % 5000 == 0 and si > 0:
                    print(f"  [GPU{gpu_id}] L{layer_idx} VQA: {si}/{n_baseline}", flush=True)

            with open(vqa_dir / f"firing_vqa_layer_{layer_idx}.json", "w") as f:
                json.dump({"layer": int(layer_idx), "n_samples": int(n_samples),
                           "n_errors": int(n_errors), "dataset": "vqa",
                           "fire_count_all": fire_count.tolist()}, f)
            print(f"[Firing GPU{gpu_id}] VQA L{layer_idx}: {n_samples} samples done", flush=True)
            del sae; torch.cuda.empty_cache(); gc.collect()

    # ── Pass 2: DocVQA ALL labeled (validation, 5349) ──
    if remaining_dvqa:
        print(f"[Firing GPU{gpu_id}] Loading DocVQA validation (all 5349)...", flush=True)
        dvqa = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        print(f"[Firing GPU{gpu_id}] DocVQA: {len(dvqa)} samples, layers: {remaining_dvqa}", flush=True)

        for layer_idx in remaining_dvqa:
            ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
            sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                          device=device, cache_dir=HF_CACHE)
            sae.eval()
            fire_count = np.zeros(D_SAE, dtype=np.int64)
            n_samples = n_failed = 0

            for si in range(len(dvqa)):
                ex       = dvqa[si]
                question = str(ex.get("question", "")).strip()
                img      = ex.get("image")
                if img is None or not question:
                    n_failed += 1; continue
                try:
                    img    = img.convert("RGB")
                    prompt = f"answer en {question}"
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        img, prompt, processor, model_raw, device=device)
                    img_s, img_e = get_image_token_positions(input_ids)

                    with torch.no_grad():
                        with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                              pixel_values=pixel_values, use_cache=False) as _:
                            layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()
                    act = layer_out.detach().squeeze(0).float()
                    with torch.no_grad():
                        codes = sae.encode(act).detach()

                    seq_len  = codes.shape[0]
                    txt_mask = torch.ones(seq_len, dtype=torch.bool, device=codes.device)
                    if img_e > img_s:
                        txt_mask[img_s:img_e] = False
                    txt_codes = codes[txt_mask]
                    if txt_codes.shape[0] > 0:
                        fired = (txt_codes > 0).any(dim=0).cpu().numpy()
                        fire_count += fired.astype(np.int64)
                    n_samples += 1
                except Exception as e:
                    n_failed += 1
                    if n_failed <= 3:
                        print(f"  [GPU{gpu_id}] DocVQA err L{layer_idx} si={si}: {e}", flush=True)
                    continue
                if (si + 1) % 500 == 0:
                    print(f"  [GPU{gpu_id}] L{layer_idx} DocVQA: {si+1}/{len(dvqa)}", flush=True)

            with open(dvqa_dir / f"firing_docvqa_layer_{layer_idx}.json", "w") as f:
                json.dump({"layer": int(layer_idx), "n_samples": int(n_samples),
                           "n_failed": int(n_failed), "dataset": "docvqa",
                           "fire_count_all": fire_count.tolist()}, f)
            print(f"[Firing GPU{gpu_id}] DocVQA L{layer_idx}: {n_samples} samples, "
                  f"{n_failed} failed. >10%: {(fire_count > n_samples * 0.1).sum()}", flush=True)
            del sae; torch.cuda.empty_cache(); gc.collect()

    del nns_model, model_raw, processor
    torch.cuda.empty_cache(); gc.collect()


def step5_firing(n_gpus=N_GPUS):
    print("\n" + "=" * 60)
    print(f"[Step 5] Firing: VQAv2 {N_FIRING_SAMPLES} + DocVQA val all 5349")
    print("=" * 60)

    # Write split indices
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    dvqa = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    write_splits(len(dvqa))
    del dvqa

    layers_per_worker = math.ceil(N_LAYERS / n_gpus)
    assignments = []
    for w in range(n_gpus):
        start = w * layers_per_worker
        end   = min(start + layers_per_worker, N_LAYERS)
        layers = list(range(start, end))
        if layers:
            assignments.append((w, layers))

    for g, layers in assignments:
        print(f"  GPU {g}: layers {layers}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for gpu_id, layers in assignments:
        p = mp.Process(target=_firing_worker, args=(gpu_id, layers))
        p.start(); processes.append(p)
    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        print(f"[ERROR] Workers {failed} failed!")
    else:
        print("[Step 5] Done.")


# ─────────────────────────── Step 6 Fisher ───────────────────────────

def step6_fisher():
    print("\n" + "=" * 60)
    print("[Step 6] Fisher Test — VQAv2 vs DocVQA (all 5349)")
    print("=" * 60)

    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests
    import pandas as pd

    vqa_dir  = ANALYSIS_DIR / "firing_vqa"
    dvqa_dir = ANALYSIS_DIR / "firing_docvqa"
    out_dir  = ANALYSIS_DIR / "docvqa_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer_idx in range(N_LAYERS):
        vqa_path  = vqa_dir  / f"firing_vqa_layer_{layer_idx}.json"
        dvqa_path = dvqa_dir / f"firing_docvqa_layer_{layer_idx}.json"
        if not vqa_path.exists() or not dvqa_path.exists():
            print(f"  L{layer_idx}: SKIP (missing)"); continue

        vqa_data  = json.load(open(vqa_path))
        dvqa_data = json.load(open(dvqa_path))
        n_vqa     = vqa_data["n_samples"]
        n_dvqa    = dvqa_data["n_samples"]
        fire_vqa  = np.array(vqa_data["fire_count_all"])
        fire_dvqa = np.array(dvqa_data["fire_count_all"])
        if n_vqa == 0 or n_dvqa == 0: continue

        print(f"  L{layer_idx}: n_vqa={n_vqa}, n_docvqa={n_dvqa}", flush=True)
        for fi in range(D_SAE):
            c_vqa  = int(fire_vqa[fi])
            c_dvqa = int(fire_dvqa[fi])
            if c_vqa == 0 and c_dvqa == 0: continue
            rows.append({"layer": layer_idx, "feature": fi,
                         "c_vqa": c_vqa, "n_vqa": n_vqa,
                         "c_docvqa": c_dvqa, "n_docvqa": n_dvqa})

    if not rows:
        print("  No features with nonzero firing"); return {"total_docvqa": 0}

    df = pd.DataFrame(rows)
    print(f"  {len(df)} features tested across {df['layer'].nunique()} layers")

    pvals, odds = [], []
    for _, r in df.iterrows():
        c_d  = min(int(r.c_docvqa), int(r.n_docvqa))
        c_v  = min(int(r.c_vqa), int(r.n_vqa))
        table = [[c_d, max(0, int(r.n_docvqa) - c_d)],
                 [c_v, max(0, int(r.n_vqa)   - c_v)]]
        try:
            o, p = fisher_exact(table, alternative="greater")
            odds.append(o if not math.isinf(o) else 1e9)
            pvals.append(p)
        except ValueError:
            odds.append(1.0); pvals.append(1.0)

    df["odds_ratio"] = odds
    df["p_raw"]      = pvals
    df["freq_docvqa"] = df.c_docvqa / df.n_docvqa
    df["freq_vqa"]    = df.c_vqa    / df.n_vqa
    df["freq_diff"]   = df.freq_docvqa - df.freq_vqa
    df["p_adj"]       = multipletests(df.p_raw, method="fdr_bh")[1]

    keep    = (df.odds_ratio >= ODDS_THR) & (df.freq_diff >= MIN_FREQ_DIFF)
    spatial = df.loc[keep].sort_values("odds_ratio", ascending=False).copy()
    print(f"  DocVQA-enriched features: {len(spatial)}")
    for layer_idx in sorted(spatial["layer"].unique()):
        n = (spatial["layer"] == layer_idx).sum()
        print(f"    L{layer_idx}: {n} features")

    spatial.to_csv(out_dir / "docvqa_features.csv", index=False)
    df.to_csv(out_dir / "all_features_stats.csv", index=False)

    summary = {"total_docvqa": len(spatial), "odds_threshold": ODDS_THR,
               "min_freq_diff": MIN_FREQ_DIFF, "total_tested": len(df)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_dir}/docvqa_features.csv")
    return summary


# ─────────────────────────── Step 7 Lexical ───────────────────────────

def _lexical_worker(gpu_id, feature_assignments):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset

    out_dir = ANALYSIS_DIR / "lexical"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return

    print(f"[Lexical GPU{gpu_id}] {len(feature_assignments)} features", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_raw)

    dvqa = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

    layer_features = defaultdict(list)
    for (layer_idx, feature_idx) in feature_assignments:
        layer_features[layer_idx].append(feature_idx)

    results = []
    top_k              = 5
    scan_samples       = 500
    activation_threshold = 0.01

    for layer_idx in sorted(layer_features.keys()):
        features  = layer_features[layer_idx]
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()

        feature_cols = {f: [] for f in features}
        scan_count   = min(scan_samples, len(dvqa))

        for scan_i in range(scan_count):
            try:
                ex    = dvqa[scan_i]
                image = ex["image"].convert("RGB")
                prompt = f"answer en {ex['question']}"
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    image, prompt, processor, model_raw, device=device)
                with torch.no_grad():
                    with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                          pixel_values=pixel_values, use_cache=False) as _:
                        layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()
                act = layer_out.detach().squeeze(0).float()
                with torch.no_grad():
                    codes = sae.encode(act.to(device)).detach().cpu()
                for f in features:
                    max_act = float(codes[:, f].max().item())
                    if max_act > 0:
                        feature_cols[f].append((max_act, scan_i))
            except Exception:
                continue

        for feature_idx in features:
            candidates = sorted(feature_cols[feature_idx], key=lambda x: -x[0])[:top_k]
            if not candidates:
                results.append({"layer": layer_idx, "feature": feature_idx,
                                 "passed": False, "n_tested": 0}); continue

            passed = True; n_tested = 0
            for (mag, dvqa_idx) in candidates:
                try:
                    ex    = dvqa[dvqa_idx]
                    image = ex["image"].convert("RGB")
                    best_generic_max = 0.0
                    for raw_prompt in GENERIC_PROMPTS:
                        prompt = f"answer en {raw_prompt}"
                        input_ids, attn_mask, pixel_values = process_vlm_inputs(
                            image, prompt, processor, model_raw, device=device)
                        with torch.no_grad():
                            with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                                  pixel_values=pixel_values, use_cache=False) as _:
                                layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()
                        act = layer_out.detach().squeeze(0).float()
                        with torch.no_grad():
                            codes = sae.encode(act.to(device)).detach().cpu()
                        best_generic_max = max(best_generic_max,
                                               float(codes[:, feature_idx].max().item()))
                    n_tested += 1
                    if best_generic_max <= activation_threshold:
                        passed = False; break
                except Exception:
                    continue

            results.append({"layer": layer_idx, "feature": feature_idx,
                             "passed": passed, "n_tested": n_tested})
            print(f"[Lexical GPU{gpu_id}] L{layer_idx} F{feature_idx}: "
                  f"{'PASS' if passed else 'FAIL'}", flush=True)

        del sae; torch.cuda.empty_cache()

    out_path = out_dir / f"lexical_results_w{gpu_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"[Lexical GPU{gpu_id}] {passed_count}/{len(results)} passed", flush=True)


def step7_lexical(n_gpus=N_GPUS):
    print("\n" + "=" * 60)
    print("[Step 7] Lexical Artifact Filtering")
    print("=" * 60)

    import pandas as pd
    csv_path = ANALYSIS_DIR / "docvqa_features" / "docvqa_features.csv"
    if not csv_path.exists():
        print("  No DocVQA features found. Run step 6 first."); return

    df         = pd.read_csv(csv_path)
    candidates = [(int(r["layer"]), int(r["feature"])) for _, r in df.iterrows()]
    print(f"  {len(candidates)} candidates")
    if not candidates: return

    (ANALYSIS_DIR / "lexical").mkdir(parents=True, exist_ok=True)
    features_per_worker = math.ceil(len(candidates) / n_gpus)
    assignments = []
    for w in range(n_gpus):
        start = w * features_per_worker
        end   = min(start + features_per_worker, len(candidates))
        if candidates[start:end]:
            assignments.append((w, candidates[start:end]))

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for gpu_id, feats in assignments:
        p = mp.Process(target=_lexical_worker, args=(gpu_id, feats))
        p.start(); processes.append(p)
    for p in processes:
        p.join()


# ─────────────────────────── Step 8 Intersection ───────────────────────────

def step8_intersection():
    print("\n" + "=" * 60)
    print("[Step 8] Intersection (adapted ∩ DocVQA-enriched ∩ lexical)")
    print("=" * 60)

    import pandas as pd
    out_dir = ANALYSIS_DIR / "final_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Adapted features — reuse from OCR pipeline (same SAEs, same model)
    adapted_by_layer = {}
    adapted_path = ANALYSIS_OCR_DIR / "adapted" / "adapted_features_results.csv"
    if adapted_path.exists():
        with open(adapted_path) as f:
            for row in csv.DictReader(f):
                layer = int(row["layer"])
                adapted_by_layer[layer] = set(ast.literal_eval(row["adapted_indices"]))
    print(f"  Adapted: {sum(len(v) for v in adapted_by_layer.values())} features")

    # DocVQA-enriched
    spatial_by_layer = {}
    spatial_path = ANALYSIS_DIR / "docvqa_features" / "docvqa_features.csv"
    if spatial_path.exists():
        df = pd.read_csv(spatial_path)
        for layer in df["layer"].unique():
            spatial_by_layer[int(layer)] = set(df[df["layer"] == layer]["feature"].tolist())
    print(f"  DocVQA-enriched: {sum(len(v) for v in spatial_by_layer.values())} features")

    # Lexical passed
    lexical_passed = {}
    lexical_dir = ANALYSIS_DIR / "lexical"
    if lexical_dir.exists():
        for fp in sorted(lexical_dir.glob("lexical_results_w*.json")):
            with open(fp) as f:
                for r in json.load(f):
                    if r["passed"]:
                        lexical_passed.setdefault(r["layer"], set()).add(r["feature"])
    print(f"  Lexical passed: {sum(len(v) for v in lexical_passed.values())} features")

    # Intersection
    final_features = []
    for layer in sorted(set(adapted_by_layer.keys()) | set(spatial_by_layer.keys())):
        adapted = adapted_by_layer.get(layer, set())
        spatial = spatial_by_layer.get(layer, set())
        common  = adapted & spatial
        if lexical_passed:
            common = common & lexical_passed.get(layer, set())
        for fi in sorted(common):
            final_features.append({"layer": layer, "feature": fi})
        if common:
            print(f"  L{layer}: adapted={len(adapted)}, docvqa={len(spatial)}, "
                  f"lexical={len(lexical_passed.get(layer, set()))}, final={len(common)}")

    if final_features:
        df_out = pd.DataFrame(final_features)
        # Merge odds_ratio from DocVQA features for ablation ranking
        if spatial_path.exists():
            df_spatial = pd.read_csv(spatial_path)[["layer", "feature", "odds_ratio"]]
            df_out = df_out.merge(df_spatial, on=["layer", "feature"], how="left")
        df_out.to_csv(out_dir / "final_docvqa_features.csv", index=False)

    summary = {"total_final": len(final_features),
               "total_adapted": sum(len(v) for v in adapted_by_layer.values()),
               "total_docvqa":  sum(len(v) for v in spatial_by_layer.values()),
               "total_lexical": sum(len(v) for v in lexical_passed.values())}
    with open(out_dir / "intersection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Final DocVQA features: {len(final_features)}")
    return summary


# ─────────────────────────── Main ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--gpus", type=int, default=N_GPUS)
    args = ap.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if 5 in args.step: step5_firing(n_gpus=args.gpus)
    if 6 in args.step: step6_fisher()
    if 7 in args.step: step7_lexical(n_gpus=args.gpus)
    if 8 in args.step: step8_intersection()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"[DONE] DocVQA pipeline complete in {elapsed/3600:.1f}h")
    print(f"  Results: {ANALYSIS_DIR}")
    print(f"  Splits: {ANALYSIS_DIR}/splits.json")
    print(f"  Final features: {ANALYSIS_DIR}/final_features/final_docvqa_features.csv")


if __name__ == "__main__":
    main()
