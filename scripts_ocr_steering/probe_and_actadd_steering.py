#!/usr/bin/env python3
"""
Probe + Activation-Addition Steering for spatial SAE features.

Phase 1 — Collect (GPU, distributed):
  For each assigned layer, run all VSR samples through model+SAE.
  Save per-sample: SAE feature activations (mean over text tokens) +
  residual mean vector + correct/incorrect label.
  Output: probe_layer_{L}.npz with keys:
    activations: (N, D_SAE) float16  — mean SAE acts per sample
    residuals:   (N, D_MODEL) float16 — mean residual per sample
    correct:     (N,) bool            — baseline correct?
    relation:    (N,) str array       — VSR relation label

Phase 2 — Probe (CPU):
  For each layer, train logistic regression: SAE activations -> correct/incorrect.
  Report per-layer AUC and accuracy vs. chance. Also per-relation breakdowns.
  Output: probe_results.json

Phase 3 — Activation Addition (GPU):
  Compute correctness direction: mean(residuals[correct]) - mean(residuals[~correct])
  per layer. Sweep injection alphas. Measure VSR + ctrl accuracy delta.
  This is the Turner et al. 2023 approach — injects a NATURAL residual direction
  rather than a sparse SAE decoder vector.
  Output: actadd_results.json

Usage:
  python3 probe_and_actadd_steering.py --phase 1 --out-dir /path/to/out --n-gpus 8
  python3 probe_and_actadd_steering.py --phase 2 --out-dir /path/to/out
  python3 probe_and_actadd_steering.py --phase 3 --out-dir /path/to/out --n-gpus 8
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", message=".*PaliGemma.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

sys.path.insert(0, str(Path(__file__).parent))
from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

MODEL_NAME   = os.environ.get("MODEL_PATH", "google/paligemma2-3b-mix-448")
N_LAYERS     = 26
D_SAE        = 16384
D_MODEL      = 2304
VSR_DATASET  = "cambridgeltl/vsr_random"

CHECKPOINT_DIR  = Path(os.environ.get("VLMSCOPE_CKPT_DIR",
                        "/data1/vlm_scope_sae_mix448_textonly/checkpoints"))
HF_CACHE        = os.environ.get("HF_HOME", "/data1/hf_cache")
IMAGE_CACHE_DIR = Path(os.environ.get("VSR_IMAGE_CACHE",
                        "/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache"))

CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}

# Activation-addition alphas to sweep
ACTADD_ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Top features from ablation_summary (layer, feat_idx) — used for SAE probe
TOP_ABLATION_FEATURES = [
    (9, 387), (8, 16146), (14, 10561), (10, 14319), (12, 11550),
    (25, 6552), (11, 12278), (5, 14464), (15, 8643), (23, 12652),
    (9, 7540),  (7, 13215), (4, 14233), (10, 8858), (6, 7539),
    (12, 686),  (11, 9639), (7, 8291),  (5, 4658),  (13, 15219),
]


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_vsr_prompt(statement: str) -> str:
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\n"
        "Answer:"
    )


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap; no_ids -= overlap
    return yes_ids, no_ids


def _load_vsr(splits=("train", "dev", "test")):
    import requests
    from datasets import Dataset, concatenate_datasets
    from PIL import Image as PILImage

    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Load from arrow files directly — avoids cache_dir key mismatch issues
    arrow_root = Path(os.environ.get("HF_DATASETS_CACHE",
                                      "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"))
    # Find the arrow dir
    arrow_dir = None
    for candidate in sorted((arrow_root / "cambridgeltl___vsr_random").rglob("*.arrow"),
                             key=lambda p: p.stat().st_size, reverse=True):
        arrow_dir = candidate.parent
        break
    if arrow_dir is None:
        raise RuntimeError(f"No VSR arrow files found under {arrow_root}")

    ds_list = []
    for split in splits:
        arrow_file = arrow_dir / f"vsr_random-{split}.arrow"
        if not arrow_file.exists():
            print(f"[WARN] Arrow file not found: {arrow_file}")
            continue
        try:
            ds_list.append(Dataset.from_file(str(arrow_file)))
        except Exception as e:
            print(f"[WARN] Could not load VSR split {split}: {e}")
    if not ds_list:
        raise RuntimeError(f"No VSR splits loaded from {arrow_dir}")

    vsr = concatenate_datasets(ds_list)
    samples = []
    for item in vsr:
        img_url  = item.get("image_link", "")
        caption  = item.get("caption", "")
        label    = int(item.get("label", 0))
        relation = item.get("relation", "")
        url_hash = hashlib.md5(img_url.encode()).hexdigest()
        cache_path = IMAGE_CACHE_DIR / f"{url_hash}.jpg"
        if not cache_path.exists() and img_url.startswith("http"):
            try:
                r = requests.get(img_url, timeout=10)
                r.raise_for_status()
                img = PILImage.open(BytesIO(r.content)).convert("RGB")
                img.save(str(cache_path), "JPEG")
            except Exception:
                pass
        samples.append({"caption": caption, "label": label,
                         "relation": relation, "img_path": str(cache_path)})
    return samples


def _load_model_and_tok():
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    print(f"[INFO] Loading model {MODEL_NAME}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE,
                                              local_files_only=True)
    print(f"[INFO] Processor loaded.", flush=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE,
        local_files_only=True
    )
    print(f"[INFO] Model loaded.", flush=True)
    return model, processor



# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Collect per-sample activations
# ──────────────────────────────────────────────────────────────────────────────

def _phase1_worker(gpu_id: int, layers: list, out_dir: Path, n_gpus: int):
    import traceback
    try:
        _phase1_worker_inner(gpu_id, layers, out_dir, n_gpus)
    except Exception as e:
        print(f"[Phase1 GPU{gpu_id}] FATAL: {e}", flush=True)
        traceback.print_exc()


def _phase1_worker_inner(gpu_id: int, layers: list, out_dir: Path, n_gpus: int):
    import nnsight
    from nnsight import NNsight
    from PIL import Image
    import warnings
    warnings.filterwarnings("ignore")

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    print(f"[Phase1 GPU{gpu_id}] layers={layers}", flush=True)

    model_raw, processor = _load_model_and_tok()
    print(f"[Phase1 GPU{gpu_id}] Moving model to {device}...", flush=True)
    model_raw = model_raw.to(device).eval()
    print(f"[Phase1 GPU{gpu_id}] Model on device.", flush=True)
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)
    print(f"[Phase1 GPU{gpu_id}] yes_ids={yes_ids}, no_ids={no_ids}", flush=True)

    samples_file = out_dir / "vsr_samples.json"
    with open(samples_file) as f:
        vsr_all = json.load(f)
    print(f"[Phase1 GPU{gpu_id}] {len(vsr_all)} VSR samples", flush=True)

    print(f"[Phase1 GPU{gpu_id}] Ready.", flush=True)

    for layer_idx in layers:
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"[Phase1 GPU{gpu_id}] MISSING ckpt {ckpt_path}, skip")
            continue

        out_file = out_dir / f"probe_layer_{layer_idx}.npz"
        if out_file.exists():
            print(f"[Phase1 GPU{gpu_id}] L{layer_idx} already done, skip")
            continue

        # Load SAE on CPU to avoid OOM (model already uses ~21GB GPU)
        sae = initialize_jumprelu_sae(layer_idx=layer_idx,
                                      checkpoint_path=str(ckpt_path),
                                      device="cpu")
        sae.eval()

        # Reinitialize NNsight per layer — reusing across layers causes hangs
        nns_model = NNsight(model_raw)
        print(f"[Phase1 GPU{gpu_id}] L{layer_idx} NNsight initialized.", flush=True)

        all_acts     = []  # (N, D_SAE) float16
        all_residuals= []  # (N, D_MODEL) float16
        all_correct  = []  # bool
        all_relations= []  # str

        for vi, sample in enumerate(vsr_all):
            if vi % 500 == 0:
                print(f"[Phase1 GPU{gpu_id}] L{layer_idx}: {vi}/{len(vsr_all)}")
            try:
                img_path = sample["img_path"]
                if not Path(img_path).exists():
                    continue
                image = Image.open(img_path).convert("RGB")
                prompt = _build_vsr_prompt(sample["caption"])
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    image, prompt, processor, model_raw, device=device)
                _, img_end = get_image_token_positions(input_ids)

                with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                     pixel_values=pixel_values):
                    residual_saved = nns_model.model.language_model.layers[layer_idx].output[0][0, img_end:].save()
                    logits_saved   = nns_model.output.logits.save()

                residual = residual_saved  # (T_text, D_MODEL)
                logits   = logits_saved    # (1, T_total, V)

                # Baseline prediction
                last_logits = logits[0, -1]
                yes_score = max(last_logits[yid].item() for yid in yes_ids)
                no_score  = max(last_logits[nid].item() for nid in no_ids)
                pred_yes  = yes_score > no_score
                gt_yes    = bool(sample["label"])
                correct   = (pred_yes == gt_yes)

                # SAE encode residual: mean over text tokens (SAE on CPU)
                with torch.no_grad():
                    res_cpu = residual.float().cpu()  # (T, D_MODEL) on CPU
                    acts = sae.encode(res_cpu)        # (T, D_SAE) on CPU
                    mean_acts = acts.mean(dim=0).half()     # (D_SAE,)
                    mean_res  = res_cpu.mean(dim=0).half()  # (D_MODEL,)

                all_acts.append(mean_acts.numpy())
                all_residuals.append(mean_res.numpy())
                all_correct.append(correct)
                all_relations.append(sample["relation"])

            except Exception as e:
                continue

        if not all_acts:
            print(f"[Phase1 GPU{gpu_id}] L{layer_idx}: no samples collected, skip")
            continue

        acts_arr = np.stack(all_acts).astype(np.float16)
        res_arr  = np.stack(all_residuals).astype(np.float16)
        corr_arr = np.array(all_correct, dtype=bool)
        rel_arr  = np.array(all_relations, dtype=object)

        np.savez_compressed(str(out_file),
                            activations=acts_arr,
                            residuals=res_arr,
                            correct=corr_arr,
                            relations=rel_arr)
        print(f"[Phase1 GPU{gpu_id}] L{layer_idx} saved: {len(all_acts)} samples, "
              f"acc={corr_arr.mean():.3f}, file={out_file}")

        del sae
        torch.cuda.empty_cache()

    print(f"[Phase1 GPU{gpu_id}] Done.")


def run_phase1(out_dir: Path, n_gpus: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load VSR + build image cache ONCE in main process, save sample list to disk
    print("[Phase1] Pre-loading VSR dataset and image cache...")
    samples = _load_vsr(splits=("train", "dev", "test"))
    # Filter to only samples with cached images
    samples = [s for s in samples if Path(s["img_path"]).exists()]
    print(f"[Phase1] {len(samples)} samples with cached images")
    samples_file = out_dir / "vsr_samples.json"
    with open(samples_file, "w") as f:
        json.dump(samples, f)

    layers_per_gpu = [[] for _ in range(n_gpus)]
    for i in range(N_LAYERS):
        layers_per_gpu[i % n_gpus].append(i)

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id in range(n_gpus):
        if not layers_per_gpu[gpu_id]:
            continue
        p = ctx.Process(target=_phase1_worker,
                        args=(gpu_id, layers_per_gpu[gpu_id], out_dir, n_gpus))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[Phase1] All workers done.")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Linear probe on SAE activations
# ──────────────────────────────────────────────────────────────────────────────

def run_phase2(out_dir: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    results = {}

    # Unique spatial relations across all layers (pool data)
    # Also run per-layer and pooled-all-layers probes

    print("[Phase2] Loading per-layer npz files...")
    layer_data = {}
    for npz_file in sorted(out_dir.glob("probe_layer_*.npz")):
        layer_idx = int(npz_file.stem.split("_")[-1])
        d = np.load(str(npz_file), allow_pickle=True)
        layer_data[layer_idx] = {
            "acts": d["activations"].astype(np.float32),
            "residuals": d["residuals"].astype(np.float32),
            "correct": d["correct"],
            "relations": d["relations"],
        }
        print(f"  L{layer_idx}: {d['activations'].shape[0]} samples, "
              f"acc={d['correct'].mean():.3f}")

    if not layer_data:
        print("[Phase2] No npz files found. Run Phase 1 first.")
        return

    # Per-layer probe: use only top ablation features for that layer
    layer_results = {}
    top_feats_by_layer = defaultdict(list)
    for (l, f) in TOP_ABLATION_FEATURES:
        top_feats_by_layer[l].append(f)

    for layer_idx, data in sorted(layer_data.items()):
        acts    = data["acts"]      # (N, D_SAE)
        correct = data["correct"]   # (N,)
        rels    = data["relations"] # (N,)
        N       = len(correct)
        chance  = correct.mean()

        # Feature subsets to probe:
        # (a) top ablation features at this layer
        # (b) all firing features (top-500 by variance)
        probes = {}

        # (a) ablation features
        abl_feats = top_feats_by_layer.get(layer_idx, [])
        if abl_feats:
            X_abl = acts[:, abl_feats]
            probes["ablation_features"] = X_abl

        # (b) top-500 variance features
        feat_vars = acts.var(axis=0)
        top500_idx = np.argsort(feat_vars)[::-1][:500]
        X_top500 = acts[:, top500_idx]
        probes["top500_var_features"] = X_top500

        # (c) full residual mean
        probes["residual_mean"] = data["residuals"]

        layer_results[layer_idx] = {"n_samples": N, "chance_acc": float(chance)}

        for probe_name, X in probes.items():
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            # 5-fold CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accs, fold_aucs = [], []
            for tr_idx, te_idx in cv.split(X_s, correct):
                clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
                clf.fit(X_s[tr_idx], correct[tr_idx])
                preds = clf.predict(X_s[te_idx])
                probs = clf.predict_proba(X_s[te_idx])[:, 1]
                fold_accs.append(accuracy_score(correct[te_idx], preds))
                try:
                    fold_aucs.append(roc_auc_score(correct[te_idx], probs))
                except Exception:
                    fold_aucs.append(0.5)

            mean_acc = float(np.mean(fold_accs))
            mean_auc = float(np.mean(fold_aucs))
            print(f"  L{layer_idx} [{probe_name}]: acc={mean_acc:.3f} "
                  f"(chance={chance:.3f}, +{mean_acc-chance:+.3f}), AUC={mean_auc:.3f}")
            layer_results[layer_idx][probe_name] = {
                "cv_acc": mean_acc,
                "cv_auc": mean_auc,
                "delta_acc": mean_acc - chance,
            }

        # Per-relation breakdown using ablation features
        if abl_feats and len(set(rels)) > 1:
            per_rel = {}
            X_abl = acts[:, abl_feats]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_abl)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X_s, correct)
            for rel in sorted(set(rels)):
                mask = rels == rel
                if mask.sum() < 10:
                    continue
                rel_acc = accuracy_score(correct[mask], clf.predict(X_s[mask]))
                per_rel[rel] = {
                    "n": int(mask.sum()),
                    "acc": float(rel_acc),
                    "chance": float(correct[mask].mean()),
                }
            layer_results[layer_idx]["per_relation"] = per_rel

    # Pooled probe: stack all layers' top ablation features
    all_acts_list, all_correct_list = [], []
    for layer_idx, data in sorted(layer_data.items()):
        abl_feats = top_feats_by_layer.get(layer_idx, [])
        if abl_feats:
            all_acts_list.append(data["acts"][:, abl_feats])
            all_correct_list.append(data["correct"])

    if all_acts_list:
        # Use first layer's correct labels (all should be same samples same order)
        X_pool = np.hstack(all_acts_list)
        y_pool = all_correct_list[0]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_pool)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        for tr, te in cv.split(X_s, y_pool):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X_s[tr], y_pool[tr])
            fold_accs.append(accuracy_score(y_pool[te], clf.predict(X_s[te])))
        pooled_acc = float(np.mean(fold_accs))
        chance = float(y_pool.mean())
        print(f"\n  POOLED all-layer ablation features: acc={pooled_acc:.3f} "
              f"(chance={chance:.3f}, +{pooled_acc-chance:+.3f})")
        results["pooled_ablation"] = {
            "cv_acc": pooled_acc, "chance_acc": chance,
            "delta_acc": pooled_acc - chance
        }

    results["per_layer"] = {str(k): v for k, v in layer_results.items()}
    out_path = out_dir / "probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Phase2] Saved results to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Activation Addition steering
# ──────────────────────────────────────────────────────────────────────────────

def _phase3_worker(gpu_id: int, layers: list, out_dir: Path,
                   alphas: list, n_gpus: int):
    """
    For each assigned layer:
    1. Load probe_layer_{L}.npz, compute correctness_direction =
       mean(residuals[correct]) - mean(residuals[~correct])
    2. On VSR test split + ctrl, inject alpha * direction at layer L
    3. Measure accuracy delta
    """
    import nnsight
    from nnsight import NNsight
    from PIL import Image
    import warnings
    warnings.filterwarnings("ignore")

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    model_raw, processor = _load_model_and_tok()
    model_raw = model_raw.to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    # Load VSR test — use pre-built samples file if available (avoids concurrent downloads)
    test_samples_file = out_dir / "vsr_test_samples.json"
    if test_samples_file.exists():
        with open(test_samples_file) as f:
            vsr_test = json.load(f)
    else:
        vsr_test = _load_vsr(splits=("test",))
    print(f"[Phase3 GPU{gpu_id}] {len(vsr_test)} VSR test samples")

    nns_model = NNsight(model_raw)
    layer_results = {}

    for layer_idx in layers:
        npz_file = out_dir / f"probe_layer_{layer_idx}.npz"
        if not npz_file.exists():
            print(f"[Phase3 GPU{gpu_id}] L{layer_idx}: no probe npz, skip")
            continue

        # Compute correctness direction from train+dev data
        d = np.load(str(npz_file), allow_pickle=True)
        residuals = d["residuals"].astype(np.float32)  # (N, D_MODEL)
        correct   = d["correct"]                        # (N,) bool

        correct_mean   = residuals[correct].mean(axis=0)    # (D_MODEL,)
        incorrect_mean = residuals[~correct].mean(axis=0)   # (D_MODEL,)
        direction = correct_mean - incorrect_mean            # (D_MODEL,)
        direction_norm = np.linalg.norm(direction)
        direction_unit = direction / (direction_norm + 1e-8)
        direction_t = torch.tensor(direction_unit, dtype=torch.bfloat16, device=device)

        print(f"[Phase3 GPU{gpu_id}] L{layer_idx}: direction norm={direction_norm:.3f}, "
              f"n_correct={correct.sum()}, n_incorrect={(~correct).sum()}")

        alpha_results = {}
        for alpha in [0.0] + alphas:
            n_correct_vsr = 0
            n_total_vsr   = 0

            for vi, sample in enumerate(vsr_test):
                try:
                    img_path = sample["img_path"]
                    if not Path(img_path).exists():
                        continue
                    image = Image.open(img_path).convert("RGB")
                    is_spatial = sample["relation"] not in CONTROL_RELATIONS
                    if not is_spatial:
                        continue

                    prompt = _build_vsr_prompt(sample["caption"])
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        image, prompt, processor, model_raw, device=device)
                    _, img_end = get_image_token_positions(input_ids)

                    if alpha == 0.0:
                        with nns_model.trace(input_ids=input_ids,
                                             attention_mask=attn_mask,
                                             pixel_values=pixel_values):
                            logits_saved = nns_model.output.logits.save()
                    else:
                        with nns_model.trace(input_ids=input_ids,
                                             attention_mask=attn_mask,
                                             pixel_values=pixel_values):
                            nns_model.model.language_model.layers[layer_idx].output[0][0, img_end:] += \
                                alpha * direction_t.unsqueeze(0)
                            logits_saved = nns_model.output.logits.save()

                    logits = logits_saved
                    last_logits = logits[0, -1]
                    yes_score = max(last_logits[yid].item() for yid in yes_ids)
                    no_score  = max(last_logits[nid].item() for nid in no_ids)
                    pred_yes  = yes_score > no_score
                    gt_yes    = bool(sample["label"])
                    n_correct_vsr += int(pred_yes == gt_yes)
                    n_total_vsr   += 1
                except Exception:
                    continue

            if n_total_vsr > 0:
                acc = n_correct_vsr / n_total_vsr
                alpha_results[str(alpha)] = {"acc": acc, "n": n_total_vsr}
                print(f"[Phase3 GPU{gpu_id}] L{layer_idx} alpha={alpha}: "
                      f"acc={acc:.4f} ({n_correct_vsr}/{n_total_vsr})")

        layer_results[layer_idx] = alpha_results
        torch.cuda.empty_cache()

    out_path = out_dir / f"actadd_results_gpu{gpu_id}.json"
    with open(out_path, "w") as f:
        json.dump(layer_results, f, indent=2)
    print(f"[Phase3 GPU{gpu_id}] Saved {out_path}")


def run_phase3(out_dir: Path, n_gpus: int):
    # Only steer layers that have probe data and top ablation features
    top_layers = sorted(set(l for l, _ in TOP_ABLATION_FEATURES))

    layers_per_gpu = [[] for _ in range(n_gpus)]
    for i, l in enumerate(top_layers):
        layers_per_gpu[i % n_gpus].append(l)

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id in range(n_gpus):
        if not layers_per_gpu[gpu_id]:
            continue
        p = ctx.Process(target=_phase3_worker,
                        args=(gpu_id, layers_per_gpu[gpu_id], out_dir,
                              ACTADD_ALPHAS, n_gpus))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Aggregate results
    all_results = {}
    for gpu_id in range(n_gpus):
        rfile = out_dir / f"actadd_results_gpu{gpu_id}.json"
        if rfile.exists():
            with open(rfile) as f:
                all_results.update(json.load(f))

    # Compute deltas vs baseline (alpha=0)
    summary_rows = []
    for layer_idx, alpha_results in sorted(all_results.items(), key=lambda x: int(x[0])):
        baseline_acc = alpha_results.get("0.0", {}).get("acc")
        if baseline_acc is None:
            continue
        row = {"layer": int(layer_idx), "baseline_acc": baseline_acc}
        for alpha in ACTADD_ALPHAS:
            ar = alpha_results.get(str(alpha), {})
            if ar:
                delta = ar["acc"] - baseline_acc
                row[f"alpha_{alpha}_acc"]   = ar["acc"]
                row[f"alpha_{alpha}_delta"] = delta
        summary_rows.append(row)
        print(f"L{layer_idx}: baseline={baseline_acc:.4f}, " +
              " | ".join(f"α={a}: Δ={row.get(f'alpha_{a}_delta', float('nan')):+.4f}"
                         for a in ACTADD_ALPHAS))

    summary = {"per_layer": {str(r["layer"]): r for r in summary_rows},
               "all_results": all_results}
    out_path = out_dir / "actadd_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Phase3] Summary saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--out-dir", type=str,
                        default="/data1/vlm_scope_sae_mix448_textonly/analysis/probe_actadd")
    parser.add_argument("--n-gpus", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.phase == 1:
        run_phase1(out_dir, args.n_gpus)
    elif args.phase == 2:
        run_phase2(out_dir)
    elif args.phase == 3:
        run_phase3(out_dir, args.n_gpus)


if __name__ == "__main__":
    main()
