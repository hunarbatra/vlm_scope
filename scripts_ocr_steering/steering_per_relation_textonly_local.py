#!/usr/bin/env python3
"""
Per-relation VSR feature steering for PaliGemma2 spatial features.
Mirror of ablation: adds alpha * feature_direction instead of projecting it out.

For each top-N feature (by selectivity from final ablation CSV), sweeps
alpha in STEERING_ALPHAS = [-20, -10, -5, +5, +10, +20] and records
∆VSR, ∆Ctrl, ∆VQA at each alpha. Negative alpha = suppress (should mirror
ablation). Positive alpha = amplify (should boost spatial accuracy).

Input: final ablation_summary.csv (has layer, feature, relations, delta_vsr, delta_ctrl).
Usage:
    python3 steering_per_relation_textonly_local.py \
        --ablation-csv /path/to/ablation_summary.csv \
        --top-n 20 \
        --gpus 0 1 2 3 4 5 6 7
"""

import os
import sys
import json
import math
import csv
import hashlib
import argparse
import warnings
import gc
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# ---- Config ----
MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384
CHECKPOINT_DIR = Path(os.environ.get("VLMSCOPE_CKPT_DIR", "/data1/vlm_scope_sae_mix448_textonly/checkpoints"))
ANALYSIS_DIR = Path(os.environ.get("VLMSCOPE_ANALYSIS_DIR", "/data1/vlm_scope_sae_mix448_textonly/analysis"))
HF_CACHE = "/data1/hf_cache/hub"
IMAGE_CACHE_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

VSR_DATASET = "cambridgeltl/vsr_random"
VQA_MAX = 1000
CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}
STEERING_ALPHAS = [-20.0, -10.0, -5.0, 5.0, 10.0, 20.0]  # negative = suppress, positive = amplify


def _build_vsr_prompt(statement: str) -> str:
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\n"
        "Answer:"
    )


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES", " true", "true", "True"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO", " false", "false", "False"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids -= overlap
    return yes_ids, no_ids


def _predict_yesno(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


def _ablation_worker(gpu_id, feature_assignments, out_dir, vqa_max):
    """Per-relation ablation worker."""
    import requests
    from PIL import Image
    from datasets import load_dataset, concatenate_datasets

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return

    print(f"[Ablation GPU{gpu_id}] {len(feature_assignments)} features", flush=True)

    # Load model
    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=False)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=False
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # Load full VSR (all splits, matching original --split train+dev+test)
    print(f"[Ablation GPU{gpu_id}] Loading VSR (all splits)...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    train_ds = load_dataset(VSR_DATASET, data_files=data_files, split="train")
    dev_ds = load_dataset(VSR_DATASET, data_files=data_files, split="dev")
    test_ds = load_dataset(VSR_DATASET, data_files=data_files, split="test")
    vsr_all = concatenate_datasets([train_ds, dev_ds, test_ds])
    print(f"[Ablation GPU{gpu_id}] VSR total: {len(vsr_all)} samples", flush=True)

    # Load VQA yes/no
    print(f"[Ablation GPU{gpu_id}] Loading VQA yes/no...", flush=True)
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno = []
    for i in range(len(vqa)):
        ex = vqa[i]
        at = str(ex.get("answer_type", "")).lower()
        mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
        if at == "yes/no" and mc in {"yes", "no"}:
            vqa_yesno.append((i, 1 if mc == "yes" else 0))
            if len(vqa_yesno) >= vqa_max:
                break
    print(f"[Ablation GPU{gpu_id}] VQA yes/no: {len(vqa_yesno)}", flush=True)

    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_vsr_image(ex):
        url = ex.get("image_link", "")
        if not url.startswith("http"):
            return None
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = IMAGE_CACHE_DIR / f"{url_hash}.jpg"
        try:
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(cache_path, "JPEG")
            return img
        except Exception:
            return None

    def _do_steering_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end, alpha):
        """Add alpha * feature_vec to all 3 taps across 26 layers, text tokens only.
        Steering mirror of 3-point ablation — amplify instead of project out.
        """
        add = alpha * feature_vec  # (d_in,)
        with nns_model.trace(
            input_ids=input_ids, attention_mask=attn_mask,
            pixel_values=pixel_values,
        ) as tr:
            for l in range(N_LAYERS):
                nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:] += add
                nns_model.model.language_model.layers[l].mlp.output[0, img_end:] += add
                nns_model.model.language_model.layers[l].output[0][0, img_end:] += add
            logits_saved = nns_model.output.logits.save()
        return logits_saved

    def _run_vsr_eval(indices, feature_vec, steered=False, alpha=0.0):
        """Run VSR evaluation on given indices, optionally with ablation."""
        vsr_c = vsr_t = ctrl_c = ctrl_t = 0
        per_relation = defaultdict(lambda: {"correct": 0, "total": 0})

        for vi in indices:
            ex = vsr_all[vi]
            img = _load_vsr_image(ex)
            if img is None:
                continue
            caption = str(ex.get("caption", "")).strip()
            label = int(ex.get("label", 0))
            relation = ex.get("relation", "")
            prompt = _build_vsr_prompt(caption)

            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model_raw if not steered else nns_model._module, device=device)

                if steered:
                    _, img_end = get_image_token_positions(input_ids)
                    logits_saved = _do_steering_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end, alpha)
                    pred = _predict_yesno(logits_saved[0, -1, :], yes_ids, no_ids)
                else:
                    with torch.inference_mode():
                        out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                        pixel_values=pixel_values, use_cache=False)
                    pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0

            vsr_t += 1
            if pred == label:
                vsr_c += 1
            if relation in CONTROL_RELATIONS:
                ctrl_t += 1
                if pred == label:
                    ctrl_c += 1
            per_relation[relation]["total"] += 1
            if pred == label:
                per_relation[relation]["correct"] += 1

        return {
            "vsr_acc": vsr_c / max(vsr_t, 1) * 100,
            "vsr_correct": vsr_c, "vsr_total": vsr_t,
            "ctrl_acc": ctrl_c / max(ctrl_t, 1) * 100,
            "ctrl_correct": ctrl_c, "ctrl_total": ctrl_t,
            "per_relation": dict(per_relation),
        }

    def _run_vqa_eval(feature_vec, steered=False, alpha=0.0):
        """Run VQA yes/no evaluation, optionally with steering."""
        vqa_c = vqa_t = 0
        for qi, label in vqa_yesno:
            ex = vqa[qi]
            img = ex.get("image")
            if img is None:
                continue
            from PIL import Image as PILImage
            if isinstance(img, PILImage.Image):
                img = img.convert("RGB")
            else:
                continue
            question = ex.get("question", "")
            prompt = (
                "Answer the following question with only 'Yes' or 'No':\n"
                f"Question: {question.strip()}\n"
                "Answer:"
            )
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model_raw if not steered else nns_model._module, device=device)
                if steered:
                    _, img_end = get_image_token_positions(input_ids)
                    logits_saved = _do_steering_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end, alpha)
                    pred = _predict_yesno(logits_saved[0, -1, :], yes_ids, no_ids)
                else:
                    with torch.inference_mode():
                        out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                        pixel_values=pixel_values, use_cache=False)
                    pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            vqa_t += 1
            if pred == label:
                vqa_c += 1

        return {
            "vqa_acc": vqa_c / max(vqa_t, 1) * 100,
            "vqa_correct": vqa_c, "vqa_total": vqa_t,
        }

    # --- Compute baselines ---
    # Baseline for VQA (shared across all features)
    vqa_baseline_path = out_dir / f"vqa_baseline_gpu{gpu_id}.json"
    if vqa_baseline_path.exists():
        with open(vqa_baseline_path) as f:
            vqa_baseline = json.load(f)
        print(f"[Ablation GPU{gpu_id}] Loaded cached VQA baseline: {vqa_baseline['vqa_acc']:.1f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing VQA baseline...", flush=True)
        vqa_baseline = _run_vqa_eval(None, steered=False)
        with open(vqa_baseline_path, "w") as f:
            json.dump(vqa_baseline, f, indent=2)
        print(f"[Ablation GPU{gpu_id}] VQA baseline: {vqa_baseline['vqa_acc']:.1f}%", flush=True)

    # Pre-compute indices by relation for fast filtering
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        rel = vsr_all[vi].get("relation", "")
        relation_indices[rel].append(vi)
    print(f"[Ablation GPU{gpu_id}] Pre-indexed {len(relation_indices)} relations", flush=True)

    # Cache baselines per relation set (hash -> baseline)
    baseline_cache = {}

    # --- Per-feature steering (alpha sweep) ---
    for feat_i, (layer_idx, feature_idx, relations_str, odds_ratio) in enumerate(feature_assignments):
        result_path = out_dir / f"steering_L{layer_idx}_F{feature_idx}.json"
        if result_path.exists():
            print(f"[Steering GPU{gpu_id}] L{layer_idx}/F{feature_idx}: already done, skip", flush=True)
            continue

        if not relations_str:
            continue

        relations = [r.strip() for r in relations_str.split(";") if r.strip()]
        if not relations:
            continue

        filtered_indices = []
        for rel in relations:
            filtered_indices.extend(relation_indices.get(rel, []))
        if not filtered_indices:
            continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            baseline_vsr = _run_vsr_eval(filtered_indices, None, steered=False)
            baseline_cache[rel_key] = baseline_vsr
            print(f"  [Baseline GPU{gpu_id}] VSR baseline for [{rel_key}] ({len(filtered_indices)} samples): "
                  f"{baseline_vsr['vsr_acc']:.1f}%", flush=True)
        baseline_vsr = baseline_cache[rel_key]

        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()
        feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
        del sae
        torch.cuda.empty_cache()

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "relations": relations,
            "odds_ratio": odds_ratio,
            "n_vsr_samples": len(filtered_indices),
            "baseline_vsr_acc": baseline_vsr["vsr_acc"],
            "baseline_ctrl_acc": baseline_vsr["ctrl_acc"],
            "baseline_vqa_acc": vqa_baseline["vqa_acc"],
            "per_relation_baseline": baseline_vsr.get("per_relation", {}),
            "alphas": {},
        }

        for alpha in STEERING_ALPHAS:
            print(f"[Steering GPU{gpu_id}] [{feat_i+1}/{len(feature_assignments)}] "
                  f"L{layer_idx}/F{feature_idx} α={alpha:+g} relations=[{rel_key}] ({len(filtered_indices)} samp)", flush=True)
            st_vsr = _run_vsr_eval(filtered_indices, feature_vec, steered=True, alpha=alpha)
            st_vqa = _run_vqa_eval(feature_vec, steered=True, alpha=alpha)
            result["alphas"][str(alpha)] = {
                "steered_vsr_acc": st_vsr["vsr_acc"],
                "steered_ctrl_acc": st_vsr["ctrl_acc"],
                "steered_vqa_acc": st_vqa["vqa_acc"],
                "delta_vsr": st_vsr["vsr_acc"] - baseline_vsr["vsr_acc"],
                "delta_ctrl": st_vsr["ctrl_acc"] - baseline_vsr["ctrl_acc"],
                "delta_vqa": st_vqa["vqa_acc"] - vqa_baseline["vqa_acc"],
                "per_relation_steered": st_vsr.get("per_relation", {}),
            }
            print(f"  α={alpha:+g}: ∆VSR={result['alphas'][str(alpha)]['delta_vsr']:+.2f}% "
                  f"∆Ctrl={result['alphas'][str(alpha)]['delta_ctrl']:+.2f}% "
                  f"∆VQA={result['alphas'][str(alpha)]['delta_vqa']:+.2f}%", flush=True)

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"[Ablation GPU{gpu_id}] All features done.", flush=True)


def main():
    global CHECKPOINT_DIR, ANALYSIS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", type=str,
                        default=str(ANALYSIS_DIR / "ablation_per_relation_full" / "ablation_summary.csv"),
                        help="Final VSR ablation_summary.csv (has layer, feature, relations, delta_vsr, delta_ctrl)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top-N features by selectivity (delta_vsr - delta_ctrl)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--vqa-max", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--analysis-dir", type=str, default=str(ANALYSIS_DIR))
    args = parser.parse_args()
    os.environ["VLMSCOPE_CKPT_DIR"] = args.checkpoint_dir
    os.environ["VLMSCOPE_ANALYSIS_DIR"] = args.analysis_dir
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    ANALYSIS_DIR = Path(args.analysis_dir)
    print(f"[Config] CHECKPOINT_DIR={CHECKPOINT_DIR}")
    print(f"[Config] ANALYSIS_DIR={ANALYSIS_DIR}")

    # Load top-N features by selectivity from final ablation CSV
    rows = []
    with open(args.ablation_csv) as f:
        for row in csv.DictReader(f):
            row["sel"] = float(row["delta_vsr"]) - float(row["delta_ctrl"])
            row["layer"] = int(row["layer"])
            row["feature"] = int(row["feature"])
            rows.append(row)
    rows.sort(key=lambda r: r["sel"])
    top_rows = rows[:args.top_n]
    features = [
        (r["layer"], r["feature"], r["relations"], float(r["odds_ratio"]))
        for r in top_rows if r.get("relations", "").strip()
    ]
    print(f"Loaded {len(features)} features (top-{args.top_n} by selectivity)")
    print(f"\nFeatures to steer:")
    for l, f, rels, or_ in features:
        sel = next(r["sel"] for r in top_rows if r["layer"] == l and r["feature"] == f)
        print(f"  L{l}/F{f}: sel={sel:+.2f}  OR={or_:.1f}  rels={rels[:60]}")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / "steering_per_relation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distribute across GPUs (round-robin for even load)
    n_gpus = len(args.gpus)
    chunks = [[] for _ in range(n_gpus)]
    for i, feat in enumerate(features):
        chunks[i % n_gpus].append(feat)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for i, gpu_id in enumerate(args.gpus):
        if chunks[i]:
            print(f"  GPU {gpu_id}: {len(chunks[i])} features")
            p = mp.Process(target=_ablation_worker,
                           args=(gpu_id, chunks[i], str(out_dir), args.vqa_max))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

    # Collect results into summary CSV (one row per feature × alpha)
    all_results = []
    for p in sorted(Path(out_dir).glob("steering_L*_F*.json")):
        with open(p) as f:
            all_results.append(json.load(f))

    if all_results:
        summary_path = out_dir / "steering_summary.csv"
        alphas = STEERING_ALPHAS
        fieldnames = ["layer", "feature", "relations", "odds_ratio", "n_vsr_samples",
                      "baseline_vsr_acc", "baseline_ctrl_acc", "baseline_vqa_acc"]
        for a in alphas:
            fieldnames += [f"delta_vsr_{a:+g}", f"delta_ctrl_{a:+g}", f"delta_vqa_{a:+g}"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                row = {
                    "layer": r["layer"], "feature": r["feature"],
                    "relations": "; ".join(r.get("relations", [])),
                    "odds_ratio": f"{r.get('odds_ratio', 0):.2f}",
                    "n_vsr_samples": r.get("n_vsr_samples", 0),
                    "baseline_vsr_acc": f"{r.get('baseline_vsr_acc', 0):.2f}",
                    "baseline_ctrl_acc": f"{r.get('baseline_ctrl_acc', 0):.2f}",
                    "baseline_vqa_acc": f"{r.get('baseline_vqa_acc', 0):.2f}",
                }
                for a in alphas:
                    adat = r.get("alphas", {}).get(str(a), {})
                    row[f"delta_vsr_{a:+g}"] = f"{adat.get('delta_vsr', 0):.2f}"
                    row[f"delta_ctrl_{a:+g}"] = f"{adat.get('delta_ctrl', 0):.2f}"
                    row[f"delta_vqa_{a:+g}"] = f"{adat.get('delta_vqa', 0):.2f}"
                writer.writerow(row)

        print(f"\nSteering summary: {summary_path} ({len(all_results)} features)")
        print(f"Alphas tested: {alphas}")
        for r in sorted(all_results,
                        key=lambda x: x.get("alphas", {}).get("20.0", {}).get("delta_vsr", 0),
                        reverse=True)[:10]:
            rels = "; ".join(r.get("relations", []))[:40]
            a20 = r.get("alphas", {}).get("20.0", {})
            print(f"  L{r['layer']:>2}/F{r['feature']:>5} ∆VSR@α=+20: {a20.get('delta_vsr',0):+.2f}%  "
                  f"∆Ctrl: {a20.get('delta_ctrl',0):+.2f}%  {rels}")
        print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
