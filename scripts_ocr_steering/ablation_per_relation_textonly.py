#!/usr/bin/env python3
"""
Per-relation VSR ablation + VQA ablation for PaliGemma2 spatial features.
Matches original ablation pipeline exactly:

1. For each feature, filter VSR to its derived relations (from derive_relations_textonly.py)
2. Run ablation on that subset (3-point projection across all layers)
3. Also run separate VQA yes/no ablation (1000 samples)
4. Report ∆VSR (per-relation), ∆Ctrl, ∆VQA, VSR OR (from spatial features CSV)

Usage:
    python3 ablation_per_relation_textonly.py \
        --features /path/to/final_spatial_visual_features.csv \
        --relations-csv /path/to/feature_relations.csv \
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
CHECKPOINT_DIR = Path("/data1/hbatra/mmdiff/vlm_scope_sae_mix448_textonly/checkpoints")
ANALYSIS_DIR = Path("/data1/hbatra/mmdiff/vlm_scope_sae_mix448_textonly/analysis")
HF_CACHE = "/data1/hbatra/mmdiff/hf_cache/hub"
IMAGE_CACHE_DIR = Path("/data1/hbatra/mmdiff/vlm_scope_sae_mix448_textonly/vsr_image_cache")

os.environ["HF_DATASETS_CACHE"] = "/data1/hbatra/mmdiff/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hbatra/mmdiff/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

VSR_DATASET = "cambridgeltl/vsr_random"
VQA_MAX = 1000
CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}


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

    def _do_ablation_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end):
        """Project out feature direction from all layers on text tokens.
        Matches original: attn output, MLP output, AND residual stream (layer output).
        """
        fv = feature_vec.unsqueeze(0)  # (1, d_in)
        with nns_model.trace(
            input_ids=input_ids, attention_mask=attn_mask,
            pixel_values=pixel_values,
        ) as tr:
            for l in range(N_LAYERS):
                # 1. Self-attention output
                attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                attn_proj = (attn_out @ fv.T) * fv
                attn_out -= attn_proj
                # 2. MLP output
                mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                mlp_proj = (mlp_out @ fv.T) * fv
                mlp_out -= mlp_proj
                # 3. Layer output (residual stream)
                layer_out = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                layer_proj = (layer_out @ fv.T) * fv
                layer_out -= layer_proj
            logits_saved = nns_model.output.logits.save()
        return logits_saved

    def _run_vsr_eval(indices, feature_vec, ablated=False):
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
                    img, prompt, processor, model_raw if not ablated else nns_model._module, device=device)

                if ablated:
                    _, img_end = get_image_token_positions(input_ids)
                    logits_saved = _do_ablation_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end)
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

    def _run_vqa_eval(feature_vec, ablated=False):
        """Run VQA yes/no evaluation, optionally with ablation."""
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
                    img, prompt, processor, model_raw if not ablated else nns_model._module, device=device)
                if ablated:
                    _, img_end = get_image_token_positions(input_ids)
                    logits_saved = _do_ablation_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end)
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
        vqa_baseline = _run_vqa_eval(None, ablated=False)
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

    # --- Per-feature ablation ---
    for feat_i, (layer_idx, feature_idx, relations_str, odds_ratio) in enumerate(feature_assignments):
        result_path = out_dir / f"ablation_L{layer_idx}_F{feature_idx}.json"
        if result_path.exists():
            print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: already done, skip", flush=True)
            continue

        if not relations_str:
            print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: no relations, skip", flush=True)
            continue

        # Parse relations and get filtered indices
        relations = [r.strip() for r in relations_str.split(";") if r.strip()]
        if not relations:
            continue

        filtered_indices = []
        for rel in relations:
            filtered_indices.extend(relation_indices.get(rel, []))

        if not filtered_indices:
            print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: no samples for {relations}, skip", flush=True)
            continue

        # Compute or load baseline for this relation set
        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            baseline_vsr = _run_vsr_eval(filtered_indices, None, ablated=False)
            baseline_cache[rel_key] = baseline_vsr
            print(f"  [Baseline GPU{gpu_id}] VSR baseline for [{rel_key}] ({len(filtered_indices)} samples): "
                  f"{baseline_vsr['vsr_acc']:.1f}%", flush=True)
        baseline_vsr = baseline_cache[rel_key]

        # Load SAE decoder direction
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()
        feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
        del sae
        torch.cuda.empty_cache()

        print(f"[Ablation GPU{gpu_id}] [{feat_i+1}/{len(feature_assignments)}] "
              f"L{layer_idx}/F{feature_idx} relations=[{rel_key}] ({len(filtered_indices)} samples)...", flush=True)

        # Ablated VSR (per-relation)
        ablated_vsr = _run_vsr_eval(filtered_indices, feature_vec, ablated=True)

        # Ablated VQA
        ablated_vqa = _run_vqa_eval(feature_vec, ablated=True)

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "relations": relations,
            "odds_ratio": odds_ratio,
            "n_vsr_samples": len(filtered_indices),
            # VSR results (per-relation)
            "baseline_vsr_acc": baseline_vsr["vsr_acc"],
            "baseline_vsr_total": baseline_vsr["vsr_total"],
            "ablated_vsr_acc": ablated_vsr["vsr_acc"],
            "ablated_vsr_total": ablated_vsr["vsr_total"],
            "delta_vsr": ablated_vsr["vsr_acc"] - baseline_vsr["vsr_acc"],
            # Control
            "baseline_ctrl_acc": baseline_vsr["ctrl_acc"],
            "ablated_ctrl_acc": ablated_vsr["ctrl_acc"],
            "delta_ctrl": ablated_vsr["ctrl_acc"] - baseline_vsr["ctrl_acc"],
            # VQA
            "baseline_vqa_acc": vqa_baseline["vqa_acc"],
            "ablated_vqa_acc": ablated_vqa["vqa_acc"],
            "delta_vqa": ablated_vqa["vqa_acc"] - vqa_baseline["vqa_acc"],
            # Per-relation breakdown
            "per_relation_baseline": baseline_vsr.get("per_relation", {}),
            "per_relation_ablated": ablated_vsr.get("per_relation", {}),
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: "
              f"∆VSR={result['delta_vsr']:+.1f}% ({len(filtered_indices)} samp), "
              f"∆Ctrl={result['delta_ctrl']:+.1f}%, "
              f"∆VQA={result['delta_vqa']:+.1f}%, OR={odds_ratio:.1f}", flush=True)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"[Ablation GPU{gpu_id}] All features done.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV with layer,feature columns")
    parser.add_argument("--relations-csv", required=True, help="CSV from derive_relations_textonly.py")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--vqa-max", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    # Load features
    features_raw = []
    with open(args.features) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features_raw.append((int(row["layer"]), int(row["feature"])))
    print(f"Loaded {len(features_raw)} features")

    # Load relations mapping
    relations_map = {}
    with open(args.relations_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["layer"]), int(row["feature"]))
            relations_map[key] = row.get("relations", "")

    # Load odds ratios from spatial features CSV
    or_map = {}
    spatial_path = ANALYSIS_DIR / "spatial" / "spatial_features.csv"
    if spatial_path.exists():
        with open(spatial_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                or_map[(int(row["layer"]), int(row["feature"]))] = float(row.get("odds_ratio", 1.0))

    # Load priority scores for sorting
    cosine_dir = ANALYSIS_DIR / "cosines"
    energy_dir = ANALYSIS_DIR / "energy"
    cosine_map, ev_map = {}, {}
    for layer_idx in range(N_LAYERS):
        cos_path = cosine_dir / f"cosines_layer_{layer_idx}.npy"
        if cos_path.exists():
            cosines = np.load(cos_path)
            for fi in range(len(cosines)):
                cosine_map[(layer_idx, fi)] = float(cosines[fi])
        ev_path = energy_dir / f"Et_layer_{layer_idx}.npy"
        if ev_path.exists():
            evs = np.load(ev_path)
            for fi in range(len(evs)):
                ev_map[(layer_idx, fi)] = float(evs[fi])

    # Build feature list with relations and priority
    scored = []
    skipped_no_relations = 0
    for layer, feature in features_raw:
        key = (layer, feature)
        rel_str = relations_map.get(key, "")
        if not rel_str:
            skipped_no_relations += 1
            continue
        cos = cosine_map.get(key, 0.9)
        odds = or_map.get(key, 1.0)
        ev = ev_map.get(key, 0.01)
        score = (1.0 - cos) * odds * ev
        scored.append((layer, feature, rel_str, odds, score))

    scored.sort(key=lambda x: -x[4])  # Highest priority first
    features = [(l, f, r, o) for l, f, r, o, _ in scored]

    print(f"Features with relations: {len(features)} (skipped {skipped_no_relations} without relations)")
    print(f"\nTop 10 by priority:")
    for i, (l, f, r, o, s) in enumerate(scored[:10]):
        print(f"  {i+1}. L{l}/F{f}: score={s:.4f}, OR={o:.1f}, relations={r[:60]}")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / "ablation_per_relation"

    # Distribute across GPUs
    n_gpus = len(args.gpus)
    per_gpu = math.ceil(len(features) / n_gpus)
    assignments = []
    for i, gpu_id in enumerate(args.gpus):
        start = i * per_gpu
        end = min(start + per_gpu, len(features))
        worker_feats = features[start:end]
        if worker_feats:
            assignments.append((gpu_id, worker_feats))
            print(f"  GPU {gpu_id}: {len(worker_feats)} features")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for gpu_id, feats in assignments:
        p = mp.Process(target=_ablation_worker,
                       args=(gpu_id, feats, str(out_dir), args.vqa_max))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Collect results into summary
    all_results = []
    for p in sorted(Path(out_dir).glob("ablation_L*_F*.json")):
        with open(p) as f:
            all_results.append(json.load(f))

    if all_results:
        all_results.sort(key=lambda r: r.get("delta_vsr", 0))
        summary_path = out_dir / "ablation_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "layer", "feature", "relations", "odds_ratio",
                "delta_vsr", "delta_ctrl", "delta_vqa",
                "baseline_vsr_acc", "ablated_vsr_acc", "n_vsr_samples",
                "baseline_vqa_acc", "ablated_vqa_acc",
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    "layer": r["layer"], "feature": r["feature"],
                    "relations": "; ".join(r.get("relations", [])),
                    "odds_ratio": r.get("odds_ratio", ""),
                    "delta_vsr": f"{r.get('delta_vsr', 0):.2f}",
                    "delta_ctrl": f"{r.get('delta_ctrl', 0):.2f}",
                    "delta_vqa": f"{r.get('delta_vqa', 0):.2f}",
                    "baseline_vsr_acc": f"{r.get('baseline_vsr_acc', 0):.2f}",
                    "ablated_vsr_acc": f"{r.get('ablated_vsr_acc', 0):.2f}",
                    "n_vsr_samples": r.get("n_vsr_samples", 0),
                    "baseline_vqa_acc": f"{r.get('baseline_vqa_acc', 0):.2f}",
                    "ablated_vqa_acc": f"{r.get('ablated_vqa_acc', 0):.2f}",
                })

        print(f"\n{'='*80}")
        print(f"Per-Relation Ablation Summary ({len(all_results)} features)")
        print(f"{'='*80}")
        print(f"\n{'Layer':>5} {'Feat':>6} {'∆VSR':>8} {'∆Ctrl':>8} {'∆VQA':>8} {'OR':>6} {'N':>5} {'Relations'}")
        for r in all_results[:30]:
            rels = "; ".join(r.get("relations", []))[:40]
            print(f"  L{r['layer']:>2}  F{r['feature']:>5}  {r['delta_vsr']:+7.2f}%  "
                  f"{r['delta_ctrl']:+7.2f}%  {r['delta_vqa']:+7.2f}%  "
                  f"{r.get('odds_ratio', 0):5.1f}  {r.get('n_vsr_samples', 0):>4}  {rels}")
        print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
