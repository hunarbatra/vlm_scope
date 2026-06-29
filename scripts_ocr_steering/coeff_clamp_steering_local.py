#!/usr/bin/env python3
"""
Label-conditioned feature coefficient clamping for top spatial SAE features.

For each (feature, relation-subset) pair:
  - Split VSR samples by ground-truth label (GT=Yes → true_idx, GT=No → false_idx)
  - Apply POSITIVE scale (amplify) to true_idx: should push model toward "Yes" for real Yes
  - Apply SUPPRESSION scale (<1.0) to false_idx: should push model toward "No" for real No
  - Report per-split accuracy + combined accuracy at each (pos_scale, neg_scale) pair

Delta computation (single-layer, SAE-coefficient scaling):
  residual = layer[L].output[0][text_tokens]          # baseline pass
  acts     = sae.encode(residual)                      # [n_text, d_sae]
  delta    = (scale - 1) * acts[:, F] * W_dec[F]      # [n_text, d_model]
  injected at layer[L].output[0][text_tokens] only.

Scales tested: LABEL_COND_PAIRS = [(pos, neg), ...]
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

sys.path.insert(0, str(Path(__file__).parent))
from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384

CHECKPOINT_DIR = Path(os.environ.get("VLMSCOPE_CKPT_DIR",
                                      "/data1/vlm_scope_sae_mix448_textonly/checkpoints"))
ANALYSIS_DIR = Path(os.environ.get("VLMSCOPE_ANALYSIS_DIR",
                                    "/data1/vlm_scope_sae_mix448_textonly/analysis"))
HF_CACHE = os.environ.get("HF_HOME", "/data1/hf_cache")

IMAGE_CACHE_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")

VSR_DATASET = "cambridgeltl/vsr_random"
VQA_MAX = 1000
CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}

# (pos_scale for GT=Yes samples, neg_scale for GT=No samples)
# pos_scale > 1.0 amplifies feature; neg_scale < 1.0 suppresses feature
LABEL_COND_PAIRS = [
    (1.5, 0.75),   # mild
    (2.0, 0.5),    # moderate
    (3.0, 0.25),   # strong
    (5.0, 0.0),    # maximum (fully suppress on GT=No)
]


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


def _predict_yesno(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


def _coeff_clamp_worker(gpu_id, feature_assignments, out_dir, vqa_max):
    import requests
    from PIL import Image
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)

    # Load VSR (all splits)
    print(f"[Clamp GPU{gpu_id}] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split="train"),
        load_dataset(VSR_DATASET, data_files=data_files, split="dev"),
        load_dataset(VSR_DATASET, data_files=data_files, split="test"),
    ])
    print(f"[Clamp GPU{gpu_id}] VSR total: {len(vsr_all)}", flush=True)

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

    def _baseline_pass(input_ids, attn_mask, pixel_values, layer_idx, img_end):
        with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                              pixel_values=pixel_values):
            residual_saved = nns_model.model.language_model.layers[layer_idx].output[0][0, img_end:].save()
            logits_saved = nns_model.output.logits.save()
        return logits_saved, residual_saved

    def _clamped_pass(input_ids, attn_mask, pixel_values, layer_idx, img_end, delta):
        with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                              pixel_values=pixel_values):
            nns_model.model.language_model.layers[layer_idx].output[0][0, img_end:] += delta
            logits_saved = nns_model.output.logits.save()
        return logits_saved

    def _run_eval_with_clamp(indices, sae, feature_idx, scale, layer_idx):
        """Run VSR eval on `indices` with coefficient clamped at `scale` (1.0 = baseline)."""
        correct = total = 0
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
                    img, prompt, processor, model_raw, device=device)
                _, img_end = get_image_token_positions(input_ids)

                if scale == 1.0:
                    with torch.inference_mode():
                        out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                        pixel_values=pixel_values, use_cache=False)
                    pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
                else:
                    logits_base, residual = _baseline_pass(
                        input_ids, attn_mask, pixel_values, layer_idx, img_end)
                    with torch.no_grad():
                        acts = sae.encode(residual.detach())
                        feat_acts = acts[:, feature_idx]
                        w_dec_f = sae.W_dec[feature_idx].to(device)
                        delta = (scale - 1) * feat_acts.unsqueeze(-1) * w_dec_f.unsqueeze(0)
                    logits_steered = _clamped_pass(
                        input_ids, attn_mask, pixel_values, layer_idx, img_end, delta)
                    pred = _predict_yesno(logits_steered[0, -1, :], yes_ids, no_ids)

            except Exception:
                pred = 0

            total += 1
            if pred == label: correct += 1
            per_relation[relation]["total"] += 1
            if pred == label: per_relation[relation]["correct"] += 1

        return {
            "acc": 100 * correct / max(total, 1),
            "correct": correct, "total": total,
            "per_relation": dict(per_relation),
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-index VSR by relation
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        rel = vsr_all[vi].get("relation", "")
        relation_indices[rel].append(vi)
    print(f"[Clamp GPU{gpu_id}] Pre-indexed {len(relation_indices)} VSR relations", flush=True)

    for feat_i, (layer_idx, feature_idx, relations_str, odds_ratio) in enumerate(feature_assignments):
        result_path = out_dir / f"lc_clamp_L{layer_idx}_F{feature_idx}.json"
        if result_path.exists():
            print(f"[Clamp GPU{gpu_id}] L{layer_idx}/F{feature_idx}: cached, skip", flush=True)
            continue
        if not relations_str: continue
        relations = [r.strip() for r in relations_str.split(";") if r.strip()]
        if not relations: continue

        # Gather and split by label
        all_idx = []
        for rel in relations:
            all_idx.extend(relation_indices.get(rel, []))
        if not all_idx: continue

        true_idx  = [vi for vi in all_idx if int(vsr_all[vi].get("label", 0)) == 1]
        false_idx = [vi for vi in all_idx if int(vsr_all[vi].get("label", 0)) == 0]

        print(f"[Clamp GPU{gpu_id}] [{feat_i+1}/{len(feature_assignments)}] "
              f"L{layer_idx}/F{feature_idx} [{';'.join(relations)[:50]}] "
              f"n={len(all_idx)} (yes={len(true_idx)} no={len(false_idx)})", flush=True)

        # Baseline (no intervention, both splits)
        base_true  = _run_eval_with_clamp(true_idx,  None, feature_idx, 1.0, layer_idx)
        base_false = _run_eval_with_clamp(false_idx, None, feature_idx, 1.0, layer_idx)
        base_combined_correct = base_true["correct"] + base_false["correct"]
        base_combined_total   = base_true["total"]   + base_false["total"]
        base_combined_acc = 100 * base_combined_correct / max(base_combined_total, 1)
        print(f"  [Baseline] yes_acc={base_true['acc']:.1f}%  no_acc={base_false['acc']:.1f}%  "
              f"combined={base_combined_acc:.1f}%", flush=True)

        # Load SAE
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "relations": relations, "odds_ratio": odds_ratio,
            "n_true": len(true_idx), "n_false": len(false_idx),
            "n_total": len(all_idx),
            "baseline": {
                "true_acc":     base_true["acc"],
                "false_acc":    base_false["acc"],
                "combined_acc": base_combined_acc,
            },
            "label_conditioned": {},
        }

        for pos_scale, neg_scale in LABEL_COND_PAIRS:
            key = f"pos{pos_scale}_neg{neg_scale}"
            steered_true  = _run_eval_with_clamp(true_idx,  sae, feature_idx, pos_scale, layer_idx)
            steered_false = _run_eval_with_clamp(false_idx, sae, feature_idx, neg_scale, layer_idx)
            comb_correct = steered_true["correct"] + steered_false["correct"]
            comb_total   = steered_true["total"]   + steered_false["total"]
            comb_acc = 100 * comb_correct / max(comb_total, 1)
            delta_true     = steered_true["acc"]  - base_true["acc"]
            delta_false    = steered_false["acc"] - base_false["acc"]
            delta_combined = comb_acc             - base_combined_acc
            result["label_conditioned"][key] = {
                "pos_scale": pos_scale, "neg_scale": neg_scale,
                "true_acc":     steered_true["acc"],
                "false_acc":    steered_false["acc"],
                "combined_acc": comb_acc,
                "delta_true":     delta_true,
                "delta_false":    delta_false,
                "delta_combined": delta_combined,
            }
            print(f"  [pos×{pos_scale}, neg×{neg_scale}]  "
                  f"yes: {base_true['acc']:.1f}→{steered_true['acc']:.1f}% ({delta_true:+.1f})  "
                  f"no:  {base_false['acc']:.1f}→{steered_false['acc']:.1f}% ({delta_false:+.1f})  "
                  f"combined: {base_combined_acc:.1f}→{comb_acc:.1f}% ({delta_combined:+.1f})", flush=True)

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[Clamp GPU{gpu_id}] Done.", flush=True)


def main():
    global CHECKPOINT_DIR, ANALYSIS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", type=str,
                        default=str(ANALYSIS_DIR / "ablation_per_relation_full" / "ablation_summary.csv"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--analysis-dir", type=str, default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    ANALYSIS_DIR = Path(args.analysis_dir)
    os.environ["VLMSCOPE_CKPT_DIR"] = args.checkpoint_dir
    os.environ["VLMSCOPE_ANALYSIS_DIR"] = args.analysis_dir

    # Load top-N features by selectivity
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
    print(f"Top-{args.top_n} features by selectivity:")
    for l, f, rels, or_ in features:
        sel = next(r["sel"] for r in top_rows if r["layer"] == l and r["feature"] == f)
        print(f"  L{l}/F{f}: sel={sel:+.2f}  rels={rels[:60]}")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / "coeff_clamp_label_cond"
    out_dir.mkdir(parents=True, exist_ok=True)

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
            p = mp.Process(target=_coeff_clamp_worker,
                           args=(gpu_id, chunks[i], str(out_dir), 0))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

    # Aggregate summary CSV
    all_results = []
    for jp in sorted(out_dir.glob("lc_clamp_L*_F*.json")):
        with open(jp) as f:
            all_results.append(json.load(f))

    if all_results:
        summary_path = out_dir / "lc_clamp_summary.csv"
        fieldnames = ["layer", "feature", "relations", "n_true", "n_false",
                      "baseline_true_acc", "baseline_false_acc", "baseline_combined_acc"]
        for pos, neg in LABEL_COND_PAIRS:
            k = f"pos{pos}_neg{neg}"
            fieldnames += [f"{k}_delta_true", f"{k}_delta_false", f"{k}_delta_combined",
                           f"{k}_combined_acc"]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_results:
                row = {
                    "layer": r["layer"], "feature": r["feature"],
                    "relations": "; ".join(r.get("relations", [])),
                    "n_true": r.get("n_true", 0), "n_false": r.get("n_false", 0),
                    "baseline_true_acc":     f"{r['baseline']['true_acc']:.2f}",
                    "baseline_false_acc":    f"{r['baseline']['false_acc']:.2f}",
                    "baseline_combined_acc": f"{r['baseline']['combined_acc']:.2f}",
                }
                for pos, neg in LABEL_COND_PAIRS:
                    k = f"pos{pos}_neg{neg}"
                    lc = r.get("label_conditioned", {}).get(k, {})
                    row[f"{k}_delta_true"]     = f"{lc.get('delta_true', 0):.2f}"
                    row[f"{k}_delta_false"]    = f"{lc.get('delta_false', 0):.2f}"
                    row[f"{k}_delta_combined"] = f"{lc.get('delta_combined', 0):.2f}"
                    row[f"{k}_combined_acc"]   = f"{lc.get('combined_acc', 0):.2f}"
                w.writerow(row)

        print(f"\nLabel-conditioned clamping summary: {summary_path} ({len(all_results)} features)")
        print(f"\nTop results by ∆combined at (×3, ×0.25):")
        key = "pos3.0_neg0.25"
        for r in sorted(all_results,
                        key=lambda x: x.get("label_conditioned", {}).get(key, {}).get("delta_combined", 0),
                        reverse=True)[:10]:
            lc = r.get("label_conditioned", {}).get(key, {})
            rels = "; ".join(r.get("relations", []))[:40]
            print(f"  L{r['layer']:>2}/F{r['feature']:>5}  "
                  f"base={r['baseline']['combined_acc']:.1f}%  "
                  f"∆true={lc.get('delta_true',0):+.1f}%  "
                  f"∆false={lc.get('delta_false',0):+.1f}%  "
                  f"∆combined={lc.get('delta_combined',0):+.1f}%  [{rels}]")


if __name__ == "__main__":
    main()
