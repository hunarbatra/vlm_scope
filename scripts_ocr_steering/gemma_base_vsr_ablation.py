#!/usr/bin/env python3
"""
VSR ablation for Gemma-2-2B (text-only, vanilla gemma-scope SAEs).

For each of the top-N mix-448 spatial features, runs the same 3-point
projection ablation used in the mix-448 pipeline, but on:
  - Gemma-2-2B with the caption text only (no image)
  - Vanilla gemma-scope-2b-pt-res SAE weights (no fine-tuning)

Prompt format (text-only True/False):
  "Is the following statement true or false?\nStatement: <caption>\nAnswer:"
  → predict True / False from next-token logits

Reports:
  - baseline_vsr_acc  (% True/False correct)
  - ablated_vsr_acc
  - delta_vsr  (ablated - baseline, like mix-448 paper)
  - n_vsr_samples

Allows direct comparison:
  Gemma-2-2B (text-only)  →  mix-448 VLM  (same feature, same relation subset)

Usage:
    python3 gemma_base_vsr_ablation.py \
        --ablation-csv /data1/vlm_scope_sae_mix448_textonly/analysis/ablation_per_relation_full/ablation_summary.csv \
        --out-dir /data1/vlm_scope_sae_mix448_textonly/analysis/gemma_base_vsr_ablation \
        --n-gpus 8

    # Summary only (after run):
    python3 gemma_base_vsr_ablation.py --summary-only \
        --ablation-csv ... --out-dir ...
"""

import os
import sys
import json
import math
import csv
import argparse
import warnings
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

GEMMA_MODEL_PATH = (
    Path(os.environ.get("HF_HOME", "/data1/hbatra/mmdiff/hf_cache"))
    / "hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
)
HF_CACHE = str(Path(os.environ.get("HF_HOME", "/data1/hbatra/mmdiff/hf_cache")) / "hub")
HF_DATASETS_CACHE = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
VSR_DATASET = "cambridgeltl/vsr_random"
N_LAYERS = 26
TOP_N = 20

os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HOME"] = str(Path(HF_CACHE).parent)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DATASETS_OFFLINE"] = "1"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_prompt(caption: str) -> str:
    return (
        "Is the following statement true or false?\n"
        f"Statement: {caption.strip()}\n"
        "Answer:"
    )


def _get_true_false_ids(tokenizer):
    true_ids, false_ids = set(), set()
    for t in [" True", "True", " true", "TRUE", " Yes", "Yes", " yes"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            true_ids.add(toks[0])
    for t in [" False", "False", " false", "FALSE", " No", "No", " no"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            false_ids.add(toks[0])
    overlap = true_ids & false_ids
    true_ids -= overlap
    false_ids -= overlap
    return true_ids, false_ids


def _predict(logits, true_ids, false_ids):
    probs = torch.softmax(logits, dim=-1)
    t = probs[list(true_ids)].sum().item() if true_ids else 0.0
    f = probs[list(false_ids)].sum().item() if false_ids else 0.0
    d = t + f
    return 1 if (t / d if d > 0 else 0) > 0.5 else 0


def load_top_features(ablation_csv: str, top_n: int):
    import pandas as pd
    df = pd.read_csv(ablation_csv)
    df["sel"] = df["delta_vsr"] - df["delta_ctrl"]
    top = df.sort_values("sel").head(top_n)[
        ["layer", "feature", "delta_vsr", "delta_ctrl", "sel", "relations"]
    ].reset_index(drop=True)
    return top


def load_feature_relations(relations_csv: str) -> dict:
    rel_map = {}
    with open(relations_csv) as f:
        for row in csv.DictReader(f):
            key = (int(row["layer"]), int(row["feature"]))
            rel_map[key] = row.get("relations", "")
    return rel_map


# ─── Worker ──────────────────────────────────────────────────────────────────

def worker_fn(gpu_id: int, feature_assignments: list, out_dir: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset, concatenate_datasets

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GemmaAbl GPU{gpu_id}] Loading Gemma-2-2B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(GEMMA_MODEL_PATH), local_files_only=True)
    model_raw = AutoModelForCausalLM.from_pretrained(
        str(GEMMA_MODEL_PATH), dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()
    model_raw.requires_grad_(False)
    print(f"[GemmaAbl GPU{gpu_id}] Model loaded.", flush=True)

    true_ids, false_ids = _get_true_false_ids(tokenizer)

    # Load all VSR splits
    print(f"[GemmaAbl GPU{gpu_id}] Loading VSR...", flush=True)
    ds_dict = load_dataset(VSR_DATASET, cache_dir=HF_DATASETS_CACHE)
    vsr_all = concatenate_datasets(list(ds_dict.values()))
    print(f"[GemmaAbl GPU{gpu_id}] VSR: {len(vsr_all)} samples", flush=True)

    # Pre-tokenise all prompts
    prompts = [_build_prompt(ex["caption"]) for ex in vsr_all]
    labels  = [int(ex["label"]) for ex in vsr_all]
    rels    = [ex["relation"] for ex in vsr_all]

    for feat_idx, feat_row in feature_assignments:
        layer_idx  = int(feat_row["layer"])
        feature_id = int(feat_row["feature"])
        rel_str    = str(feat_row["relations"])
        out_file   = out_dir / f"gemma_abl_L{layer_idx}_F{feature_id}.json"

        if out_file.exists():
            print(f"[GemmaAbl GPU{gpu_id}] L{layer_idx}/F{feature_id} skip", flush=True)
            continue

        print(
            f"[GemmaAbl GPU{gpu_id}] [{feat_idx+1}/{len(feature_assignments)}] "
            f"L{layer_idx}/F{feature_id} mix-448_drop={feat_row['delta_vsr']:.1f}%",
            flush=True,
        )

        # Filter VSR to this feature's relations
        target_rels = {r.strip().lower() for r in rel_str.split(";")}
        indices = [i for i, r in enumerate(rels) if r.lower() in target_rels]
        if not indices:
            print(f"[GemmaAbl GPU{gpu_id}] L{layer_idx}/F{feature_id} no matching VSR samples, skip", flush=True)
            continue

        # Load vanilla gemma-scope SAE (no fine-tuned checkpoint)
        sae = initialize_jumprelu_sae(
            layer_idx=layer_idx,
            checkpoint_path=None,  # vanilla gemma-scope weights
            device="cpu",
            cache_dir=HF_CACHE,
        )
        sae.eval()
        W_dec = sae.W_dec[feature_id].to(torch.float32)  # (D_MODEL,) on CPU
        feature_vec = (W_dec / (W_dec.norm() + 1e-8)).to(device=device, dtype=torch.bfloat16)
        del sae
        gc.collect()

        def _make_proj_hook(fv):
            """Returns a hook that projects out fv from a tensor (1, T, D)."""
            def hook_fn(module, input, output):
                # attn: output is tuple (tensor, ...), mlp: output is tensor, layer: tuple
                if isinstance(output, tuple):
                    x = output[0]  # (1, T, D)
                    x = x - (x @ fv.T) * fv
                    return (x,) + output[1:]
                else:
                    return output - (output @ fv.T) * output.new_tensor(fv)
            return hook_fn

        def run_eval(ablated: bool):
            correct = total = 0
            hooks = []
            if ablated:
                fv = feature_vec.unsqueeze(0)  # (1, D)

                def make_attn_hook(fv):
                    def h(module, inp, out):
                        x = out[0]  # (1, T, D)
                        x = x - (x @ fv.T) * fv
                        return (x,) + out[1:]
                    return h

                def make_mlp_hook(fv):
                    def h(module, inp, out):
                        return out - (out @ fv.T) * fv
                    return h

                def make_layer_hook(fv):
                    def h(module, inp, out):
                        x = out[0]  # (1, T, D)
                        x = x - (x @ fv.T) * fv
                        return (x,) + out[1:]
                    return h

                for l in range(N_LAYERS):
                    layer = model_raw.model.layers[l]
                    hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(fv)))
                    hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(fv)))
                    hooks.append(layer.register_forward_hook(make_layer_hook(fv)))

            for vi in indices:
                enc = tokenizer(prompts[vi], return_tensors="pt",
                                max_length=192, truncation=True).to(device)
                input_ids   = enc["input_ids"]
                attn_mask   = enc["attention_mask"]
                gt          = labels[vi]

                with torch.no_grad():
                    out = model_raw(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits[0, -1]

                pred = _predict(logits, true_ids, false_ids)
                correct += int(pred == gt)
                total   += 1

                if total % 200 == 0:
                    print(f"[GemmaAbl GPU{gpu_id}] L{layer_idx}/F{feature_id} "
                          f"{'abl' if ablated else 'base'}: {total}/{len(indices)}", flush=True)

            for h in hooks:
                h.remove()
            return correct, total

        base_correct, n = run_eval(ablated=False)
        abl_correct, _  = run_eval(ablated=True)

        base_acc = 100 * base_correct / n if n else 0.0
        abl_acc  = 100 * abl_correct  / n if n else 0.0
        delta    = abl_acc - base_acc

        result = {
            "layer": layer_idx,
            "feature": feature_id,
            "relations": rel_str,
            "n_vsr_samples": n,
            "baseline_vsr_acc": base_acc,
            "ablated_vsr_acc": abl_acc,
            "delta_vsr": delta,
            "mix448_delta_vsr": float(feat_row["delta_vsr"]),
            "mix448_selectivity": float(feat_row["sel"]),
        }

        with open(out_file, "w") as fh:
            json.dump(result, fh, indent=2)

        print(
            f"[GemmaAbl GPU{gpu_id}] L{layer_idx}/F{feature_id}: "
            f"base={base_acc:.1f}% → abl={abl_acc:.1f}% "
            f"delta_gemma={delta:+.2f}% | mix448_drop={feat_row['delta_vsr']:.1f}%",
            flush=True,
        )

        del feature_vec
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[GemmaAbl GPU{gpu_id}] Done.", flush=True)


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(out_dir: str):
    results = []
    for f in sorted(Path(out_dir).glob("gemma_abl_L*_F*.json")):
        results.append(json.load(open(f)))
    if not results:
        print("No results yet.")
        return

    results.sort(key=lambda x: x["mix448_delta_vsr"])

    print(f"\n{'L':>3} {'F':>6} {'Relation':<32} {'N':>5} "
          f"{'Gemma-base':>10} {'Gemma-abl':>10} {'∆Gemma':>8} {'∆mix-448':>9}")
    print("─" * 90)
    for r in results:
        print(
            f"{r['layer']:>3} {r['feature']:>6} {str(r['relations'])[:32]:<32} "
            f"{r['n_vsr_samples']:>5} "
            f"{r['baseline_vsr_acc']:>10.1f}% "
            f"{r['ablated_vsr_acc']:>10.1f}% "
            f"{r['delta_vsr']:>+8.2f}% "
            f"{r['mix448_delta_vsr']:>+9.2f}%"
        )

    deltas_g = np.array([r["delta_vsr"] for r in results])
    deltas_m = np.array([r["mix448_delta_vsr"] for r in results])
    if len(results) > 2:
        corr = float(np.corrcoef(deltas_g, deltas_m)[0, 1])
        print(f"\nCorrelation (∆Gemma vs ∆mix-448): r={corr:.3f}")
        print(f"Mean ∆Gemma={deltas_g.mean():+.2f}%  Mean ∆mix-448={deltas_m.mean():+.2f}%")
        stronger_in_mix = int((deltas_m < deltas_g).sum())
        print(f"Features where mix-448 drops MORE than Gemma: {stronger_in_mix}/{len(results)}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", required=True,
                        help="mix-448 ablation_summary.csv")
    parser.add_argument("--relations-csv", default=None,
                        help="Optional feature_relations.csv override (uses ablation-csv relations if omitted)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        print_summary(args.out_dir)
        return

    top_features = load_top_features(args.ablation_csv, args.top_n)
    print(f"Top {len(top_features)} features:")
    print(top_features[["layer", "feature", "delta_vsr", "relations"]].to_string(index=False))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    feat_list = list(top_features.iterrows())
    n_gpus = min(args.n_gpus, len(feat_list))
    per_gpu = math.ceil(len(feat_list) / n_gpus)
    assignments = [feat_list[i*per_gpu:(i+1)*per_gpu] for i in range(n_gpus)]
    assignments = [a for a in assignments if a]

    if n_gpus == 1:
        worker_fn(0, assignments[0], args.out_dir)
    else:
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=worker_fn, args=(gpu_id, chunk, args.out_dir))
            for gpu_id, chunk in enumerate(assignments)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    print_summary(args.out_dir)


if __name__ == "__main__":
    main()
