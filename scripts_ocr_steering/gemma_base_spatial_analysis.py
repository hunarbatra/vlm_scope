"""
Gemma-2-2B spatial feature analysis using vanilla gemma-scope SAEs.

For each of the top-20 mix-448 spatial features (by ablation drop), measures:
  - mean activation on spatial-relation tokens in VSR captions
  - mean activation on non-spatial control captions
  - selectivity = spatial_mean - control_mean
  - firing rate (fraction of captions where feature fires > 0)

This provides the "base LLM" baseline for the three-way comparison:
  Gemma-2-2B (text only, no vision) → pt-448 → mix-448

Since Gemma-2-2B has no VQA head, ablation accuracy is not measurable.
Instead: if the feature fires strongly in base Gemma, the spatial circuit
was already present in the text model; mix-448 fine-tuning then *wires* it
to image-grounded reasoning (causal power = ablation drop).

Usage (single GPU):
    python3 gemma_base_spatial_analysis.py \
        --ablation-csv /data1/vlm_scope_sae_mix448_textonly/analysis/ablation_per_relation_full/ablation_summary.csv \
        --out-dir /data1/vlm_scope_sae_mix448_textonly/analysis/gemma_base_spatial \
        --n-gpus 1

Usage (8 GPU):
    python3 gemma_base_spatial_analysis.py \
        --ablation-csv /data1/vlm_scope_sae_mix448_textonly/analysis/ablation_per_relation_full/ablation_summary.csv \
        --out-dir /data1/vlm_scope_sae_mix448_textonly/analysis/gemma_base_spatial \
        --n-gpus 8
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path

import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import initialize_jumprelu_sae

# ─── Config ─────────────────────────────────────────────────────────────────

GEMMA_MODEL_NAME = "google/gemma-2-2b"
HF_CACHE = os.environ.get("HF_HOME", "/data1/hbatra/mmdiff/hf_cache")
HF_DATASETS_CACHE = os.environ.get(
    "HF_DATASETS_CACHE",
    "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache",
)
GEMMA_MODEL_PATH = (
    Path(HF_CACHE)
    / "hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
)
GEMMA_SCOPE_CACHE = Path(HF_CACHE) / "hub"

TOP_N = 20          # top-N features by ablation selectivity
BATCH_SIZE = 16     # captions per forward pass
MAX_SEQ_LEN = 128   # max tokens per caption

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_top_features(ablation_csv: str, top_n: int = TOP_N):
    """Return top-N (layer, feature, delta_vsr, relations) by selectivity."""
    df = pd.read_csv(ablation_csv)
    df["sel"] = df["delta_vsr"] - df["delta_ctrl"]
    top = df.sort_values("sel").head(top_n)[
        ["layer", "feature", "delta_vsr", "delta_ctrl", "sel", "relations"]
    ].copy()
    return top.reset_index(drop=True)


def load_vsr_captions(hf_cache: str):
    """Load all VSR captions + relations from cached arrow files."""
    import datasets
    os.environ["HF_DATASETS_CACHE"] = hf_cache
    os.environ["DATASETS_OFFLINE"] = "1"
    ds = datasets.load_dataset(
        "cambridgeltl/vsr_random",
        split="train+validation+test",
        cache_dir=hf_cache,
    )
    captions = [r["caption"] for r in ds]
    relations = [r["relation"] for r in ds]
    labels = [r["label"] for r in ds]
    return captions, relations, labels


def get_spatial_relation_token_indices(tokens: list[str], relation: str) -> list[int]:
    """Return token indices that overlap with the spatial relation phrase."""
    rel_words = set(relation.lower().replace("_", " ").split())
    indices = []
    for i, tok in enumerate(tokens):
        clean = tok.lower().lstrip("▁").lstrip("Ġ")
        if clean in rel_words or any(w.startswith(clean) for w in rel_words if len(clean) > 2):
            indices.append(i)
    return indices if indices else list(range(len(tokens)))  # fallback: all tokens


# ─── Worker ──────────────────────────────────────────────────────────────────

def worker_fn(gpu_id: int, feature_assignments: list, captions: list,
              relations: list, labels: list, out_dir: str, args_dict: dict):
    """Per-GPU worker: analyse assigned (layer, feature) pairs."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from nnsight import NNsight

    device = f"cuda:{gpu_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[GPU{gpu_id}] Loading Gemma-2-2B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(GEMMA_MODEL_PATH), local_files_only=True
    )
    model_raw = AutoModelForCausalLM.from_pretrained(
        str(GEMMA_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model_raw.eval()
    print(f"[GPU{gpu_id}] Model loaded.", flush=True)

    # Encode all captions once upfront (reused across features)
    print(f"[GPU{gpu_id}] Tokenising {len(captions)} captions...", flush=True)
    all_input_ids = []
    all_attention_masks = []
    all_token_lists = []
    for cap in captions:
        enc = tokenizer(
            cap,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding=False,
        )
        all_input_ids.append(enc["input_ids"][0])
        all_attention_masks.append(enc["attention_mask"][0])
        all_token_lists.append(tokenizer.convert_ids_to_tokens(enc["input_ids"][0]))

    # Identify spatial vs non-spatial captions
    # Spatial: any VSR caption whose relation is in top features' relations
    top_relations = set()
    for _, feat_row in feature_assignments:
        for r in str(feat_row["relations"]).split(";"):
            top_relations.add(r.strip().lower())

    spatial_idx = [i for i, r in enumerate(relations) if r.lower() in top_relations]
    # Control: captions with no spatial keywords at all (simple nouns/verbs)
    spatial_kw = {"left", "right", "above", "below", "behind", "front", "inside",
                  "outside", "top", "bottom", "near", "close", "touching", "on",
                  "under", "beside", "next", "against", "ahead", "across"}
    control_idx = [
        i for i, cap in enumerate(captions)
        if not any(kw in cap.lower().split() for kw in spatial_kw)
    ][:len(spatial_idx)]  # balance sizes

    print(f"[GPU{gpu_id}] Spatial captions: {len(spatial_idx)}, control: {len(control_idx)}", flush=True)

    for feat_idx, feat_row in feature_assignments:
        layer_idx = int(feat_row["layer"])
        feature_id = int(feat_row["feature"])
        out_file = out_path / f"gemma_L{layer_idx}_F{feature_id}.json"
        if out_file.exists():
            print(f"[GPU{gpu_id}] L{layer_idx}/F{feature_id} already done, skip", flush=True)
            continue

        print(f"[GPU{gpu_id}] [{feat_idx+1}/{len(feature_assignments)}] L{layer_idx}/F{feature_id} "
              f"(mix-448 delta_vsr={feat_row['delta_vsr']:.1f}%)", flush=True)

        # Load vanilla gemma-scope SAE (no checkpoint = base weights)
        sae = initialize_jumprelu_sae(
            layer_idx=layer_idx,
            checkpoint_path=None,  # vanilla gemma-scope
            device="cpu",
            cache_dir=str(GEMMA_SCOPE_CACHE),
        )
        sae.eval()

        # Collect per-caption activations for this feature
        # Reinitialize NNsight per feature (avoids stale state across features)
        nns_model = NNsight(model_raw)

        def collect_activations(indices):
            """Return (mean_on_relation_tokens, mean_all_tokens, firing_rate) over index set."""
            rel_acts = []   # activation at spatial-relation token positions
            all_acts = []   # activation averaged over all tokens
            fired = 0

            batches = [indices[i:i+BATCH_SIZE] for i in range(0, len(indices), BATCH_SIZE)]
            for batch in batches:
                # Pad batch
                batch_ids = [all_input_ids[i] for i in batch]
                max_len = max(t.shape[0] for t in batch_ids)
                padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
                attn = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
                for bi, t in enumerate(batch_ids):
                    padded[bi, :t.shape[0]] = t
                    attn[bi, :t.shape[0]] = 1
                padded = padded.to(device)
                attn = attn.to(device)

                # Collect residual stream at layer_idx
                with torch.no_grad():
                    with nns_model.trace({"input_ids": padded, "attention_mask": attn}):
                        resid = nns_model.model.layers[layer_idx].output[0].save()

                resid_np = resid.value.float().cpu().numpy()  # (B, T, D_MODEL)

                for bi, sample_i in enumerate(batch):
                    seq_len = int(attn[bi].sum().item())
                    resid_sample = resid_np[bi, :seq_len]  # (T, D_MODEL)

                    # SAE encode
                    with torch.no_grad():
                        acts = sae.encode(
                            torch.tensor(resid_sample, dtype=torch.bfloat16)
                        )  # (T, D_SAE)
                    feat_acts = acts[:, feature_id].numpy()  # (T,)

                    # Relation-token positions
                    rel_positions = get_spatial_relation_token_indices(
                        all_token_lists[sample_i], relations[sample_i]
                    )
                    rel_val = float(feat_acts[rel_positions].mean()) if rel_positions else 0.0
                    all_val = float(feat_acts.mean())

                    rel_acts.append(rel_val)
                    all_acts.append(all_val)
                    if float(feat_acts.max()) > 0:
                        fired += 1

                del resid
                torch.cuda.empty_cache()

            n = len(indices)
            return {
                "mean_rel_tokens": float(np.mean(rel_acts)) if rel_acts else 0.0,
                "mean_all_tokens": float(np.mean(all_acts)) if all_acts else 0.0,
                "firing_rate": fired / n if n > 0 else 0.0,
                "n": n,
            }

        spatial_stats = collect_activations(spatial_idx)
        control_stats = collect_activations(control_idx)

        result = {
            "layer": layer_idx,
            "feature": feature_id,
            "relation": str(feat_row["relations"]),
            "mix448_delta_vsr": float(feat_row["delta_vsr"]),
            "mix448_selectivity": float(feat_row["sel"]),
            "gemma_base": {
                "spatial": spatial_stats,
                "control": control_stats,
                "selectivity_rel_tokens": spatial_stats["mean_rel_tokens"] - control_stats["mean_rel_tokens"],
                "selectivity_all_tokens": spatial_stats["mean_all_tokens"] - control_stats["mean_all_tokens"],
                "firing_rate_ratio": (
                    spatial_stats["firing_rate"] / control_stats["firing_rate"]
                    if control_stats["firing_rate"] > 0 else float("inf")
                ),
            },
        }

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[GPU{gpu_id}] L{layer_idx}/F{feature_id} done — "
            f"gemma_sel={result['gemma_base']['selectivity_rel_tokens']:+.3f} "
            f"(spatial_mean={spatial_stats['mean_rel_tokens']:.3f}, "
            f"ctrl_mean={control_stats['mean_rel_tokens']:.3f}, "
            f"fire_ratio={result['gemma_base']['firing_rate_ratio']:.2f}x) "
            f"vs mix448_drop={feat_row['delta_vsr']:.1f}%",
            flush=True,
        )

    print(f"[GPU{gpu_id}] All features done.", flush=True)


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(out_dir: str, ablation_csv: str):
    """Print comparison table: gemma-base firing vs mix-448 ablation drop."""
    results = []
    for f in sorted(Path(out_dir).glob("gemma_L*_F*.json")):
        d = json.load(open(f))
        results.append(d)

    if not results:
        print("No results yet.")
        return

    results.sort(key=lambda x: x["mix448_delta_vsr"])
    print(f"\n{'Layer':>5} {'Feat':>6} {'Relation':<30} {'mix-448 drop':>12} "
          f"{'Gemma sel(rel)':>14} {'Gemma sel(all)':>14} {'Fire ratio':>10}")
    print("-" * 100)
    for r in results:
        gb = r["gemma_base"]
        print(
            f"{r['layer']:>5} {r['feature']:>6} {str(r['relation'])[:30]:<30} "
            f"{r['mix448_delta_vsr']:>12.1f}% "
            f"{gb['selectivity_rel_tokens']:>+14.3f} "
            f"{gb['selectivity_all_tokens']:>+14.3f} "
            f"{gb['firing_rate_ratio']:>10.2f}x"
        )

    # Correlation: does higher mix-448 causal power correlate with higher base Gemma firing?
    drops = np.array([r["mix448_delta_vsr"] for r in results])
    sels = np.array([r["gemma_base"]["selectivity_rel_tokens"] for r in results])
    if len(drops) > 2:
        corr = float(np.corrcoef(drops, sels)[0, 1])
        print(f"\nCorrelation (mix-448 drop vs gemma-base selectivity): r={corr:.3f}")
        print("Positive r → features that fire more in base Gemma also have larger causal drops in mix-448")
        print("Near-zero r → causal wiring is VLM-training-specific, not inherited from text pretraining")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--summary-only", action="store_true",
                        help="Just print summary from existing results")
    args = parser.parse_args()

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DATASETS_OFFLINE"] = "1"

    if args.summary_only:
        print_summary(args.out_dir, args.ablation_csv)
        return

    top_features = load_top_features(args.ablation_csv, top_n=args.top_n)
    print(f"Top {len(top_features)} features loaded:")
    print(top_features[["layer", "feature", "delta_vsr", "relations"]].to_string(index=False))

    print("\nLoading VSR captions...")
    captions, relations, labels = load_vsr_captions(HF_DATASETS_CACHE)
    print(f"  {len(captions)} captions loaded")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Distribute features across GPUs
    feat_list = list(top_features.iterrows())  # [(idx, row), ...]
    n_gpus = min(args.n_gpus, len(feat_list))
    per_gpu = math.ceil(len(feat_list) / n_gpus)
    assignments = []
    for i in range(n_gpus):
        chunk = feat_list[i * per_gpu: (i + 1) * per_gpu]
        if chunk:
            assignments.append(chunk)

    if n_gpus == 1:
        worker_fn(
            gpu_id=0,
            feature_assignments=assignments[0],
            captions=captions,
            relations=relations,
            labels=labels,
            out_dir=args.out_dir,
            args_dict=vars(args),
        )
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for gpu_id, chunk in enumerate(assignments):
            p = ctx.Process(
                target=worker_fn,
                args=(gpu_id, chunk, captions, relations, labels, args.out_dir, vars(args)),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print("\n=== SUMMARY ===")
    print_summary(args.out_dir, args.ablation_csv)


if __name__ == "__main__":
    main()
