#!/usr/bin/env python3
"""
VSR ablation for PaliGemma2-3B-pt-448 on the top-N spatial features from the
full text-only SAE run (mix-448 ablation_summary.csv).

Comparison story (all three use the SAME W_dec feature vectors):
  Gemma-2-2B text-only  →  pt-448 (vision pretrain, no instruct)  →  mix-448

Method (identical to mix-448 ablation):
  3-point projection across all 26 layers (attn_out, mlp_out, layer_out),
  text tokens only (positions after image tokens).

pt-448 baseline accuracy on VSR is lower than mix-448 (not instruct-tuned),
but the ablation delta tells us whether these feature directions are CAUSALLY
present in the pt backbone vs only becoming causal after mix fine-tuning.

Usage:
    # Download model first (one-time, needs internet):
    python3 pt448_vsr_ablation.py --download-only

    # Run ablation:
    python3 pt448_vsr_ablation.py \
        --ablation-csv /data1/vlm_scope_sae_mix448_textonly/analysis/ablation_per_relation_full/ablation_summary.csv \
        --out-dir /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_vsr_ablation \
        --n-gpus 8

    # Summary only:
    python3 pt448_vsr_ablation.py --summary-only \
        --ablation-csv ... --out-dir ...
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
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

PT_MODEL_NAME   = "google/paligemma2-3b-pt-448"
HF_CACHE        = "/data1/hf_cache/hub"
HF_DATASETS_CACHE = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
IMAGE_CACHE_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET     = "cambridgeltl/vsr_random"
HF_TOKEN        = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

N_LAYERS = 26
TOP_N    = 100

os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = HF_TOKEN


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def load_top_features(ablation_csv: str, top_n: int):
    import pandas as pd
    df = pd.read_csv(ablation_csv)
    df["sel"] = df["delta_vsr"] - df["delta_ctrl"]
    top = df.sort_values("sel").head(top_n)[
        ["layer", "feature", "delta_vsr", "delta_ctrl", "sel", "relations"]
    ].reset_index(drop=True)
    return top


def _load_vsr_image(ex):
    import requests
    from PIL import Image
    from io import BytesIO
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = IMAGE_CACHE_DIR / f"{url_hash}.jpg"
    try:
        if cache_path.exists():
            return Image.open(cache_path).convert("RGB")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        img.save(cache_path, "JPEG")
        return img
    except Exception:
        return None


# ─── Download helper ─────────────────────────────────────────────────────────

def download_model():
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    print(f"Downloading {PT_MODEL_NAME} to {HF_CACHE}...")
    AutoProcessor.from_pretrained(
        PT_MODEL_NAME, cache_dir=HF_CACHE, token=HF_TOKEN
    )
    PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, token=HF_TOKEN
    )
    print("Download complete.")


# ─── Worker ──────────────────────────────────────────────────────────────────

def worker_fn(gpu_id: int, feature_assignments: list, out_dir: str):
    from PIL import Image
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from datasets import load_dataset, concatenate_datasets

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return

    print(f"[PT448Abl GPU{gpu_id}] Loading pt-448...", flush=True)
    processor = AutoProcessor.from_pretrained(
        PT_MODEL_NAME, cache_dir=HF_CACHE, token=HF_TOKEN
    )
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, token=HF_TOKEN
    ).to(device).eval()
    model_raw.requires_grad_(False)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)

    # Compute img_end once (fixed for all samples at 448px resolution)
    import numpy as np
    _dummy_img = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
    _dummy_inputs = processor(text="a", images=_dummy_img, return_tensors="pt")
    _ids = _dummy_inputs["input_ids"][0].tolist()
    _IMAGE_TOKEN_ID = 257152
    _img_toks = [i for i, t in enumerate(_ids) if t == _IMAGE_TOKEN_ID]
    img_end = max(_img_toks) + 1 if _img_toks else 0
    print(f"[PT448Abl GPU{gpu_id}] img_end={img_end} (image tokens 0:{img_end})", flush=True)

    print(f"[PT448Abl GPU{gpu_id}] Model loaded.", flush=True)

    # Load VSR
    print(f"[PT448Abl GPU{gpu_id}] Loading VSR...", flush=True)
    ds_dict = load_dataset(VSR_DATASET, cache_dir=HF_DATASETS_CACHE)
    vsr_all = concatenate_datasets(list(ds_dict.values()))
    print(f"[PT448Abl GPU{gpu_id}] VSR: {len(vsr_all)} samples", flush=True)

    prompts = [_build_vsr_prompt(ex["caption"]) for ex in vsr_all]
    labels  = [int(ex["label"]) for ex in vsr_all]
    rels    = [ex["relation"] for ex in vsr_all]

    def make_attn_hook(fv, ie):
        def h(module, inp, out):
            x = out[0]  # (1, T, D)
            xt = x[:, ie:]
            x[:, ie:] = xt - (xt @ fv.T) * fv
            return (x,) + out[1:]
        return h

    def make_mlp_hook(fv, ie):
        def h(module, inp, out):
            xt = out[:, ie:]
            out = out.clone()
            out[:, ie:] = xt - (xt @ fv.T) * fv
            return out
        return h

    def make_layer_hook(fv, ie):
        def h(module, inp, out):
            x = out[0]  # (1, T, D)
            xt = x[:, ie:]
            x[:, ie:] = xt - (xt @ fv.T) * fv
            return (x,) + out[1:]
        return h

    for feat_idx, feat_row in feature_assignments:
        layer_idx  = int(feat_row["layer"])
        feature_id = int(feat_row["feature"])
        rel_str    = str(feat_row["relations"])
        out_file   = out_dir / f"pt448_abl_L{layer_idx}_F{feature_id}.json"

        if out_file.exists():
            print(f"[PT448Abl GPU{gpu_id}] L{layer_idx}/F{feature_id} skip", flush=True)
            continue

        print(
            f"[PT448Abl GPU{gpu_id}] [{feat_idx+1}/{len(feature_assignments)}] "
            f"L{layer_idx}/F{feature_id} mix_drop={feat_row['delta_vsr']:.1f}%",
            flush=True,
        )

        target_rels = {r.strip().lower() for r in rel_str.split(";")}
        indices = [i for i, r in enumerate(rels) if r.lower() in target_rels]
        if not indices:
            print(f"[PT448Abl GPU{gpu_id}] L{layer_idx}/F{feature_id} no VSR samples, skip", flush=True)
            continue

        # Load vanilla gemma-scope W_dec for this feature
        sae = initialize_jumprelu_sae(
            layer_idx=layer_idx,
            checkpoint_path=None,
            device="cpu",
            cache_dir=HF_CACHE,
        )
        sae.eval()
        W_dec = sae.W_dec[feature_id].to(torch.float32)
        feature_vec = (W_dec / (W_dec.norm() + 1e-8)).to(device=device, dtype=torch.bfloat16)
        del sae
        gc.collect()

        def run_eval(ablated: bool):
            correct = total = skip = 0
            hooks = []
            if ablated:
                fv = feature_vec.unsqueeze(0)  # (1, D)
                for l in range(N_LAYERS):
                    layer = model_raw.language_model.layers[l]
                    hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(fv, img_end)))
                    hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(fv, img_end)))
                    hooks.append(layer.register_forward_hook(make_layer_hook(fv, img_end)))

            for vi in indices:
                img = _load_vsr_image(vsr_all[vi])
                if img is None:
                    skip += 1
                    continue

                try:
                    inputs = processor(
                        text=prompts[vi],
                        images=img,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    input_ids = inputs["input_ids"]
                    attn_mask = inputs["attention_mask"]
                    pix_vals  = inputs.get("pixel_values")
                    gt        = labels[vi]

                    with torch.no_grad():
                        out = model_raw(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            pixel_values=pix_vals,
                            use_cache=False,
                        )
                    pred_logits = out.logits[0, -1]
                except Exception as e:
                    if total + skip < 3:
                        print(f"[PT448Abl GPU{gpu_id}] EXCEPTION vi={vi} ablated={ablated}: {type(e).__name__}: {e}", flush=True)
                    skip += 1
                    continue

                pred = _predict_yesno(pred_logits, yes_ids, no_ids)
                correct += int(pred == gt)
                total   += 1

                if total % 200 == 0:
                    print(
                        f"[PT448Abl GPU{gpu_id}] L{layer_idx}/F{feature_id} "
                        f"{'abl' if ablated else 'base'}: {total}/{len(indices)-skip}",
                        flush=True,
                    )

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
            "delta_vsr_pt448": delta,
            "mix448_delta_vsr": float(feat_row["delta_vsr"]),
            "mix448_selectivity": float(feat_row["sel"]),
        }

        with open(out_file, "w") as fh:
            json.dump(result, fh, indent=2)

        print(
            f"[PT448Abl GPU{gpu_id}] L{layer_idx}/F{feature_id}: "
            f"base={base_acc:.1f}% → abl={abl_acc:.1f}% "
            f"∆pt448={delta:+.2f}% | ∆mix448={feat_row['delta_vsr']:+.2f}%",
            flush=True,
        )

        del feature_vec
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[PT448Abl GPU{gpu_id}] Done.", flush=True)


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(out_dir: str, gemma_abl_dir: str = None):
    results = []
    for f in sorted(Path(out_dir).glob("pt448_abl_L*_F*.json")):
        results.append(json.load(open(f)))
    if not results:
        print("No pt-448 results yet.")
        return

    # Load gemma-base results if available for 3-way comparison
    gemma = {}
    if gemma_abl_dir and Path(gemma_abl_dir).exists():
        for f in Path(gemma_abl_dir).glob("gemma_abl_L*_F*.json"):
            r = json.load(open(f))
            gemma[(r["layer"], r["feature"])] = r["delta_vsr"]

    results.sort(key=lambda x: x["mix448_delta_vsr"])

    header = f"{'L':>3} {'F':>6} {'Relation':<28} {'N':>5} {'base%':>6} {'∆pt448':>8} {'∆mix448':>9}"
    if gemma:
        header += f" {'∆gemma':>8}"
    print(f"\n{header}")
    print("─" * (90 + (9 if gemma else 0)))

    for r in results:
        key = (r["layer"], r["feature"])
        line = (
            f"{r['layer']:>3} {r['feature']:>6} {str(r['relations'])[:28]:<28} "
            f"{r['n_vsr_samples']:>5} "
            f"{r['baseline_vsr_acc']:>6.1f}% "
            f"{r['delta_vsr_pt448']:>+8.2f}% "
            f"{r['mix448_delta_vsr']:>+9.2f}%"
        )
        if gemma:
            g = gemma.get(key, float("nan"))
            line += f" {g:>+8.2f}%" if not np.isnan(g) else f" {'n/a':>8}"
        print(line)

    deltas_pt  = np.array([r["delta_vsr_pt448"] for r in results])
    deltas_mix = np.array([r["mix448_delta_vsr"] for r in results])
    print(f"\nMean ∆pt448={deltas_pt.mean():+.2f}%  Mean ∆mix448={deltas_mix.mean():+.2f}%")
    if len(results) > 2:
        corr = float(np.corrcoef(deltas_pt, deltas_mix)[0, 1])
        print(f"Correlation (∆pt448 vs ∆mix448): r={corr:.3f}")
        stronger_mix = int((deltas_mix < deltas_pt).sum())
        print(f"Features where mix-448 drops MORE than pt-448: {stronger_mix}/{len(results)}")
    if gemma:
        deltas_g = np.array([gemma.get((r["layer"], r["feature"]), np.nan) for r in results])
        valid = ~np.isnan(deltas_g)
        if valid.sum() > 0:
            print(f"Mean ∆gemma={deltas_g[valid].mean():+.2f}%  (n={valid.sum()})")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", required=False,
                        help="mix-448 ablation_summary.csv (full run)")
    parser.add_argument("--out-dir", required=False,
                        help="Output dir for pt448_abl_L*_F*.json")
    parser.add_argument("--gemma-abl-dir", default=None,
                        help="Optional: gemma_base_vsr_ablation dir for 3-way comparison in summary")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--download-only", action="store_true",
                        help="Just download pt-448 model and exit")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.download_only:
        download_model()
        return

    if not args.ablation_csv or not args.out_dir:
        parser.error("--ablation-csv and --out-dir are required unless --download-only or --summary-only")

    if args.summary_only:
        print_summary(
            args.out_dir,
            gemma_abl_dir=args.gemma_abl_dir,
        )
        return

    top_features = load_top_features(args.ablation_csv, args.top_n)
    print(f"Top {len(top_features)} features:")
    print(top_features[["layer", "feature", "delta_vsr", "relations"]].to_string(index=False))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)


    feat_list  = list(top_features.iterrows())
    n_gpus     = min(args.n_gpus, len(feat_list))
    per_gpu    = math.ceil(len(feat_list) / n_gpus)
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

    print_summary(args.out_dir, gemma_abl_dir=args.gemma_abl_dir)


if __name__ == "__main__":
    main()
