#!/usr/bin/env python3
"""
Identify spatial SAE features by comparing firing statistics between
VQA (baseline) and VSR (spatial-reasoning) datasets.

Usage
-----
python features/spatial/find_spatial_features.py \
       --vqa-json  results/feature_firing_analysis/vqa_text_only/feature_firing_analysis_0_50000.json \
       --vsr-json  results/feature_firing_analysis/vsr_text_only/feature_firing_analysis_0_3800.json \
       --method    pretrained \
       --p-thr     0.01 \
       --odds-thr  1.5 \
       --out       results/spatial_analysis/suspect_spatial_features_text-only.csv \
       [--vqa-basic results/feature_firing_analysis/vqa_text_only/basic_metrics_text-only_0_50000.pt] \
       [--vsr-basic results/feature_firing_analysis/vsr_text_only/basic_metrics_text-only_0_3800.pt]

    
CUDA_VISIBLE_DEVICES=4 python features/spatial/find_spatial_features.py \
       --vqa-json  results/feature_firing_analysis/vqa_text_only/feature_firing_analysis_0_50000.json \
       --vsr-json  results/feature_firing_analysis/vqa_spatial_text_only/feature_firing_analysis_0_20000.json \
       --method    text-only \
       --min-diff  0.005 \
       --odds-thr  3 \
       --out       results/stage_3/spatial/spatial_features_vqa_th3.csv \
       
python features/spatial/find_spatial_features.py \
       --vqa-json  results/feature_firing_analysis/vqa_text_only/feature_firing_analysis_0_50000.json \
       --vsr-json  results/feature_firing_analysis/vsr_text_only/feature_firing_analysis_0_3800.json \
       --method    text-only \
       --min-diff  0.05 \
       --odds-thr  3 \
       --out       results/stage_3/spatial/spatial_features_vsr.csv \
       [--vqa-basic results/feature_firing_analysis/vqa_text_only/basic_metrics_text-only_0_50000.pt] \
       [--vsr-basic results/feature_firing_analysis/vsr_text_only/basic_metrics_text-only_0_3800.pt]



Note: If --vqa-basic/--vsr-basic are provided, image-token-only firing counts
      will be used for numerators, while denominators remain total tokens from JSON
      (minimal change; keeps backward compatibility).
"""
import json, argparse, numpy as np, pandas as pd, math, sys, os
from pathlib import Path
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests   # pip install statsmodels
import torch

def load_json(path, method):
    with open(path) as f:
        data = json.load(f)["feature_firing_frequencies"][method]
    return data        # dict[layer][feature] = {...}

def load_image_firing_counts(basic_path):
    """Load per-layer image firing counts saved by tracking scripts.
    Returns dict[int layer] -> dict[int feature] -> int count
    """
    if basic_path is None or not os.path.exists(basic_path):
        return None
    m = torch.load(basic_path, map_location="cpu")
    img = m.get("image_firing_counts", {})
    # Normalize keys to int
    out = {}
    for layer, feat_map in img.items():
        try:
            L = int(layer)
        except Exception:
            L = layer
        out[L] = {int(k): int(v) for k, v in feat_map.items()}
    return out

def collect_rows(vqa, vsr, vqa_img_counts=None, vsr_img_counts=None):
    rows = []
    common_layers = sorted(set(map(int, vqa.keys())) & set(map(int, vsr.keys())))
    print(f"[INFO] Processing {len(common_layers)} common layers")
    
    total_features = 0
    skipped_features = 0
    
    for L in common_layers:
        vqa_layer, vsr_layer = vqa[str(L)], vsr[str(L)]
        common_feats = set(map(int, vqa_layer.keys())) & set(map(int, vsr_layer.keys()))
        total_features += len(common_feats)
        
        for F in common_feats:
            v = vqa_layer[str(F)]
            s = vsr_layer[str(F)]
            
            # Validate data
            if v["firing_count"] > v["total_tokens"] or s["firing_count"] > s["total_tokens"]:
                print(f"[WARN] Invalid data for layer {L}, feature {F}:")
                print(f"  VQA: {v['firing_count']} > {v['total_tokens']}")
                print(f"  VSR: {s['firing_count']} > {s['total_tokens']}")
                skipped_features += 1
                continue
            
            # If image-only counts are available, use them for numerators
            c_vqa = v["firing_count"]
            c_vsr = s["firing_count"]
            if vqa_img_counts is not None:
                c_vqa = int(vqa_img_counts.get(L, {}).get(F, 0))
            if vsr_img_counts is not None:
                c_vsr = int(vsr_img_counts.get(L, {}).get(F, 0))

            row = dict(layer=L, feature=F,
                       c_vqa=c_vqa,  n_vqa=v["total_tokens"],
                       c_vsr=c_vsr,  n_vsr=s["total_tokens"])
            rows.append(row)
    
    print(f"[INFO] Processed {len(rows)} features, skipped {skipped_features} invalid ones")
    return pd.DataFrame(rows)

def add_stats(df):
    """Add freq diff, odds ratio, p-value (Fisher) to DataFrame."""
    pvals, odds = [], []
    for _, r in df.iterrows():
        # Handle edge cases where firing_count > total_tokens
        c_vsr = min(r.c_vsr, r.n_vsr)
        c_vqa = min(r.c_vqa, r.n_vqa)
        
        # Ensure non-negative values for contingency table
        table = [[c_vsr, max(0, r.n_vsr - c_vsr)],
                 [c_vqa, max(0, r.n_vqa - c_vqa)]]
        
        try:
            o, p = fisher_exact(table, alternative="greater")
            odds.append(o if not math.isinf(o) else 1e9)
            pvals.append(p)
        except ValueError as e:
            print(f"[WARN] Fisher test failed for layer {r.layer}, feature {r.feature}: {e}")
            print(f"  VSR: {c_vsr}/{r.n_vsr}, VQA: {c_vqa}/{r.n_vqa}")
            odds.append(1.0)  # neutral odds ratio
            pvals.append(1.0)  # no significance
    
    df["odds_ratio"] = odds
    df["p_raw"] = pvals
    df["freq_vsr"] = df.c_vsr / df.n_vsr
    df["freq_vqa"] = df.c_vqa / df.n_vqa
    df["freq_diff"] = df.freq_vsr - df.freq_vqa
    return df



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa-json", required=True)
    ap.add_argument("--vsr-json", required=True)
    ap.add_argument("--method", default="pretrained")
    ap.add_argument("--p-thr",  type=float, default=0.01,
                    help="FDR-corrected p-value threshold (not used for filtering)")
    ap.add_argument("--odds-thr", type=float, default=2.0,
                    help="Minimum odds-ratio (VSR vs VQA) to keep")
    ap.add_argument("--min-diff", type=float, default=1e-3,
                    help="Minimum absolute frequency difference")
    ap.add_argument("--out", required=True, help="Output CSV file")
    ap.add_argument("--vqa-basic", default=None, help="Optional path to VQA basic metrics .pt (to use image-only numerators)")
    ap.add_argument("--vsr-basic", default=None, help="Optional path to VSR basic metrics .pt (to use image-only numerators)")
    args = ap.parse_args()

    vqa = load_json(args.vqa_json, args.method)
    vsr = load_json(args.vsr_json, args.method)

    vqa_img_counts = load_image_firing_counts(args.vqa_basic)
    vsr_img_counts = load_image_firing_counts(args.vsr_basic)

    df = collect_rows(vqa, vsr, vqa_img_counts, vsr_img_counts)
    df = add_stats(df)

    # FDR / BH adjustment
    df["p_adj"] = multipletests(df.p_raw, method="fdr_bh")[1]

    # Filter spatial - using only frequency difference and odds ratio (no p-value threshold)
    keep =  (df.odds_ratio >= args.odds_thr) \
          & (df.freq_diff >= args.min_diff)
    spatial = df.loc[keep].sort_values("odds_ratio", ascending=False)

    print(f"[INFO] Spatial features found: {len(spatial)} "
          f"out of {len(df)} ({100*len(spatial)/len(df):.1f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spatial.to_csv(out_path, index=False)
    print(f"[INFO] Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()