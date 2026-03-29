#!/usr/bin/env python3
"""
Identify OCR-enriched SAE features by comparing firing statistics between
VQA (baseline) and OCR-focused VQA subsets.

Usage
-----
python features/ocr/find_ocr_enriched_features.py \
       --vqa-json  results/feature_firing_analysis/vqa_text_only/feature_firing_analysis_0_50000.json \
       --ocr-json  results/feature_firing_analysis/vqa_ocr/feature_firing_analysis_0_20000.json \
       --method    text-only \
       --min-diff  0.005 \
       --odds-thr  4 \
       --out       results/ocr_analysis/suspect_ocr_features.csv \
       [--vqa-basic results/feature_firing_analysis/vqa/basic_metrics_pretrained_0_50000.pt] \
       [--ocr-basic results/feature_firing_analysis/vqa_ocr/basic_metrics_pretrained_0_20000.pt]

Note: If --vqa-basic/--ocr-basic are provided, image-token-only firing counts
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


def collect_rows(vqa, ocr, vqa_img_counts=None, ocr_img_counts=None):
    rows = []
    common_layers = sorted(set(map(int, vqa.keys())) & set(map(int, ocr.keys())))
    print(f"[INFO] Processing {len(common_layers)} common layers")

    total_features = 0
    skipped_features = 0

    for L in common_layers:
        vqa_layer, ocr_layer = vqa[str(L)], ocr[str(L)]
        common_feats = set(map(int, vqa_layer.keys())) & set(map(int, ocr_layer.keys()))
        total_features += len(common_feats)

        for F in common_feats:
            v = vqa_layer[str(F)]
            o = ocr_layer[str(F)]

            # Validate data
            if v["firing_count"] > v["total_tokens"] or o["firing_count"] > o["total_tokens"]:
                print(f"[WARN] Invalid data for layer {L}, feature {F}:")
                print(f"  VQA: {v['firing_count']} > {v['total_tokens']}")
                print(f"  OCR: {o['firing_count']} > {o['total_tokens']}")
                skipped_features += 1
                continue

            # If image-only counts are available, use them for numerators
            c_vqa = v["firing_count"]
            c_ocr = o["firing_count"]
            if vqa_img_counts is not None:
                c_vqa = int(vqa_img_counts.get(L, {}).get(F, 0))
            if ocr_img_counts is not None:
                c_ocr = int(ocr_img_counts.get(L, {}).get(F, 0))

            row = dict(layer=L, feature=F,
                       c_vqa=c_vqa,  n_vqa=v["total_tokens"],
                       c_ocr=c_ocr,  n_ocr=o["total_tokens"]) 
            rows.append(row)

    print(f"[INFO] Processed {len(rows)} features, skipped {skipped_features} invalid ones")
    return pd.DataFrame(rows)


def add_stats(df):
    """Add freq diff, odds ratio, p-value (Fisher) to DataFrame.
    Alternative is 'greater' meaning OCR > VQA.
    """
    pvals, odds = [], []
    for _, r in df.iterrows():
        c_ocr = min(r.c_ocr, r.n_ocr)
        c_vqa = min(r.c_vqa, r.n_vqa)

        table = [[c_ocr, max(0, r.n_ocr - c_ocr)],
                 [c_vqa, max(0, r.n_vqa - c_vqa)]]

        try:
            o, p = fisher_exact(table, alternative="greater")
            odds.append(o if not math.isinf(o) else 1e9)
            pvals.append(p)
        except ValueError as e:
            print(f"[WARN] Fisher test failed for layer {r.layer}, feature {r.feature}: {e}")
            print(f"  OCR: {c_ocr}/{r.n_ocr}, VQA: {c_vqa}/{r.n_vqa}")
            odds.append(1.0)
            pvals.append(1.0)

    df["odds_ratio"] = odds
    df["p_raw"] = pvals
    df["freq_ocr"] = df.c_ocr / df.n_ocr
    df["freq_vqa"] = df.c_vqa / df.n_vqa
    df["freq_diff"] = df.freq_ocr - df.freq_vqa
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa-json", required=True)
    ap.add_argument("--ocr-json", required=True)
    ap.add_argument("--method", default="pretrained")
    ap.add_argument("--p-thr",  type=float, default=0.01,
                    help="FDR-corrected p-value threshold (not used for filtering)")
    ap.add_argument("--odds-thr", type=float, default=2.0,
                    help="Minimum odds-ratio (OCR vs VQA) to keep")
    ap.add_argument("--min-diff", type=float, default=1e-3,
                    help="Minimum absolute frequency difference")
    ap.add_argument("--out", required=True, help="Output CSV file")
    ap.add_argument("--vqa-basic", default=None, help="Optional path to VQA basic metrics .pt (to use image-only numerators)")
    ap.add_argument("--ocr-basic", default=None, help="Optional path to OCR basic metrics .pt (to use image-only numerators)")
    args = ap.parse_args()

    vqa = load_json(args.vqa_json, args.method)
    ocr = load_json(args.ocr_json, args.method)

    vqa_img_counts = load_image_firing_counts(args.vqa_basic)
    ocr_img_counts = load_image_firing_counts(args.ocr_basic)

    df = collect_rows(vqa, ocr, vqa_img_counts, ocr_img_counts)
    df = add_stats(df)

    # FDR / BH adjustment
    df["p_adj"] = multipletests(df.p_raw, method="fdr_bh")[1]

    # Filter OCR-enriched features - using only frequency difference and odds ratio
    keep =  (df.odds_ratio >= args.odds_thr) \
          & (df.freq_diff >= args.min_diff)
    ocr_enriched = df.loc[keep].sort_values("odds_ratio", ascending=False)

    print(f"[INFO] OCR-enriched features found: {len(ocr_enriched)} "
          f"out of {len(df)} ({100*len(ocr_enriched)/len(df):.1f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_enriched.to_csv(out_path, index=False)
    print(f"[INFO] Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()


