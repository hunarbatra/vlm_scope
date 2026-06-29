#!/usr/bin/env python3
"""
Step 6 (safety variant) — Fisher exact test per feature:
  features firing significantly more on HallusionBench-unsafe than on VQAv2 baseline.

Runs two passes:
  (a) Overall: HallusionBench-unsafe (all 835) vs VQA 50K baseline  → "unsafe features"
  (b) Per-category: each of 6 categories vs VQA baseline       → category-specific features

Outputs:
  analysis_hallucination/_greedy/halluc_pertoken/halluc_features_pertoken.csv
  analysis_hallucination/_greedy/halluc_pertoken/all_features_stats_pertoken.csv
  analysis_hallucination/_greedy/halluc_pertoken_by_subcat/halluc_features_<CAT>.csv   (per category)

Thresholds (match Step 6 of VSR pipeline):
  odds_ratio >= 3.0, freq_diff >= 0.05
"""
import json, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
VQA_DIR    = ROOT / "analysis" / "firing_vqa_pertoken"                          # reused from VSR pipeline
UNSAFE_DIR = ROOT / "analysis_hallucination/_greedy" / "firing_incorrect_pertoken"
CAT_DIR    = ROOT / "analysis_hallucination/_greedy" / "firing_incorrect_by_subcat"
OUT_ALL    = ROOT / "analysis_hallucination/_greedy" / "halluc_pertoken"
OUT_CAT    = ROOT / "analysis_hallucination/_greedy" / "halluc_pertoken_by_subcat"
OUT_ALL.mkdir(parents=True, exist_ok=True)
OUT_CAT.mkdir(parents=True, exist_ok=True)

N_LAYERS = 26
D_SAE    = 16384
ODDS_THR = 3.0
MIN_FREQ_DIFF = 0.05


def fisher_pass(vqa_layers_data, unsafe_layers_data, out_stem):
    """Run Fisher test for each layer + save combined CSV."""
    rows = []
    for layer_idx in range(N_LAYERS):
        if layer_idx not in vqa_layers_data or layer_idx not in unsafe_layers_data: continue
        vqa_d = vqa_layers_data[layer_idx]
        uns_d = unsafe_layers_data[layer_idx]
        n_vqa = vqa_d["n_tokens"]
        fire_vqa = np.asarray(vqa_d["fire_count"], dtype=np.int64)
        n_uns = uns_d["n_tokens"]
        fire_uns = np.asarray(uns_d["fire_count"], dtype=np.int64)
        if n_vqa == 0 or n_uns == 0: continue
        for fi in range(D_SAE):
            c_vqa = int(fire_vqa[fi]); c_uns = int(fire_uns[fi])
            if c_vqa == 0 and c_uns == 0: continue
            rows.append({
                "layer": layer_idx, "feature": fi,
                "c_vqa": c_vqa, "n_vqa": n_vqa,
                "c_uns": c_uns, "n_uns": n_uns,
            })
    if not rows:
        print(f"  [WARN] no rows for {out_stem}"); return pd.DataFrame()
    df = pd.DataFrame(rows)
    pvals, odds = [], []
    for _, r in df.iterrows():
        cu = min(int(r.c_uns), int(r.n_uns))
        cv = min(int(r.c_vqa), int(r.n_vqa))
        table = [[cu, max(0, int(r.n_uns) - cu)],
                 [cv, max(0, int(r.n_vqa) - cv)]]
        try:
            o, p = fisher_exact(table, alternative="greater")
            odds.append(o if not math.isinf(o) else 1e9)
            pvals.append(p)
        except ValueError:
            odds.append(1.0); pvals.append(1.0)
    df["odds_ratio"] = odds
    df["p_raw"] = pvals
    df["freq_uns"]  = df.c_uns  / df.n_uns
    df["freq_vqa"]  = df.c_vqa  / df.n_vqa
    df["freq_diff"] = df.freq_uns - df.freq_vqa
    df["p_adj"] = multipletests(df.p_raw, method="fdr_bh")[1]
    return df


def main():
    # Load VQA baseline firings
    print("[INFO] Loading VQA baseline firings...")
    vqa_layers = {}
    for l in range(N_LAYERS):
        p = VQA_DIR / f"firing_vqa_layer_{l}.json"
        if p.exists(): vqa_layers[l] = json.load(open(p))
    print(f"  VQA layers loaded: {sorted(vqa_layers.keys())}")

    # --- Pass (a): overall UNSAFE vs VQA ---
    print("\n[Pass a] Overall HallusionBench-unsafe vs VQA")
    unsafe_layers = {}
    for l in range(N_LAYERS):
        p = UNSAFE_DIR / f"firing_halluc_layer_{l}.json"
        if p.exists(): unsafe_layers[l] = json.load(open(p))
    df = fisher_pass(vqa_layers, unsafe_layers, "all")
    if not df.empty:
        df.to_csv(OUT_ALL / "all_features_stats_pertoken.csv", index=False)
        keep = (df.odds_ratio >= ODDS_THR) & (df.freq_diff >= MIN_FREQ_DIFF)
        unsafe = df.loc[keep].sort_values("odds_ratio", ascending=False).copy()
        unsafe.to_csv(OUT_ALL / "halluc_features_pertoken.csv", index=False)
        print(f"  unsafe features (overall): {len(unsafe)} (of {len(df)} tested)")
        for lyr in sorted(unsafe.layer.unique()):
            n = (unsafe.layer == lyr).sum()
            print(f"    L{lyr}: {n} features")

    # --- Pass (b): per-category ---
    print("\n[Pass b] Per-category HallusionBench-unsafe vs VQA")
    # Discover categories from filenames
    cats = set()
    for p in CAT_DIR.glob("firing_halluc_*_layer_*.json"):
        # filename: firing_halluc_<CAT>_layer_<L>.json
        name = p.stem  # firing_halluc_Violent_layer_10
        parts = name.split("_layer_")[0]
        cat = parts[len("firing_halluc_"):]
        cats.add(cat)
    cats = sorted(cats)
    print(f"  categories found: {cats}")
    summary = {}
    for cat in cats:
        layers_data = {}
        for l in range(N_LAYERS):
            p = CAT_DIR / f"firing_halluc_{cat}_layer_{l}.json"
            if p.exists(): layers_data[l] = json.load(open(p))
        if not layers_data:
            print(f"  [{cat}] no layer data"); continue
        df_cat = fisher_pass(vqa_layers, layers_data, cat)
        if df_cat.empty: continue
        keep = (df_cat.odds_ratio >= ODDS_THR) & (df_cat.freq_diff >= MIN_FREQ_DIFF)
        unsafe_cat = df_cat.loc[keep].sort_values("odds_ratio", ascending=False).copy()
        unsafe_cat.to_csv(OUT_CAT / f"halluc_features_{cat}.csv", index=False)
        df_cat.to_csv(OUT_CAT / f"all_features_stats_{cat}.csv", index=False)
        summary[cat] = len(unsafe_cat)
        print(f"  [{cat}] {len(unsafe_cat)} unsafe features (N_samples={layers_data[next(iter(layers_data))]['n_samples']})")

    overall_n = 0
    if not df.empty:
        overall_n = int(((df.odds_ratio >= ODDS_THR) & (df.freq_diff >= MIN_FREQ_DIFF)).sum())
    with open(OUT_ALL / "summary.json", "w") as f:
        json.dump({"overall": overall_n,
                   "per_category": summary,
                   "odds_threshold": ODDS_THR, "min_freq_diff": MIN_FREQ_DIFF}, f, indent=2)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
