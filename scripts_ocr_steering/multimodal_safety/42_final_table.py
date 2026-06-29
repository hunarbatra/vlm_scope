#!/usr/bin/env python3
"""
Final 4-metric table — DELTA format. Columns:
  category     = primary category (argmax-OR over the 6 per-cat Fisher CSVs)
  sub-category = the per-cat slot we picked the feature for (top-K-per-cat)
  ΔVLSBench_ASR = ablated_asr_unsafe − 100   (negative = ASR dropped)
  ΔVQA          = ablated_vqa_acc  − baseline_vqa_acc
  ΔCtrl_ASR     = ablated_asr_benign − 0     (positive = harm introduced on safe inputs)
  OR            = odds_ratio_in_cat
"""
from pathlib import Path
import pandas as pd

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
COMBINED = ROOT / "analysis_safety" / "ablation_results_combined_v2.csv"
OUT_CSV  = ROOT / "analysis_safety" / "final_ablation_table.csv"
OUT_BEST = ROOT / "analysis_safety" / "final_best_per_subcategory.csv"
OUT_CATSUB = ROOT / "analysis_safety" / "final_max_drop_per_cat_subcat.csv"


def main():
    df = pd.read_csv(COMBINED)

    proj = df[[
        "layer", "feature", "primary_category", "selected_for_category", "is_control",
        "delta_asr_unsafe", "delta_vqa", "delta_asr_benign",
        "odds_ratio_in_cat",
    ]].rename(columns={
        "primary_category":      "category",
        "selected_for_category": "sub-category",
        "delta_asr_unsafe":      "ΔVLSBench_ASR",
        "delta_vqa":             "ΔVQA",
        "delta_asr_benign":      "ΔCtrl_ASR",
        "odds_ratio_in_cat":     "OR",
    })
    proj = proj.sort_values(["is_control", "ΔVLSBench_ASR"])
    proj.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {OUT_CSV} ({len(proj)} rows)")

    fmt = lambda x: f"{x:+.2f}" if abs(x) < 1000 else f"{x:.0f}"
    test = proj[proj.is_control == 0]
    ctrl = proj[proj.is_control == 1]

    print("\n" + "="*82)
    print("BASELINES — VLSBench_ASR=100% (eval=originally-UNSAFE), "
          "VQA=86.5%, Ctrl_ASR≈0% (MSSBench-safe)")
    print("="*82)
    print(f"\n{'Group':>8} | {'n':>3} | {'ΔVLSBench_ASR':>14} | {'ΔVQA':>6} | {'ΔCtrl_ASR':>10}")
    print("-"*60)
    for lbl, sub in [("test", test), ("control", ctrl)]:
        print(f"{lbl:>8} | {len(sub):>3} | "
              f"{sub['ΔVLSBench_ASR'].mean():>+13.2f}% | "
              f"{sub['ΔVQA'].mean():>+5.2f}% | "
              f"{sub['ΔCtrl_ASR'].mean():>+9.2f}%")

    print("\n=== ALL 60 TEST FEATURES (sorted by ΔVLSBench_ASR) ===")
    print(test[["layer","feature","category","sub-category",
                "ΔVLSBench_ASR","ΔVQA","ΔCtrl_ASR","OR"]]
          .to_string(index=False, float_format=fmt))

    print("\n=== ALL 20 CONTROL FEATURES ===")
    print(ctrl[["layer","feature","ΔVLSBench_ASR","ΔVQA","ΔCtrl_ASR","OR"]]
          .to_string(index=False, float_format=fmt))

    # --- Concise: max-drop per unique (category, sub-category) pair ---
    print("\n=== Max ΔVLSBench_ASR drop per unique (category, sub-category) pair ===")
    test_cs = test.copy()
    idx = test_cs.groupby(["category", "sub-category"])["ΔVLSBench_ASR"].idxmin()
    catsub = test_cs.loc[idx].sort_values("ΔVLSBench_ASR")
    catsub.to_csv(OUT_CATSUB, index=False)
    print(catsub[["category","sub-category","layer","feature",
                  "ΔVLSBench_ASR","ΔVQA","ΔCtrl_ASR","OR"]]
          .to_string(index=False, float_format=fmt))
    print(f"\n  → wrote {OUT_CATSUB} ({len(catsub)} unique cat × sub-cat pairs)")

    # --- Best per sub-category (matches earlier "best per category") ---
    print("\n=== Best per sub-category (lowest ΔVLSBench_ASR; test only) ===")
    best = (test.sort_values("ΔVLSBench_ASR")
                .groupby("sub-category")
                .head(1)[["sub-category","layer","feature","category",
                          "ΔVLSBench_ASR","ΔVQA","ΔCtrl_ASR","OR"]])
    best.to_csv(OUT_BEST, index=False)
    print(best.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
