#!/usr/bin/env python3
"""
Summarize the FULL ablation sweep:
  - VLSBench ablation:  ablation_results_v2_full/
  - MSSBench-safe ctrl: ablation_results_v2_mssbench_safe/
  - VQA in v2_full / vqa_L*_F*.json (1000 yes/no samples)
  - features:           features_to_ablate_full.csv

Output:
  analysis_safety/ablation_results_combined_v2_full.csv
  analysis_safety/final_ablation_table_v2.csv  (delta-format projection)
  analysis_safety/final_max_drop_per_cat_subcat_v2.csv
  analysis_safety/final_best_per_subcategory_v2.csv
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
ANALYSIS_DIR = ROOT / "analysis_safety"
FEATURES_CSV = ANALYSIS_DIR / "ablation_input" / "features_to_ablate_full.csv"
UNSAFE_DIR   = ANALYSIS_DIR / "ablation_results_v2_full"
SAFE_DIR     = ANALYSIS_DIR / "ablation_results_v2_mssbench_safe"
OUT_COMBINED = ANALYSIS_DIR / "ablation_results_combined_v2_full.csv"
OUT_TABLE    = ANALYSIS_DIR / "final_ablation_table_v2.csv"
OUT_CATSUB   = ANALYSIS_DIR / "final_max_drop_per_cat_subcat_v2.csv"
OUT_BEST     = ANALYSIS_DIR / "final_best_per_subcategory_v2.csv"


def compute_asr(judge_path):
    n_unsafe = n_judged = 0
    per_cat = defaultdict(lambda: {"u": 0, "t": 0})
    for line in open(judge_path):
        r = json.loads(line)
        lab = r.get("judge_label")
        if lab not in ("UNSAFE", "SAFE"): continue
        n_judged += 1
        cat = r.get("category", "?")
        per_cat[cat]["t"] += 1
        if lab == "UNSAFE":
            n_unsafe += 1
            per_cat[cat]["u"] += 1
    asr = n_unsafe / max(n_judged, 1) * 100
    per_cat_asr = {c: (d["u"] / max(d["t"], 1) * 100) for c, d in per_cat.items()}
    return asr, n_unsafe, n_judged, per_cat_asr


def main():
    df_feat = pd.read_csv(FEATURES_CSV)
    print(f"[MAIN] {len(df_feat)} features")

    rows = []
    n_skipped = 0
    for i, row in df_feat.iterrows():
        L = int(row["layer"]); F = int(row["feature"])
        ju  = UNSAFE_DIR / f"judgments_L{L}_F{F}.jsonl"
        jb  = SAFE_DIR   / f"judgments_L{L}_F{F}.jsonl"
        vqa = UNSAFE_DIR / f"vqa_L{L}_F{F}.json"
        if not ju.exists():
            n_skipped += 1; continue
        asr_u, n_u, n_ju_, per_cat_asr = compute_asr(ju)
        asr_b = n_b = n_jb_ = None
        if jb.exists():
            asr_b, n_b, n_jb_, _ = compute_asr(jb)
        vqa_acc = vqa_base = delta_vqa = None
        if vqa.exists():
            vj = json.load(open(vqa))
            vqa_acc  = vj.get("vqa_acc")
            vqa_base = vj.get("baseline_vqa_acc")
            delta_vqa = vj.get("delta_vqa")
        rows.append({
            "layer": L, "feature": F,
            "primary_category": row["primary_category"],
            "selected_for_category": row["selected_for_category"],
            "is_control": int(row["is_control"]),
            "odds_ratio_in_cat": float(row["odds_ratio_in_cat"]),
            "ablated_asr_unsafe": asr_u,
            "delta_asr_unsafe": asr_u - 100.0,
            "n_judged_unsafe": n_ju_,
            "ablated_asr_benign": asr_b,
            "delta_asr_benign": (asr_b if asr_b is not None else None),  # baseline=0
            "n_judged_benign": n_jb_,
            "baseline_vqa_acc": vqa_base,
            "ablated_vqa_acc": vqa_acc,
            "delta_vqa": delta_vqa,
            "per_cat_asr": json.dumps(per_cat_asr),
        })
    df = pd.DataFrame(rows)
    print(f"[MAIN] summarized {len(df)} features (skipped {n_skipped} not yet judged)")
    df.to_csv(OUT_COMBINED, index=False)
    print(f"[OK] wrote {OUT_COMBINED}")

    # ---- delta-format projection ----
    proj = df[[
        "layer","feature","primary_category","selected_for_category","is_control",
        "delta_asr_unsafe","delta_vqa","delta_asr_benign","odds_ratio_in_cat",
    ]].rename(columns={
        "primary_category":      "category",
        "selected_for_category": "sub-category",
        "delta_asr_unsafe":      "ΔVLSBench_ASR",
        "delta_vqa":             "ΔVQA",
        "delta_asr_benign":      "ΔCtrl_ASR",
        "odds_ratio_in_cat":     "OR",
    }).sort_values(["is_control","ΔVLSBench_ASR"])
    proj.to_csv(OUT_TABLE, index=False)
    print(f"[OK] wrote {OUT_TABLE}")

    fmt = lambda x: f"{x:+.2f}" if pd.notna(x) and abs(x) < 1000 else (f"{x:.0f}" if pd.notna(x) else "—")
    test = proj[proj.is_control == 0]
    ctrl = proj[proj.is_control == 1]

    print("\n" + "="*82)
    print("BASELINES — VLSBench_ASR=100% (eval=ALL 835 baseline-UNSAFE), "
          "VQA=86.5%, Ctrl_ASR≈0%")
    print("="*82)
    print(f"\n{'Group':>8} | {'n':>4} | {'ΔVLSBench_ASR':>15} | {'ΔVQA':>7} | {'ΔCtrl_ASR':>11}")
    print("-"*64)
    for lbl, sub in [("test", test), ("control", ctrl)]:
        print(f"{lbl:>8} | {len(sub):>4} | "
              f"{sub['ΔVLSBench_ASR'].mean():>+14.2f}% | "
              f"{sub['ΔVQA'].mean():>+6.2f}% | "
              f"{sub['ΔCtrl_ASR'].mean():>+10.2f}%")

    print("\n=== Top-30 features by ΔVLSBench_ASR ===")
    print(test.sort_values("ΔVLSBench_ASR").head(30)[
        ["layer","feature","category","sub-category","ΔVLSBench_ASR","ΔVQA","ΔCtrl_ASR","OR"]
    ].to_string(index=False, float_format=fmt))

    print("\n=== Max ΔVLSBench_ASR drop per unique (category, sub-category) pair ===")
    test_cs = test.copy()
    idx = test_cs.groupby(["category","sub-category"])["ΔVLSBench_ASR"].idxmin()
    catsub = test_cs.loc[idx].sort_values("ΔVLSBench_ASR")
    catsub.to_csv(OUT_CATSUB, index=False)
    print(catsub[["category","sub-category","layer","feature",
                  "ΔVLSBench_ASR","ΔVQA","ΔCtrl_ASR","OR"]]
          .to_string(index=False, float_format=fmt))
    print(f"\n  → wrote {OUT_CATSUB} ({len(catsub)} pairs)")

    print("\n=== Best per sub-category (lowest ΔVLSBench_ASR; test only) ===")
    best = (test.sort_values("ΔVLSBench_ASR")
                .groupby("sub-category")
                .head(1)[["sub-category","layer","feature","category",
                          "ΔVLSBench_ASR","ΔVQA","ΔCtrl_ASR","OR"]])
    best.to_csv(OUT_BEST, index=False)
    print(best.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
