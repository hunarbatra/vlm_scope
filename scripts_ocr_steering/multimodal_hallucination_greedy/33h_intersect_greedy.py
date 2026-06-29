#!/usr/bin/env python3
"""
Step 8 (safety variant) — intersect:
  adapted-features  ∩  unsafe-features  ∩  lexical-pass

The VSR Step 8 intersected (adapted ∩ spatial ∩ lexical). Here we intersect
(adapted ∩ unsafe-overall ∩ lexical-pass) to produce the final set of
"causally-relevant unsafe features".

Also produces per-category intersections:
  final_halluc_features_<CAT>.csv = halluc_<SUBCAT>.csv  ∩  lexical-pass  ∩  adapted

"adapted features" = output of the original Step 4 (saved at analysis/adapted.csv)

Outputs:
  analysis_hallucination/_greedy/final/final_halluc_features.csv           (overall)
  analysis_hallucination/_greedy/final/final_halluc_features_<CAT>.csv     (per category)
  analysis_hallucination/_greedy/final/summary.json
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
ADAPTED_CSV   = ROOT / "analysis" / "adapted" / "adapted_features_results.csv"   # from Step 4
HALLUC_CSV    = ROOT / "analysis_hallucination/_greedy" / "halluc_pertoken" / "halluc_features_pertoken.csv"
LEX_DIR       = ROOT / "analysis_hallucination/_greedy" / "lexical"
CAT_DIR       = ROOT / "analysis_hallucination/_greedy" / "halluc_pertoken_by_subcat"
OUT_DIR       = ROOT / "analysis_hallucination/_greedy" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lexical_pass():
    passed = set()    # {(layer, feature)}
    for p in sorted(LEX_DIR.glob("lexical_results_w*.json")):
        d = json.load(open(p))
        for r in d.get("results", []):
            if r.get("passed"):
                passed.add((int(r["layer"]), int(r["feature"])))
    return passed


def main():
    if not HALLUC_CSV.exists():
        print("[FATAL] halluc_features_pertoken.csv missing"); return
    unsafe = pd.read_csv(HALLUC_CSV)
    print(f"[INFO] unsafe (Step 6 pass): {len(unsafe)}")

    # Adapted set — optional. CSV has one row/layer with `adapted_indices` as a list-like string.
    import ast
    adapted_set = None
    if ADAPTED_CSV.exists():
        ad = pd.read_csv(ADAPTED_CSV)
        adapted_set = set()
        for _, r in ad.iterrows():
            layer = int(r["layer"])
            try:
                idxs = ast.literal_eval(r["adapted_indices"])
            except Exception:
                idxs = []
            for fi in idxs:
                adapted_set.add((layer, int(fi)))
        print(f"[INFO] adapted (Step 4): {len(adapted_set)} (layer,feature) pairs across {ad['layer'].nunique()} layers")
    else:
        print(f"[WARN] {ADAPTED_CSV} missing — skipping adapted ∩")

    # Lexical-passed
    lex_passed = load_lexical_pass()
    print(f"[INFO] lexical-passed: {len(lex_passed)}")

    # Build intersection
    def apply_filters(df):
        df = df.copy()
        df["key"] = list(zip(df["layer"].astype(int), df["feature"].astype(int)))
        if lex_passed:
            df = df[df["key"].isin(lex_passed)]
        if adapted_set:
            df = df[df["key"].isin(adapted_set)]
        df = df.drop(columns=["key"])
        return df

    final = apply_filters(unsafe)
    print(f"\n[FINAL overall] {len(final)} features after all filters")
    print(f"  per-layer: {dict(final.groupby('layer').size())}")
    final.sort_values("odds_ratio", ascending=False).to_csv(
        OUT_DIR / "final_halluc_features.csv", index=False)

    # Per-category
    summary = {"overall": len(final), "per_category": {}}
    for cat_csv in sorted(CAT_DIR.glob("halluc_features_*.csv")):
        cat = cat_csv.stem[len("halluc_features_"):]
        df = pd.read_csv(cat_csv)
        n0 = len(df)
        df2 = apply_filters(df)
        df2.sort_values("odds_ratio", ascending=False).to_csv(
            OUT_DIR / f"final_halluc_features_{cat}.csv", index=False)
        summary["per_category"][cat] = {"step6": n0, "final": len(df2)}
        print(f"  [{cat}] step6={n0} -> final={len(df2)}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
