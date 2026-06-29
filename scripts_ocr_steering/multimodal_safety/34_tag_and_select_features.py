#!/usr/bin/env python3
"""
Step 9 (safety) — tag each final unsafe feature with its primary category and
select a compact ablation set + matched random control set.

Inputs:
  analysis_safety/final/final_unsafe_features.csv           (1061 features)
  analysis_safety/final/final_unsafe_features_<CAT>.csv     (per-category finals)
  analysis_safety/unsafe_pertoken_by_cat/unsafe_features_<CAT>.csv (pre-filter per-cat Fisher)
  analysis/adapted/adapted_features_results.csv             (for control pool)

Selection strategy:
  - test set   = top-K (default 10) per category by per-cat OR within the final set
                  (deduplicated; each feature tagged with its argmax-OR category).
  - control set = random features matched to the same layer distribution,
                  drawn from "adapted" features that are NOT in any per-category
                  Step 6 unsafe set (i.e. features that are not unsafe-signalling).

Outputs:
  analysis_safety/ablation_input/features_to_ablate.csv
  analysis_safety/ablation_input/selection_summary.json
"""
import ast, json, random
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
FINAL_CSV   = ROOT / "analysis_safety" / "final" / "final_unsafe_features.csv"
FINAL_DIR   = ROOT / "analysis_safety" / "final"
CAT_DIR     = ROOT / "analysis_safety" / "unsafe_pertoken_by_cat"
ADAPTED_CSV = ROOT / "analysis" / "adapted" / "adapted_features_results.csv"
OUT_DIR     = ROOT / "analysis_safety" / "ablation_input"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["Erotic", "Hate", "Illegal_Activity", "Privacy", "Self-Harm", "Violent"]
TOP_K_PER_CAT = 10
N_CONTROL = 20
SEED = 17


def load_final():
    df = pd.read_csv(FINAL_CSV)
    return set(zip(df.layer.astype(int), df.feature.astype(int)))


def load_per_cat_final():
    """{cat: DataFrame(layer, feature, odds_ratio, ...)} restricted to final-passed."""
    out = {}
    for cat in CATEGORIES:
        p = FINAL_DIR / f"final_unsafe_features_{cat}.csv"
        if not p.exists():
            continue
        out[cat] = pd.read_csv(p)
    return out


def load_step6_union():
    """Union of all per-category Step 6 Fisher passes (pre lexical/adapted filtering).
    Any feature here is 'unsafe-signalling' for at least one category, so we
    exclude these from the control pool."""
    union = set()
    for cat in CATEGORIES:
        p = CAT_DIR / f"unsafe_features_{cat}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        union.update(zip(df.layer.astype(int), df.feature.astype(int)))
    return union


def load_adapted():
    df = pd.read_csv(ADAPTED_CSV)
    out = set()
    for _, r in df.iterrows():
        layer = int(r["layer"])
        try:
            idxs = ast.literal_eval(r["adapted_indices"])
        except Exception:
            idxs = []
        for fi in idxs:
            out.add((layer, int(fi)))
    return out


def main():
    random.seed(SEED)
    final = load_final()
    per_cat = load_per_cat_final()
    step6_union = load_step6_union()
    adapted = load_adapted()
    print(f"[INFO] final unsafe features: {len(final)}")
    print(f"[INFO] per-category finals: {sorted(per_cat.keys())}")
    print(f"[INFO] Step 6 union (any-category unsafe): {len(step6_union)}")
    print(f"[INFO] adapted pool: {len(adapted)}")

    # --- Tag every final feature with its primary (argmax-OR) category ---
    primary_or = defaultdict(dict)  # feat -> {cat: OR}
    for cat, df in per_cat.items():
        for _, r in df.iterrows():
            key = (int(r["layer"]), int(r["feature"]))
            primary_or[key][cat] = float(r["odds_ratio"])

    feat_to_cat = {}
    for key in final:
        cat_ors = primary_or.get(key, {})
        if not cat_ors:
            feat_to_cat[key] = "Unknown"
            continue
        feat_to_cat[key] = max(cat_ors, key=cat_ors.get)

    # --- Pick top-K per category within the final set ---
    test_rows = []
    seen = set()
    for cat in CATEGORIES:
        df = per_cat.get(cat)
        if df is None:
            continue
        # restrict to features also in final overall
        mask = df.apply(
            lambda r: (int(r["layer"]), int(r["feature"])) in final, axis=1
        )
        df_final = df[mask].copy()
        df_final = df_final.sort_values("odds_ratio", ascending=False)
        count = 0
        for _, r in df_final.iterrows():
            key = (int(r["layer"]), int(r["feature"]))
            if key in seen:
                continue
            seen.add(key)
            test_rows.append({
                "layer": key[0], "feature": key[1],
                "primary_category": feat_to_cat[key],
                "selected_for_category": cat,
                "odds_ratio_in_cat": float(r["odds_ratio"]),
                "freq_diff_in_cat": float(r["freq_diff"]),
                "top_rank_in_cat": count + 1,
                "is_control": 0,
            })
            count += 1
            if count >= TOP_K_PER_CAT:
                break
        print(f"  [{cat}] picked {count} top features")

    # --- Build control set: same layer distribution, NOT in any Step 6 union ---
    layer_counts = defaultdict(int)
    for r in test_rows:
        layer_counts[r["layer"]] += 1

    control_pool = adapted - step6_union
    pool_by_layer = defaultdict(list)
    for (l, f) in control_pool:
        pool_by_layer[l].append(f)

    control_rows = []
    # scale: ~N_CONTROL total, distributed proportionally across layers used by test set
    total_test = sum(layer_counts.values())
    for l, n in layer_counts.items():
        n_ctrl = max(1, round(N_CONTROL * n / total_test))
        pool = pool_by_layer.get(l, [])
        if len(pool) < n_ctrl:
            n_ctrl = len(pool)
        sampled = random.sample(pool, n_ctrl)
        for fi in sampled:
            control_rows.append({
                "layer": l, "feature": fi,
                "primary_category": "CONTROL",
                "selected_for_category": "CONTROL",
                "odds_ratio_in_cat": 0.0,
                "freq_diff_in_cat": 0.0,
                "top_rank_in_cat": 0,
                "is_control": 1,
            })
    # Trim control set to exactly N_CONTROL if oversized
    if len(control_rows) > N_CONTROL:
        control_rows = random.sample(control_rows, N_CONTROL)
    print(f"  [CONTROL] picked {len(control_rows)} features")

    all_rows = test_rows + control_rows
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT_DIR / "features_to_ablate.csv", index=False)

    summary = {
        "n_total": len(df_out),
        "n_test": len(test_rows),
        "n_control": len(control_rows),
        "per_category_count": {
            cat: int((df_out.selected_for_category == cat).sum()) for cat in CATEGORIES
        },
        "primary_category_distribution": dict(df_out.primary_category.value_counts()),
        "layer_distribution": dict(df_out.layer.value_counts().sort_index()),
    }
    with open(OUT_DIR / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(f"\n[DONE] wrote {OUT_DIR}/features_to_ablate.csv ({len(df_out)} rows)")
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
