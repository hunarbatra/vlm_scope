#!/usr/bin/env python3
"""
Build features_to_ablate_full.csv covering ALL 1,061 final unsafe features
plus ~60 matched-random control features.

Output: analysis_safety/ablation_input/features_to_ablate_full.csv
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
OUT_CSV     = OUT_DIR / "features_to_ablate_full.csv"

CATEGORIES = ["Erotic", "Hate", "Illegal_Activity", "Privacy", "Self-Harm", "Violent"]
N_CONTROL = 60   # scaled up from 20 since the test set went from 60→1061
SEED = 17


def load_final():
    return pd.read_csv(FINAL_CSV)


def load_per_cat_final():
    return {cat: pd.read_csv(FINAL_DIR / f"final_unsafe_features_{cat}.csv")
            for cat in CATEGORIES if (FINAL_DIR / f"final_unsafe_features_{cat}.csv").exists()}


def load_step6_union():
    union = set()
    for cat in CATEGORIES:
        p = CAT_DIR / f"unsafe_features_{cat}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p)
        union.update(zip(df.layer.astype(int), df.feature.astype(int)))
    return union


def load_adapted():
    df = pd.read_csv(ADAPTED_CSV)
    out = set()
    for _, r in df.iterrows():
        layer = int(r["layer"])
        try: idxs = ast.literal_eval(r["adapted_indices"])
        except Exception: idxs = []
        for fi in idxs: out.add((layer, int(fi)))
    return out


def main():
    random.seed(SEED)
    final = load_final()
    final_set = set(zip(final.layer.astype(int), final.feature.astype(int)))
    per_cat = load_per_cat_final()
    step6_union = load_step6_union()
    adapted = load_adapted()
    print(f"[INFO] final unsafe features: {len(final_set)}")
    print(f"[INFO] step6 union: {len(step6_union)}")
    print(f"[INFO] adapted pool: {len(adapted)}")

    # Tag every final feature with its primary (argmax-OR) category
    primary_or = defaultdict(dict)
    for cat, df in per_cat.items():
        for _, r in df.iterrows():
            key = (int(r["layer"]), int(r["feature"]))
            primary_or[key][cat] = float(r["odds_ratio"])

    rows = []
    for _, r in final.iterrows():
        key = (int(r["layer"]), int(r["feature"]))
        cat_ors = primary_or.get(key, {})
        primary = max(cat_ors, key=cat_ors.get) if cat_ors else "Unknown"
        # selected_for_category = primary (since we're keeping all 1,061 here)
        rows.append({
            "layer": key[0], "feature": key[1],
            "primary_category": primary,
            "selected_for_category": primary,
            "odds_ratio_in_cat": float(r["odds_ratio"]),  # overall OR
            "freq_diff_in_cat": float(r["freq_diff"]),
            "top_rank_in_cat": 0,  # not applicable in full sweep
            "is_control": 0,
        })
    print(f"[INFO] test rows: {len(rows)}")

    # Control set — matched layer dist over the test set, drawn from
    # adapted - step6_union. Scaled to N_CONTROL across used layers.
    layer_counts = defaultdict(int)
    for r in rows: layer_counts[r["layer"]] += 1
    control_pool = adapted - step6_union
    pool_by_layer = defaultdict(list)
    for (l, f) in control_pool: pool_by_layer[l].append(f)
    total_test = sum(layer_counts.values())
    ctrl_rows = []
    for l, n in layer_counts.items():
        n_ctrl = max(1, round(N_CONTROL * n / total_test))
        pool = pool_by_layer.get(l, [])
        if len(pool) < n_ctrl: n_ctrl = len(pool)
        for fi in random.sample(pool, n_ctrl):
            ctrl_rows.append({
                "layer": l, "feature": fi,
                "primary_category": "CONTROL",
                "selected_for_category": "CONTROL",
                "odds_ratio_in_cat": 0.0,
                "freq_diff_in_cat": 0.0,
                "top_rank_in_cat": 0,
                "is_control": 1,
            })
    if len(ctrl_rows) > N_CONTROL:
        ctrl_rows = random.sample(ctrl_rows, N_CONTROL)
    print(f"[INFO] control rows: {len(ctrl_rows)}")

    df_out = pd.DataFrame(rows + ctrl_rows)
    df_out.to_csv(OUT_CSV, index=False)
    summary = {
        "n_total": len(df_out),
        "n_test":  len(rows),
        "n_control": len(ctrl_rows),
        "primary_category_distribution": {
            k: int(v) for k, v in df_out.primary_category.value_counts().items()
        },
        "layer_distribution": {
            int(k): int(v) for k, v in df_out.layer.value_counts().sort_index().items()
        },
    }
    (OUT_DIR / "selection_summary_full.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n[DONE] wrote {OUT_CSV} ({len(df_out)} rows)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
