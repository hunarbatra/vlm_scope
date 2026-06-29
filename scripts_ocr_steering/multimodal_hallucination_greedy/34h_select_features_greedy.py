#!/usr/bin/env python3
"""
Step 9 (hallucination) — select features to ablate.

Strategy: use ALL final hallucination features + a matched-layer random control
set drawn from adapted features that are NOT Fisher-selected (no hallucination
signal by construction). Mirrors 43_select_full_sweep.py from safety.

Outputs:
  analysis_hallucination/_greedy/ablation_input/features_to_ablate.csv
  analysis_hallucination/_greedy/ablation_input/selection_summary.json
"""
import ast, json, random
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
FINAL_CSV   = ROOT / "analysis_hallucination/_greedy" / "final" / "final_halluc_features.csv"
FINAL_DIR   = ROOT / "analysis_hallucination/_greedy" / "final"
SC_DIR      = ROOT / "analysis_hallucination/_greedy" / "halluc_pertoken_by_subcat"
HALLUC_CSV  = ROOT / "analysis_hallucination/_greedy" / "halluc_pertoken" / "halluc_features_pertoken.csv"
ADAPTED_CSV = ROOT / "analysis" / "adapted" / "adapted_features_results.csv"
OUT_DIR     = ROOT / "analysis_hallucination/_greedy" / "ablation_input"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CONTROL = 60
SEED = 17


def load_step6_union():
    """Hallucination Fisher-passed pool — exclude from control candidates."""
    if not HALLUC_CSV.exists(): return set()
    df = pd.read_csv(HALLUC_CSV)
    return set(zip(df.layer.astype(int), df.feature.astype(int)))


def load_adapted():
    df = pd.read_csv(ADAPTED_CSV)
    out = set()
    for _, r in df.iterrows():
        layer = int(r["layer"])
        try: idxs = ast.literal_eval(r["adapted_indices"])
        except Exception: idxs = []
        for fi in idxs: out.add((layer, int(fi)))
    return out


def load_per_subcat_primary():
    """feat -> argmax-OR subcategory across per-subcat CSVs (for optional tagging)."""
    primary = defaultdict(dict)
    for p in sorted(SC_DIR.glob("halluc_features_*.csv")):
        sc = p.stem[len("halluc_features_"):]
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            key = (int(r["layer"]), int(r["feature"]))
            primary[key][sc] = float(r["odds_ratio"])
    return primary


def main():
    random.seed(SEED)
    final = pd.read_csv(FINAL_CSV)
    print(f"[INFO] final hallucination features: {len(final)}")

    primary_or = load_per_subcat_primary()
    test_rows = []
    for _, r in final.iterrows():
        key = (int(r["layer"]), int(r["feature"]))
        sc_ors = primary_or.get(key, {})
        primary_subcat = max(sc_ors, key=sc_ors.get) if sc_ors else "Overall"
        test_rows.append({
            "layer": key[0], "feature": key[1],
            "primary_subcategory": primary_subcat,
            "selected_for_subcategory": primary_subcat,
            "odds_ratio": float(r["odds_ratio"]),
            "freq_diff": float(r["freq_diff"]),
            "is_control": 0,
        })
    print(f"[INFO] test rows: {len(test_rows)}")

    # Matched-layer control pool: adapted \ Fisher-selected
    union = load_step6_union(); adapted = load_adapted()
    control_pool = adapted - union
    pool_by_layer = defaultdict(list)
    for (l, f) in control_pool: pool_by_layer[l].append(f)

    layer_counts = defaultdict(int)
    for r in test_rows: layer_counts[r["layer"]] += 1
    total_test = max(sum(layer_counts.values()), 1)

    ctrl_rows = []
    for l, n in layer_counts.items():
        n_ctrl = max(1, round(N_CONTROL * n / total_test))
        pool = pool_by_layer.get(l, [])
        if len(pool) < n_ctrl: n_ctrl = len(pool)
        for fi in random.sample(pool, n_ctrl):
            ctrl_rows.append({
                "layer": l, "feature": fi,
                "primary_subcategory": "CONTROL",
                "selected_for_subcategory": "CONTROL",
                "odds_ratio": 0.0, "freq_diff": 0.0,
                "is_control": 1,
            })
    if len(ctrl_rows) > N_CONTROL:
        ctrl_rows = random.sample(ctrl_rows, N_CONTROL)
    print(f"[INFO] control rows: {len(ctrl_rows)}")

    df_out = pd.DataFrame(test_rows + ctrl_rows)
    df_out.to_csv(OUT_DIR / "features_to_ablate.csv", index=False)

    summary = {
        "n_total": len(df_out), "n_test": len(test_rows), "n_control": len(ctrl_rows),
        "primary_subcategory_distribution": {
            k: int(v) for k, v in df_out.primary_subcategory.value_counts().items()
        },
        "layer_distribution": {
            int(k): int(v) for k, v in df_out.layer.value_counts().sort_index().items()
        },
    }
    with open(OUT_DIR / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] wrote {OUT_DIR}/features_to_ablate.csv ({len(df_out)} rows)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
