"""
Run Fisher exact test: VQA vs VSR firing.

Reads existing firing stats from Modal volume:
  - VQA: firing_jumprelu/firing_layer_{i}.json (n_all=50000, fire_count_all)
  - VSR: firing_vsr_jumprelu/firing_vsr_layer_{i}.json (n_vsr=7529, fire_count_vsr)

Then compares using Fisher exact test with global FDR (matching old LLaVA-MORE pipeline).

Usage:
    MODAL_PROFILE=hunar-oxford modal run modal_fisher_vsr.py
"""

import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("vlm-scope-fisher-vsr")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "statsmodels", "pandas", "tqdm")
)

N_LAYERS = 26
D_SAE = 16384
RESULTS_BASE = "/vol/results/paligemma2"
SAE_TYPE = "jumprelu"


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=3600,
)
def fisher_test_vsr():
    """Compare VQA vs VSR firing using Fisher exact test.

    Matches old LLaVA-MORE pipeline:
    - Global FDR across all layers/features
    - Filter by odds_ratio + freq_diff (not p_adj)
    - Test ALL features (adapted intersection done separately)
    """
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    vqa_firing_dir = Path(RESULTS_BASE) / "analysis" / f"firing{sae_suffix}"
    vsr_firing_dir = Path(RESULTS_BASE) / "analysis" / f"firing_vsr{sae_suffix}"
    out_dir = Path(RESULTS_BASE) / "analysis" / f"spatial_vsr{sae_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load adapted features for post-hoc intersection
    adapted_path = Path(RESULTS_BASE) / "analysis" / f"adapted{sae_suffix}" / "adapted_features_results.csv"
    adapted_by_layer = {}
    if adapted_path.exists():
        import ast, csv
        with open(adapted_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer"])
                indices = ast.literal_eval(row["adapted_indices"])
                adapted_by_layer[layer] = set(indices)
        print(f"[INFO] Adapted features: {sum(len(v) for v in adapted_by_layer.values())} total across {len(adapted_by_layer)} layers")

    ODDS_THR = 2.0
    MIN_FREQ_DIFF = 0.005

    # Collect all rows across all layers
    all_rows = []
    for layer_idx in tqdm(range(N_LAYERS), desc="Loading firing stats"):
        vqa_path = vqa_firing_dir / f"firing_layer_{layer_idx}.json"
        vsr_path = vsr_firing_dir / f"firing_vsr_layer_{layer_idx}.json"

        if not vqa_path.exists() or not vsr_path.exists():
            print(f"  L{layer_idx}: SKIP (missing)")
            continue

        with open(vqa_path) as f:
            vqa_data = json.load(f)
        with open(vsr_path) as f:
            vsr_data = json.load(f)

        # VQA: standard format
        n_vqa = vqa_data["n_all"]
        fire_vqa = np.array(vqa_data["fire_count_all"])

        # VSR: may be in old format (n_vsr, fire_count_vsr) or new format (n_samples, fire_count_all)
        if "n_samples" in vsr_data:
            n_vsr = vsr_data["n_samples"]
            fire_vsr = np.array(vsr_data["fire_count_all"])
        elif "n_vsr" in vsr_data:
            n_vsr = vsr_data["n_vsr"]
            fire_vsr = np.array(vsr_data["fire_count_vsr"])
        else:
            print(f"  L{layer_idx}: SKIP (unknown VSR format, keys: {list(vsr_data.keys())})")
            continue

        if n_vqa == 0 or n_vsr == 0:
            print(f"  L{layer_idx}: SKIP (n_vqa={n_vqa}, n_vsr={n_vsr})")
            continue

        print(f"  L{layer_idx}: n_vqa={n_vqa}, n_vsr={n_vsr}, "
              f"vqa_nonzero={np.count_nonzero(fire_vqa)}, vsr_nonzero={np.count_nonzero(fire_vsr)}")

        for fi in range(D_SAE):
            all_rows.append({
                "layer": layer_idx,
                "feature": fi,
                "c_vqa": int(fire_vqa[fi]),
                "n_vqa": n_vqa,
                "c_vsr": int(fire_vsr[fi]),
                "n_vsr": n_vsr,
            })

    if not all_rows:
        print("[ERROR] No data collected!")
        volume.commit()
        return {"total_spatial": 0, "error": "no data"}

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df)}, layers: {df['layer'].nunique()}")

    # Compute frequencies
    df["freq_vqa"] = df["c_vqa"] / df["n_vqa"]
    df["freq_vsr"] = df["c_vsr"] / df["n_vsr"]
    df["freq_diff"] = df["freq_vsr"] - df["freq_vqa"]

    # Fisher exact test for all features
    print("\nRunning Fisher exact tests...")
    p_values = []
    odds_ratios = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Fisher tests"):
        c_vsr = min(int(r["c_vsr"]), int(r["n_vsr"]))
        c_vqa = min(int(r["c_vqa"]), int(r["n_vqa"]))
        table = [[c_vsr, max(0, int(r["n_vsr"]) - c_vsr)],
                 [c_vqa, max(0, int(r["n_vqa"]) - c_vqa)]]
        try:
            o, p = fisher_exact(table, alternative="greater")
            odds_ratios.append(o if not np.isinf(o) else 1e9)
            p_values.append(p)
        except Exception:
            odds_ratios.append(1.0)
            p_values.append(1.0)

    df["odds_ratio"] = odds_ratios
    df["p_raw"] = p_values

    # Global FDR correction
    df["p_adj"] = multipletests(df["p_raw"].values, method="fdr_bh")[1]

    # Filter: odds_ratio + freq_diff (matching old pipeline)
    spatial = df[(df["odds_ratio"] >= ODDS_THR) & (df["freq_diff"] >= MIN_FREQ_DIFF)].copy()
    spatial = spatial.sort_values("odds_ratio", ascending=False)

    print(f"\n{'='*60}")
    print(f"[RESULT] {len(spatial)} spatial features (before adapted intersection)")
    print(f"  across {spatial['layer'].nunique()} layers")
    spatial.to_csv(out_dir / "spatial_features_vsr_all.csv", index=False)

    # Intersect with adapted
    if adapted_by_layer:
        mask = spatial.apply(
            lambda r: int(r["feature"]) in adapted_by_layer.get(int(r["layer"]), set()),
            axis=1
        )
        final = spatial[mask].copy()
        print(f"[RESULT] {len(final)} spatial+adapted features across {final['layer'].nunique()} layers")
    else:
        final = spatial
        print("[WARN] No adapted features — using all spatial")

    final_dir = Path(RESULTS_BASE) / "analysis" / f"final_features_vsr{sae_suffix}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(final_dir / "final_spatial_visual_features.csv", index=False)
    spatial.to_csv(out_dir / "spatial_features_vsr.csv", index=False)

    # Per-layer summary
    print(f"\n{'='*60}")
    print("Per-layer breakdown:")
    print(f"{'Layer':<8} {'All spatial':>12} {'Adapted∩':>12}")
    for l in range(N_LAYERS):
        n_all = len(spatial[spatial["layer"] == l])
        n_adapted = len(final[final["layer"] == l]) if adapted_by_layer else n_all
        if n_all > 0:
            print(f"  L{l:<5} {n_all:>12} {n_adapted:>12}")

    # Print top features
    print(f"\n{'='*60}")
    print(f"Top 30 (adapted ∩ spatial):")
    print(f"{'Layer':<6} {'Feat':<8} {'OR':>8} {'f_vqa':>8} {'f_vsr':>8} {'diff':>8} {'p_adj':>10}")
    for _, r in final.head(30).iterrows():
        print(f"  L{int(r['layer']):<4} F{int(r['feature']):<6} "
              f"{r['odds_ratio']:>8.2f} {r['freq_vqa']:>8.4f} {r['freq_vsr']:>8.4f} "
              f"{r['freq_diff']:>8.4f} {r['p_adj']:>10.2e}")

    summary = {
        "total_spatial_all": int(len(spatial)),
        "total_spatial_adapted": int(len(final)),
        "odds_threshold": ODDS_THR,
        "min_freq_diff": MIN_FREQ_DIFF,
        "fdr_method": "global_bh",
        "n_vqa": int(df["n_vqa"].iloc[0]),
        "n_vsr": int(df["n_vsr"].iloc[0]),
        "method": "VQA vs VSR sample-level, Fisher exact, global FDR",
    }
    with open(out_dir / "spatial_vsr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    print("Running Fisher exact test: VQA vs VSR firing...")
    summary = fisher_test_vsr.remote()
    print(f"\nResult: {summary}")
    print("\nDone!")
