#!/usr/bin/env python3
"""
Sweep epsilon (Ev threshold) and cosine percentile thresholds to assess selection robustness.

Loads precomputed arrays produced by compute_feature_metrics.py and cosine similarities
to evaluate how sensitive adapted-feature selection is to threshold choices.

Outputs (in --output_dir):
  - robustness_summary.csv                 # metrics per (epsilon, cosine_percentile)
  - heatmap_counts.png                     # selected feature count heatmap
  - heatmap_jaccard.png                    # Jaccard vs baseline heatmap
  - heatmap_layer_corr.png                 # Per-layer count correlation vs baseline heatmap
  - heatmaps_grid.png                      # combined figure for quick inspection
  - baseline_summary.json                  # baseline stats and parameters

Example:
  python features/adapted/sweep_threshold_robustness.py \
    --metrics_dir /homes/55/lachin/llama-scope-finetune-3/results/stage_2/metrics_run_text_only \
    --output_dir /homes/55/lachin/llama-scope-finetune-3/results/stage_2/robustness_ev1e-3_cos25 \
    --base_epsilon 1e-3 --base_cosine_percentile 25 \
    --eps_values 5e-4 1e-3 2e-3 5e-3 \
    --cos_percentiles 15 20 25 30 35
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


def configure_matplotlib_style():
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })


def select_indices(ev: np.ndarray, cosines: np.ndarray, epsilon: float, cosine_percentile: float) -> np.ndarray:
    """Select global indices with Ev > epsilon and cosine <= bottom percentile threshold.

    If cosine_percentile == 0.0, ignore cosine threshold (Ev-only selection),
    matching the selection behavior in select_and_plot_adapted_features.py.
    """
    mask = ev > float(epsilon)
    if float(cosine_percentile) > 0.0:
        threshold = float(np.percentile(cosines, float(cosine_percentile)))
        mask &= (cosines <= threshold)
    return np.where(mask)[0]


def counts_by_layer(selected_indices: np.ndarray, layer_indices: np.ndarray, num_layers: int | None = None) -> np.ndarray:
    if num_layers is None:
        num_layers = int(layer_indices.max()) + 1 if layer_indices.size else 0
    return np.bincount(layer_indices[selected_indices], minlength=num_layers)


def safe_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.union1d(a, b)
    if union.size == 0:
        return float("nan")
    inter = np.intersect1d(a, b)
    return float(inter.size) / float(union.size)


def retention_and_precision(candidate: np.ndarray, baseline: np.ndarray) -> Tuple[float, float]:
    inter = np.intersect1d(candidate, baseline)
    retention = float(inter.size) / float(baseline.size) if baseline.size else float("nan")
    precision = float(inter.size) / float(candidate.size) if candidate.size else float("nan")
    return retention, precision


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_grid_metrics(ev: np.ndarray, cos: np.ndarray, layer_idx: np.ndarray,
                         eps_values: List[float], cos_percentiles: List[float],
                         base_eps: float, base_pct: float) -> Dict[str, np.ndarray | List[float]]:
    """Compute metrics across epsilon x cosine grid and return matrices.

    Returns dict with keys:
      - counts, jaccard, retention, precision, layer_corr: arrays shape (len(eps), len(pct))
    """
    eps_values = [float(e) for e in eps_values]
    cos_percentiles = [float(p) for p in cos_percentiles]

    num_layers = int(layer_idx.max()) + 1 if layer_idx.size else 0

    baseline_sel = np.array(sorted(select_indices(ev, cos, base_eps, base_pct)), dtype=np.int64)
    baseline_counts = counts_by_layer(baseline_sel, layer_idx, num_layers=num_layers)

    E = len(eps_values)
    P = len(cos_percentiles)
    counts = np.zeros((E, P), dtype=np.int64)
    jaccard = np.full((E, P), np.nan, dtype=np.float64)
    retention = np.full((E, P), np.nan, dtype=np.float64)
    precision = np.full((E, P), np.nan, dtype=np.float64)
    layer_corr = np.full((E, P), np.nan, dtype=np.float64)

    for i, eps in enumerate(eps_values):
        for j, pct in enumerate(cos_percentiles):
            sel = np.array(sorted(select_indices(ev, cos, eps, pct)), dtype=np.int64)
            counts[i, j] = sel.size
            jaccard[i, j] = safe_jaccard(sel, baseline_sel)
            r, p = retention_and_precision(sel, baseline_sel)
            retention[i, j] = r
            precision[i, j] = p
            sel_counts = counts_by_layer(sel, layer_idx, num_layers=num_layers)
            layer_corr[i, j] = pearson_corr(sel_counts, baseline_counts)

    return {
        "counts": counts,
        "jaccard": jaccard,
        "retention": retention,
        "precision": precision,
        "layer_corr": layer_corr,
        "eps_values": eps_values,
        "cos_percentiles": cos_percentiles,
        "baseline_count": int(baseline_sel.size),
    }


def plot_heatmap(matrix: np.ndarray, eps_values: List[float], pct_values: List[float],
                 title: str, cbar_label: str, out_path: Path,
                 norm=None, cmap: str = 'viridis', baseline_loc: Tuple[int, int] | None = None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(matrix, aspect='auto', origin='lower', cmap=cmap, norm=norm)
    ax.set_xlabel('Cosine Percentile (bottom)')
    ax.set_ylabel('Epsilon (Ev threshold)')
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(pct_values)))
    ax.set_xticklabels([f"{p:.0f}%" for p in pct_values])

    # Use scientific notation for epsilon ticks when appropriate
    def fmt_eps(e: float) -> str:
        return f"{e:.0e}" if (e < 0.01 or e >= 100) else f"{e:g}"

    ax.set_yticks(range(len(eps_values)))
    ax.set_yticklabels([fmt_eps(e) for e in eps_values])

    if baseline_loc is not None:
        bi, bj = baseline_loc
        ax.scatter([bj], [bi], s=80, marker='o', facecolors='none', edgecolors='white', linewidths=2, label='Baseline')
        ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True, framealpha=0.9, facecolor='white', edgecolor='gray')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_combined_grid(metrics: Dict[str, np.ndarray | List[float]], base_idx: Tuple[int, int], out_path: Path):
    eps_values: List[float] = metrics["eps_values"]  # type: ignore[index]
    pct_values: List[float] = metrics["cos_percentiles"]  # type: ignore[index]
    counts = metrics["counts"]  # type: ignore[index]
    jacc = metrics["jaccard"]  # type: ignore[index]
    corr = metrics["layer_corr"]  # type: ignore[index]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].imshow(counts, aspect='auto', origin='lower', cmap='magma', norm=LogNorm(vmin=max(1, int(np.nanmin(counts))), vmax=max(1, int(np.nanmax(counts)))))
    axes[0].set_title('Selected Count', fontsize=14, fontweight='bold')

    im1 = axes[1].imshow(jacc, aspect='auto', origin='lower', cmap='viridis', vmin=0.0, vmax=1.0)
    axes[1].set_title('Jaccard vs Baseline', fontsize=14, fontweight='bold')

    im2 = axes[2].imshow(corr, aspect='auto', origin='lower', cmap='coolwarm', norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
    axes[2].set_title('Per-layer Count Correlation', fontsize=14, fontweight='bold')

    for ax in axes:
        ax.set_xlabel('Cosine Percentile (bottom)')
        ax.set_xticks(range(len(pct_values)))
        ax.set_xticklabels([f"{p:.0f}%" for p in pct_values])
        ax.set_ylabel('Epsilon (Ev threshold)')
        ax.set_yticks(range(len(eps_values)))
        ax.set_yticklabels([f"{e:.0e}" if (e < 0.01 or e >= 100) else f"{e:g}" for e in eps_values])
        bi, bj = base_idx
        ax.scatter([bj], [bi], s=70, marker='o', facecolors='none', edgecolors='white', linewidths=2)

    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label('Count')
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label('Jaccard')
    cbar2 = fig.colorbar(im2, ax=axes[2])
    cbar2.set_label('Pearson r')

    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sweep epsilon and cosine percentile thresholds for robustness analysis")
    parser.add_argument("--metrics_dir", required=True, help="Directory with plotting_data_global_Ev.npy, plotting_data_global_cosines.npy, plotting_data_layer_indices.npy")
    parser.add_argument("--output_dir", required=True, help="Directory to write robustness outputs")
    parser.add_argument("--base_epsilon", type=float, default=1e-3, help="Baseline Ev threshold")
    parser.add_argument("--base_cosine_percentile", type=float, default=25.0, help="Baseline bottom percentile for cosine")
    parser.add_argument("--eps_values", type=float, nargs='*', default=[5e-4, 1e-3, 2e-3, 5e-3], help="List of epsilon values to sweep")
    parser.add_argument("--cos_percentiles", type=float, nargs='*', default=[15.0, 20.0, 25.0, 30.0, 35.0], help="List of bottom cosine percentiles to sweep")
    args = parser.parse_args()

    configure_matplotlib_style()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load arrays
    ev = np.load(metrics_dir / "plotting_data_global_Ev.npy")
    cos = np.load(metrics_dir / "plotting_data_global_cosines.npy")
    layer_indices = np.load(metrics_dir / "plotting_data_layer_indices.npy").astype(int)

    # Compute grid metrics
    metrics = compute_grid_metrics(
        ev=ev,
        cos=cos,
        layer_idx=layer_indices,
        eps_values=list(args.eps_values),
        cos_percentiles=list(args.cos_percentiles),
        base_eps=float(args.base_epsilon),
        base_pct=float(args.base_cosine_percentile),
    )

    eps_values: List[float] = metrics["eps_values"]  # type: ignore[index]
    pct_values: List[float] = metrics["cos_percentiles"]  # type: ignore[index]
    counts = metrics["counts"]  # type: ignore[index]
    jaccard = metrics["jaccard"]  # type: ignore[index]
    retention = metrics["retention"]  # type: ignore[index]
    precision = metrics["precision"]  # type: ignore[index]

    # Locate baseline indices on the grid, if present
    try:
        base_i = eps_values.index(float(args.base_epsilon))
    except ValueError:
        base_i = int(np.argmin(np.abs(np.array(eps_values) - float(args.base_epsilon))))
    try:
        base_j = pct_values.index(float(args.base_cosine_percentile))
    except ValueError:
        base_j = int(np.argmin(np.abs(np.array(pct_values) - float(args.base_cosine_percentile))))

    # Write summary CSV
    records: List[Dict] = []
    for i, eps in enumerate(eps_values):
        for j, pct in enumerate(pct_values):
            records.append({
                "epsilon": float(eps),
                "cosine_percentile": float(pct),
                "count": int(counts[i, j]),
                "jaccard_vs_baseline": float(jaccard[i, j]) if np.isfinite(jaccard[i, j]) else float("nan"),
                "retention_wrt_baseline": float(retention[i, j]) if np.isfinite(retention[i, j]) else float("nan"),
                "precision_wrt_baseline": float(precision[i, j]) if np.isfinite(precision[i, j]) else float("nan"),
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir / "robustness_summary.csv", index=False)

    # Save baseline summary
    baseline_summary = {
        "base_epsilon": float(args.base_epsilon),
        "base_cosine_percentile": float(args.base_cosine_percentile),
        "baseline_selected_count": int(metrics["baseline_count"]),  # type: ignore[index]
        "grid_eps_values": [float(e) for e in eps_values],
        "grid_cos_percentiles": [float(p) for p in pct_values],
    }
    with open(out_dir / "baseline_summary.json", "w") as f:
        json.dump(baseline_summary, f, indent=2)

    # Individual heatmaps
    plot_heatmap(
        counts.astype(float), eps_values, pct_values,
        title="Selected Feature Count",
        cbar_label="Count",
        out_path=out_dir / "heatmap_counts.png",
        norm=LogNorm(vmin=max(1, int(np.nanmin(counts))), vmax=max(1, int(np.nanmax(counts)))),
        cmap='magma',
        baseline_loc=(base_i, base_j),
    )
    plot_heatmap(
        jaccard, eps_values, pct_values,
        title="Jaccard vs Baseline",
        cbar_label="Jaccard",
        out_path=out_dir / "heatmap_jaccard.png",
        norm=None, cmap='viridis', baseline_loc=(base_i, base_j),
    )
    plot_heatmap(
        metrics["layer_corr"], eps_values, pct_values,  # type: ignore[index]
        title="Per-layer Count Correlation",
        cbar_label="Pearson r",
        out_path=out_dir / "heatmap_layer_corr.png",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0), cmap='coolwarm', baseline_loc=(base_i, base_j),
    )

    # Combined grid
    plot_combined_grid(metrics, base_idx=(base_i, base_j), out_path=out_dir / "heatmaps_grid.png")

    print(f"Robustness sweep complete. Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()





