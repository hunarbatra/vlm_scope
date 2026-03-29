#!/usr/bin/env python3
"""
Plot aggregated attribution results:
- Reads aggregated_attribution_from_all_features.json
- Reconstructs average normalized layer distributions (per absolute layer index)
- Reconstructs layer×head score matrices from aggregated head scores
- Produces publication-style plots similar to attribution_patching.py


python experiments/plot_aggregated_attribution.py \
  --input-json results/experiments/aggregated_attribution_from_passed_features_odds_gt_3.json \
  --output-dir results/experiments/plots_odds_gt_3

python experiments/plot_aggregated_attribution.py \
  --input-json results/experiments/filtered.json \
  --output-dir results/experiments/plots_filtered \
  --top-k-heads 30  
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def configure_matplotlib_style() -> None:
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': False,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def reconstruct_layer_vector(avg_layer_distribution: List[Dict[str, float]]) -> np.ndarray:
    if not avg_layer_distribution:
        return np.zeros((0,), dtype=np.float32)
    max_layer = max(int(x["layer"]) for x in avg_layer_distribution)
    vec = np.zeros((max_layer + 1,), dtype=np.float32)
    for item in avg_layer_distribution:
        li = int(item["layer"])
        vec[li] = float(item["avg_norm_score"].__float__() if hasattr(item["avg_norm_score"], "__float__") else item["avg_norm_score"])  # robust cast
    return vec


def reconstruct_layer_head_matrix(head_scores: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str], List[str]]:
    # head_scores: list of {name: "LxHy", total_norm_score: float}
    if not head_scores:
        return np.zeros((0, 0), dtype=np.float32), [], []
    layers, heads = set(), set()
    parsed: List[Tuple[int, int, float]] = []
    for h in head_scores:
        name = str(h.get("name", ""))
        if not name.startswith("L") or "H" not in name:
            continue
        try:
            l_idx = int(name.split("H")[0][1:])
            h_idx = int(name.split("H")[1])
        except Exception:
            continue
        score = float(h.get("total_norm_score", 0.0))
        parsed.append((l_idx, h_idx, score))
        layers.add(l_idx)
        heads.add(h_idx)

    if not parsed:
        return np.zeros((0, 0), dtype=np.float32), [], []

    max_layer = max(layers)
    max_head = max(heads)
    mat = np.zeros((max_layer + 1, max_head + 1), dtype=np.float32)
    for l, h, v in parsed:
        if l >= 0 and h >= 0:
            mat[l, h] = v

    layer_labels = [f"L{l}" for l in range(mat.shape[0])]
    head_labels = [f"H{h}" for h in range(mat.shape[1])]
    return mat, layer_labels, head_labels


def plot_layer_line(vec: np.ndarray, title: str, color: str, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(range(len(vec)), vec, color=color, linewidth=3, marker='o', markersize=6,
            alpha=0.85, markerfacecolor='white', markeredgewidth=2)
    ax.set_xlabel("Layer Index", fontsize=14, fontweight='bold')
    ax.set_ylabel("Avg normalized attribution", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    if len(vec) > 0:
        step = max(1, max(1, len(vec) // 10))
        ax.set_xticks(range(0, len(vec), step))
    ax.tick_params(axis='both', which='major', labelsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_head_heatmap(mat: np.ndarray, title: str, cmap: str, out_path: Path) -> None:
    if mat.size == 0:
        # create an empty placeholder
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No head data", ha='center', va='center')
        ax.axis('off')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.heatmap(mat, cmap=cmap, xticklabels=range(mat.shape[1]), yticklabels=range(mat.shape[0]),
                cbar_kws={"label": "Aggregated normalized score", "shrink": 0.8}, ax=ax,
                square=True, linewidths=0, cbar=True)
    ax.set_xlabel("Attention Head", fontsize=14, fontweight='bold')
    ax.set_ylabel("Layer Index", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # Tweak colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_ylabel("Aggregated normalized score", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_top_heads_bar(head_scores: List[Dict[str, float]], top_k: int, title: str, color: str, out_path: Path) -> None:
    if not head_scores:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No head data", ha='center', va='center')
        ax.axis('off')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    sorted_scores = sorted(head_scores, key=lambda x: float(x.get("total_norm_score", 0.0)), reverse=True)[:top_k]
    names = [s.get("name", "?") for s in sorted_scores]
    vals = [float(s.get("total_norm_score", 0.0)) for s in sorted_scores]
    fig, ax = plt.subplots(1, 1, figsize=(min(14, 1 + 0.5 * len(names)), 6))
    ax.bar(range(len(names)), vals, color=color, alpha=0.9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Aggregated normalized score", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregated attribution results")
    parser.add_argument("--input-json", type=Path, default=Path(
        "/homes/55/lachin/llama-scope-finetune-3/results/experiments/aggregated_attribution_from_all_features.json"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/homes/55/lachin/llama-scope-finetune-3/results/experiments/plots"
    ))
    parser.add_argument("--top-k-heads", type=int, default=25)
    args = parser.parse_args()

    configure_matplotlib_style()

    payload = json.loads(args.input_json.read_text())
    methods = payload.get("methods", {})

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Colors chosen similar to attribution_patching.py
    method_colors = {
        "A": "#E84D8A",   # pink/magenta
        "B": "#FEB326",   # golden
    }
    method_cmaps = {
        "A": "magma",
        "B": "cividis",
    }

    for method_key in ("A", "B"):
        m = methods.get(method_key, {})
        # Layer vector
        layer_vec = reconstruct_layer_vector(m.get("avg_layer_distribution", []))
        plot_layer_line(
            layer_vec,
            title=("Layer-wise Aggregated Attribution (Method %s)" % method_key),
            color=method_colors.get(method_key, "#333333"),
            out_path=args.output_dir / ("layer_scores_method_%s.png" % method_key),
        )
        # Heads matrix
        mat, _, _ = reconstruct_layer_head_matrix(m.get("aggregated_top_heads", []))
        plot_head_heatmap(
            mat,
            title=("Attention Head Aggregated Attribution (Method %s)" % method_key),
            cmap=method_cmaps.get(method_key, "viridis"),
            out_path=args.output_dir / ("head_scores_method_%s.png" % method_key),
        )
        # Top-K heads bar chart
        plot_top_heads_bar(
            m.get("aggregated_top_heads", []),
            top_k=args.top_k_heads,
            title=("Top-%d Heads (Aggregated, Method %s)" % (args.top_k_heads, method_key)),
            color=method_colors.get(method_key, "#333333"),
            out_path=args.output_dir / ("top_heads_method_%s.png" % method_key),
        )


if __name__ == "__main__":
    main()
