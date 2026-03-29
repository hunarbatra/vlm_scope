#!/usr/bin/env python3
"""
Simplified selection and plotting of adapted features using only Vision Energy (Ev).

Inputs (produced by compute_feature_metrics.py):
  - plotting_data_global_Ev.npy
  - plotting_data_global_cosines.npy
  - plotting_data_layer_indices.npy

This script:
  1) Loads Ev and cosine similarities
  2) Selects features with Ev > --epsilon and bottom --cosine_percentile% cosine
  3) Saves adapted indices and per-layer summaries
  4) Plots Ev (x-axis, log) vs cosine (y-axis, linear)

CUDA_VISIBLE_DEVICES=1 python features/adapted/select_and_plot_adapted_features.py \
  --metrics_dir /homes/55/lachin/llama-scope-finetune-3/results/stage_2/metrics_run_text_only \
  --output_dir /homes/55/lachin/llama-scope-finetune-3/results/stage_2/adapted_features_ev_gt_0p001 \
  --epsilon 0.01 \
  --cosine_percentile 25.0 \
  --overlay_csv results/stage_3/spatial/spatial_features_vqa.csv /homes/55/lachin/llama-scope-finetune-3/results/stage6/top_200_features_ablation_filtered_renamed_layers_4plus_final_filtered.csv /homes/55/lachin/llama-scope-finetune-3/results/top_200_ocr_features.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def select_adapted_features_ev(ev: np.ndarray, cosines: np.ndarray, epsilon: float, cosine_percentile: float) -> Set[int]:
    """Return global indices where Ev > epsilon AND cosine <= bottom percentile threshold."""
    if ev.shape[0] != cosines.shape[0]:
        raise ValueError("Ev and cosine arrays must have the same length")
    ev_mask = ev > float(epsilon)
    
    # Temporarily ignore cosine threshold - include all features with Ev > epsilon
    if cosine_percentile == 0.0:
        print("WARNING: Ignoring cosine threshold - selecting all features with Ev > epsilon")
        return set(np.where(ev_mask)[0].tolist())
    
    cos_threshold = float(np.percentile(cosines, float(cosine_percentile)))
    cos_mask = cosines <= cos_threshold
    return set(np.where(ev_mask & cos_mask)[0].tolist())


def plot_ev_vs_cosine(ev: np.ndarray, cosines: np.ndarray, adapted_indices: Set[int], out_dir: Path, epsilon: float,
                      overlay_indices: List[np.ndarray] | None = None, overlay_labels: List[str] | None = None,
                      layer_offsets: Dict[int, int] | None = None, layer_counts: Dict[int, int] | None = None):
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

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.scatter(ev, cosines, c='#D3D3D3', alpha=0.3, s=1, label='All Features', zorder=1)

    # Adapted (Ev and low-cosine) first so overlays render on top
    if adapted_indices:
        sel = np.array(sorted(list(adapted_indices)))
        ax.scatter(ev[sel], cosines[sel], c='#F2BECF', alpha=0.9, s=7, label=f'Ev > {epsilon} & low cosine', zorder=2)

    # Overlays on top
    if overlay_indices and len(overlay_indices) > 0:
        palette = ['#6CA6CD', '#FF6B6B', '#7E2FBC', '#7E6BFF', '#FFA41B']
        markers = ['s', 'x', 'd', '^', 'v']
        for i, inds in enumerate(overlay_indices):
            if inds is None or len(inds) == 0:
                continue
            color = palette[i % len(palette)]
            marker = markers[i % len(markers)]
            label = overlay_labels[i] if overlay_labels and i < len(overlay_labels) else f'Overlay {i+1}'
            inds_sorted = np.sort(inds)
            
            # Special handling for "Top Spatial Features" - use red cross, bold, and bigger
            if i == 1 and len(overlay_indices) >= 2:  # Second overlay is "Top Spatial Features"
                ax.scatter(ev[inds_sorted], cosines[inds_sorted], c='#FF6B6B', alpha=0.85, s=13, 
                          marker='x', label=label, zorder=3)
            else:
                ax.scatter(ev[inds_sorted], cosines[inds_sorted], c=color, alpha=0.85, s=10, 
                          marker=marker, label=label, zorder=3)

            # Highlight intersection with adapted on top-most layer
            if adapted_indices:
                inter = np.intersect1d(inds_sorted, np.fromiter(adapted_indices, dtype=np.int64))
                if inter.size > 0:
                    # Special handling for "Top Spatial Features" intersection
                    if i == 1 and len(overlay_indices) >= 2:  # Second overlay is "Top Spatial Features"
                        ax.scatter(ev[inter], cosines[inter], c='#FF6B6B', alpha=1.0, s=16, 
                                  marker='x', linewidths=2, edgecolors='black', 
                                  label=f'{label} (Ev>{epsilon})', zorder=4)
                    else:
                        ax.scatter(ev[inter], cosines[inter], c=color, alpha=1.0, s=16, marker=marker,
                                   edgecolors='black', linewidth=0.6, label=f'{label} (Ev>{epsilon})', zorder=4)

    ax.set_xlabel('Modality Preference (Visual Energy)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Geometric Reorientation (Cosine Similarity)', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of SAE Features', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)

    # Use log scale for Ev if any positive values
    if np.any(ev > 0):
        ax.set_xscale('log')
        pos = ev[ev > 0]
        q001 = float(np.quantile(pos, 0.001)) if pos.size > 10 else float(np.min(pos))
        q999 = float(np.quantile(pos, 0.999)) if pos.size > 10 else float(np.max(pos))
        x_min = max(1e-12, q001)
        x_max = max(x_min * 1.1, q999 * 1.1)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3, which='both')
        ax.minorticks_on()
    else:
        # Fallback to linear if no positive values
        ev_max = float(np.max(ev))
        ev_min = float(np.min(ev))
        pad = 0.05 * (ev_max - ev_min) if ev_max > ev_min else 1e-6
        ax.set_xlim(ev_min - pad, ev_max + pad)
        ax.grid(True, alpha=0.3)

    # Create proper legend with all plotted elements
    legend_elements = []
    
    # Add "All Features" to legend
    total_features = ev.shape[0]
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D3D3D3', 
                                     markersize=8, alpha=0.3, label=f'All Features ({total_features:,})'))
    
    # Add adapted features to legend if they exist
    if adapted_indices:
        adapted_count = len(adapted_indices)
        adapted_percentage = (adapted_count / total_features) * 100
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F2BECF', 
                                         markersize=8, alpha=0.9, label=f'Adapted Features ({adapted_count:,})'))
        # Add adapted percentage as a separate legend entry
        legend_elements.append(plt.Line2D([0], [0], marker='', color='w', 
                                         label=f'Adapted Percentage: {adapted_percentage:.2f}%'))
    
    # Add overlay features to legend
    if overlay_indices and len(overlay_indices) > 0:
        palette = ['#6CA6CD', '#FF6B6B', '#7E2FBC', '#7E2FBC', '#FFA41B']  # third and fourth overlays are purple (#7E2FBC)
        markers = ['s', 'x', 'd', '^', 'v']
        # Define specific labels for overlays
        specific_labels = ['Spatial Candidates', 'Top Spatial Features']
        
        for i, inds in enumerate(overlay_indices):
            if inds is None or len(inds) == 0:
                continue
            color = palette[i % len(palette)]
            marker = markers[i % len(markers)]
            # Use specific labels if available, otherwise fall back to overlay_labels or default
            if i < len(specific_labels):
                base_label = specific_labels[i]
            else:
                base_label = overlay_labels[i] if overlay_labels and i < len(overlay_labels) else f'Overlay {i+1}'
            # Add count to the label
            count = len(inds)
            label = f'{base_label} ({count:,})'
            
            # Special handling for "Top Spatial Features" - use red cross, bold, and bigger
            if base_label == 'Top Spatial Features':
                legend_elements.append(plt.Line2D([0], [0], marker='x', color='#FF6B6B', 
                                                 markersize=10, markeredgewidth=2, alpha=0.75, label=label))
            else:
                legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, 
                                                 markersize=8, alpha=0.85, label=label))
    
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02), fontsize=12, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.9, facecolor='white', edgecolor='gray')
    plt.tight_layout()
    plt.savefig(out_dir / "global_feature_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Select features using Ev > epsilon and plot")
    parser.add_argument("--metrics_dir", required=True, help="Directory with precomputed metrics")
    parser.add_argument("--output_dir", required=True, help="Directory to save selection and plots")
    parser.add_argument("--epsilon", type=float, default=0.001, help="Threshold for selecting Ev > epsilon")
    parser.add_argument("--cosine_percentile", type=float, default=20.0, help="Bottom percentile for cosine (low similarity)")
    parser.add_argument("--overlay_csv", type=str, nargs='*', default=[], help="Paths to CSVs (columns: layer,feature[,name]) to overlay")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Ev, cosine, and layer indices
    ev = np.load(metrics_dir / "plotting_data_global_Ev.npy")
    cosines = np.load(metrics_dir / "plotting_data_global_cosines.npy")
    layer_indices = np.load(metrics_dir / "plotting_data_layer_indices.npy")

    # Compute per-layer counts and offsets
    layer_counts: Dict[int, int] = {}
    for l in layer_indices:
        li = int(l)
        layer_counts[li] = layer_counts.get(li, 0) + 1
    layer_offsets: Dict[int, int] = {}
    _offset = 0
    for li in sorted(layer_counts.keys()):
        layer_offsets[li] = _offset
        _offset += layer_counts[li]

    # Load overlays
    overlay_indices: List[np.ndarray] = []
    overlay_labels: List[str] = []
    for csv_path in args.overlay_csv:
        try:
            df = pd.read_csv(csv_path)
            overlay_set: Set[int] = set()
            for _, row in df.iterrows():
                layer = int(row['layer'])
                feat = int(row['feature'])
                if layer in layer_offsets and layer in layer_counts and 0 <= feat < layer_counts[layer]:
                    overlay_set.add(layer_offsets[layer] + feat)
            overlay_indices.append(np.array(sorted(list(overlay_set)), dtype=np.int64))
            overlay_labels.append(Path(csv_path).stem)
            print(f"Loaded overlay CSV '{csv_path}' with {len(overlay_set)} features")
        except Exception as e:
            print(f"Warning: Failed to load overlay CSV '{csv_path}': {e}")

    # Select adapted features by Ev > epsilon AND low cosine (bottom percentile)
    adapted_indices = select_adapted_features_ev(ev, cosines, args.epsilon, args.cosine_percentile)
    adapted_global = np.array(sorted(list(adapted_indices)), dtype=np.int64)

    # Print intersections of overlays with adapted features
    if overlay_indices:
        print("\nOverlay intersections with adapted features:")
        spans = [(li, layer_offsets[li], layer_offsets[li] + layer_counts[li]) for li in sorted(layer_counts.keys())]

        def global_to_layer_feature(gidx: int):
            for li, start, end in spans:
                if start <= gidx < end:
                    return li, int(gidx - start)
            return None, None

        for i, (label, inds) in enumerate(zip(overlay_labels, overlay_indices)):
            if inds is None or len(inds) == 0:
                print(f"  {label}: 0 in overlay (skipped)")
                continue
            inter = np.intersect1d(inds, adapted_global)
            count = int(inter.size)
            print(f"  {label}: {count} intersecting features")
            if count > 0:
                max_show = 100
                pairs = []
                for gi in inter[:max_show]:
                    L, f = global_to_layer_feature(int(gi))
                    if L is not None:
                        pairs.append((int(L), int(f)))
                print(f"    Layer-Feature pairs (first {min(max_show, count)}): {pairs}")
                if count > max_show:
                    print(f"    ... {count - max_show} more")
            
            # Also show non-intersecting features for each overlay
            non_intersecting = np.setdiff1d(inds, adapted_global)
            non_intersecting_count = int(non_intersecting.size)
            if non_intersecting_count > 0:
                print(f"  {label}: {non_intersecting_count} non-intersecting features")
                max_show = 100
                pairs = []
                ev_values = []
                cosine_values = []
                for gi in non_intersecting[:max_show]:
                    L, f = global_to_layer_feature(int(gi))
                    if L is not None:
                        pairs.append((int(L), int(f)))
                        # Get Ev and cosine values for this feature
                        ev_value = ev[gi]
                        cosine_value = cosines[gi]
                        ev_values.append(ev_value)
                        cosine_values.append(cosine_value)
                print(f"    Layer-Feature pairs (first {min(max_show, non_intersecting_count)}): {pairs}")
                print(f"    Ev values (first {min(max_show, non_intersecting_count)}): {ev_values}")
                print(f"    Cosine values (first {min(max_show, non_intersecting_count)}): {cosine_values}")
                
                # Print detailed analysis
                print(f"    Analysis:")
                print(f"      Ev threshold: {args.epsilon}")
                print(f"      Cosine percentile threshold: {args.cosine_percentile}%")
                if args.cosine_percentile > 0.0:
                    cos_threshold = float(np.percentile(cosines, float(args.cosine_percentile)))
                    print(f"      Cosine threshold value: {cos_threshold:.6f}")
                    
                    # Count how many fail each criterion
                    ev_failures = sum(1 for ev_val in ev_values if ev_val <= args.epsilon)
                    cos_failures = sum(1 for cos_val in cosine_values if cos_val > cos_threshold)
                    print(f"      Features failing Ev criterion (Ev <= {args.epsilon}): {ev_failures}/{len(ev_values)}")
                    print(f"      Features failing cosine criterion (cos > {cos_threshold:.6f}): {cos_failures}/{len(cosine_values)}")
                else:
                    # When cosine_percentile is 0.0, only Ev criterion applies
                    ev_failures = sum(1 for ev_val in ev_values if ev_val <= args.epsilon)
                    print(f"      Features failing Ev criterion (Ev <= {args.epsilon}): {ev_failures}/{len(ev_values)}")
                    print(f"      (Cosine criterion ignored)")
                
                if non_intersecting_count > max_show:
                    print(f"    ... {non_intersecting_count - max_show} more")

    # Print non-intersecting features from the second overlay (if it exists)
    if len(overlay_indices) >= 2 and overlay_indices[1] is not None and len(overlay_indices[1]) > 0:
        print(f"\nNon-intersecting features from second overlay ({overlay_labels[1]}):")
        second_overlay = overlay_indices[1]
        non_intersecting = np.setdiff1d(second_overlay, adapted_global)
        non_intersecting_count = int(non_intersecting.size)
        print(f"  {overlay_labels[1]}: {non_intersecting_count} non-intersecting features")
        
        if non_intersecting_count > 0:
            max_show = 100
            pairs = []
            ev_values = []
            cosine_values = []
            for gi in non_intersecting[:max_show]:
                L, f = global_to_layer_feature(int(gi))
                if L is not None:
                    pairs.append((int(L), int(f)))
                    # Get Ev and cosine values for this feature
                    ev_value = ev[gi]
                    cosine_value = cosines[gi]
                    ev_values.append(ev_value)
                    cosine_values.append(cosine_value)
            print(f"    Layer-Feature pairs (first {min(max_show, non_intersecting_count)}): {pairs}")
            print(f"    Ev values (first {min(max_show, non_intersecting_count)}): {ev_values}")
            print(f"    Cosine values (first {min(max_show, non_intersecting_count)}): {cosine_values}")
            
            # Print detailed analysis
            print(f"    Analysis:")
            print(f"      Ev threshold: {args.epsilon}")
            print(f"      Cosine percentile threshold: {args.cosine_percentile}%")
            cos_threshold = float(np.percentile(cosines, float(args.cosine_percentile)))
            print(f"      Cosine threshold value: {cos_threshold:.6f}")
            
            # Count how many fail each criterion
            ev_failures = sum(1 for ev_val in ev_values if ev_val <= args.epsilon)
            cos_failures = sum(1 for cos_val in cosine_values if cos_val > cos_threshold)
            print(f"      Features failing Ev criterion (Ev <= {args.epsilon}): {ev_failures}/{len(ev_values)}")
            print(f"      Features failing cosine criterion (cos > {cos_threshold:.6f}): {cos_failures}/{len(cosine_values)}")
            
            if non_intersecting_count > max_show:
                print(f"    ... {non_intersecting_count - max_show} more")
    elif len(overlay_indices) < 2:
        print("\nNo second overlay found to analyze non-intersecting features")

    # Build per-layer results
    results: List[Dict] = []
    layer_start = 0
    for layer_idx in sorted(layer_counts.keys()):
        count = layer_counts[layer_idx]
        cos_slice = cosines[layer_start:layer_start + count]
        ev_slice = ev[layer_start:layer_start + count]

        if adapted_global.size > 0:
            in_layer_mask = (adapted_global >= layer_start) & (adapted_global < layer_start + count)
            layer_adapted_local = (adapted_global[in_layer_mask] - layer_start).astype(int).tolist()
        else:
            layer_adapted_local = []

        result = {
            "layer": layer_idx,
            "num_features": int(count),
            "num_adapted": int(len(layer_adapted_local)),
            "adapted_indices": sorted(layer_adapted_local),
            "adapted_data": {int(idx): {"Ev": float(ev_slice[idx]), "cosine_similarity": float(cos_slice[idx])} for idx in layer_adapted_local},
            "mean_cosine": float(np.mean(cos_slice)) if len(cos_slice) else float("nan"),
            "mean_Ev": float(np.mean(ev_slice)) if len(ev_slice) else float("nan"),
            "std_Ev": float(np.std(ev_slice)) if len(ev_slice) else float("nan"),
        }
        if len(layer_adapted_local) > 0:
            result.update({
                "adapted_mean_cosine": float(np.mean(cos_slice[layer_adapted_local])),
                "adapted_mean_Ev": float(np.mean(ev_slice[layer_adapted_local])),
                "adapted_min_cosine": float(np.min(cos_slice[layer_adapted_local])),
                "adapted_max_Ev": float(np.max(ev_slice[layer_adapted_local])),
            })

        results.append(result)
        layer_start += count

    # Save selection outputs
    np.save(out_dir / "adapted_indices.npy", adapted_global)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "adapted_features_results.csv", index=False)

    # Helper to convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(out_dir / "adapted_features_results.json", 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    # Plot
    plot_ev_vs_cosine(ev, cosines, adapted_indices, out_dir, args.epsilon,
                      overlay_indices=overlay_indices, overlay_labels=overlay_labels,
                      layer_offsets=layer_offsets, layer_counts=layer_counts)

    # Summary
    total_adapted = int(len(adapted_indices))
    total_features = int(cosines.shape[0])
    print(f"Selected {total_adapted} features (Ev > {args.epsilon}, cosine <= P{args.cosine_percentile}%) out of {total_features} ({(total_adapted/total_features)*100:.2f}%)")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()


