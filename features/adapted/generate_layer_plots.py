#!/usr/bin/env python3
"""
Minimal script to generate adapted features per layer plots from saved results.

This script loads the saved adapted_features_results.csv file and creates:
1. adapted_features_per_layer.png - Bar chart of adapted features per layer
2. cosine_similarity_comparison.png - Line plot comparing overall vs adapted cosine similarities

Usage:
    python features/adapted/generate_layer_plots.py --results_dir results/stage_2/adapted_features/text-only_50k_20
    python features/adapted/generate_layer_plots.py --results_dir results/stage_2/adapted_features/text-only_50k_15_35_sensitivity
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_adapted_analysis(results: list, out_dir: Path):
    """Create comprehensive visualization of adapted feature analysis."""
    # Filter valid results
    valid_results = [r for r in results if 'num_adapted' in r]
    if not valid_results:
        print("No valid data for plotting")
        return
    
    df = pd.DataFrame(valid_results)
    
    # 1. Number of adapted features per layer - separate plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.bar(df['layer'], df['num_adapted'], color='#4A90E2', alpha=0.8)  # Consistent blue
    ax1.set_xlabel('Layer Index', fontsize=14, fontfamily='serif', fontweight='bold')
    ax1.set_ylabel('Number of Adapted Features', fontsize=14, fontfamily='serif', fontweight='bold')
    # ax1.set_title('Adapted Features per Layer (Global Selection)', fontsize=14, fontfamily='serif', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Apply serif fonts to tick labels
    ax1.tick_params(axis='both', which='major', labelsize=10)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.savefig(out_dir / "adapted_features_per_layer.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cosine similarity: Overall vs Adapted comparison - separate plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(df['layer'], df['mean_cosine'], 'o-', color='#4A90E2', label='Overall Mean', linewidth=2.5, markersize=6)  # Consistent blue
    
    # Add adapted feature statistics if available
    adapted_data = df[df['adapted_mean_cosine'].notna()]
    if len(adapted_data) > 0:
        ax2.plot(adapted_data['layer'], adapted_data['adapted_mean_cosine'], 
                'o-', color='#F5A623', label='Adapted Mean', linewidth=2.5, markersize=6)  # Consistent orange
    
    ax2.set_xlabel('Layer Index', fontsize=14, fontfamily='serif', fontweight='bold')
    ax2.set_ylabel('Cosine Similarity', fontsize=14, fontfamily='serif', fontweight='bold')
    # ax2.set_title('Cosine Similarity: Overall vs Adapted Features', fontsize=14, fontfamily='serif', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12, frameon=False)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Apply serif fonts to tick labels
    ax2.tick_params(axis='both', which='major', labelsize=10)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontfamily('serif')
    
    # Apply serif fonts and bold to legend text
    legend = ax2.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_similarity_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Adapted features per layer plot saved to: {out_dir / 'adapted_features_per_layer.png'}")
    print(f"Cosine similarity comparison plot saved to: {out_dir / 'cosine_similarity_comparison.png'}")

def load_results_from_csv(csv_path: Path):
    """Load results from CSV file and convert to the expected format."""
    print(f"Loading results from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Convert DataFrame to list of dictionaries (expected format)
    results = []
    for _, row in df.iterrows():
        result = {
            'layer': int(row['layer']),
            'num_features': int(row['num_features']),
            'num_adapted': int(row['num_adapted']),
            'mean_cosine': float(row['mean_cosine']) if pd.notna(row['mean_cosine']) else float('nan'),
            'mean_Ev': float(row['mean_Ev']) if pd.notna(row['mean_Ev']) else float('nan'),
            'std_Ev': float(row['std_Ev']) if pd.notna(row['std_Ev']) else float('nan'),
        }
        
        # Add adapted feature statistics if available
        if 'adapted_mean_cosine' in row and pd.notna(row['adapted_mean_cosine']):
            result['adapted_mean_cosine'] = float(row['adapted_mean_cosine'])
        if 'adapted_mean_Ev' in row and pd.notna(row['adapted_mean_Ev']):
            result['adapted_mean_Ev'] = float(row['adapted_mean_Ev'])
        if 'adapted_min_cosine' in row and pd.notna(row['adapted_min_cosine']):
            result['adapted_min_cosine'] = float(row['adapted_min_cosine'])
        if 'adapted_max_Ev' in row and pd.notna(row['adapted_max_Ev']):
            result['adapted_max_Ev'] = float(row['adapted_max_Ev'])
            
        results.append(result)
    
    print(f"Loaded {len(results)} layer results")
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate adapted features per layer plots from saved results")
    parser.add_argument("--results_dir", required=True, help="Directory containing adapted_features_results.csv")
    parser.add_argument("--output_dir", help="Output directory for plots (defaults to results_dir)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    
    # Check if results file exists
    csv_path = results_dir / "adapted_features_results.csv"
    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        print("Make sure the directory contains adapted_features_results.csv")
        return
    
    # Load results
    results = load_results_from_csv(csv_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"Generating plots in {output_dir}...")
    plot_adapted_analysis(results, output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
