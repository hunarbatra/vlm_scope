#!/usr/bin/env python3
"""
Script to extract features that are BOTH in the top X% cosine similarity AND bottom Y% variance gap.
Run this from the project root directory.

Usage:
    python features/adapted/extract_top_features.py                                    # Use default paths (5% each)
    python features/adapted/extract_top_features.py --data-dir path/to/data            # Custom data directory
    python features/adapted/extract_top_features.py --output-dir path/to/output        # Custom output directory
    python features/adapted/extract_top_features.py --cosine-percent 10 --variance-percent 5  # Custom percentages
    python features/adapted/extract_top_features.py --help                            # Show all options

The script will:
1. Load the saved plotting data from the specified adapted features directory
2. Identify the top X% highest cosine similarity features
3. Identify the bottom Y% variance gap features
4. Find the intersection (features that are in BOTH sets)
5. Save intersection features to a CSV file with layer and feature columns

Examples:
    # Use default paths (5% cosine, 5% variance)
    python features/adapted/extract_top_features.py
    
    # Custom experiment directory
    python features/adapted/extract_top_features.py --data-dir results/stage_2/adapted_features/my_experiment
    
    # Custom percentages
    python features/adapted/extract_top_features.py --cosine-percent 10 --variance-percent 3
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from datetime import datetime

def load_plotting_data(data_dir: Path):
    """Load the saved plotting data from the specified directory."""
    print(f"Loading plotting data from {data_dir}...")
    
    # Load the saved plotting data
    global_cosines = np.load(data_dir / "plotting_data_global_cosines.npy")
    global_gaps = np.load(data_dir / "plotting_data_global_gaps.npy")
    adapted_indices_list = np.load(data_dir / "plotting_data_adapted_indices.npy")
    
    # Convert back to set
    adapted_indices = set(adapted_indices_list)
    
    # Load metadata for verification
    with open(data_dir / "plotting_data_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(global_cosines):,} total features")
    print(f"Loaded {len(adapted_indices):,} adapted features")
    print(f"Data saved on: {metadata['saved_timestamp']}")
    
    return global_cosines, global_gaps, adapted_indices, metadata

def extract_top_cosine_features(global_cosines, global_gaps, adapted_indices, top_percent=5, reverse=False):
    """Extract the top N% highest cosine similarity features (or bottom N% if reverse=True)."""
    if reverse:
        print(f"Extracting bottom {top_percent}% lowest cosine similarity features...")
    else:
        print(f"Extracting top {top_percent}% highest cosine similarity features...")
    
    # Calculate the number of features to extract
    num_features = len(global_cosines)
    num_top_features = int(num_features * top_percent / 100)
    
    # Get the indices of the cosine similarity features
    if reverse:
        # Bottom cosine similarity (lowest values)
        top_cosine_indices = np.argsort(global_cosines)[:num_top_features]
    else:
        # Top cosine similarity (highest values)
        top_cosine_indices = np.argsort(global_cosines)[-num_top_features:]
    
    # Create DataFrame with feature information
    top_cosine_data = []
    for idx in top_cosine_indices:
        is_adapted = idx in adapted_indices
        top_cosine_data.append({
            'global_index': idx,
            'cosine_similarity': global_cosines[idx],
            'variance_gap': global_gaps[idx],
            'is_adapted': is_adapted,
            'layer': idx // (num_features // 32),  # Assuming 32 layers
            'feature_in_layer': idx % (num_features // 32)
        })
    
    # Sort by cosine similarity (descending)
    top_cosine_df = pd.DataFrame(top_cosine_data)
    top_cosine_df = top_cosine_df.sort_values('cosine_similarity', ascending=False)
    
    print(f"Extracted {len(top_cosine_df)} top cosine similarity features")
    print(f"Cosine similarity range: {top_cosine_df['cosine_similarity'].min():.4f} - {top_cosine_df['cosine_similarity'].max():.4f}")
    print(f"Adapted features in top cosine: {top_cosine_df['is_adapted'].sum()}")
    
    return top_cosine_df

def extract_top_variance_gap_features(global_cosines, global_gaps, adapted_indices, top_percent=5, reverse=False):
    """Extract the top N% lowest variance gap features (or top N% highest if reverse=True)."""
    if reverse:
        print(f"Extracting top {top_percent}% highest variance gap features...")
    else:
        print(f"Extracting top {top_percent}% lowest variance gap features...")
    
    # Calculate the number of features to extract
    num_features = len(global_cosines)
    num_top_features = int(num_features * top_percent / 100)
    
    # Get the indices of the variance gap features
    if reverse:
        # Top variance gap (highest values)
        top_gap_indices = np.argsort(global_gaps)[-num_top_features:]
    else:
        # Bottom variance gap (lowest values)
        top_gap_indices = np.argsort(global_gaps)[:num_top_features]
    
    # Create DataFrame with feature information
    top_gap_data = []
    for idx in top_gap_indices:
        is_adapted = idx in adapted_indices
        top_gap_data.append({
            'global_index': idx,
            'cosine_similarity': global_cosines[idx],
            'variance_gap': global_gaps[idx],
            'is_adapted': is_adapted,
            'layer': idx // (num_features // 32),  # Assuming 32 layers
            'feature_in_layer': idx % (num_features // 32)
        })
    
    # Sort by variance gap (ascending)
    top_gap_df = pd.DataFrame(top_gap_data)
    top_gap_df = top_gap_df.sort_values('variance_gap', ascending=True)
    
    print(f"Extracted {len(top_gap_df)} top variance gap features")
    print(f"Variance gap range: {top_gap_df['variance_gap'].min():.6f} - {top_gap_df['variance_gap'].max():.6f}")
    print(f"Adapted features in top variance gap: {top_gap_df['is_adapted'].sum()}")
    
    return top_gap_df

def save_intersection_features_to_csv(df, output_path, cosine_percent, variance_percent, metadata):
    """Save intersection features DataFrame to CSV with layer and feature columns."""
    print(f"Saving intersection features to {output_path}...")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a simplified DataFrame with just layer and feature columns
    simple_df = df[['layer', 'feature_in_layer']].copy()
    simple_df.columns = ['layer', 'feature']
    
    # Add metadata as comments at the top of the file
    with open(output_path, 'w') as f:
        f.write(f"# Intersection Features (Both Top {cosine_percent}% Cosine Similarity AND Bottom {variance_percent}% Variance Gap)\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Original data saved on: {metadata['saved_timestamp']}\n")
        f.write(f"# Features that are BOTH in top {cosine_percent}% cosine similarity AND bottom {variance_percent}% variance gap\n")
        f.write(f"# Total intersection features: {len(df):,}\n")
        f.write(f"# Columns: layer, feature\n")
        f.write("#\n")
    
    # Append the simplified DataFrame to the file
    simple_df.to_csv(output_path, mode='a', index=False)
    
    print(f"Saved {len(simple_df)} intersection features to {output_path}")

def create_summary_report(top_cosine_df, top_gap_df, output_dir, metadata):
    """Create a summary report of the extracted features."""
    print("Creating summary report...")
    
    summary_path = output_dir / "feature_extraction_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("Feature Extraction Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original data saved on: {metadata['saved_timestamp']}\n\n")
        
        f.write("Top 5% Highest Cosine Similarity Features:\n")
        f.write("-" * 45 + "\n")
        f.write(f"Total features: {len(top_cosine_df):,}\n")
        f.write(f"Cosine similarity range: {top_cosine_df['cosine_similarity'].min():.4f} - {top_cosine_df['cosine_similarity'].max():.4f}\n")
        f.write(f"Adapted features: {top_cosine_df['is_adapted'].sum():,} ({top_cosine_df['is_adapted'].mean()*100:.1f}%)\n")
        f.write(f"Layer distribution:\n")
        layer_counts = top_cosine_df['layer'].value_counts().sort_index()
        for layer, count in layer_counts.items():
            f.write(f"  Layer {layer:2d}: {count:4d} features\n")
        f.write("\n")
        
        f.write("Top 5% Lowest Variance Gap Features:\n")
        f.write("-" * 42 + "\n")
        f.write(f"Total features: {len(top_gap_df):,}\n")
        f.write(f"Variance gap range: {top_gap_df['variance_gap'].min():.6f} - {top_gap_df['variance_gap'].max():.6f}\n")
        f.write(f"Adapted features: {top_gap_df['is_adapted'].sum():,} ({top_gap_df['is_adapted'].mean()*100:.1f}%)\n")
        f.write(f"Layer distribution:\n")
        layer_counts = top_gap_df['layer'].value_counts().sort_index()
        for layer, count in layer_counts.items():
            f.write(f"  Layer {layer:2d}: {count:4d} features\n")
        f.write("\n")
        
        # Find overlap between the two sets
        cosine_indices = set(top_cosine_df['global_index'])
        gap_indices = set(top_gap_df['global_index'])
        overlap = cosine_indices.intersection(gap_indices)
        
        f.write("Overlap Analysis:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Features in both top cosine and top variance gap: {len(overlap):,}\n")
        f.write(f"Overlap percentage: {len(overlap)/len(top_cosine_df)*100:.1f}%\n")
    
    print(f"Summary report saved to {summary_path}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Extract top 5% highest cosine similarity and lowest variance gap features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python features/adapted/extract_top_features.py
  
  # Custom adapted features directory
  python features/adapted/extract_top_features.py --data-dir results/stage_2/adapted_features/my_experiment
  
  # Custom output directory
  python features/adapted/extract_top_features.py --output-dir results/feature_analysis
  
  # Custom percentages
  python features/adapted/extract_top_features.py --cosine-percent 10 --variance-percent 5
        """
    )
    
    parser.add_argument('--data-dir', type=str, 
                       default="results/stage_2/adapted_features/text-only_50k_20",
                       help='Directory containing adapted features plotting data (default: results/stage_2/adapted_features/text-only_50k_20)')
    
    parser.add_argument('--output-dir', type=str,
                       default="results/feature_analysis",
                       help='Directory to save extracted features (default: results/feature_analysis)')
    
    parser.add_argument('--cosine-percent', type=float, default=5.0,
                       help='Percentage of top cosine similarity features to extract (default: 5.0)')
    
    parser.add_argument('--variance-percent', type=float, default=5.0,
                       help='Percentage of bottom variance gap features to extract (default: 5.0)')
    
    parser.add_argument('--reverse', action='store_true',
                       help='Reverse selection: bottom cosine similarity and top variance gap (default: top cosine, bottom variance)')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cosine_percent = args.cosine_percent
    variance_percent = args.variance_percent
    
    print(f"Configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Cosine similarity percent: {cosine_percent}%")
    print(f"  Variance gap percent: {variance_percent}%")
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        print("Make sure you're running this from the project root directory")
        print("Use --data-dir to specify a different directory")
        return
    
    # Load the plotting data
    global_cosines, global_gaps, adapted_indices, metadata = load_plotting_data(data_dir)
    
    # Extract cosine similarity features
    top_cosine_df = extract_top_cosine_features(global_cosines, global_gaps, adapted_indices, cosine_percent, args.reverse)
    
    # Extract variance gap features
    top_gap_df = extract_top_variance_gap_features(global_cosines, global_gaps, adapted_indices, variance_percent, args.reverse)
    
    # Find features that are in BOTH sets (intersection)
    cosine_indices = set(top_cosine_df['global_index'])
    gap_indices = set(top_gap_df['global_index'])
    intersection_indices = cosine_indices.intersection(gap_indices)
    
    print(f"Found {len(intersection_indices)} features that are both top {cosine_percent}% cosine similarity AND bottom {variance_percent}% variance gap")
    
    # Create DataFrame for intersection features
    intersection_data = []
    for idx in intersection_indices:
        is_adapted = idx in adapted_indices
        intersection_data.append({
            'global_index': idx,
            'cosine_similarity': global_cosines[idx],
            'variance_gap': global_gaps[idx],
            'is_adapted': is_adapted,
            'layer': idx // (len(global_cosines) // 32),  # Assuming 32 layers
            'feature_in_layer': idx % (len(global_cosines) // 32)
        })
    
    intersection_df = pd.DataFrame(intersection_data)
    intersection_df = intersection_df.sort_values('cosine_similarity', ascending=False)
    
    # Save intersection features to a single CSV file
    output_path = output_dir / f"intersection_features_cosine_{cosine_percent}pct_variance_{variance_percent}pct.csv"
    save_intersection_features_to_csv(intersection_df, output_path, cosine_percent, variance_percent, metadata)
    
    # Create summary report
    create_summary_report(top_cosine_df, top_gap_df, output_dir, metadata)
    
    print(f"\nExtraction complete!")
    print(f"Files saved to {output_dir}:")
    print(f"  - {output_path.name} (Intersection features: both top {cosine_percent}% cosine AND bottom {variance_percent}% variance gap)")
    print(f"  - feature_extraction_summary.txt (Summary statistics)")
    print(f"\nCSV file contains layer,feature columns and can be used with other scripts.")

if __name__ == "__main__":
    main()
