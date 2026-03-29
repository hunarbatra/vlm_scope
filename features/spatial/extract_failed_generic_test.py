#!/usr/bin/env python3
"""
Extract features that haven't passed the generic test from the JSON summary file.

This script loads the common_features_summary JSON file and creates a CSV file
containing only the features that haven't passed the generic test, formatted
similarly to common_features_generic_filtered.csv.

Usage:
    python features/spatial/extract_failed_generic_test.py \
        --json_file results/stage_4/common_features_summary_35_2_with_passed_with_relations.json \
        --output_csv results/stage_3/adapted_spatial_features_text-only_15_35/common_features_failed_generic_test.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path

def load_common_features_csv(csv_path: str) -> pd.DataFrame:
    """Load the original common features CSV to get the base data."""
    print(f"Loading common features from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} common features")
    return df

def load_json_summary(json_path: str) -> dict:
    """Load the JSON summary file, handling potential JSON parsing issues."""
    print(f"Loading JSON summary from {json_path}...")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded summary for {len(data['features'])} features")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("Attempting to fix common JSON issues...")
        
        # Read the file content and try to fix common issues
        with open(json_path, 'r') as f:
            content = f.read()
        
        # Fix trailing commas in arrays and objects
        import re
        # Remove trailing commas before closing brackets/braces
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        try:
            data = json.loads(content)
            print(f"Successfully loaded summary for {len(data['features'])} features after fixing JSON")
            return data
        except json.JSONDecodeError as e2:
            print(f"Still unable to parse JSON after fixes: {e2}")
            print("Please check the JSON file for syntax errors")
            raise

def extract_failed_generic_test_features(json_data: dict, common_features_df: pd.DataFrame) -> pd.DataFrame:
    """Extract features that haven't passed the generic test."""
    print("Extracting features that haven't passed the generic test...")
    
    failed_features = []
    
    for feature_key, feature_data in json_data['features'].items():
        layer = feature_data['layer']
        feature = feature_data['feature']
        
        # Check if this feature passed the generic test
        # Look for the 'passed' field in the feature data
        passed = feature_data.get('passed', True)  # Default to True if not specified
        
        if not passed:
            # Find the corresponding row in the common features CSV
            matching_row = common_features_df[
                (common_features_df['layer'] == layer) & 
                (common_features_df['feature'] == feature)
            ]
            
            if not matching_row.empty:
                failed_features.append(matching_row.iloc[0].to_dict())
            else:
                print(f"Warning: Could not find feature {layer}_{feature} in common features CSV")
    
    print(f"Found {len(failed_features)} features that haven't passed the generic test")
    
    if failed_features:
        return pd.DataFrame(failed_features)
    else:
        # Return empty DataFrame with the same columns as the original
        return pd.DataFrame(columns=common_features_df.columns)

def main():
    parser = argparse.ArgumentParser(description="Extract features that haven't passed the generic test")
    parser.add_argument("--json_file", required=True, help="Path to common_features_summary JSON file")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path")
    parser.add_argument("--common_features_csv", 
                       default="results/stage_3/adapted_spatial_features_text-only_15_35/common_features_detailed.csv",
                       help="Path to original common features CSV file")
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    output_path = Path(args.output_csv)
    common_features_path = Path(args.common_features_csv)
    
    # Check if files exist
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return
    
    if not common_features_path.exists():
        print(f"Error: Common features CSV not found at {common_features_path}")
        return
    
    # Load data
    json_data = load_json_summary(json_path)
    common_features_df = load_common_features_csv(common_features_path)
    
    # Extract failed features
    failed_features_df = extract_failed_generic_test_features(json_data, common_features_df)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    failed_features_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(failed_features_df)} failed features to {output_path}")
    
    if len(failed_features_df) > 0:
        print(f"\nFirst few failed features:")
        print(failed_features_df[['layer', 'feature', 'odds_ratio', 'freq_diff', 'p_adj']].head())
        
        print(f"\nSummary statistics:")
        print(f"  Total failed features: {len(failed_features_df)}")
        print(f"  Layers with failed features: {failed_features_df['layer'].nunique()}")
        print(f"  Mean odds ratio: {failed_features_df['odds_ratio'].mean():.2f}")
        print(f"  Mean frequency difference: {failed_features_df['freq_diff'].mean():.4f}")
    else:
        print("No features found that haven't passed the generic test.")

if __name__ == "__main__":
    main()
