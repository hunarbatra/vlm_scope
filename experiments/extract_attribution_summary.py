#!/usr/bin/env python3
"""
Script to extract top heads and layer scores from attribution results,
removing detailed head scores to create a more concise summary.
"""

import json
import sys
from pathlib import Path

def extract_attribution_summary(input_json_path, output_json_path):
    """
    Extract top heads and layer scores from attribution results.
    
    Args:
        input_json_path: Path to the generated_attribution_results.json file
        output_json_path: Path for the output JSON file
    """
    print(f"Loading attribution data from {input_json_path}...")
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    features = data.get('features', {})
    
    # Create summary data structure
    summary_data = {
        "features": {}
    }
    
    for feature_name, feature_data in features.items():
        layer = feature_data.get('layer')
        feature_id = feature_data.get('feature')
        attribution = feature_data.get('attribution', {})
        
        # Extract only the data we want
        summary_attribution = {
            "layer_scores_method_A": attribution.get('layer_scores_method_A', []),
            "layer_scores_method_B": attribution.get('layer_scores_method_B', []),
            "top_heads_method_A": attribution.get('top_heads_method_A', []),
            "top_heads_method_B": attribution.get('top_heads_method_B', [])
        }
        
        summary_data["features"][feature_name] = {
            "layer": layer,
            "feature": feature_id,
            "attribution": summary_attribution
        }
    
    # Write summary to output file
    print(f"Writing summary to {output_json_path}...")
    
    with open(output_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print summary statistics
    total_features = len(summary_data["features"])
    print(f"\nSummary:")
    print(f"Total features processed: {total_features}")
    print(f"Output file size reduced by removing detailed head scores")
    print(f"Summary saved to: {output_json_path}")

def main():
    # Define paths
    input_json = "/homes/55/lachin/llama-scope-finetune-3/results/stage6/generated_attribution_results.json"
    output_json = "/homes/55/lachin/llama-scope-finetune-3/results/experiments/attribution_summary.json"
    
    # Check if input file exists
    if not Path(input_json).exists():
        print(f"Error: Input file {input_json} not found!")
        sys.exit(1)
    
    try:
        extract_attribution_summary(input_json, output_json)
        print(f"\nExtraction complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()








