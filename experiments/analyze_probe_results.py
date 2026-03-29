#!/usr/bin/env python3
"""
Script to analyze probe results and create a CSV file showing layer features
for both passed and failed features.
"""

import json
import csv
import sys
from pathlib import Path

def analyze_probe_results(json_file_path, output_csv_path):
    """
    Analyze probe results and create CSV with layer features for passed/failed features.
    
    Args:
        json_file_path: Path to the generated_probe_results_hard.json file
        output_csv_path: Path for the output CSV file
    """
    print(f"Loading data from {json_file_path}...")
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    features = data.get('features', {})
    
    # Prepare data for CSV
    csv_data = []
    
    for feature_name, feature_data in features.items():
        layer = feature_data.get('layer')
        feature_id = feature_data.get('feature')
        method = feature_data.get('method', 'unknown')
        probe_data = feature_data.get('probe', {})
        passed = probe_data.get('passed_generic_test', False)
        records = probe_data.get('records', [])
        
        # Count records for additional context
        num_records = len(records)
        
        csv_data.append({
            'feature_name': feature_name,
            'layer': layer,
            'feature_id': feature_id,
            'method': method,
            'passed_generic_test': passed,
            'num_records': num_records
        })
    
    # Sort by layer, then by feature_id for better organization
    csv_data.sort(key=lambda x: (x['layer'], x['feature_id']))
    
    # Write to CSV
    print(f"Writing results to {output_csv_path}...")
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['feature_name', 'layer', 'feature_id', 'method', 'passed_generic_test', 'num_records']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    # Print summary statistics
    total_features = len(csv_data)
    passed_features = sum(1 for row in csv_data if row['passed_generic_test'])
    failed_features = total_features - passed_features
    
    print(f"\nSummary:")
    print(f"Total features: {total_features}")
    print(f"Passed features: {passed_features}")
    print(f"Failed features: {failed_features}")
    print(f"Pass rate: {passed_features/total_features*100:.1f}%")
    
    # Show layer distribution
    layer_stats = {}
    for row in csv_data:
        layer = row['layer']
        if layer not in layer_stats:
            layer_stats[layer] = {'total': 0, 'passed': 0, 'failed': 0}
        layer_stats[layer]['total'] += 1
        if row['passed_generic_test']:
            layer_stats[layer]['passed'] += 1
        else:
            layer_stats[layer]['failed'] += 1
    
    print(f"\nLayer distribution:")
    for layer in sorted(layer_stats.keys()):
        stats = layer_stats[layer]
        pass_rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"Layer {layer}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")

def main():
    # Define paths
    json_file = "/homes/55/lachin/llama-scope-finetune-3/results/stage6/generated_probe_results_hard.json"
    output_csv = "/homes/55/lachin/llama-scope-finetune-3/experiments/probe_results_analysis.csv"
    
    # Check if input file exists
    if not Path(json_file).exists():
        print(f"Error: Input file {json_file} not found!")
        sys.exit(1)
    
    try:
        analyze_probe_results(json_file, output_csv)
        print(f"\nAnalysis complete! Results saved to {output_csv}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()








