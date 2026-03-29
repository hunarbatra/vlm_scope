"""
Find common features between spatial features (from find_spatial_features.py) 
and adapted features (from adapted_features_results.csv).

Usage:
CUDA_VISIBLE_DEVICES=7 python features/spatial/find_common_adapted_spatial.py \
  --spatial-file results/spatial_analysis/suspect_spatial_features_vqa_vqa.csv \
  --adapted-file results/adapted_features/text-only_50k_10/adapted_features_results.csv \
  --output-dir results/common_features_vqa_vqa_text-only

python features/spatial/find_adapted_spatial.py \
  --spatial-file results/spatial_analysis/suspect_spatial_features_vqa_vqa_text-only.csv \
  --adapted-file results/stage_2/adapted_features/text-only_50k_15_35_sensitivity/adapted_features_results.csv \
  --output-dir results/stage_3/adapted_spatial_features_text-only_15_35

"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import ast
from collections import defaultdict

class CommonAdaptedFeatureFinder:
    def __init__(self, spatial_file, adapted_file, output_dir="common_features_robust"):
        self.spatial_file = Path(spatial_file)
        self.adapted_file = Path(adapted_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.spatial_features = {}
        self.adapted_features = {}
        self.common_features = {}
        
    def load_spatial_features(self):
        """Load spatial features from CSV file."""
        print(f"[INFO] Loading spatial features from {self.spatial_file}")
        
        df = pd.read_csv(self.spatial_file)
        print(f"[INFO] Loaded {len(df)} spatial features")
        
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer]
            self.spatial_features[layer] = set(layer_data['feature'].tolist())
            print(f"  Layer {layer}: {len(self.spatial_features[layer])} features")
        
        return df
    
    def load_adapted_features(self):
        """Load adapted features from CSV file."""
        print(f"[INFO] Loading adapted features from {self.adapted_file}")
        
        df = pd.read_csv(self.adapted_file)
        print(f"[INFO] Loaded {len(df)} adapted feature entries")
        
        for _, row in df.iterrows():
            layer = row['layer']
            adapted_indices_str = row['adapted_indices']
            
            try:
                adapted_indices = ast.literal_eval(adapted_indices_str)
                self.adapted_features[layer] = set(adapted_indices)
                print(f"  Layer {layer}: {len(self.adapted_features[layer])} adapted features")
            except (ValueError, SyntaxError) as e:
                print(f"[WARN] Failed to parse adapted indices for layer {layer}: {e}")
                self.adapted_features[layer] = set()
        
        return df
    
    def find_common_features(self):
        """Find intersection of spatial and adapted features."""
        print(f"\n[INFO] Finding common features...")
        
        common_count = 0
        spatial_count = 0
        adapted_count = 0
        
        for layer in sorted(set(self.spatial_features.keys()) | set(self.adapted_features.keys())):
            spatial_set = self.spatial_features.get(layer, set())
            adapted_set = self.adapted_features.get(layer, set())
            
            common_set = spatial_set & adapted_set
            self.common_features[layer] = common_set
            
            spatial_count += len(spatial_set)
            adapted_count += len(adapted_set)
            common_count += len(common_set)
            
            print(f"  Layer {layer}:")
            print(f"    Spatial features: {len(spatial_set)}")
            print(f"    Adapted features: {len(adapted_set)}")
            print(f"    Common features: {len(common_set)}")
            if common_set:
                print(f"    Common feature examples: {sorted(list(common_set))[:5]}")
        
        print(f"\n[INFO] Summary:")
        print(f"  Total spatial features: {spatial_count}")
        print(f"  Total adapted features: {adapted_count}")
        print(f"  Total common features: {common_count}")
        print(f"  Common percentage: {100 * common_count / max(spatial_count, 1):.2f}% of spatial features")
        
        return common_count
    

    

    
    def save_results(self):
        """Save results to files."""
        print(f"\n[INFO] Saving results...")
        
        spatial_df = pd.read_csv(self.spatial_file)
        adapted_df = pd.read_csv(self.adapted_file)
        
        detailed_common_data = []
        for layer in sorted(self.common_features.keys()):
            common_features = self.common_features[layer]
            if not common_features:
                continue
            
            layer_spatial = spatial_df[spatial_df['layer'] == layer]
            layer_spatial_dict = dict(zip(layer_spatial['feature'], layer_spatial.to_dict('records')))
            
            layer_adapted = adapted_df[adapted_df['layer'] == layer].iloc[0]
            adapted_indices = ast.literal_eval(layer_adapted['adapted_indices'])
            adapted_set = set(adapted_indices)
            
            for feature in sorted(common_features):
                spatial_stats = {}
                if feature in layer_spatial_dict:
                    spatial_data = layer_spatial_dict[feature]
                    spatial_stats = {
                        'odds_ratio': spatial_data['odds_ratio'],
                        'freq_diff': spatial_data['freq_diff'],
                        'p_adj': spatial_data['p_adj'],
                        'freq_vsr': spatial_data['freq_vsr'],
                        'freq_vqa': spatial_data['freq_vqa'],
                        'c_vsr': spatial_data['c_vsr'],
                        'c_vqa': spatial_data['c_vqa'],
                        'n_vsr': spatial_data['n_vsr'],
                        'n_vqa': spatial_data['n_vqa']
                    }
                
                adapted_stats = {
                    'is_adapted': feature in adapted_set,
                    'layer_mean_cosine': layer_adapted['mean_cosine'],
                    'layer_mean_variance_gap': layer_adapted['mean_variance_gap'],
                    'layer_adapted_mean_cosine': layer_adapted.get('adapted_mean_cosine', None),
                    'layer_adapted_mean_gap': layer_adapted.get('adapted_mean_gap', None)
                }
                
                if feature in adapted_set:
                    adapted_data = layer_adapted['adapted_data']
                    if isinstance(adapted_data, str):
                        adapted_data = ast.literal_eval(adapted_data)
                    
                    if feature in adapted_data:
                        feature_data = adapted_data[feature]
                        adapted_stats.update({
                            'feature_variance_gap': feature_data.get('variance_gap', None),
                            'feature_cosine_similarity': feature_data.get('cosine_similarity', None)
                        })
                    else:
                        adapted_stats.update({
                            'feature_variance_gap': None,
                            'feature_cosine_similarity': None
                        })
                else:
                    adapted_stats.update({
                        'feature_variance_gap': None,
                        'feature_cosine_similarity': None
                    })
                
                detailed_common_data.append({
                    'layer': layer,
                    'feature': feature,
                    **spatial_stats,
                    **adapted_stats
                })
        
        if detailed_common_data:
            detailed_df = pd.DataFrame(detailed_common_data)
            detailed_df.to_csv(self.output_dir / "common_features_detailed.csv", index=False)
            print(f"[INFO] Saved {len(detailed_common_data)} detailed common features to {self.output_dir / 'common_features_detailed.csv'}")
        
        simple_common_data = []
        for layer in sorted(self.common_features.keys()):
            for feature in sorted(self.common_features[layer]):
                simple_common_data.append({
                    'layer': layer,
                    'feature': feature
                })
        
        if simple_common_data:
            simple_df = pd.DataFrame(simple_common_data)
            simple_df.to_csv(self.output_dir / "common_features.csv", index=False)
            print(f"[INFO] Saved {len(simple_common_data)} common features to {self.output_dir / 'common_features.csv'}")
        
        total_common = sum(len(features) for features in self.common_features.values())
        total_spatial = sum(len(features) for features in self.spatial_features.values())
        total_adapted = sum(len(features) for features in self.adapted_features.values())
        
        summary = {
            'total_common_features': total_common,
            'total_spatial_features': total_spatial,
            'total_adapted_features': total_adapted,
            'common_percentage_of_spatial': 100 * total_common / max(total_spatial, 1),
            'layers_with_common_features': len([l for l in self.common_features if self.common_features[l]]),
            'layer_breakdown': {}
        }
        
        for layer in sorted(self.common_features.keys()):
            layer_key = int(layer) if hasattr(layer, 'item') else layer
            summary['layer_breakdown'][layer_key] = {
                'spatial_features': len(self.spatial_features.get(layer, set())),
                'adapted_features': len(self.adapted_features.get(layer, set())),
                'common_features': len(self.common_features[layer]),
                'common_feature_list': sorted(list(self.common_features[layer]))
            }
        
        with open(self.output_dir / "common_features_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"[INFO] Saved summary to {self.output_dir / 'common_features_summary.json'}")
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("=== Finding Common Features Between Spatial and Adapted Features ===")
        
        self.load_spatial_features()
        self.load_adapted_features()
        
        common_count = self.find_common_features()
        
        if common_count == 0:
            print("\n[WARN] No common features found!")
            return
        
        self.save_results()
        
        print(f"\n[INFO] Analysis complete! Results saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Find common features between spatial and adapted features")
    parser.add_argument("--spatial-file", required=True, 
                       help="Path to spatial features CSV file")
    parser.add_argument("--adapted-file", required=True,
                       help="Path to adapted features CSV file")
    parser.add_argument("--output-dir", default="results/common_features_robust",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    analyzer = CommonAdaptedFeatureFinder(
        spatial_file=args.spatial_file,
        adapted_file=args.adapted_file,
        output_dir=args.output_dir
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 
