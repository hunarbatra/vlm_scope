#!/usr/bin/env python3
"""
Generate clean, paper-ready FVU curves from wandb data.

This script processes wandb export data to create publication-quality plots
comparing different training methods (full, random, image-only, text-only) across layers.

Usage:
python finetune/vqa/generate_fvu_curves.py --wandb_dir results/wandb --output_dir results/stage_1/fvu_plots --split both --save_data



"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 20,
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
    'legend.fontsize': 20,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class FVUCurveGenerator:
    """Generate FVU curves from wandb data."""
    
    def __init__(self, wandb_dir: str):
        """Initialize with wandb directory path."""
        self.wandb_dir = Path(wandb_dir)
        self.data = {}
        self.methods = ['full', 'random', 'image', 'text']  # 4 training methods
        self.splits = ['train', 'val']  # Both training and validation data
        
    def load_data(self) -> None:
        """Load and process wandb export data."""
        print("Loading wandb data...")
        
        # Load data for each method and split combination
        for method in self.methods:
            for split in self.splits:
                file_name = f"{method}_{split}.csv"
                file_path = self.wandb_dir / file_name
                
                if not file_path.exists():
                    print(f"Warning: {file_name} not found, skipping...")
                    continue
                
                print(f"Loading {file_name}...")
                df = self._load_csv(file_path)
                
                # Store with method_split as key
                key = f"{method}_{split}"
                self.data[key] = df
        
        print(f"Loaded data for: {list(self.data.keys())}")
        print(f"Data shapes: {[(key, df.shape) for key, df in self.data.items()]}")
    
    def _get_method_from_key(self, key: str) -> str:
        """Extract method from data key (e.g., 'full_train' -> 'full')."""
        return key.split('_')[0]
    
    def _get_split_from_key(self, key: str) -> str:
        """Extract split from data key (e.g., 'full_train' -> 'train')."""
        return key.split('_')[1]
    
    def _load_csv(self, csv_file: Path) -> pd.DataFrame:
        """Load and clean CSV data."""
        df = pd.read_csv(csv_file)
        
        # Clean data
        clean_df = self._clean_data(df)
        
        return clean_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the dataframe."""
        # Find the tokens column (might have different names)
        token_col = None
        for col in df.columns:
            if 'total_tokens' in col or 'tokens' in col.lower():
                token_col = col
                break
        
        if token_col is None:
            raise ValueError("Could not find tokens column in data")
        
        # Create a clean dataframe with tokens and FVU data
        clean_df = pd.DataFrame()
        clean_df['tokens'] = df[token_col]
        
        # Extract FVU columns for each layer
        for layer_idx in range(32):
            fvu_col = None
            for col in df.columns:
                if f'layer_{layer_idx}' in col and 'fvu' in col.lower():
                    # Skip MIN/MAX columns, use main column
                    if '__MIN' not in col and '__MAX' not in col:
                        fvu_col = col
                        break
            
            if fvu_col:
                clean_df[f'layer_{layer_idx}'] = df[fvu_col]
        
        # Remove rows with NaN values
        clean_df = clean_df.dropna()
        
        # Sort by tokens
        clean_df = clean_df.sort_values('tokens')
        
        return clean_df
    
    def generate_layer_comparison_plot(self, split: str = 'train', save_path: str = None) -> None:
        """Generate layer-by-layer FVU comparison plot for specified split."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"Generating layer comparison plots for {split} data...")
        
        fig, axes = plt.subplots(4, 8, figsize=(24, 12))
        # fig.suptitle(f'FVU Comparison: 4 Training Methods Across Layers ({split.capitalize()} Data)', 
        #              fontsize=16, fontweight='bold', y=0.98)
        
        # Define colors for each method
        colors = {'full': '#7F58AF', 'random': '#64C5EB', 'image': '#E84D8A', 'text': '#FEB326'}
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for layer_idx in range(32):
            ax = axes_flat[layer_idx]
            
            # Plot data for each method
            for method in self.methods:
                key = f"{method}_{split}"
                if key in self.data:
                    df = self.data[key]
                    
                    if f'layer_{layer_idx}' in df.columns:
                        # Convert tokens to millions for readability
                        tokens_millions = df['tokens'] / 1e6
                        fvu_values = df[f'layer_{layer_idx}']
                        
                        ax.plot(tokens_millions, fvu_values, 
                               label=method.capitalize(), 
                               linewidth=2.5, 
                               marker='o', 
                               markersize=4,
                               alpha=0.85,
                               color=colors.get(method, None))
            
            ax.set_title(f'Layer {layer_idx}', fontsize=18, fontweight='bold')
            ax.set_xlabel('Training Steps', fontsize=20, fontweight='bold')
            ax.set_ylabel('FVU', fontsize=20, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits for better comparison
            all_fvu = []
            for method in self.methods:
                key = f"{method}_{split}"
                if key in self.data and f'layer_{layer_idx}' in self.data[key].columns:
                    all_fvu.extend(self.data[key][f'layer_{layer_idx}'].dropna().values)
            
            if all_fvu:
                y_min, y_max = min(all_fvu), max(all_fvu)
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        # Add a common legend
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer comparison plot saved to {save_path}")
        else:
            filename = f'fvu_layer_comparison_{split}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Layer comparison plot saved to {filename}")
        
        plt.show()
    
    def generate_aggregate_comparison_plot(self, split: str = 'train', save_path: str = None) -> None:
        """Generate aggregate FVU comparison plots (two separate plots)."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"Generating aggregate comparison plots for {split} data...")
        
        # Define colors for each method
        colors = {'full': '#7F58AF', 'random': '#64C5EB', 'image': '#E84D8A', 'text': '#FEB326'}
        
        # Plot 1: Mean FVU across all layers
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        # fig1.suptitle(f'Mean FVU Across All Layers ({split.capitalize()} Data)', 
        #               fontsize=16, fontweight='bold')
        
        for method in self.methods:
            key = f"{method}_{split}"
            if key in self.data:
                df = self.data[key]
                
                # Calculate mean FVU across all layers
                layer_cols = [col for col in df.columns if col.startswith('layer_')]
                if layer_cols:
                    mean_fvu = df[layer_cols].mean(axis=1)
                    tokens_millions = df['tokens'] / 1e6
                    
                    ax1.plot(tokens_millions, mean_fvu, 
                            label=method.capitalize(), 
                            linewidth=3, 
                            marker='o', 
                            markersize=5,
                            alpha=0.85,
                            color=colors.get(method, None))
        
        # ax1.set_title('Training Progress: Mean FVU Across All Layers', fontsize=20, fontweight='bold')
        ax1.set_xlabel('Training Steps', fontsize=22, fontweight='bold')
        ax1.set_ylabel('Mean FVU', fontsize=22, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=20)
        ax1.set_ylim(bottom=0)  # Start y-axis from 0 for better visualization
        
        plt.tight_layout()
        
        # Save first plot
        if save_path:
            # Extract base path and create separate filenames
            base_path = Path(save_path).parent / Path(save_path).stem
            plot1_path = f"{base_path}_mean_fvu.png"
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
            print(f"Mean FVU plot saved to {plot1_path}")
        else:
            filename = f'fvu_mean_fvu_{split}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Mean FVU plot saved to {filename}")
        
        plt.show()
        
        # Plot 2: Final FVU improvement over random baseline
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        fig2.suptitle(f'FVU Improvement vs Random Baseline ({split.capitalize()} Data)', 
                      fontsize=22, fontweight='bold')
        
        final_improvements = {}
        random_key = f"random_{split}"
        
        if random_key in self.data:
            random_df = self.data[random_key]
            random_layer_cols = [col for col in random_df.columns if col.startswith('layer_')]
            random_final_fvu = random_df[random_layer_cols].iloc[-1].values
            
            for method in self.methods:
                if method == 'random':
                    continue
                    
                key = f"{method}_{split}"
                if key in self.data:
                    df = self.data[key]
                    layer_cols = [col for col in df.columns if col.startswith('layer_')]
                    if layer_cols:
                        final_fvu = df[layer_cols].iloc[-1].values
                        improvement = ((random_final_fvu - final_fvu) / random_final_fvu) * 100
                        final_improvements[method] = improvement
        
        if final_improvements:
            methods = list(final_improvements.keys())
            improvements = [np.mean(final_improvements[method]) for method in methods]
            
            bars = ax2.bar(methods, improvements, 
                          color=[colors.get(method, '#666666') for method in methods],
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            ax2.set_title('Final Performance: Improvement Over Random Baseline', fontsize=20, fontweight='bold')
            ax2.set_ylabel('Mean Improvement (%)', fontsize=20, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Add horizontal line at 0% for reference
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Save second plot
        if save_path:
            base_path = Path(save_path).parent / Path(save_path).stem
            plot2_path = f"{base_path}_improvement.png"
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            print(f"Improvement plot saved to {plot2_path}")
        else:
            filename = f'fvu_improvement_{split}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Improvement plot saved to {filename}")
        
        plt.show()
    
    def generate_heatmap_plot(self, split: str = 'train', save_path: str = None) -> None:
        """Generate heatmap showing FVU across layers and training progression."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"Generating heatmap plot for {split} data...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'FVU Heatmaps: Layer vs Training Progress ({split.capitalize()} Data)', 
                     fontsize=22, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for idx, method in enumerate(self.methods):
            key = f"{method}_{split}"
            if key in self.data and idx < len(axes_flat):
                df = self.data[key]
                ax = axes_flat[idx]
                
                # Get layer columns
                layer_cols = [col for col in df.columns if col.startswith('layer_')]
                if layer_cols:
                    # Create heatmap data
                    heatmap_data = df[layer_cols].T  # Transpose so layers are on y-axis
                    
                    # Sample data points for visualization if too many
                    if heatmap_data.shape[1] > 50:
                        step = heatmap_data.shape[1] // 50
                        heatmap_data = heatmap_data.iloc[:, ::step]
                    
                    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis_r', interpolation='nearest')
                    
                    ax.set_title(f'{method.capitalize()}', fontsize=20, fontweight='bold')
                    ax.set_xlabel('Training Progress', fontsize=20, fontweight='bold')
                    ax.set_ylabel('Layer Index', fontsize=20, fontweight='bold')
                    
                    # Set y-axis labels to show every 4th layer
                    layer_indices = [int(col.split('_')[1]) for col in layer_cols]
                    y_ticks = range(0, len(layer_indices), 4)
                    y_labels = [str(layer_indices[i]) for i in y_ticks]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('FVU', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap plot saved to {save_path}")
        else:
            filename = f'fvu_heatmap_{split}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Heatmap plot saved to {filename}")
        
        plt.show()
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for all methods."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Generating summary statistics...")
        
        summary_data = []
        
        for key, df in self.data.items():
            method = self._get_method_from_key(key)
            split = self._get_split_from_key(key)
            
            # Get layer columns
            layer_cols = [col for col in df.columns if col.startswith('layer_')]
            
            if layer_cols:
                # Calculate statistics
                initial_fvu = df[layer_cols].iloc[0].mean()
                final_fvu = df[layer_cols].iloc[-1].mean()
                improvement = ((initial_fvu - final_fvu) / initial_fvu) * 100
                
                summary_data.append({
                    'Method': method,
                    'Split': split,
                    'Initial_FVU': initial_fvu,
                    'Final_FVU': final_fvu,
                    'Improvement_%': improvement,
                    'Total_Tokens': df['tokens'].iloc[-1],
                    'Num_Layers': len(layer_cols),
                    'Data_Points': len(df)
                })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def generate_val_fvu_table(self) -> pd.DataFrame:
        """Generate a focused table showing final FVU values for all methods on validation set."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Generating validation FVU comparison table...")
        
        val_data = []
        
        for method in self.methods:
            key = f"{method}_val"
            if key in self.data:
                df = self.data[key]
                
                # Get layer columns
                layer_cols = [col for col in df.columns if col.startswith('layer_')]
                
                if layer_cols:
                    # Get final FVU values for each layer
                    final_fvu_values = df[layer_cols].iloc[-1].values
                    
                    # Calculate statistics
                    mean_fvu = np.mean(final_fvu_values)
                    std_fvu = np.std(final_fvu_values)
                    min_fvu = np.min(final_fvu_values)
                    max_fvu = np.max(final_fvu_values)
                    
                    val_data.append({
                        'Method': method.capitalize(),
                        'Mean FVU': f"{mean_fvu:.4f}",
                        'Std FVU': f"{std_fvu:.4f}",
                        'Min FVU': f"{min_fvu:.4f}",
                        'Max FVU': f"{max_fvu:.4f}",
                        'Total Tokens (M)': f"{df['tokens'].iloc[-1] / 1e6:.1f}"
                    })
        
        val_df = pd.DataFrame(val_data)
        
        # Print the table in a nice format
        print("\n" + "="*80)
        print("VALIDATION SET FVU COMPARISON TABLE")
        print("="*80)
        print(val_df.to_string(index=False))
        print("="*80)
        
        return val_df
    
    def generate_layer_wise_val_table(self) -> pd.DataFrame:
        """Generate a detailed table showing FVU values for each layer across all methods on validation set."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Generating layer-wise validation FVU table...")
        
        # Get validation data for all methods
        val_methods = {}
        for method in self.methods:
            key = f"{method}_val"
            if key in self.data:
                val_methods[method] = self.data[key]
        
        if not val_methods:
            print("No validation data found for any method!")
            return pd.DataFrame()
        
        # Find common layers across all methods
        all_layers = set()
        for method, df in val_methods.items():
            layer_cols = [col for col in df.columns if col.startswith('layer_')]
            all_layers.update([int(col.split('_')[1]) for col in layer_cols])
        
        all_layers = sorted(list(all_layers))
        
        # Create the table
        table_data = []
        for layer_idx in all_layers:
            row = {'Layer': layer_idx}
            
            for method in self.methods:
                if method in val_methods:
                    df = val_methods[method]
                    layer_col = f'layer_{layer_idx}'
                    
                    if layer_col in df.columns:
                        final_fvu = df[layer_col].iloc[-1]
                        row[f'{method.capitalize()}'] = f"{final_fvu:.4f}"
                    else:
                        row[f'{method.capitalize()}'] = "N/A"
                else:
                    row[f'{method.capitalize()}'] = "N/A"
            
            table_data.append(row)
        
        layer_df = pd.DataFrame(table_data)
        
        # Print the table
        print("\n" + "="*100)
        print("LAYER-WISE VALIDATION FVU VALUES")
        print("="*100)
        print(layer_df.to_string(index=False))
        print("="*100)
        
        return layer_df
    
    def save_processed_data(self, output_dir: str) -> None:
        """Save processed data to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Saving processed data to {output_dir}...")
        
        for key, df in self.data.items():
            filename = f"processed_{key}.csv"
            filepath = output_path / filename
            df.to_csv(filepath, index=False)
            print(f"Saved {filename}")
        
        # Save summary statistics
        summary_df = self.generate_summary_statistics()
        summary_path = output_path / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary_statistics.csv")
        
        # Save validation FVU comparison table
        val_fvu_df = self.generate_val_fvu_table()
        val_fvu_path = output_path / "validation_fvu_comparison.csv"
        val_fvu_df.to_csv(val_fvu_path, index=False)
        print(f"Saved validation_fvu_comparison.csv")
        
        # Save layer-wise validation FVU table
        layer_val_df = self.generate_layer_wise_val_table()
        layer_val_path = output_path / "layer_wise_validation_fvu.csv"
        layer_val_df.to_csv(layer_val_path, index=False)
        print(f"Saved layer_wise_validation_fvu.csv")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Generate FVU curves from wandb data')
    parser.add_argument('--wandb_dir', '-w', type=str, default='results/wandb',
                       help='Directory containing wandb CSV files (default: results/wandb)')
    parser.add_argument('--output_dir', '-o', type=str, default='fvu_plots',
                       help='Output directory for plots and processed data (default: fvu_plots)')
    parser.add_argument('--split', '-s', type=str, choices=['train', 'val', 'both'], default='both',
                       help='Which data split to plot (default: both)')
    parser.add_argument('--save_data', action='store_true',
                       help='Save processed data to CSV files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = FVUCurveGenerator(args.wandb_dir)
    
    try:
        # Load data
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        generator.load_data()
        
        # Generate plots
        splits_to_plot = ['train', 'val'] if args.split == 'both' else [args.split]
        
        for split in splits_to_plot:
            print(f"\n{'='*60}")
            print(f"GENERATING PLOTS FOR {split.upper()} DATA")
            print("="*60)
            
            # Layer comparison plot
            layer_save_path = output_path / f'fvu_layer_comparison_{split}.png'
            generator.generate_layer_comparison_plot(split=split, save_path=str(layer_save_path))
            
            # Aggregate comparison plots (now generates two separate plots)
            agg_save_path = output_path / f'fvu_aggregate_comparison_{split}.png'
            generator.generate_aggregate_comparison_plot(split=split, save_path=str(agg_save_path))
            
            # Heatmap plot
            heatmap_save_path = output_path / f'fvu_heatmap_{split}.png'
            generator.generate_heatmap_plot(split=split, save_path=str(heatmap_save_path))
        
        # Generate summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print("="*60)
        generator.generate_summary_statistics()

        # Generate validation FVU table
        print(f"\n{'='*60}")
        print("VALIDATION SET FVU COMPARISON TABLE")
        print("="*60)
        generator.generate_val_fvu_table()

        # Generate layer-wise validation FVU table
        print(f"\n{'='*60}")
        print("LAYER-WISE VALIDATION FVU VALUES")
        print("="*60)
        generator.generate_layer_wise_val_table()
        
        # Save processed data if requested
        if args.save_data:
            print(f"\n{'='*60}")
            print("SAVING PROCESSED DATA")
            print("="*60)
            generator.save_processed_data(args.output_dir)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"All plots saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())