#!/usr/bin/env python3
"""
Extract top VQA samples from vqa_all_spatial directory.

This script:
1. Reads sample_info.json from a specific feature directory in vqa_all_spatial
2. Loads the top samples from the VQA dataset
3. Creates a folder with the images and questions
4. Uses the same styling and fonts as visualize_feature.py

Usage:
python autointerp/extract_vqa_samples.py \
    --feature-dir results/stage_4/feature_samples/full/vqa_all_spatial/pretrained_layer_26_feature_807 \
    --output-dir results/stage_4/vqa_top_samples \
    --top-k 20
"""

import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np
from pathlib import Path
import sys
from datasets import load_dataset
import textwrap

# Set up matplotlib for publication-quality figures (same as visualize_feature.py)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'text.usetex': False,  # Set to True if you have LaTeX installed
})


def load_sample_info(feature_dir: Path) -> list:
    """Load sample_info.json from the feature directory."""
    sample_info_file = feature_dir / "sample_info.json"
    
    if not sample_info_file.exists():
        raise FileNotFoundError(f"sample_info.json not found: {sample_info_file}")
    
    with open(sample_info_file, 'r') as f:
        return json.load(f)


def load_vqa_samples(samples_info: list, top_k: int = 20) -> list:
    """Load the top K VQA samples from the dataset."""
    # Load VQA dataset
    print("[INFO] Loading VQA dataset...")
    vqa_dataset = load_dataset("lmms-lab/VQAv2", split="validation")
    print(f"[INFO] Loaded {len(vqa_dataset)} VQA samples")
    
    samples_data = []
    
    for i, sample_info in enumerate(samples_info[:top_k]):
        try:
            sample_idx = sample_info["sample_idx"]
            
            # Get the sample from the VQA dataset
            sample = vqa_dataset[sample_idx]
            image = sample["image"].convert("RGB")
            question = sample["question"]
            answer = sample.get("answer", "N/A")
            
            samples_data.append({
                'image': image,
                'question': question,
                'answer': answer,
                'rank': sample_info["rank"],
                'magnitude': sample_info["magnitude"],
                'sample_idx': sample_idx
            })
            print(f"Loaded VQA sample {sample_idx}: {question[:50]}...")
            
        except Exception as e:
            print(f"Warning: Could not load VQA sample {sample_info.get('sample_idx', 'unknown')}: {e}")
            # Create a placeholder image
            placeholder = Image.new('RGB', (224, 224), color='gray')
            samples_data.append({
                'image': placeholder,
                'question': f"Error loading sample: {e}",
                'answer': "N/A",
                'rank': sample_info["rank"],
                'magnitude': sample_info["magnitude"],
                'sample_idx': sample_info.get('sample_idx', -1)
            })
    
    return samples_data


def create_top_samples_visualization(samples_data: list, output_path: str = None) -> None:
    """Create a clean visualization of the top samples."""
    
    # Set up the figure with clean proportions
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.6, 5, 0.8], hspace=0.05, wspace=0.1)
    
    # Title section
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # Image grid - Clean and simple
    ax_images = fig.add_subplot(gs[1, :])
    ax_images.axis('off')
    
    # Create a 2x5 grid of images with reduced spacing between rows
    img_gs = GridSpec(2, 5, left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.2, hspace=-0.2)
    
    # Create image axes
    img_axes = []
    for i in range(10):
        row = i // 5
        col = i % 5
        ax = fig.add_subplot(img_gs[row, col])
        img_axes.append(ax)
        
        if i < len(samples_data):
            sample = samples_data[i]
            ax.imshow(sample['image'])
            ax.axis('off')
            
            # Simple border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#666')
                spine.set_linewidth(1)
            
            # Sample info
            rank = sample['rank']
            magnitude = sample['magnitude']
            question = sample['question']
            
            # Wrap long questions
            wrapped_question = textwrap.fill(question, width=25)
            
            # Title with rank, magnitude, and question
            title_text = f"Rank {rank}\nMag: {magnitude:.2f}\n{wrapped_question}"
            ax.set_title(title_text, fontsize=9, pad=20, fontweight='bold')
    
    # Bottom section
    ax_bottom = fig.add_subplot(gs[2, :])
    ax_bottom.axis('off')
    
    # Save the visualization
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()


def save_individual_samples(samples_data: list, output_dir: Path) -> None:
    """Save individual sample images with questions embedded above them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples_data):
        # Create a new image with space for the question above
        original_img = sample['image']
        img_width, img_height = original_img.size
        
        # Create a new image with extra height for the question
        question_height = 80  # Reduced height for question text
        new_height = img_height + question_height
        new_img = Image.new('RGB', (img_width, new_height), color='white')
        
        # Paste the original image below the question area
        new_img.paste(original_img, (0, question_height))
        
        # Add question text above the image
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(new_img)
        
        # Try to use serif fonts matching visualize_feature.py style
        try:
            # Try Times New Roman or DejaVu Serif (Linux)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 30)
        except:
            try:
                # Try Times New Roman alternative
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 30)
            except:
                try:
                    # Try macOS Times New Roman
                    font = ImageFont.truetype("/System/Library/Fonts/Times New Roman.ttf", 30)
                except:
                    try:
                        # Try Windows Times New Roman
                        font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 30)
                    except:
                        # Fall back to default serif font
                        font = ImageFont.load_default()
        
        # Draw the question text
        question = sample['question']
        rank = sample['rank']
        magnitude = sample['magnitude']
        
        # Wrap long questions to fit the image width using actual font metrics
        words = question.split()
        lines = []
        current_line = ""
        max_width = img_width - 20  # 20 pixel margin
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            # Get actual text width using font metrics
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width > max_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # Draw the question lines centered with balanced spacing
        y_offset = question_height - 65  # Position with balanced spacing from image
        for line in lines:
            # Center each line horizontally
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x_center = (img_width - text_width) // 2
            
            draw.text((x_center, y_offset), line, fill='black', font=font)
            y_offset += 25
        
        # Save the combined image
        image_filename = f"sample_{sample['rank']:02d}_rank_{sample['rank']}.png"
        image_path = output_dir / image_filename
        new_img.save(image_path, "PNG")
        
        print(f"Saved sample {sample['rank']}: {image_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Extract top VQA samples from vqa_all_spatial directory")
    parser.add_argument("--feature-dir", type=str, required=True,
                        help="Path to the feature directory containing sample_info.json (e.g., results/stage_4/feature_samples/full/vqa_all_spatial/pretrained_layer_26_feature_807)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for extracted samples")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top samples to extract (default: 20)")
    parser.add_argument("--create-viz", action="store_true",
                        help="Create a visualization of all top samples")
    
    args = parser.parse_args()
    
    try:
        # Load sample info
        feature_dir = Path(args.feature_dir)
        if not feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
        
        print(f"[INFO] Loading sample info from: {feature_dir}")
        samples_info = load_sample_info(feature_dir)
        print(f"[INFO] Found {len(samples_info)} samples, extracting top {args.top_k}")
        
        # Load VQA samples
        samples_data = load_vqa_samples(samples_info, args.top_k)
        
        if len(samples_data) == 0:
            print("Warning: No samples loaded")
            return
        
        # Set up output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual samples
        print(f"Saving {len(samples_data)} samples to {output_dir}")
        save_individual_samples(samples_data, output_dir)
        
        # Create visualization if requested
        if args.create_viz:
            viz_path = output_dir / "top_samples_visualization.png"
            print(f"Creating visualization...")
            create_top_samples_visualization(samples_data, str(viz_path))
        
        # Print summary
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"  Feature directory: {feature_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Samples extracted: {len(samples_data)}")
        print(f"  Files created: {len(samples_data)} individual images")
        if args.create_viz:
            print(f"  Visualization created: top_samples_visualization.png")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
