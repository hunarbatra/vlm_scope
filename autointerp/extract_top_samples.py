"""
Extract top samples from sample_info.json and create a folder with images and questions.

This script:
1. Reads sample_info.json from a feature directory
2. Loads the top samples from the VQA dataset
3. Creates a folder with the images and questions

Usage:
CUDA_VISIBLE_DEVICES=7 python autointerp/extract_top_samples.py \
    --base-dir results/stage_4/feature_samples \
    --layer  25 \
    --feature 245 \
    --output-dir results/stage_4/manual_top_test \
    --cache-dir /scratch/local/ssd/lachin/vsr_image_cache
    --vqa-only


    CUDA_VISIBLE_DEVICES=7 python autointerp/extract_top_samples.py \
  --base-dir results/stage_4/feature_samples \
  --dataset-dir results/stage_4/feature_samples/full/vqa_hallucinated \
  --dataset-type vqa \
  --layer 11 \
  --feature 18529 \
  --output-dir results/stage_4/manual_top_test \
  --top-k 20 \
  --create-viz
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
import shutil

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


class DatasetLoader:
    """Load and manage different datasets."""
    
    def __init__(self):
        self.vqa_dataset = None
        self.vsr_dataset = None
        self._vqa_spatial_indices = None
        self._vqa_spatial_cache_dir = None
        self._vqa_spatial_cache_file = None
    
    def get_vqa_dataset(self):
        """Lazy load VQA dataset - same as extract_top_samples.py."""
        if self.vqa_dataset is None:
            print("[INFO] Loading VQA dataset...")
            self.vqa_dataset = load_dataset("lmms-lab/VQAv2", split="validation")
            print(f"[INFO] Loaded {len(self.vqa_dataset)} VQA samples")
        return self.vqa_dataset
    
    def configure_vqa_spatial_cache(self, cache_dir: str | None, cache_file: str | None):
        self._vqa_spatial_cache_dir = cache_dir
        self._vqa_spatial_cache_file = cache_file

    def get_vqa_spatial_indices(self) -> list[int] | None:
        """Load spatial filter indices from cache if available.

        Returns list of base VQAv2 indices corresponding to the spatial subset, or None if not found.
        """
        if self._vqa_spatial_indices is not None:
            return self._vqa_spatial_indices

        candidates: list[Path] = []
        if self._vqa_spatial_cache_file:
            p = Path(self._vqa_spatial_cache_file)
            if p.exists():
                candidates.append(p)

        search_dirs = []
        if self._vqa_spatial_cache_dir:
            search_dirs.append(Path(self._vqa_spatial_cache_dir))
        search_dirs.append(Path(".cache/vqa_spatial_filter"))
        for d in search_dirs:
            if d.exists():
                for f in sorted(d.glob("indices_validation_*.json")):
                    candidates.append(f)

        for f in candidates:
            try:
                payload = json.loads(Path(f).read_text())
                indices = payload.get("indices") or payload.get("filtered_indices")
                if indices and isinstance(indices, list):
                    self._vqa_spatial_indices = [int(x) for x in indices]
                    print(f"[INFO] Loaded VQA spatial indices from {f} ({len(self._vqa_spatial_indices)} entries)")
                    return self._vqa_spatial_indices
            except Exception as e:
                print(f"[WARN] Failed reading spatial indices from {f}: {e}")

        print("[WARN] No VQA spatial indices cache found; vqa_spatial indices may not resolve correctly")
        return None

    def get_vsr_dataset(self):
        """Lazy load VSR dataset - using same structure as extract_vsr_features.py."""
        if self.vsr_dataset is None:
            print("[INFO] Loading VSR dataset...")
            data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
            dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="train")
            
            print(f"[INFO] Loaded {len(dataset)} samples from VSR")
            self.vsr_dataset = dataset
        return self.vsr_dataset


def load_sample_info(feature_dir: Path) -> list:
    """Load sample_info.json from the feature directory."""
    sample_info_file = feature_dir / "sample_info.json"
    
    if not sample_info_file.exists():
        raise FileNotFoundError(f"sample_info.json not found: {sample_info_file}")
    
    with open(sample_info_file, 'r') as f:
        return json.load(f)


def load_image_with_cache(url: str, cache_dir: str = "/scratch/local/ssd/lachin/vsr_image_cache", timeout: int = 10) -> Image.Image:
    """Load image with disk caching to avoid re-downloading (same as ablate_head_vsr_nnsight.py)."""
    import os
    import hashlib
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{url_hash}.jpg")
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load cached image {cache_path}: {e}")
    try:
        import requests
        from io import BytesIO
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        try:
            img.save(cache_path, "JPEG")
        except Exception as e:
            print(f"[WARN] Failed to cache image: {e}")
        return img
    except Exception as e:
        print(f"[WARN] Failed to download image {url}: {e}")
        return Image.new("RGB", (224, 224), (128, 128, 128))


def load_top_samples_from_dataset(dataset, samples_info: list, dataset_type: str, top_k: int = 10, spatial_indices: list[int] | None = None, cache_dir: str = "/scratch/local/ssd/lachin/vsr_image_cache") -> list:
    """Load the top K sample images from a dataset with proper spatial indices handling.

    If dataset_type is 'vqa_spatial', sample_indices are relative to the filtered
    spatial subset ordering. In that case, use spatial_indices (a list mapping
    filtered index → base VQAv2 index) to look up the correct base sample.
    """
    samples_data = []
    
    warned_missing_stable_fields = False
    for i, sample_info in enumerate(samples_info[:top_k]):
        try:
            sample_idx = sample_info["sample_idx"]
            base_idx = sample_idx
            stored_image_link = sample_info.get("image_link")
            stored_caption = sample_info.get("caption")
            
            if dataset_type == "vqa_spatial":
                if spatial_indices is None:
                    raise ValueError("VQA spatial indices mapping not provided")
                if sample_idx < 0 or sample_idx >= len(spatial_indices):
                    raise IndexError(f"VQA spatial sample_idx {sample_idx} out of range {len(spatial_indices)}")
                base_idx = int(spatial_indices[sample_idx])
            
            if dataset_type == "vsr":
                if stored_caption is not None or stored_image_link is not None:
                    question = stored_caption if stored_caption is not None else "N/A"
                    answer = "N/A"
                    image_url = stored_image_link
                else:
                    if not warned_missing_stable_fields and (stored_image_link is None and stored_caption is None):
                        print("[WARN] sample_info.json lacks 'image_link'/'caption'. Indices may not match current VSR dataset. Re-run extraction to embed stable fields.")
                        warned_missing_stable_fields = True
                    sample = dataset[base_idx]
                    question = sample.get("caption", "N/A")
                    answer = "N/A"  # VSR doesn't have answers
                    image_url = sample.get("image_link", None)
                if image_url is None:
                    image = Image.new('RGB', (224, 224), color='gray')
                else:
                    try:
                        if isinstance(image_url, str):
                            image = load_image_with_cache(image_url, cache_dir=cache_dir)
                        else:
                            image = image_url.convert("RGB")
                    except Exception as img_e:
                        print(f"Warning: Could not load VSR image from {image_url}: {img_e}")
                        image = Image.new('RGB', (224, 224), color='gray')
            else:
                sample = dataset[base_idx]
                image = sample["image"].convert("RGB")
                question = sample["question"]
                answer = sample.get("answer", "N/A")
            
            samples_data.append({
                'image': image,
                'question': question,
                'answer': answer,
                'rank': sample_info["rank"],
                'magnitude': sample_info["magnitude"],
                'sample_idx': base_idx
            })
            print(f"Loaded {dataset_type} sample {base_idx}: {question[:50]}...")
            
        except Exception as e:
            print(f"Warning: Could not load {dataset_type} sample {sample_info.get('sample_idx', 'unknown')}: {e}")
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
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.6, 5, 0.8], hspace=0.05, wspace=0.1)
    
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    ax_images = fig.add_subplot(gs[1, :])
    ax_images.axis('off')
    
    img_gs = GridSpec(2, 5, left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.2, hspace=-0.2)
    
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
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#666')
                spine.set_linewidth(1)
            
            rank = sample['rank']
            magnitude = sample['magnitude']
            question = sample['question']
            
            wrapped_question = textwrap.fill(question, width=25)
            
            title_text = f"Rank {rank}\nMag: {magnitude:.2f}\n{wrapped_question}"
            ax.set_title(title_text, fontsize=9, pad=20, fontweight='bold')
    
    ax_bottom = fig.add_subplot(gs[2, :])
    ax_bottom.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()


def save_individual_samples(samples_data: list, output_dir: Path) -> None:
    """Save individual sample images with questions embedded above them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples_data):
        original_img = sample['image']
        img_width, img_height = original_img.size
        
        question_height = 80  # Reduced height for question text
        new_height = img_height + question_height
        new_img = Image.new('RGB', (img_width, new_height), color='white')
        
        new_img.paste(original_img, (0, question_height))
        
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(new_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 30)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 30)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Times New Roman.ttf", 30)
                except:
                    try:
                        font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 30)
                    except:
                        font = ImageFont.load_default()
        
        question = sample['question']
        rank = sample['rank']
        magnitude = sample['magnitude']
        
        words = question.split()
        lines = []
        current_line = ""
        max_width = img_width - 20  # 20 pixel margin
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
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
        
        y_offset = question_height - 65  # Position with balanced spacing from image
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x_center = (img_width - text_width) // 2
            
            draw.text((x_center, y_offset), line, fill='black', font=font)
            y_offset += 25
        
        image_filename = f"sample_{sample['rank']:02d}_rank_{sample['rank']}.png"
        image_path = output_dir / image_filename
        new_img.save(image_path, "PNG")
        
        print(f"Saved sample {sample['rank']}: {image_path.name}")


def process_feature_across_datasets(base_dir: Path, layer: int, feature: int, top_k: int, only_vqa: bool = False) -> dict:
    """Process a feature across datasets and return results.

    If only_vqa is True, restrict processing to the VQA dataset.
    """
    dataset_dirs = {
        'vqa': base_dir / 'vqa_all_spatial',
        'vqa_spatial': base_dir / 'vqa_spatial_all_spatial',
        'vsr': base_dir / 'vsr_all_spatial_fixed'
    }
    if only_vqa:
        dataset_dirs = {
            'vqa': base_dir / 'vqa_all_spatial'
        }
    
    results = {}
    
    for dataset_name, dataset_dir in dataset_dirs.items():
        if not dataset_dir.exists():
            print(f"[WARN] Dataset directory not found: {dataset_dir}")
            continue
            
        candidates = [
            dataset_dir / f"text-only_layer_{layer}_feature_{feature}",
            dataset_dir / f"layer_{layer}_feature_{feature}",
        ]
        feature_dir = next((p for p in candidates if p.exists()), None)
        
        if feature_dir is None:
            print(f"[WARN] Feature directory not found in {dataset_name}: tried {candidates}")
            continue
        
        try:
            samples_info = load_sample_info(feature_dir)
            if samples_info:
                results[dataset_name] = {
                    'feature_dir': feature_dir,
                    'samples_info': samples_info,
                    'total_samples': len(samples_info)
                }
                print(f"[INFO] Found {len(samples_info)} samples in {dataset_name}")
        except Exception as e:
            print(f"[WARN] Error loading {dataset_name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract top samples from sample_info.json across datasets")
    parser.add_argument("--base-dir", type=str, required=True, 
                        help="Base directory containing the three dataset folders (e.g., results/stage_4/feature_samples)")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer number (e.g., 31)")
    parser.add_argument("--feature", type=int, required=True,
                        help="Feature number (e.g., 19124)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for extracted samples")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top samples to extract (default: 20)")
    parser.add_argument("--create-viz", action="store_true",
                        help="Create a visualization of all top samples")
    parser.add_argument("--only-vqa", action="store_true",
                        help="Process only the VQA dataset (skip VQA spatial and VSR)")
    parser.add_argument("--vqa-spatial-cache-dir", type=str, default=None,
                        help="Directory containing cached spatial VQA indices (indices_validation_*.json)")
    parser.add_argument("--vqa-spatial-cache-file", type=str, default=None,
                        help="Explicit path to a cached spatial indices JSON file")
    parser.add_argument("--cache-dir", type=str, default="/scratch/local/ssd/lachin/tmp/vsr_image_cache_test",
                        help="Directory to cache downloaded VSR images")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to a single dataset directory that contains per-feature subfolders (e.g., results/stage_4/feature_samples/full/vqa_ocr)")
    parser.add_argument("--dataset-type", type=str, default="vqa", choices=["vqa", "vqa_spatial", "vsr"],
                        help="Type of the dataset located at --dataset-dir (default: vqa)")
    
    args = parser.parse_args()
    
    try:
        if args.dataset_dir is not None:
            single_dir = Path(args.dataset_dir)
            if not single_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {single_dir}")
            print(f"[INFO] Processing single dataset dir: {single_dir} (type={args.dataset_type})")
            candidates = [
                single_dir / f"text-only_layer_{args.layer}_feature_{args.feature}",
                single_dir / f"layer_{args.layer}_feature_{args.feature}",
            ]
            feature_dir = next((p for p in candidates if p.exists()), None)
            if feature_dir is None:
                raise FileNotFoundError(f"Feature directory not found in {single_dir}: tried {candidates}")

            samples_info = load_sample_info(feature_dir)
            print(f"[INFO] Found {len(samples_info)} samples in single dataset")

            base_output_dir = Path(args.output_dir)
            feature_output_dir = base_output_dir / f"layer_{args.layer}_feature_{args.feature}"
            feature_output_dir.mkdir(parents=True, exist_ok=True)
            dataset_output_dir = feature_output_dir / single_dir.name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

            dataset_loader = DatasetLoader()
            dataset_loader.configure_vqa_spatial_cache(args.vqa_spatial_cache_dir, args.vqa_spatial_cache_file)

            if args.dataset_type == 'vqa_spatial':
                spatial_indices = dataset_loader.get_vqa_spatial_indices()
                if spatial_indices is None:
                    raise FileNotFoundError("VQA spatial indices not found; provide --vqa-spatial-cache-dir or --vqa-spatial-cache-file")
                samples_data = load_top_samples_from_dataset(dataset_loader.get_vqa_dataset(), samples_info, 'vqa_spatial', args.top_k, spatial_indices, args.cache_dir)
            elif args.dataset_type == 'vsr':
                samples_data = load_top_samples_from_dataset(dataset_loader.get_vsr_dataset(), samples_info, 'vsr', args.top_k, cache_dir=args.cache_dir)
            else:  # vqa
                samples_data = load_top_samples_from_dataset(dataset_loader.get_vqa_dataset(), samples_info, 'vqa', args.top_k)

            if len(samples_data) == 0:
                print(f"[WARN] No samples loaded from {single_dir}")
            else:
                print(f"  Saving {len(samples_data)} samples to {dataset_output_dir}")
                save_individual_samples(samples_data, dataset_output_dir)
                if args.create_viz:
                    viz_path = dataset_output_dir / "top_samples_visualization.png"
                    print(f"  Creating visualization...")
                    create_top_samples_visualization(samples_data, str(viz_path))
        else:
            base_dir = Path(args.base_dir)
            if not base_dir.exists():
                raise FileNotFoundError(f"Base directory not found: {base_dir}")
            
            scope_label = "VQA only" if args.only_vqa else "all datasets"
            print(f"[INFO] Processing feature layer_{args.layer}_feature_{args.feature} across {scope_label}...")
            dataset_results = process_feature_across_datasets(base_dir, args.layer, args.feature, args.top_k, args.only_vqa)
            
            if not dataset_results:
                raise FileNotFoundError(f"No feature data found for layer {args.layer}, feature {args.feature} in any dataset")
            
            print(f"[INFO] Found data in {len(dataset_results)} datasets: {list(dataset_results.keys())}")
            
            base_output_dir = Path(args.output_dir)
            feature_output_dir = base_output_dir / f"layer_{args.layer}_feature_{args.feature}"
            feature_output_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_loader = DatasetLoader()
            dataset_loader.configure_vqa_spatial_cache(args.vqa_spatial_cache_dir, args.vqa_spatial_cache_file)
            
            total_samples_processed = 0
            for dataset_name, dataset_data in dataset_results.items():
                print(f"\n[INFO] Processing {dataset_name} dataset...")
                
                dataset_output_dir = feature_output_dir / dataset_name
                dataset_output_dir.mkdir(parents=True, exist_ok=True)
                
                samples_info = dataset_data['samples_info']
                print(f"  Found {len(samples_info)} samples, extracting top {args.top_k}")
                
                if dataset_name == 'vqa_spatial':
                    spatial_indices = dataset_loader.get_vqa_spatial_indices()
                    if spatial_indices is None:
                        print(f"  Warning: VQA spatial indices not found. Cannot load VQA spatial samples.")
                        continue
                    samples_data = load_top_samples_from_dataset(dataset_loader.get_vqa_dataset(), samples_info, dataset_name, args.top_k, spatial_indices, args.cache_dir)
                elif dataset_name == 'vsr':
                    samples_data = load_top_samples_from_dataset(dataset_loader.get_vsr_dataset(), samples_info, dataset_name, args.top_k, cache_dir=args.cache_dir)
                else: # vqa and vqa_all_spatial
                    samples_data = load_top_samples_from_dataset(dataset_loader.get_vqa_dataset(), samples_info, dataset_name, args.top_k)
                
                if len(samples_data) > 0:
                    print(f"  Saving {len(samples_data)} samples to {dataset_output_dir}")
                    save_individual_samples(samples_data, dataset_output_dir)
                    total_samples_processed += len(samples_data)
                    
                    if args.create_viz:
                        viz_path = dataset_output_dir / "top_samples_visualization.png"
                        print(f"  Creating visualization...")
                        create_top_samples_visualization(samples_data, str(viz_path))
                else:
                    print(f"  Warning: No samples loaded from {dataset_name}")
        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"  Layer: {args.layer}")
        print(f"  Feature: {args.feature}")
        print(f"  Output directory: {feature_output_dir}")
        print(f"  Datasets processed: {list(dataset_results.keys())}")
        print(f"  Total samples extracted: {total_samples_processed}")
        print(f"  Files created: {total_samples_processed} individual images")
        if args.create_viz:
            print(f"  Visualizations created for each dataset")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
