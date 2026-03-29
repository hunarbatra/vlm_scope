"""
Extract top 10 samples from all features across multiple directories and create a comprehensive summary.

This script:
1. Goes through vqa_all_spatial, vqa_spatial_all_spatial, and vsr_all_spatial directories
2. For each feature, extracts top 10 samples from each dataset
3. Creates a structured JSON with feature -> dataset -> samples organization

Usage:
python autointerp/extract_all_features_summary.py --base-dir results/stage_4/feature_samples --output-file results/stage_4/feature_summary.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datasets import load_dataset
from PIL import Image
import re, hashlib
import tqdm


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

    def get_vqa_spatial_indices(self) -> List[int] | None:
        """Load spatial filter indices from cache if available.

        Returns list of base VQAv2 indices corresponding to the spatial subset, or None if not found.
        """
        if self._vqa_spatial_indices is not None:
            return self._vqa_spatial_indices

        candidates: List[Path] = []
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
            
            dataset = dataset.filter(lambda x: x["label"] == 1)
            print(f"[INFO] Loaded {len(dataset)} true statements from VSR")
            self.vsr_dataset = dataset
        return self.vsr_dataset


def load_sample_info(feature_dir: Path) -> List[Dict]:
    """Load sample_info.json from a feature directory."""
    sample_info_file = feature_dir / "sample_info.json"
    
    if not sample_info_file.exists():
        return []
    
    try:
        with open(sample_info_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {sample_info_file}: {e}")
        return []


def get_top_samples_from_dataset(dataset, sample_indices: List[int], dataset_type: str, top_k: int = 10, spatial_indices: List[int] | None = None) -> List[Dict]:
    """Extract top K samples from a dataset - same pattern as extract_top_samples.py.

    If dataset_type is 'vqa_spatial', sample_indices are relative to the filtered
    spatial subset ordering. In that case, use spatial_indices (a list mapping
    filtered index → base VQAv2 index) to look up the correct base sample.
    """
    samples = []
    
    for i, sample_info in enumerate(sample_indices[:top_k]):
        try:
            sample_idx = sample_info["sample_idx"]
            base_idx = sample_idx
            if dataset_type == "vqa_spatial":
                if spatial_indices is None:
                    raise ValueError("VQA spatial indices mapping not provided")
                if sample_idx < 0 or sample_idx >= len(spatial_indices):
                    raise IndexError(f"VQA spatial sample_idx {sample_idx} out of range {len(spatial_indices)}")
                base_idx = int(spatial_indices[sample_idx])
            sample = dataset[base_idx]
            
            if dataset_type == "vsr":
                question = sample.get("caption", "N/A")
            else:
                question = sample.get("question", "N/A")
            
            samples.append({
                'rank': sample_info["rank"],
                'sample_idx': base_idx,
                'magnitude': sample_info["magnitude"],
                'question': question
            })
            
        except Exception as e:
            print(f"Warning: Could not load sample {sample_info.get('sample_idx', 'unknown')}: {e}")
            samples.append({
                'rank': sample_info["rank"],
                'sample_idx': sample_info.get("sample_idx", -1),
                'magnitude': sample_info["magnitude"],
                'question': f"Error loading sample: {e}"
            })
    
    return samples


def process_feature_directories(base_dir: Path, dataset_loader: DatasetLoader, top_k: int = 10) -> Dict:
    """Process all feature directories and extract top samples."""
    
    dataset_dirs = {
        'vqa': base_dir / 'vqa_all_spatial',
        'vqa_spatial': base_dir / 'vqa_spatial_all_spatial', 
        'vsr': base_dir / 'vsr_all_spatial_fixed'
    }
    
    existing_dirs = {name: path for name, path in dataset_dirs.items() if path.exists()}
    if not existing_dirs:
        raise FileNotFoundError(f"No dataset directories found in {base_dir}")
    
    print(f"[INFO] Processing {len(existing_dirs)} dataset directories:")
    for name, path in existing_dirs.items():
        print(f"  - {name}: {path}")
    
    all_features = set()
    name_pattern = re.compile(r'^(?:text-only_)?layer_(\d+)_feature_(\d+)$')
    for dataset_name, dataset_dir in existing_dirs.items():
        if dataset_dir.exists():
            for feature_dir in dataset_dir.iterdir():
                if not feature_dir.is_dir():
                    continue
                m = name_pattern.match(feature_dir.name)
                if not m:
                    continue
                try:
                    layer = int(m.group(1))
                    feature = int(m.group(2))
                    all_features.add((layer, feature))
                except Exception:
                    continue
    
    print(f"[INFO] Found {len(all_features)} unique features across all datasets")
    
    feature_summary = {}
    
    for layer, feature in tqdm.tqdm(sorted(all_features), desc="Processing features"):
        feature_key = f"layer_{layer}_feature_{feature}"
        feature_summary[feature_key] = {
            'layer': layer,
            'feature': feature,
            'datasets': {}
        }
        
        for dataset_name, dataset_dir in existing_dirs.items():
            if not dataset_dir.exists():
                continue
                
            candidates = [
                dataset_dir / f"text-only_layer_{layer}_feature_{feature}",
                dataset_dir / f"layer_{layer}_feature_{feature}",
            ]
            feature_dir = next((p for p in candidates if p.exists()), None)
            if feature_dir is None:
                continue
            
            samples_info = load_sample_info(feature_dir)
            if not samples_info:
                continue
            
            if dataset_name == 'vqa':
                dataset = dataset_loader.get_vqa_dataset()
                dataset_type = 'vqa'
            elif dataset_name == 'vqa_spatial':
                dataset = dataset_loader.get_vqa_dataset()
                dataset_type = 'vqa_spatial'
            elif dataset_name == 'vsr':
                dataset = dataset_loader.get_vsr_dataset()
                dataset_type = 'vsr'
            else:
                continue
            
            spatial_indices = None
            if dataset_type == 'vqa_spatial':
                spatial_indices = dataset_loader.get_vqa_spatial_indices()
            top_samples = get_top_samples_from_dataset(dataset, samples_info, dataset_type, top_k, spatial_indices)
            
            if top_samples:
                feature_summary[feature_key]['datasets'][dataset_name] = {
                    'total_samples': len(samples_info),
                    'top_samples': top_samples
                }
    
    return feature_summary, existing_dirs


def main():
    parser = argparse.ArgumentParser(description="Extract top samples from all features across multiple datasets")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base directory containing the three dataset folders")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top samples to extract per feature (default: 10)")
    parser.add_argument("--vqa-spatial-cache-dir", type=str, default=None,
                        help="Directory containing cached spatial VQA indices (indices_validation_*.json)")
    parser.add_argument("--vqa-spatial-cache-file", type=str, default=None,
                        help="Explicit path to a cached spatial indices JSON file")
    
    args = parser.parse_args()
    
    try:
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
        
        dataset_loader = DatasetLoader()
        dataset_loader.configure_vqa_spatial_cache(args.vqa_spatial_cache_dir, args.vqa_spatial_cache_file)
        
        print(f"[INFO] Starting feature extraction from {base_dir}")
        feature_summary, existing_dirs = process_feature_directories(base_dir, dataset_loader, args.top_k)
        
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_output = {
            'metadata': {
                'script_version': 'extract_all_features_summary.py',
                'base_directory': str(base_dir),
                'top_k': args.top_k,
                'total_features_processed': len(feature_summary),
                'dataset_directories_found': list(existing_dirs.keys())
            },
            'features': feature_summary
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"  Base directory: {base_dir}")
        print(f"  Output file: {output_path}")
        print(f"  Features processed: {len(feature_summary)}")
        print(f"  Top samples per feature: {args.top_k}")
        
        dataset_counts = {}
        for feature_data in feature_summary.values():
            num_datasets = len(feature_data['datasets'])
            dataset_counts[num_datasets] = dataset_counts.get(num_datasets, 0) + 1
        
        print(f"\nDataset coverage:")
        for num_datasets, count in sorted(dataset_counts.items()):
            print(f"  {num_datasets} dataset(s): {count} features")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
