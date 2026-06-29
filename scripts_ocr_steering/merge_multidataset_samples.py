#!/usr/bin/env python3
"""
Merge per-dataset feature samples into a single common features JSON.
Matches the original extract_common_features_summary.py output format.

Usage:
    python3 merge_multidataset_samples.py \
        --samples-dir /path/to/multidataset_feature_samples \
        --features /path/to/final_spatial_visual_features.csv \
        --output /path/to/dataset_all_features.json
"""

import argparse
import csv
import json
from pathlib import Path

ANALYSIS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=str, default=None)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=10, help="Top K samples per dataset to include")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir) if args.samples_dir else ANALYSIS_DIR / "multidataset_feature_samples"
    output_path = Path(args.output) if args.output else ANALYSIS_DIR / "dataset_all_features.json"

    # Load features
    features = []
    with open(args.features) as f:
        for row in csv.DictReader(f):
            features.append((int(row["layer"]), int(row["feature"])))
    print(f"Loaded {len(features)} features")

    # Discover datasets
    ds_names = sorted([d.name for d in samples_dir.iterdir() if d.is_dir()])
    print(f"Datasets found: {ds_names}")

    # Build feature summary
    feature_summary = {}
    for layer, feat in features:
        fkey = f"layer_{layer}_feature_{feat}"
        entry = {"layer": layer, "feature": feat, "datasets": {}}

        for ds_name in ds_names:
            sample_path = (samples_dir / ds_name / f"layer_{layer}" /
                          f"text-only_layer_{layer}_feature_{feat}" / "sample_info.json")
            if not sample_path.exists():
                continue
            with open(sample_path) as f:
                samples = json.load(f)
            if not samples:
                continue

            # Sort by magnitude, keep top-K
            samples.sort(key=lambda x: x.get("magnitude", 0), reverse=True)
            top = samples[:args.top_k]

            # Format to match original structure
            formatted = []
            for rank, s in enumerate(top, 1):
                formatted.append({
                    "rank": rank,
                    "sample_idx": s.get("sample_idx", s.get("base_idx", -1)),
                    "magnitude": s.get("magnitude", 0),
                    "question": s.get("text", s.get("caption", "")),
                })

            entry["datasets"][ds_name] = {
                "total_samples": len(samples),
                "top_samples": formatted,
            }

        if entry["datasets"]:
            feature_summary[fkey] = entry

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final = {
        "metadata": {
            "total_features": len(feature_summary),
            "datasets": ds_names,
            "top_k": args.top_k,
        },
        "features": feature_summary,
    }
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

    # Stats
    ds_counts = {ds: 0 for ds in ds_names}
    for fdata in feature_summary.values():
        for ds in fdata["datasets"]:
            ds_counts[ds] += 1
    print(f"\nMerged {len(feature_summary)} features:")
    for ds, cnt in ds_counts.items():
        print(f"  {ds}: {cnt} features with samples")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
