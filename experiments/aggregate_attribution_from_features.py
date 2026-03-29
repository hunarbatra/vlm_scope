#!/usr/bin/env python3
"""
Aggregate normalized attribution summaries for a set of features.
- Inputs: CSV with columns: layer,feature; attribution_summary.json
- For each feature, use only layers/heads strictly before the feature's layer.
- Normalize per-feature distributions before aggregating across features.
- Outputs a JSON with averaged layer distributions and aggregated top heads per method.

python experiments/aggregate_attribution_from_features.py \
  --features-csv /homes/55/lachin/llama-scope-finetune-3/results/experiments/features_accuracy_drop_gt_2_with_odds_ratio.csv \
  --attribution-json /homes/55/lachin/llama-scope-finetune-3/results/experiments/attribution_summary.json \
  --output-json /homes/55/lachin/llama-scope-finetune-3/results/experiments/filtered.json

python experiments/aggregate_attribution_from_features.py --features-csv results/stage6/top_200_features_ablation_filtered_renamed.csv --attribution-json results/experiments/attribution_summary.json --output-json results/stage6/aggregated_attribution_from_filtered_features.json


"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_features(csv_path: Path) -> List[Tuple[int, int]]:
    features: List[Tuple[int, int]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features.append((int(row["layer"]), int(row["feature"])))
            except Exception:
                continue
    return features


def aggregate(features_csv: Path, attribution_json: Path, output_json: Path) -> None:
    features = load_features(features_csv)

    with attribution_json.open("r") as f:
        data = json.load(f)
    feature_map: Dict[str, dict] = data.get("features", {})

    method_stats = {
        "A": {
            "layer_sum": {},           # layer_index -> accumulated normalized score
            "layer_count": {},         # layer_index -> count of features contributing to this index
            "weighted_layer_index_sum": 0.0,
            "feature_count": 0,
            "head_score_sum": {},      # "LxHy" -> accumulated normalized head score
            "head_eligibility_count": {},  # "LxHy" -> how many features had this head eligible
        },
        "B": {
            "layer_sum": {},
            "layer_count": {},
            "weighted_layer_index_sum": 0.0,
            "feature_count": 0,
            "head_score_sum": {},
            "head_eligibility_count": {},
        },
    }

    missing = 0

    for layer, feat_id in features:
        key = f"layer_{layer}_feature_{feat_id}"
        entry = feature_map.get(key)
        if not entry:
            missing += 1
            continue
        attribution = entry.get("attribution", {})

        for method_key, tag in (("A", "method_A"), ("B", "method_B")):
            # Normalize layer scores before feature's layer
            all_layer_scores: List[float] = attribution.get(f"layer_scores_{tag}", [])
            pre_scores = all_layer_scores[:layer] if layer > 0 else []
            total = float(sum(pre_scores))
            if total > 0.0 and pre_scores:
                norm = [v / total for v in pre_scores]
                for li, v in enumerate(norm):
                    method_stats[method_key]["layer_sum"][li] = (
                        method_stats[method_key]["layer_sum"].get(li, 0.0) + v
                    )
                    method_stats[method_key]["layer_count"][li] = (
                        method_stats[method_key]["layer_count"].get(li, 0) + 1
                    )
                weighted_avg_index = sum(i * v for i, v in enumerate(norm))
                method_stats[method_key]["weighted_layer_index_sum"] += weighted_avg_index
            method_stats[method_key]["feature_count"] += 1

            # Normalize top head scores for heads before feature's layer
            heads: List[dict] = attribution.get(f"top_heads_{tag}", [])
            heads = [h for h in heads if int(h.get("layer", -1)) < layer]
            
            # Count eligibility for all heads before this feature's layer
            for li in range(layer):
                for hi in range(32):  # Assuming 32 heads per layer
                    head_name = f"L{li}H{hi}"
                    method_stats[method_key]["head_eligibility_count"][head_name] = (
                        method_stats[method_key]["head_eligibility_count"].get(head_name, 0) + 1
                    )
            
            head_total = float(sum(h.get("score", 0.0) for h in heads))
            if head_total > 0.0 and heads:
                for h in heads:
                    name = f"L{int(h.get('layer', -1))}H{int(h.get('head', -1))}"
                    method_stats[method_key]["head_score_sum"][name] = (
                        method_stats[method_key]["head_score_sum"].get(name, 0.0)
                        + (h.get("score", 0.0) / head_total)
                    )

    # Prepare output
    output = {"meta": {}, "methods": {}}
    output["meta"] = {
        "input_features_csv": str(features_csv),
        "attribution_summary_json": str(attribution_json),
        "missing_features": missing,
    }

    for method_key in ("A", "B"):
        ms = method_stats[method_key]
        avg_layer_distribution = []
        for li in sorted(ms["layer_sum"].keys()):
            denom = ms["layer_count"].get(li, 1)
            avg_layer_distribution.append(
                {"layer": li, "avg_norm_score": ms["layer_sum"][li] / denom}
            )
        # Normalize head scores by eligibility count
        top_heads = []
        for name, score_sum in ms["head_score_sum"].items():
            eligibility_count = ms["head_eligibility_count"].get(name, 1)
            normalized_score = score_sum / eligibility_count
            top_heads.append({"name": name, "total_norm_score": normalized_score})
        
        top_heads = sorted(top_heads, key=lambda x: x["total_norm_score"], reverse=True)
        avg_weighted_layer_index = (
            ms["weighted_layer_index_sum"] / ms["feature_count"]
            if ms["feature_count"]
            else 0.0
        )
        output["methods"][method_key] = {
            "features_processed": ms["feature_count"],
            "avg_weighted_layer_index": avg_weighted_layer_index,
            "avg_layer_distribution": avg_layer_distribution,
            "aggregated_top_heads": top_heads,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate attribution from feature list")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path(
            "/homes/55/lachin/llama-scope-finetune-3/results/experiments/all_features.csv"
        ),
        help="Path to features CSV with columns layer,feature",
    )
    parser.add_argument(
        "--attribution-json",
        type=Path,
        default=Path(
            "/homes/55/lachin/llama-scope-finetune-3/results/experiments/attribution_summary.json"
        ),
        help="Path to attribution_summary.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "/homes/55/lachin/llama-scope-finetune-3/results/experiments/aggregated_attribution_from_all_features.json"
        ),
        help="Path to write aggregated JSON output",
    )
    args = parser.parse_args()

    aggregate(args.features_csv, args.attribution_json, args.output_json)


if __name__ == "__main__":
    main()
