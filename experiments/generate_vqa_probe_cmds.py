#!/usr/bin/env python3
"""
Generate (and optionally run) VQA spatial probe commands for features in a CSV.

For each row with columns `layer,feature` this script:
- Builds a command to run features/spatial/probe_vqa_spatial_features.py
- Optionally executes it
- Optionally aggregates probe outputs into a consolidated JSON
results/stage_3/spatial/spatial_features_vqa.csv
Example:
CUDA_VISIBLE_DEVICES=1 python experiments/generate_vqa_probe_cmds.py \
  --csv-file results/stage_3/spatial/spatial_features_vqa.csv \
  --vqa-samples-dir results/stage_4/feature_samples/vqa_all_spatial \
  --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k \
  --method text-only \
  --samples-per-feature 5 \
  --vqa-split validation \
  --execute \
  --save-results \
  --results-json results/stage6/generated_probe_results_hard2.json
"""

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def build_command(
    layer: int,
    feature: int,
    vqa_samples_dir: str,
    sae_checkpoint_dir: str,
    method: str,
    samples_per_feature: int,
    vqa_split: str,
) -> str:
    parts: List[str] = [
        "python", "features/spatial/probe_vqa_spatial_features.py",
        "--vqa-samples-dir", vqa_samples_dir,
        "--sae-checkpoint-dir", sae_checkpoint_dir,
        "--method", method,
        "--layer", str(layer),
        "--feature", str(feature),
        "--samples-per-feature", str(samples_per_feature),
        "--vqa-split", vqa_split,
    ]
    return " ".join(shlex.quote(p) for p in parts)


def maybe_read_json(path: Path) -> Dict:
    try:
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"features": {}}


def append_probe_results(
    out_payload: Dict,
    layer: int,
    feature: int,
    method: str,
    vqa_samples_dir: Path,
) -> None:
    """Read per-layer CSV written by probe script and merge rows for this feature."""
    csv_name = f"visual_probe_layer{layer}_{method}_generic_passed.csv"
    csv_path = vqa_samples_dir / csv_name
    feat_key = f"layer_{layer}_feature_{feature}"

    entry = {
        "layer": int(layer),
        "feature": int(feature),
        "method": method,
        "probe": {
            "passed_generic_test": False,
            "records": [],
        },
    }

    if not csv_path.exists():
        out_payload.setdefault("features", {})[feat_key] = entry
        return

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row_layer = int(row.get("layer", -1))
                    row_feature = int(row.get("feature", -1))
                except Exception:
                    continue
                if row_layer == layer and row_feature == feature:
                    # Keep a compact subset of columns
                    rec = {
                        "sample_idx": int(row.get("sample_idx", -1)),
                        "magnitude": float(row.get("magnitude", 0.0)),
                        "rank": int(row.get("rank", 0)),
                        "orig_max_all": float(row.get("orig_max_all", 0.0)),
                        "orig_max_img": float(row.get("orig_max_img", 0.0)),
                        "orig_max_txt": float(row.get("orig_max_txt", 0.0)),
                        "generic_max_all": float(row.get("generic_max_all", 0.0)),
                        "generic_max_img": float(row.get("generic_max_img", 0.0)),
                        "generic_max_txt": float(row.get("generic_max_txt", 0.0)),
                        "generic_fired_any": row.get("generic_fired_any", "False") in ("True", "true", True),
                        "generic_fired_in_img": row.get("generic_fired_in_img", "False") in ("True", "true", True),
                        "question": row.get("question", ""),
                    }
                    entry["probe"]["records"].append(rec)

        if entry["probe"]["records"]:
            entry["probe"]["passed_generic_test"] = True
        out_payload.setdefault("features", {})[feat_key] = entry
    except Exception:
        out_payload.setdefault("features", {})[feat_key] = entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate/run VQA spatial probe commands from CSV features")
    ap.add_argument("--csv-file", required=True, help="CSV with columns: layer,feature")
    ap.add_argument("--vqa-samples-dir", required=True, help="Directory with feature samples (vqa_all_spatial)")
    ap.add_argument("--sae-checkpoint-dir", required=True, help="Directory with SAE checkpoints (per layer)")
    ap.add_argument("--method", default="text-only", help="SAE method name (e.g., text-only, pretrained)")
    ap.add_argument("--samples-per-feature", type=int, default=5, help="Top samples per feature to probe")
    ap.add_argument("--vqa-split", default="validation", choices=["train", "validation", "test"]) 
    ap.add_argument("--limit", type=int, default=0, help="Limit number of features (0=all)")
    ap.add_argument("--execute", action="store_true", help="Actually run each command (otherwise just print)")
    ap.add_argument("--output-cmds", default="probe_commands.txt", help="File to write generated commands")
    ap.add_argument("--save-results", action="store_true", help="Aggregate per-feature probe results into one JSON")
    ap.add_argument("--results-json", default="results/stage6/generated_probe_results.json")

    args = ap.parse_args()

    csv_path = Path(args.csv_file)
    vqa_samples_dir = Path(args.vqa_samples_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not vqa_samples_dir.exists():
        raise FileNotFoundError(f"VQA samples dir not found: {vqa_samples_dir}")

    commands: List[str] = []
    out_payload: Dict = maybe_read_json(Path(args.results_json)) if args.save_results else {"features": {}}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            try:
                layer = int(row.get("layer") or row.get("Layer"))
                feature = int(row.get("feature") or row.get("Feature"))
            except Exception:
                continue

            cmd = build_command(
                layer=layer,
                feature=feature,
                vqa_samples_dir=str(vqa_samples_dir),
                sae_checkpoint_dir=args.sae_checkpoint_dir,
                method=str(args.method),
                samples_per_feature=int(args.samples_per_feature),
                vqa_split=str(args.vqa_split),
            )

            print(cmd)
            commands.append(cmd)

            if args.execute:
                try:
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Command failed (code {e.returncode}): {cmd}")

            if args.save_results:
                append_probe_results(out_payload, layer, feature, str(args.method), vqa_samples_dir)
                try:
                    out_path = Path(args.results_json)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(out_payload, indent=2))
                    print(f"[INFO] Updated results JSON → {out_path}")
                except Exception as e:
                    print(f"[WARN] Failed to write results JSON: {e}")

            count += 1
            if args.limit and count >= args.limit:
                break

    try:
        out_cmds = Path(args.output_cmds)
        out_cmds.write_text("\n".join(commands) + ("\n" if commands else ""))
        print(f"[INFO] Wrote {len(commands)} command(s) → {out_cmds}")
    except Exception as e:
        print(f"[WARN] Failed to write commands file: {e}")


if __name__ == "__main__":
    main()


