"""
Automated interpretation of common features using attention overlay images.

This script:
1) Reads features from a common_features_summary JSON
2) For each feature, collects up to N top samples that have TWO attention overlay images
   (produced by features/spatial/attn_viz_common_features.py with top_k=2 heads)
3) Packages each sample as: dataset name, question/caption text, magnitude, and two overlay images
4) Sends them to a vision-capable model with strict JSON response formatting
5) Saves a concise, precise one-sentence description per feature

Notes:
- Expects overlay images under: {attn_viz_dir}/{feature_key}/attn_top_B/
- Filenames follow: {ds}_sample{idx}_L{layer}_H{head}.png
- We only include samples where we can find exactly two overlay images (two heads)

Example:
python autointerp/auto_interp_common_features_attn_images.py \
  --common-summary-path results/stage_4/common_features_summary_35_2_with_passed_with_relations.json \
  --attn-viz-dir results/common_attn_viz \
  --output-dir results/auto_interp_common_features_attn \
  --samples-per-feature 5 \
  --feature-filter layer_16_feature_176

CUDA_VISIBLE_DEVICES=7 python autointerp/auto_interp_common_features_attn_images.py \
  --common-summary-path results/experiments/dataset_all_features.json \
  --attn-viz-dir results/common_attn_viz \
  --output-dir results/auto_interp_common_features_attn \
  --samples-per-feature 5 \
  --feature-filter layer_15_feature_10748 \
  --precompute-overlays \
  --combined-attrib-path results/experiments/attribution_summary.json \
  --viz-top-k 1 \
  --viz-samples-extra 15 \
    --precompute-overlays 
"""

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import subprocess

import requests
from datasets import load_dataset
from PIL import Image as PILImage

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _read_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def _encode_image_file_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")

def _encode_pil_to_base64(img: PILImage.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _load_vqa_validation():
    try:
        return load_dataset("lmms-lab/VQAv2", split="validation")
    except Exception:
        return None

def _load_vqa_spatial_indices(cache_dir: str = ".cache/vqa_spatial_filter") -> List[int]:
    candidates: List[Path] = []
    indices: List[int] = []
    try:
        search_dirs = [Path(cache_dir), Path(".cache/vqa_spatial_filter")]
        for d in search_dirs:
            if d.exists():
                candidates.extend(sorted(d.glob("indices_validation_*.json")))
        for f in candidates:
            try:
                payload = json.loads(Path(f).read_text())
                vals = payload.get("indices") or payload.get("filtered_indices")
                if vals and isinstance(vals, list):
                    indices = [int(x) for x in vals]
                    break
            except Exception:
                continue
    except Exception:
        pass
    return indices

def _to_base_vqa_idx(dataset_name: str, idx: int, spatial_index_map: List[int]) -> Optional[int]:
    try:
        if dataset_name == "vqa":
            return int(idx)
        if dataset_name == "vqa_spatial":
            if spatial_index_map and 0 <= int(idx) < len(spatial_index_map):
                return int(spatial_index_map[int(idx)])
    except Exception:
        pass
    return None


def _collect_overlay_pairs_for_feature(
    feature_key: str,
    attn_viz_dir: Path,
    heads_per_sample: int = 1,
) -> Dict[Tuple[str, int], List[Tuple[int, int, Path]]]:
    """Collect overlay images for a feature and group them by (dataset, sample_idx).

    Returns mapping:
        (ds_name, sample_idx) -> List[(layer, head, image_path)]
    """
    base_dir = attn_viz_dir / feature_key / "attn_top_B"
    groups: Dict[Tuple[str, int], List[Tuple[int, int, Path]]] = {}
    if not base_dir.exists():
        print(f"[DEBUG] Directory does not exist: {base_dir}")
        return groups

    print(f"[DEBUG] Looking for images in: {base_dir}")
    png_files = list(base_dir.glob("*.png"))
    print(f"[DEBUG] Found {len(png_files)} PNG files")
    
    pattern = re.compile(r"^(?P<ds>[^_]+(?:_[^_]+)*)_sample(?P<idx>\d+)_L(?P<layer>\d+)_H(?P<head>\d+)\.png$")
    for p in sorted(png_files):
        print(f"[DEBUG] Checking file: {p.name}")
        m = pattern.match(p.name)
        if not m:
            print(f"[DEBUG] Pattern did not match: {p.name}")
            continue
        ds = m.group("ds")
        idx = int(m.group("idx"))
        ly = int(m.group("layer"))
        hd = int(m.group("head"))
        key = (ds, idx)
        groups.setdefault(key, []).append((ly, hd, p))
        print(f"[DEBUG] Matched: {ds}, {idx}, L{ly}, H{hd}")

    print(f"[DEBUG] Found {len(groups)} sample groups")
    groups = {k: v for k, v in groups.items() if len(v) >= heads_per_sample}
    print(f"[DEBUG] After filtering for >={heads_per_sample} overlays: {len(groups)} groups")
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda t: (t[0], t[1]))[:heads_per_sample]
    return groups


def _build_sample_records(
    feature_key: str,
    feature_entry: dict,
    overlay_groups: Dict[Tuple[str, int], List[Tuple[int, int, Path]]],
    samples_per_feature: int,
    heads_per_sample: int = 1,
) -> List[dict]:
    """Select up to samples_per_feature records that have overlays and attach metadata.

    We prefer samples that appear in the common summary's merged top list (by magnitude).
    """
    preferred: List[Tuple[str, int, float, str]] = []  # (ds, idx, magnitude, text)
    datasets = (feature_entry.get("datasets") or {})
    for ds_name, ds_block in datasets.items():
        for s in (ds_block.get("top_samples") or []):
            try:
                idx = int(s.get("sample_idx"))
                mag = float(s.get("magnitude", 0.0))
                text = s.get("question") or s.get("caption") or ""
                preferred.append((ds_name, idx, mag, text))
            except Exception:
                continue
    preferred.sort(key=lambda t: t[2], reverse=True)

    used = []
    seen_pairs: set[Tuple[str, int]] = set()
    seen_texts: set[str] = set()
    for ds, idx, mag, text in preferred:
        key = (ds if ds != "vqa_spatial" else "vqa_spatial", int(idx))
        if key in seen_pairs:
            continue
        if key not in overlay_groups:
            continue
        lyhd_paths = overlay_groups[key]
        if len(lyhd_paths) < heads_per_sample:
            continue
        text_norm = (text or "").strip().lower()
        if text_norm in seen_texts:
            continue
        used.append({
            "dataset": ds,
            "sample_idx": idx,
            "magnitude": mag,
            "text": text,
            "overlays": [
                {"layer": ly, "head": hd, "path": str(p)} for (ly, hd, p) in lyhd_paths[:heads_per_sample]
            ]
        })
        seen_pairs.add(key)
        seen_texts.add(text_norm)
        if 0 < samples_per_feature <= len(used):
            break

    return used


def _create_api_prompt_with_images(feature_key: str, samples: List[dict], heads_per_sample: int = 1) -> Dict:
    """Create a multimodal prompt that includes text and attention overlay images."""
    system_message = (
        "You are analysing individual neurons based on their top activating samples, each with an image, question, and attention overlay.\n\n"
        "Task: Write a precise ONE-SENTENCE description of what the neuron detects.\n\n"
        "Guidelines:\n"
        "- Base your description on the visual regions highlighted by attention overlays and how they align with the question text.\n"
        "- Focus on consistent visual–spatial patterns (objects, parts, relations, or configurations).\n"
        "- Be specific and concrete; avoid vague or generic phrases.\n"
        "- Keep the output to one short, lower-case sentence with no hedging.\n"
        "- Return strict JSON only with fields.\n"
    )

    content: List[dict] = []
    content.append({"type": "text", "text": f"Feature: {feature_key}. Analyze the following samples."})
    for i, s in enumerate(samples):
        header = (
            f"Sample {i+1}: dataset={s['dataset']}, sample_idx={s['sample_idx']}, "
            f"magnitude={s.get('magnitude', 0):.4f}.\n"
            f"Text: {s.get('text', '')}"
        )
        content.append({"type": "text", "text": header})
        for ov in s.get("overlays", [])[:heads_per_sample]:
            try:
                b64 = _encode_image_file_to_base64(Path(ov["path"]))
                content.append({
                    "type": "text",
                    "text": f"overlay: L{ov['layer']} H{ov['head']}"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            except Exception:
                continue

    schema = (
        "Return strict JSON matching exactly:\n"
        "{\n"
        "  \"description\": \"one concise sentence\"\n"
        "}"
    )
    content.append({"type": "text", "text": schema})

    user_message = {
        "role": "user",
        "content": content,
    }

    return {
        "system_message": system_message,
        "user_message": user_message,
    }


def _create_eval_prompt_with_overlays(description: str, samples: List[dict], heads_per_sample: int = 1) -> Dict:
    """Create a prompt to classify each sample (with overlays) as match (1) or not (0)."""
    system_message = (
        "You are validating a neuron description by reviewing short examples (each has an image/text and attention overlays).\n\n"
        "Task: For each sample, decide if it reasonably matches the neuron description. Output 1 if the description is supported; otherwise 0.\n\n"
        "Guidelines:\n"
        "- Use both the overlay-highlighted regions and the accompanying text; let the text clarify ambiguous visuals when helpful.\n"
        "- Be tolerant of minor mismatches; look for the main idea rather than exact wording.\n"
        "- Prefer consistency across similar cases.\n\n"
        "Output format:\n"
        "- Return JSON only.\n"
        "- Use exactly: {\"classifications\": [<0 or 1 per sample, in order>]}.\n"
    )

    content: List[dict] = []
    content.append({"type": "text", "text": f"Neuron description: {description}"})
    for i, s in enumerate(samples):
        header = (
            f"Sample {i+1}: dataset={s['dataset']}, sample_idx={s['sample_idx']}\n"
            f"Text: {s.get('text', '')}"
        )
        content.append({"type": "text", "text": header})
        if s.get("image_b64"):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{s['image_b64']}"}
            })
        for ov in s.get("overlays", [])[:heads_per_sample]:
            try:
                b64 = _encode_image_file_to_base64(Path(ov["path"]))
                content.append({"type": "text", "text": f"overlay: L{ov['layer']} H{ov['head']}"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            except Exception:
                continue

    schema = (
        "Return strict JSON matching exactly:\n"
        "{\n"
        "  \"classifications\": [0, 1, 0]\n"
        "}"
    )
    content.append({"type": "text", "text": schema})

    return {
        "system_message": system_message,
        "user_message": {"role": "user", "content": content},
    }


def _compute_f1(preds: List[int], labels: List[int]) -> float:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return (2.0 * tp) / denom


def _call_api(prompt: Dict, api_key: str, timeout_s: int = 120) -> Optional[Dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": prompt["system_message"]},
            prompt["user_message"],
        ],
        "max_tokens": 400,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return None


def _maybe_generate_overlays(
    common_summary_path: Path,
    attn_viz_dir: Path,
    combined_attrib_path: Optional[Path],
    feature_filter: str,
    viz_top_k: int,
    viz_samples_per_feature: int,
    manifest_path: Optional[Path],
) -> None:
    """Optionally invoke the attention visualization script to precompute overlays.

    This generates overlays under attn_viz_dir so that evaluation can include them.
    """
    try:
        if not combined_attrib_path:
            print("[WARN] --combined-attrib-path not provided; cannot precompute overlays")
            return
        script_path = Path(__file__).resolve().parent.parent / "features" / "spatial" / "attn_viz_common_features.py"
        if not script_path.exists():
            print(f"[WARN] Visualization script not found at {script_path}")
            return
        cmd = [
            "python", str(script_path),
            "--common-summary-path", str(common_summary_path),
            "--combined-attrib-path", str(combined_attrib_path),
            "--output-dir", str(attn_viz_dir),
            "--top-k", str(int(viz_top_k)),
            "--samples-per-feature", str(int(viz_samples_per_feature)),
        ]
        if feature_filter:
            cmd.extend(["--feature-filter", feature_filter])
        if manifest_path:
            cmd.extend(["--manifest-path", str(manifest_path)])
        print(f"[INFO] Precomputing overlays via: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[WARN] Failed to precompute overlays: {e}")


def process_features(
    common_summary_path: Path,
    attn_viz_dir: Path,
    output_dir: Path,
    api_key: str,
    samples_per_feature: int = 5,
    delay_s: float = 1.0,
    feature_filter: str = "",
    heads_per_sample: int = 1,
    precompute_overlays: bool = False,
    combined_attrib_path: Optional[Path] = None,
    viz_top_k: int = 1,
    viz_samples_extra: int = 10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_json(common_summary_path)
    features = payload.get("features", {})

    feature_keys = sorted(features.keys())
    if feature_filter:
        feature_keys = [k for k in feature_keys if feature_filter in k]

    print(f"[INFO] Processing {len(feature_keys)} features")

    ds_vqa = _load_vqa_validation()
    spatial_index_map = _load_vqa_spatial_indices()

    if precompute_overlays:
        try:
            manifest: Dict[str, List[Dict[str, int]]] = {}
            for fkey in feature_keys:
                info = features.get(fkey) or {}
                ds_info = info.get("datasets") or {}
                merged: List[Tuple[str, int, float, str]] = []
                for ds_name, ds_block in ds_info.items():
                    for s in (ds_block.get("top_samples") or []):
                        try:
                            merged.append((ds_name, int(s.get("sample_idx")), float(s.get("magnitude", 0.0)), s.get("question") or s.get("caption") or ""))
                        except Exception:
                            continue
                merged.sort(key=lambda t: t[2], reverse=True)
                take = max(0, int(samples_per_feature) + int(viz_samples_extra))
                pairs = [{"dataset": ds, "sample_idx": idx} for ds, idx, _, _ in merged[:take]]
                if pairs:
                    manifest[fkey] = pairs
            manifest_path = attn_viz_dir / "manifest_for_overlays.json"
            attn_viz_dir.mkdir(parents=True, exist_ok=True)
            (manifest_path).write_text(json.dumps(manifest, indent=2))
        except Exception as e:
            print(f"[WARN] Failed to build manifest for overlays: {e}")
            manifest_path = None

        viz_samples_per_feature = max(0, int(samples_per_feature) + int(viz_samples_extra))
        _maybe_generate_overlays(
            common_summary_path=common_summary_path,
            attn_viz_dir=attn_viz_dir,
            combined_attrib_path=combined_attrib_path,
            feature_filter=feature_filter,
            viz_top_k=viz_top_k,
            viz_samples_per_feature=viz_samples_per_feature,
            manifest_path=manifest_path,
        )
    results = []

    for i, fkey in enumerate(feature_keys):
        info = features.get(fkey) or {}
        layer = int(info.get("layer", -1))
        feat = int(info.get("feature", -1))
        print(f"[INFO] ({i+1}/{len(feature_keys)}) {fkey} -> L{layer} F{feat}")

        overlay_groups = _collect_overlay_pairs_for_feature(fkey, attn_viz_dir, heads_per_sample=heads_per_sample)
        if not overlay_groups:
            print(f"[WARN] No overlay images found for {fkey} in {attn_viz_dir}")
            continue

        samples = _build_sample_records(fkey, info, overlay_groups, samples_per_feature, heads_per_sample=heads_per_sample)
        if not samples:
            print(f"[WARN] No eligible samples (with overlays) for {fkey}")
            continue

        prompt = _create_api_prompt_with_images(fkey, samples, heads_per_sample=heads_per_sample)
        interp = _call_api(prompt, api_key)
        if not interp:
            print(f"[WARN] Failed to get interpretation for {fkey}")
            continue

        desc = str((interp.get("description") if isinstance(interp, dict) else "") or "").strip()
        f1_conf = None
        eval_details = None

        used_pairs = {(s["dataset"], int(s["sample_idx"])) for s in samples}
        text_lookup: Dict[Tuple[str, int], str] = {}
        try:
            ds_info_local = (info.get("datasets") or {})
            for ds_name_l, ds_block_l in ds_info_local.items():
                for s in (ds_block_l.get("top_samples") or []):
                    try:
                        idx_l = int(s.get("sample_idx"))
                        txt_l = s.get("question") or s.get("caption") or ""
                        text_lookup[(ds_name_l, idx_l)] = txt_l
                    except Exception:
                        continue
        except Exception:
            pass

        preferred_mag: List[Tuple[str, int, float, str]] = []
        ds_info_pref = (info.get("datasets") or {})
        for ds_name_p, ds_block_p in ds_info_pref.items():
            for s in (ds_block_p.get("top_samples") or []):
                try:
                    preferred_mag.append((ds_name_p, int(s.get("sample_idx")), float(s.get("magnitude", 0.0)), s.get("question") or s.get("caption") or ""))
                except Exception:
                    continue
        preferred_mag.sort(key=lambda t: t[2], reverse=True)

        pos_candidates: List[dict] = []
        for ds_name, idx, _mag, txt in preferred_mag:
            pair = (ds_name, int(idx))
            if pair in used_pairs:
                continue
            if pair not in overlay_groups:
                continue
            lyhd = overlay_groups[pair]
            if not lyhd:
                continue
            pos_candidates.append({
                "dataset": ds_name,
                "sample_idx": int(idx),
                "text": text_lookup.get(pair, txt),
                "overlays": [
                    {"layer": ly, "head": hd, "path": str(p)} for (ly, hd, p) in lyhd[:heads_per_sample]
                ],
            })
            if len(pos_candidates) >= 5:
                break

        eval_pos = pos_candidates

        eval_neg: List[dict] = []  # List of dicts with dataset, sample_idx, text, image_b64
        if ds_vqa is not None:
            used_base_vqa: set[int] = set()
            for ds_name_u, idx_u in used_pairs:
                base_idx = _to_base_vqa_idx(ds_name_u, idx_u, spatial_index_map)
                if base_idx is not None:
                    used_base_vqa.add(base_idx)
            pos_pairs = set()
            for r in eval_pos:
                pos_pairs.add((r["dataset"], int(r["sample_idx"])))
                base_idx = _to_base_vqa_idx(r["dataset"], int(r["sample_idx"]), spatial_index_map)
                if base_idx is not None:
                    used_base_vqa.add(base_idx)

            vqa_len = len(ds_vqa)
            tried = set()
            while len(eval_neg) < 5 and len(tried) < vqa_len * 3:
                ridx = random.randint(0, vqa_len - 1)
                if ridx in tried:
                    continue
                tried.add(ridx)
                if ridx in used_base_vqa:
                    continue
                if ("vqa", ridx) in pos_pairs:
                    continue
                q_text = str(ds_vqa[ridx].get("question", "")).strip()
                try:
                    image = ds_vqa[ridx]["image"].convert("RGB")
                    b64 = _encode_pil_to_base64(image)
                    eval_neg.append({
                        "dataset": "vqa",
                        "sample_idx": ridx,
                        "text": q_text,
                        "image_b64": b64,
                    })
                except Exception:
                    continue

        eval_pos_payload: List[dict] = [{**s, "label": 1} for s in eval_pos]
        eval_neg_payload: List[dict] = [{**s, "label": 0} for s in eval_neg]

        r1_pos_n = min(3, len(eval_pos_payload))
        r1_neg_n = min(2, len(eval_neg_payload))
        r2_pos_n = min(2, max(0, len(eval_pos_payload) - r1_pos_n))
        r2_neg_n = min(3, max(0, len(eval_neg_payload) - r1_neg_n))

        r1_payload = eval_pos_payload[:r1_pos_n] + eval_neg_payload[:r1_neg_n]
        r2_payload = eval_pos_payload[r1_pos_n:r1_pos_n + r2_pos_n] + eval_neg_payload[r1_neg_n:r1_neg_n + r2_neg_n]

        all_preds: List[int] = []
        all_labels: List[int] = []
        selected_payloads: List[dict] = []
        for payload in (r1_payload, r2_payload):
            if not payload:
                continue
            random.shuffle(payload)
            eval_prompt = _create_eval_prompt_with_overlays(desc, payload, heads_per_sample=heads_per_sample)
            eval_resp = _call_api(eval_prompt, api_key)
            try:
                round_preds = [int(x) for x in (eval_resp.get("classifications") or [])]
                round_labels = [int(s.get("label", 0)) for s in payload][: len(round_preds)]
                all_preds.extend(round_preds)
                all_labels.extend(round_labels)
                selected_payloads.extend(payload[: len(round_preds)])
            except Exception:
                continue

        if all_preds and all_labels:
            f1_conf = _compute_f1(all_preds, all_labels)

        if selected_payloads:
            results_detail = []
            for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
                sample = selected_payloads[i]
                correct = pred == label
                result_type = "tp" if pred == 1 and label == 1 else "fp" if pred == 1 and label == 0 else "tn" if pred == 0 and label == 0 else "fn"
                results_detail.append({
                    "dataset": sample["dataset"],
                    "sample_idx": sample["sample_idx"],
                    "text": sample.get("text", ""),
                    "true_label": label,
                    "predicted": pred,
                    "correct": correct,
                    "result_type": result_type,
                })

            eval_pos_save = [{
                "dataset": s["dataset"],
                "sample_idx": s["sample_idx"],
                "text": s.get("text", ""),
            } for s in selected_payloads if s.get("label") == 1]
            eval_neg_save = [{
                "dataset": s["dataset"],
                "sample_idx": s["sample_idx"],
                "text": s.get("text", ""),
            } for s in selected_payloads if s.get("label") == 0]
            eval_details = {
                "positives_used": eval_pos_save,
                "negatives_used": eval_neg_save,
                "predictions": all_preds,
                "f1": f1_conf,
                "detailed_results": results_detail,
            }

        record = {
            "feature_key": fkey,
            "layer": layer,
            "feature": feat,
            "samples_used": samples,
            "interpretation": interp,
            "validation": eval_details,
        }

        out_file = output_dir / f"{fkey}.json"
        with open(out_file, "w") as f:
            json.dump(record, f, indent=2)
        results.append(record)
        print(f"[INFO] Saved {out_file}")

        time.sleep(delay_s)

    summary = {
        "total_features": len(feature_keys),
        "processed_features": len(results),
        "results": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Complete. Summary at {output_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Auto-interpret common features using attention overlay images")
    parser.add_argument("--common-summary-path", type=str, required=True,
                        help="Path to common_features_summary_*.json")
    parser.add_argument("--attn-viz-dir", type=str, default="results/common_attn_viz",
                        help="Base directory where attn_viz_common_features.py saved outputs")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save interpretations")
    parser.add_argument("--heads-per-sample", type=int, default=1,
                        help="Number of overlay heads per sample to include (K)")
    parser.add_argument("--precompute-overlays", action="store_true",
                        help="If set, generate overlays with top-1 head before evaluation")
    parser.add_argument("--combined-attrib-path", type=str, default=None,
                        help="Path to attribution JSON for overlay generation (required if --precompute-overlays)")
    parser.add_argument("--viz-top-k", type=int, default=1,
                        help="Top-k heads to use when precomputing overlays (default 1)")
    parser.add_argument("--viz-samples-extra", type=int, default=10,
                        help="Extra samples per feature to visualize beyond interpretation set for eval")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key or set OPENAI_API_KEY in env")
    parser.add_argument("--samples-per-feature", type=int, default=5,
                        help="Max number of samples (each with overlay) per feature")
    parser.add_argument("--delay-s", type=float, default=1.0,
                        help="Delay between API calls in seconds")
    parser.add_argument("--feature-filter", type=str, default="",
                        help="Optional substring filter for feature keys")
    args = parser.parse_args()

    common_summary_path = Path(args.common_summary_path)
    attn_viz_dir = Path(args.attn_viz_dir)
    output_dir = Path(args.output_dir)

    if not common_summary_path.exists():
        print(f"[ERROR] Common summary not found: {common_summary_path}")
        return
    if not attn_viz_dir.exists():
        print(f"[ERROR] Attention viz dir not found: {attn_viz_dir}")
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] No API key provided. Pass --api-key or set OPENAI_API_KEY in env.")
        return

    process_features(
        common_summary_path=common_summary_path,
        attn_viz_dir=attn_viz_dir,
        output_dir=output_dir,
        api_key=api_key,
        samples_per_feature=args.samples_per_feature,
        delay_s=args.delay_s,
        feature_filter=args.feature_filter,
        heads_per_sample=args.heads_per_sample,
        precompute_overlays=args.precompute_overlays,
        combined_attrib_path=Path(args.combined_attrib_path) if args.combined_attrib_path else None,
        viz_top_k=args.viz_top_k,
        viz_samples_extra=args.viz_samples_extra,
    )


if __name__ == "__main__":
    main()


