"""
Concise autointerp pipeline using RAW samples (no attention overlays).

This script:
- Reads a common features JSON (same structure used by features/spatial/attn_viz_common_features.py)
- For each feature, gathers top samples across datasets (vqa, vqa_spatial, vsr)
- Loads the corresponding raw image and text (question/caption)
- Sends up to N samples per feature to a vision-capable API
- Saves one concise one-sentence interpretation per feature as JSON, plus a summary

Input JSON structure: payload["features"][feature_key] with keys:
- layer, feature, datasets: { vqa: {top_samples: [...]}, vqa_spatial: {...}, vsr: {...} }
Each top_samples entry should contain sample_idx, magnitude, and question/caption.

Usage:
python autointerp/auto_interp_common_features_raw_samples.py \
  --common-summary-path results/experiments/dataset_all_features.json \
  --output-dir results/auto_interp_common_features_raw \
  --samples-per-feature 5 \
  --feature-filter layer_15_feature_10748
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import requests
from datasets import load_dataset
from PIL import Image as PILImage

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



class SampleRow:
    def __init__(self, feature_key: str, dataset: str, sample_idx: int, magnitude: float, text: str, layer: Optional[int], feature: Optional[int]):
        self.feature_key = feature_key
        self.dataset = dataset
        self.sample_idx = sample_idx
        self.magnitude = magnitude
        self.text = text
        self.layer = layer
        self.feature = feature



def _encode_pil_to_base64(img: PILImage) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _safe_int(x, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_common_features(common_summary_path: Path, feature_filter: str, samples_per_feature: int) -> Dict[str, Dict]:
    with open(common_summary_path, "r") as f:
        payload = json.load(f)

    features = payload.get("features", {})
    result: Dict[str, Dict] = {}
    spatial_index_map = _load_vqa_spatial_indices()

    for feature_key, info in features.items():
        if feature_filter and feature_filter not in feature_key:
            continue
        try:
            layer = int(info.get("layer"))
        except Exception:
            layer = -1
        try:
            feat_id = int(info.get("feature"))
        except Exception:
            feat_id = -1
        ds_info = info.get("datasets", {})

        merged_with_base: List[Dict] = []
        for ds_name in ("vqa", "vqa_spatial", "vsr"):
            ds_block = ds_info.get(ds_name) or {}
            for s in (ds_block.get("top_samples") or []):
                try:
                    idx = _safe_int(s.get("sample_idx"))
                    mag = _safe_float(s.get("magnitude"), 0.0)
                    text = s.get("question") if ds_name != "vsr" else s.get("caption")
                    if idx is None:
                        continue
                    base_vqa_idx: Optional[int] = None
                    if ds_name == "vqa":
                        base_vqa_idx = int(idx)
                    elif ds_name == "vqa_spatial":
                        try:
                            if spatial_index_map and 0 <= int(idx) < len(spatial_index_map):
                                base_vqa_idx = int(spatial_index_map[int(idx)])
                        except Exception:
                            base_vqa_idx = None
                    merged_with_base.append({
                        "sample_row": SampleRow(
                            feature_key=feature_key,
                            dataset=ds_name,
                            sample_idx=int(idx),
                            magnitude=float(mag),
                            text=str(text or ""),
                            layer=layer,
                            feature=feat_id,
                        ),
                        "_base_vqa_idx": base_vqa_idx,
                    })
                except Exception:
                    continue

        if not merged_with_base:
            continue

        merged_with_base.sort(key=lambda r: r["sample_row"].magnitude, reverse=True)

        seen_base: set = set()
        seen_pair: set = set()
        deduped_rows: List[SampleRow] = []
        for rec in merged_with_base:
            r: SampleRow = rec["sample_row"]
            base = rec.get("_base_vqa_idx")
            pair = (r.dataset, r.sample_idx)
            if base is not None:
                if base in seen_base:
                    continue
                seen_base.add(base)
            else:
                if pair in seen_pair:
                    continue
                seen_pair.add(pair)
            deduped_rows.append(r)

        top_subset = deduped_rows[:samples_per_feature] if samples_per_feature > 0 else list(deduped_rows)

        result[feature_key] = {
            "layer": layer,
            "feature": feat_id,
            "samples": top_subset,
            "all_samples_sorted": deduped_rows,
        }
    return result


def _load_vqa_validation():
    return load_dataset("lmms-lab/VQAv2", split="validation")


def _load_vsr_train():
    try:
        data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        return load_dataset("cambridgeltl/vsr_random", data_files=data_files, split="train")
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
    """Map a (dataset, idx) to base VQA validation index if possible.

    - For 'vqa': returns idx
    - For 'vqa_spatial': returns spatial_index_map[idx] if available
    - Otherwise: returns None
    """
    try:
        if dataset_name == "vqa":
            return int(idx)
        if dataset_name == "vqa_spatial":
            if spatial_index_map and 0 <= int(idx) < len(spatial_index_map):
                return int(spatial_index_map[int(idx)])
    except Exception:
        pass
    return None

def _create_api_prompt_with_images(feature_key: str, samples: List[Dict]) -> Dict:
    system_message = (
    "You are analyzing individual neurons using their top activating samples, each with an image and a question.\n\n"
    "Task: Produce a concise ONE SENTENCE description that completes the phrase: 'this neuron activates for ...'.\n\n"
    "Guidelines:\n"
        "- Base your description on pattens supported by both image and text.\n"
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
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{s['image_b64']}"}
        })

    schema = (
        "Return strict JSON matching exactly:\n"
        "{\n"
        "  \"description\": \"one concise sentence\"\n"
        "}"
    )
    content.append({"type": "text", "text": schema})

    user_message = {"role": "user", "content": content}
    return {"system_message": system_message, "user_message": user_message}


def _create_eval_prompt_with_images(description: str, samples: List[Dict]) -> Dict:
    """Create a prompt to classify each sample as matches_description (1) or not (0)."""
    system_message = (
    "You are validating a neuron description by reviewing short examples (each has an image and a brief text).\n\n"
    "Task: For each sample, decide if it reasonably matches the neuron description. Output 1 if the description is supported; otherwise 0.\n\n"
    "Guidelines:\n"
    "- Use both the image and the text; let the text clarify ambiguous visuals when helpful.\n"
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
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{s['image_b64']}"}
        })

    schema = (
        "Return strict JSON matching exactly:\n"
        "{\n"
        "  \"classifications\": [0, 1, 0]\n"
        "}"
    )
    content.append({"type": "text", "text": schema})

    user_message = {"role": "user", "content": content}
    return {"system_message": system_message, "user_message": user_message}


def _compute_f1(preds: List[int], labels: List[int]) -> float:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return (2.0 * tp) / denom


def _call_api(prompt: Dict, api_key: str, timeout_s: int = 120) -> Optional[Dict]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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



def process_features(
    common_summary_path: Path,
    output_dir: Path,
    api_key: str,
    samples_per_feature: int = 5,
    delay_s: float = 1.0,
    feature_filter: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading common features JSON: {common_summary_path}")
    feature_map = _parse_common_features(common_summary_path, feature_filter, samples_per_feature)
    feature_keys = sorted(feature_map.keys())
    print(f"[INFO] Processing {len(feature_keys)} features")

    print("[INFO] Loading datasets...")
    ds_vqa = _load_vqa_validation()
    ds_vsr = _load_vsr_train()
    spatial_index_map = _load_vqa_spatial_indices()

    results = []
    for i, fkey in enumerate(feature_keys):
        meta = feature_map[fkey]
        layer = int(meta.get("layer", -1))
        feat = int(meta.get("feature", -1))
        sample_rows: List[SampleRow] = meta["samples"]
        all_rows: List[SampleRow] = meta.get("all_samples_sorted", sample_rows)

        print(f"[INFO] ({i+1}/{len(feature_keys)}) {fkey} -> L{layer} F{feat}")

        samples_payload: List[Dict] = []
        seen_texts: set = set()
        for r in all_rows:
            if len(samples_payload) >= 5:
                break
            ds_name = r.dataset
            idx = int(r.sample_idx)
            mag = float(r.magnitude)
            text_override = (r.text or "").strip()

            try:
                if ds_name in ("vqa", "vqa_spatial"):
                    base_idx = idx
                    if ds_name == "vqa_spatial" and spatial_index_map and 0 <= idx < len(spatial_index_map):
                        base_idx = int(spatial_index_map[idx])
                    sample = ds_vqa[base_idx]
                    image = sample["image"].convert("RGB")
                    default_text = str(sample.get("question", "")).strip()
                    text = text_override if text_override else (default_text if default_text else "Answer the question.")
                elif ds_name == "vsr":
                    sample = ds_vsr[idx] if ds_vsr is not None else None
                    image_url = sample.get("image_link") if sample is not None else None
                    caption = str(sample.get("caption", "")).strip() if sample is not None else ""
                    text = text_override if text_override else (caption if caption else "Describe the image.")
                    try:
                        if image_url:
                            resp = requests.get(image_url, timeout=10)
                            resp.raise_for_status()
                            image = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                        else:
                            image = PILImage.new("RGB", (224, 224), (128, 128, 128))
                    except Exception:
                        image = PILImage.new("RGB", (224, 224), (128, 128, 128))
                else:
                    continue

                text_norm = text.strip().lower()
                if text_norm in seen_texts:
                    continue
                seen_texts.add(text_norm)
                b64 = _encode_pil_to_base64(image)
                samples_payload.append({
                    "dataset": ds_name,
                    "sample_idx": idx,
                    "magnitude": mag,
                    "text": text,
                    "image_b64": b64,
                })
            except Exception as e:
                print(f"[WARN] Failed to load sample {idx} from {ds_name}: {e}")
                continue

        if not samples_payload:
            print(f"[WARN] No valid samples for {fkey}")
            continue

        prompt = _create_api_prompt_with_images(fkey, samples_payload)
        interp = _call_api(prompt, api_key)
        if not interp:
            print(f"[WARN] Failed to get interpretation for {fkey}")
            continue

        samples_for_saving = []
        for s in samples_payload:
            samples_for_saving.append({
                "dataset": s["dataset"],
                "sample_idx": s["sample_idx"],
                "magnitude": s["magnitude"],
                "text": s["text"],
            })

        desc = str(interp.get("description", "")).strip()
        orig_conf = interp.get("confidence", None)

        used_keys = {(s["dataset"], int(s["sample_idx"])) for s in samples_for_saving}
        pos_candidates: List[SampleRow] = []
        for r in all_rows:
            key = (r.dataset, int(r.sample_idx))
            if key in used_keys:
                continue
            pos_candidates.append(r)
        eval_pos = pos_candidates[:5]

        eval_neg: List[Tuple[str, int, str]] = []  # (dataset, idx, text)
        if ds_vqa is not None:
            used_base_vqa: set[int] = set()
            for ds_name, idx_used in used_keys:
                base_idx = _to_base_vqa_idx(ds_name, idx_used, spatial_index_map)
                if base_idx is not None:
                    used_base_vqa.add(base_idx)
            pos_pairs = set()
            for r in eval_pos:
                pos_pairs.add((r.dataset, int(r.sample_idx)))
                base_idx = _to_base_vqa_idx(r.dataset, r.sample_idx, spatial_index_map)
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
                eval_neg.append(("vqa", ridx, q_text))

        eval_pos_payload: List[Dict] = []
        for r in eval_pos:
            try:
                if r.dataset in ("vqa", "vqa_spatial"):
                    base_idx = r.sample_idx
                    if r.dataset == "vqa_spatial" and spatial_index_map and 0 <= r.sample_idx < len(spatial_index_map):
                        base_idx = int(spatial_index_map[r.sample_idx])
                    sample = ds_vqa[base_idx]
                    image = sample["image"].convert("RGB")
                    default_text = str(sample.get("question", "")).strip()
                    text = r.text if r.text else (default_text if default_text else "Answer the question.")
                elif r.dataset == "vsr":
                    sample = ds_vsr[r.sample_idx] if ds_vsr is not None else None
                    image_url = sample.get("image_link") if sample is not None else None
                    caption = str(sample.get("caption", "")).strip() if sample is not None else ""
                    text = r.text if r.text else (caption if caption else "Describe the image.")
                    try:
                        if image_url:
                            resp = requests.get(image_url, timeout=10)
                            resp.raise_for_status()
                            image = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                        else:
                            image = PILImage.new("RGB", (224, 224), (128, 128, 128))
                    except Exception:
                        image = PILImage.new("RGB", (224, 224), (128, 128, 128))
                else:
                    continue
                b64 = _encode_pil_to_base64(image)
                eval_pos_payload.append({
                    "dataset": r.dataset,
                    "sample_idx": r.sample_idx,
                    "text": text,
                    "image_b64": b64,
                    "label": 1,
                })
            except Exception:
                continue

        eval_neg_payload: List[Dict] = []
        for ds_name, idx, txt in eval_neg:
            try:
                image = ds_vqa[idx]["image"].convert("RGB")
                b64 = _encode_pil_to_base64(image)
                eval_neg_payload.append({
                    "dataset": ds_name,
                    "sample_idx": idx,
                    "text": txt if txt else "Answer the question.",
                    "image_b64": b64,
                    "label": 0,
                })
            except Exception:
                continue

        r1_pos_n = min(3, len(eval_pos_payload))
        r1_neg_n = min(2, len(eval_neg_payload))
        r2_pos_n = min(2, max(0, len(eval_pos_payload) - r1_pos_n))
        r2_neg_n = min(3, max(0, len(eval_neg_payload) - r1_neg_n))

        r1_payload = eval_pos_payload[:r1_pos_n] + eval_neg_payload[:r1_neg_n]
        r2_payload = eval_pos_payload[r1_pos_n:r1_pos_n + r2_pos_n] + eval_neg_payload[r1_neg_n:r1_neg_n + r2_neg_n]

        f1_conf = None
        eval_details = None
        all_preds: List[int] = []
        all_labels: List[int] = []
        selected_payloads: List[Dict] = []

        for payload in (r1_payload, r2_payload):
            if not payload:
                continue
            random.shuffle(payload)
            eval_prompt = _create_eval_prompt_with_images(desc, payload)
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
                    "text": sample["text"],
                    "true_label": label,
                    "predicted": pred,
                    "correct": correct,
                    "result_type": result_type
                })

            eval_pos_save = [{
                "dataset": s["dataset"],
                "sample_idx": s["sample_idx"],
                "text": s["text"],
            } for s in selected_payloads if s.get("label") == 1]
            eval_neg_save = [{
                "dataset": s["dataset"],
                "sample_idx": s["sample_idx"],
                "text": s["text"],
            } for s in selected_payloads if s.get("label") == 0]
            preds = all_preds
            eval_details = {
                "positives_used": eval_pos_save,
                "negatives_used": eval_neg_save,
                "predictions": preds if 'preds' in locals() else None,
                "f1": f1_conf,
                "detailed_results": results_detail,
            }

        final_interp = {
            "description": desc,
        }

        record = {
            "feature_key": fkey,
            "layer": layer,
            "feature": feat,
            "samples_used": samples_for_saving,
            "interpretation": final_interp,
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
    parser = argparse.ArgumentParser(description="Auto-interpret common features using RAW samples (no overlays)")
    parser.add_argument("--common-summary-path", type=str, required=True, help="Path to common_features_summary JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save interpretations")
    parser.add_argument("--api-key", type=str, default=None, help="API key or set OPENAI_API_KEY in env")
    parser.add_argument("--samples-per-feature", type=int, default=5, help="Max number of samples per feature")
    parser.add_argument("--delay-s", type=float, default=1.0, help="Delay between API calls in seconds")
    parser.add_argument("--feature-filter", type=str, default="", help="Optional substring filter for feature keys")
    args = parser.parse_args()

    common_summary_path = Path(args.common_summary_path)
    output_dir = Path(args.output_dir)
    if not common_summary_path.exists():
        print(f"[ERROR] Common summary not found: {common_summary_path}")
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] No API key provided. Pass --api-key or set OPENAI_API_KEY in env.")
        return

    process_features(
        common_summary_path=common_summary_path,
        output_dir=output_dir,
        api_key=api_key,
        samples_per_feature=args.samples_per_feature,
        delay_s=args.delay_s,
        feature_filter=args.feature_filter,
    )


if __name__ == "__main__":
    main()


