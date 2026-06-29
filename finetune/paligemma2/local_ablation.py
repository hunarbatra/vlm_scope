#!/usr/bin/env python3
"""
Local multi-GPU ablation for PaliGemma2 spatial features.

Evaluates causal impact of ablating each feature's decoder direction
from all transformer layers (matching original ablate_sae_feature_vsr.py).

Measures: ∆VSR accuracy, ∆Control VSR, ∆VQA yes/no accuracy.

Usage:
    python3 -u local_ablation.py --features /path/to/features.csv --gpus 6 7
"""

import os
import sys
import json
import math
import time
import hashlib
import argparse
import warnings
import csv
from pathlib import Path
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# ---- Config ----
MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448/checkpoints")
ANALYSIS_DIR = Path("/data1/vlm_scope_sae_mix448/analysis")
HF_CACHE = "/data1/vlm_scope_sae_mix448/hf_cache/hub"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/vlm_scope_sae_mix448/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Eval config
VSR_DATASET = "cambridgeltl/vsr_random"
VQA_MAX = 1000      # max VQA yes/no samples
CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}

IMAGE_CACHE_DIR = Path("/data1/vlm_scope_sae_mix448/vsr_image_cache")


def _build_vsr_prompt(statement: str) -> str:
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\n"
        "Answer:"
    )


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES", " true", "true", "True"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO", " false", "false", "False"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids -= overlap
    return yes_ids, no_ids


def _predict_yesno(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


def _ablation_worker(gpu_id, feature_assignments, out_dir, vqa_max):
    """Ablate features and measure ∆VSR, ∆VQA, ∆Ctrl."""
    import requests
    from PIL import Image

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return

    print(f"[Ablation GPU{gpu_id}] {len(feature_assignments)} features", flush=True)

    # Load model
    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=False)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=False
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # Load VSR
    print(f"[Ablation GPU{gpu_id}] Loading VSR...", flush=True)
    vsr = load_dataset(VSR_DATASET, split="test")
    print(f"[Ablation GPU{gpu_id}] VSR test: {len(vsr)} samples", flush=True)

    # Load VQA yes/no
    print(f"[Ablation GPU{gpu_id}] Loading VQA yes/no...", flush=True)
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno = []
    for i in range(len(vqa)):
        ex = vqa[i]
        at = str(ex.get("answer_type", "")).lower()
        mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
        if at == "yes/no" and mc in {"yes", "no"}:
            vqa_yesno.append((i, 1 if mc == "yes" else 0))
            if len(vqa_yesno) >= vqa_max:
                break
    print(f"[Ablation GPU{gpu_id}] VQA yes/no: {len(vqa_yesno)}", flush=True)

    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_vsr_image(ex):
        url = ex.get("image_link", "")
        if not url.startswith("http"):
            return None
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = IMAGE_CACHE_DIR / f"{url_hash}.jpg"
        try:
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(cache_path, "JPEG")
            return img
        except Exception:
            return None

    def _do_ablation_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end):
        """Project out feature direction from all layers on text tokens.

        Matches original: projects from attn output, MLP output, AND residual stream
        (layer output) — all three sub-components per layer.
        """
        fv = feature_vec.unsqueeze(0)  # (1, d_in)
        with nns_model.trace(
            input_ids=input_ids, attention_mask=attn_mask,
            pixel_values=pixel_values,
        ) as tr:
            for l in range(N_LAYERS):
                # 1. Self-attention output
                attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                attn_proj = (attn_out @ fv.T) * fv
                attn_out -= attn_proj

                # 2. MLP output
                mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                mlp_proj = (mlp_out @ fv.T) * fv
                mlp_out -= mlp_proj

                # 3. Layer output (residual stream) — output is (hidden_states,) tuple
                layer_out = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                layer_proj = (layer_out @ fv.T) * fv
                layer_out -= layer_proj
            logits_saved = nns_model.output.logits.save()
        return logits_saved

    # --- Baseline pass (no ablation) ---
    baseline_path = out_dir / f"baseline_gpu{gpu_id}.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"[Ablation GPU{gpu_id}] Loaded cached baseline: VSR={baseline['vsr_acc']:.1f}%, "
              f"Ctrl={baseline['ctrl_acc']:.1f}%, VQA={baseline['vqa_acc']:.1f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing baseline...", flush=True)
        vsr_c = vsr_t = ctrl_c = ctrl_t = 0
        for vi in range(len(vsr)):
            ex = vsr[vi]
            img = _load_vsr_image(ex)
            if img is None:
                continue
            caption = str(ex.get("caption", "")).strip()
            label = int(ex.get("label", 0))
            relation = ex.get("relation", "")
            prompt = _build_vsr_prompt(caption)
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model_raw, device=device)
                with torch.inference_mode():
                    out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                    pixel_values=pixel_values, use_cache=False)
                pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            vsr_t += 1
            if pred == label:
                vsr_c += 1
            if relation in CONTROL_RELATIONS:
                ctrl_t += 1
                if pred == label:
                    ctrl_c += 1
            if vsr_t % 100 == 0:
                print(f"  [Baseline GPU{gpu_id}] VSR {vsr_t}: {vsr_c/vsr_t*100:.1f}%", flush=True)

        vqa_c = vqa_t = 0
        for qi, label in vqa_yesno:
            ex = vqa[qi]
            img = ex.get("image")
            if img is None:
                continue
            from PIL import Image as PILImage
            if isinstance(img, PILImage.Image):
                img = img.convert("RGB")
            else:
                continue
            question = ex.get("question", "")
            prompt = (
                "Answer the following question with only 'Yes' or 'No':\n"
                f"Question: {question.strip()}\n"
                "Answer:"
            )
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model_raw, device=device)
                with torch.inference_mode():
                    out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                    pixel_values=pixel_values, use_cache=False)
                pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            vqa_t += 1
            if pred == label:
                vqa_c += 1
            if vqa_t % 200 == 0:
                print(f"  [Baseline GPU{gpu_id}] VQA {vqa_t}: {vqa_c/vqa_t*100:.1f}%", flush=True)

        baseline = {
            "vsr_acc": vsr_c / max(vsr_t, 1) * 100,
            "vsr_correct": vsr_c, "vsr_total": vsr_t,
            "ctrl_acc": ctrl_c / max(ctrl_t, 1) * 100,
            "ctrl_correct": ctrl_c, "ctrl_total": ctrl_t,
            "vqa_acc": vqa_c / max(vqa_t, 1) * 100,
            "vqa_correct": vqa_c, "vqa_total": vqa_t,
        }
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"[Ablation GPU{gpu_id}] Baseline: VSR={baseline['vsr_acc']:.1f}%, "
              f"Ctrl={baseline['ctrl_acc']:.1f}%, VQA={baseline['vqa_acc']:.1f}%", flush=True)

    # --- Per-feature ablation ---
    for feat_i, (layer_idx, feature_idx) in enumerate(feature_assignments):
        result_path = out_dir / f"ablation_L{layer_idx}_F{feature_idx}.json"
        if result_path.exists():
            print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: already done, skip", flush=True)
            continue

        # Load SAE decoder direction
        ckpt_path = CHECKPOINT_DIR / f"pretrained_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()
        feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
        del sae
        torch.cuda.empty_cache()

        print(f"[Ablation GPU{gpu_id}] [{feat_i+1}/{len(feature_assignments)}] "
              f"L{layer_idx}/F{feature_idx}...", flush=True)

        # VSR ablation
        vsr_c = vsr_t = ctrl_c = ctrl_t = 0
        for vi in range(len(vsr)):
            ex = vsr[vi]
            img = _load_vsr_image(ex)
            if img is None:
                continue
            caption = str(ex.get("caption", "")).strip()
            label = int(ex.get("label", 0))
            relation = ex.get("relation", "")
            prompt = _build_vsr_prompt(caption)
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, nns_model._module, device=device)
                _, img_end = get_image_token_positions(input_ids)
                logits_saved = _do_ablation_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end)
                pred = _predict_yesno(logits_saved[0, -1, :], yes_ids, no_ids)
            except Exception as e:
                if vsr_t < 3:
                    print(f"  [ERROR] GPU{gpu_id} L{layer_idx}F{feature_idx} VSR {vi}: {e}", flush=True)
                pred = 0
            vsr_t += 1
            if pred == label:
                vsr_c += 1
            if relation in CONTROL_RELATIONS:
                ctrl_t += 1
                if pred == label:
                    ctrl_c += 1

        # VQA ablation
        vqa_c = vqa_t = 0
        for qi, label in vqa_yesno:
            ex = vqa[qi]
            img = ex.get("image")
            if img is None:
                continue
            from PIL import Image as PILImage
            if isinstance(img, PILImage.Image):
                img = img.convert("RGB")
            else:
                continue
            question = ex.get("question", "")
            prompt = (
                "Answer the following question with only 'Yes' or 'No':\n"
                f"Question: {question.strip()}\n"
                "Answer:"
            )
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, nns_model._module, device=device)
                _, img_end = get_image_token_positions(input_ids)
                logits_saved = _do_ablation_trace(input_ids, attn_mask, pixel_values, feature_vec, img_end)
                pred = _predict_yesno(logits_saved[0, -1, :], yes_ids, no_ids)
            except Exception as e:
                if vqa_t < 3:
                    print(f"  [ERROR] GPU{gpu_id} L{layer_idx}F{feature_idx} VQA {qi}: {e}", flush=True)
                pred = 0
            vqa_t += 1
            if pred == label:
                vqa_c += 1

        vsr_acc = vsr_c / max(vsr_t, 1) * 100
        ctrl_acc = ctrl_c / max(ctrl_t, 1) * 100
        vqa_acc = vqa_c / max(vqa_t, 1) * 100

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "vsr_acc": vsr_acc, "vsr_correct": vsr_c, "vsr_total": vsr_t,
            "ctrl_acc": ctrl_acc, "ctrl_correct": ctrl_c, "ctrl_total": ctrl_t,
            "vqa_acc": vqa_acc, "vqa_correct": vqa_c, "vqa_total": vqa_t,
            "delta_vsr": vsr_acc - baseline["vsr_acc"],
            "delta_ctrl": ctrl_acc - baseline["ctrl_acc"],
            "delta_vqa": vqa_acc - baseline["vqa_acc"],
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: "
              f"∆VSR={result['delta_vsr']:+.1f}%, ∆Ctrl={result['delta_ctrl']:+.1f}%, "
              f"∆VQA={result['delta_vqa']:+.1f}%", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV with layer,feature columns")
    parser.add_argument("--gpus", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--vqa-max", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: ANALYSIS_DIR/ablation)")
    args = parser.parse_args()

    # Load features and sort by priority: (1 - cosine) * odds_ratio * Ev
    features_raw = []
    with open(args.features) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features_raw.append((int(row["layer"]), int(row["feature"])))
    print(f"Loaded {len(features_raw)} features from {args.features}")

    # Load per-feature metrics for priority sorting
    cosine_dir = ANALYSIS_DIR / "cosines"
    energy_dir = ANALYSIS_DIR / "energy"
    spatial_path = ANALYSIS_DIR / "spatial" / "spatial_features.csv"

    cosine_map, ev_map, or_map = {}, {}, {}
    for layer_idx in range(N_LAYERS):
        cos_path = cosine_dir / f"cosines_layer_{layer_idx}.npy"
        if cos_path.exists():
            cosines = np.load(cos_path)
            for fi in range(len(cosines)):
                cosine_map[(layer_idx, fi)] = float(cosines[fi])
        ev_path = energy_dir / f"Ev_layer_{layer_idx}.npy"
        if ev_path.exists():
            evs = np.load(ev_path)
            for fi in range(len(evs)):
                ev_map[(layer_idx, fi)] = float(evs[fi])

    if spatial_path.exists():
        with open(spatial_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                or_map[(int(row["layer"]), int(row["feature"]))] = float(row.get("odds_ratio", 1.0))

    scored = []
    for layer, feature in features_raw:
        key = (layer, feature)
        cos = cosine_map.get(key, 0.9)
        odds = or_map.get(key, 1.0)
        ev = ev_map.get(key, 0.01)
        score = (1.0 - cos) * odds * ev
        scored.append((layer, feature, score, cos, odds, ev))

    scored.sort(key=lambda x: -x[2])  # Highest priority first
    features = [(l, f) for l, f, *_ in scored]

    print(f"Priority-sorted {len(features)} features by (1-cosine)*OR*Ev:")
    for i, (l, f, s, c, o, e) in enumerate(scored[:10]):
        print(f"  {i+1}. L{l}/F{f}: score={s:.4f} (cos={c:.3f}, OR={o:.1f}, Ev={e:.3f})")
    if len(scored) > 10:
        print(f"  ... ({len(scored)-10} more)")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / "ablation"

    # Distribute features across GPUs
    n_gpus = len(args.gpus)
    per_gpu = math.ceil(len(features) / n_gpus)
    assignments = []
    for i, gpu_id in enumerate(args.gpus):
        start = i * per_gpu
        end = min(start + per_gpu, len(features))
        worker_feats = features[start:end]
        if worker_feats:
            assignments.append((gpu_id, worker_feats))
            print(f"  GPU {gpu_id}: {len(worker_feats)} features")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for gpu_id, feats in assignments:
        p = mp.Process(target=_ablation_worker,
                       args=(gpu_id, feats, str(out_dir), args.vqa_max))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Collect all results into summary CSV
    all_results = []
    for p in sorted(Path(out_dir).glob("ablation_L*_F*.json")):
        with open(p) as f:
            all_results.append(json.load(f))

    if all_results:
        all_results.sort(key=lambda r: r.get("delta_vsr", 0))
        summary_path = out_dir / "ablation_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "layer", "feature", "delta_vsr", "delta_ctrl", "delta_vqa",
                "vsr_acc", "ctrl_acc", "vqa_acc",
                "vsr_total", "ctrl_total", "vqa_total",
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

        print(f"\n{'='*60}")
        print(f"Ablation Summary ({len(all_results)} features)")
        print(f"{'='*60}")
        # Load any baseline for reference
        baselines = list(Path(out_dir).glob("baseline_gpu*.json"))
        if baselines:
            with open(baselines[0]) as f:
                bl = json.load(f)
            print(f"Baseline: VSR={bl['vsr_acc']:.1f}%, Ctrl={bl['ctrl_acc']:.1f}%, VQA={bl['vqa_acc']:.1f}%")
        print(f"\n{'Layer':>5} {'Feat':>6} {'∆VSR':>7} {'∆Ctrl':>7} {'∆VQA':>7}")
        for r in all_results:
            print(f"  L{r['layer']:>2}  F{r['feature']:>5}  {r['delta_vsr']:+6.1f}%  "
                  f"{r['delta_ctrl']:+6.1f}%  {r['delta_vqa']:+6.1f}%")
        print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
