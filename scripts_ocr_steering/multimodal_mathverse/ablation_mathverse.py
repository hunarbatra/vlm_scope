#!/usr/bin/env python3
"""
MathVerse causal ablation for text-only SAE features.

Mirrors ablation_per_relation_ocr.py, adapted for MathVerse MCQ evaluation.
- Dataset: hunarbatra/MathVerse_Vision_MCQ (testmini, 430 samples)
- Task: MCQ 4-choice (A/B/C/D) — scored by argmax logit over choice tokens
- Ablation: 3-point projection (attn_out, mlp_out, layer_out) on text tokens, all 26 layers
- Capability control: VQA yes/no (1000 samples)
- Input: final_math_features.csv (from step 8 intersection)

Usage:
  python3 ablation_mathverse.py \
      --features /data1/vlm_scope_sae_mix448_textonly/analysis_mathverse/final_features/final_math_features.csv \
      --gpus 0 1 2 3 4 5 6 7
"""

import os
import sys
import json
import math
import csv
import argparse
import warnings
import gc
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
ANALYSIS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_mathverse")
OCR_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr")
HF_CACHE = "/data1/hf_cache/hub"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

VQA_MAX = 1000


def _get_choice_ids(tokenizer):
    choice_ids = {}
    for letter in "ABCD":
        ids_ = set()
        for form in [letter, f" {letter}", f"({letter})", f" ({letter})"]:
            try:
                t = tokenizer.encode(form, add_special_tokens=False)
                if t:
                    ids_.add(t[0])
            except Exception:
                pass
        choice_ids[letter] = ids_
    return choice_ids


def _predict_mcq(logits, choice_ids):
    p = torch.softmax(logits.float(), dim=-1)
    scores = {l: sum(p[i].item() for i in ids_) for l, ids_ in choice_ids.items()}
    return max(scores, key=scores.get)


def _parse_gt(s):
    m = re.search(r'([A-D])', str(s))
    return m.group(1) if m else None


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


class ProjectionAblator:
    """3-point projection: project out feature_vec from attn_out + mlp_out + layer_out on text tokens."""

    def __init__(self, model, feature_vec, n_layers=N_LAYERS):
        self.model = model
        self.fv = feature_vec.view(1, -1)
        self.n_layers = n_layers
        self.img_end = 0
        self.handles = []

    def set_img_end(self, img_end):
        self.img_end = int(img_end)

    def _proj(self, x):
        if x.shape[1] > 1:
            start = min(self.img_end, x.shape[1])
            sub = x[:, start:, :]
            coef = sub @ self.fv.T
            x[:, start:, :] = sub - coef * self.fv
        else:
            coef = x @ self.fv.T
            x.sub_(coef * self.fv)
        return x

    def _attn_hook(self):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = list(out)
                out[0] = self._proj(out[0])
                return tuple(out)
            return self._proj(out)
        return hook

    def _mlp_hook(self):
        def hook(module, inp, out):
            return self._proj(out)
        return hook

    def _layer_hook(self):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = list(out)
                out[0] = self._proj(out[0])
                return tuple(out)
            return self._proj(out)
        return hook

    def install(self):
        layers = self.model.model.language_model.layers
        for l in range(self.n_layers):
            self.handles.append(layers[l].self_attn.register_forward_hook(self._attn_hook()))
            self.handles.append(layers[l].mlp.register_forward_hook(self._mlp_hook()))
            self.handles.append(layers[l].register_forward_hook(self._layer_hook()))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _ablation_worker(gpu_id, feature_assignments, out_dir, vqa_max):
    from datasets import load_dataset

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return

    print(f"[Ablation GPU{gpu_id}] {len(feature_assignments)} features", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = processor.tokenizer
    choice_ids = _get_choice_ids(tokenizer)
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model.parameters()).dtype

    print(f"[Ablation GPU{gpu_id}] Loading MathVerse testmini...", flush=True)
    math_ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")
    N = len(math_ds)
    print(f"[Ablation GPU{gpu_id}] MathVerse: {N} samples", flush=True)

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

    def _run_math_eval(feature_vec):
        ablator = None
        if feature_vec is not None:
            ablator = ProjectionAblator(model, feature_vec).install()
        c = t = 0
        try:
            for si in range(N):
                ex = math_ds[si]
                img = ex.get("image")
                if img is None:
                    continue
                gt = _parse_gt(ex.get("answer", ""))
                if gt is None:
                    continue
                try:
                    img = img.convert("RGB")
                    iids, attn, pv = process_vlm_inputs(
                        img, f"answer en {ex['prompt']}", processor, model, device=device)
                    _, img_end = get_image_token_positions(iids)
                    if ablator is not None:
                        ablator.set_img_end(img_end)
                    with torch.no_grad():
                        out = model(input_ids=iids, attention_mask=attn,
                                    pixel_values=pv, use_cache=False)
                    pred = _predict_mcq(out.logits[0, -1, :], choice_ids)
                    t += 1
                    if pred == gt:
                        c += 1
                except Exception:
                    pass
        finally:
            if ablator is not None:
                ablator.remove()
        return {"math_acc": c / max(t, 1) * 100, "math_correct": c, "math_total": t}

    def _run_vqa_eval(feature_vec):
        ablator = None
        if feature_vec is not None:
            ablator = ProjectionAblator(model, feature_vec).install()
        c = t = 0
        try:
            for qi, label in vqa_yesno:
                ex = vqa[qi]
                img = ex.get("image")
                if img is None:
                    continue
                img = img.convert("RGB")
                question = ex.get("question", "")
                prompt = (
                    "Answer the following question with only 'Yes' or 'No':\n"
                    f"Question: {question.strip()}\nAnswer:"
                )
                try:
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        img, prompt, processor, model, device=device)
                    _, img_end = get_image_token_positions(input_ids)
                    if ablator is not None:
                        ablator.set_img_end(img_end)
                    with torch.inference_mode():
                        out = model(input_ids=input_ids, attention_mask=attn_mask,
                                    pixel_values=pixel_values, use_cache=False)
                    pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
                except Exception:
                    pred = 0
                t += 1
                if pred == label:
                    c += 1
        finally:
            if ablator is not None:
                ablator.remove()
        return {"vqa_acc": c / max(t, 1) * 100, "vqa_correct": c, "vqa_total": t}

    # Baselines
    math_base_path = out_dir / f"math_baseline_gpu{gpu_id}.json"
    if math_base_path.exists():
        math_baseline = json.load(open(math_base_path))
        print(f"[Ablation GPU{gpu_id}] Loaded Math baseline: {math_baseline['math_acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing Math baseline...", flush=True)
        math_baseline = _run_math_eval(None)
        with open(math_base_path, "w") as f:
            json.dump(math_baseline, f, indent=2)
        print(f"[Ablation GPU{gpu_id}] Math baseline: {math_baseline['math_acc']:.2f}% "
              f"(n={math_baseline['math_total']})", flush=True)

    vqa_base_path = out_dir / f"vqa_baseline_gpu{gpu_id}.json"
    if vqa_base_path.exists():
        vqa_baseline = json.load(open(vqa_base_path))
        print(f"[Ablation GPU{gpu_id}] Loaded VQA baseline: {vqa_baseline['vqa_acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing VQA baseline...", flush=True)
        vqa_baseline = _run_vqa_eval(None)
        with open(vqa_base_path, "w") as f:
            json.dump(vqa_baseline, f, indent=2)
        print(f"[Ablation GPU{gpu_id}] VQA baseline: {vqa_baseline['vqa_acc']:.2f}%", flush=True)

    # Per-feature ablation
    for feat_i, (layer_idx, feature_idx, odds_ratio) in enumerate(feature_assignments):
        result_path = out_dir / f"ablation_L{layer_idx}_F{feature_idx}.json"
        if result_path.exists():
            print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: cached, skip", flush=True)
            continue

        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
        del sae
        torch.cuda.empty_cache()

        print(f"[Ablation GPU{gpu_id}] [{feat_i+1}/{len(feature_assignments)}] "
              f"L{layer_idx}/F{feature_idx}...", flush=True)

        ablated_math = _run_math_eval(feature_vec)
        ablated_vqa = _run_vqa_eval(feature_vec)

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "odds_ratio": odds_ratio,
            "baseline_math_acc": math_baseline["math_acc"],
            "ablated_math_acc": ablated_math["math_acc"],
            "delta_math": ablated_math["math_acc"] - math_baseline["math_acc"],
            "math_total": math_baseline["math_total"],
            "baseline_vqa_acc": vqa_baseline["vqa_acc"],
            "ablated_vqa_acc": ablated_vqa["vqa_acc"],
            "delta_vqa": ablated_vqa["vqa_acc"] - vqa_baseline["vqa_acc"],
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: "
              f"∆Math={result['delta_math']:+.2f}%, "
              f"∆VQA={result['delta_vqa']:+.2f}%, OR={odds_ratio:.1f}", flush=True)

        del feature_vec
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[Ablation GPU{gpu_id}] Done.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=None)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--vqa-max", type=int, default=VQA_MAX)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--summarize", action="store_true",
                        help="Print summary of completed ablation results and exit")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / "ablation_mathverse"

    if args.summarize:
        _print_summary(out_dir)
        return

    if args.features is None:
        print("ERROR: --features required when not using --summarize"); sys.exit(1)

    features_raw = []
    with open(args.features) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features_raw.append((int(row["layer"]), int(row["feature"])))
    print(f"Loaded {len(features_raw)} features from {args.features}")

    # Load odds ratios from math_features.csv
    or_map = {}
    math_feat_path = ANALYSIS_DIR / "math_features" / "math_features.csv"
    if math_feat_path.exists():
        with open(math_feat_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                or_map[(int(row["layer"]), int(row["feature"]))] = float(row.get("odds_ratio", 1.0))

    # Score features: (1-cosine) * OR * Ev  (same priority as OCR pipeline)
    cosine_dir = OCR_DIR / "cosines"
    energy_dir = OCR_DIR / "energy"
    cosine_map, ev_map = {}, {}
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

    scored = []
    for layer, feature in features_raw:
        key = (layer, feature)
        cos = cosine_map.get(key, 0.9)
        odds = or_map.get(key, 1.0)
        ev = ev_map.get(key, 0.01)
        score = (1.0 - cos) * odds * ev
        scored.append((layer, feature, odds, score))
    scored.sort(key=lambda x: -x[3])
    features = [(l, f, o) for l, f, o, _ in scored]

    print(f"\nTop 10 by priority score:")
    for i, (l, f, o, s) in enumerate(scored[:10]):
        print(f"  {i+1}. L{l}/F{f}: score={s:.4f}, OR={o:.1f}")

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

    _print_summary(out_dir)


def _print_summary(out_dir):
    out_dir = Path(out_dir)
    all_results = []
    for p in sorted(out_dir.glob("ablation_L*_F*.json")):
        with open(p) as f:
            all_results.append(json.load(f))

    if not all_results:
        print("No ablation results found.")
        return

    all_results.sort(key=lambda x: x["delta_math"])

    print(f"\n{'='*80}")
    print(f"MATHVERSE ABLATION SUMMARY  ({len(all_results)} features)")
    print(f"{'='*80}")
    print(f"  {'Key':<16} {'∆Math':>8}  {'∆VQA':>7}  {'OR':>6}")
    for r in all_results[:15]:
        key = f"L{r['layer']}_F{r['feature']}"
        print(f"  {key:<16} {r['delta_math']:+7.2f}%  {r['delta_vqa']:+6.2f}%  {r['odds_ratio']:.1f}")

    print(f"\nTop 5 steering targets (biggest ablation drop on MathVerse):")
    for r in all_results[:5]:
        print(f"  L{r['layer']}_F{r['feature']}: ∆Math={r['delta_math']:+.2f}%, "
              f"baseline={r['baseline_math_acc']:.2f}%, OR={r['odds_ratio']:.1f}")

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved → {out_dir}/ablation_summary.json")


if __name__ == "__main__":
    main()
