#!/usr/bin/env python3
"""
OCR-Bench ablation for PaliGemma2 text-only SAE features.

Mirrors the VSR per-relation ablation, but:
  - Dataset: echo840/OCRBench (full test split)  — no per-relation subsets, no correctness filter
  - Task: free-form generation (not yes/no logits). Scored by OCR-Bench substring match
    against the ground-truth answer list.
  - Ablation: 3-point projection (attn_out, mlp_out, layer_out) on text tokens, all 26 layers.
    Uses forward hooks so it composes cleanly with model.generate().
  - Capability control: VQA yes/no (1000 samples, same as spatial pipeline).

Per feature, reports:
  - ∆OCR (accuracy delta on OCR-Bench)
  - Per-subtask breakdown (regular_text, irregular_text, artistic_text, handwriting,
    digit_string, non-semantic, scene_text, document_text, key_information_extraction,
    handwritten_mathematical_expression — whatever `dataset` field OCR-Bench uses)
  - ∆VQA (capability control)

Usage:
  python3 ablation_per_relation_ocr.py \
      --features /data1/vlm_scope_sae_mix448_textonly/analysis_ocr/final_features/final_ocr_features.csv \
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
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# ---- Config ----
MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
ANALYSIS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr")
HF_CACHE = "/data1/hf_cache/hub"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

OCR_DATASET = "echo840/OCRBench"
VQA_MAX = 1000
MAX_NEW_TOKENS = 64


# ---- OCR-Bench scoring (official substring-match) ----
def _ocr_correct(response: str, gt_list) -> bool:
    """OCR-Bench official metric: substring match (case-insensitive), either direction.
    The leaderboard accepts pred that contains gt OR gt that contains pred — we mirror that.
    """
    if response is None:
        return False
    if isinstance(gt_list, str):
        gt_list = [gt_list]
    resp = response.strip().lower()
    if not resp:
        return False
    for gt in gt_list:
        gt_l = str(gt).strip().lower()
        if not gt_l:
            continue
        if gt_l in resp or resp in gt_l:
            return True
    return False


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


# ---- Hook-based 3-point projection ablation ----
class ProjectionAblator:
    """Attach forward hooks on every layer that project out `feature_vec` from
    (attn_out, mlp_out, layer_out) on text-token positions only.

    img_end_getter is a zero-arg callable returning the current sample's
    first-text-token index (set by the caller before forward/generate).
    """

    def __init__(self, model, feature_vec, n_layers=N_LAYERS):
        self.model = model
        self.fv = feature_vec.view(1, -1)  # (1, d)
        self.n_layers = n_layers
        self.img_end = 0
        self.handles = []

    def set_img_end(self, img_end):
        self.img_end = int(img_end)

    def _proj(self, x):
        # x: (B, T, d). Ablate only positions >= img_end on the *first* forward pass.
        # During generate(), later tokens are at position 0 (KV-cached). We ablate any
        # position 0 tokens too, since they are always text (the just-generated token).
        if x.shape[1] > 1:
            # Prompt pass — positions are absolute, ablate img_end onward.
            start = min(self.img_end, x.shape[1])
            sub = x[:, start:, :]
            coef = sub @ self.fv.T  # (B, t, 1)
            x[:, start:, :] = sub - coef * self.fv
        else:
            # Single-token decode step — always text.
            coef = x @ self.fv.T
            x.sub_(coef * self.fv)
        return x

    def _make_attn_hook(self):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = list(out)
                out[0] = self._proj(out[0])
                return tuple(out)
            return self._proj(out)
        return hook

    def _make_mlp_hook(self):
        def hook(module, inp, out):
            return self._proj(out)
        return hook

    def _make_layer_hook(self):
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
            self.handles.append(layers[l].self_attn.register_forward_hook(self._make_attn_hook()))
            self.handles.append(layers[l].mlp.register_forward_hook(self._make_mlp_hook()))
            self.handles.append(layers[l].register_forward_hook(self._make_layer_hook()))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _ablation_worker(gpu_id, feature_assignments, out_dir, vqa_max):
    from PIL import Image as PILImage
    from datasets import load_dataset

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent))
    # utils.py lives in the sibling text-only scripts dir (one level up)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return

    print(f"[Ablation GPU{gpu_id}] {len(feature_assignments)} features", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=False)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=False
    ).to(device).eval()
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model.parameters()).dtype

    # ---- Load OCR-Bench ----
    print(f"[Ablation GPU{gpu_id}] Loading OCR-Bench (test)...", flush=True)
    ocr = load_dataset(OCR_DATASET, split="test")
    print(f"[Ablation GPU{gpu_id}] OCR-Bench: {len(ocr)} samples", flush=True)

    # ---- Load VQA yes/no capability control ----
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

    # ---- Load VQA-clean (no-OCR) control set — non-OCR analog (VSR's CONTROL_RELATIONS equivalent) ----
    vqa_clean_path = ANALYSIS_DIR / "vqa_clean_yesno" / "indices.json"
    vqa_clean = []
    if vqa_clean_path.exists():
        clean_meta = json.load(open(vqa_clean_path))
        vqa_clean = [(d["vqa_index"], int(d["label"])) for d in clean_meta]
        print(f"[Ablation GPU{gpu_id}] VQA-clean control: {len(vqa_clean)} samples", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] WARN: no VQA-clean control at {vqa_clean_path}", flush=True)

    def _run_ocr_eval(feature_vec):
        """OCR-Bench eval. If feature_vec is None, runs baseline (no ablation)."""
        ablator = None
        if feature_vec is not None:
            ablator = ProjectionAblator(model, feature_vec).install()

        by_cat = defaultdict(lambda: {"c": 0, "t": 0})
        total_c = total_t = 0
        try:
            for si in range(len(ocr)):
                sample = ocr[si]
                question = str(sample.get("question", "")).strip()
                img = sample.get("image")
                if img is None or not question:
                    continue
                gt_list = sample.get("answer", [])
                if isinstance(gt_list, str):
                    gt_list = [gt_list]
                if not gt_list:
                    continue
                cat = str(sample.get("question_type", "unknown")).strip() or "unknown"

                try:
                    img = img.convert("RGB") if isinstance(img, PILImage.Image) else None
                    if img is None:
                        continue
                    prompt = f"answer en {question}"
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        img, prompt, processor, model, device=device)
                    _, img_end = get_image_token_positions(input_ids)
                    if ablator is not None:
                        ablator.set_img_end(img_end)

                    with torch.inference_mode():
                        out = model.generate(
                            input_ids=input_ids, attention_mask=attn_mask,
                            pixel_values=pixel_values,
                            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                            use_cache=True,
                        )
                    gen_ids = out[0, input_ids.shape[1]:]
                    resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    ok = _ocr_correct(resp, gt_list)
                except Exception:
                    ok = False

                total_t += 1
                by_cat[cat]["t"] += 1
                if ok:
                    total_c += 1
                    by_cat[cat]["c"] += 1
        finally:
            if ablator is not None:
                ablator.remove()

        return {
            "ocr_acc": total_c / max(total_t, 1) * 100,
            "ocr_correct": total_c, "ocr_total": total_t,
            "per_category": {k: {"acc": v["c"] / max(v["t"], 1) * 100,
                                  "correct": v["c"], "total": v["t"]}
                             for k, v in by_cat.items()},
        }

    def _run_yesno_eval(feature_vec, index_label_pairs):
        """Yes/no eval over (vqa_index, label) pairs via next-token logits."""
        ablator = None
        if feature_vec is not None:
            ablator = ProjectionAblator(model, feature_vec).install()

        c = t = 0
        try:
            for qi, label in index_label_pairs:
                ex = vqa[qi]
                img = ex.get("image")
                if img is None:
                    continue
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

        return {"acc": c / max(t, 1) * 100, "correct": c, "total": t}

    def _run_vqa_eval(feature_vec):
        """VQA yes/no capability control (raw VQAv2)."""
        r = _run_yesno_eval(feature_vec, vqa_yesno)
        return {"vqa_acc": r["acc"], "vqa_correct": r["correct"], "vqa_total": r["total"]}

    def _run_ctrl_eval(feature_vec):
        """VQA-clean (no-OCR) control — non-OCR analog (like VSR's CONTROL_RELATIONS)."""
        if not vqa_clean:
            return {"ctrl_acc": 0.0, "ctrl_correct": 0, "ctrl_total": 0}
        r = _run_yesno_eval(feature_vec, vqa_clean)
        return {"ctrl_acc": r["acc"], "ctrl_correct": r["correct"], "ctrl_total": r["total"]}

    # --- Baselines (shared across features) ---
    ocr_baseline_path = out_dir / f"ocr_baseline_gpu{gpu_id}.json"
    if ocr_baseline_path.exists():
        with open(ocr_baseline_path) as f:
            ocr_baseline = json.load(f)
        print(f"[Ablation GPU{gpu_id}] Loaded OCR baseline: {ocr_baseline['ocr_acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing OCR baseline...", flush=True)
        ocr_baseline = _run_ocr_eval(None)
        with open(ocr_baseline_path, "w") as f:
            json.dump(ocr_baseline, f, indent=2)
        print(f"[Ablation GPU{gpu_id}] OCR baseline: {ocr_baseline['ocr_acc']:.2f}%", flush=True)

    vqa_baseline_path = out_dir / f"vqa_baseline_gpu{gpu_id}.json"
    if vqa_baseline_path.exists():
        with open(vqa_baseline_path) as f:
            vqa_baseline = json.load(f)
        print(f"[Ablation GPU{gpu_id}] Loaded VQA baseline: {vqa_baseline['vqa_acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing VQA baseline...", flush=True)
        vqa_baseline = _run_vqa_eval(None)
        with open(vqa_baseline_path, "w") as f:
            json.dump(vqa_baseline, f, indent=2)
        print(f"[Ablation GPU{gpu_id}] VQA baseline: {vqa_baseline['vqa_acc']:.2f}%", flush=True)

    ctrl_baseline_path = out_dir / f"ctrl_baseline_gpu{gpu_id}.json"
    if ctrl_baseline_path.exists():
        with open(ctrl_baseline_path) as f:
            ctrl_baseline = json.load(f)
        print(f"[Ablation GPU{gpu_id}] Loaded CTRL baseline: {ctrl_baseline['ctrl_acc']:.2f}% "
              f"(n={ctrl_baseline['ctrl_total']})", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing CTRL (VQA-clean no-OCR) baseline...", flush=True)
        ctrl_baseline = _run_ctrl_eval(None)
        with open(ctrl_baseline_path, "w") as f:
            json.dump(ctrl_baseline, f, indent=2)
        print(f"[Ablation GPU{gpu_id}] CTRL baseline: {ctrl_baseline['ctrl_acc']:.2f}% "
              f"(n={ctrl_baseline['ctrl_total']})", flush=True)

    # --- Per-feature ablation ---
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

        ablated_ocr = _run_ocr_eval(feature_vec)
        ablated_vqa = _run_vqa_eval(feature_vec)
        ablated_ctrl = _run_ctrl_eval(feature_vec)

        # Per-category deltas
        per_cat_delta = {}
        base_cat = ocr_baseline.get("per_category", {})
        abl_cat = ablated_ocr.get("per_category", {})
        for k in base_cat:
            if k in abl_cat:
                per_cat_delta[k] = {
                    "baseline_acc": base_cat[k]["acc"],
                    "ablated_acc": abl_cat[k]["acc"],
                    "delta": abl_cat[k]["acc"] - base_cat[k]["acc"],
                    "total": base_cat[k]["total"],
                }

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "odds_ratio": odds_ratio,
            "baseline_ocr_acc": ocr_baseline["ocr_acc"],
            "ablated_ocr_acc": ablated_ocr["ocr_acc"],
            "delta_ocr": ablated_ocr["ocr_acc"] - ocr_baseline["ocr_acc"],
            "ocr_total": ocr_baseline["ocr_total"],
            "baseline_vqa_acc": vqa_baseline["vqa_acc"],
            "ablated_vqa_acc": ablated_vqa["vqa_acc"],
            "delta_vqa": ablated_vqa["vqa_acc"] - vqa_baseline["vqa_acc"],
            "baseline_ctrl_acc": ctrl_baseline["ctrl_acc"],
            "ablated_ctrl_acc": ablated_ctrl["ctrl_acc"],
            "delta_ctrl": ablated_ctrl["ctrl_acc"] - ctrl_baseline["ctrl_acc"],
            "ctrl_total": ctrl_baseline["ctrl_total"],
            "per_category": per_cat_delta,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: "
              f"∆OCR={result['delta_ocr']:+.2f}%, "
              f"∆Ctrl={result['delta_ctrl']:+.2f}%, "
              f"∆VQA={result['delta_vqa']:+.2f}%, OR={odds_ratio:.1f}", flush=True)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"[Ablation GPU{gpu_id}] Done.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV with layer,feature columns")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--vqa-max", type=int, default=VQA_MAX)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    features_raw = []
    with open(args.features) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features_raw.append((int(row["layer"]), int(row["feature"])))
    print(f"Loaded {len(features_raw)} features")

    # Odds ratios from step6 ocr_features.csv
    or_map = {}
    ocr_feat_path = ANALYSIS_DIR / "ocr" / "ocr_features.csv"
    if ocr_feat_path.exists():
        with open(ocr_feat_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                or_map[(int(row["layer"]), int(row["feature"]))] = float(row.get("odds_ratio", 1.0))

    cosine_dir = ANALYSIS_DIR / "cosines"
    energy_dir = ANALYSIS_DIR / "energy"
    cosine_map, ev_map = {}, {}
    for layer_idx in range(N_LAYERS):
        cos_path = cosine_dir / f"cosines_layer_{layer_idx}.npy"
        if cos_path.exists():
            cosines = np.load(cos_path)
            for fi in range(len(cosines)):
                cosine_map[(layer_idx, fi)] = float(cosines[fi])
        ev_path = energy_dir / f"Ev_layer_{layer_idx}.npy"
        if not ev_path.exists():
            ev_path = energy_dir / f"Et_layer_{layer_idx}.npy"
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

    print(f"\nTop 10 by priority:")
    for i, (l, f, o, s) in enumerate(scored[:10]):
        print(f"  {i+1}. L{l}/F{f}: score={s:.4f}, OR={o:.1f}")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / "ablation_ocr"

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

    # Collect summary
    all_results = []
    for p in sorted(Path(out_dir).glob("ablation_L*_F*.json")):
        with open(p) as f:
            all_results.append(json.load(f))

    if all_results:
        # Sort by most degrading (most negative ∆OCR)
        all_results.sort(key=lambda r: r.get("delta_ocr", 0))
        summary_path = out_dir / "ablation_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "layer", "feature", "odds_ratio",
                "delta_ocr", "delta_ctrl", "delta_vqa",
                "baseline_ocr_acc", "ablated_ocr_acc", "ocr_total",
                "baseline_ctrl_acc", "ablated_ctrl_acc", "ctrl_total",
                "baseline_vqa_acc", "ablated_vqa_acc",
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    "layer": r["layer"], "feature": r["feature"],
                    "odds_ratio": r.get("odds_ratio", ""),
                    "delta_ocr": f"{r.get('delta_ocr', 0):.2f}",
                    "delta_ctrl": f"{r.get('delta_ctrl', 0):.2f}",
                    "delta_vqa": f"{r.get('delta_vqa', 0):.2f}",
                    "baseline_ocr_acc": f"{r.get('baseline_ocr_acc', 0):.2f}",
                    "ablated_ocr_acc": f"{r.get('ablated_ocr_acc', 0):.2f}",
                    "ocr_total": r.get("ocr_total", 0),
                    "baseline_ctrl_acc": f"{r.get('baseline_ctrl_acc', 0):.2f}",
                    "ablated_ctrl_acc": f"{r.get('ablated_ctrl_acc', 0):.2f}",
                    "ctrl_total": r.get("ctrl_total", 0),
                    "baseline_vqa_acc": f"{r.get('baseline_vqa_acc', 0):.2f}",
                    "ablated_vqa_acc": f"{r.get('ablated_vqa_acc', 0):.2f}",
                })

        print(f"\n{'='*80}")
        print(f"OCR-Bench Ablation Summary ({len(all_results)} features)")
        print(f"{'='*80}")
        print(f"\n{'Layer':>5} {'Feat':>6} {'∆OCR':>8} {'∆Ctrl':>8} {'∆VQA':>8} {'OR':>6}")
        for r in all_results[:30]:
            print(f"  L{r['layer']:>2}  F{r['feature']:>5}  "
                  f"{r['delta_ocr']:+7.2f}%  {r.get('delta_ctrl', 0):+7.2f}%  "
                  f"{r['delta_vqa']:+7.2f}%  {r.get('odds_ratio', 0):5.1f}")
        print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
