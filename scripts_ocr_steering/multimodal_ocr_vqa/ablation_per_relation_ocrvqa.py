#!/usr/bin/env python3
"""
OCR-VQA ablation for PaliGemma2 text-only SAE features.

Mirrors the VSR per-relation ablation, but for the free-form OCR-VQA task:
  - 10K random rows from the OCR-VQA test split (one (Q, A) pair per row, fixed by indices file)
  - 3-point projection ablation (attn_out, mlp_out, layer_out) on text tokens, all 26 layers,
    via forward hooks (works cleanly with model.generate())
  - Score: substring match against the ground-truth answer (case-insensitive, either direction)
  - Capability controls (in addition to ∆OCR-VQA):
      ∆Ctrl   — VQA-clean (no-OCR) yes/no, 500 samples (analog of VSR CONTROL_RELATIONS)
      ∆VQA    — raw VQAv2 yes/no, 1000 samples

Usage:
  python3 ablation_per_relation_ocrvqa.py \
      --features /data1/.../analysis_ocrvqa/final_features/final_ocrvqa_features.csv \
      --gpus 0 1 2 3 4 5 6 7
"""
import os, sys, json, math, csv, argparse, warnings, gc
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
CHECKPOINT_DIR = Path(os.environ.get("VLMSCOPE_CKPT_DIR",
                       "/data1/vlm_scope_sae_mix448_textonly/checkpoints"))
ANALYSIS_DIR = Path(os.environ.get("VLMSCOPE_ANALYSIS_DIR",
                     "/data1/vlm_scope_sae_mix448_textonly/analysis_ocrvqa"))
HF_CACHE = "/data1/hf_cache/hub"
OCRVQA_TEST_INDICES = ANALYSIS_DIR / "ocrvqa_indices_test.json"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

OCRVQA_DATASET = "howard-hou/OCR-VQA"
VQA_MAX = 1000
MAX_NEW_TOKENS = 32


def _ocrvqa_correct(response, gt) -> bool:
    """OCR-VQA scoring: case-insensitive substring match (either direction).
    `gt` may be a single string (one answer per (row, q_idx) sample)."""
    if response is None: return False
    if isinstance(gt, list): gt_list = gt
    else: gt_list = [gt]
    resp = response.strip().lower()
    if not resp: return False
    for g in gt_list:
        gl = str(g).strip().lower()
        if not gl: continue
        if gl in resp or resp in gl:
            return True
    return False


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES", " true", "true", "True"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO", " false", "false", "False"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap; no_ids -= overlap
    return yes_ids, no_ids


def _predict_yesno(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


class ProjectionAblator:
    """Forward-hook 3-point projection: removes feature_vec direction from
    (attn_out, mlp_out, layer_out) on text-token positions for all N_LAYERS.
    Works with model.generate() (handles single-token decode passes too).
    """

    def __init__(self, model, feature_vec, n_layers=N_LAYERS):
        self.model = model
        self.fv = feature_vec.view(1, -1)
        self.n_layers = n_layers
        self.img_end = 0
        self.handles = []

    def set_img_end(self, ie):
        self.img_end = int(ie)

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

    def _hook_attn(self):
        def f(m, i, o):
            if isinstance(o, tuple):
                o = list(o); o[0] = self._proj(o[0]); return tuple(o)
            return self._proj(o)
        return f

    def _hook_mlp(self):
        def f(m, i, o):
            return self._proj(o)
        return f

    def _hook_layer(self):
        def f(m, i, o):
            if isinstance(o, tuple):
                o = list(o); o[0] = self._proj(o[0]); return tuple(o)
            return self._proj(o)
        return f

    def install(self):
        layers = self.model.model.language_model.layers
        for l in range(self.n_layers):
            self.handles.append(layers[l].self_attn.register_forward_hook(self._hook_attn()))
            self.handles.append(layers[l].mlp.register_forward_hook(self._hook_mlp()))
            self.handles.append(layers[l].register_forward_hook(self._hook_layer()))
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

    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return
    print(f"[Ablation GPU{gpu_id}] {len(feature_assignments)} features", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE
    ).to(device).eval()
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model.parameters()).dtype

    # ---- OCR-VQA test slice ----
    print(f"[Ablation GPU{gpu_id}] Loading OCR-VQA test split + indices...", flush=True)
    ocrvqa_test = load_dataset(OCRVQA_DATASET, split="test")
    with open(OCRVQA_TEST_INDICES) as f:
        ocrvqa_idx = json.load(f)
    print(f"[Ablation GPU{gpu_id}] OCR-VQA test: {len(ocrvqa_idx)} samples", flush=True)

    # ---- VQA yes/no capability control ----
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno = []
    for i in range(len(vqa)):
        ex = vqa[i]
        at = str(ex.get("answer_type", "")).lower()
        mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
        if at == "yes/no" and mc in {"yes", "no"}:
            vqa_yesno.append((i, 1 if mc == "yes" else 0))
            if len(vqa_yesno) >= vqa_max: break
    print(f"[Ablation GPU{gpu_id}] VQA yes/no: {len(vqa_yesno)}", flush=True)

    # ---- VQA-clean (no-OCR) control set ----
    vqa_clean_path = ANALYSIS_DIR / "vqa_clean_yesno" / "indices.json"
    vqa_clean = []
    if vqa_clean_path.exists():
        clean_meta = json.load(open(vqa_clean_path))
        vqa_clean = [(d["vqa_index"], int(d["label"])) for d in clean_meta]
        print(f"[Ablation GPU{gpu_id}] VQA-clean control: {len(vqa_clean)} samples", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] WARN: no VQA-clean control at {vqa_clean_path}", flush=True)

    def _run_ocrvqa_eval(feature_vec):
        ablator = None
        if feature_vec is not None:
            ablator = ProjectionAblator(model, feature_vec).install()
        c = t = 0
        try:
            for meta in ocrvqa_idx:
                row = ocrvqa_test[meta["row_idx"]]
                img = row.get("image")
                question = meta["question"]
                gt = meta["answer"]
                if img is None or not question or not gt: continue
                try:
                    img = img.convert("RGB") if isinstance(img, PILImage.Image) else None
                    if img is None: continue
                    prompt = f"answer en {question}"
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                    _, img_end = get_image_token_positions(iids)
                    if ablator is not None: ablator.set_img_end(img_end)
                    with torch.inference_mode():
                        out = model.generate(input_ids=iids, attention_mask=attn,
                                              pixel_values=pv,
                                              max_new_tokens=MAX_NEW_TOKENS,
                                              do_sample=False, use_cache=True)
                    gen = tokenizer.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
                    ok = _ocrvqa_correct(gen, gt)
                except Exception:
                    ok = False
                t += 1
                if ok: c += 1
        finally:
            if ablator is not None: ablator.remove()
        return {"acc": c / max(t, 1) * 100, "correct": c, "total": t}

    def _run_yesno_eval(feature_vec, pairs):
        ablator = None
        if feature_vec is not None:
            ablator = ProjectionAblator(model, feature_vec).install()
        c = t = 0
        try:
            for qi, label in pairs:
                ex = vqa[qi]
                img = ex.get("image")
                if img is None or not isinstance(img, PILImage.Image): continue
                img = img.convert("RGB")
                question = ex.get("question", "")
                prompt = ("Answer the following question with only 'Yes' or 'No':\n"
                          f"Question: {question.strip()}\nAnswer:")
                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                    _, img_end = get_image_token_positions(iids)
                    if ablator is not None: ablator.set_img_end(img_end)
                    with torch.inference_mode():
                        out = model(input_ids=iids, attention_mask=attn,
                                    pixel_values=pv, use_cache=False)
                    pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
                except Exception:
                    pred = 0
                t += 1
                if pred == label: c += 1
        finally:
            if ablator is not None: ablator.remove()
        return {"acc": c / max(t, 1) * 100, "correct": c, "total": t}

    def _run_vqa_eval(fv):
        r = _run_yesno_eval(fv, vqa_yesno)
        return {"vqa_acc": r["acc"], "vqa_correct": r["correct"], "vqa_total": r["total"]}

    def _run_ctrl_eval(fv):
        if not vqa_clean:
            return {"ctrl_acc": 0.0, "ctrl_correct": 0, "ctrl_total": 0}
        r = _run_yesno_eval(fv, vqa_clean)
        return {"ctrl_acc": r["acc"], "ctrl_correct": r["correct"], "ctrl_total": r["total"]}

    # --- Baselines (cached per GPU) ---
    ocr_baseline_path = out_dir / f"ocrvqa_baseline_gpu{gpu_id}.json"
    if ocr_baseline_path.exists():
        ocr_baseline = json.load(open(ocr_baseline_path))
        print(f"[Ablation GPU{gpu_id}] cached OCR-VQA baseline: {ocr_baseline['acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] computing OCR-VQA baseline...", flush=True)
        ocr_baseline = _run_ocrvqa_eval(None)
        with open(ocr_baseline_path, "w") as f: json.dump(ocr_baseline, f, indent=2)
        print(f"[Ablation GPU{gpu_id}] OCR-VQA baseline: {ocr_baseline['acc']:.2f}%", flush=True)

    vqa_baseline_path = out_dir / f"vqa_baseline_gpu{gpu_id}.json"
    if vqa_baseline_path.exists():
        vqa_baseline = json.load(open(vqa_baseline_path))
        print(f"[Ablation GPU{gpu_id}] cached VQA baseline: {vqa_baseline['vqa_acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] computing VQA baseline...", flush=True)
        vqa_baseline = _run_vqa_eval(None)
        with open(vqa_baseline_path, "w") as f: json.dump(vqa_baseline, f, indent=2)

    ctrl_baseline_path = out_dir / f"ctrl_baseline_gpu{gpu_id}.json"
    if ctrl_baseline_path.exists():
        ctrl_baseline = json.load(open(ctrl_baseline_path))
        print(f"[Ablation GPU{gpu_id}] cached CTRL baseline: {ctrl_baseline['ctrl_acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] computing CTRL baseline...", flush=True)
        ctrl_baseline = _run_ctrl_eval(None)
        with open(ctrl_baseline_path, "w") as f: json.dump(ctrl_baseline, f, indent=2)

    # --- Per-feature ablation ---
    for feat_i, (layer_idx, feature_idx, odds_ratio) in enumerate(feature_assignments):
        result_path = out_dir / f"ablation_L{layer_idx}_F{feature_idx}.json"
        if result_path.exists():
            print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: cached", flush=True)
            continue

        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()
        feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        print(f"[Ablation GPU{gpu_id}] [{feat_i+1}/{len(feature_assignments)}] "
              f"L{layer_idx}/F{feature_idx}...", flush=True)

        ablated_ocr = _run_ocrvqa_eval(feature_vec)
        ablated_vqa = _run_vqa_eval(feature_vec)
        ablated_ctrl = _run_ctrl_eval(feature_vec)

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "odds_ratio": odds_ratio,
            "baseline_ocrvqa_acc": ocr_baseline["acc"],
            "ablated_ocrvqa_acc": ablated_ocr["acc"],
            "delta_ocrvqa": ablated_ocr["acc"] - ocr_baseline["acc"],
            "ocrvqa_total": ocr_baseline["total"],
            "baseline_vqa_acc": vqa_baseline["vqa_acc"],
            "ablated_vqa_acc": ablated_vqa["vqa_acc"],
            "delta_vqa": ablated_vqa["vqa_acc"] - vqa_baseline["vqa_acc"],
            "baseline_ctrl_acc": ctrl_baseline["ctrl_acc"],
            "ablated_ctrl_acc": ablated_ctrl["ctrl_acc"],
            "delta_ctrl": ablated_ctrl["ctrl_acc"] - ctrl_baseline["ctrl_acc"],
            "ctrl_total": ctrl_baseline["ctrl_total"],
        }
        with open(result_path, "w") as f: json.dump(result, f, indent=2)

        print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: "
              f"∆OCRVQA={result['delta_ocrvqa']:+.2f}%, "
              f"∆Ctrl={result['delta_ctrl']:+.2f}%, "
              f"∆VQA={result['delta_vqa']:+.2f}%, OR={odds_ratio:.1f}", flush=True)

        torch.cuda.empty_cache(); gc.collect()

    print(f"[Ablation GPU{gpu_id}] Done.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--vqa-max", type=int, default=VQA_MAX)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    features_raw = []
    with open(args.features) as f:
        for r in csv.DictReader(f):
            features_raw.append((int(r["layer"]), int(r["feature"])))
    print(f"Loaded {len(features_raw)} features")

    # Pull odds ratios from Step 6 output
    or_map = {}
    fp = ANALYSIS_DIR / "ocrvqa_pertoken" / "ocrvqa_features_pertoken.csv"
    if fp.exists():
        for r in csv.DictReader(open(fp)):
            or_map[(int(r["layer"]), int(r["feature"]))] = float(r.get("odds_ratio", 1.0))

    cosine_map, ev_map = {}, {}
    for layer_idx in range(N_LAYERS):
        cp = ANALYSIS_DIR / "cosines" / f"cosines_layer_{layer_idx}.npy"
        if cp.exists():
            cs = np.load(cp)
            for fi in range(len(cs)): cosine_map[(layer_idx, fi)] = float(cs[fi])
        ep = ANALYSIS_DIR / "energy" / f"Ev_layer_{layer_idx}.npy"
        if not ep.exists(): ep = ANALYSIS_DIR / "energy" / f"Et_layer_{layer_idx}.npy"
        if ep.exists():
            es = np.load(ep)
            for fi in range(len(es)): ev_map[(layer_idx, fi)] = float(es[fi])

    scored = []
    for layer, feature in features_raw:
        k = (layer, feature)
        cos = cosine_map.get(k, 0.9)
        odds = or_map.get(k, 1.0)
        ev = ev_map.get(k, 0.01)
        score = (1.0 - cos) * odds * ev
        scored.append((layer, feature, odds, score))
    scored.sort(key=lambda x: -x[2])  # sort by OR descending
    features = [(l, f, o) for l, f, o, _ in scored]

    print(f"\nTop 10 by OR:")
    for i, (l, f, o, s) in enumerate(scored[:10]):
        print(f"  {i+1}. L{l}/F{f}: OR={o:.1f}, score={s:.4f}")

    out_dir = Path(args.out_dir) if args.out_dir else ANALYSIS_DIR / "ablation_ocrvqa"

    n_gpus = len(args.gpus)
    per_gpu = math.ceil(len(features) / n_gpus)
    assignments = []
    for i, gpu_id in enumerate(args.gpus):
        start = i * per_gpu
        end = min(start + per_gpu, len(features))
        wf = features[start:end]
        if wf:
            assignments.append((gpu_id, wf))
            print(f"  GPU {gpu_id}: {len(wf)} features")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    procs = []
    for gpu_id, feats in assignments:
        p = mp.Process(target=_ablation_worker,
                       args=(gpu_id, feats, str(out_dir), args.vqa_max))
        p.start()
        procs.append(p)
    for p in procs: p.join()

    # Collect summary
    all_results = []
    for p in sorted(Path(out_dir).glob("ablation_L*_F*.json")):
        all_results.append(json.load(open(p)))

    if all_results:
        all_results.sort(key=lambda r: r.get("delta_ocrvqa", 0))
        summary_path = out_dir / "ablation_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "layer", "feature", "odds_ratio",
                "delta_ocrvqa", "delta_ctrl", "delta_vqa",
                "baseline_ocrvqa_acc", "ablated_ocrvqa_acc", "ocrvqa_total",
                "baseline_ctrl_acc", "ablated_ctrl_acc", "ctrl_total",
                "baseline_vqa_acc", "ablated_vqa_acc",
            ])
            w.writeheader()
            for r in all_results:
                w.writerow({
                    "layer": r["layer"], "feature": r["feature"],
                    "odds_ratio": r.get("odds_ratio", ""),
                    "delta_ocrvqa": f"{r.get('delta_ocrvqa', 0):.2f}",
                    "delta_ctrl": f"{r.get('delta_ctrl', 0):.2f}",
                    "delta_vqa": f"{r.get('delta_vqa', 0):.2f}",
                    "baseline_ocrvqa_acc": f"{r.get('baseline_ocrvqa_acc', 0):.2f}",
                    "ablated_ocrvqa_acc": f"{r.get('ablated_ocrvqa_acc', 0):.2f}",
                    "ocrvqa_total": r.get("ocrvqa_total", 0),
                    "baseline_ctrl_acc": f"{r.get('baseline_ctrl_acc', 0):.2f}",
                    "ablated_ctrl_acc": f"{r.get('ablated_ctrl_acc', 0):.2f}",
                    "ctrl_total": r.get("ctrl_total", 0),
                    "baseline_vqa_acc": f"{r.get('baseline_vqa_acc', 0):.2f}",
                    "ablated_vqa_acc": f"{r.get('ablated_vqa_acc', 0):.2f}",
                })
        print(f"\n{'='*80}\nOCR-VQA Ablation Summary ({len(all_results)} features)\n{'='*80}")
        print(f"\n{'Layer':>5} {'Feat':>6} {'∆OCRVQA':>9} {'∆Ctrl':>8} {'∆VQA':>8} {'OR':>6}")
        for r in all_results[:30]:
            print(f"  L{r['layer']:>2}  F{r['feature']:>5}  "
                  f"{r['delta_ocrvqa']:+8.2f}%  {r.get('delta_ctrl',0):+7.2f}%  "
                  f"{r['delta_vqa']:+7.2f}%  {r.get('odds_ratio',0):5.1f}")
        print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
