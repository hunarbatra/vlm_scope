#!/usr/bin/env python3
"""
DocVQA ablation for PaliGemma2 text-only SAE features.

Mirrors ablation_per_relation_ocr.py but evaluates on DocVQA validation (5349 samples)
instead of OCR-Bench. Uses 3-point projection ablation on text tokens.

Usage:
  python3 ablation_per_feature_docvqa.py \
      --features /data1/vlm_scope_sae_mix448_textonly/analysis_docvqa/final_features/final_docvqa_features.csv \
      --gpus 6 7 2>&1 | tee /tmp/docvqa_ablation.log
"""
import os, sys, json, math, csv, argparse, warnings, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

MODEL_NAME     = "google/paligemma2-3b-mix-448"
N_LAYERS       = 26
D_SAE          = 16384
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
ANALYSIS_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_docvqa")
HF_CACHE       = "/data1/hf_cache/hub"
MAX_NEW_TOKENS = 64
VQA_CTRL_PATH  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/vqa_clean_yesno/indices.json")

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def _correct(response, gt_list):
    if response is None: return False
    if isinstance(gt_list, str): gt_list = [gt_list]
    resp = response.strip().lower()
    if not resp: return False
    for gt in gt_list:
        gt_l = str(gt).strip().lower()
        if not gt_l: continue
        if gt_l in resp or resp in gt_l: return True
    return False


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
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
    def __init__(self, model, feature_vec, n_layers=N_LAYERS):
        self.model    = model
        self.fv       = feature_vec.view(1, -1)
        self.n_layers = n_layers
        self.img_end  = 0
        self.handles  = []

    def set_img_end(self, img_end): self.img_end = int(img_end)

    def _proj(self, x):
        if x.shape[1] > 1:
            start = min(self.img_end, x.shape[1])
            sub   = x[:, start:, :]
            coef  = sub @ self.fv.T
            x[:, start:, :] = sub - coef * self.fv
        else:
            coef = x @ self.fv.T
            x.sub_(coef * self.fv)
        return x

    def _attn_hook(self):
        def h(mod, inp, out):
            if isinstance(out, tuple):
                lst = list(out); lst[0] = self._proj(lst[0]); return tuple(lst)
            return self._proj(out)
        return h

    def _mlp_hook(self):
        def h(mod, inp, out): return self._proj(out)
        return h

    def _layer_hook(self):
        def h(mod, inp, out):
            if isinstance(out, tuple):
                lst = list(out); lst[0] = self._proj(lst[0]); return tuple(lst)
            return self._proj(out)
        return h

    def install(self):
        layers = self.model.model.language_model.layers
        for l in range(self.n_layers):
            self.handles.append(layers[l].self_attn.register_forward_hook(self._attn_hook()))
            self.handles.append(layers[l].mlp.register_forward_hook(self._mlp_hook()))
            self.handles.append(layers[l].register_forward_hook(self._layer_hook()))
        return self

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def _ablation_worker(gpu_id, feature_assignments, out_dir_str):
    from PIL import Image as PILImage
    from datasets import load_dataset

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return

    print(f"[Ablation GPU{gpu_id}] {len(feature_assignments)} features", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model     = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model.parameters()).dtype

    print(f"[Ablation GPU{gpu_id}] Loading DocVQA validation...", flush=True)
    dvqa = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    print(f"[Ablation GPU{gpu_id}] DocVQA: {len(dvqa)} samples", flush=True)

    # VQA-ctrl
    vqa_ctrl = []
    if VQA_CTRL_PATH.exists():
        meta = json.load(open(VQA_CTRL_PATH))
        vqa_ctrl = [(d["vqa_index"], int(d["label"])) for d in meta]
        vqa_ds   = load_dataset("lmms-lab/VQAv2", split="validation")
        print(f"[Ablation GPU{gpu_id}] VQA-ctrl: {len(vqa_ctrl)}", flush=True)
    else:
        vqa_ds = None
        print(f"[Ablation GPU{gpu_id}] WARN: no VQA-ctrl at {VQA_CTRL_PATH}", flush=True)

    def _run_docvqa(feature_vec):
        ablator = ProjectionAblator(model, feature_vec).install() if feature_vec is not None else None
        by_type = defaultdict(lambda: {"c": 0, "t": 0})
        total_c = total_t = 0
        try:
            for si in range(len(dvqa)):
                ex       = dvqa[si]
                question = str(ex.get("question", "")).strip()
                img      = ex.get("image")
                gt_list  = ex.get("answers", [])
                qtypes   = ex.get("question_types", ["unknown"])
                if img is None or not question or not gt_list: continue
                try:
                    img    = img.convert("RGB")
                    prompt = f"answer en {question}"
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        img, prompt, processor, model, device=device)
                    _, img_end = get_image_token_positions(input_ids)
                    if ablator: ablator.set_img_end(img_end)
                    with torch.inference_mode():
                        out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                              pixel_values=pixel_values,
                                              max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                              use_cache=True)
                    resp = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
                    ok   = _correct(resp, gt_list)
                except Exception:
                    ok = False
                total_t += 1
                if ok: total_c += 1
                for qt in qtypes:
                    by_type[qt]["t"] += 1
                    if ok: by_type[qt]["c"] += 1
        finally:
            if ablator: ablator.remove()
        return {"acc": total_c / max(total_t, 1) * 100, "correct": total_c, "total": total_t,
                "by_type": {qt: {"acc": v["c"]/max(v["t"],1)*100, "correct": v["c"], "total": v["t"]}
                            for qt, v in by_type.items()}}

    def _run_ctrl(feature_vec):
        if not vqa_ctrl or vqa_ds is None:
            return {"acc": 0.0, "correct": 0, "total": 0}
        ablator = ProjectionAblator(model, feature_vec).install() if feature_vec is not None else None
        c = t = 0
        try:
            for qi, label in vqa_ctrl:
                ex  = vqa_ds[qi]
                img = ex.get("image")
                if img is None: continue
                try:
                    img    = img.convert("RGB") if isinstance(img, PILImage.Image) else None
                    if img is None: continue
                    prompt = f"answer en {ex['question']}"
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        img, prompt, processor, model, device=device)
                    _, img_end = get_image_token_positions(input_ids)
                    if ablator: ablator.set_img_end(img_end)
                    with torch.inference_mode():
                        out = model(input_ids=input_ids, attention_mask=attn_mask,
                                    pixel_values=pixel_values, use_cache=False)
                    pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
                except Exception:
                    pred = 0
                t += 1
                if pred == label: c += 1
        finally:
            if ablator: ablator.remove()
        return {"acc": c / max(t, 1) * 100, "correct": c, "total": t}

    # Baselines
    base_path = out_dir / f"docvqa_baseline_gpu{gpu_id}.json"
    if base_path.exists():
        baseline = json.load(open(base_path))
        print(f"[Ablation GPU{gpu_id}] Loaded baseline: {baseline['acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing DocVQA baseline...", flush=True)
        baseline = _run_docvqa(None)
        json.dump(baseline, open(base_path, "w"), indent=2)
        print(f"[Ablation GPU{gpu_id}] Baseline: {baseline['acc']:.2f}%", flush=True)

    ctrl_base_path = out_dir / f"ctrl_baseline_gpu{gpu_id}.json"
    if ctrl_base_path.exists():
        ctrl_baseline = json.load(open(ctrl_base_path))
        print(f"[Ablation GPU{gpu_id}] Loaded ctrl baseline: {ctrl_baseline['acc']:.2f}%", flush=True)
    else:
        print(f"[Ablation GPU{gpu_id}] Computing ctrl baseline...", flush=True)
        ctrl_baseline = _run_ctrl(None)
        json.dump(ctrl_baseline, open(ctrl_base_path, "w"), indent=2)
        print(f"[Ablation GPU{gpu_id}] Ctrl baseline: {ctrl_baseline['acc']:.2f}%", flush=True)

    # Per-feature ablation
    for feat_i, (layer_idx, feature_idx, odds_ratio) in enumerate(feature_assignments):
        result_path = out_dir / f"ablation_L{layer_idx}_F{feature_idx}.json"
        if result_path.exists():
            print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: cached", flush=True)
            continue

        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        print(f"[Ablation GPU{gpu_id}] [{feat_i+1}/{len(feature_assignments)}] "
              f"L{layer_idx}/F{feature_idx}...", flush=True)

        ablated     = _run_docvqa(fv)
        ctrl_ablated= _run_ctrl(fv)

        base_type = baseline.get("by_type", {})
        abl_type  = ablated.get("by_type", {})
        per_type_delta = {qt: {"baseline": base_type[qt]["acc"], "ablated": abl_type[qt]["acc"],
                               "delta": abl_type[qt]["acc"] - base_type[qt]["acc"],
                               "total": base_type[qt]["total"]}
                         for qt in base_type if qt in abl_type}

        result = {
            "layer": layer_idx, "feature": feature_idx, "odds_ratio": odds_ratio,
            "baseline_acc": baseline["acc"], "ablated_acc": ablated["acc"],
            "delta_docvqa": ablated["acc"] - baseline["acc"],
            "per_type_delta": per_type_delta,
            "ctrl_baseline": ctrl_baseline["acc"], "ctrl_ablated": ctrl_ablated["acc"],
            "delta_ctrl": ctrl_ablated["acc"] - ctrl_baseline["acc"],
        }
        json.dump(result, open(result_path, "w"), indent=2)
        print(f"[Ablation GPU{gpu_id}] L{layer_idx}/F{feature_idx}: "
              f"ΔDocVQA={result['delta_docvqa']:+.2f}%  ΔCtrl={result['delta_ctrl']:+.2f}%", flush=True)

    del model, processor
    torch.cuda.empty_cache(); gc.collect()


def consolidate(out_dir):
    """Merge per-feature results into ablation_summary.csv."""
    import pandas as pd
    out_dir = Path(out_dir)
    rows = []
    for fp in sorted(out_dir.glob("ablation_L*_F*.json")):
        with open(fp) as f:
            r = json.load(f)
        rows.append({"layer": r["layer"], "feature": r["feature"],
                     "odds_ratio": r.get("odds_ratio", 0),
                     "baseline_acc": r["baseline_acc"], "ablated_acc": r["ablated_acc"],
                     "delta_docvqa": r["delta_docvqa"],
                     "ctrl_baseline": r.get("ctrl_baseline", 0),
                     "ctrl_ablated": r.get("ctrl_ablated", 0),
                     "delta_ctrl": r.get("delta_ctrl", 0)})
    if not rows:
        print("No ablation results found."); return
    df = pd.DataFrame(rows).sort_values("delta_docvqa")
    df.to_csv(out_dir / "ablation_summary.csv", index=False)
    print(f"Summary: {len(df)} features → {out_dir}/ablation_summary.csv")
    print(f"Top drops:")
    for _, r in df.head(10).iterrows():
        print(f"  L{int(r.layer)}/F{int(r.feature)}: ΔDocVQA={r.delta_docvqa:+.2f}%  ΔCtrl={r.delta_ctrl:+.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str,
                    default=str(ANALYSIS_DIR / "final_features" / "final_docvqa_features.csv"))
    ap.add_argument("--gpus", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--consolidate-only", action="store_true")
    args = ap.parse_args()

    out_dir = ANALYSIS_DIR / "ablation_docvqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.consolidate_only:
        consolidate(out_dir); return

    import pandas as pd
    df = pd.read_csv(args.features)
    print(f"Features to ablate: {len(df)}")

    # Build (layer, feature, odds_ratio) tuples
    feats = []
    for _, r in df.iterrows():
        odds = float(r.get("odds_ratio", 0))
        feats.append((int(r["layer"]), int(r["feature"]), odds))

    n_gpus = len(args.gpus)
    per_worker = math.ceil(len(feats) / n_gpus)
    assignments = [(args.gpus[w], feats[w*per_worker:min((w+1)*per_worker, len(feats))])
                   for w in range(n_gpus) if feats[w*per_worker:min((w+1)*per_worker, len(feats))]]

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for gpu_id, fa in assignments:
        p = mp.Process(target=_ablation_worker, args=(gpu_id, fa, str(out_dir)))
        p.start(); processes.append(p)
    for p in processes:
        p.join()

    consolidate(out_dir)


if __name__ == "__main__":
    main()
