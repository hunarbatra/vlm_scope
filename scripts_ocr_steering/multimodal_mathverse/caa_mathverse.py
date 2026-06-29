#!/usr/bin/env python3
"""
CAA steering for MathVerse (mix-448).

Two modes:
  1. BASELINE: Rimsky-style middle-layer CAA (layer 13), alpha sweep.
     v = mean(last-text-token residual | correct) - mean(... | incorrect)
     Inject α·unit(v) on text tokens at layer 13.

  2. MMDIFF: Feature-specific CAA with W_dec boost at feature's home layer.
     For each top-K MMDiff math feature F at layer lF:
       v_lF   = CAA vector at lF (correct − incorrect last-text-token means)
       W      = unit(W_dec[F])
       steer  = v_unit + (β − 1) · <v_unit, W> · W   (β=1 = plain CAA)
       inject α · steer at lF

Evaluation: MathVerse testmini (430 samples, MCQ A/B/C/D)
Capability control: VQA yes/no (1000 samples)

Usage:
  CUDA_VISIBLE_DEVICES=X python3 caa_mathverse.py --mode baseline
  CUDA_VISIBLE_DEVICES=X python3 caa_mathverse.py --mode mmdiff \
      --ablation-dir /data1/vlm_scope_sae_mix448_textonly/analysis_mathverse/ablation_mathverse \
      --top-k 5
"""
import os, sys, json, gc, re, warnings, argparse, csv
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL = "google/paligemma2-3b-mix-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
ANALYSIS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_mathverse")
HF_CACHE = "/data1/hf_cache/hub"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER = 13
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0]
BETAS = [1.0, 3.0, 10.0]
VQA_MAX = 1000
MAX_VQA_DROP = 2.0


def _get_choice_ids(tokenizer):
    choice_ids = {}
    for letter in "ABCD":
        ids_ = set()
        for form in [letter, f" {letter}", f"({letter})", f" ({letter})"]:
            try:
                t = tokenizer.encode(form, add_special_tokens=False)
                if t: ids_.add(t[0])
            except Exception: pass
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
    for t in [" Yes", "Yes", " yes"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids; yes_ids -= overlap; no_ids -= overlap
    return yes_ids, no_ids


def _predict_yesno(logits, yes_ids, no_ids):
    p = torch.softmax(logits, dim=-1)
    y = p[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = p[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


class LayerInjector:
    def __init__(self, model, layer_idx, steer_vec):
        self.model = model
        self.layer = layer_idx
        self.sv = steer_vec.view(1, -1)
        self.img_end = 0
        self.handle = None

    def set_img_end(self, ie): self.img_end = int(ie)

    def _hook(self, module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] > 1:
            start = min(self.img_end, x.shape[1])
            x[:, start:, :] = x[:, start:, :] + self.sv
        else:
            x.add_(self.sv)
        return (x,) + out[1:] if isinstance(out, tuple) else x

    def install(self):
        self.handle = self.model.model.language_model.layers[self.layer].register_forward_hook(self._hook)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove(); self.handle = None


def run_math_eval_hidden(model, processor, math_ds, tokenizer, device, layer_idx,
                          from_utils):
    """Eval + collect last-text-token residuals at layer_idx (for CAA vector)."""
    process_vlm_inputs, get_image_token_positions = from_utils

    collected = {"x": None}
    def _hook(module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        collected["x"] = x.detach()
        return out
    h = model.model.language_model.layers[layer_idx].register_forward_hook(_hook)

    choice_ids = _get_choice_ids(tokenizer)
    results, hidden = [], {}
    N = len(math_ds)
    try:
        for si in range(N):
            ex = math_ds[si]
            img = ex.get("image")
            gt = _parse_gt(ex.get("answer", ""))
            if img is None or gt is None: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(
                    img, f"answer en {ex['prompt']}", processor, model, device=device)
                _, img_end = get_image_token_positions(iids)
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn,
                                pixel_values=pv, use_cache=False)
                hidden[si] = out[0][0, -1, :].float().cpu().clone() if hasattr(out, '__getitem__') else \
                             collected["x"][0, -1, :].float().cpu().clone()
                # Use collected hidden (from hook) not logits
                hidden[si] = collected["x"][0, -1, :].float().cpu().clone()
                pred = _predict_mcq(out.logits[0, -1, :], choice_ids)
                results.append({"si": si, "correct": bool(pred == gt)})
            except Exception: continue
            if (si + 1) % 100 == 0:
                c = sum(1 for r in results if r["correct"])
                print(f"  hidden collection L{layer_idx}: {si+1}/{N} "
                      f"acc={100*c/max(len(results),1):.2f}%", flush=True)
    finally:
        h.remove()

    c = sum(1 for r in results if r["correct"])
    return {"acc": 100*c/max(len(results),1), "correct": c, "total": len(results),
            "results": results}, hidden


def run_math_eval(model, processor, math_ds, tokenizer, device, injector=None,
                   from_utils=None):
    process_vlm_inputs, get_image_token_positions = from_utils
    choice_ids = _get_choice_ids(tokenizer)
    c = t = 0
    for si in range(len(math_ds)):
        ex = math_ds[si]
        img = ex.get("image")
        gt = _parse_gt(ex.get("answer", ""))
        if img is None or gt is None: continue
        try:
            img = img.convert("RGB")
            iids, attn, pv = process_vlm_inputs(
                img, f"answer en {ex['prompt']}", processor, model, device=device)
            _, img_end = get_image_token_positions(iids)
            if injector is not None: injector.set_img_end(img_end)
            with torch.no_grad():
                out = model(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
            pred = _predict_mcq(out.logits[0, -1, :], choice_ids)
            t += 1
            if pred == gt: c += 1
        except Exception: pass
    return {"acc": 100*c/max(t,1), "correct": c, "total": t}


def run_vqa_eval(model, processor, vqa, vqa_yesno, tokenizer, device, injector=None,
                  from_utils=None):
    process_vlm_inputs, get_image_token_positions = from_utils
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    c = t = 0
    for qi, label in vqa_yesno:
        ex = vqa[qi]
        img = ex.get("image")
        if img is None: continue
        try:
            img = img.convert("RGB")
            prompt = (f"Answer the following question with only 'Yes' or 'No':\n"
                      f"Question: {ex.get('question', '').strip()}\nAnswer:")
            iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
            _, img_end = get_image_token_positions(iids)
            if injector is not None: injector.set_img_end(img_end)
            with torch.inference_mode():
                out = model(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
            pred = _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
            t += 1
            if pred == label: c += 1
        except Exception: pass
    return {"acc": 100*c/max(t,1), "correct": c, "total": t}


def build_caa_vector(hidden, results):
    pos, neg = [], []
    si_to_correct = {r["si"]: r["correct"] for r in results}
    for si, h in hidden.items():
        (pos if si_to_correct.get(si, False) else neg).append(h)
    if not pos or not neg:
        return None, len(pos), len(neg)
    v = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
    return v, len(pos), len(neg)


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    if "W_dec" in d: return d["W_dec"][fi].float()
    return None


def _select_top_features(ablation_dir, k, max_vqa_drop):
    ablation_dir = Path(ablation_dir)
    rows = []
    for p in sorted(ablation_dir.glob("ablation_L*_F*.json")):
        with open(p) as f:
            r = json.load(f)
        rows.append(r)
    rows.sort(key=lambda r: r["delta_math"])
    gated = [r for r in rows if r["delta_vqa"] >= -max_vqa_drop]
    if len(gated) >= k: return gated[:k]
    return rows[:k]


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import process_vlm_inputs, get_image_token_positions
    from_utils = (process_vlm_inputs, get_image_token_positions)

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "mmdiff"], default="baseline")
    ap.add_argument("--ablation-dir", default=str(ANALYSIS_DIR / "ablation_mathverse"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--betas", type=float, nargs="+", default=BETAS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-vqa-drop", type=float, default=MAX_VQA_DROP)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    out_dir = ANALYSIS_DIR / f"caa_{args.mode}_mathverse"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading mix-448 on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = processor.tokenizer

    print("[INFO] Loading MathVerse testmini...", flush=True)
    math_ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")
    N = len(math_ds)
    print(f"[INFO] {N} MathVerse samples", flush=True)

    print("[INFO] Loading VQA yes/no...", flush=True)
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno = []
    for i in range(len(vqa)):
        ex = vqa[i]
        at = str(ex.get("answer_type", "")).lower()
        mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
        if at == "yes/no" and mc in {"yes", "no"}:
            vqa_yesno.append((i, 1 if mc == "yes" else 0))
            if len(vqa_yesno) >= VQA_MAX: break
    print(f"[INFO] VQA yes/no: {len(vqa_yesno)}", flush=True)

    # Collect hidden states at steering layer for CAA vector
    if args.mode == "baseline":
        layer_idx = MIDDLE_LAYER
    else:
        top_features = _select_top_features(args.ablation_dir, args.top_k, args.max_vqa_drop)
        print(f"\nTop {len(top_features)} MMDiff math features:")
        for i, r in enumerate(top_features):
            print(f"  {i+1}. L{r['layer']}/F{r['feature']}  ∆Math={r['delta_math']:+.2f}  ∆VQA={r['delta_vqa']:+.2f}")
        # Use most-impactful feature's layer for CAA vector collection
        layer_idx = top_features[0]["layer"]

    # Build CAA vector
    hidden_cache = out_dir / f"hidden_L{layer_idx}.pt"
    per_sample_cache = out_dir / f"per_sample_L{layer_idx}.json"
    caa_cache = out_dir / f"caa_L{layer_idx}.pt"

    if caa_cache.exists():
        d = torch.load(caa_cache, map_location="cpu")
        v_raw = d["v"]
        n_pos, n_neg = d["n_pos"], d["n_neg"]
        results_base = d["results"]
        print(f"[INFO] Loaded CAA L{layer_idx}: n_pos={n_pos}, n_neg={n_neg}", flush=True)
    else:
        print(f"[INFO] Computing baseline + hidden states @ L{layer_idx}...", flush=True)
        base_info, hidden = run_math_eval_hidden(
            model, processor, math_ds, tokenizer, device, layer_idx, from_utils)
        v_raw, n_pos, n_neg = build_caa_vector(hidden, base_info["results"])
        results_base = base_info["results"]
        torch.save({"v": v_raw, "n_pos": n_pos, "n_neg": n_neg,
                    "layer": layer_idx, "results": results_base}, caa_cache)
        print(f"[INFO] CAA L{layer_idx}: n_pos={n_pos}, n_neg={n_neg}, "
              f"baseline={base_info['acc']:.2f}%", flush=True)

    if v_raw is None:
        print("[ERROR] CAA vector is None — not enough pos/neg samples. Exiting.")
        return

    # Baseline math accuracy (no steering)
    base_math_path = out_dir / "baseline_math.json"
    if base_math_path.exists():
        base_math = json.load(open(base_math_path))
        print(f"[INFO] Baseline math: {base_math['acc']:.2f}%", flush=True)
    else:
        print("[INFO] Computing math baseline...", flush=True)
        base_math = run_math_eval(model, processor, math_ds, tokenizer, device,
                                   injector=None, from_utils=from_utils)
        with open(base_math_path, "w") as f: json.dump(base_math, f, indent=2)
        print(f"[INFO] Baseline math: {base_math['acc']:.2f}%", flush=True)

    base_vqa_path = out_dir / "baseline_vqa.json"
    if base_vqa_path.exists():
        base_vqa = json.load(open(base_vqa_path))
        print(f"[INFO] Baseline VQA: {base_vqa['acc']:.2f}%", flush=True)
    else:
        print("[INFO] Computing VQA baseline...", flush=True)
        base_vqa = run_vqa_eval(model, processor, vqa, vqa_yesno, tokenizer, device,
                                 injector=None, from_utils=from_utils)
        with open(base_vqa_path, "w") as f: json.dump(base_vqa, f, indent=2)
        print(f"[INFO] Baseline VQA: {base_vqa['acc']:.2f}%", flush=True)

    all_results = []

    if args.mode == "baseline":
        v_unit = v_raw / v_raw.norm().clamp(min=1e-8)
        print(f"\n[BASELINE CAA] layer={layer_idx}, alphas={args.alphas}", flush=True)
        for alpha in args.alphas:
            steer = (alpha * v_unit).to(device).to(next(model.parameters()).dtype)
            inj = LayerInjector(model, layer_idx, steer).install()
            math_r = run_math_eval(model, processor, math_ds, tokenizer, device,
                                    injector=inj, from_utils=from_utils)
            vqa_r  = run_vqa_eval(model, processor, vqa, vqa_yesno, tokenizer, device,
                                   injector=inj, from_utils=from_utils)
            inj.remove()
            res = {
                "mode": "baseline", "layer": layer_idx, "alpha": alpha, "beta": 1.0,
                "math_acc": math_r["acc"], "delta_math": math_r["acc"] - base_math["acc"],
                "vqa_acc": vqa_r["acc"],   "delta_vqa":  vqa_r["acc"]  - base_vqa["acc"],
                "n_pos": n_pos, "n_neg": n_neg,
            }
            all_results.append(res)
            print(f"  α={alpha}: math={math_r['acc']:.2f}% (Δ={res['delta_math']:+.2f}pp)  "
                  f"vqa={vqa_r['acc']:.2f}% (Δ={res['delta_vqa']:+.2f}pp)", flush=True)
            torch.cuda.empty_cache()

    else:  # mmdiff
        for fi, feat in enumerate(top_features):
            fl = feat["layer"]
            ff = feat["feature"]
            feat_key = f"L{fl}_F{ff}"

            # Get CAA vector at this feature's layer
            if fl != layer_idx:
                caa_fl_cache = out_dir / f"caa_L{fl}.pt"
                if caa_fl_cache.exists():
                    d = torch.load(caa_fl_cache, map_location="cpu")
                    v_fl = d["v"]; n_pos_fl = d["n_pos"]; n_neg_fl = d["n_neg"]
                else:
                    print(f"[INFO] Computing hidden @ L{fl}...", flush=True)
                    base_fl, hidden_fl = run_math_eval_hidden(
                        model, processor, math_ds, tokenizer, device, fl, from_utils)
                    v_fl, n_pos_fl, n_neg_fl = build_caa_vector(hidden_fl, base_fl["results"])
                    torch.save({"v": v_fl, "n_pos": n_pos_fl, "n_neg": n_neg_fl,
                                "layer": fl, "results": base_fl["results"]}, caa_fl_cache)
                    print(f"[INFO] CAA L{fl}: n_pos={n_pos_fl}, n_neg={n_neg_fl}", flush=True)
            else:
                v_fl = v_raw; n_pos_fl = n_pos; n_neg_fl = n_neg

            if v_fl is None:
                print(f"[SKIP] L{fl}: no CAA vector"); continue

            v_unit_fl = v_fl / v_fl.norm().clamp(min=1e-8)

            # Load W_dec for this feature
            w_dec = _load_wdec(fl, ff)
            if w_dec is None:
                print(f"[SKIP] {feat_key}: no W_dec"); continue
            w_unit = (w_dec / w_dec.norm().clamp(min=1e-8)).to(device).to(torch.float32)
            v_unit_dev = v_unit_fl.to(device).to(torch.float32)
            coeff = (v_unit_dev @ w_unit).item()
            print(f"\n[MMDiff] {feat_key}: <v_unit, W_dec>={coeff:.4f}, "
                  f"n_pos={n_pos_fl}, n_neg={n_neg_fl}", flush=True)

            for alpha in args.alphas:
                for beta in args.betas:
                    res_key = f"{feat_key}_a{alpha}_b{beta}"
                    res_path = out_dir / f"result_{feat_key}_a{alpha}_b{beta}.json"
                    if res_path.exists():
                        print(f"  [SKIP] {res_key}"); continue

                    # steer = v_unit + (β-1) * coeff * W
                    steer = v_unit_dev + (beta - 1.0) * coeff * w_unit
                    steer = (alpha * steer).to(next(model.parameters()).dtype)

                    inj = LayerInjector(model, fl, steer).install()
                    math_r = run_math_eval(model, processor, math_ds, tokenizer, device,
                                           injector=inj, from_utils=from_utils)
                    vqa_r  = run_vqa_eval(model, processor, vqa, vqa_yesno, tokenizer, device,
                                          injector=inj, from_utils=from_utils)
                    inj.remove()

                    res = {
                        "mode": "mmdiff", "layer": fl, "feature": ff,
                        "alpha": alpha, "beta": beta,
                        "coeff": coeff,
                        "math_acc": math_r["acc"], "delta_math": math_r["acc"] - base_math["acc"],
                        "vqa_acc": vqa_r["acc"],   "delta_vqa":  vqa_r["acc"]  - base_vqa["acc"],
                        "n_pos": n_pos_fl, "n_neg": n_neg_fl,
                    }
                    all_results.append(res)
                    with open(res_path, "w") as f: json.dump(res, f, indent=2)
                    print(f"  α={alpha} β={beta}: math={math_r['acc']:.2f}% (Δ={res['delta_math']:+.2f}pp)  "
                          f"vqa={vqa_r['acc']:.2f}% (Δ={res['delta_vqa']:+.2f}pp)", flush=True)
                    torch.cuda.empty_cache()

    # Save summary
    summary_path = out_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "baseline_math_acc": base_math["acc"],
            "baseline_vqa_acc": base_vqa["acc"],
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"MATHVERSE CAA — {args.mode.upper()} RESULTS")
    print(f"  Baseline math: {base_math['acc']:.2f}%")
    print(f"  Baseline VQA:  {base_vqa['acc']:.2f}%")
    print(f"{'='*70}")
    all_results.sort(key=lambda x: -x["delta_math"])
    for r in all_results[:10]:
        feat = f"L{r['layer']}" if args.mode == "baseline" else f"L{r['layer']}_F{r['feature']}"
        print(f"  {feat} α={r['alpha']} β={r.get('beta',1)}: "
              f"math={r['math_acc']:.2f}% (Δ={r['delta_math']:+.2f}pp)  "
              f"vqa={r['vqa_acc']:.2f}% (Δ={r['delta_vqa']:+.2f}pp)")
    print(f"\nResults saved → {summary_path}")


if __name__ == "__main__":
    main()
