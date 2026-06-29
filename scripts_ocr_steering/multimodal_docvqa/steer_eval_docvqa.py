#!/usr/bin/env python3
"""
DocVQA CAA steering eval: mix→pt, Recipe A (Middle) and Recipe D (BB+W(γ)).

Vector: mix-448 correct/incorrect on DocVQA val[0:4278].
Eval on: DocVQA val[4279:5348] (1070 samples).

Recipe A (MIDDLE):
  inject α·unit(v_L13) at L13 of pt-448 (text tokens only)

Recipe D (BB+W(γ)) — run once per feature F at layer lF:
  Backbone: inject α·unit(v_L) at ALL backbone layers {L15,L17,L19,L21}
  At lF additionally: inject γ·unit(W_dec[F])
  Total at lF: α·unit(v_lF) + γ·unit(W_dec[F])
  Sweep: α ∈ {0.5,1,2,5,10,20}, γ ∈ {1,3,10}

5 DocVQA features (home layers = backbone layers):
  L15/F3923  — Scene Text
  L17/F13602 — Scene Text
  L19/F10089 — Scene Text VQA  (spotlight: layout, free_text)
  L21/F9577  — Digit String    (spotlight: form, table/list)
  L19/F14093 — Irregular Text  (spotlight: handwritten, free_text)

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 steer_eval_docvqa.py --gpu 0 --condition middle
  CUDA_VISIBLE_DEVICES=1 python3 steer_eval_docvqa.py --gpu 0 --condition recipe_d --feature L15F3923
"""
import os, sys, json, warnings, argparse
from pathlib import Path
from collections import defaultdict

import torch

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

PT_MODEL    = "google/paligemma2-3b-pt-448"
SAE_CKPT    = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
VEC_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_docvqa/steering_docvqa")
SPLITS_JSON = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_docvqa/splits.json")

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MAX_NEW_TOKENS  = 64
ALPHAS          = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
GAMMAS          = [1.0, 3.0, 10.0]
MIDDLE_LAYER    = 13
BACKBONE_LAYERS = [15, 17, 19, 21]

FEATURES = [
    {"layer": 15, "feature": 3923,  "label": "L15F3923",
     "category": "Scene Text",     "spotlight_types": ["layout", "form"]},
    {"layer": 17, "feature": 13602, "label": "L17F13602",
     "category": "Scene Text",     "spotlight_types": ["layout", "form"]},
    {"layer": 19, "feature": 10089, "label": "L19F10089",
     "category": "Scene Text VQA", "spotlight_types": ["layout", "free_text"]},
    {"layer": 21, "feature": 9577,  "label": "L21F9577",
     "category": "Digit String",   "spotlight_types": ["form", "table/list"]},
    {"layer": 19, "feature": 14093, "label": "L19F14093",
     "category": "Irregular Text", "spotlight_types": ["handwritten", "free_text"]},
]


def _correct(resp, gt_list):
    if resp is None: return False
    if isinstance(gt_list, str): gt_list = [gt_list]
    r = resp.strip().lower()
    for gt in gt_list:
        g = str(gt).strip().lower()
        if g and (g in r or r in g): return True
    return False


def _load_vec(tag, layer):
    p = VEC_DIR / f"{tag}_caa_L{layer}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Vector not found: {p}")
    d = torch.load(p, map_location="cpu")
    return d["v_unit"]


def _load_wdec_unit(layer, feature):
    p = SAE_CKPT / f"text-only_layer_{layer}.pt"
    if not p.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {p}")
    d = torch.load(p, map_location="cpu", weights_only=True)
    if "W_dec" in d:
        w = d["W_dec"][feature].float()
    else:
        for k in ["decoder.weight", "decoder"]:
            if k in d:
                W = d[k]
                w = (W[feature] if W.shape[0] > W.shape[1] else W.T[feature]).float()
                break
        else:
            raise KeyError(f"No W_dec key in {p}")
    return w / w.norm().clamp(min=1e-8)


class MultiLayerInjector:
    """Injects pre-scaled steering vectors at multiple layers (text tokens only)."""
    def __init__(self, model, layer_vecs):
        # layer_vecs: {layer_int: tensor shape (1, D), already scaled}
        self.model      = model
        self.layer_vecs = layer_vecs
        self.img_end    = 0
        self.handles    = []

    def set_img_end(self, ie):
        self.img_end = int(ie)

    def _make_hook(self, sv):
        def _hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if x.shape[1] > 1:  # prefill
                s = min(self.img_end, x.shape[1])
                x[:, s:, :] = x[:, s:, :] + sv
            else:               # generation step
                x.add_(sv)
            return (x,) + out[1:] if isinstance(out, tuple) else x
        return _hook

    def install(self):
        for l, sv in self.layer_vecs.items():
            self.handles.append(
                self.model.model.language_model.layers[l].register_forward_hook(
                    self._make_hook(sv)))
        return self

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def eval_docvqa(model, processor, tokenizer, samples, device, injector=None):
    from utils import process_vlm_inputs, get_image_token_positions

    c = t = 0
    by_type = defaultdict(lambda: {"c": 0, "t": 0})
    for i, sample in enumerate(samples):
        try:
            img = sample.get("image")
            q   = str(sample.get("question", "")).strip()
            gt  = sample.get("answers") or []
            if isinstance(gt, str): gt = [gt]
            if img is None or not q or not gt: continue
            img = img.convert("RGB")
            iids, attn, pv = process_vlm_inputs(
                img, f"answer en {q}", processor, model, device=device)
            if injector is not None:
                _, img_end = get_image_token_positions(iids)
                injector.set_img_end(img_end)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=iids, attention_mask=attn, pixel_values=pv,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            resp = tokenizer.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
            ok = _correct(resp, gt)
        except Exception:
            ok = False
        t += 1
        if ok: c += 1
        qtype = str(sample.get("question_type", sample.get("type", "unknown")))
        by_type[qtype]["t"] += 1
        if ok: by_type[qtype]["c"] += 1
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(samples)} acc={100*c/max(t,1):.2f}%", flush=True)

    per_type = {k: {"acc": 100*v["c"]/max(v["t"],1), "c": v["c"], "t": v["t"]}
                for k, v in by_type.items()}
    return {"acc": 100*c/max(t,1), "correct": c, "total": t, "per_type": per_type}


def run_middle(model, processor, tokenizer, samples, device,
               alphas, v_mid, dtype, base_acc, all_results, results_path):
    print(f"\n── MIDDLE (L{MIDDLE_LAYER}) ──", flush=True)
    for alpha in alphas:
        rk = f"middle_a{alpha:g}"
        if rk in all_results and all_results[rk].get("total", 0) > 0:
            print(f"  [SKIP α={alpha:g}] {all_results[rk]['acc']:.2f}% "
                  f"Δ={all_results[rk]['delta']:+.2f}%", flush=True)
            continue
        sv  = (v_mid * alpha).to(dtype).to(device).view(1, -1)
        inj = MultiLayerInjector(model, {MIDDLE_LAYER: sv}).install()
        try:
            res = eval_docvqa(model, processor, tokenizer, samples, device, inj)
        finally:
            inj.remove()
        delta = res["acc"] - base_acc
        all_results[rk] = {"acc": res["acc"], "delta": delta, "alpha": alpha,
                           "condition": "middle", "correct": res["correct"],
                           "total": res["total"], "per_type": res["per_type"]}
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
        print(f"  [MIDDLE α={alpha:g}] {res['acc']:.2f}%  Δ={delta:+.2f}%", flush=True)


def run_recipe_d_feature(model, processor, tokenizer, samples, device,
                         feat, alphas, gammas, backbone_vecs, dtype,
                         base_acc, all_results, results_path):
    """Recipe D for one feature: backbone CAA at all backbone layers + γ·unit(W_dec[F]) at lF."""
    lF    = feat["layer"]
    label = feat["label"]
    w_unit = _load_wdec_unit(lF, feat["feature"]).to(dtype).to(device)

    print(f"\n── RECIPE_D  {label} ({feat['category']}) ──", flush=True)
    for alpha in alphas:
        for gamma in gammas:
            rk = f"reciped_{label}_a{alpha:g}_g{gamma:g}"
            if rk in all_results and all_results[rk].get("total", 0) > 0:
                print(f"  [SKIP α={alpha:g} γ={gamma:g}] {all_results[rk]['acc']:.2f}% "
                      f"Δ={all_results[rk]['delta']:+.2f}%", flush=True)
                continue
            # Backbone: α·unit(v_L) at all backbone layers; at lF also add γ·unit(W_dec[F])
            layer_vecs = {}
            for l, v_u in backbone_vecs.items():
                sv = (v_u * alpha).to(dtype).to(device).view(1, -1)
                if l == lF:
                    sv = sv + (w_unit * gamma).view(1, -1)
                layer_vecs[l] = sv
            inj = MultiLayerInjector(model, layer_vecs).install()
            try:
                res = eval_docvqa(model, processor, tokenizer, samples, device, inj)
            finally:
                inj.remove()
            delta = res["acc"] - base_acc
            all_results[rk] = {
                "acc": res["acc"], "delta": delta, "alpha": alpha, "gamma": gamma,
                "condition": "recipe_d", "feature": label,
                "correct": res["correct"], "total": res["total"],
                "per_type": res["per_type"],
            }
            with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
            print(f"  [{label} α={alpha:g} γ={gamma:g}] {res['acc']:.2f}%  Δ={delta:+.2f}%",
                  flush=True)


def print_summary(all_results, alphas, gammas, base_acc, results_path):
    print(f"\n{'='*72}")
    print(f"DocVQA CAA Steering  —  pt-448 base {base_acc:.2f}%")
    print(f"{'='*72}")
    print(f"\nMIDDLE (L{MIDDLE_LAYER}):")
    for alpha in alphas:
        rk = f"middle_a{alpha:g}"
        if rk in all_results:
            r = all_results[rk]
            print(f"  α={alpha:g}  {r['acc']:.2f}%  Δ={r['delta']:+.2f}%")
    print(f"\nRECIPE_D (best per feature):")
    for feat in FEATURES:
        label = feat["label"]
        candidates = [(rk, v) for rk, v in all_results.items()
                      if rk.startswith(f"reciped_{label}_")]
        if not candidates: continue
        best_rk, best = max(candidates, key=lambda x: x[1].get("delta", -999))
        print(f"  {label}  α={best['alpha']:g} γ={best['gamma']:g}  "
              f"{best['acc']:.2f}%  Δ={best['delta']:+.2f}%")
        if "per_type" in best:
            for stype in feat["spotlight_types"]:
                pt = best["per_type"].get(stype)
                if pt:
                    print(f"    └ {stype}: {pt['acc']:.1f}% ({pt['c']}/{pt['t']})")
    print(f"\nResults: {results_path}")


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent.parent))

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",       type=str, default="0")
    ap.add_argument("--alphas",    type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--gammas",    type=float, nargs="+", default=GAMMAS)
    ap.add_argument("--condition", choices=["middle", "recipe_d", "both"], default="both")
    ap.add_argument("--feature",   type=str, default=None,
                    help="Run only this feature label (e.g. L19F10089). None = all.")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Per-feature files avoid race conditions when multiple GPUs run in parallel
    if args.feature:
        results_path = OUT_DIR / f"results_{args.feature}.json"
    elif args.condition == "middle":
        results_path = OUT_DIR / "results_middle.json"
    else:
        results_path = OUT_DIR / "results.json"

    print("[INFO] Loading CAA vectors...", flush=True)
    v_mid         = _load_vec("docvqa", MIDDLE_LAYER)
    backbone_vecs = {l: _load_vec("docvqa", l) for l in BACKBONE_LAYERS}
    print(f"  backbone layers: {list(backbone_vecs.keys())}", flush=True)

    print("[INFO] Loading DocVQA test split...", flush=True)
    ds      = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    splits  = json.load(open(SPLITS_JSON))
    samples = [ds[i] for i in splits["test"]]
    print(f"  {len(samples)} test samples", flush=True)

    print("[INFO] Loading pt-448...", flush=True)
    pt_proc  = AutoProcessor.from_pretrained(PT_MODEL)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    pt_tok   = pt_proc.tokenizer
    dtype    = next(pt_model.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    if "baseline" not in all_results:
        print("[INFO] pt-448 baseline...", flush=True)
        base = eval_docvqa(pt_model, pt_proc, pt_tok, samples, device)
        all_results["baseline"] = base
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
        print(f"  baseline: {base['acc']:.2f}% ({base['correct']}/{base['total']})", flush=True)
    base_acc = all_results["baseline"]["acc"]
    print(f"[BASE] {base_acc:.2f}%", flush=True)

    features_to_run = FEATURES
    if args.feature:
        features_to_run = [f for f in FEATURES if f["label"] == args.feature]
        if not features_to_run:
            print(f"[ERROR] Unknown feature label: {args.feature}. "
                  f"Valid: {[f['label'] for f in FEATURES]}", flush=True)
            sys.exit(1)

    if args.condition in ("middle", "both"):
        run_middle(pt_model, pt_proc, pt_tok, samples, device,
                   args.alphas, v_mid, dtype, base_acc, all_results, results_path)

    if args.condition in ("recipe_d", "both"):
        for feat in features_to_run:
            run_recipe_d_feature(pt_model, pt_proc, pt_tok, samples, device,
                                 feat, args.alphas, args.gammas, backbone_vecs,
                                 dtype, base_acc, all_results, results_path)

    print_summary(all_results, args.alphas, args.gammas, base_acc, results_path)
    print(f"[DONE] Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
