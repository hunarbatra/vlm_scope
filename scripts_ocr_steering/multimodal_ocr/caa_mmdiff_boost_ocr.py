#!/usr/bin/env python3
"""
MMDiff-CAA (Recipe G: project + amplify) steering for OCR-Bench (mix-448).

For each top-K MMDiff OCR feature F at its home layer lF:
  v_lF       = CAA vector at layer lF (correct − incorrect last-text-token means)
  v_unit     = v_lF / ||v_lF||
  W          = unit(W_dec[F])
  coeff      = <v_unit, W>
  proj       = coeff · W                              # F's component of v_unit
  steer(β)   = v_unit + (β − 1) · proj                # boost F's component
  inject  α · steer(β)  at layer lF only

Sweeps α ∈ {0.5, 1, 2, 5} × β ∈ {1, 3, 10}.
β = 1 reduces to plain CAA at lF (feature-aware-layer baseline).
Eval = full OCR-Bench test split.

Top features are loaded from the ablation summary (sorted by most-negative ∆OCR,
bounded by capability control via ∆VQA).

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_mmdiff_boost_ocr.py \
        --ablation-csv /data1/vlm_scope_sae_mix448_textonly/analysis_ocr/ablation_ocr/ablation_summary.csv \
        --top-k 10
"""
import os, sys, json, gc, warnings, argparse, csv
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MODEL = "google/paligemma2-3b-mix-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
BASELINE_OUT = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/caa_baseline_ocr")
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/caa_mmdiff_boost_ocr")
HF_CACHE = "/data1/hf_cache/hub"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [0.5, 1.0, 2.0, 5.0]
BETAS = [1.0, 3.0, 10.0]        # β=1 ≡ plain CAA at lF; β>1 = MMDiff-boost
MAX_NEW_TOKENS = 64
# Selection gates: drop features that hurt non-OCR controls by more than these (pp)
MAX_VQA_DROP = 2.0
MAX_CTRL_DROP = 2.0


def _ocr_correct(response, gt_list):
    if response is None: return False
    if isinstance(gt_list, str): gt_list = [gt_list]
    resp = response.strip().lower()
    if not resp: return False
    for gt in gt_list:
        gt_l = str(gt).strip().lower()
        if not gt_l: continue
        if gt_l in resp or resp in gt_l: return True
    return False


class LayerInjector:
    def __init__(self, model, layer_idx, steer_vec):
        self.model = model
        self.layer = layer_idx
        self.sv = steer_vec.view(1, -1)
        self.img_end = 0
        self.handle = None

    def set_img_end(self, ie):
        self.img_end = int(ie)

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
            self.handle.remove()
            self.handle = None


def run_ocr_eval_hidden(model, processor, ocr, tokenizer, device, layer_idx):
    """Baseline eval + collect last-text-token residual at `layer_idx`."""
    from utils import process_vlm_inputs, get_image_token_positions

    collected = {"x": None}
    def _hook(module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        collected["x"] = x.detach()
        return out
    h = model.model.language_model.layers[layer_idx].register_forward_hook(_hook)

    results, hidden = [], {}
    try:
        for si in range(len(ocr)):
            sample = ocr[si]
            q = str(sample.get("question", "")).strip()
            img = sample.get("image")
            gt = sample.get("answer", [])
            if isinstance(gt, str): gt = [gt]
            if img is None or not q or not gt:
                continue
            try:
                img = img.convert("RGB") if isinstance(img, PILImage.Image) else None
                if img is None: continue
                prompt = f"answer en {q}"
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                with torch.inference_mode():
                    _ = model(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                hidden[si] = collected["x"][0, -1, :].float().cpu().clone()
                with torch.inference_mode():
                    out = model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                          max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
                gen = tokenizer.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
                ok = _ocr_correct(gen, gt)
            except Exception:
                ok = False; gen = ""
            results.append({"si": si, "cat": str(sample.get("question_type", "?")),
                            "correct": bool(ok), "response": gen})
            if (si + 1) % 200 == 0:
                c = sum(1 for r in results if r["correct"])
                print(f"    baseline L{layer_idx}: {si+1}/{len(ocr)} acc={100*c/len(results):.2f}%", flush=True)
    finally:
        h.remove()

    c = sum(1 for r in results if r["correct"])
    return {"acc": 100*c/max(len(results),1), "correct": c, "total": len(results),
            "results": results}, hidden


def run_ocr_eval(model, processor, ocr, tokenizer, device, injector):
    from utils import process_vlm_inputs, get_image_token_positions

    c = t = 0
    by_cat = defaultdict(lambda: {"c": 0, "t": 0})
    for si in range(len(ocr)):
        sample = ocr[si]
        q = str(sample.get("question", "")).strip()
        img = sample.get("image")
        gt = sample.get("answer", [])
        if isinstance(gt, str): gt = [gt]
        if img is None or not q or not gt:
            continue
        try:
            img = img.convert("RGB") if isinstance(img, PILImage.Image) else None
            if img is None: continue
            prompt = f"answer en {q}"
            iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
            _, img_end = get_image_token_positions(iids)
            injector.set_img_end(img_end)
            with torch.inference_mode():
                out = model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                      max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            gen = tokenizer.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
            ok = _ocr_correct(gen, gt)
        except Exception:
            ok = False
        cat = str(sample.get("question_type", "?"))
        t += 1; by_cat[cat]["t"] += 1
        if ok: c += 1; by_cat[cat]["c"] += 1
    return {"acc": 100*c/max(t,1), "correct": c, "total": t,
            "per_cat": {k: {"acc": 100*v["c"]/max(v["t"],1), "c": v["c"], "t": v["t"]}
                        for k, v in by_cat.items()}}


def compute_caa_at_layer(model, processor, ocr, tokenizer, device, layer_idx, baseline_dir):
    """Run baseline at layer_idx to collect per-sample last-text-token hidden states,
    then build CAA = mean(correct) - mean(incorrect).
    """
    caa_cache = baseline_dir / f"caa_L{layer_idx}.pt"
    hidden_cache = baseline_dir / f"hidden_L{layer_idx}.pt"
    per_sample_path = baseline_dir / f"per_sample_L{layer_idx}.json"

    if caa_cache.exists():
        d = torch.load(caa_cache, map_location="cpu")
        return d["v"], d["n_pos"], d["n_neg"]

    if hidden_cache.exists() and per_sample_path.exists():
        hidden = torch.load(hidden_cache, map_location="cpu")
        results = json.load(open(per_sample_path))
    else:
        print(f"    computing baseline + hidden @ L{layer_idx}...", flush=True)
        base, hidden = run_ocr_eval_hidden(model, processor, ocr, tokenizer, device, layer_idx)
        results = base["results"]
        torch.save(hidden, hidden_cache)
        with open(per_sample_path, "w") as f: json.dump(results, f)

    pos, neg = [], []
    for r in results:
        si = r["si"]
        if si not in hidden: continue
        (pos if r["correct"] else neg).append(hidden[si])
    if not pos or not neg:
        return None, len(pos), len(neg)
    v = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
    torch.save({"v": v, "n_pos": len(pos), "n_neg": len(neg), "layer": layer_idx}, caa_cache)
    return v, len(pos), len(neg)


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    if "W_dec" in d:
        return d["W_dec"][fi].float()
    # Fallback for other checkpoint layouts
    for k in ["decoder.weight", "decoder"]:
        if k in d:
            W = d[k]
            return (W.T if W.shape[0] != d.get("W_dec_shape", [0])[0] else W)[fi].float()
    return None


def _select_top_features(ablation_csv, k, max_vqa_drop, max_ctrl_drop):
    """Pick top-K features that:
      (a) drop OCR significantly (most negative delta_ocr first)
      (b) preserve VQA-clean control within max_ctrl_drop pp
      (c) preserve raw VQA within max_vqa_drop pp
    These are the "OCR-specific" features — strong OCR effect, low collateral.
    If gates leave < k, relax in order: drop ctrl gate, then drop vqa gate.
    """
    rows = []
    with open(ablation_csv) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "layer": int(r["layer"]), "feature": int(r["feature"]),
                    "delta_ocr": float(r["delta_ocr"]),
                    "delta_ctrl": float(r.get("delta_ctrl", 0)),
                    "delta_vqa": float(r["delta_vqa"]),
                    "baseline_ocr_acc": float(r.get("baseline_ocr_acc", 0)),
                    "ablated_ocr_acc": float(r.get("ablated_ocr_acc", 0)),
                })
            except (ValueError, KeyError):
                continue

    # Sort by most-negative delta_ocr (strongest OCR-degrading first)
    rows.sort(key=lambda r: r["delta_ocr"])

    # Gate 1: both ctrl AND vqa preserved
    strict = [r for r in rows
              if r["delta_ctrl"] >= -max_ctrl_drop and r["delta_vqa"] >= -max_vqa_drop]
    if len(strict) >= k:
        return strict[:k]
    # Relax: only vqa preserved
    relaxed = [r for r in rows if r["delta_vqa"] >= -max_vqa_drop]
    if len(relaxed) >= k:
        return relaxed[:k]
    # Last resort: ungated
    return rows[:k]


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-csv", required=True,
                    help="ablation_ocr/ablation_summary.csv — sort key for top features")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--betas", type=float, nargs="+", default=BETAS)
    ap.add_argument("--max-vqa-drop", type=float, default=MAX_VQA_DROP)
    ap.add_argument("--max-ctrl-drop", type=float, default=MAX_CTRL_DROP)
    args = ap.parse_args()

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    top = _select_top_features(args.ablation_csv, args.top_k,
                                 args.max_vqa_drop, args.max_ctrl_drop)
    print(f"[INFO] Top {len(top)} MMDiff OCR features "
          f"(gated ∆Ctrl ≥ {-args.max_ctrl_drop} pp AND ∆VQA ≥ {-args.max_vqa_drop} pp):")
    for i, r in enumerate(top):
        print(f"  {i+1:2}. L{r['layer']}/F{r['feature']}  "
              f"∆OCR={r['delta_ocr']:+.2f}  "
              f"∆Ctrl={r.get('delta_ctrl',0):+.2f}  "
              f"∆VQA={r['delta_vqa']:+.2f}")

    print(f"[INFO] Loading {MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = proc.tokenizer
    dtype = next(model.parameters()).dtype

    print("[INFO] Loading OCR-Bench...", flush=True)
    ocr = load_dataset("echo840/OCRBench", split="test")

    BASELINE_OUT.mkdir(parents=True, exist_ok=True)

    # Baseline OCR acc (reuse from caa_baseline_ocr if present)
    base_results_path = BASELINE_OUT / "results.json"
    base_acc = None
    if base_results_path.exists():
        try:
            base_acc = json.load(open(base_results_path))["base"]["acc"]
        except Exception:
            pass
    if base_acc is None:
        # Compute baseline without hidden collection
        print("[INFO] Computing baseline OCR acc...", flush=True)
        # Reuse any existing per-sample from another layer to avoid redundant generation
        for p in BASELINE_OUT.glob("per_sample_L*.json"):
            rs = json.load(open(p))
            c = sum(1 for r in rs if r["correct"]); n = len(rs)
            if n > 0:
                base_acc = 100 * c / n
                print(f"  reused {p.name}: base={base_acc:.2f}% ({c}/{n})", flush=True)
                break
    if base_acc is None:
        print("[ERROR] No baseline found; run caa_baseline_ocr.py first.", flush=True)
        return
    print(f"[BASE] {base_acc:.2f}%", flush=True)

    all_results = json.load(open(results_path)) if results_path.exists() else {}
    all_results["_base"] = {"acc": base_acc}

    # --- Per-feature: build CAA at lF, then α×β sweep ---
    for r in top:
        lF, fF = r["layer"], r["feature"]
        key = f"L{lF}_F{fF}"
        print(f"\n--- {key}  base={base_acc:.2f}% ---", flush=True)

        v, n_pos, n_neg = compute_caa_at_layer(model, proc, ocr, tokenizer, device,
                                                lF, BASELINE_OUT)
        if v is None:
            print(f"  [SKIP] no pos/neg split at L{lF}", flush=True); continue
        v_unit = v / v.norm().clamp(min=1e-8)

        w = _load_wdec(lF, fF)
        if w is None:
            print(f"  [SKIP] no W_dec for L{lF}/F{fF}", flush=True); continue
        w_unit = w / w.norm().clamp(min=1e-8)
        coeff = (v_unit * w_unit).sum().item()
        proj = coeff * w_unit
        print(f"  ||v||={v.norm():.3f}  cos(v,W_dec)={coeff:+.3f}  n_pos={n_pos} n_neg={n_neg}",
              flush=True)

        for beta in args.betas:
            v_boost = v_unit + (beta - 1.0) * proj
            for alpha in args.alphas:
                rk = f"{key}_b{beta:g}_a{alpha:g}"
                if rk in all_results and all_results[rk].get("total", 0) > 0:
                    rp = all_results[rk]
                    print(f"  [SKIP β={beta:g} α={alpha:g}] {rp['acc']:.2f}% Δ={rp['delta']:+.2f}%",
                          flush=True)
                    continue
                sv = (v_boost * alpha).to(dtype).to(device)
                injector = LayerInjector(model, lF, sv).install()
                try:
                    res = run_ocr_eval(model, proc, ocr, tokenizer, device, injector)
                finally:
                    injector.remove()
                delta = res["acc"] - base_acc
                all_results[rk] = {"acc": res["acc"], "delta": delta,
                                    "alpha": alpha, "beta": beta, "layer": lF, "feature": fF,
                                    "coeff": coeff, "correct": res["correct"],
                                    "total": res["total"], "per_cat": res["per_cat"]}
                with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
                print(f"  [β={beta:g} α={alpha:g}] {res['acc']:.2f}%  Δ={delta:+.2f}%  "
                      f"({res['correct']}/{res['total']})", flush=True)

        gc.collect(); torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"MMDiff-CAA boost  —  base {base_acc:.2f}%")
    print(f"{'='*80}")
    print(f"  {'Feature':<12} {'cos(v,W)':>10} {'plain Δ':>10} {'boost best':>14} {'(β,α)':>10}")
    for r in top:
        k = f"L{r['layer']}_F{r['feature']}"
        coeff = None
        plain = None
        best = None
        for key, v in all_results.items():
            if not key.startswith(k): continue
            if key == k: continue
            if coeff is None: coeff = v.get("coeff")
            b = v.get("beta"); a = v.get("alpha")
            if b == 1.0:
                if plain is None or v["delta"] > plain["delta"]:
                    plain = v
            if best is None or v["delta"] > best["delta"]:
                best = v
        if best is None: continue
        pl = f"{plain['delta']:+.2f}%" if plain else "—"
        bs = f"{best['delta']:+.2f}%"
        pp = f"(β={best['beta']:g},α={best['alpha']:g})"
        cs = f"{coeff:+.3f}" if coeff is not None else "—"
        print(f"  {k:<12} {cs:>10} {pl:>10} {bs:>14} {pp:>10}")
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
