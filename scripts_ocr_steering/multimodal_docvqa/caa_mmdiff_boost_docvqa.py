#!/usr/bin/env python3
"""
DocVQA CAA steering: mix-448 → pt-448, two conditions.

Vector construction (mix-448 on DocVQA val[0:4278]):
  v_L = mean(hidden @ correct answers, layer L)
      - mean(hidden @ incorrect answers, layer L)

Condition A — MIDDLE:
  inject α·unit(v_13) at L13 of pt-448  (no feature boost)

Condition B — FEATURE (Recipe D):
  steer(β) = unit(v_19 + β·W_dec[F10089])
  inject α·steer(β) at L19 of pt-448

Eval: DocVQA val[4279:5348] (1070 held-out samples), substring-match accuracy.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -B caa_mmdiff_boost_docvqa.py
"""
import os, sys, json, gc, warnings, argparse
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL   = "google/paligemma2-3b-mix-448"
PT_MODEL    = "google/paligemma2-3b-pt-448"
SAE_CKPT    = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
ANALYSIS    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_docvqa")
OUT_DIR     = ANALYSIS / "caa_mmdiff_boost_docvqa"
SPLITS_JSON = ANALYSIS / "splits.json"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER   = 13
FEATURE_LAYER  = 19
FEATURE_IDX    = 10089
ALPHAS         = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
BETAS          = [1.0, 3.0, 10.0]   # β=1 → plain CAA at lF; β>1 → W_dec boost
MAX_NEW_TOKENS = 64


def _correct(resp, gt_list):
    if resp is None: return False
    if isinstance(gt_list, str): gt_list = [gt_list]
    r = resp.strip().lower()
    if not r: return False
    for gt in gt_list:
        g = str(gt).strip().lower()
        if g and (g in r or r in g): return True
    return False


def _load_wdec(layer, feature):
    p = SAE_CKPT / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    if "W_dec" in d:
        return d["W_dec"][feature].float()
    for k in ["decoder.weight", "decoder"]:
        if k in d:
            W = d[k]
            return (W[feature] if W.shape[0] > W.shape[1] else W.T[feature]).float()
    return None


class LayerInjector:
    def __init__(self, model, layer_idx, steer_vec):
        self.model = model; self.layer = layer_idx
        self.sv = steer_vec.view(1, -1); self.img_end = 0; self.handle = None

    def set_img_end(self, ie): self.img_end = int(ie)

    def _hook(self, module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] > 1:
            s = min(self.img_end, x.shape[1])
            x[:, s:, :] = x[:, s:, :] + self.sv
        else:
            x.add_(self.sv)
        return (x,) + out[1:] if isinstance(out, tuple) else x

    def install(self):
        self.handle = self.model.model.language_model.layers[self.layer]\
            .register_forward_hook(self._hook)
        return self

    def remove(self):
        if self.handle: self.handle.remove(); self.handle = None


def collect_hidden_and_labels(model, processor, tokenizer, samples, layer_idx, device, tag):
    """Run mix-448 on samples, collect last-text-token hidden @ layer_idx,
    split by correct/incorrect. Returns (pos_vecs, neg_vecs)."""
    from utils import process_vlm_inputs

    cache = OUT_DIR / f"hidden_{tag}_L{layer_idx}.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        print(f"  [cache] {tag} L{layer_idx}: pos={d['pos'].shape[0]} neg={d['neg'].shape[0]}", flush=True)
        return d["pos"], d["neg"]

    collected = {"x": None}
    def _hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        collected["x"] = x.detach(); return out
    h = model.model.language_model.layers[layer_idx].register_forward_hook(_hook)

    pos, neg = [], []
    try:
        for i, sample in enumerate(samples):
            try:
                img = sample.get("image")
                q   = str(sample.get("question", "")).strip()
                gt  = sample.get("answers", [])
                if isinstance(gt, str): gt = [gt]
                if img is None or not q or not gt: continue
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(img, f"answer en {q}", processor, model, device=device)
                with torch.inference_mode():
                    model(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                hidden = collected["x"][0, -1, :].float().cpu()
                # generate to check correctness
                with torch.inference_mode():
                    out_ = model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                          max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
                resp = tokenizer.decode(out_[0, iids.shape[1]:], skip_special_tokens=True)
                (pos if _correct(resp, gt) else neg).append(hidden)
            except Exception:
                continue
            if (i + 1) % 500 == 0:
                print(f"  {tag} L{layer_idx}: {i+1}/{len(samples)} pos={len(pos)} neg={len(neg)}", flush=True)
    finally:
        h.remove()

    pos_t = torch.stack(pos) if pos else torch.zeros(0, 2304)
    neg_t = torch.stack(neg) if neg else torch.zeros(0, 2304)
    torch.save({"pos": pos_t, "neg": neg_t}, cache)
    print(f"  {tag} L{layer_idx}: pos={pos_t.shape[0]} neg={neg_t.shape[0]} saved", flush=True)
    return pos_t, neg_t


def build_caa(model, processor, tokenizer, samples, layer_idx, device):
    """CAA = mean(correct hidden) - mean(incorrect hidden) at layer_idx."""
    caa_cache = OUT_DIR / f"caa_L{layer_idx}.pt"
    if caa_cache.exists():
        d = torch.load(caa_cache, map_location="cpu")
        print(f"  [cache] CAA L{layer_idx}: ||v||={d['v'].norm():.3f} n+={d['n_pos']} n-={d['n_neg']}", flush=True)
        return d["v"]

    pos, neg = collect_hidden_and_labels(model, processor, tokenizer, samples, layer_idx, device, "train")
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        print(f"  [WARN] empty pos/neg at L{layer_idx}", flush=True); return None
    v = pos.mean(0) - neg.mean(0)
    torch.save({"v": v, "n_pos": pos.shape[0], "n_neg": neg.shape[0]}, caa_cache)
    print(f"  CAA L{layer_idx}: ||v||={v.norm():.3f} n+={pos.shape[0]} n-={neg.shape[0]}", flush=True)
    return v


def eval_pt(model, processor, tokenizer, samples, device, injector=None):
    """Eval pt-448 on samples, optionally with injector."""
    from utils import process_vlm_inputs, get_image_token_positions

    c = t = 0
    by_type = defaultdict(lambda: {"c": 0, "t": 0})
    for i, sample in enumerate(samples):
        try:
            img = sample.get("image")
            q   = str(sample.get("question", "")).strip()
            gt  = sample.get("answers", [])
            if isinstance(gt, str): gt = [gt]
            if img is None or not q or not gt: continue
            img = img.convert("RGB")
            iids, attn, pv = process_vlm_inputs(img, f"answer en {q}", processor, model, device=device)
            if injector is not None:
                _, img_end = get_image_token_positions(iids)
                injector.set_img_end(img_end)
            with torch.inference_mode():
                out = model.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                     max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            resp = tokenizer.decode(out[0, iids.shape[1]:], skip_special_tokens=True)
            ok = _correct(resp, gt)
        except Exception:
            ok = False
        t += 1
        if ok: c += 1
        for qt in (sample.get("question_types") or ["unknown"]):
            by_type[qt]["t"] += 1
            if ok: by_type[qt]["c"] += 1
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(samples)} acc={100*c/max(t,1):.2f}%", flush=True)

    per_type = {k: {"acc": 100*v["c"]/max(v["t"],1), "c": v["c"], "t": v["t"]}
                for k, v in by_type.items()}
    return {"acc": 100*c/max(t,1), "correct": c, "total": t, "per_type": per_type}


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent.parent))

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    splits = json.load(open(SPLITS_JSON))
    train_idx = splits["train"]
    test_idx  = splits["test"]

    print("[INFO] Loading DocVQA...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    train_samples = [ds[i] for i in train_idx]
    test_samples  = [ds[i] for i in test_idx]
    print(f"  train={len(train_samples)} test={len(test_samples)}", flush=True)

    # Load W_dec for L19/F10089
    print(f"[INFO] Loading W_dec for L{FEATURE_LAYER}/F{FEATURE_IDX}...", flush=True)
    w = _load_wdec(FEATURE_LAYER, FEATURE_IDX)
    if w is None:
        print("[ERROR] W_dec not found.", flush=True); return
    w_unit = (w / w.norm().clamp(min=1e-8))

    # ── Phase 1: build CAA vectors with mix-448 ─────────────────────────────
    print(f"\n[PHASE 1] Building CAA vectors with mix-448...", flush=True)
    mix_proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    mix_tok   = mix_proc.tokenizer

    v_mid  = build_caa(mix_model, mix_proc, mix_tok, train_samples, MIDDLE_LAYER, device)
    v_feat = build_caa(mix_model, mix_proc, mix_tok, train_samples, FEATURE_LAYER, device)

    del mix_model; gc.collect(); torch.cuda.empty_cache()

    if v_mid is None or v_feat is None:
        print("[ERROR] CAA build failed.", flush=True); return

    v_mid_unit  = v_mid  / v_mid.norm().clamp(min=1e-8)
    v_feat_unit = v_feat / v_feat.norm().clamp(min=1e-8)
    coeff = (v_feat_unit * w_unit).sum().item()
    print(f"  cos(v_L{FEATURE_LAYER}, W_dec[F{FEATURE_IDX}]) = {coeff:+.4f}", flush=True)

    # ── Phase 2: steer pt-448, eval on test set ─────────────────────────────
    print(f"\n[PHASE 2] Loading pt-448 for steering eval...", flush=True)
    pt_proc  = AutoProcessor.from_pretrained(PT_MODEL)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    pt_tok   = pt_proc.tokenizer
    dtype    = next(pt_model.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # pt-448 baseline
    if "baseline" not in all_results:
        print("[INFO] pt-448 baseline...", flush=True)
        base = eval_pt(pt_model, pt_proc, pt_tok, test_samples, device)
        all_results["baseline"] = base
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
        print(f"  pt-448 baseline: {base['acc']:.2f}% ({base['correct']}/{base['total']})", flush=True)
    base_acc = all_results["baseline"]["acc"]
    print(f"[BASE] {base_acc:.2f}%", flush=True)

    # Condition A: MIDDLE — inject unit(v_13) at L13, sweep α
    print(f"\n── Condition A: MIDDLE (L{MIDDLE_LAYER}) ──", flush=True)
    for alpha in ALPHAS:
        rk = f"middle_a{alpha:g}"
        if rk in all_results and all_results[rk].get("total", 0) > 0:
            r = all_results[rk]
            print(f"  [SKIP α={alpha:g}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
            continue
        sv  = (v_mid_unit * alpha).to(dtype).to(device)
        inj = LayerInjector(pt_model, MIDDLE_LAYER, sv).install()
        try:
            res = eval_pt(pt_model, pt_proc, pt_tok, test_samples, device, inj)
        finally:
            inj.remove()
        delta = res["acc"] - base_acc
        all_results[rk] = {"acc": res["acc"], "delta": delta, "alpha": alpha,
                            "condition": "middle", "layer": MIDDLE_LAYER,
                            "correct": res["correct"], "total": res["total"],
                            "per_type": res["per_type"]}
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
        print(f"  [MIDDLE α={alpha:g}] {res['acc']:.2f}%  Δ={delta:+.2f}%", flush=True)

    # Condition B: FEATURE (Recipe D) — inject unit(v_19 + β·W_dec) at L19, sweep β×α
    print(f"\n── Condition B: FEATURE L{FEATURE_LAYER}/F{FEATURE_IDX} (Recipe D) ──", flush=True)
    for beta in BETAS:
        v_boost = v_feat_unit + (beta - 1.0) * (coeff * w_unit)
        v_boost = v_boost / v_boost.norm().clamp(min=1e-8)
        for alpha in ALPHAS:
            rk = f"feature_b{beta:g}_a{alpha:g}"
            if rk in all_results and all_results[rk].get("total", 0) > 0:
                r = all_results[rk]
                print(f"  [SKIP β={beta:g} α={alpha:g}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
                continue
            sv  = (v_boost * alpha).to(dtype).to(device)
            inj = LayerInjector(pt_model, FEATURE_LAYER, sv).install()
            try:
                res = eval_pt(pt_model, pt_proc, pt_tok, test_samples, device, inj)
            finally:
                inj.remove()
            delta = res["acc"] - base_acc
            all_results[rk] = {"acc": res["acc"], "delta": delta, "alpha": alpha, "beta": beta,
                                "condition": "feature", "layer": FEATURE_LAYER,
                                "feature": FEATURE_IDX, "coeff": coeff,
                                "correct": res["correct"], "total": res["total"],
                                "per_type": res["per_type"]}
            with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
            print(f"  [FEAT β={beta:g} α={alpha:g}] {res['acc']:.2f}%  Δ={delta:+.2f}%", flush=True)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"DocVQA CAA Steering  —  pt-448 base {base_acc:.2f}%")
    print(f"{'='*70}")
    print(f"\nCondition A — MIDDLE (L{MIDDLE_LAYER}):")
    print(f"  {'α':>6}  {'acc':>8}  {'Δ':>8}")
    best_mid = max((v for k, v in all_results.items() if k.startswith("middle_")),
                   key=lambda x: x.get("delta", -999), default=None)
    for alpha in ALPHAS:
        rk = f"middle_a{alpha:g}"
        if rk not in all_results: continue
        r = all_results[rk]
        star = " ◄" if r == best_mid else ""
        print(f"  {alpha:>6g}  {r['acc']:>7.2f}%  {r['delta']:>+7.2f}%{star}")

    print(f"\nCondition B — FEATURE L{FEATURE_LAYER}/F{FEATURE_IDX} (cos={coeff:+.4f}):")
    print(f"  {'β':>5} {'α':>6}  {'acc':>8}  {'Δ':>8}")
    best_feat = max((v for k, v in all_results.items() if k.startswith("feature_")),
                    key=lambda x: x.get("delta", -999), default=None)
    for beta in BETAS:
        for alpha in ALPHAS:
            rk = f"feature_b{beta:g}_a{alpha:g}"
            if rk not in all_results: continue
            r = all_results[rk]
            star = " ◄" if r == best_feat else ""
            print(f"  {beta:>5g} {alpha:>6g}  {r['acc']:>7.2f}%  {r['delta']:>+7.2f}%{star}")

    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
