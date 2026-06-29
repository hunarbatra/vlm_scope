#!/usr/bin/env python3
"""
Recipe comparison — mix-src → pt-448, per-feature R(F)∩test subsets for OCR-Bench.

Mirrors caa_recipe_compare_mix_to_pt_devtest.py exactly, but for OCR-Bench:
  - MEANPOOL_DIR: analysis_ocr/mix_hidden_cache/vi_{si:05d}.pt
  - SAE_ACTS_DIR: analysis_ocr/sae_acts/acts_L{L}_F{F}.json
  - Labels: "correct" bool stored in each vi_NNNNN.pt
  - Correctness at eval: model response contains/is contained by gt_answer
  - Train: indices 0..799; Test: indices 800..999

Recipes evaluated:
  A. MIDDLE              α · unit(v_CAA[L13]) @ L13
  B. CAA_SAE_DOWN        α · unit(v_CAA[lF]) @ lF..25
  C. BACKBONE            α · unit(v_CAA[L]) @ each L in {17,19,20,21}
  D. BACKBONE+WDEC       (C) plus γ · W_dec[F] at lF only
  E. SPATIAL_LAYER       α · unit(v_CAA[lF]) @ lF only
  F. SPATIAL+WDEC        (E) plus γ · W_dec[F] at lF

Usage:
    CUDA_VISIBLE_DEVICES=X python3 caa_recipe_ocr.py
"""
import os, sys, json, gc, warnings
from pathlib import Path

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
SAE_CKPT_DIR = SAE_ROOT / "checkpoints"
MEANPOOL_DIR = SAE_ROOT / "analysis_ocr/mix_hidden_cache"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts"
OUT_DIR      = SAE_ROOT / "analysis_ocr/caa_recipe_results"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END     = 800
MIDDLE_LAYER  = 13
CACHED_LAYERS = [17, 19, 20, 21]  # OCR backbone (feature layers)
NUM_LAYERS    = 26
ALPHAS        = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
GAMMAS        = [1.0, 3.0, 10.0]
MAX_NEW_TOKENS = 64

# OCR features confirmed via steer_eval_ocr.py (top by Δ on full 1000-sample set)
SPATIAL_FEATURES = [
    {"layer": 17, "feature": 13602, "key": "L17_F13602"},
    {"layer": 19, "feature": 10089, "key": "L19_F10089"},
    {"layer": 19, "feature": 14093, "key": "L19_F14093"},
    {"layer": 20, "feature": 10687, "key": "L20_F10687"},
    {"layer": 21, "feature": 9577,  "key": "L21_F9577"},
]


def _correct_ocr(resp, gt):
    if resp is None: return False
    r = resp.strip().lower()
    g = str(gt).strip().lower()
    return bool(g) and (g in r or r in g)


def compute_meanpool_caa(train_indices, layer):
    """Label-aware CAA at `layer` from meanpool cache on train split."""
    pos = neg = None
    pn = nn = 0
    for si in train_indices:
        p = MEANPOOL_DIR / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if layer not in d: continue
        v = d[layer].float()
        if d.get("correct", False):
            pos = v.clone() if pos is None else pos + v
            pn += 1
        else:
            neg = v.clone() if neg is None else neg + v
            nn += 1
    if pos is None or neg is None or pn == 0 or nn == 0:
        return None
    print(f"  L{layer}: n_pos={pn} n_neg={nn}", flush=True)
    return pos / pn - neg / nn


def _load_wdec(layer, feature_idx):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][feature_idx].float()


def run_eval(tag, inject_pairs, test_indices, ds, model, processor, tok, device,
             base_acc, result_key, all_results, results_path):
    from utils import process_vlm_inputs, get_image_token_positions

    if result_key in all_results and all_results[result_key].get("n", 0) > 0:
        r = all_results[result_key]
        print(f"  [SKIP {tag}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
        return all_results

    img_end_r = [0]

    def make_hook(sv_):
        def f(m, inp, out):
            ie = img_end_r[0]
            h = out[0] if isinstance(out, tuple) else out
            h[0, ie:] = h[0, ie:] + sv_.unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return f

    c = t = 0
    for si in test_indices:
        ex  = ds[si]
        img = ex.get("image")
        q   = str(ex.get("question", "")).strip()
        gt  = str(ex.get("answer", "")).strip()
        if img is None or not q: continue
        hooks = []
        try:
            img = img.convert("RGB")
            iids, attn, pv = process_vlm_inputs(
                img, f"answer en {q}", processor, model, device=device)
            _, img_end_r[0] = get_image_token_positions(iids)
            for (l, sv) in inject_pairs:
                hooks.append(model.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=iids, attention_mask=attn, pixel_values=pv,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            for h in hooks:
                try: h.remove()
                except: pass
            resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
            t += 1
            c += int(_correct_ocr(resp, gt))
        except Exception:
            for h in hooks:
                try: h.remove()
                except: pass

    if t == 0: return all_results
    acc = c / t * 100
    delta = acc - base_acc
    all_results[result_key] = {"acc": acc, "delta": delta, "n": t}
    print(f"  [{tag}] {acc:.2f}%  Δ={delta:+.2f}%  ({c}/{t})", flush=True)
    with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
    return all_results


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("=" * 80)
    print("RECIPE COMPARE — mix-src → pt-448, OCR-Bench R(F)∩test subsets")
    print("=" * 80, flush=True)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")
    train_indices = list(range(TRAIN_END))
    test_indices  = list(range(TRAIN_END, len(ds)))
    print(f"  train={len(train_indices)} test={len(test_indices)}", flush=True)

    # Compute CAA vectors from per-sample hidden cache
    needed_layers = set(CACHED_LAYERS) | {sf["layer"] for sf in SPATIAL_FEATURES} | {MIDDLE_LAYER}
    for sf in SPATIAL_FEATURES:
        for l in range(sf["layer"], NUM_LAYERS):
            needed_layers.add(l)
    needed_layers = sorted(needed_layers)
    print(f"[INFO] Computing meanpool CAA at {len(needed_layers)} layers from {len(train_indices)} train samples...", flush=True)

    caa_raw  = {}
    caa_unit = {}
    for l in needed_layers:
        v = compute_meanpool_caa(train_indices, l)
        if v is not None:
            caa_raw[l]  = v
            caa_unit[l] = v / v.norm().clamp(min=1e-8)
    for l in sorted(caa_unit.keys()):
        print(f"  L{l}: raw norm={caa_raw[l].norm():.3f}", flush=True)
    gc.collect()

    # R(F) ∩ test subsets
    rF = {}
    for sf in SPATIAL_FEATURES:
        k  = sf["key"]
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists():
            print(f"  [WARN] missing {ap}", flush=True)
            continue
        ad  = json.load(open(ap))
        ak  = {int(x) for x in ad.get("acts", {}).keys() if ad["acts"][x] > 0}
        tsi = [si for si in test_indices if si in ak]
        rF[k] = {"indices": tsi}
        print(f"  {k}: R(F)∩test = {len(tsi)}", flush=True)

    # Load pt-448
    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    proc  = AutoProcessor.from_pretrained(PT_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok   = proc.tokenizer
    dtype = next(model.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Per-feature R(F)∩test baselines
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        if k not in rF: continue
        bk = f"{k}_rF_base"
        if bk in all_results:
            print(f"  [{k}] rF base: {all_results[bk]['acc']:.2f}% (n={all_results[bk]['n']}) [cached]", flush=True)
            continue
        bc = bt = 0
        for si in rF[k]["indices"]:
            ex  = ds[si]
            img = ex.get("image")
            q   = str(ex.get("question", "")).strip()
            gt  = str(ex.get("answer", "")).strip()
            if img is None or not q: continue
            try:
                from utils import process_vlm_inputs, get_image_token_positions
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(
                    img, f"answer en {q}", proc, model, device=device)
                with torch.no_grad():
                    out_ids = model.generate(
                        input_ids=iids, attention_mask=attn, pixel_values=pv,
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                bt += 1
                bc += int(_correct_ocr(resp, gt))
            except Exception:
                continue
        all_results[bk] = {"acc": bc / max(bt, 1) * 100, "n": bt}
        print(f"  [{k}] rF base: {all_results[bk]['acc']:.2f}% (n={bt})", flush=True)
    with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # ==== 6 recipes per feature ====
    for sf in SPATIAL_FEATURES:
        key, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if key not in rF: continue
        test_si = rF[key]["indices"]
        if len(test_si) < 5:
            print(f"[{key}] too few samples ({len(test_si)}) — skip", flush=True)
            continue
        base_rF = all_results[f"{key}_rF_base"]["acc"]

        print(f"\n--- {key}  n={len(test_si)}  base={base_rF:.2f}%  layer=L{lF} ---", flush=True)

        # Recipe A: MIDDLE
        if MIDDLE_LAYER in caa_unit:
            for alpha in ALPHAS:
                sv = (caa_unit[MIDDLE_LAYER] * alpha).to(dtype).to(device)
                rkey = f"{key}_A_middle_a{alpha}"
                all_results = run_eval(
                    f"{key}/A_MIDDLE α={alpha}", [(MIDDLE_LAYER, sv)],
                    test_si, ds, model, proc, tok, device, base_rF,
                    rkey, all_results, results_path)

        # Recipe B: CAA_SAE_DOWN
        if lF in caa_unit:
            v_lF = caa_unit[lF]
            for alpha in ALPHAS:
                sv = (v_lF * alpha).to(dtype).to(device)
                inject = [(l, sv) for l in range(lF, NUM_LAYERS)]
                rkey = f"{key}_B_sae_down_a{alpha}"
                all_results = run_eval(
                    f"{key}/B_SAE_DOWN α={alpha}", inject,
                    test_si, ds, model, proc, tok, device, base_rF,
                    rkey, all_results, results_path)

        # Recipe C: BACKBONE
        for alpha in ALPHAS:
            inject = [(l, (caa_unit[l] * alpha).to(dtype).to(device))
                      for l in CACHED_LAYERS if l in caa_unit]
            rkey = f"{key}_C_backbone_a{alpha}"
            all_results = run_eval(
                f"{key}/C_BACKBONE α={alpha}", inject,
                test_si, ds, model, proc, tok, device, base_rF,
                rkey, all_results, results_path)

        # Recipe D: BACKBONE + W_dec
        w_dec = _load_wdec(lF, fi)
        if w_dec is not None:
            for alpha in ALPHAS:
                for gamma in GAMMAS:
                    inject = []
                    for l in CACHED_LAYERS:
                        if l not in caa_unit: continue
                        if l == lF:
                            sv = (caa_unit[l] * alpha + w_dec * gamma).to(dtype).to(device)
                        else:
                            sv = (caa_unit[l] * alpha).to(dtype).to(device)
                        inject.append((l, sv))
                    rkey = f"{key}_D_bb_wdec_a{alpha}_g{gamma}"
                    all_results = run_eval(
                        f"{key}/D_BB+WDEC α={alpha} γ={gamma}", inject,
                        test_si, ds, model, proc, tok, device, base_rF,
                        rkey, all_results, results_path)

        # Recipe E: SPATIAL_LAYER
        if lF in caa_unit:
            for alpha in ALPHAS:
                sv = (caa_unit[lF] * alpha).to(dtype).to(device)
                rkey = f"{key}_E_spatial_a{alpha}"
                all_results = run_eval(
                    f"{key}/E_SPATIAL_LAYER α={alpha}", [(lF, sv)],
                    test_si, ds, model, proc, tok, device, base_rF,
                    rkey, all_results, results_path)

        # Recipe F: SPATIAL + W_dec
        if lF in caa_unit and w_dec is not None:
            for alpha in ALPHAS:
                for gamma in GAMMAS:
                    sv = (caa_unit[lF] * alpha + w_dec * gamma).to(dtype).to(device)
                    rkey = f"{key}_F_spatial_wdec_a{alpha}_g{gamma}"
                    all_results = run_eval(
                        f"{key}/F_SPAT+WDEC α={alpha} γ={gamma}", [(lF, sv)],
                        test_si, ds, model, proc, tok, device, base_rF,
                        rkey, all_results, results_path)

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*120}\nSUMMARY — OCR-Bench R(F)∩test per-feature, best Δ per recipe\n{'='*120}", flush=True)
    print(f"  {'Feature':<14} {'Layer':<5} {'N':>4} {'Base':>7}  "
          f"{'A MIDDLE':>10}  {'B SAE_DOWN':>11}  {'C BACKBONE':>11}  {'D BB+WDEC (γ)':>14}  "
          f"{'E SPATIAL':>10}  {'F SPAT+WDEC (γ)':>16}")
    print("  " + "-"*120)
    for sf in SPATIAL_FEATURES:
        k = sf["key"]; lF = sf["layer"]
        bk = f"{k}_rF_base"
        if bk not in all_results: continue
        n  = all_results[bk]["n"]
        ba = all_results[bk]["acc"]

        def best_alpha(prefix):
            best = None
            for a in ALPHAS:
                r = all_results.get(f"{prefix}_a{a}")
                if r and (best is None or r["delta"] > best[0]):
                    best = (r["delta"], a)
            return best

        def best_ag(pref):
            best = None
            for a in ALPHAS:
                for g in GAMMAS:
                    r = all_results.get(f"{pref}_a{a}_g{g}")
                    if r and (best is None or r["delta"] > best[0]):
                        best = (r["delta"], a, g)
            return best

        aA = best_alpha(f"{k}_A_middle")
        aB = best_alpha(f"{k}_B_sae_down")
        aC = best_alpha(f"{k}_C_backbone")
        aD = best_ag(f"{k}_D_bb_wdec")
        aE = best_alpha(f"{k}_E_spatial")
        aF = best_ag(f"{k}_F_spatial_wdec")

        fmt  = lambda x: f"{x[0]:+.2f}%" if x else "—"
        fmtG = lambda x: f"{x[0]:+.2f}% (γ={x[2]:g})" if x else "—"
        print(f"  {k:<14} L{lF:<4} {n:>4} {ba:>6.2f}%  "
              f"{fmt(aA):>10}  {fmt(aB):>11}  {fmt(aC):>11}  {fmtG(aD):>14}  "
              f"{fmt(aE):>10}  {fmtG(aF):>16}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
