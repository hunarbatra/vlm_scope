#!/usr/bin/env python3
"""
Paired CAA steering with the "ocr" prompt — pt-448 inference uses just "ocr".

CAA built per-feature from R(F)∩all-1000 paired contrasts (cache from
build_paired_cache_ocr_prompt.py, prompts use "ocr\\n{ans}").

Recipes:
  A_MIDDLE   alpha · unit(v[L13]) @ L13                         (baseline)
  D_BB+WDEC  alpha · unit(v[L]) @ {17,19,20,21} + gamma·W_dec[F] @ lF (mmdiff)

Eval: lenient substring (OCR-Bench official).

Usage:
  CUDA_VISIBLE_DEVICES=X python3 -B scripts/caa_paired_recipe_ocr_prompt.py
"""
import os, sys, json, gc, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
SAE_CKPT_DIR = SAE_ROOT / "checkpoints"
PAIR_CACHE   = SAE_ROOT / "analysis_ocr/paired_cache_ocrprompt"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts"
OUT_DIR      = SAE_ROOT / "analysis_ocr/caa_paired_recipe_ocrprompt"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END     = 1000
MIDDLE_LAYER  = 13
BACKBONE_LAYERS = [17, 19, 20, 21]
NUM_LAYERS    = 26
ALPHAS        = [1.0, 2.0, 5.0, 10.0, 20.0]
GAMMAS        = [1.0, 3.0, 10.0]
MAX_NEW_TOKENS = 256

SPATIAL_FEATURES = [
    {"layer": 17, "feature": 13602, "key": "L17_F13602"},
    {"layer": 21, "feature": 9577,  "key": "L21_F9577"},
]


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def _correct_lenient(resp, gt):
    if resp is None: return False
    r = str(resp).strip().lower()
    g = str(gt).strip().lower()
    return bool(g) and (g in r or r in g)


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][fi].float()


def compute_paired_caa(layer, indices):
    pos = neg = None; n = 0
    for si in indices:
        p = PAIR_CACHE / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception: continue
        if "pos" not in d or "neg" not in d: continue
        if layer not in d["pos"] or layer not in d["neg"]: continue
        hp = d["pos"][layer].float(); hn = d["neg"][layer].float()
        pos = hp.clone() if pos is None else pos + hp
        neg = hn.clone() if neg is None else neg + hn
        n += 1
    if n == 0 or pos is None: return None, 0
    return (pos - neg) / n, n


def run_eval(tag, inject_pairs, test_indices, ds, model, processor, tok, device,
             base_acc, result_key, all_results, results_path):
    sys.path.insert(0, str(Path(__file__).parent))
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
        gt  = _parse_gt(ex.get("answer"))
        if img is None or not gt: continue
        hooks = []
        try:
            img = img.convert("RGB")
            iids, attn, pv = process_vlm_inputs(
                img, "ocr", processor, model, device=device)
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
            c += int(_correct_lenient(resp, gt))
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

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("=" * 80)
    print("PAIRED CAA — 'ocr' prompt — pt-448 OCR-Bench R(F)∩all-1000")
    print("=" * 80, flush=True)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")
    test_indices = list(range(len(ds)))
    print(f"  total samples for eval={len(test_indices)} (R(F) ∩ all 1000)", flush=True)

    # R(F) over all-1000
    rF = {}
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists(): continue
        ad = json.load(open(ap))
        ak = {int(x) for x, v in ad.get("acts", {}).items() if v > 0}
        rF[k] = [v for v in test_indices if v in ak]
        print(f"  R({k}) over all 1000: n={len(rF[k])}", flush=True)

    # Per-feature paired CAA from R(F)
    needed_layers = sorted(set(BACKBONE_LAYERS) | {MIDDLE_LAYER}
                           | {sf["layer"] for sf in SPATIAL_FEATURES})
    print(f"\n[INFO] Computing per-feature paired CAA at {len(needed_layers)} layers", flush=True)
    caa_unit_by_feat = {}
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        if k not in rF or len(rF[k]) < 5: continue
        caa_unit_by_feat[k] = {}
        for l in needed_layers:
            v, n = compute_paired_caa(l, indices=rF[k])
            if v is not None:
                caa_unit_by_feat[k][l] = v / v.norm().clamp(min=1e-8)
                print(f"  [{k}] L{l}: paired n={n} raw_norm={v.norm():.3f}", flush=True)
    if not caa_unit_by_feat:
        print("[ERROR] No CAA. Did the cache build?")
        return
    gc.collect()

    # Load pt-448
    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tok = proc.tokenizer
    dtype = next(mdl.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Per-feature R(F) baselines (lenient, "ocr" prompt)
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        if k not in rF: continue
        bk = f"{k}_rF_base"
        if bk in all_results: continue
        bc = bt = 0
        for si in rF[k]:
            ex = ds[si]
            img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            if img is None or not gt: continue
            try:
                img = img.convert("RGB")
                from utils import process_vlm_inputs
                iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
                with torch.no_grad():
                    out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                           pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False, use_cache=True)
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                bt += 1; bc += int(_correct_lenient(resp, gt))
            except Exception: continue
        all_results[bk] = {"acc": bc/max(bt,1)*100, "n": bt}
        print(f"  [{k}] rF base: {all_results[bk]['acc']:.2f}% (n={bt})", flush=True)
    with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # Recipes
    for sf in SPATIAL_FEATURES:
        key, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if key not in rF or key not in caa_unit_by_feat: continue
        vis_F = rF[key]
        if len(vis_F) < 5: continue
        base_rF = all_results[f"{key}_rF_base"]["acc"]
        caa_unit = caa_unit_by_feat[key]
        print(f"\n--- {key} n={len(vis_F)} base={base_rF:.2f}% layer=L{lF} ---", flush=True)

        # A_MIDDLE
        if MIDDLE_LAYER in caa_unit:
            for a in ALPHAS:
                sv = (caa_unit[MIDDLE_LAYER] * a).to(dtype).to(device)
                rk = f"{key}_A_middle_a{a}"
                all_results = run_eval(
                    f"{key}/A_MIDDLE α={a}", [(MIDDLE_LAYER, sv)],
                    vis_F, ds, mdl, proc, tok, device,
                    base_rF, rk, all_results, results_path)

        # D_BB+WDEC
        w_dec = _load_wdec(lF, fi)
        if w_dec is not None:
            for a in ALPHAS:
                for g in GAMMAS:
                    inject = []
                    for l in BACKBONE_LAYERS:
                        if l not in caa_unit: continue
                        if l == lF:
                            sv = (caa_unit[l] * a + w_dec * g).to(dtype).to(device)
                        else:
                            sv = (caa_unit[l] * a).to(dtype).to(device)
                        inject.append((l, sv))
                    rk = f"{key}_D_bb_wdec_a{a}_g{g}"
                    all_results = run_eval(
                        f"{key}/D_BB+WDEC α={a} γ={g}", inject,
                        vis_F, ds, mdl, proc, tok, device,
                        base_rF, rk, all_results, results_path)

        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}\nSUMMARY — 'ocr' prompt, per-feature CAA, lenient metric\n{'='*90}", flush=True)
    print(f"  {'Feature':<14} {'L':<3} {'N':>4} {'Base':>8}  "
          f"{'A MIDDLE':>22}  {'D BB+WDEC (mmdiff)':>26}")
    print("  " + "-"*90)
    for sf in SPATIAL_FEATURES:
        k = sf["key"]; lF = sf["layer"]
        bk = f"{k}_rF_base"
        if bk not in all_results: continue
        n = all_results[bk]["n"]; ba = all_results[bk]["acc"]
        def best_a(pref):
            best = None
            for a in ALPHAS:
                r = all_results.get(f"{pref}_a{a}")
                if r and (best is None or r["delta"] > best[0]):
                    best = (r["delta"], a, r["acc"])
            return best
        def best_ag(pref):
            best = None
            for a in ALPHAS:
                for g in GAMMAS:
                    r = all_results.get(f"{pref}_a{a}_g{g}")
                    if r and (best is None or r["delta"] > best[0]):
                        best = (r["delta"], a, g, r["acc"])
            return best
        aA = best_a(f"{k}_A_middle")
        aD = best_ag(f"{k}_D_bb_wdec")
        fmt  = lambda x: f"{x[2]:.2f}% Δ={x[0]:+.2f} (a{x[1]:g})" if x else "—"
        fmtG = lambda x: f"{x[3]:.2f}% Δ={x[0]:+.2f} (a{x[1]:g}/g{x[2]:g})" if x else "—"
        print(f"  {k:<14} L{lF:<2} {n:>4} {ba:>7.2f}%  {fmt(aA):>22}  {fmtG(aD):>26}")
    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
