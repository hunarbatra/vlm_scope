#!/usr/bin/env python3
"""
MathVerse Step 8: CAA steering recipe eval — mix-src → mix-448, top math features.

Exactly mirrors caa_recipe_compare_mix_to_pt_devtest.py but for MathVerse:
- Source hidden states: mix-448 (MCQ prompts, answer en <prompt>)
- Eval model:          mix-448 (MCQ logit comparison — no generate needed)
- Correctness:         argmax over A/B/C/D logits
- Recipes A/C/D/E/F   (B excluded — not useful for small layer sets)

For each top feature F at layer lF:
  Eval on R(F)∩test with 6 recipes × alpha/gamma sweeps.

Output: analysis_mathverse/caa_recipe_results/results.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -u mathverse_step5_caa_recipe.py
    (Run after all prior steps complete.)
"""
import os, sys, json, gc, re, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL  = "google/paligemma2-3b-mix-448"
SAE_ROOT   = Path("/data1/vlm_scope_sae_mix448_textonly")
CACHE_DIR  = SAE_ROOT / "analysis_mathverse/mix_hidden"
SAE_ACTS_DIR = SAE_ROOT / "analysis_mathverse/sae_acts"
ABL_PATH   = SAE_ROOT / "analysis_mathverse/ablation_results.json"
CKPT_DIR   = SAE_ROOT / "checkpoints"
OUT_DIR    = SAE_ROOT / "analysis_mathverse/caa_recipe_results"

TRAIN_END  = 344
MIDDLE_LAYER = 13
NUM_LAYERS   = 26
# Include negative alphas — features with OR<1 fire on incorrect samples,
# so injecting the CAA vector in the negative direction may help
ALPHAS       = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0]
GAMMAS       = [1.0, 3.0, 10.0]
TOP_N        = 8

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def _get_choice_ids(tok):
    choice_ids = {}
    for letter in "ABCD":
        ids_ = set()
        for form in [letter, f" {letter}", f"({letter})", f" ({letter})"]:
            try:
                t = tok.encode(form, add_special_tokens=False)
                if t: ids_.add(t[0])
            except: pass
        choice_ids[letter] = ids_
    return choice_ids


def _predict_mcq(logits, choice_ids):
    p = torch.softmax(logits.float(), dim=-1)
    scores = {l: sum(p[i].item() for i in ids_) for l, ids_ in choice_ids.items()}
    return max(scores, key=scores.get)


def _parse_gt(s):
    m = re.search(r'([A-D])', str(s))
    return m.group(1) if m else None


def compute_meanpool_caa(correct_map, layer, train_idx):
    pos = neg = None; pn = nn = 0
    for si in train_idx:
        p = CACHE_DIR / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception: continue
        if layer not in d: continue
        v = d[layer].float()
        if correct_map.get(si, False):
            pos = v.clone() if pos is None else pos + v; pn += 1
        else:
            neg = v.clone() if neg is None else neg + v; nn += 1
    if pos is None or neg is None or pn == 0 or nn == 0: return None
    return pos/pn - neg/nn


def _load_wdec(layer, feature_idx):
    p = CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][feature_idx].float()


def run_eval(tag, inject_pairs, test_vis, ds, choice_ids, base_acc,
             result_key, all_results, results_path, model, proc, device):
    from utils import process_vlm_inputs, get_image_token_positions
    if result_key in all_results and all_results[result_key].get("n", 0) > 0:
        r = all_results[result_key]
        print(f"  [SKIP {tag}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
        return all_results

    img_end_r = [0]
    def make_hook(sv_):
        def f(m, i, o):
            ie = img_end_r[0]
            h = o[0] if isinstance(o, tuple) else o
            h[0, ie:] = h[0, ie:] + sv_.unsqueeze(0)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        return f

    c = t = 0
    for si in test_vis:
        ex = ds[si]
        img = ex.get("image")
        if img is None: continue
        hooks = []
        try:
            img = img.convert("RGB")
            iids, attn, pv = process_vlm_inputs(
                img, f"answer en {ex['prompt']}", proc, model, device=device)
            _, img_end_r[0] = get_image_token_positions(iids)
            for (l, sv) in inject_pairs:
                hooks.append(model.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
            with torch.no_grad():
                out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
            for h in hooks:
                try: h.remove()
                except: pass
            pred = _predict_mcq(out.logits[0, -1, :], choice_ids)
            gt   = _parse_gt(ex.get("answer", ""))
            if gt:
                t += 1; c += int(pred == gt)
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
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("=" * 80)
    print("MATHVERSE CAA RECIPE EVAL — mix-src → mix-448, R(F)∩test subsets")
    print("=" * 80, flush=True)

    # Get top features from ablation — both ends:
    # - most-negative drop: ablating hurts → amplify with positive alpha
    # - most-positive drop: ablating helps → suppress with negative alpha
    abl = json.load(open(ABL_PATH))
    drops = [(k, v) for k, v in abl.items() if k != "base" and "drop" in v]
    drops.sort(key=lambda x: x[1]["drop"])
    # Bottom N (negative drop = amplify) + Top N (positive drop = suppress)
    n_each = TOP_N // 2
    bottom = drops[:n_each]          # most negative → amplify
    top    = drops[-(n_each):]       # most positive → suppress
    # Deduplicate and combine, keeping the most extreme on both ends
    seen = set()
    combined = []
    for k, v in (bottom + top):
        if k not in seen:
            seen.add(k)
            combined.append((k, v))
    SPATIAL_FEATURES = [
        {"layer": v["layer"], "feature": v["feature"], "key": k,
         "suppress": v["drop"] > 0}  # True = suppress (use negative alpha)
        for k, v in combined
    ]
    print(f"[INFO] Steering targets (amplify=positive drop<0, suppress=positive drop>0):")
    for sf in SPATIAL_FEATURES:
        action = "SUPPRESS(neg α)" if sf["suppress"] else "AMPLIFY(pos α)"
        print(f"  {sf['key']}: drop={abl[sf['key']]['drop']:+.2f}%  → {action}", flush=True)

    print("[INFO] Loading MathVerse...", flush=True)
    ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")
    N  = len(ds)
    train_idx = list(range(TRAIN_END))
    test_idx  = list(range(TRAIN_END, N))

    # Load correctness labels
    corr_path = SAE_ROOT / "analysis_mathverse/correctness.json"
    corr_data = json.load(open(corr_path))
    correct   = {int(k): v for k, v in corr_data["correct"].items()}

    # Compute CAA vectors at all needed layers
    needed_layers = set([MIDDLE_LAYER]) | {sf["layer"] for sf in SPATIAL_FEATURES}
    # Also add all downstream layers for recipe B
    for sf in SPATIAL_FEATURES:
        for l in range(sf["layer"], NUM_LAYERS):
            needed_layers.add(l)
    needed_layers = sorted(needed_layers)

    print(f"[INFO] Computing CAA at {len(needed_layers)} layers...", flush=True)
    caa_raw, caa_unit = {}, {}
    for l in needed_layers:
        v = compute_meanpool_caa(correct, l, train_idx)
        if v is not None:
            caa_raw[l]  = v
            caa_unit[l] = v / v.norm().clamp(min=1e-8)
    for l in sorted(caa_unit.keys()):
        print(f"  L{l}: norm={caa_raw[l].norm():.3f}", flush=True)
    gc.collect()

    # Build backbone layers = union of all feature layers
    BACKBONE_LAYERS = sorted({sf["layer"] for sf in SPATIAL_FEATURES} | {MIDDLE_LAYER})

    # R(F) subsets
    rF = {}
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists():
            print(f"  [WARN] No acts file for {k}", flush=True); continue
        ad = json.load(open(ap))
        ak = {int(x) for x in ad["acts"] if ad["acts"][x] > 0}
        tvis = [si for si in test_idx if si in ak]
        rF[k] = {"vis": tvis, "labels": [correct.get(si, False) for si in tvis]}
        print(f"  {k}: R(F)∩test = {len(tvis)}/{len(test_idx)}", flush=True)

    print(f"\n[INFO] Loading {MIX_MODEL}...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    choice_ids = _get_choice_ids(proc.tokenizer)
    dtype = next(model.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Per-feature R(F)∩test baseline
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        if k not in rF: continue
        bk = f"{k}_rF_base"
        if bk in all_results: continue
        bc = bt = 0
        for si in rF[k]["vis"]:
            ex = ds[si]; img = ex.get("image")
            if img is None: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(
                    img, f"answer en {ex['prompt']}", proc, model, device=device)
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                pred = _predict_mcq(out.logits[0, -1, :], choice_ids)
                gt   = _parse_gt(ex.get("answer", ""))
                if gt: bt += 1; bc += int(pred == gt)
            except Exception: pass
        base_rf = bc / max(bt, 1) * 100
        all_results[bk] = {"acc": base_rf, "n": bt}
        print(f"  [{k}] R(F)∩test base: {base_rf:.2f}% (n={bt})", flush=True)
    with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # ===== 6 Recipes per feature =====
    for sf in SPATIAL_FEATURES:
        key, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if key not in rF: continue
        vis_F = rF[key]["vis"]
        if len(vis_F) < 3:
            print(f"[{key}] too few R(F)∩test ({len(vis_F)}) — skip", flush=True); continue
        base_rF = all_results[f"{key}_rF_base"]["acc"]

        print(f"\n--- {key}  n={len(vis_F)}  base={base_rF:.2f}%  L{lF} ---", flush=True)

        # Recipe A: MIDDLE
        if MIDDLE_LAYER in caa_unit:
            for alpha in ALPHAS:
                sv = (caa_unit[MIDDLE_LAYER] * alpha).to(dtype).to(device)
                all_results = run_eval(
                    f"{key}/A_MIDDLE α={alpha}", [(MIDDLE_LAYER, sv)],
                    vis_F, ds, choice_ids, base_rF,
                    f"{key}_A_middle_a{alpha}", all_results, results_path,
                    model, proc, device)

        # Recipe B: CAA_SAE_DOWN (lF..25)
        if lF in caa_unit:
            for alpha in ALPHAS:
                sv = (caa_unit[lF] * alpha).to(dtype).to(device)
                inject = [(l, sv) for l in range(lF, NUM_LAYERS) if l in caa_unit]
                all_results = run_eval(
                    f"{key}/B_SAE_DOWN α={alpha}", inject,
                    vis_F, ds, choice_ids, base_rF,
                    f"{key}_B_sae_down_a{alpha}", all_results, results_path,
                    model, proc, device)

        # Recipe C: BACKBONE
        for alpha in ALPHAS:
            inject = [(l, (caa_unit[l]*alpha).to(dtype).to(device))
                      for l in BACKBONE_LAYERS if l in caa_unit]
            all_results = run_eval(
                f"{key}/C_BACKBONE α={alpha}", inject,
                vis_F, ds, choice_ids, base_rF,
                f"{key}_C_backbone_a{alpha}", all_results, results_path,
                model, proc, device)

        # Recipe D: BACKBONE + W_dec
        w_dec = _load_wdec(lF, fi)
        if w_dec is not None:
            for alpha in ALPHAS:
                for gamma in GAMMAS:
                    inject = []
                    for l in BACKBONE_LAYERS:
                        if l not in caa_unit: continue
                        if l == lF:
                            sv = (caa_unit[l]*alpha + w_dec*gamma).to(dtype).to(device)
                        else:
                            sv = (caa_unit[l]*alpha).to(dtype).to(device)
                        inject.append((l, sv))
                    all_results = run_eval(
                        f"{key}/D_BB+WDEC α={alpha} γ={gamma}", inject,
                        vis_F, ds, choice_ids, base_rF,
                        f"{key}_D_bb_wdec_a{alpha}_g{gamma}", all_results, results_path,
                        model, proc, device)

        # Recipe E: SPATIAL_LAYER
        if lF in caa_unit:
            for alpha in ALPHAS:
                sv = (caa_unit[lF]*alpha).to(dtype).to(device)
                all_results = run_eval(
                    f"{key}/E_SPATIAL α={alpha}", [(lF, sv)],
                    vis_F, ds, choice_ids, base_rF,
                    f"{key}_E_spatial_a{alpha}", all_results, results_path,
                    model, proc, device)

        # Recipe F: SPATIAL + W_dec
        if lF in caa_unit and w_dec is not None:
            for alpha in ALPHAS:
                for gamma in GAMMAS:
                    sv = (caa_unit[lF]*alpha + w_dec*gamma).to(dtype).to(device)
                    all_results = run_eval(
                        f"{key}/F_SPAT+WDEC α={alpha} γ={gamma}", [(lF, sv)],
                        vis_F, ds, choice_ids, base_rF,
                        f"{key}_F_spatial_wdec_a{alpha}_g{gamma}", all_results, results_path,
                        model, proc, device)

        gc.collect(); torch.cuda.empty_cache()

    # ===== Summary =====
    print(f"\n{'='*120}")
    print("SUMMARY — R(F)∩test per-feature, best Δ per recipe")
    print(f"{'='*120}", flush=True)
    print(f"  {'Feature':<14} {'N':>4} {'Base':>7}  "
          f"{'A MIDDLE':>10}  {'B SAE_DOWN':>11}  {'C BACKBONE':>11}  "
          f"{'D BB+WDEC':>11}  {'E SPATIAL':>10}  {'F SPAT+WDEC':>12}")
    print("  " + "-"*120)
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        bk = f"{k}_rF_base"
        if bk not in all_results: continue
        n  = all_results[bk]["n"]
        ba = all_results[bk]["acc"]

        def best_a(pfx):
            b = None
            for a in ALPHAS:
                r = all_results.get(f"{pfx}_a{a}")
                if r and (b is None or r["delta"] > b[0]): b = (r["delta"], a)
            return b

        def best_ag(pfx):
            b = None
            for a in ALPHAS:
                for g in GAMMAS:
                    r = all_results.get(f"{pfx}_a{a}_g{g}")
                    if r and (b is None or r["delta"] > b[0]): b = (r["delta"], a, g)
            return b

        fmt  = lambda x: f"{x[0]:+.2f}%" if x else "—"
        fmtG = lambda x: f"{x[0]:+.2f}%(γ{x[2]:g})" if x else "—"
        aA = best_a(f"{k}_A_middle");     aB = best_a(f"{k}_B_sae_down")
        aC = best_a(f"{k}_C_backbone");   aD = best_ag(f"{k}_D_bb_wdec")
        aE = best_a(f"{k}_E_spatial");    aF = best_ag(f"{k}_F_spatial_wdec")
        print(f"  {k:<14} {n:>4} {ba:>6.2f}%  "
              f"{fmt(aA):>10}  {fmt(aB):>11}  {fmt(aC):>11}  "
              f"{fmtG(aD):>11}  {fmt(aE):>10}  {fmtG(aF):>12}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
