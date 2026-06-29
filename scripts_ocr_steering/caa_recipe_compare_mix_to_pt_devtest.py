#!/usr/bin/env python3
"""
Recipe comparison — mix-src → pt-448, per-feature R(F)∩test subsets.

For each of the 10 spatial SAE features F at layer lF, evaluate 6 recipes:

  A. MIDDLE              α · unit(v_meanpool[L13]) @ L13
                         Plain Rimsky middle-layer CAA, shared vector. Baseline.

  B. CAA_SAE_DOWN        α · unit(v_meanpool[lF]) @ each of {lF, lF+1, ..., 25}
                         The recipe that gave +15.38% on "ahead of".

  C. BACKBONE            α · unit(v_meanpool[L]) @ each L in {4,6,9,11,12,13,14,15}
                         All-8 multi-layer CAA (+4.42% full test / +10-12% R(F)).

  D. BACKBONE+WDEC       (C) plus γ · W_dec[F] at lF only
                         The +15.62% recipe for L12_F2257 "facing".

  E. SPATIAL_LAYER       α · unit(v_meanpool[lF]) @ lF only
                         Single-layer at the feature's SAE layer. Isolates layer
                         effect (vs MIDDLE at L13) without spreading downstream.

  F. SPATIAL+WDEC        (E) plus γ · W_dec[F] at lF (same layer)
                         Single-layer spatial CAA + feature significance.
                         Cleanest test of "feature direction helps at lF".

Uses the MEANPOOL cache (pt448_hidden_delta/mix_hidden/, built from mix-448).
Vectors UNIT-NORMED, label-aware, α·unit convention.

Extraction: train+dev (samples 0..8776). Evaluation: R(F) ∩ test (8777..10971).

Alphas: A,B,C sweep α ∈ {0.5, 1.0, 2.0, 5.0}.
                D sweeps γ ∈ {1, 3, 10} at each α.

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_recipe_compare_mix_to_pt.py
"""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

PT_MODEL = "google/paligemma2-3b-pt-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_recipe_compare_mix_to_pt_devtest")
IMAGE_CACHE  = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 7680
MIDDLE_LAYER   = 13
CACHED_LAYERS  = [4, 6, 9, 11, 12, 13, 14, 15]   # for BACKBONE
NUM_LAYERS     = 26                                # PaliGemma2 LM layers 0..25
ALPHAS         = [0.5, 1.0, 2.0, 5.0]
GAMMAS         = [1.0, 3.0, 10.0]

SPATIAL_FEATURES = [
    {"layer": 9,  "feature": 387,   "key": "L9_F387"},
    {"layer": 14, "feature": 10561, "key": "L14_F10561"},
    {"layer": 11, "feature": 12278, "key": "L11_F12278"},
    {"layer": 9,  "feature": 7540,  "key": "L9_F7540"},
    {"layer": 4,  "feature": 14233, "key": "L4_F14233"},
    {"layer": 6,  "feature": 7539,  "key": "L6_F7539"},
    {"layer": 11, "feature": 9639,  "key": "L11_F9639"},
    {"layer": 13, "feature": 15219, "key": "L13_F15219"},
    {"layer": 15, "feature": 220,   "key": "L15_F220"},
    {"layer": 12, "feature": 2257,  "key": "L12_F2257"},
]


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    y, n = set(), set()
    for t in [" Yes","Yes"," yes","YES"]:
        tt = tok.encode(t, add_special_tokens=False)
        if tt: y.add(tt[0])
    for t in [" No","No"," no","NO"]:
        tt = tok.encode(t, add_special_tokens=False)
        if tt: n.add(tt[0])
    o = y & n; y -= o; n -= o
    return y, n

def _predict(logits, yids, nids):
    p = torch.softmax(logits.float(), dim=-1)
    y = p[list(yids)].sum().item() if yids else 1e-9
    nn = p[list(nids)].sum().item() if nids else 1e-9
    return 1 if (y/(y+nn) if y+nn > 0 else 0.5) > 0.5 else 0

def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(cp, "JPEG"); return img
    except Exception: return None

def _load_wdec(layer, feature_idx):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][feature_idx].float()


def compute_meanpool_caa(vsr_labels, layer):
    """Label-aware CAA at `layer` from meanpool cache on train+dev."""
    pos = neg = None; pn = nn = 0
    for vi in range(TRAIN_END):
        p = MEANPOOL_DIR / f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception: continue
        if layer not in d: continue
        v = d[layer].float()
        if int(vsr_labels[vi]) == 1:
            pos = v.clone() if pos is None else pos + v; pn += 1
        else:
            neg = v.clone() if neg is None else neg + v; nn += 1
    if pos is None or neg is None: return None
    return pos/pn - neg/nn


def run_eval(tag, inject_pairs, test_vis, test_labels, base_acc,
             result_key, all_results, results_path,
             model, processor, yes_ids, no_ids, device, vsr_all):
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
    for vi, lbl in zip(test_vis, test_labels):
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        pt = _build_vsr_prompt(str(ex.get("caption", "")))
        hooks = []
        try:
            iids, attn, pv = process_vlm_inputs(img, pt, processor, model, device=device)
            _, img_end_r[0] = get_image_token_positions(iids)
            for (l, sv) in inject_pairs:
                hooks.append(model.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
            with torch.no_grad():
                out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
            for h in hooks:
                try: h.remove()
                except: pass
            pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
            t += 1; c += int(pred == lbl)
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
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("=" * 80)
    print("RECIPE COMPARE — mix-src → pt-448, 4 recipes × 10 features on R(F)∩test")
    print("=" * 80, flush=True)

    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s: f"{s}.jsonl"}, split=s)
        for s in ["train", "dev", "test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label", 0)) for vi in range(len(vsr_all))]
    test_vis_full = list(range(7680, len(vsr_all)))

    # Compute CAA unit vectors at every cached layer + each unique feature layer
    needed_layers = set(CACHED_LAYERS) | {sf["layer"] for sf in SPATIAL_FEATURES} | {MIDDLE_LAYER}
    # For CAA_SAE_DOWN we also need CAA at every downstream layer lF..25
    for sf in SPATIAL_FEATURES:
        for l in range(sf["layer"], NUM_LAYERS):
            needed_layers.add(l)
    needed_layers = sorted(needed_layers)
    print(f"[INFO] Computing meanpool CAA at {len(needed_layers)} layers...", flush=True)

    caa_raw = {}
    caa_unit = {}
    for l in needed_layers:
        v = compute_meanpool_caa(vsr_labels, l)
        if v is not None:
            caa_raw[l] = v
            caa_unit[l] = v / v.norm().clamp(min=1e-8)
    for l in sorted(caa_unit.keys()):
        print(f"  L{l}: raw norm={caa_raw[l].norm():.3f}", flush=True)
    gc.collect()

    # R(F)∩test subsets
    rF = {}
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists(): continue
        ad = json.load(open(ap))
        ak = {int(x) for x in ad.get("acts", {}).keys()}
        tvis = [v for v in test_vis_full if v in ak]
        rF[k] = {
            "vis": tvis,
            "labels": [vsr_labels[v] for v in tvis],
            "relations": ad.get("relations", []),
        }

    # Load model
    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)
    dtype = next(mdl.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Per-feature R(F)∩test baselines
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        if k not in rF: continue
        bk = f"{k}_rF_base"
        if bk in all_results: continue
        bc = bt = 0
        for vi, lbl in zip(rF[k]["vis"], rF[k]["labels"]):
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            pt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                iids, attn, pv = process_vlm_inputs(img, pt, proc, mdl, device=device)
                with torch.no_grad():
                    out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv)
                pred = _predict(out.logits[0, -1, :], yids, nids)
                bt += 1; bc += int(pred == lbl)
            except Exception: continue
        all_results[bk] = {"acc": bc/max(bt,1)*100, "n": bt, "relations": rF[k]["relations"]}
        print(f"  [{k}] rF base: {all_results[bk]['acc']:.2f}% (n={bt})", flush=True)
    with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # ==== 4 recipes per feature ====
    for sf in SPATIAL_FEATURES:
        key, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if key not in rF: continue
        vis_F = rF[key]["vis"]; labels_F = rF[key]["labels"]
        if len(vis_F) < 5:
            print(f"[{key}] too few samples ({len(vis_F)}) — skip", flush=True); continue
        base_rF = all_results[f"{key}_rF_base"]["acc"]

        print(f"\n--- {key}  n={len(vis_F)}  base={base_rF:.2f}%  layer=L{lF}  relations={rF[key]['relations']} ---", flush=True)

        # -- Recipe A: MIDDLE (shared L13) --
        if MIDDLE_LAYER in caa_unit:
            for alpha in ALPHAS:
                sv = (caa_unit[MIDDLE_LAYER] * alpha).to(dtype).to(device)
                rkey = f"{key}_A_middle_a{alpha}"
                all_results = run_eval(
                    f"{key}/A_MIDDLE α={alpha}",
                    [(MIDDLE_LAYER, sv)],
                    vis_F, labels_F, base_rF,
                    rkey, all_results, results_path,
                    mdl, proc, yids, nids, device, vsr_all,
                )

        # -- Recipe B: CAA_SAE_DOWN (lF's vector at lF..25) --
        if lF in caa_unit:
            v_lF = caa_unit[lF]
            for alpha in ALPHAS:
                sv = (v_lF * alpha).to(dtype).to(device)
                inject = [(l, sv) for l in range(lF, NUM_LAYERS)]
                rkey = f"{key}_B_sae_down_a{alpha}"
                all_results = run_eval(
                    f"{key}/B_SAE_DOWN α={alpha}",
                    inject,
                    vis_F, labels_F, base_rF,
                    rkey, all_results, results_path,
                    mdl, proc, yids, nids, device, vsr_all,
                )

        # -- Recipe C: BACKBONE (all-8 unit vectors, per-layer) --
        for alpha in ALPHAS:
            inject = []
            for l in CACHED_LAYERS:
                if l in caa_unit:
                    inject.append((l, (caa_unit[l] * alpha).to(dtype).to(device)))
            rkey = f"{key}_C_backbone_a{alpha}"
            all_results = run_eval(
                f"{key}/C_BACKBONE α={alpha}",
                inject,
                vis_F, labels_F, base_rF,
                rkey, all_results, results_path,
                mdl, proc, yids, nids, device, vsr_all,
            )

        # -- Recipe D: BACKBONE + γ·W_dec[F] at lF --
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
                        f"{key}/D_BB+WDEC α={alpha} γ={gamma}",
                        inject,
                        vis_F, labels_F, base_rF,
                        rkey, all_results, results_path,
                        mdl, proc, yids, nids, device, vsr_all,
                    )

        # -- Recipe E: SPATIAL_LAYER — α · unit(v_CAA[lF]) @ lF only --
        if lF in caa_unit:
            for alpha in ALPHAS:
                sv = (caa_unit[lF] * alpha).to(dtype).to(device)
                rkey = f"{key}_E_spatial_a{alpha}"
                all_results = run_eval(
                    f"{key}/E_SPATIAL_LAYER α={alpha}",
                    [(lF, sv)],
                    vis_F, labels_F, base_rF,
                    rkey, all_results, results_path,
                    mdl, proc, yids, nids, device, vsr_all,
                )

        # -- Recipe F: SPATIAL + γ·W_dec[F] single layer at lF --
        if lF in caa_unit and w_dec is not None:
            for alpha in ALPHAS:
                for gamma in GAMMAS:
                    sv = (caa_unit[lF] * alpha + w_dec * gamma).to(dtype).to(device)
                    rkey = f"{key}_F_spatial_wdec_a{alpha}_g{gamma}"
                    all_results = run_eval(
                        f"{key}/F_SPAT+WDEC α={alpha} γ={gamma}",
                        [(lF, sv)],
                        vis_F, labels_F, base_rF,
                        rkey, all_results, results_path,
                        mdl, proc, yids, nids, device, vsr_all,
                    )

        gc.collect(); torch.cuda.empty_cache()

    # Summary — per-feature per-relation, all 6 recipes
    print(f"\n{'='*140}\nSUMMARY — R(F)∩test per-feature, per-relation, best Δ per recipe\n{'='*140}", flush=True)
    print(f"  {'Feature':<14} {'Layer':<5} {'Relations':<30} {'N':>4} {'Base':>7}  "
          f"{'A MIDDLE':>10}  {'B SAE_DOWN':>11}  {'C BACKBONE':>11}  {'D BB+WDEC (γ)':>14}  "
          f"{'E SPATIAL':>10}  {'F SPAT+WDEC (γ)':>16}")
    print("  " + "-"*140)
    for sf in SPATIAL_FEATURES:
        k = sf["key"]; lF = sf["layer"]
        bk = f"{k}_rF_base"
        if bk not in all_results: continue
        n = all_results[bk]["n"]; ba = all_results[bk]["acc"]
        rels = all_results[bk].get("relations", [])
        rel_str = ", ".join(rels)[:28]

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
        print(f"  {k:<14} L{lF:<4} {rel_str:<30} {n:>4} {ba:>6.2f}%  "
              f"{fmt(aA):>10}  {fmt(aB):>11}  {fmt(aC):>11}  {fmtG(aD):>14}  "
              f"{fmt(aE):>10}  {fmtG(aF):>16}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
