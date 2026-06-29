#!/usr/bin/env python3
"""
Rimsky-style label-aware paired CAA — mix-src → steer pt-448.
Three conditions per spatial feature F at layer lF:

  1. MIDDLE          α · v_paired[L13] injected at L13
                     (one vector shared across all features)

  2. SPATIAL_LAYER   α · v_paired[lF] injected at lF
                     (feature-specific layer, no W_dec)

  3. SPATIAL+WDEC    α · v_paired[lF] + γ · W_dec[F] injected at lF
                     (spatial layer CAA plus feature significance)

Vectors are RAW (not unit-normalized). Multipliers α ∈ {-3, -2, -1, 1, 2, 3}.
γ-sweep for condition 3: {1, 3, 10}.

Evaluates on R(F) ∩ VSR_test subset for per-feature results, plus full VSR test
for MIDDLE as a single shared reference.

Vectors come from /data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken/
(label-aware paired " Yes"/" No" cache, computed on mix-448).
Steering target is pt-448 (mix→pt).

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_rimsky_mix_to_pt_3cond.py
"""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

PT_MODEL = "google/paligemma2-3b-pt-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
PAIRED_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken")
SAE_ACTS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_rimsky_mix_to_pt_3cond")
IMAGE_CACHE  = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END    = 8777
MIDDLE_LAYER = 13
MULTIPLIERS  = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
GAMMAS       = [1.0, 3.0, 10.0]

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


def compute_paired_caa(vsr_labels, layers):
    """Label-aware paired CAA at each requested layer, from train+dev paired cache."""
    print(f"[STEP] Computing Rimsky label-aware paired CAA at layers {layers}...", flush=True)
    acc = {l: None for l in layers}
    n = 0
    for vi in range(TRAIN_END):
        p = PAIRED_DIR / f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception: continue
        if "yes" not in d or "no" not in d: continue
        label = int(vsr_labels[vi])
        n += 1
        for l in layers:
            if l not in d["yes"] or l not in d["no"]: continue
            h_yes = d["yes"][l].float()
            h_no  = d["no"][l].float()
            diff = (h_yes - h_no) if label == 1 else (h_no - h_yes)
            acc[l] = diff.clone() if acc[l] is None else acc[l] + diff
    out = {}
    for l in layers:
        if acc[l] is None: continue
        v = acc[l] / n
        out[l] = v
        print(f"  v_CAA L{l}: norm={v.norm():.3f}  n={n}  (RAW)", flush=True)
    return out


def run_eval(tag, inject_pairs, test_vis, test_labels, base_acc,
             result_key, all_results, results_path,
             model, processor, yes_ids, no_ids, device, vsr_all):
    """inject_pairs = list of (layer, scaled_vector_on_GPU). Evaluate on test_vis."""
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

    print("=" * 72)
    print("Rimsky-style paired CAA — mix-src → pt-448  —  3 conditions per feature")
    print("=" * 72, flush=True)

    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s: f"{s}.jsonl"}, split=s)
        for s in ["train", "dev", "test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label", 0)) for vi in range(len(vsr_all))]
    test_vis_full = list(range(TRAIN_END, len(vsr_all)))
    test_labels_full = [vsr_labels[vi] for vi in test_vis_full]
    print(f"[INFO] train+dev: {TRAIN_END}, test: {len(test_vis_full)}", flush=True)

    # Compute CAA at L13 and each unique feature layer
    feat_layers = sorted(set([MIDDLE_LAYER] + [sf["layer"] for sf in SPATIAL_FEATURES]))
    caa = compute_paired_caa(vsr_labels, feat_layers)
    gc.collect()

    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)
    dtype = next(mdl.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Full-test baseline
    if "full_base" not in all_results:
        print(f"\n[BASELINE] pt-448 full VSR test...", flush=True)
        bc = bt = 0
        for i, (vi, lbl) in enumerate(zip(test_vis_full, test_labels_full)):
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
            if (i + 1) % 500 == 0:
                print(f"  baseline {i+1}/{len(test_vis_full)}  acc={bc/max(bt,1)*100:.2f}%", flush=True)
        all_results["full_base"] = {"acc": bc/max(bt,1)*100, "n": bt}
        print(f"[BASELINE] full test: {all_results['full_base']['acc']:.2f}% (n={bt})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
    base_full = all_results["full_base"]["acc"]

    # R(F)∩test subsets + baselines
    rF = {}
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists(): continue
        ad = json.load(open(ap))
        ak = {int(x) for x in ad.get("acts", {}).keys()}
        tvis = [v for v in test_vis_full if v in ak]
        rF[k] = {"vis": tvis, "labels": [vsr_labels[v] for v in tvis], "relations": ad.get("relations", [])}
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
        print(f"  [{k}] R(F)∩test base: {all_results[bk]['acc']:.2f}% (n={bt})", flush=True)
    with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # ==== CONDITION 1: MIDDLE (full test, shared L13 vector) ====
    print(f"\n{'='*72}\nCOND 1: MIDDLE — α·v_paired[L{MIDDLE_LAYER}] @ L{MIDDLE_LAYER}, full VSR test\n{'='*72}", flush=True)
    if MIDDLE_LAYER in caa:
        for mult in MULTIPLIERS:
            rkey = f"middle_full_m{mult}"
            sv = (caa[MIDDLE_LAYER] * mult).to(dtype).to(device)
            all_results = run_eval(
                f"MIDDLE/full m={mult:+g}",
                [(MIDDLE_LAYER, sv)],
                test_vis_full, test_labels_full, base_full,
                rkey, all_results, results_path,
                mdl, proc, yids, nids, device, vsr_all,
            )

    # ==== Per-feature CONDITIONS 2 & 3 on R(F) subset ====
    for sf in SPATIAL_FEATURES:
        key, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if key not in rF: continue
        if lF not in caa:
            print(f"[{key}] L{lF} CAA missing — skip", flush=True); continue
        vis_F = rF[key]["vis"]; labels_F = rF[key]["labels"]
        if len(vis_F) < 5:
            print(f"[{key}] too few samples ({len(vis_F)}) — skip", flush=True); continue
        base_rF = all_results[f"{key}_rF_base"]["acc"]

        print(f"\n--- {key}  (n={len(vis_F)}, base={base_rF:.2f}%, layer=L{lF}) ---", flush=True)

        # Cond 2: SPATIAL_LAYER — α·v_paired[lF] at lF
        for mult in MULTIPLIERS:
            sv = (caa[lF] * mult).to(dtype).to(device)
            rkey = f"{key}_spatial_m{mult}"
            all_results = run_eval(
                f"{key}/SPATIAL m={mult:+g}",
                [(lF, sv)],
                vis_F, labels_F, base_rF,
                rkey, all_results, results_path,
                mdl, proc, yids, nids, device, vsr_all,
            )

        # Cond 3: SPATIAL+WDEC — α·v_paired[lF] + γ·W_dec[F] at lF
        w_dec = _load_wdec(lF, fi)
        if w_dec is None:
            print(f"  [{key}] W_dec missing — skip cond 3", flush=True); continue
        w_dec_gpu = w_dec.to(dtype).to(device)
        for mult in MULTIPLIERS:
            for gamma in GAMMAS:
                sv = (caa[lF] * mult + w_dec * gamma).to(dtype).to(device)
                rkey = f"{key}_spatialwdec_m{mult}_g{gamma}"
                all_results = run_eval(
                    f"{key}/SPATIAL+WDEC m={mult:+g} γ={gamma:g}",
                    [(lF, sv)],
                    vis_F, labels_F, base_rF,
                    rkey, all_results, results_path,
                    mdl, proc, yids, nids, device, vsr_all,
                )

        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*100}\nSUMMARY\n{'='*100}", flush=True)
    print(f"Full VSR test baseline: {base_full:.2f}%")
    print(f"\nCOND 1 — MIDDLE (full VSR test, shared L{MIDDLE_LAYER} vector):")
    for m in MULTIPLIERS:
        r = all_results.get(f"middle_full_m{m}")
        if r: print(f"  m={m:+g}: {r['acc']:.2f}% Δ={r['delta']:+.2f}%")

    print(f"\nCOND 2 & 3 — per-feature R(F) subsets (best Δ):")
    print(f"  {'Feature':<14} {'N':>4} {'Base':>7}  {'SPAT best Δ':>13}  {'SPAT+W best Δ':>14}  {'Best (m, γ)':>14}")
    print("  " + "-"*90)
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        bk = f"{k}_rF_base"
        if bk not in all_results: continue
        n = all_results[bk]["n"]; ba = all_results[bk]["acc"]
        # SPATIAL best
        spat_best = None
        for m in MULTIPLIERS:
            r = all_results.get(f"{k}_spatial_m{m}")
            if r and (spat_best is None or r["delta"] > spat_best[0]):
                spat_best = (r["delta"], m)
        # SPATIAL+WDEC best
        sw_best = None
        for m in MULTIPLIERS:
            for g in GAMMAS:
                r = all_results.get(f"{k}_spatialwdec_m{m}_g{g}")
                if r and (sw_best is None or r["delta"] > sw_best[0]):
                    sw_best = (r["delta"], m, g)
        sp_s = f"{spat_best[0]:+.2f}% (m={spat_best[1]:+g})" if spat_best else "—"
        sw_s = f"{sw_best[0]:+.2f}%" if sw_best else "—"
        best_mg = f"(m={sw_best[1]:+g}, γ={sw_best[2]:g})" if sw_best else "—"
        print(f"  {k:<14} {n:>4} {ba:>6.2f}%  {sp_s:>13}  {sw_s:>14}  {best_mg:>14}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
