#!/usr/bin/env python3
"""
Spatial-feature intensification using MULTI-LAYER CAA as the spine.

Strategy:
  1) Fixed backbone: α · unit(v_caa[L]) injected at every cached layer L ∈ {4,6,9,11,12,13,14,15}
     with α = best_alpha (from caa_all_layers_meanpool).
  2) Plus feature-specific injection at its own layer lF:
     γ · W_dec[F] at lF.

We test 3 conditions per feature on R(F) ∩ test subset and on full test:
  BACKBONE_ONLY   multi-layer CAA only (no feature boost)  — ceiling control
  W_DEC_ONLY      γ · W_dec[F] only at lF (no CAA)          — pure feature signal
  BACKBONE+WDEC   both combined                              — intensification

γ sweep: {1, 3, 10} × α ∈ {1.0} (backbone fixed).

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_spatial_boost_multilayer.py
"""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

PT_MODEL = "google/paligemma2-3b-pt-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_spatial_boost_multilayer")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 8777
CACHED_LAYERS = [4, 6, 9, 11, 12, 13, 14, 15]
BACKBONE_ALPHA = 1.0  # the "winning" α for all-layer CAA
GAMMAS = [1.0, 3.0, 10.0]   # W_dec multiplier

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
    d = y + nn
    return 1 if (y/d if d > 0 else 0.5) > 0.5 else 0

def _load_image(ex):
    url = ex.get("image_link","")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(cp, "JPEG"); return img
    except Exception: return None

def compute_meanpool_caa(vsr_labels, layer):
    pos = neg = None; pn = nn = 0
    for vi in range(TRAIN_END):
        p = MEANPOOL_DIR / f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=True)
        except: continue
        if layer not in d: continue
        v = d[layer].float()
        if int(vsr_labels[vi]) == 1:
            pos = v.clone() if pos is None else pos+v; pn += 1
        else:
            neg = v.clone() if neg is None else neg+v; nn += 1
    if pos is None: return None
    return pos/pn - neg/nn

def _load_wdec(layer, feature_idx):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][feature_idx].float()


def run_eval(tag, inject_pairs, test_vis, test_labels, base_acc,
             result_key, all_results, results_path,
             model, processor, yes_ids, no_ids, device, vsr_all):
    """inject_pairs = list of (layer, scaled_vector_on_GPU). Run on test_vis."""
    from utils import process_vlm_inputs, get_image_token_positions
    if result_key in all_results and all_results[result_key].get("n",0) > 0:
        r = all_results[result_key]
        print(f"  [SKIP {tag}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
        return all_results
    img_end_r = [0]
    def make_hook(sv_):
        def f(m,i,o):
            ie = img_end_r[0]
            h = o[0] if isinstance(o,tuple) else o
            h[0,ie:] = h[0,ie:] + sv_.unsqueeze(0)
            return (h,)+o[1:] if isinstance(o,tuple) else h
        return f
    c = t = 0
    for vi, lbl in zip(test_vis, test_labels):
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        pt = _build_vsr_prompt(str(ex.get("caption","")))
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
            pred = _predict(out.logits[0,-1,:], yes_ids, no_ids)
            t += 1; c += int(pred == lbl)
        except Exception:
            for h in hooks:
                try: h.remove()
                except: pass
    if t == 0: return all_results
    acc = c/t*100; d = acc - base_acc
    all_results[result_key] = {"acc": acc, "delta": d, "n": t}
    print(f"  [{tag}] {acc:.2f}%  Δ={d:+.2f}%  ({c}/{t})", flush=True)
    with open(results_path,"w") as f: json.dump(all_results, f, indent=2)
    return all_results


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print(f"[INFO] Multi-layer CAA backbone (α={BACKBONE_ALPHA}) + γ·W_dec[F] boost per feature", flush=True)

    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s:f"{s}.jsonl"}, split=s)
        for s in ["train","dev","test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label",0)) for vi in range(len(vsr_all))]
    test_vis_full = list(range(TRAIN_END, len(vsr_all)))
    test_labels_full = [vsr_labels[vi] for vi in test_vis_full]

    # CAA vectors at all cached layers
    caa = {}
    for l in CACHED_LAYERS:
        v = compute_meanpool_caa(vsr_labels, l)
        if v is not None:
            caa[l] = v / v.norm().clamp(min=1e-8)
            print(f"  CAA L{l}: raw norm={v.norm():.3f}", flush=True)
    gc.collect()

    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Baselines
    base_acc_full = 53.53
    shared = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_find_working_layer/results.json")
    if shared.exists():
        try: base_acc_full = json.load(open(shared))["base"]["acc"]
        except: pass
    all_results["full_base"] = {"acc": base_acc_full, "n": 2195}
    print(f"[BASE] full: {base_acc_full:.2f}%", flush=True)

    # R(F)∩test subsets
    rF = {}
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        ap = SAE_ACTS_DIR / f"acts_{key}.json"
        if not ap.exists(): continue
        ad = json.load(open(ap))
        ak = {int(k) for k in ad.get("acts", {}).keys()}
        tvis = [v for v in test_vis_full if v in ak]
        rF[key] = {"vis": tvis, "labels": [vsr_labels[v] for v in tvis], "relations": ad.get("relations",[])}

    # Compute rF baselines
    from utils import process_vlm_inputs
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        if k not in rF: continue
        bk = f"{k}_rF_base"
        if bk in all_results: continue
        bc = bt = 0
        for vi, lbl in zip(rF[k]["vis"], rF[k]["labels"]):
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            pt = _build_vsr_prompt(str(ex.get("caption","")))
            try:
                iids, attn, pv = process_vlm_inputs(img, pt, proc, mdl, device=device)
                with torch.no_grad():
                    out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv)
                pred = _predict(out.logits[0,-1,:], yids, nids)
                bt += 1; bc += int(pred == lbl)
            except Exception: continue
        all_results[bk] = {"acc": bc/max(bt,1)*100, "n": bt, "relations": rF[k]["relations"]}
        print(f"  [{k}] rF base: {all_results[bk]['acc']:.2f}% (n={bt})", flush=True)
    with open(results_path,"w") as f: json.dump(all_results,f,indent=2)

    # Precompute backbone: scaled unit CAA vectors at each layer, moved to GPU
    dtype = next(mdl.parameters()).dtype
    backbone_sv = {l: (caa[l] * BACKBONE_ALPHA).to(dtype).to(device) for l in caa}

    # Backbone-only on full test (ceiling reference)
    if "backbone_full" not in all_results:
        print(f"[BACKBONE_ONLY full test]...", flush=True)
        inject = [(l, sv) for l, sv in backbone_sv.items()]
        run_eval(f"BACKBONE/full", inject, test_vis_full, test_labels_full, base_acc_full,
                 "backbone_full", all_results, results_path, mdl, proc, yids, nids, device, vsr_all)

    # Per-feature evals
    for sf in SPATIAL_FEATURES:
        k, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if k not in rF: continue
        vis_F, labels_F = rF[k]["vis"], rF[k]["labels"]
        base_rF = all_results[f"{k}_rF_base"]["acc"]

        # Backbone-only on rF (for collateral check)
        bk_rF = f"{k}_backbone_rF"
        if bk_rF not in all_results:
            inject = [(l, sv) for l, sv in backbone_sv.items()]
            run_eval(f"{k}/BACKBONE/rF", inject, vis_F, labels_F, base_rF,
                     bk_rF, all_results, results_path, mdl, proc, yids, nids, device, vsr_all)

        # W_dec only at lF, γ sweep (no backbone)
        w_dec = _load_wdec(lF, fi)
        if w_dec is None: continue
        for gamma in GAMMAS:
            sv_wdec_only = (w_dec * gamma).to(dtype).to(device)
            inject = [(lF, sv_wdec_only)]
            rkey = f"{k}_wdec_only_g{gamma}_rF"
            run_eval(f"{k}/WDEC_only(g{gamma})/rF", inject, vis_F, labels_F, base_rF,
                     rkey, all_results, results_path, mdl, proc, yids, nids, device, vsr_all)

        # Backbone + γ·W_dec at lF
        for gamma in GAMMAS:
            w_dec_scaled = (w_dec * gamma).to(dtype).to(device)
            # Backbone + boost: for lF layer, replace or add to the backbone injection
            inject = []
            for l, sv in backbone_sv.items():
                if l == lF:
                    inject.append((l, sv + w_dec_scaled))
                else:
                    inject.append((l, sv))
            rkey = f"{k}_backbone_wdec_g{gamma}_rF"
            run_eval(f"{k}/BB+WDEC(g{gamma})/rF", inject, vis_F, labels_F, base_rF,
                     rkey, all_results, results_path, mdl, proc, yids, nids, device, vsr_all)

        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
