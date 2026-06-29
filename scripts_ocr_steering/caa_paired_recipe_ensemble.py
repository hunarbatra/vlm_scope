#!/usr/bin/env python3
"""
Multi-feature ENSEMBLE steering: inject CAA + W_dec for ALL 8 winner features
simultaneously. Tests if combined "OCR capability" pressure works better than
single-feature D_BB+WDEC.

Per sample: at each backbone layer, sum CAA contributions from all features
whose R(F) contains this sample. At each feature's home layer, also add γ·W_dec[F].
"""
import os, sys, json, gc, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
SAE_CKPT_DIR = SAE_ROOT / "checkpoints"
PAIR_CACHE   = SAE_ROOT / "analysis_ocr/paired_cache_ocrprompt"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts_ocrprompt"
OUT_DIR      = SAE_ROOT / "analysis_ocr/caa_paired_ensemble"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER = 13
BACKBONE_LAYERS = [17, 19, 20, 21]
ALPHAS  = [1.0, 2.0, 5.0, 10.0]
GAMMAS  = [1.0, 3.0, 10.0]
MAX_NEW_TOKENS = 64

# All 8 winner features
WINNERS = [
    {"layer": 21, "feature": 13072, "key": "L21_F13072"},
    {"layer": 19, "feature": 9893,  "key": "L19_F9893"},
    {"layer": 21, "feature": 677,   "key": "L21_F677"},
    {"layer": 19, "feature": 8866,  "key": "L19_F8866"},
    {"layer": 17, "feature": 9368,  "key": "L17_F9368"},
    {"layer": 17, "feature": 12336, "key": "L17_F12336"},
    {"layer": 21, "feature": 10675, "key": "L21_F10675"},
    {"layer": 19, "feature": 89,    "key": "L19_F89"},
]


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def _correct(resp, gt):
    if not resp: return False
    r = str(resp).strip().lower(); g = str(gt).strip().lower()
    return bool(g) and (g in r or r in g)


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][fi].float()


def compute_paired_caa(layer, indices):
    pos = neg = None; n = 0
    for si in indices:
        p = PAIR_CACHE / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=False)
        except: continue
        if "pos" not in d or "neg" not in d: continue
        if layer not in d["pos"] or layer not in d["neg"]: continue
        hp = d["pos"][layer].float(); hn = d["neg"][layer].float()
        pos = hp.clone() if pos is None else pos + hp
        neg = hn.clone() if neg is None else neg + hn
        n += 1
    if n == 0 or pos is None: return None, 0
    return (pos - neg) / n, n


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("="*80)
    print("ENSEMBLE STEERING — all 8 winner features simultaneously")
    print("="*80, flush=True)

    ds = load_dataset("echo840/OCRBench", split="test")
    test_indices = list(range(len(ds)))

    # Per-feature R(F) sets and W_decs
    rF_per = {}
    w_decs = {}
    feat_layers = set()
    for f in WINNERS:
        k, l, fi = f["key"], f["layer"], f["feature"]
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists():
            print(f"  [SKIP {k}] acts missing"); continue
        ad = json.load(open(ap))
        rF_per[k] = {int(x) for x, v in ad.get("acts", {}).items() if v > 0}
        w_decs[k] = _load_wdec(l, fi)
        feat_layers.add(l)
        print(f"  {k}: R(F)={len(rF_per[k])}", flush=True)

    # UNION of all R(F) — this is the eval set (samples where any winner fires)
    rF_union = sorted(set().union(*rF_per.values()))
    print(f"\n  R_union (any winner fires): n={len(rF_union)}", flush=True)

    # CAA per-layer per-feature (built from each feature's R(F))
    needed_layers = sorted(set(BACKBONE_LAYERS) | {MIDDLE_LAYER} | feat_layers)
    print(f"\n[INFO] Computing per-feature CAA at {len(needed_layers)} layers", flush=True)
    caa_unit_by_feat = {}
    for f in WINNERS:
        k = f["key"]
        if k not in rF_per: continue
        caa_unit_by_feat[k] = {}
        for l in needed_layers:
            v, n = compute_paired_caa(l, indices=sorted(rF_per[k]))
            if v is not None:
                caa_unit_by_feat[k][l] = v / v.norm().clamp(min=1e-8)

    print(f"\n[Loading {PT_MODEL}]", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok = proc.tokenizer
    dtype = next(mdl.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # ---- Baseline on R_union ----
    bk = "ensemble_rF_base"
    if bk not in all_results:
        bc = bt = 0
        for si in rF_union:
            ex = ds[si]; img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            if img is None or not gt: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
                with torch.no_grad():
                    out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                           pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False, use_cache=True)
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                bt += 1; bc += int(_correct(resp, gt))
            except: continue
        all_results[bk] = {"acc": bc/max(bt,1)*100, "n": bt}
        print(f"  Baseline R_union: {all_results[bk]['acc']:.2f}% (n={bt})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    base = all_results[bk]["acc"]

    # ---- ENSEMBLE eval: sample-conditioned injection ----
    img_end_r = [0]

    def make_hook(get_sv):
        def f(m, inp, out):
            ie = img_end_r[0]
            h = out[0] if isinstance(out, tuple) else out
            sv = get_sv()
            if sv is not None:
                h[0, ie:] = h[0, ie:] + sv.unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return f

    # We can't trivially make sample-conditioned hooks per-layer without
    # rewiring. Simpler approach: precompute per-sample injection vectors
    # before each generation, and use closures.

    def run_ensemble(alpha, gamma):
        c = t = 0
        active_svs = {}
        for L in BACKBONE_LAYERS:
            active_svs[L] = None

        # Set up persistent hooks that read from active_svs dict
        hooks = []
        for L in BACKBONE_LAYERS:
            def make_layer_hook(l):
                def hook_fn(m, inp, out):
                    ie = img_end_r[0]
                    h = out[0] if isinstance(out, tuple) else out
                    sv = active_svs.get(l)
                    if sv is not None:
                        h[0, ie:] = h[0, ie:] + sv.unsqueeze(0)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                return hook_fn
            hooks.append(mdl.model.language_model.layers[L].register_forward_hook(make_layer_hook(L)))

        try:
            for si in rF_union:
                ex = ds[si]; img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
                if img is None or not gt: continue

                # Find active features for this sample
                active_feats = [f for f in WINNERS if f["key"] in rF_per and si in rF_per[f["key"]]]
                if not active_feats: continue

                # Build per-layer injection vectors
                # For backbone layer L: sum of α·unit(v_CAA_F[L]) over all active features F
                # For feature's home layer lF: also add γ·W_dec[F]
                for L in BACKBONE_LAYERS:
                    sv_layer = None
                    for f in active_feats:
                        k = f["key"]; lF = f["layer"]
                        if L not in caa_unit_by_feat.get(k, {}): continue
                        contrib = caa_unit_by_feat[k][L] * alpha
                        if L == lF:
                            contrib = contrib + w_decs[k] * gamma
                        if sv_layer is None:
                            sv_layer = contrib.clone()
                        else:
                            sv_layer = sv_layer + contrib
                    active_svs[L] = sv_layer.to(dtype).to(device) if sv_layer is not None else None

                try:
                    img = img.convert("RGB")
                    iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
                    _, img_end_r[0] = get_image_token_positions(iids)
                    with torch.no_grad():
                        out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                               pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                               do_sample=False, use_cache=True)
                    resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                    t += 1; c += int(_correct(resp, gt))
                except Exception:
                    continue

            return c, t
        finally:
            for h in hooks:
                try: h.remove()
                except: pass

    for alpha in ALPHAS:
        for gamma in GAMMAS:
            rk = f"ensemble_a{alpha}_g{gamma}"
            if rk in all_results and all_results[rk].get("n", 0) > 0:
                r = all_results[rk]
                print(f"  [SKIP α={alpha} γ={gamma}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
                continue
            c, t = run_ensemble(alpha, gamma)
            if t == 0: continue
            acc = c / t * 100
            delta = acc - base
            all_results[rk] = {"acc": acc, "delta": delta, "n": t}
            print(f"  [ENSEMBLE α={alpha} γ={gamma}] {acc:.2f}% Δ={delta:+.2f}% ({c}/{t})", flush=True)
            with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    print(f"\n[DONE] ensemble. base={base:.2f}%", flush=True)


if __name__ == "__main__":
    main()
