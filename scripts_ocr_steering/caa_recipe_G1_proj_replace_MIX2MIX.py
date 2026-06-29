#!/usr/bin/env python3
"""
Recipe G1 — SPATIAL + FEATURE PROJECT-AND-REPLACE at lF only.

For each spatial feature F at layer lF:
  coeff = v_CAA_unit[lF] · W_dec_F_unit           # existing component (tiny)
  v_remainder = v_CAA_unit[lF] − coeff·W_dec_F_unit
  steer = α · ( v_remainder + β · W_dec_F_unit )   # set F-coefficient = β exactly
  inject steer @ lF only

β ∈ {1, 3, 10, 30}  — TARGET coefficient along W_dec[F] (replaces existing)
α ∈ {0.5, 1, 2, 5}  — overall scale

Contrast:
  F (SPAT+WDEC):  v_CAA + γ·W_dec      (adds W_dec on top; doesn't remove existing)
  G BOOST:        v_CAA + (β−1)·proj   (amplifies v_CAA's tiny W_dec component)
  G1 REPLACE:     v_remainder + β·W_dec (strips component, then sets fresh magnitude)

Eval: extract from train [0..7679], eval on R(F) ∩ (dev+test) [7680..10971].

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_recipe_G1_proj_replace.py
"""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL = "google/paligemma2-3b-mix-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_recipe_G1_proj_replace_MIX2MIX")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 7680
ALPHAS = [0.5, 1.0, 2.0, 5.0]
BETAS  = [1.0, 3.0, 10.0, 30.0]

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

def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][fi].float()

def compute_meanpool_caa(vsr_labels, layer):
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

    print("=" * 72)
    print("Recipe G1 — SPATIAL + PROJECT-AND-REPLACE at lF")
    print("=" * 72, flush=True)

    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s: f"{s}.jsonl"}, split=s)
        for s in ["train", "dev", "test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label", 0)) for vi in range(len(vsr_all))]
    test_vis_full = list(range(TRAIN_END, len(vsr_all)))
    print(f"[INFO] train extract: {TRAIN_END}, dev+test eval: {len(test_vis_full)}", flush=True)

    # Precompute steer vectors for every (feature, β)
    g1_vecs = {}    # key -> {beta: vec}
    diagnostic = {}
    for sf in SPATIAL_FEATURES:
        k, lF, fi = sf["key"], sf["layer"], sf["feature"]
        v = compute_meanpool_caa(vsr_labels, lF)
        w = _load_wdec(lF, fi)
        if v is None or w is None: continue
        v_unit = v / v.norm().clamp(min=1e-8)
        w_unit = w / w.norm().clamp(min=1e-8)
        coeff = (v_unit * w_unit).sum().item()
        v_remainder = v_unit - coeff * w_unit
        diagnostic[k] = {"v_norm": v.norm().item(), "coeff": coeff, "v_rem_norm": v_remainder.norm().item()}
        g1_vecs[k] = {}
        for beta in BETAS:
            # set F-coefficient exactly to β (replacing the existing tiny coeff)
            g1_vecs[k][beta] = v_remainder + beta * w_unit
        print(f"  [{k}] L{lF}: coeff={coeff:+.3f} ||v_rem||={v_remainder.norm():.3f}  (G1: F-coeff=β, not β+coeff)", flush=True)

    print(f"\n[INFO] Loading {MIX_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(MIX_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)
    dtype = next(mdl.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}
    all_results["_diagnostic"] = diagnostic

    # R(F)∩(dev+test) subsets
    rF = {}
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists(): continue
        ad = json.load(open(ap))
        ak = {int(x) for x in ad.get("acts", {}).keys()}
        tvis = [v for v in test_vis_full if v in ak]
        rF[k] = {"vis": tvis, "labels": [vsr_labels[v] for v in tvis], "relations": ad.get("relations", [])}

    # Reuse baselines if available from sibling runs
    shared_candidates = [
        Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_recipe_compare_mix_to_pt_devtest/results.json"),
        Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_recipe_G_spat_boost_devtest/results.json"),
    ]
    shared = {}
    for c in shared_candidates:
        if c.exists():
            try: shared.update(json.load(open(c)))
            except: pass

    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        if k not in rF: continue
        bk = f"{k}_rF_base"
        if bk in all_results: continue
        if bk in shared:
            all_results[bk] = shared[bk]
            print(f"  [{k}] rF base (shared): {shared[bk]['acc']:.2f}% (n={shared[bk]['n']})", flush=True); continue
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

    # G1 sweep
    for sf in SPATIAL_FEATURES:
        k, lF = sf["key"], sf["layer"]
        if k not in rF or k not in g1_vecs: continue
        vis_F = rF[k]["vis"]; labels_F = rF[k]["labels"]
        if len(vis_F) < 5:
            print(f"[{k}] too few samples — skip", flush=True); continue
        base_rF = all_results[f"{k}_rF_base"]["acc"]
        print(f"\n--- {k}  n={len(vis_F)}  base={base_rF:.2f}%  L{lF} ---", flush=True)

        for beta in BETAS:
            v_g1 = g1_vecs[k][beta]
            for alpha in ALPHAS:
                sv = (v_g1 * alpha).to(dtype).to(device)
                rkey = f"{k}_G1_replace_b{beta}_a{alpha}"
                all_results = run_eval(
                    f"{k}/G1_REPLACE β={beta:g} α={alpha:g}",
                    [(lF, sv)],
                    vis_F, labels_F, base_rF,
                    rkey, all_results, results_path,
                    mdl, proc, yids, nids, device, vsr_all,
                )
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}\nRecipe G1 — best per feature\n{'='*80}", flush=True)
    print(f"  {'Feature':<14} {'N':>4} {'Base':>7}  {'coeff':>7}  {'G1 best Δ':>10}  {'(β, α)':>12}")
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        bk = f"{k}_rF_base"
        if bk not in all_results: continue
        n = all_results[bk]["n"]; ba = all_results[bk]["acc"]
        coeff = diagnostic.get(k, {}).get("coeff", 0)
        best = None
        for b in BETAS:
            for a in ALPHAS:
                r = all_results.get(f"{k}_G1_replace_b{b}_a{a}")
                if r and (best is None or r["delta"] > best[0]):
                    best = (r["delta"], b, a)
        bs = f"{best[0]:+.2f}%" if best else "—"
        pars = f"(β={best[1]:g}, α={best[2]:g})" if best else "—"
        print(f"  {k:<14} {n:>4} {ba:>6.2f}%  {coeff:>+7.3f}  {bs:>10}  {pars:>12}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
