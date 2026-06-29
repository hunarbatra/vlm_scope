#!/usr/bin/env python3
"""
COND 4 — SPATIAL+BOOST (project+amplify) — mix-src → pt-448.

For each feature F at layer lF:
  proj  = (v_paired[lF] · W_dec_F_unit) · W_dec_F_unit
  steer = v_paired[lF] + (β − 1) · proj
        = v_paired[lF] amplified along W_dec[F]'s axis
  inject  mult · steer  at lF

β ∈ {2, 3, 5, 10}  multiplier sweep m ∈ {-3,-2,-1,1,2,3}.

Distinct from SPATIAL+WDEC (adds γ·W_dec as a new direction): BOOST stays on
v_paired's manifold and only amplifies the component already aligned with W_dec[F].
If F is causally relevant, this should preferentially boost the feature-relevant
component of the truth direction; if F is irrelevant, this degenerates to identity
(proj ≈ 0 → steer ≈ v_paired).

Evaluates on R(F) ∩ VSR test (same subsets as cond 2/3).

Vectors: label-aware paired from mix-448 paired cache (train+dev). RAW.

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_rimsky_mix_to_pt_boost.py
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
OUT_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_rimsky_mix_to_pt_boost")
IMAGE_CACHE  = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END    = 8777
MULTIPLIERS  = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
BETAS        = [2.0, 3.0, 5.0, 10.0]

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
    return d["W_dec"][feature_idx].float()  # already unit norm


def compute_paired_caa(vsr_labels, layers):
    print(f"[STEP] Computing Rimsky paired CAA at layers {layers}...", flush=True)
    acc = {l: None for l in layers}; n = 0
    for vi in range(TRAIN_END):
        p = PAIRED_DIR / f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception: continue
        if "yes" not in d or "no" not in d: continue
        label = int(vsr_labels[vi]); n += 1
        for l in layers:
            if l not in d["yes"] or l not in d["no"]: continue
            h_yes = d["yes"][l].float(); h_no = d["no"][l].float()
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
    print("COND 4: SPATIAL+BOOST (project+amplify) — mix-src → pt-448")
    print("=" * 72, flush=True)

    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s: f"{s}.jsonl"}, split=s)
        for s in ["train", "dev", "test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label", 0)) for vi in range(len(vsr_all))]
    test_vis_full = list(range(TRAIN_END, len(vsr_all)))

    feat_layers = sorted(set(sf["layer"] for sf in SPATIAL_FEATURES))
    caa = compute_paired_caa(vsr_labels, feat_layers)
    gc.collect()

    # Precompute BOOST steer vectors per feature per β
    # steer(F, β) = v_paired[lF] + (β-1) · proj(v_paired[lF] onto W_dec[F])
    boost_vectors = {}
    for sf in SPATIAL_FEATURES:
        k, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if lF not in caa: continue
        w_dec = _load_wdec(lF, fi)
        if w_dec is None: continue
        w_dec_unit = w_dec / w_dec.norm().clamp(min=1e-8)  # already unit but be safe
        v = caa[lF]
        # scalar projection coefficient
        coeff = (v * w_dec_unit).sum().item()
        proj = coeff * w_dec_unit
        boost_vectors[k] = {}
        for beta in BETAS:
            steer = v + (beta - 1.0) * proj
            boost_vectors[k][beta] = steer
        cos_vw = (v / v.norm().clamp(min=1e-8) * w_dec_unit).sum().item()
        print(f"  [{k}] L{lF}: coeff(v·W_dec_unit)={coeff:+.3f}  cos={cos_vw:+.3f}  ||v||={v.norm():.3f}  ||proj||={abs(coeff):.3f}", flush=True)

    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)
    dtype = next(mdl.parameters()).dtype

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # R(F)∩test subsets
    rF = {}
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists(): continue
        ad = json.load(open(ap))
        ak = {int(x) for x in ad.get("acts", {}).keys()}
        tvis = [v for v in test_vis_full if v in ak]
        rF[k] = {"vis": tvis, "labels": [vsr_labels[v] for v in tvis], "relations": ad.get("relations", [])}

    # R(F) baselines (share with 3-cond script if available)
    shared_results_path = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_rimsky_mix_to_pt_3cond/results.json")
    shared = {}
    if shared_results_path.exists():
        try: shared = json.load(open(shared_results_path))
        except: pass

    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        if k not in rF: continue
        bk = f"{k}_rF_base"
        if bk in all_results: continue
        if bk in shared:
            all_results[bk] = shared[bk]
            print(f"  [{k}] rF base (shared): {shared[bk]['acc']:.2f}% (n={shared[bk]['n']})", flush=True)
            continue
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

    # Run BOOST sweep per feature
    for sf in SPATIAL_FEATURES:
        k, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if k not in rF or k not in boost_vectors: continue
        vis_F = rF[k]["vis"]; labels_F = rF[k]["labels"]
        if len(vis_F) < 5:
            print(f"[{k}] too few samples ({len(vis_F)}) — skip", flush=True); continue
        base_rF = all_results[f"{k}_rF_base"]["acc"]

        print(f"\n--- {k}  (n={len(vis_F)}, base={base_rF:.2f}%, L{lF}) ---", flush=True)

        for beta in BETAS:
            v_boost = boost_vectors[k][beta]
            for mult in MULTIPLIERS:
                sv = (v_boost * mult).to(dtype).to(device)
                rkey = f"{k}_boost_b{beta}_m{mult}"
                all_results = run_eval(
                    f"{k}/BOOST β={beta:g} m={mult:+g}",
                    [(lF, sv)],
                    vis_F, labels_F, base_rF,
                    rkey, all_results, results_path,
                    mdl, proc, yids, nids, device, vsr_all,
                )
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*100}\nBOOST SUMMARY — best (β, m) per feature\n{'='*100}", flush=True)
    print(f"  {'Feature':<14} {'N':>4} {'Base':>7}  {'BOOST best Δ':>14}  {'(β, m)':>12}")
    print("  " + "-"*70)
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        bk = f"{k}_rF_base"
        if bk not in all_results: continue
        n = all_results[bk]["n"]; ba = all_results[bk]["acc"]
        best = None
        for beta in BETAS:
            for m in MULTIPLIERS:
                r = all_results.get(f"{k}_boost_b{beta}_m{m}")
                if r and (best is None or r["delta"] > best[0]):
                    best = (r["delta"], beta, m)
        best_s = f"{best[0]:+.2f}%" if best else "—"
        params = f"(β={best[1]:g}, m={best[2]:+g})" if best else "—"
        print(f"  {k:<14} {n:>4} {ba:>6.2f}%  {best_s:>14}  {params:>12}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
