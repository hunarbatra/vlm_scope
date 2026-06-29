#!/usr/bin/env python3
"""
Canonical Rimsky-style PAIRED CAA — mix-src → steer pt-448.

Three conditions per feature F at layer lF:

  PAIRED_MIDDLE
      v_paired[L13] = mean over all 10k samples of ( h_yes[L13] - h_no[L13] )
      where h_yes/h_no are extracted at the appended " Yes"/" No" answer-token
      position. Inject α · unit(v_paired[L13]) at L13.

  PAIRED_SPATIAL_FBOOST  (the spatial+feature-aware variant)
      v_paired[lF] = mean( h_yes[lF] - h_no[lF] ) at the feature's SAE layer.
      Δc_F = mean over all 10k samples of ( SAE_F(h_yes[lF]) - SAE_F(h_no[lF]) )
      Inject α · unit(v_paired[lF]) + γ · Δc_F · W_dec[F] at lF.
      γ is the "feature intensification" knob — sweeping it tests directly
      whether boosting feature F's reconstruction direction adds steering signal.

  PAIRED_SPATIAL_FONLY   (strongest causal test of feature F alone)
      Inject α · Δc_F · W_dec[F] at lF — pure SAE feature signal, no CAA.
      If the spatial features we identified are causal for VSR, this should
      produce a positive Δ on R(F) on its own.

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B pt448_true_caa_paired_mix_to_pt.py
"""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

PT_MODEL        = "google/paligemma2-3b-pt-448"
MIDDLE_LAYER    = 13
SAE_CKPT_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")

PAIRED_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken")
SAE_ACTS_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR         = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_paired_mix_to_pt")
IMAGE_CACHE     = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET     = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
# γ multipliers for F_WDEC: weight of W_dec[F] in unit(unit(v_paired) + γ·W_dec[F])
GAMMAS = [0.5, 1.0, 3.0, 10.0]
# (Δc_F-scaled FBOOST/FONLY removed: chosen spatial features don't fire at the
# answer-token position — selected by text-token activations, OOD here. We
# instead test W_dec[F] direction at multiple weights, plus pure W_dec[F] as FONLY.)

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


def _build_vsr_prompt(statement):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\nAnswer:"
    )

def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap; no_ids -= overlap
    return yes_ids, no_ids

def _predict(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item()  if no_ids  else 1e-9
    d = y + n
    return 1 if (y / d if d > 0 else 0.5) > 0.5 else 0

def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h  = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None

def _load_paired(vi):
    p = PAIRED_DIR / f"vi_{vi:05d}.pt"
    if not p.exists(): return None
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return None

def _load_sae(layer):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return {
        "W_enc":     d["W_enc"].float(),       # [2304, 16384]
        "b_enc":     d["b_enc"].float(),       # [16384]
        "W_dec":     d["W_dec"].float(),       # [16384, 2304] — rows are unit-norm
        "threshold": d["threshold"].float(),   # [16384]
    }

def _sae_feature_act(h, sae, feat_idx):
    """JumpReLU activation of feature `feat_idx` on hidden state h [2304]."""
    pre = h @ sae["W_enc"][:, feat_idx] + sae["b_enc"][feat_idx]
    thr = sae["threshold"][feat_idx]
    return (pre * (pre > thr)).item()


# ─────────────────────── Vector + scalar computation ──────────
def compute_vectors_and_deltas(N, vsr_labels):
    """
    Build LABEL-AWARE Rimsky paired CAA:
      v_middle             — at L13 over all samples, mean(h_correct - h_wrong)
      v_spatial[key]       — at lF over all samples (one per unique layer), same formulation
      W_dec_F[key]         — unit-norm SAE decoder direction for feature F at lF

    For label=1 samples (Yes is correct): contribute (h_yes - h_no)
    For label=0 samples (No is correct):  contribute (h_no - h_yes)
    """
    print("[STEP 1] Computing label-aware paired CAA vectors...", flush=True)

    layers_needed = sorted(set([MIDDLE_LAYER] + [sf["layer"] for sf in SPATIAL_FEATURES]))
    print(f"  layers needed: {layers_needed}", flush=True)

    # Load SAEs for all spatial-feature layers (only used to extract W_dec[F])
    saes = {}
    for sf in SPATIAL_FEATURES:
        l = sf["layer"]
        if l not in saes:
            saes[l] = _load_sae(l)
            if saes[l] is None:
                print(f"  [WARN] SAE missing for L{l}", flush=True)

    # CAA accumulators
    pair_sum  = {l: None for l in layers_needed}
    pair_n    = {l: 0    for l in layers_needed}

    # LABEL-AWARE: for label=1 samples, " Yes" is correct (positive); for label=0, " No" is correct.
    # v = mean over all samples of (h_correct - h_wrong) — the "be correct" direction.
    n_loaded = n_pos = n_neg = 0
    for vi in range(N):
        d = _load_paired(vi)
        if d is None: continue
        if "yes" not in d or "no" not in d: continue
        label = int(vsr_labels[vi])
        n_loaded += 1
        if label == 1: n_pos += 1
        else:          n_neg += 1

        for l in layers_needed:
            if l not in d["yes"] or l not in d["no"]: continue
            h_yes = d["yes"][l].float()
            h_no  = d["no"][l].float()
            if label == 1:
                diff = h_yes - h_no   # Yes is correct
            else:
                diff = h_no  - h_yes  # No is correct
            if pair_sum[l] is None: pair_sum[l] = diff.clone()
            else:                   pair_sum[l] = pair_sum[l] + diff
            pair_n[l] += 1

        if (vi + 1) % 1000 == 0:
            print(f"    {vi+1}/{N} samples loaded={n_loaded}", flush=True)

    print(f"  total loaded: {n_loaded}  (label=1: {n_pos}  label=0: {n_neg})", flush=True)

    v_middle = (pair_sum[MIDDLE_LAYER] / pair_n[MIDDLE_LAYER]) if pair_n[MIDDLE_LAYER] > 0 else None
    if v_middle is not None:
        print(f"  PAIRED_MIDDLE  L{MIDDLE_LAYER}: n={pair_n[MIDDLE_LAYER]}  norm={v_middle.norm():.4f}", flush=True)

    v_spatial = {}
    W_dec_F   = {}
    for sf in SPATIAL_FEATURES:
        key, l, fi = sf["key"], sf["layer"], sf["feature"]
        if pair_n[l] == 0:
            print(f"  [{key}] no samples at L{l} — skip", flush=True); continue
        v_spatial[key] = pair_sum[l] / pair_n[l]
        W_dec_F[key] = saes[l]["W_dec"][fi]   # already unit norm
        cos_v_w = (v_spatial[key] / v_spatial[key].norm().clamp(min=1e-8) * W_dec_F[key]).sum().item()
        print(f"  [{key}] L{l}: v_norm={v_spatial[key].norm():.3f}  cos(v_paired,W_dec[F])={cos_v_w:+.3f}", flush=True)

    return v_middle, v_spatial, W_dec_F


# ─────────────────────── Hook-based steering ──────────────────
def run_steer_sweep(cond_tag, steer_vec, steer_layer, alphas, rel_vis, rel_labels,
                    base_acc, result_key, all_results, results_path,
                    model, processor, yes_ids, no_ids, device, vsr_all):
    from utils import process_vlm_inputs, get_image_token_positions

    sv_norm   = steer_vec / steer_vec.norm().clamp(min=1e-8)
    img_end_r = [0]

    for alpha in alphas:
        akey = str(alpha)
        if akey in all_results.get(result_key, {}) and \
                all_results[result_key][akey].get("n", 0) > 0:
            r = all_results[result_key][akey]
            print(f"  [SKIP {cond_tag}] α={alpha}: {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
            continue

        sv_gpu = (sv_norm * alpha).to(next(model.parameters()).dtype).to(device)

        def make_hook(sv_=sv_gpu):
            def hook_fn(module, input, output):
                ie = img_end_r[0]
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[0, ie:] = hidden[0, ie:] + sv_.unsqueeze(0)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook_fn

        correct = total = 0
        for vi, label in zip(rel_vis, rel_labels):
            ex  = vsr_all[vi]
            img = _load_image(ex)
            if img is None: continue
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            hook_h = None
            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                hook_h = model.model.language_model.layers[steer_layer].register_forward_hook(make_hook())
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                hook_h.remove(); hook_h = None
                pred    = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                total  += 1
                correct += int(pred == label)
            except Exception as e:
                if hook_h is not None:
                    try: hook_h.remove()
                    except Exception: pass
                if total < 3:
                    print(f"    [WARN] vi={vi}: {e}", flush=True)

        if total == 0: continue
        acc   = correct / total * 100
        delta = acc - base_acc
        all_results.setdefault(result_key, {})[akey] = {"acc": acc, "delta": delta, "n": total}
        print(f"  [{cond_tag}] α={alpha}: {acc:.2f}% Δ={delta:+.2f}% ({correct}/{total})", flush=True)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("Canonical PAIRED CAA — mix-src → steer pt-448")
    print("LABEL-AWARE: MIDDLE + SPATIAL + F_WDEC γ-sweep + FONLY (W_dec[F] only)")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    vsr_labels = [int(vsr_all[vi].get("label", 0)) for vi in range(len(vsr_all))]
    v_middle, v_spatial, W_dec_F = compute_vectors_and_deltas(len(vsr_all), vsr_labels)
    gc.collect()

    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    for sf in SPATIAL_FEATURES:
        key, layer, feat_idx = sf["key"], sf["layer"], sf["feature"]

        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            print(f"\n[{key}] acts missing — skip", flush=True); continue
        acts_data  = json.load(open(acts_path))
        acts       = acts_data.get("acts", {})
        relations  = acts_data.get("relations", [])
        rel_vis    = [int(k) for k in acts.keys()]
        rel_labels = [int(vsr_all[vi].get("label", 0)) for vi in rel_vis]

        # Baseline
        base_key = f"{key}_base"
        if base_key not in all_results:
            print(f"\n[{key}] pt-448 baseline on R(F) (n={len(rel_vis)}, rel={relations})...", flush=True)
            from utils import process_vlm_inputs
            bc = bt = 0
            for vi, lbl in zip(rel_vis, rel_labels):
                ex  = vsr_all[vi]
                img = _load_image(ex)
                if img is None: continue
                prompt = _build_vsr_prompt(str(ex.get("caption", "")))
                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                    with torch.no_grad():
                        out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                    bt += 1; bc += int(pred == lbl)
                except Exception: continue
            base_acc = bc / max(bt, 1) * 100
            all_results[base_key] = {"acc": base_acc, "n": bt, "relations": relations}
            print(f"  [{key}] baseline: {base_acc:.2f}% (n={bt})", flush=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            base_acc = all_results[base_key]["acc"]
            print(f"\n[{key}] baseline (cached): {base_acc:.2f}%  rel={relations}", flush=True)

        # ── PAIRED_MIDDLE (single shared vector for all features) ──
        if v_middle is not None:
            print(f"  [{key}] PAIRED_MIDDLE steer (L{MIDDLE_LAYER}→L{MIDDLE_LAYER})...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/MIDDLE", v_middle, MIDDLE_LAYER, ALPHAS,
                rel_vis, rel_labels, base_acc,
                f"{key}_paired_middle", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )

        # ── PAIRED_SPATIAL: α · unit(v_paired[lF]) — paired CAA at lF, no W_dec ──
        if key in v_spatial:
            print(f"  [{key}] PAIRED_SPATIAL steer (L{layer}→L{layer})...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/SPATIAL", v_spatial[key], layer, ALPHAS,
                rel_vis, rel_labels, base_acc,
                f"{key}_paired_spatial", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )

        # ── PAIRED_F_WDEC γ-sweep:  α · unit( unit(v_paired[lF]) + γ · W_dec[F] ) ──
        # γ=0.5 → mostly v_paired; γ=1 → 50/50; γ=3,10 → tilted toward W_dec[F]
        if key in v_spatial and key in W_dec_F:
            v_sp_unit = v_spatial[key] / v_spatial[key].norm().clamp(min=1e-8)
            for gamma in GAMMAS:
                steer = v_sp_unit + gamma * W_dec_F[key]
                tag  = f"F_WDEC_g{gamma}"
                rkey = f"{key}_paired_f_wdec_g{gamma}"
                print(f"  [{key}] {tag} steer (L{layer}→L{layer})...", flush=True)
                all_results = run_steer_sweep(
                    f"{key}/{tag}", steer, layer, ALPHAS,
                    rel_vis, rel_labels, base_acc,
                    rkey, all_results, results_path,
                    model, processor, yes_ids, no_ids, device, vsr_all,
                )

        # ── PAIRED_FONLY: α · W_dec[F] only — does the SAE feature direction help on its own? ──
        if key in W_dec_F:
            print(f"  [{key}] FONLY steer (L{layer}→L{layer}, pure W_dec[F])...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/FONLY", W_dec_F[key], layer, ALPHAS,
                rel_vis, rel_labels, base_acc,
                f"{key}_paired_fonly", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )

        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("PAIRED CAA — mix-src → pt-448  (best Δ across α-sweep per condition)")
    print(f"{'='*80}")
    print(f"  {'Feature':<14} {'N':>5} {'Base':>6}  {'MIDDLE':>8}  {'SPATIAL':>8}  {'F_W(g.5)':>9}  {'F_W(g1)':>8}  {'F_W(g3)':>8}  {'F_W(g10)':>9}  {'FONLY':>8}")
    print("  " + "-"*100)
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        base = all_results.get(f"{key}_base", {})
        if not base: continue
        ba = base["acc"]; n = base["n"]
        def bd(rkey):
            r = all_results.get(rkey, {})
            if not r: return "—"
            return f"{max(v.get('delta',-999) for v in r.values()):+.2f}%"
        print(f"  {key:<14} {n:>5} {ba:>5.2f}% "
              f" {bd(key+'_paired_middle'):>8} "
              f" {bd(key+'_paired_spatial'):>8} "
              f" {bd(key+'_paired_f_wdec_g0.5'):>9} "
              f" {bd(key+'_paired_f_wdec_g1.0'):>8} "
              f" {bd(key+'_paired_f_wdec_g3.0'):>8} "
              f" {bd(key+'_paired_f_wdec_g10.0'):>9} "
              f" {bd(key+'_paired_fonly'):>8}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
