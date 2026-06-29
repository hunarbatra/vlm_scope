#!/usr/bin/env python3
"""
Spatial-feature intensification on full VSR test, using the WINNER config from
caa_find_working_layer.py.

Reads:
  /data1/vlm_scope_sae_mix448_textonly/analysis/caa_find_working_layer/results.json
  → determines winner (meanpool vs paired), best layer, best α.

For each of the 10 spatial SAE features F at layer lF:
  Condition A — BASELINE_CAA      : α_best · unit(v_winner[best_layer]) at best_layer
                                     (same for every feature; the ceiling)
  Condition B — SPATIAL_ONLY      : α · unit(v_winner[lF]) at lF
  Condition C — SPATIAL+WDEC      : α · unit(unit(v_winner[lF]) + γ·W_dec[F]) at lF
  Condition D — SPATIAL+WDEC_MULTI: multi-α in C, pick best γ per feature

Evaluates on:
  1) Full VSR test (2195 samples) — "does boosting F help overall?"
  2) R(F) subset of test — "does boosting F help its own relations specifically?"

The R(F) test is what ablation experiments mirror: removing F hurts R(F).
If intensifying F helps R(F) while barely affecting full-VSR, that's the
clean causal claim.

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_spatial_feature_boost.py
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

PT_MODEL    = "google/paligemma2-3b-pt-448"
SAE_CKPT_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
MEANPOOL_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
PAIRED_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken")
SAE_ACTS_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
WINNER_RESULTS = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_find_working_layer/results.json")

OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_spatial_feature_boost")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 8777
CACHED_LAYERS = [4, 6, 9, 11, 12, 13, 14, 15]
GAMMAS = [0.5, 1.0, 3.0]

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


def determine_winner():
    """Read caa_find_working_layer results.json and figure out winner config."""
    if not WINNER_RESULTS.exists():
        raise SystemExit(f"Winner results not found: {WINNER_RESULTS}. Run caa_find_working_layer.py first.")
    d = json.load(open(WINNER_RESULTS))
    base_acc = d["base"]["acc"]
    best = None
    for k, v in d.items():
        if k == "base" or not isinstance(v, dict): continue
        for alpha_key, rec in v.items():
            if not isinstance(rec, dict) or "delta" not in rec: continue
            if best is None or rec["delta"] > best["delta"]:
                best = {"condition": k, "alpha": alpha_key, **rec}
    if best is None:
        raise SystemExit("No deltas found in winner results.")
    # Parse condition: meanpool_Ln or paired_Ln or meanpool_multi_...
    parts = best["condition"].split("_")
    variant = parts[0]   # 'meanpool' or 'paired'
    # single layer if _Lnn, multi if contains 'multi'
    multi = "multi" in best["condition"]
    single_layer = None
    if not multi and parts[-1].startswith("L"):
        try: single_layer = int(parts[-1][1:])
        except Exception: pass
    print(f"[WINNER] variant={variant}  layer={single_layer or 'multi'}  α={best['alpha']}  Δ={best['delta']:+.2f}% (acc={best['acc']:.2f}%)", flush=True)
    return variant, single_layer, float(best["alpha"]), base_acc, d


def compute_caa_vectors(vsr_labels, variant, layers):
    """Compute CAA vectors for given variant at requested layers, train+dev only."""
    print(f"[STEP] Computing {variant} CAA at {layers}...", flush=True)
    if variant == "meanpool":
        pos = {l: None for l in layers}; neg = {l: None for l in layers}
        pn = nn = 0
        for vi in range(TRAIN_END):
            p = MEANPOOL_DIR / f"vi_{vi:05d}.pt"
            if not p.exists(): continue
            try: d = torch.load(p, map_location="cpu", weights_only=True)
            except Exception: continue
            label = int(vsr_labels[vi])
            for l in layers:
                if l not in d: continue
                v = d[l].float()
                if label == 1: pos[l] = v.clone() if pos[l] is None else pos[l] + v
                else:          neg[l] = v.clone() if neg[l] is None else neg[l] + v
            if label == 1: pn += 1
            else:          nn += 1
        out = {}
        for l in layers:
            if pos[l] is None or neg[l] is None: continue
            out[l] = pos[l]/pn - neg[l]/nn
            print(f"  {variant} L{l}: norm={out[l].norm():.3f}", flush=True)
        return out
    else:  # paired
        acc = {l: None for l in layers}; n = 0
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
                diff = (d["yes"][l].float() - d["no"][l].float()) if label == 1 else (d["no"][l].float() - d["yes"][l].float())
                acc[l] = diff.clone() if acc[l] is None else acc[l] + diff
        out = {}
        for l in layers:
            if acc[l] is None: continue
            out[l] = acc[l] / n
            print(f"  {variant} L{l}: norm={out[l].norm():.3f}", flush=True)
        return out


def _load_wdec(layer, feature_idx):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][feature_idx].float()


def run_test_eval(tag, inject_pairs, alphas, test_vis, test_labels, base_acc,
                  result_key, all_results, results_path,
                  model, processor, yes_ids, no_ids, device, vsr_all):
    from utils import process_vlm_inputs, get_image_token_positions
    unit_pairs = [(l, v / v.norm().clamp(min=1e-8)) for (l, v) in inject_pairs]
    img_end_r = [0]
    for alpha in alphas:
        akey = str(alpha)
        if akey in all_results.get(result_key, {}) and all_results[result_key][akey].get("n", 0) > 0:
            r = all_results[result_key][akey]
            print(f"  [SKIP {tag}] α={alpha}: {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True); continue
        sv_gpu = [(l, (uv * alpha).to(next(model.parameters()).dtype).to(device)) for (l, uv) in unit_pairs]
        def make_hook(sv_):
            def hook_fn(module, input, output):
                ie = img_end_r[0]
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[0, ie:] = hidden[0, ie:] + sv_.unsqueeze(0)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook_fn
        correct = total = 0
        for vi, lbl in zip(test_vis, test_labels):
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None: continue
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            hooks = []
            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                for (l, sv) in sv_gpu:
                    hooks.append(model.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                for h in hooks:
                    try: h.remove()
                    except Exception: pass
                pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                total += 1; correct += int(pred == lbl)
            except Exception as e:
                for h in hooks:
                    try: h.remove()
                    except Exception: pass
                if total < 3: print(f"    [WARN] vi={vi}: {e}", flush=True)
        if total == 0: continue
        acc = correct / total * 100
        delta = acc - base_acc
        all_results.setdefault(result_key, {})[akey] = {"acc": acc, "delta": delta, "n": total}
        print(f"  [{tag}] α={alpha}: {acc:.2f}% Δ={delta:+.2f}% ({correct}/{total})", flush=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
    return all_results


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    variant, best_layer, best_alpha, base_acc_global, _ = determine_winner()
    if best_layer is None: best_layer = 13  # fallback

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label", 0)) for vi in range(len(vsr_all))]
    test_vis = list(range(TRAIN_END, len(vsr_all)))
    test_labels = [vsr_labels[vi] for vi in test_vis]

    # Compute CAA at all layers we might need
    caa = compute_caa_vectors(vsr_labels, variant, CACHED_LAYERS)
    gc.collect()

    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # ── Baselines on full test and on each R(F) subset ──
    if "full_base" not in all_results:
        print(f"\n[BASELINE] pt-448 full VSR test (n={len(test_vis)})...", flush=True)
        from utils import process_vlm_inputs
        bc = bt = 0
        for vi, lbl in zip(test_vis, test_labels):
            ex = vsr_all[vi]
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
        base_full = bc / max(bt, 1) * 100
        all_results["full_base"] = {"acc": base_full, "n": bt}
        print(f"  full base: {base_full:.2f}% (n={bt})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
    else:
        base_full = all_results["full_base"]["acc"]
        print(f"[BASELINE] full_base (cached): {base_full:.2f}%", flush=True)

    # R(F) subsets within test: intersect acts keys with test range
    rF_subsets = {}
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists(): continue
        ad = json.load(open(acts_path))
        acts_keys = {int(k) for k in ad.get("acts", {}).keys()}
        test_rF = [vi for vi in test_vis if vi in acts_keys]
        rF_subsets[key] = {"vis": test_rF, "labels": [vsr_labels[v] for v in test_rF], "relations": ad.get("relations", [])}
        print(f"  [{key}] R(F)∩test = {len(test_rF)}", flush=True)

    # Baseline on each R(F)∩test
    for key, sub in rF_subsets.items():
        bk = f"{key}_rF_base"
        if bk not in all_results:
            bc = bt = 0
            for vi, lbl in zip(sub["vis"], sub["labels"]):
                ex = vsr_all[vi]; img = _load_image(ex)
                if img is None: continue
                prompt = _build_vsr_prompt(str(ex.get("caption", "")))
                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                    with torch.no_grad():
                        out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                    bt += 1; bc += int(pred == lbl)
                except Exception: continue
            ba = bc / max(bt, 1) * 100
            all_results[bk] = {"acc": ba, "n": bt, "relations": sub["relations"]}
            print(f"  [{key}] R(F)∩test base: {ba:.2f}% (n={bt})", flush=True)
    with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # ── Per-feature: SPATIAL_ONLY + SPATIAL+WDEC γ-sweep ──
    per_feat_alphas = [1.0, 2.0, 5.0, 10.0]
    for sf in SPATIAL_FEATURES:
        key, lF, fi = sf["key"], sf["layer"], sf["feature"]
        if key not in rF_subsets: continue
        test_vis_F = rF_subsets[key]["vis"]
        test_labels_F = rF_subsets[key]["labels"]
        base_rF = all_results[f"{key}_rF_base"]["acc"]
        if lF not in caa:
            print(f"[{key}] L{lF} CAA missing — skip", flush=True); continue

        # SPATIAL_ONLY on full test
        all_results = run_test_eval(
            f"{key}/SPATIAL/full", [(lF, caa[lF])], per_feat_alphas,
            test_vis, test_labels, base_full,
            f"{key}_spatial_full", all_results, results_path,
            model, processor, yes_ids, no_ids, device, vsr_all,
        )
        # SPATIAL_ONLY on R(F)
        all_results = run_test_eval(
            f"{key}/SPATIAL/rF", [(lF, caa[lF])], per_feat_alphas,
            test_vis_F, test_labels_F, base_rF,
            f"{key}_spatial_rF", all_results, results_path,
            model, processor, yes_ids, no_ids, device, vsr_all,
        )

        # SPATIAL+WDEC γ-sweep on R(F) (where the feature is supposed to matter)
        w_dec = _load_wdec(lF, fi)
        if w_dec is None: continue
        v_caa_unit = caa[lF] / caa[lF].norm().clamp(min=1e-8)
        for gamma in GAMMAS:
            steer = v_caa_unit + gamma * w_dec
            all_results = run_test_eval(
                f"{key}/WDEC_g{gamma}/rF", [(lF, steer)], per_feat_alphas,
                test_vis_F, test_labels_F, base_rF,
                f"{key}_wdec_g{gamma}_rF", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )
            # Also evaluate on full test for collateral
            all_results = run_test_eval(
                f"{key}/WDEC_g{gamma}/full", [(lF, steer)], per_feat_alphas,
                test_vis, test_labels, base_full,
                f"{key}_wdec_g{gamma}_full", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*90}\nSPATIAL FEATURE BOOST SUMMARY\n{'='*90}", flush=True)
    print(f"Full VSR test baseline: {all_results['full_base']['acc']:.2f}%")
    print(f"\n{'Feature':<14} {'rF n':>5} {'rF base':>8}  {'SPAT (rF Δ)':>12} {'WDEC.5 (rF Δ)':>14} {'WDEC1 (rF Δ)':>13} {'WDEC3 (rF Δ)':>13}  {'SPAT (full Δ)':>14}")
    for sf in SPATIAL_FEATURES:
        k = sf["key"]
        bk = f"{k}_rF_base"; fb = f"{k}_spatial_full"; rb = f"{k}_spatial_rF"
        if bk not in all_results: continue
        def bd(rkey):
            r = all_results.get(rkey, {})
            if not r: return "—"
            return f"{max(v.get('delta',-999) for v in r.values()):+.2f}%"
        print(f"  {k:<12} {all_results[bk]['n']:>5} {all_results[bk]['acc']:>7.2f}%  {bd(rb):>12} {bd(k+'_wdec_g0.5_rF'):>14} {bd(k+'_wdec_g1.0_rF'):>13} {bd(k+'_wdec_g3.0_rF'):>13}  {bd(fb):>14}")
    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
