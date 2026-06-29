#!/usr/bin/env python3
"""
True CAA steering for pt-448.

Computes label-contrastive steering vectors from mix-448 hidden states:
    v[l] = mean(h_mix[l] | label=1) - mean(h_mix[l] | label=0)

Two conditions compared:
  GLOBAL   v_global[l] computed over all ~10,972 VSR samples.
           Used to steer pt-448 on each R(F) relation subset.
  FEATURE  v_feat[F] computed at feature layer lF, using only
           R(F) samples where SAE feature F fires (coeff > 0).
           Used to steer pt-448 on that same R(F) subset.

This is the canonical CAA design (Rimsky et al. 2023), sourcing
the contrastive direction from mix-448 (stronger spatial model)
to improve pt-448 (weaker base model).

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 -B pt448_true_caa.py
"""

import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# ─────────────────────── Config ───────────────────────────────
PT_MODEL       = "google/paligemma2-3b-pt-448"
MIX_HIDDEN_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

SAE_LAYERS = [4, 6, 9, 11, 12, 14, 15]
ALPHAS     = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

SPATIAL_FEATURES = [
    {"layer": 4,  "feature": 14233, "key": "L4_F14233"},
    {"layer": 6,  "feature": 7539,  "key": "L6_F7539"},
    {"layer": 9,  "feature": 387,   "key": "L9_F387"},
    {"layer": 9,  "feature": 7540,  "key": "L9_F7540"},
    {"layer": 11, "feature": 12278, "key": "L11_F12278"},
    {"layer": 12, "feature": 2257,  "key": "L12_F2257"},
    {"layer": 14, "feature": 10561, "key": "L14_F10561"},
    {"layer": 15, "feature": 220,   "key": "L15_F220"},
]

# ─────────────────────── Helpers ──────────────────────────────
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
    p_yes = max(y / d if d > 0 else 0.5, 1e-7)
    return 1 if p_yes > 0.5 else 0

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

def _load_h_mix(vi, layers):
    """Load mix hidden states for sample vi at requested layers."""
    path = MIX_HIDDEN_DIR / f"vi_{vi:05d}.pt"
    if not path.exists():
        return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return {l: d[l].float() for l in layers if l in d}
    except Exception:
        return None


# ─────────────────────── Step 1: Global CAA vector ────────────
def compute_global_caa_vectors(vsr_all):
    """
    v_global[l] = mean(h_mix[l] | label=1) - mean(h_mix[l] | label=0)
    over all N VSR samples at SAE_LAYERS.
    """
    print("[STEP 1] Computing global CAA vectors from mix-448 hidden states...", flush=True)
    pos_sums  = {l: None for l in SAE_LAYERS}
    neg_sums  = {l: None for l in SAE_LAYERS}
    pos_count = {l: 0    for l in SAE_LAYERS}
    neg_count = {l: 0    for l in SAE_LAYERS}
    skipped   = 0

    for vi in range(len(vsr_all)):
        label = int(vsr_all[vi].get("label", 0))
        h = _load_h_mix(vi, SAE_LAYERS)
        if h is None:
            skipped += 1
            continue
        for l in SAE_LAYERS:
            if l not in h:
                continue
            vec = h[l]
            if label == 1:
                pos_sums[l]  = vec.clone() if pos_sums[l] is None else pos_sums[l] + vec
                pos_count[l] += 1
            else:
                neg_sums[l]  = vec.clone() if neg_sums[l] is None else neg_sums[l] + vec
                neg_count[l] += 1

        if (vi + 1) % 2000 == 0:
            print(f"  {vi+1}/{len(vsr_all)} processed...", flush=True)

    print(f"  Skipped {skipped} samples (no cached h_mix).", flush=True)

    v_global = {}
    for l in SAE_LAYERS:
        if pos_count[l] > 0 and neg_count[l] > 0:
            pos_mean = pos_sums[l] / pos_count[l]
            neg_mean = neg_sums[l] / neg_count[l]
            v_global[l] = pos_mean - neg_mean
            norm = v_global[l].norm().item()
            print(f"  Layer {l:2d}: pos={pos_count[l]}, neg={neg_count[l]}, "
                  f"vec_norm={norm:.4f}", flush=True)
        else:
            print(f"  Layer {l:2d}: SKIPPED (pos={pos_count[l]}, neg={neg_count[l]})", flush=True)

    return v_global


# ─────────────────────── Step 2: Feature-specific CAA vectors ─
def compute_feature_caa_vectors(vsr_all):
    """
    For each feature F at layer lF:
      Fire_F = R(F) samples where SAE coeff > 0
      v_feat[F] = mean(h_mix[lF] | vi in Fire_F, label=1)
                - mean(h_mix[lF] | vi in Fire_F, label=0)
    """
    print("\n[STEP 2] Computing feature-specific CAA vectors...", flush=True)
    v_feat = {}

    for sf in SPATIAL_FEATURES:
        layer   = sf["layer"]
        feature = sf["feature"]
        key     = sf["key"]
        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            print(f"  [{key}] acts file missing — skipping", flush=True)
            continue

        acts_data = json.load(open(acts_path))
        acts      = acts_data.get("acts", {})  # {vi_str: coeff}
        relations = acts_data.get("relations", [])

        # Filter to firing samples
        fire_vi = {int(vi_str) for vi_str, coeff in acts.items() if coeff > 0}
        total_R = len(acts)
        print(f"  [{key}] R(F)={total_R}, firing={len(fire_vi)} "
              f"({100*len(fire_vi)/max(total_R,1):.1f}%), "
              f"relations={relations}", flush=True)

        pos_sum, neg_sum = None, None
        pos_n,   neg_n   = 0, 0

        for vi in fire_vi:
            label = int(vsr_all[vi].get("label", 0))
            h     = _load_h_mix(vi, [layer])
            if h is None or layer not in h:
                continue
            vec = h[layer]
            if label == 1:
                pos_sum = vec.clone() if pos_sum is None else pos_sum + vec
                pos_n  += 1
            else:
                neg_sum = vec.clone() if neg_sum is None else neg_sum + vec
                neg_n  += 1

        if pos_n == 0 or neg_n == 0:
            print(f"  [{key}] SKIPPED: pos={pos_n}, neg={neg_n}", flush=True)
            continue

        v_feat[key] = {
            "layer":    layer,
            "feature":  feature,
            "vec":      pos_sum / pos_n - neg_sum / neg_n,
            "pos_n":    pos_n,
            "neg_n":    neg_n,
            "fire_n":   len(fire_vi),
            "total_R":  total_R,
            "relations": relations,
        }
        norm = v_feat[key]["vec"].norm().item()
        print(f"  [{key}] pos={pos_n}, neg={neg_n}, vec_norm={norm:.4f}", flush=True)

    return v_feat


# ─────────────────────── Step 3: Inference ────────────────────
def run_caa_inference(condition, steer_vec_l, steer_layer, eval_vis, eval_labels,
                      nns_model, processor, yes_ids, no_ids, model_dtype, device,
                      base_acc, alphas, result_key, results_dict):
    """
    Steer pt-448 on eval_vis (subset) with steer_vec at steer_layer.
    Returns updated results_dict.
    """
    from utils import process_vlm_inputs, get_image_token_positions

    sv_norm = steer_vec_l / steer_vec_l.norm().clamp(min=1e-8)
    sv_gpu  = sv_norm.to(model_dtype).to(device)

    for alpha in alphas:
        alpha_key = str(alpha)
        if alpha_key in results_dict.get(result_key, {}) and \
           results_dict[result_key][alpha_key].get("n", 0) > 0:
            r = results_dict[result_key][alpha_key]
            print(f"  [SKIP {condition}] α={alpha}: {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
            continue

        correct = total = 0
        for idx, vi in enumerate(eval_vis):
            ex    = vsr_all_global[vi]
            label = eval_labels[idx]
            img   = _load_image(ex)
            if img is None:
                continue
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))

            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw_global, device=device)
                _, img_end = get_image_token_positions(iids)

                sv_col = sv_gpu.unsqueeze(1)
                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    lo    = nns_model.model.language_model.layers[steer_layer].output[0][0, img_end:]
                    ones  = (lo @ sv_col) * 0.0 + 1.0
                    lo   += alpha * ones * sv_gpu.unsqueeze(0)
                    logits_s = nns_model.output.logits.save()

                pred    = _predict(logits_s[0, -1, :], yes_ids, no_ids)
                total  += 1
                correct += int(pred == label)
            except Exception as e:
                if total < 3:
                    print(f"    [WARN] vi={vi}: {e}", flush=True)
                continue

        if total == 0:
            continue
        acc   = correct / total * 100
        delta = acc - base_acc
        if result_key not in results_dict:
            results_dict[result_key] = {}
        results_dict[result_key][alpha_key] = {"acc": acc, "delta": delta, "n": total}
        print(f"  [{condition}] α={alpha}: {acc:.2f}% Δ={delta:+.2f}% ({correct}/{total})", flush=True)

    return results_dict


# ─────────────────────── Main ─────────────────────────────────

# Globals needed inside run_caa_inference (VSR dataset + model refs)
vsr_all_global   = None
model_raw_global = None


def main():
    global vsr_all_global, model_raw_global

    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("True CAA Steering: mix-448 label-contrastive vector → pt-448")
    print("Conditions: GLOBAL (all samples) vs FEATURE (R(F) firing subset)")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load VSR ──
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    vsr_all_global = vsr_all

    # ── Step 1 & 2: Compute steering vectors from cached h_mix ──
    v_global = compute_global_caa_vectors(vsr_all)
    v_feat   = compute_feature_caa_vectors(vsr_all)

    # Save vector norms for diagnostics
    diag = {
        "global": {str(l): float(v.norm()) for l, v in v_global.items()},
        "feature": {k: {
            "layer": v["layer"], "feature": v["feature"],
            "vec_norm": float(v["vec"].norm()),
            "pos_n": v["pos_n"], "neg_n": v["neg_n"],
            "fire_n": v["fire_n"], "total_R": v["total_R"],
            "fire_pct": 100 * v["fire_n"] / max(v["total_R"], 1),
        } for k, v in v_feat.items()}
    }
    with open(OUT_DIR / "vector_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n[INFO] Vector diagnostics saved to {OUT_DIR}/vector_diagnostics.json", flush=True)

    # ── Step 3: Load pt-448 ──
    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    model_raw_global = model_raw
    nns_model   = NNsight(model_raw)
    tokenizer   = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # ── Load or init results ──
    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # ── Step 4: Sweep alphas per feature, both conditions ──
    for sf in SPATIAL_FEATURES:
        key     = sf["key"]
        layer   = sf["layer"]

        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            continue
        acts_data = json.load(open(acts_path))
        acts      = acts_data.get("acts", {})
        relations = acts_data.get("relations", [])

        # R(F) samples with labels
        rel_vis    = [int(vi_str) for vi_str in acts.keys()]
        rel_labels = [int(vsr_all[vi].get("label", 0)) for vi in rel_vis]
        n_pos      = sum(rel_labels)
        n_neg      = len(rel_labels) - n_pos
        base_correct = sum(
            1 for vi, lbl in zip(rel_vis, rel_labels)
            if (_predict_from_cache_or_skip(vi, yes_ids, no_ids) == lbl
                if False else lbl == lbl)  # placeholder — compute base below
        )

        # Compute baseline acc on R(F) subset from pt-448 (no steering)
        base_key = f"{key}_base"
        if base_key not in all_results:
            print(f"\n[{key}] Computing baseline on R(F) (n={len(rel_vis)})...", flush=True)
            b_correct = b_total = 0
            for vi, lbl in zip(rel_vis, rel_labels):
                ex  = vsr_all[vi]
                img = _load_image(ex)
                if img is None: continue
                prompt = _build_vsr_prompt(str(ex.get("caption", "")))
                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                    with torch.no_grad():
                        out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                    b_total   += 1
                    b_correct += int(pred == lbl)
                except Exception:
                    continue
            base_acc_R = b_correct / max(b_total, 1) * 100
            all_results[base_key] = {"acc": base_acc_R, "n": b_total, "relations": relations}
            print(f"  [{key}] R(F) baseline: {base_acc_R:.2f}% (n={b_total})", flush=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            base_acc_R = all_results[base_key]["acc"]
            print(f"\n[{key}] R(F) baseline (cached): {base_acc_R:.2f}%  "
                  f"relations={relations}", flush=True)

        # ── Condition A: GLOBAL vector at this layer ──
        if layer not in v_global:
            print(f"  [{key}] GLOBAL vector missing at layer {layer} — skip", flush=True)
        else:
            global_key = f"{key}_global"
            print(f"  [{key}] GLOBAL steer (layer {layer})...", flush=True)
            all_results = run_caa_inference(
                condition  = f"{key}/GLOBAL",
                steer_vec_l= v_global[layer],
                steer_layer= layer,
                eval_vis   = rel_vis,
                eval_labels= rel_labels,
                nns_model  = nns_model,
                processor  = processor,
                yes_ids    = yes_ids, no_ids=no_ids,
                model_dtype= model_dtype, device=device,
                base_acc   = base_acc_R,
                alphas     = ALPHAS,
                result_key = global_key,
                results_dict=all_results,
            )
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

        # ── Condition B: FEATURE-specific vector ──
        if key not in v_feat:
            print(f"  [{key}] FEATURE vector missing — skip", flush=True)
        else:
            feat_info   = v_feat[key]
            feature_key = f"{key}_feature"
            # Eval only on firing samples for feature condition
            fire_vi  = {int(vi_str) for vi_str, c in acts.items() if c > 0}
            fire_vis = [vi for vi in rel_vis if vi in fire_vi]
            fire_lbl = [int(vsr_all[vi].get("label", 0)) for vi in fire_vis]

            print(f"  [{key}] FEATURE steer (layer {layer}, "
                  f"fire_n={len(fire_vis)}/{len(rel_vis)})...", flush=True)
            all_results = run_caa_inference(
                condition  = f"{key}/FEATURE",
                steer_vec_l= feat_info["vec"],
                steer_layer= layer,
                eval_vis   = fire_vis,
                eval_labels= fire_lbl,
                nns_model  = nns_model,
                processor  = processor,
                yes_ids    = yes_ids, no_ids=no_ids,
                model_dtype= model_dtype, device=device,
                base_acc   = base_acc_R,
                alphas     = ALPHAS,
                result_key = feature_key,
                results_dict=all_results,
            )
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

        gc.collect(); torch.cuda.empty_cache()

    # ── Final summary ──
    print(f"\n{'='*70}")
    print("True CAA Results Summary")
    print(f"{'='*70}")
    print(f"{'Feature':<16} {'Condition':<10} {'Base':>7} {'Best Δ':>8} {'Best α':>8} {'N':>6}")
    print("-" * 60)
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        base_acc_R = all_results.get(f"{key}_base", {}).get("acc", float("nan"))
        for cond in ["global", "feature"]:
            rkey = f"{key}_{cond}"
            if rkey not in all_results: continue
            r = all_results[rkey]
            if not isinstance(r, dict) or not r: continue
            best_a, best_v = max(r.items(), key=lambda x: x[1].get("delta", -999))
            print(f"  {key:<16} {cond:<10} {base_acc_R:>6.2f}%  "
                  f"{best_v['delta']:>+7.2f}%  α={best_a:<6}  n={best_v['n']}")

    print(f"\nSaved: {results_path}", flush=True)


def _predict_from_cache_or_skip(vi, yes_ids, no_ids):
    return 0  # placeholder — not used


if __name__ == "__main__":
    main()
