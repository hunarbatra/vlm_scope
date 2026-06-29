#!/usr/bin/env python3
"""
True CAA steering — generalized source/target script.

Computes label-contrastive steering vectors from a source model's
cached hidden states, then steers the target model on each R(F)
relation-specific subset.

    v[l] = mean(h_src[l] | label=1) - mean(h_src[l] | label=0)

Two conditions per feature:
  GLOBAL   v_global[l] computed over ALL ~10,972 VSR samples.
           Steers target on R(F).
  FEATURE  v_feat[F] computed at lF using only R(F) ∩ {SAE fires}.
           Steers target on R(F) (same eval set as GLOBAL for fair comparison).

Supported SOURCE × TARGET combinations:
  mix → pt   (cross-model, already run in v1 — this script is authoritative)
  pt  → pt   (canonical True CAA — how much T/F direction exists in pt itself)
  mix → mix  (upper bound — does mix improve with its own contrastive vector)

Usage:
    CUDA_VISIBLE_DEVICES=0 SOURCE=pt  TARGET=pt  python3 -B pt448_true_caa_v2.py
    CUDA_VISIBLE_DEVICES=1 SOURCE=mix TARGET=mix python3 -B pt448_true_caa_v2.py
    CUDA_VISIBLE_DEVICES=3 SOURCE=mix TARGET=pt  python3 -B pt448_true_caa_v2.py  (already running v1)
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
SOURCE = os.environ.get("SOURCE", "mix").lower()  # "mix" or "pt"
TARGET = os.environ.get("TARGET", "pt").lower()   # "mix" or "pt"

assert SOURCE in ("mix", "pt"), f"SOURCE must be mix or pt, got {SOURCE!r}"
assert TARGET in ("mix", "pt"), f"TARGET must be mix or pt, got {TARGET!r}"

PT_MODEL  = "google/paligemma2-3b-pt-448"
MIX_MODEL = "google/paligemma2-3b-mix-448"

HIDDEN_DIRS = {
    "mix": Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden"),
    "pt":  Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/pt_hidden"),
}
SAE_ACTS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_BASE     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_v2")
IMAGE_CACHE  = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET  = "cambridgeltl/vsr_random"

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

def _load_hidden(vi, src, layers):
    path = HIDDEN_DIRS[src] / f"vi_{vi:05d}.pt"
    if not path.exists():
        return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return {l: d[l].float() for l in layers if l in d}
    except Exception:
        return None


# ─────────────────────── Step 1: Global CAA vectors ───────────
def compute_global_vectors(vsr_all, src):
    print(f"[STEP 1] Global CAA vectors from {src}-448 hidden states...", flush=True)
    pos_sums  = {l: None for l in SAE_LAYERS}
    neg_sums  = {l: None for l in SAE_LAYERS}
    pos_count = {l: 0    for l in SAE_LAYERS}
    neg_count = {l: 0    for l in SAE_LAYERS}
    skipped   = 0

    for vi in range(len(vsr_all)):
        label = int(vsr_all[vi].get("label", 0))
        h = _load_hidden(vi, src, SAE_LAYERS)
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

    print(f"  Skipped {skipped} samples (no cached hidden).", flush=True)
    v_global = {}
    for l in SAE_LAYERS:
        if pos_count[l] > 0 and neg_count[l] > 0:
            v_global[l] = pos_sums[l] / pos_count[l] - neg_sums[l] / neg_count[l]
            print(f"  L{l}: pos={pos_count[l]}, neg={neg_count[l]}, "
                  f"norm={v_global[l].norm():.4f}", flush=True)
    return v_global


# ─────────────────────── Step 2: Feature-specific vectors ─────
def compute_feature_vectors(vsr_all, src):
    print(f"\n[STEP 2] Feature-specific CAA vectors from {src}-448...", flush=True)
    v_feat = {}
    for sf in SPATIAL_FEATURES:
        key   = sf["key"]
        layer = sf["layer"]
        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            print(f"  [{key}] acts file missing — skip", flush=True)
            continue

        acts_data = json.load(open(acts_path))
        acts      = acts_data.get("acts", {})
        relations = acts_data.get("relations", [])
        fire_vi   = {int(vi_s) for vi_s, c in acts.items() if c > 0}
        print(f"  [{key}] R(F)={len(acts)}, firing={len(fire_vi)} "
              f"({100*len(fire_vi)/max(len(acts),1):.1f}%), rel={relations}", flush=True)

        pos_sum = neg_sum = None
        pos_n   = neg_n   = 0
        for vi in fire_vi:
            label = int(vsr_all[vi].get("label", 0))
            h = _load_hidden(vi, src, [layer])
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
            print(f"  [{key}] SKIP: pos={pos_n}, neg={neg_n}", flush=True)
            continue

        v_feat[key] = {
            "layer": layer, "feature": sf["feature"],
            "vec": pos_sum / pos_n - neg_sum / neg_n,
            "pos_n": pos_n, "neg_n": neg_n,
            "fire_n": len(fire_vi), "total_R": len(acts),
            "relations": relations,
        }
        print(f"  [{key}] pos={pos_n}, neg={neg_n}, "
              f"norm={v_feat[key]['vec'].norm():.4f}", flush=True)
    return v_feat


# ─────────────────────── Step 3: Inference ────────────────────
def run_inference(cond_tag, steer_vec, steer_layer, eval_vis, eval_labels,
                  base_acc, alphas, result_key, all_results, results_path,
                  nns_model, processor, yes_ids, no_ids, model_dtype, device,
                  vsr_all, target_model_raw):
    from utils import process_vlm_inputs, get_image_token_positions

    sv_norm = steer_vec / steer_vec.norm().clamp(min=1e-8)
    sv_gpu  = sv_norm.to(model_dtype).to(device)
    sv_col  = sv_gpu.unsqueeze(1)

    for alpha in alphas:
        akey = str(alpha)
        if akey in all_results.get(result_key, {}) and \
                all_results[result_key][akey].get("n", 0) > 0:
            r = all_results[result_key][akey]
            print(f"  [SKIP {cond_tag}] α={alpha}: {r['acc']:.2f}% Δ={r['delta']:+.2f}%",
                  flush=True)
            continue

        correct = total = 0
        for vi, label in zip(eval_vis, eval_labels):
            ex  = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, prompt, processor, target_model_raw, device=device)
                _, img_end = get_image_token_positions(iids)

                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    lo   = nns_model.model.language_model.layers[steer_layer].output[0][0, img_end:]
                    ones = (lo @ sv_col) * 0.0 + 1.0
                    lo  += alpha * ones * sv_gpu.unsqueeze(0)
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
        all_results.setdefault(result_key, {})[akey] = {"acc": acc, "delta": delta, "n": total}
        print(f"  [{cond_tag}] α={alpha}: {acc:.2f}% Δ={delta:+.2f}% ({correct}/{total})",
              flush=True)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    tag = f"{SOURCE}_to_{TARGET}"
    print("=" * 70)
    print(f"True CAA  SOURCE={SOURCE}-448  TARGET={TARGET}-448")
    print(f"Eval: R(F) relation subsets only")
    print("=" * 70, flush=True)

    device  = "cuda:0"
    out_dir = OUT_BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    # ── Load VSR ──
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    # ── Compute steering vectors from source hidden cache ──
    v_global = compute_global_vectors(vsr_all, SOURCE)
    v_feat   = compute_feature_vectors(vsr_all, SOURCE)

    # Save diagnostics
    diag = {
        "source": SOURCE, "target": TARGET,
        "global_norms": {str(l): float(v.norm()) for l, v in v_global.items()},
        "feature_stats": {k: {
            "layer": v["layer"], "norm": float(v["vec"].norm()),
            "pos_n": v["pos_n"], "neg_n": v["neg_n"],
            "fire_n": v["fire_n"], "total_R": v["total_R"],
            "fire_pct": 100 * v["fire_n"] / max(v["total_R"], 1),
        } for k, v in v_feat.items()}
    }
    with open(out_dir / "vector_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n[INFO] Diagnostics saved.", flush=True)

    # ── Load target model ──
    target_hf = MIX_MODEL if TARGET == "mix" else PT_MODEL
    print(f"\n[INFO] Loading target model: {target_hf}...", flush=True)
    processor = AutoProcessor.from_pretrained(target_hf)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        target_hf, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model   = NNsight(model_raw)
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # ── Load / init results ──
    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # ── Per-feature sweep ──
    for sf in SPATIAL_FEATURES:
        key   = sf["key"]
        layer = sf["layer"]

        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            continue
        acts_data = json.load(open(acts_path))
        acts      = acts_data.get("acts", {})
        relations = acts_data.get("relations", [])

        # R(F) eval set — always the full relation subset for fair comparison
        rel_vis    = [int(vi_s) for vi_s in acts.keys()]
        rel_labels = [int(vsr_all[vi].get("label", 0)) for vi in rel_vis]

        # Compute or load baseline for this target on R(F)
        base_key = f"{key}_base"
        if base_key not in all_results:
            print(f"\n[{key}] Computing {TARGET}-448 baseline on R(F) (n={len(rel_vis)})...",
                  flush=True)
            b_correct = b_total = 0
            for vi, lbl in zip(rel_vis, rel_labels):
                ex  = vsr_all[vi]
                img = _load_image(ex)
                if img is None: continue
                prompt = _build_vsr_prompt(str(ex.get("caption", "")))
                try:
                    iids, attn, pv = process_vlm_inputs(
                        img, prompt, processor, model_raw, device=device)
                    with torch.no_grad():
                        out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                    b_total   += 1
                    b_correct += int(pred == lbl)
                except Exception:
                    continue
            base_acc_R = b_correct / max(b_total, 1) * 100
            all_results[base_key] = {
                "acc": base_acc_R, "n": b_total, "relations": relations
            }
            print(f"  [{key}] {TARGET}-448 R(F) baseline: {base_acc_R:.2f}% (n={b_total})",
                  flush=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            base_acc_R = all_results[base_key]["acc"]
            print(f"\n[{key}] {TARGET}-448 R(F) baseline (cached): {base_acc_R:.2f}%  "
                  f"rel={relations}", flush=True)

        # ── Condition A: GLOBAL vector at this layer ──
        if layer in v_global:
            print(f"  [{key}] GLOBAL steer (L{layer})...", flush=True)
            all_results = run_inference(
                cond_tag   = f"{key}/GLOBAL",
                steer_vec  = v_global[layer],
                steer_layer= layer,
                eval_vis   = rel_vis,
                eval_labels= rel_labels,
                base_acc   = base_acc_R,
                alphas     = ALPHAS,
                result_key = f"{key}_global",
                all_results= all_results,
                results_path=results_path,
                nns_model  = nns_model,
                processor  = processor,
                yes_ids    = yes_ids, no_ids=no_ids,
                model_dtype= model_dtype, device=device,
                vsr_all    = vsr_all,
                target_model_raw=model_raw,
            )

        # ── Condition B: FEATURE-specific vector, eval on full R(F) ──
        if key in v_feat:
            print(f"  [{key}] FEATURE steer (L{layer})...", flush=True)
            all_results = run_inference(
                cond_tag   = f"{key}/FEATURE",
                steer_vec  = v_feat[key]["vec"],
                steer_layer= layer,
                eval_vis   = rel_vis,      # full R(F) for fair comparison
                eval_labels= rel_labels,
                base_acc   = base_acc_R,
                alphas     = ALPHAS,
                result_key = f"{key}_feature",
                all_results= all_results,
                results_path=results_path,
                nns_model  = nns_model,
                processor  = processor,
                yes_ids    = yes_ids, no_ids=no_ids,
                model_dtype= model_dtype, device=device,
                vsr_all    = vsr_all,
                target_model_raw=model_raw,
            )

        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"True CAA {SOURCE}→{TARGET} — Summary")
    print(f"{'='*70}")
    print(f"{'Feature':<16} {'Cond':<10} {'Base':>7} {'Best Δ':>8} {'Best α':>8} {'N':>6}")
    print("-" * 58)
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        base_acc_R = all_results.get(f"{key}_base", {}).get("acc", float("nan"))
        for cond in ["global", "feature"]:
            rkey = f"{key}_{cond}"
            r = all_results.get(rkey, {})
            if not r: continue
            best_a, best_v = max(r.items(), key=lambda x: x[1].get("delta", -999))
            print(f"  {key:<16} {cond:<10} {base_acc_R:>6.2f}%  "
                  f"{best_v['delta']:>+7.2f}%  α={best_a:<6}  n={best_v['n']}")
    print(f"\nSaved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
