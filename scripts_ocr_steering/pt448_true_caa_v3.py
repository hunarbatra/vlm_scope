#!/usr/bin/env python3
"""
True CAA steering — hook-based, all 10 spatial features.

Computes label-contrastive steering vectors from mix-448 hidden states
(cached), then steers the target model on each R(F) relation subset
using register_forward_hook (no NNsight — avoids OOM).

    v[l] = mean(h_mix[l] | label=1) - mean(h_mix[l] | label=0)

Two conditions per feature F at layer lF:
  GLOBAL   v computed over all ~10,972 VSR samples at lF.
           Eval on full R(F).
  FEATURE  v computed over R(F) ∩ {mix-448 SAE feature F fires} at lF.
           Eval on full R(F).  (same eval set as GLOBAL — fair comparison)

Claim: FEATURE > GLOBAL because lF is the specific layer where concept F
is encoded in mix-448; restricting to firing samples gives a sharper
label-contrastive direction.

Usage:
    CUDA_VISIBLE_DEVICES=0 TARGET=pt  python3 -B pt448_true_caa_v3.py
    CUDA_VISIBLE_DEVICES=1 TARGET=mix python3 -B pt448_true_caa_v3.py
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

# ─────────────────────── Config ───────────────────────────────
TARGET = os.environ.get("TARGET", "pt").lower()
assert TARGET in ("pt", "mix"), f"TARGET must be pt or mix, got {TARGET!r}"

PT_MODEL  = "google/paligemma2-3b-pt-448"
MIX_MODEL = "google/paligemma2-3b-mix-448"

MIX_HIDDEN_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_BASE       = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_v3")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# All 10 top spatial features with canonical relations
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

def _load_h_mix(vi, layers):
    path = MIX_HIDDEN_DIR / f"vi_{vi:05d}.pt"
    if not path.exists(): return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return {l: d[l].float() for l in layers if l in d}
    except Exception:
        return None


# ─────────────────────── Vector computation ───────────────────
def compute_global_vectors(vsr_all, layers):
    """mean(h_mix[l]|label=1) - mean(h_mix[l]|label=0) over all VSR."""
    print("[STEP 1] Global CAA vectors from mix-448...", flush=True)
    pos  = {l: None for l in layers}
    neg  = {l: None for l in layers}
    pc   = {l: 0    for l in layers}
    nc   = {l: 0    for l in layers}

    for vi in range(len(vsr_all)):
        label = int(vsr_all[vi].get("label", 0))
        h = _load_h_mix(vi, layers)
        if h is None: continue
        for l in layers:
            if l not in h: continue
            v = h[l]
            if label == 1:
                pos[l] = v.clone() if pos[l] is None else pos[l] + v
                pc[l] += 1
            else:
                neg[l] = v.clone() if neg[l] is None else neg[l] + v
                nc[l] += 1
        if (vi + 1) % 2000 == 0:
            print(f"  {vi+1}/{len(vsr_all)}...", flush=True)

    v_global = {}
    for l in layers:
        if pc[l] > 0 and nc[l] > 0:
            v_global[l] = pos[l] / pc[l] - neg[l] / nc[l]
            print(f"  L{l}: pos={pc[l]}, neg={nc[l]}, norm={v_global[l].norm():.4f}", flush=True)
    return v_global


def compute_feature_vectors(vsr_all):
    """For each feature F: mean(h_mix[lF]|fire_F,label=1) - mean(h_mix[lF]|fire_F,label=0)."""
    print("\n[STEP 2] Feature-specific CAA vectors...", flush=True)
    v_feat = {}
    for sf in SPATIAL_FEATURES:
        key, layer = sf["key"], sf["layer"]
        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            print(f"  [{key}] acts missing — skip", flush=True)
            continue
        acts_data = json.load(open(acts_path))
        acts      = acts_data.get("acts", {})
        relations = acts_data.get("relations", [])
        fire_vi   = {int(k) for k, c in acts.items() if c > 0}
        print(f"  [{key}] R(F)={len(acts)}, fire={len(fire_vi)} "
              f"({100*len(fire_vi)/max(len(acts),1):.1f}%), rel={relations}", flush=True)

        pos_s = neg_s = None
        pn = nn = 0
        for vi in fire_vi:
            label = int(vsr_all[vi].get("label", 0))
            h = _load_h_mix(vi, [layer])
            if h is None or layer not in h: continue
            v = h[layer]
            if label == 1:
                pos_s = v.clone() if pos_s is None else pos_s + v; pn += 1
            else:
                neg_s = v.clone() if neg_s is None else neg_s + v; nn += 1

        if pn == 0 or nn == 0:
            print(f"  [{key}] skip: pos={pn} neg={nn}", flush=True)
            continue
        v_feat[key] = {"layer": layer, "vec": pos_s/pn - neg_s/nn,
                       "pos_n": pn, "neg_n": nn, "relations": relations}
        print(f"  [{key}] pos={pn}, neg={nn}, norm={v_feat[key]['vec'].norm():.4f}", flush=True)
    return v_feat


# ─────────────────────── Hook-based inference ─────────────────
def run_steer_sweep(cond_tag, steer_vec, steer_layer, rel_vis, rel_labels,
                    base_acc, alphas, result_key, all_results, results_path,
                    model_raw, processor, yes_ids, no_ids, device, vsr_all):
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

        sv_gpu = (sv_norm * alpha).to(next(model_raw.parameters()).dtype).to(device)

        def make_hook(sv_=sv_gpu):
            def hook_fn(module, input, output):
                ie = img_end_r[0]
                hidden = output[0]
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
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                hook_h = model_raw.model.language_model.layers[steer_layer].register_forward_hook(make_hook())
                with torch.no_grad():
                    out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv)
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


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from utils import process_vlm_inputs, get_image_token_positions
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print(f"True CAA v3 — SOURCE=mix-448  TARGET={TARGET}-448  (hook-based)")
    print(f"Eval: R(F) relation subsets only — all 10 spatial features")
    print("=" * 70, flush=True)

    device   = "cuda:0"
    out_dir  = OUT_BASE / f"mix_to_{TARGET}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    # ── Load VSR ──
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    # ── Compute feature-specific vectors (pure cache reads, no GPU) ──
    v_feat = compute_feature_vectors(vsr_all)

    # Diagnostics
    diag = {
        "target": TARGET,
        "feature_stats": {k: {"layer": v["layer"], "norm": float(v["vec"].norm()),
                              "pos_n": v["pos_n"], "neg_n": v["neg_n"]}
                          for k, v in v_feat.items()},
    }
    with open(out_dir / "vector_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print("\n[INFO] Vector diagnostics saved.", flush=True)

    # ── Load target model ──
    target_hf = MIX_MODEL if TARGET == "mix" else PT_MODEL
    print(f"\n[INFO] Loading {target_hf}...", flush=True)
    processor = AutoProcessor.from_pretrained(target_hf)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        target_hf, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # ── Per-feature sweep ──
    for sf in SPATIAL_FEATURES:
        key, layer = sf["key"], sf["layer"]

        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            print(f"\n[{key}] acts file missing — skip", flush=True)
            continue
        acts_data  = json.load(open(acts_path))
        acts       = acts_data.get("acts", {})
        relations  = acts_data.get("relations", [])
        rel_vis    = [int(k) for k in acts.keys()]
        rel_labels = [int(vsr_all[vi].get("label", 0)) for vi in rel_vis]

        # ── Baseline ──
        base_key = f"{key}_base"
        if base_key not in all_results:
            print(f"\n[{key}] {TARGET}-448 baseline on R(F) (n={len(rel_vis)})...", flush=True)
            bc = bt = 0
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

        # ── FEATURE condition ──
        if key in v_feat:
            print(f"  [{key}] FEATURE steer (L{layer})...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/FEATURE", v_feat[key]["vec"], layer,
                rel_vis, rel_labels, base_acc, ALPHAS,
                f"{key}_feature", all_results, results_path,
                model_raw, processor, yes_ids, no_ids, device, vsr_all,
            )

        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"True CAA mix→{TARGET} — Summary")
    print(f"{'='*70}")
    print(f"  {'Feature':<16} {'Rel':<30} {'Base':>7} {'FEATURE':>9} {'N':>6}")
    print("  " + "-" * 72)
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        base = all_results.get(f"{key}_base", {})
        if not base: continue
        base_acc = base["acc"]; base_n = base["n"]
        rels_str = "; ".join(base.get("relations", []))[:28]

        def best_delta(rkey):
            r = all_results.get(rkey, {})
            if not r: return "—"
            bd = max(v.get("delta", -999) for v in r.values())
            return f"{bd:+.2f}%"

        print(f"  {key:<16} {rels_str:<30} {base_acc:>6.2f}%  "
              f"{best_delta(key+'_feature'):>9}  {base_n:>6}")

    print(f"\nSaved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
