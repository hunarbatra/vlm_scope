#!/usr/bin/env python3
"""
True CAA SPATIAL_LAYER + W_dec[F] — mix→mix (self-steering).

Condition per feature F at layer lF:
  SPATIAL_LAYER_WDEC  unit(v_spatial[lF] + W_dec[F]) injected at lF.
                      v_spatial[lF] = CAA at lF over all 10,972 VSR samples.
                      W_dec[F] = SAE decoder weight for feature F (unit-norm).
                      Both are unit vectors; their sum is re-normalized before injection.

This blends the data-driven contrastive direction with the SAE's distilled
concept direction for feature F. Compared to SPATIAL_LAYER alone, this adds
the SAE's monosemantic direction for exactly feature F.

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B pt448_true_caa_v4_spatial_layer_wdec_mix_to_pt.py
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
MIX_MODEL       = "google/paligemma2-3b-mix-448"
MIDDLE_LAYER    = 13
SAE_CKPT_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")

MIX_HIDDEN_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_lasttoken")
SAE_ACTS_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR         = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix448_true_caa_v4_spatial_layer_wdec_mix_to_mix")
IMAGE_CACHE     = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET     = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

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

def _load_mix_hidden_layer(vi, layer):
    path = MIX_HIDDEN_DIR / f"vi_{vi:05d}.pt"
    if not path.exists(): return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d[layer].float() if layer in d else None
    except Exception:
        return None

def _load_wdec(layer, feature_idx):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    if not p.exists(): return None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][feature_idx].float()  # already unit-norm


# ─────────────────────── Vector computation ───────────────────
def compute_vectors(vsr_all):
    print("[STEP 1] Computing vectors from mix-448 hidden cache...", flush=True)

    # SPATIAL_LAYER: all VSR at lF (one pass per unique layer)
    v_spatial = {}
    unique_layers = sorted(set(sf["layer"] for sf in SPATIAL_FEATURES))
    for lyr in unique_layers:
        feats_at_layer = [sf for sf in SPATIAL_FEATURES if sf["layer"] == lyr]
        print(f"  SPATIAL_LAYER: L{lyr} over all {len(vsr_all)} samples...", flush=True)
        ps = ns = None; pn = nn = 0
        for vi in range(len(vsr_all)):
            label = int(vsr_all[vi].get("label", 0))
            v = _load_mix_hidden_layer(vi, lyr)
            if v is None: continue
            if label == 1:
                ps = v.clone() if ps is None else ps + v; pn += 1
            else:
                ns = v.clone() if ns is None else ns + v; nn += 1
            if (vi + 1) % 2000 == 0:
                print(f"    {vi+1}/{len(vsr_all)}...", flush=True)
        if pn > 0 and nn > 0:
            vec = ps / pn - ns / nn
            print(f"    L{lyr} pos={pn}, neg={nn}, norm={vec.norm():.4f}", flush=True)
            for sf in feats_at_layer:
                v_spatial[sf["key"]] = {"layer": lyr, "vec": vec.clone()}

    # SPATIAL_LAYER_WDEC: unit(v_spatial + W_dec[F])
    v_wdec = {}
    for sf in SPATIAL_FEATURES:
        key, lyr, feat_idx = sf["key"], sf["layer"], sf["feature"]
        if key not in v_spatial: continue
        w_dec = _load_wdec(lyr, feat_idx)
        if w_dec is None:
            print(f"  [{key}] W_dec missing — skip", flush=True); continue
        v_sl = v_spatial[key]["vec"]
        v_sl_unit = v_sl / v_sl.norm().clamp(min=1e-8)
        # w_dec is already unit-norm; sum then renormalize
        combined = v_sl_unit + w_dec
        combined = combined / combined.norm().clamp(min=1e-8)
        cos_sim = (v_sl_unit * w_dec).sum().item()
        print(f"  [{key}] SPATIAL_LAYER_WDEC: cos(v_spatial, W_dec)={cos_sim:.4f}, combined_norm_before_norm={( v_sl_unit + w_dec).norm():.4f}", flush=True)
        v_wdec[key] = {"layer": lyr, "vec": combined}

    return v_spatial, v_wdec


# ─────────────────────── Hook-based steering ──────────────────
def run_steer_sweep(cond_tag, steer_vec, steer_layer, rel_vis, rel_labels,
                    base_acc, result_key, all_results, results_path,
                    model, processor, yes_ids, no_ids, device, vsr_all):
    from utils import process_vlm_inputs, get_image_token_positions

    sv_norm   = steer_vec / steer_vec.norm().clamp(min=1e-8)
    img_end_r = [0]

    for alpha in ALPHAS:
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


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("True CAA SPATIAL_LAYER + W_dec[F] — mix→mix (self-steering)")
    print("SPATIAL_LAYER_WDEC: unit(v_spatial[lF] + W_dec[F]) → lF")
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

    v_spatial, v_wdec = compute_vectors(vsr_all)
    gc.collect()

    print(f"\n[INFO] Loading {MIX_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    for sf in SPATIAL_FEATURES:
        key, layer = sf["key"], sf["layer"]

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
            print(f"\n[{key}] mix-448 baseline on R(F) (n={len(rel_vis)}, rel={relations})...", flush=True)
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

        # SPATIAL_LAYER_WDEC condition
        if key in v_wdec:
            lyr = v_wdec[key]["layer"]
            print(f"  [{key}] SPATIAL_LAYER_WDEC steer (L{lyr}→L{lyr})...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/SPATIAL_LAYER_WDEC", v_wdec[key]["vec"], lyr,
                rel_vis, rel_labels, base_acc,
                f"{key}_spatial_layer_wdec", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )

        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("True CAA SPATIAL_LAYER_WDEC — mix→mix")
    print(f"{'='*70}")
    print(f"  {'Feature':<16} {'N':>5} {'Base':>7}  {'SPAT_WDEC Δ':>12}")
    print("  " + "-"*45)
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        base = all_results.get(f"{key}_base", {})
        if not base: continue
        ba = base["acc"]; n = base["n"]
        r = all_results.get(f"{key}_spatial_layer_wdec", {})
        best = f"{max(v.get('delta',-999) for v in r.values()):+.2f}%" if r else "—"
        print(f"  {key:<16} {n:>5} {ba:>6.2f}%  {best:>12}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
