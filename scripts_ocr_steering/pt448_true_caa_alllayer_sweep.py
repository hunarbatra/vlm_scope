#!/usr/bin/env python3
"""
True CAA all-layer injection sweep.

For each of 10 spatial features:
  1. Compute MIDDLE vector at L13 from mix-448 cache (mean-over-text-tokens).
  2. Try injecting that vector at each of the 26 layers of pt-448.
  3. Report best injection layer and corresponding accuracy delta.

Hypothesis: The optimal injection layer for pt-448 may differ from the
vector extraction layer (L13 of mix). This sweep finds the best injection
point for cross-model transfer without oracle per-feature knowledge.

Compare to:
  - pt448_true_caa_v4_mix_to_pt.py (always injects at L13 = extraction layer)
  - pt448_caa_startlayer_v2 (used SAE recon delta vectors, not True CAA)

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 -B pt448_true_caa_alllayer_sweep.py
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
PT_MODEL     = "google/paligemma2-3b-pt-448"
MIDDLE_LAYER = 13
BEST_ALPHA   = 10.0  # single alpha for speed; prior CAA shows ~10 optimal for mid-range features

MIX_HIDDEN_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_alllayer_sweep")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [2.0, 5.0, 10.0, 20.0]  # reduced sweep, just finding best layer

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

# Injection layers to test
INJECTION_LAYERS = list(range(0, 26))

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


# ─────────────────────── Vector computation ───────────────────
def compute_middle_vector(vsr_all):
    print(f"  MIDDLE: L{MIDDLE_LAYER} over all {len(vsr_all)} samples...", flush=True)
    pos_sum = neg_sum = None
    pos_n = neg_n = 0
    for vi in range(len(vsr_all)):
        label = int(vsr_all[vi].get("label", 0))
        v = _load_mix_hidden_layer(vi, MIDDLE_LAYER)
        if v is None: continue
        if label == 1:
            pos_sum = v.clone() if pos_sum is None else pos_sum + v; pos_n += 1
        else:
            neg_sum = v.clone() if neg_sum is None else neg_sum + v; neg_n += 1
        if (vi + 1) % 2000 == 0:
            print(f"    {vi+1}/{len(vsr_all)}...", flush=True)
    if pos_n == 0 or neg_n == 0:
        return None
    v_mid = pos_sum / pos_n - neg_sum / neg_n
    print(f"  MIDDLE pos={pos_n}, neg={neg_n}, norm={v_mid.norm():.4f}", flush=True)
    return v_mid


# ─────────────────────── Per-layer sweep ──────────────────────
def eval_injection_layer(inj_layer, sv_gpu, rel_vis, rel_labels,
                          model, processor, yes_ids, no_ids, device, vsr_all):
    from utils import process_vlm_inputs, get_image_token_positions

    img_end_r = [0]

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
            hook_h = model.model.language_model.layers[inj_layer].register_forward_hook(make_hook())
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

    return correct / max(total, 1) * 100, total


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("True CAA all-layer injection sweep")
    print(f"mix-448 MIDDLE L{MIDDLE_LAYER} vector → sweep pt-448 injection layer")
    print(f"alphas: {ALPHAS}, layers: 0–25")
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

    print("\n[STEP 1] Computing MIDDLE vector from mix-448 cache...", flush=True)
    v_middle = compute_middle_vector(vsr_all)
    if v_middle is None:
        print("ERROR: Could not compute MIDDLE vector", flush=True)
        return
    sv_norm = v_middle / v_middle.norm().clamp(min=1e-8)

    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)
    gc.collect()

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    for sf in SPATIAL_FEATURES:
        key, feat_layer = sf["key"], sf["layer"]

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
            print(f"\n[{key}] pt-448 baseline (n={len(rel_vis)}, rel={relations})...", flush=True)
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
            all_results[base_key] = {"acc": base_acc, "n": bt, "relations": relations, "feat_layer": feat_layer}
            print(f"  [{key}] baseline: {base_acc:.2f}% (n={bt})", flush=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            base_acc = all_results[base_key]["acc"]
            print(f"\n[{key}] baseline (cached): {base_acc:.2f}%  feat_layer=L{feat_layer}", flush=True)

        # Sweep injection layers × alphas
        sweep_key = f"{key}_layer_sweep"
        if sweep_key not in all_results:
            all_results[sweep_key] = {}

        for alpha in ALPHAS:
            akey = str(alpha)
            if akey not in all_results[sweep_key]:
                all_results[sweep_key][akey] = {}

            sv_gpu = (sv_norm * alpha).to(next(model.parameters()).dtype).to(device)
            best_delta = -999
            best_layer = -1

            for inj_layer in INJECTION_LAYERS:
                lkey = str(inj_layer)
                if lkey in all_results[sweep_key][akey] and \
                        all_results[sweep_key][akey][lkey].get("n", 0) > 0:
                    delta = all_results[sweep_key][akey][lkey].get("delta", -999)
                    if delta > best_delta:
                        best_delta = delta; best_layer = inj_layer
                    continue

                acc, n = eval_injection_layer(
                    inj_layer, sv_gpu, rel_vis, rel_labels,
                    model, processor, yes_ids, no_ids, device, vsr_all
                )
                delta = acc - base_acc
                all_results[sweep_key][akey][lkey] = {"acc": acc, "delta": delta, "n": n}
                if delta > best_delta:
                    best_delta = delta; best_layer = inj_layer

            print(f"  [{key}] α={alpha}: best_layer=L{best_layer} delta={best_delta:+.2f}%", flush=True)

        # Per-alpha best summary
        best_overall = -999
        best_al = (-1, -1)
        for akey, ldata in all_results[sweep_key].items():
            for lkey, r in ldata.items():
                d = r.get("delta", -999)
                if d > best_overall:
                    best_overall = d
                    best_al = (float(akey), int(lkey))

        all_results[sweep_key]["_best"] = {
            "alpha": best_al[0], "layer": best_al[1], "delta": best_overall
        }
        print(f"  [{key}] BEST: α={best_al[0]} L{best_al[1]} Δ={best_overall:+.2f}%", flush=True)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("True CAA All-Layer Sweep  mix→pt (MIDDLE vector, sweep injection)")
    print(f"{'='*70}")
    print(f"  {'Feature':<16} {'feat_L':>7} {'base':>7} {'bestΔ':>8} {'bestL':>7} {'bestα':>7}")
    print("  " + "-"*55)
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        base = all_results.get(f"{key}_base", {})
        if not base: continue
        sw = all_results.get(f"{key}_layer_sweep", {})
        best = sw.get("_best", {})
        print(f"  {key:<16} L{base.get('feat_layer',sf['layer']):>2}  "
              f"{base['acc']:>6.2f}%  {best.get('delta',-999):>+.2f}%  "
              f"L{best.get('layer',-1):>2}  α={best.get('alpha',-1)}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
