#!/usr/bin/env python3
"""
True CAA middle-layer baseline.

Computes label-contrastive vector at the middle layer (layer 13 of 26)
from mix-448 hidden states, then steers target model on every R(F)
relation subset using that single vector.

This is the standard CAA baseline when no layer-selection knowledge
is used. Compared against pt448_true_caa_v3.py (which uses SAE-identified
layer lF per feature), this tests whether knowing the right layer matters.

Usage:
    CUDA_VISIBLE_DEVICES=2 TARGET=pt  python3 -B pt448_true_caa_middle.py
    CUDA_VISIBLE_DEVICES=3 TARGET=mix python3 -B pt448_true_caa_middle.py
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
TARGET       = os.environ.get("TARGET", "pt").lower()
MIDDLE_LAYER = 13  # middle of 26 layers (0-25)

assert TARGET in ("pt", "mix"), f"TARGET must be pt or mix, got {TARGET!r}"

PT_MODEL  = "google/paligemma2-3b-pt-448"
MIX_MODEL = "google/paligemma2-3b-mix-448"

MIX_HIDDEN_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_BASE       = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_middle")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

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

def _load_h_mix_layer(vi, layer):
    path = MIX_HIDDEN_DIR / f"vi_{vi:05d}.pt"
    if not path.exists(): return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d[layer].float() if layer in d else None
    except Exception:
        return None


# ─────────────────────── Compute middle-layer vector ──────────
def compute_middle_vector(vsr_all):
    print(f"[STEP 1] Global True CAA vector at layer {MIDDLE_LAYER} from mix-448...", flush=True)
    pos_sum = neg_sum = None
    pos_n = neg_n = 0

    for vi in range(len(vsr_all)):
        label = int(vsr_all[vi].get("label", 0))
        v = _load_h_mix_layer(vi, MIDDLE_LAYER)
        if v is None: continue
        if label == 1:
            pos_sum = v.clone() if pos_sum is None else pos_sum + v
            pos_n  += 1
        else:
            neg_sum = v.clone() if neg_sum is None else neg_sum + v
            neg_n  += 1
        if (vi + 1) % 2000 == 0:
            print(f"  {vi+1}/{len(vsr_all)}...", flush=True)

    assert pos_n > 0 and neg_n > 0, "No samples found"
    v_mid = pos_sum / pos_n - neg_sum / neg_n
    print(f"  pos={pos_n}, neg={neg_n}, norm={v_mid.norm():.4f}", flush=True)
    return v_mid


# ─────────────────────── Hook-based inference ─────────────────
def run_steer_sweep(key, steer_vec, rel_vis, rel_labels, base_acc,
                    all_results, results_path, model_raw, processor,
                    yes_ids, no_ids, device, vsr_all):
    from utils import process_vlm_inputs, get_image_token_positions

    sv_norm   = steer_vec / steer_vec.norm().clamp(min=1e-8)
    img_end_r = [0]
    result_key = f"{key}_middle"

    for alpha in ALPHAS:
        akey = str(alpha)
        if akey in all_results.get(result_key, {}) and \
                all_results[result_key][akey].get("n", 0) > 0:
            r = all_results[result_key][akey]
            print(f"  [SKIP {key}/MIDDLE] α={alpha}: {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
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
                hook_h = model_raw.model.language_model.layers[MIDDLE_LAYER].register_forward_hook(make_hook())
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
        print(f"  [{key}/MIDDLE] α={alpha}: {acc:.2f}% Δ={delta:+.2f}% ({correct}/{total})", flush=True)

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
    print(f"True CAA Middle-Layer Baseline  TARGET={TARGET}-448  layer={MIDDLE_LAYER}")
    print(f"v_global[{MIDDLE_LAYER}] from mix-448, applied to all 10 R(F) subsets")
    print("=" * 70, flush=True)

    device  = "cuda:0"
    out_dir = OUT_BASE / f"mix_to_{TARGET}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    # ── Load VSR ──
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    # ── Compute middle-layer vector (pure cache reads) ──
    v_mid = compute_middle_vector(vsr_all)
    print(f"  v_mid norm={v_mid.norm():.4f}", flush=True)

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
        key      = sf["key"]
        feat_layer = sf["layer"]

        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            print(f"\n[{key}] acts missing — skip", flush=True)
            continue
        acts_data  = json.load(open(acts_path))
        acts       = acts_data.get("acts", {})
        relations  = acts_data.get("relations", [])
        rel_vis    = [int(k) for k in acts.keys()]
        rel_labels = [int(vsr_all[vi].get("label", 0)) for vi in rel_vis]

        # ── Baseline ──
        base_key = f"{key}_base"
        if base_key not in all_results:
            print(f"\n[{key}] {TARGET}-448 baseline on R(F) (n={len(rel_vis)}, "
                  f"rel={relations})...", flush=True)
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
            all_results[base_key] = {"acc": base_acc, "n": bt, "relations": relations,
                                     "feature_layer": feat_layer}
            print(f"  [{key}] baseline: {base_acc:.2f}% (n={bt})", flush=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            base_acc = all_results[base_key]["acc"]
            print(f"\n[{key}] baseline (cached): {base_acc:.2f}%  "
                  f"rel={relations}  feat_layer={feat_layer}", flush=True)

        # ── Middle-layer steer ──
        print(f"  [{key}] MIDDLE (L{MIDDLE_LAYER}) steer...", flush=True)
        all_results = run_steer_sweep(
            key, v_mid, rel_vis, rel_labels, base_acc,
            all_results, results_path,
            model_raw, processor, yes_ids, no_ids, device, vsr_all,
        )

        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"True CAA Middle-Layer Baseline (L{MIDDLE_LAYER})  mix→{TARGET}")
    print(f"{'='*70}")
    print(f"  {'Feature':<16} {'feat_L':>7} {'Base':>7} {'Middle Δ':>10} {'N':>6}")
    print("  " + "-" * 52)
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        base = all_results.get(f"{key}_base", {})
        if not base: continue
        base_acc   = base["acc"]
        feat_layer = base.get("feature_layer", sf["layer"])
        r = all_results.get(f"{key}_middle", {})
        if r:
            best_d = max(v.get("delta", -999) for v in r.values())
            best_n = max(v.get("n", 0)        for v in r.values())
            mid_str = f"{best_d:+.2f}%"
        else:
            mid_str = "—"; best_n = 0
        print(f"  {key:<16} L{feat_layer:>2}      {base_acc:>6.2f}%  {mid_str:>10}  {best_n:>6}")

    print(f"\nSaved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
