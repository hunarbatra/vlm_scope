#!/usr/bin/env python3
"""
Combined CAA + W_dec injection test.

Tests whether adding the SAE decoder vector W_dec[f] alongside the CAA direction
improves over pure caa_sae_down.

Injection formula:
  lo += alpha * v_caa_norm[l] + beta * v_wdec[l]

where:
  - v_caa_norm[l]: normalized per-layer CAA direction (as in caa_sae_down)
  - v_wdec[l]: W_dec decoder vector (at SAE layer, held constant across all layers)

Tests on per-relation subsets (own-relation, not full VSR) to avoid collapse.

Grid sweep:
  alpha in [0.5, 1.0, 1.5, 2.0]  (CAA component, optimal range from caa_sae_down)
  beta  in [0.0, 0.5, 1.0, 2.0, 5.0]  (W_dec component, 0.0 = pure CAA baseline)

Features tested: top 5 with best caa_sae_down results
  L4/F14233  (ahead of, α*=1.0)
  L14/F10561 (close to, α*=2.0)
  L12/F2257  (facing,   α*=1.0)
  L11/F12278 (touching, α*=0.5)
  L15/F220   (across from / at left side, α*=0.75)

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_wdec_combined_v2/

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 pt448_caa_wdec_combined_v2.py
"""

import os, sys, json, hashlib, warnings, math
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_PT    = "google/paligemma2-3b-pt-448"
N_LAYERS    = 26
CAA_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_wdec_combined_v2")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# (layer_idx, feature_idx, relations, caa_alpha_opt, start_layer)
FEATURES = [
    (4,  14233, ["ahead of"],                             1.0,  0),
    (14, 10561, ["close to"],                             2.0,  0),
    (12, 2257,  ["facing"],                               1.0,  1),
    (11, 12278, ["touching"],                             0.5,  5),
    (15, 220,   ["across from", "at the left side of"],   0.75, 15),
]

# Grid: alpha (CAA weight), beta (W_dec weight)
ALPHAS = [0.5, 1.0, 1.5, 2.0]
BETAS  = [0.0, 0.5, 1.0, 2.0, 5.0]  # 0.0 = pure caa_sae_down (baseline)


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No","No"," no","NO"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    ov = yes_ids & no_ids; yes_ids -= ov; no_ids -= ov
    return yes_ids, no_ids

def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n; p = max(y/d if d > 0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1-p, 1e-7))

def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB"); img.save(cp, "JPEG")
        return img
    except Exception: return None


def eval_combined(indices, vsr_all, caa_vecs, wdec_vecs, alpha, beta, start_layer,
                  nns_model, yes_ids, no_ids, processor, base_module, model_dtype, device):
    """Inject alpha * v_caa_norm[l] + beta * v_wdec (fixed at SAE layer) for each layer l."""
    from utils import process_vlm_inputs, get_image_token_positions
    correct = total = 0; margins = []
    for vi in indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        try:
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                 processor, base_module, device=device)
            _, img_end = get_image_token_positions(iids)
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in range(start_layer, N_LAYERS):
                    v_caa = caa_vecs[l]
                    v_wdec = wdec_vecs[l]
                    lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                    # Proxy trick for both directions
                    v_caa_col = v_caa.unsqueeze(1)
                    v_wdec_col = v_wdec.unsqueeze(1)
                    ones_caa  = (lo @ v_caa_col)  * 0.0 + 1.0
                    ones_wdec = (lo @ v_wdec_col) * 0.0 + 1.0
                    lo += alpha * ones_caa * v_caa + beta * ones_wdec * v_wdec
                logits_s = nns_model.output.logits.save()
            pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
            margins.append(m if label == 1 else -m)
        except Exception: pred = 0; margins.append(0.0)
        total += 1; correct += (pred == label)
    acc = correct / max(total, 1) * 100
    mg = sum(margins) / max(len(margins), 1)
    return acc, mg, total


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    result_path = OUT_DIR / "caa_wdec_combined_results.json"
    if result_path.exists():
        print("[SKIP] Results already exist", flush=True)
        with open(result_path) as f: results = json.load(f)
        _print_summary(results)
        return

    print(f"[INFO] Loading {MODEL_PT}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PT)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    from collections import defaultdict
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    results = {}

    for layer_idx, feature_idx, relations, alpha_opt, start_layer in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists():
            print(f"[SKIP] No CAA for {key}", flush=True); continue

        caa_saved = torch.load(caa_path)
        caa_data = caa_saved["caa_data"]

        # Build per-layer vectors
        caa_vecs  = {}  # normalized CAA per layer
        wdec_vecs = {}  # W_dec (use SAE-layer W_dec broadcast to all layers)
        wdec_sae = caa_data[layer_idx]["v_wdec"].to(model_dtype).to(device)
        wdec_norm = wdec_sae / wdec_sae.norm().clamp(min=1e-8)
        for l in range(N_LAYERS):
            src = caa_data[l] if l in caa_data else caa_data[layer_idx]
            caa_vecs[l]  = src["v_caa_norm"].to(model_dtype).to(device)
            wdec_vecs[l] = wdec_norm  # same W_dec direction at all layers

        # Build relation sample indices
        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices:
            print(f"[SKIP] {key} — no samples", flush=True); continue

        # Baseline
        print(f"\n[{key}] Relations={relations} N={len(indices)}", flush=True)
        correct = total = 0; margins = []
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                     processor, model_raw, device=device)
                with torch.inference_mode():
                    out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, m = _pm(out.logits[0, -1, :], yes_ids, no_ids)
                margins.append(m if label == 1 else -m)
            except Exception: pred = 0; margins.append(0.0)
            total += 1; correct += (pred == label)
        base_acc = correct / max(total, 1) * 100
        base_mg  = sum(margins) / max(len(margins), 1)
        print(f"  BASE: {base_acc:.2f}% margin={base_mg:.3f}", flush=True)

        results[key] = {"base_acc": base_acc, "base_mg": base_mg, "n": total,
                        "alpha_opt": alpha_opt, "grid": {}}
        best_da = -999; best_config = None

        # Grid sweep
        for alpha in ALPHAS:
            for beta in BETAS:
                acc, mg, n = eval_combined(
                    indices, vsr_all, caa_vecs, wdec_vecs, alpha, beta, start_layer,
                    nns_model, yes_ids, no_ids, processor, nns_model._module, model_dtype, device
                )
                da = acc - base_acc
                marker = " *** BEST ***" if da > best_da else ""
                if da > best_da: best_da = da; best_config = (alpha, beta)
                label_str = f"α={alpha} β={beta}"
                print(f"  {label_str:15s}: {acc:.2f}% (Δ={da:+.2f}%) mg={mg:.3f}{marker}", flush=True)
                results[key]["grid"][f"a{alpha}_b{beta}"] = {"acc": acc, "delta": da, "mg": mg}

        results[key]["best_delta"] = best_da
        results[key]["best_alpha"] = best_config[0] if best_config else None
        results[key]["best_beta"]  = best_config[1] if best_config else None
        print(f"  BEST: α={best_config[0]} β={best_config[1]} Δ={best_da:+.2f}%", flush=True)

    with open(result_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved to {result_path}", flush=True)
    _print_summary(results)


def _print_summary(results):
    print(f"\n{'='*70}")
    print("CAA + W_dec Combined Injection Summary")
    print(f"{'='*70}")
    print(f"{'Feature':<15} {'Base':>7} {'PureCAA':>9} {'BestComb':>10} {'Best α':>7} {'Best β':>7}")
    print("-" * 70)
    for key, r in results.items():
        base = r["base_acc"]
        # pure CAA = best among beta=0.0 at any alpha
        pure_caa_da = max(
            r["grid"][f"a{a}_b0.0"]["delta"] for a in [0.5, 1.0, 1.5, 2.0]
            if f"a{a}_b0.0" in r["grid"]
        ) if r["grid"] else 0.0
        best_da = r.get("best_delta", 0.0)
        ba = r.get("best_alpha", "-")
        bb = r.get("best_beta", "-")
        print(f"{key:<15} {base:>7.2f}% {pure_caa_da:>+9.2f}% {best_da:>+10.2f}% {str(ba):>7} {str(bb):>7}")


if __name__ == "__main__":
    main()
