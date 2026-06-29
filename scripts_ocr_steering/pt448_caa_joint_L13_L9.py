#!/usr/bin/env python3
"""
Joint alpha × start-layer sweep for L13/F15219 and L9/F387.

Known best results:
  L13/F15219: caa_sae_down +1.55%@α=0.5 (start=13 default), W_dec@L3 +2.68%
              startlayer_v2 best_start=22 Δ=+1.69% (only α=1.0 was tested there)
  L9/F387:    caa_sae_down +2.92%@α=0.5 (start=9 default)
              startlayer_v2 best_start=18 Δ=+2.50% (only α=1.0 was tested there)

Sweep configs:
  L13: alpha=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] × start=[0,1,3,5,8,10,13,16,20,22]
  L9:  alpha=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] × start=[0,1,3,5,8,9,12,15,18,20]

Usage:
    CUDA_VISIBLE_DEVICES=4 GPU_FEATURES=L13 python3 pt448_caa_joint_L13_L9.py  (GPU4)
    CUDA_VISIBLE_DEVICES=5 GPU_FEATURES=L9  python3 pt448_caa_joint_L13_L9.py  (GPU5)

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_joint_alphastart/
"""

import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_PT    = "google/paligemma2-3b-pt-448"
N_LAYERS    = 26
CAA_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_joint_alphastart")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

_RUN = os.environ.get("GPU_FEATURES", "BOTH")

FEATURES_ALL = [
    (13, 15219, ["behind"],                             +2.68),
    (9,  387,   ["at the right side of"],               +2.92),
]

FEATURES = [f for f in FEATURES_ALL if _RUN == "BOTH" or f"L{f[0]}" in _RUN]

SWEEP_CONFIG = {
    "L13_F15219": (
        [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        [0, 1, 3, 5, 8, 10, 13, 16, 20, 22],
    ),
    "L9_F387": (
        [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        [0, 1, 3, 5, 8, 9, 12, 15, 18, 20],
    ),
}


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


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

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
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations, known_best in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"joint_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists():
            print(f"[SKIP - no CAA] {key}", flush=True)
            continue

        caa_saved = torch.load(caa_path)
        caa_data = caa_saved["caa_data"]

        layer_vecs = {}
        for l in range(N_LAYERS):
            if l in caa_data:
                layer_vecs[l] = caa_data[l]["v_caa_norm"].to(model_dtype).to(device)
            else:
                layer_vecs[l] = caa_data[layer_idx]["v_caa_norm"].to(model_dtype).to(device)

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            print(f"[BASE] [{rel_key}] N={len(indices)}...", flush=True)
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
            acc = correct / max(total, 1) * 100
            mg = sum(margins) / max(len(margins), 1)
            baseline_cache[rel_key] = (acc, mg, total)
            print(f"[BASE] {acc:.2f}% margin={mg:.3f}", flush=True)
        base_acc, base_mg, n_total = baseline_cache[rel_key]

        alphas, starts = SWEEP_CONFIG[key]
        print(f"\n[{key}] Joint alpha×start | known_best={known_best:+.2f}%", flush=True)
        print(f"  alphas={alphas}\n  starts={starts}", flush=True)

        result = {
            "layer_idx": layer_idx, "feature_idx": feature_idx, "relations": relations,
            "known_best": known_best,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "grid": {}
        }

        best_da = -999
        best_combo = None

        for alpha in alphas:
            result["grid"][str(alpha)] = {}
            for start in starts:
                inj_layers = list(range(start, N_LAYERS))
                correct = total = 0; margins = []
                for vi in indices:
                    ex = vsr_all[vi]; img = _load_image(ex)
                    if img is None: continue
                    label = int(ex.get("label", 0))
                    try:
                        iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                             processor, nns_model._module, device=device)
                        _, img_end = get_image_token_positions(iids)
                        with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                            for l in inj_layers:
                                v_l = layer_vecs[l]
                                v_col = v_l.unsqueeze(1)
                                lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                                ones = (lo @ v_col) * 0.0 + 1.0
                                lo += alpha * ones * v_l
                            logits_s = nns_model.output.logits.save()
                        pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                        margins.append(m if label == 1 else -m)
                    except Exception: pred = 0; margins.append(0.0)
                    total += 1; correct += (pred == label)

                acc = correct / max(total, 1) * 100
                mg = sum(margins) / max(len(margins), 1)
                da = acc - base_acc; dm = mg - base_mg
                result["grid"][str(alpha)][str(start)] = {
                    "acc": acc, "delta_acc": da, "margin": mg, "delta_margin": dm,
                    "n_injection_layers": len(inj_layers)
                }
                marker = " *** NEW BEST ***" if da > best_da else ""
                if da > best_da:
                    best_da = da; best_combo = (alpha, start)
                print(f"  α={alpha:+.3f} start={start:2d} ({len(inj_layers):2d}L): {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f}{marker}", flush=True)

        result["best_alpha"] = best_combo[0] if best_combo else None
        result["best_start"] = best_combo[1] if best_combo else None
        result["best_delta_acc"] = best_da
        print(f"\n  BEST: α={best_combo[0]} start={best_combo[1]} Δ={best_da:+.2f}% (vs known {known_best:+.2f}%)", flush=True)

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        del layer_vecs; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("Joint Alpha×Start Sweep (L13+L9) — Summary")
    print(f"{'='*80}")
    print(f"{'Feature':<20} {'Relation':<35} {'BestΔ':>8} {'α':>6} {'Start':>6} {'Known':>8}")
    print("-" * 80)
    for r in all_results:
        key = f"L{r['layer_idx']}/F{r['feature_idx']}"
        print(f"{key:<20} {str(r['relations']):<35} {r['best_delta_acc']:>+8.2f}% "
              f"{str(r['best_alpha']):>6} {str(r['best_start']):>6} {r['known_best']:>+8.2f}%")


if __name__ == "__main__":
    main()
