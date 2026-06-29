#!/usr/bin/env python3
"""
Combined CAA + W_dec injection experiment.

Key observation: CAA and W_dec are nearly orthogonal (cos ≈ 0.04).
Since they span different subspaces of the residual stream, their effects
might be approximately additive when injected together.

Test: inject both CAA (sae_down strategy) AND W_dec (sae_only_down strategy)
simultaneously and see if gains combine.

Also tests: CAA-only sae_down vs W_dec-only sae_only_down vs combined.

Focus on L4/F14233 "ahead of" where both have strong individual results:
  - CAA caa_sae_down: +15.38% @ α=1
  - W_dec sae_only_down: +10.26% @ α=4
  Combined hypothesis: possibly +17-18% if truly orthogonal/additive

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_wdec_combined/

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 pt448_caa_wdec_combined.py
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

MODEL_PT       = "google/paligemma2-3b-pt-448"
N_LAYERS       = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
CAA_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_wdec_combined")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Features and their best known individual configs
FEATURES = [
    # (layer_idx, feature_idx, relations, best_caa_alpha, best_wdec_alpha, caa_best_delta, wdec_best_delta)
    (4,  14233, ["ahead of"],                            1.0, 4.0, +15.38, +10.26),
    (12, 2257,  ["facing"],                              1.0, 50.0, +8.50, +3.92),
    (11, 12278, ["touching"],                            1.0, 25.0, None, +3.36),  # CAA result TBD
    (9,  387,   ["at the right side of"],                1.0, 2.0, None, +3.12),   # CAA result TBD
    (13, 15219, ["behind"],                              1.0, 30.0, None, +2.12),   # CAA result TBD
]

# Alpha grid for combined injection
# (alpha_caa, alpha_wdec) pairs to test
COMBINED_ALPHAS = [
    (0.5, 2.0), (0.5, 4.0), (0.5, 6.0),
    (1.0, 2.0), (1.0, 4.0), (1.0, 6.0),
    (1.5, 2.0), (1.5, 4.0), (1.5, 6.0),
    (2.0, 2.0), (2.0, 4.0),
]


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


def run_injection(nns_model, indices, vsr_all, processor, model_raw, img_end_ref,
                  v_caa, v_wdec, alpha_caa, alpha_wdec, layer_idx, yes_ids, no_ids, device):
    """Run injection and return (acc, margin) for a specific alpha pair."""
    from utils import process_vlm_inputs, get_image_token_positions
    v_caa_col = v_caa.unsqueeze(1)
    v_wdec_col = v_wdec.unsqueeze(1)
    inj_layers = list(range(layer_idx, N_LAYERS))

    correct = total = 0; margins = []
    for vi in indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        try:
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                 processor, model_raw, device=device)
            _, img_end = get_image_token_positions(iids)
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in inj_layers:
                    lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                    ones = (lo @ v_caa_col) * 0.0 + 1.0
                    lo += alpha_caa * ones * v_caa
                    lo2 = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                    ones2 = (lo2 @ v_wdec_col) * 0.0 + 1.0
                    lo2 += alpha_wdec * ones2 * v_wdec
                logits_s = nns_model.output.logits.save()
            pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
            margins.append(m if label == 1 else -m)
        except Exception: pred = 0; margins.append(0.0)
        total += 1; correct += (pred == label)
    return correct / max(total, 1) * 100, sum(margins) / max(len(margins), 1)


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

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

    for layer_idx, feature_idx, relations, best_caa_a, best_wdec_a, caa_best, wdec_best in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"combined_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists():
            print(f"[SKIP - no CAA] {key}", flush=True)
            continue

        caa_saved = torch.load(caa_path)
        v_caa = caa_saved["caa_data"][layer_idx]["v_caa_norm"].to(model_dtype).to(device)

        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        v_wdec = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        v_wdec = v_wdec / v_wdec.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        cos_sim = (v_caa @ v_wdec).item()
        print(f"\n[{key}] cos(caa,wdec)={cos_sim:.4f}", flush=True)

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
        base_acc, base_mg, _ = baseline_cache[rel_key]

        result = {
            "layer_idx": layer_idx, "feature_idx": feature_idx, "relations": relations,
            "cos_caa_wdec": cos_sim,
            "caa_best_known": caa_best, "wdec_best_known": wdec_best,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "combined_results": {}
        }

        v_caa_col = v_caa.unsqueeze(1)
        v_wdec_col = v_wdec.unsqueeze(1)
        inj_layers = list(range(layer_idx, N_LAYERS))

        for alpha_caa, alpha_wdec in COMBINED_ALPHAS:
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
                            lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                            ones = (lo @ v_caa_col) * 0.0 + 1.0
                            lo += alpha_caa * ones * v_caa + alpha_wdec * ones * v_wdec
                        logits_s = nns_model.output.logits.save()
                    pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                    margins.append(m if label == 1 else -m)
                except Exception: pred = 0; margins.append(0.0)
                total += 1; correct += (pred == label)

            acc = correct / max(total, 1) * 100
            mg = sum(margins) / max(len(margins), 1)
            da = acc - base_acc
            k = f"caa{alpha_caa}_wdec{alpha_wdec}"
            result["combined_results"][k] = {"alpha_caa": alpha_caa, "alpha_wdec": alpha_wdec,
                                              "acc": acc, "delta_acc": da, "margin": mg}
            print(f"  caa={alpha_caa:+.2f} wdec={alpha_wdec:+.2f}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f}", flush=True)

        best = max(result["combined_results"].items(), key=lambda x: x[1]["delta_acc"])
        print(f"  BEST combined: {best[0]} → Δ={best[1]['delta_acc']:+.2f}%"
              f"  (CAA-only best: {caa_best if caa_best else '?'}%, W_dec-only best: {wdec_best:+.2f}%)", flush=True)

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        del v_caa, v_wdec, v_caa_col, v_wdec_col; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("CAA + W_dec Combined Injection — Summary")
    print(f"{'='*100}")
    for r in all_results:
        key = f"L{r['layer_idx']}/F{r['feature_idx']}"
        best = max(r['combined_results'].items(), key=lambda x: x[1]['delta_acc'])
        print(f"{key} {r['relations']}: best_combined={best[1]['delta_acc']:+.2f}% @ {best[0]}"
              f"  CAA-only={r['caa_best_known']}%  W_dec-only={r['wdec_best_known']:+.2f}%")


if __name__ == "__main__":
    main()
