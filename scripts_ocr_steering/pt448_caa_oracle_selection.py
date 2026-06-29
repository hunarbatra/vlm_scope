#!/usr/bin/env python3
"""
Oracle-selection injection experiment.

Uses the per-relation steering matrix to implement a theoretical ceiling:
for each VSR sample, apply the best-known feature for that sample's relation.

Best-feature-per-relation mapping derived from pt448_caa_per_relation_steer.py
(will be updated as that experiment completes; uses partial results if needed).

This answers: "What is the theoretical maximum VSR accuracy if we always apply
the best steering feature for each relation?"

Also tests: relation-aware gating (skip injection for known-harmful relations
like 'against') vs. always-inject.

All evaluated on full VSR dataset.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_oracle_selection/

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 pt448_caa_oracle_selection.py
"""

import os, sys, json, hashlib, warnings, math
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_oracle_selection")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"
PER_RELATION_JSON = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_per_relation/per_relation_steer.json")

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# All 8 features with confirmed optimal configs
ALL8 = [
    (4,  14233, 1.0,  0),
    (14, 10561, 2.0,  0),
    (12, 2257,  1.0,  1),
    (15, 220,   0.75, 15),
    (11, 12278, 0.5,  5),
    (9,  387,   0.5,  1),
    (6,  7539,  1.5,  1),
    (9,  7540,  0.25, 9),
]

# Fallback best-feature map from known partial per-relation results
# Updated as per-relation experiment completes.
# Format: relation -> (layer_idx, feature_idx) key of best feature
FALLBACK_BEST_MAP = {
    "above":       "L15_F220",
    "across from": "L6_F7539",
    "adjacent to": "L9_F387",
    "against":     None,       # skip — all features damage this relation
    "ahead of":    "L4_F14233",
    "alongside":   "L6_F7539",
}

# Relations known to be harmful for all features — skip injection
SKIP_RELATIONS = {"against"}


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


def build_oracle_map():
    """Load per-relation steer JSON if available; else use fallback."""
    if PER_RELATION_JSON.exists():
        try:
            data = json.load(open(PER_RELATION_JSON))
            oracle_map = {}
            for relation, r in data["relations"].items():
                best_key = r["best_feature"]
                best_delta = r["best_delta"]
                # Skip relations where best feature still hurts
                if best_delta <= 0:
                    oracle_map[relation] = None
                else:
                    oracle_map[relation] = best_key
            print(f"[ORACLE] Loaded {len(oracle_map)} relations from per-relation JSON", flush=True)
            return oracle_map
        except Exception as e:
            print(f"[ORACLE] Failed to load per-relation JSON: {e}, using fallback", flush=True)
    else:
        print(f"[ORACLE] Per-relation JSON not ready yet, using fallback map", flush=True)
    return dict(FALLBACK_BEST_MAP)


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    result_path = OUT_DIR / "oracle_selection_results.json"
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
    all_indices = list(range(len(vsr_all)))

    # Load all CAA vectors
    feature_vecs = {}
    for layer_idx, feature_idx, alpha_opt, start_layer in ALL8:
        key = f"L{layer_idx}_F{feature_idx}"
        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists(): continue
        caa_data = torch.load(caa_path)["caa_data"]
        layer_vecs = {}
        for l in range(N_LAYERS):
            if l in caa_data:
                layer_vecs[l] = caa_data[l]["v_caa_norm"].to(model_dtype).to(device)
            else:
                layer_vecs[l] = caa_data[layer_idx]["v_caa_norm"].to(model_dtype).to(device)
        feature_vecs[key] = (layer_vecs, alpha_opt, start_layer)
        print(f"[LOADED] {key}", flush=True)

    # Build oracle map
    oracle_map = build_oracle_map()
    print(f"[ORACLE] Map: {oracle_map}", flush=True)

    # Baseline
    print("\n[BASELINE] Full VSR...", flush=True)
    correct = total = 0; margins = []
    for vi in all_indices:
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
        if total % 1000 == 0: print(f"  baseline {total}", flush=True)
    base_acc = correct / max(total, 1) * 100
    base_mg = sum(margins) / max(len(margins), 1)
    print(f"[BASELINE] {base_acc:.2f}% margin={base_mg:.3f} N={total}", flush=True)

    results = {"baseline": {"acc": base_acc, "margin": base_mg, "n": total}, "oracle": {}}

    # Oracle injection: per-sample, inject best feature for that sample's relation
    for mode in ["oracle_inject", "oracle_gated"]:
        # oracle_inject: inject best feature, skip if no best feature
        # oracle_gated: same but also skip if relation is in SKIP_RELATIONS
        print(f"\n[{mode.upper()}] Oracle injection...", flush=True)
        correct = total = n_injected = n_skipped = 0; margins = []
        for vi in all_indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            relation = ex.get("relation", "")

            # Determine which feature to inject
            best_key = oracle_map.get(relation, None)
            skip_this = (best_key is None)
            if mode == "oracle_gated" and relation in SKIP_RELATIONS:
                skip_this = True

            try:
                iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                     processor, model_raw, device=device)
                if skip_this or best_key not in feature_vecs:
                    with torch.inference_mode():
                        out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                    pred, m = _pm(out.logits[0, -1, :], yes_ids, no_ids)
                    n_skipped += 1
                else:
                    layer_vecs, alpha, start_layer = feature_vecs[best_key]
                    _, img_end = get_image_token_positions(iids)
                    with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                        for l in range(start_layer, N_LAYERS):
                            v_l = layer_vecs[l]
                            v_col = v_l.unsqueeze(1)
                            lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                            ones = (lo @ v_col) * 0.0 + 1.0
                            lo += alpha * ones * v_l
                        logits_s = nns_model.output.logits.save()
                    pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                    n_injected += 1
                margins.append(m if label == 1 else -m)
            except Exception: pred = 0; margins.append(0.0)
            total += 1; correct += (pred == label)
            if total % 1000 == 0: print(f"  {mode} {total} (injected={n_injected}, skipped={n_skipped})", flush=True)

        acc = correct / max(total, 1) * 100
        mg = sum(margins) / max(len(margins), 1)
        da = acc - base_acc
        print(f"[{mode.upper()}] {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f} N={total} injected={n_injected} skipped={n_skipped}", flush=True)
        results["oracle"][mode] = {"acc": acc, "delta_acc": da, "margin": mg, "n": total,
                                    "n_injected": n_injected, "n_skipped": n_skipped}

    # Also test: what if we inject a fixed best-overall feature (L4/F14233 +15.38%) on all samples
    print(f"\n[FIXED_L4] L4/F14233 on all VSR (single best feature)...", flush=True)
    if "L4_F14233" in feature_vecs:
        layer_vecs, alpha, start_layer = feature_vecs["L4_F14233"]
        correct = total = 0; margins = []
        for vi in all_indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption", ""))),
                                                     processor, model_raw, device=device)
                _, img_end = get_image_token_positions(iids)
                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l in range(start_layer, N_LAYERS):
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
            if total % 1000 == 0: print(f"  fixed_L4 {total}", flush=True)
        acc = correct / max(total, 1) * 100
        mg = sum(margins) / max(len(margins), 1)
        da = acc - base_acc
        print(f"[FIXED_L4] {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f}", flush=True)
        results["fixed_L4_all_vsr"] = {"acc": acc, "delta_acc": da, "margin": mg, "n": total}

    with open(result_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved to {result_path}", flush=True)
    _print_summary(results)


def _print_summary(results):
    base = results["baseline"]["acc"]
    print(f"\n{'='*70}")
    print("Oracle Selection Summary")
    print(f"{'='*70}")
    print(f"Baseline: {base:.2f}%")
    for name, r in results.get("oracle", {}).items():
        print(f"  {name}: {r['acc']:.2f}% (Δ={r['delta_acc']:+.2f}%) injected={r.get('n_injected','?')} skipped={r.get('n_skipped','?')}")
    if "fixed_L4_all_vsr" in results:
        r = results["fixed_L4_all_vsr"]
        print(f"  fixed_L4_all_vsr: {r['acc']:.2f}% (Δ={r['delta_acc']:+.2f}%)")


if __name__ == "__main__":
    main()
