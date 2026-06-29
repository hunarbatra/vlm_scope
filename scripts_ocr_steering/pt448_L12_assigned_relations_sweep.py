#!/usr/bin/env python3
"""
Fine alpha sweep for L12/F2257 on all oracle-assigned relations.

L12/F2257 oracle alpha is 0.75 (from cross-feature sweeps).
This script tests all oracle-assigned relations for L12/F2257 to find
per-relation optimal alphas.

Oracle-assigned relations for L12/F2257:
  beneath, beyond, enclosed by, facing, near, off, outside, toward

Also tests L6/F7539 and L15/F220 as cross-feature comparisons.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L12_assigned_sweep/
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_L12_assigned_relations_sweep.py
"""

import os, sys, json, re, warnings, math
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L12_assigned_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Relations currently assigned to L12/F2257 in the oracle
TARGET_RELATIONS = [
    "beneath",
    "beyond",
    "enclosed by",
    "facing",
    "near",
    "off",
    "outside",
    "toward",
]

# Features to test
FEATURE_CONFIGS = {
    "L12_F2257": {"layer": 12, "feature": 2257, "start": 1,  "oracle_alpha": 0.75,
                  "alphas": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]},
    "L6_F7539":  {"layer": 6,  "feature": 7539, "start": 1,  "oracle_alpha": 1.5,
                  "alphas": [0.5, 1.0, 1.5, 2.0, 2.5]},
    "L15_F220":  {"layer": 15, "feature": 220,  "start": 15, "oracle_alpha": 0.7,
                  "alphas": [0.4, 0.6, 0.7, 0.9, 1.25, 1.5]},
}

ALL_RELATIONS = [
    "above", "across from", "adjacent to", "against", "ahead of", "alongside",
    "at the back of", "at the edge of", "at the left side of", "at the right side of",
    "at the side of", "attached to", "away from", "behind", "below", "beneath",
    "beside", "beyond", "by", "close to", "connected to", "consists of", "contains",
    "enclosed by", "facing", "facing away from", "far away from", "far from",
    "has as a part", "in", "in front of", "in the middle of", "inside", "into",
    "left of", "near", "next to", "off", "on", "on top of", "opposite to", "outside",
    "over", "parallel to", "part of", "perpendicular to", "right of", "surrounding",
    "touching", "toward", "under", "within",
]
_RELS_BY_LEN = sorted(ALL_RELATIONS, key=len, reverse=True)


def parse_relation(caption):
    cap = caption.lower()
    for r in _RELS_BY_LEN:
        if re.search(r'\b' + re.escape(r) + r'\b', cap):
            return r
    return None


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"


def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tok.encode(t, add_special_tokens=False); yes_ids.update(toks[:1] if toks else [])
    for t in [" No", "No", " no", "NO"]:
        toks = tok.encode(t, add_special_tokens=False); no_ids.update(toks[:1] if toks else [])
    ov = yes_ids & no_ids; yes_ids -= ov; no_ids -= ov
    return yes_ids, no_ids


def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n; p = max(y / d if d > 0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1 - p, 1e-7))


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = __import__("hashlib").md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    if cp.exists():
        try: return Image.open(cp).convert("RGB")
        except: pass
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        cp.parent.mkdir(parents=True, exist_ok=True); img.save(cp)
        return img
    except: return None


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)

    print("[INFO] Parsing relations...", flush=True)
    rel_indices = {}
    for vi in range(N):
        cap = str(vsr_all[vi].get("caption", ""))
        r = parse_relation(cap)
        if r is not None:
            rel_indices.setdefault(r, []).append(vi)
    for rel in TARGET_RELATIONS:
        n = len(rel_indices.get(rel, []))
        print(f"  {rel}: N={n}", flush=True)

    print("[INFO] Loading pt-448...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    print("[INFO] Loading CAA vectors...", flush=True)
    all_vecs = {}
    for feat_key, cfg in FEATURE_CONFIGS.items():
        path = CAA_DIR / f"caa_L{cfg['layer']}_F{cfg['feature']}.pt"
        if not path.exists():
            print(f"  [WARN] Missing {path}"); continue
        saved = torch.load(path, map_location="cpu")
        vecs = {}
        for l, ld in saved.get("caa_data", {}).items():
            v = ld.get("v_caa_norm")
            if v is not None:
                vecs[int(l)] = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)
        all_vecs[feat_key] = vecs
        print(f"  [LOADED] {feat_key} ({len(vecs)} layers)", flush=True)

    summary = {}

    for rel in TARGET_RELATIONS:
        indices = rel_indices.get(rel, [])
        if not indices:
            print(f"\n[SKIP] {rel} — no samples", flush=True); continue

        result_path = OUT_DIR / f"rel_{rel.replace(' ', '_')}.json"
        if result_path.exists():
            print(f"[SKIP] {rel} already done", flush=True)
            with open(result_path) as f:
                r = json.load(f)
            summary[rel] = r
            continue

        print(f"\n[RELATION] '{rel}'  N={len(indices)}", flush=True)

        # Baseline
        correct = total = 0
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            lbl = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, _build_vsr_prompt(str(ex.get("caption", ""))),
                    processor_pt, model_pt, device=device)
                with torch.inference_mode():
                    out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, _ = _pm(out.logits[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            total += 1; correct += (pred == lbl)
        base_acc = correct / max(total, 1) * 100
        print(f"  BASE: {base_acc:.2f}%  N={total}", flush=True)

        rel_results = {"relation": rel, "n": total, "base_acc": base_acc, "features": {}}

        for feat_key, cfg in FEATURE_CONFIGS.items():
            vecs = all_vecs.get(feat_key)
            if vecs is None: continue
            start = cfg["start"]
            alphas = cfg["alphas"]
            oracle_alpha = cfg["oracle_alpha"]

            print(f"  [FEAT {feat_key}] oracle_α={oracle_alpha}", flush=True)
            feat_results = {}
            best_delta = -999; best_alpha = None

            for alpha in alphas:
                correct = total = 0
                for vi in indices:
                    ex = vsr_all[vi]; img = _load_image(ex)
                    if img is None: continue
                    lbl = int(ex.get("label", 0))
                    try:
                        iids, attn, pv = process_vlm_inputs(
                            img, _build_vsr_prompt(str(ex.get("caption", ""))),
                            processor_pt, model_pt, device=device)
                        _, img_end = get_image_token_positions(iids)
                        with nns_pt.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                            for l in range(start, N_LAYERS):
                                if l not in vecs: continue
                                v_l = vecs[l]
                                v_col = v_l.unsqueeze(1)
                                lo = nns_pt.model.language_model.layers[l].output[0][0, img_end:]
                                ones = (lo @ v_col) * 0.0 + 1.0
                                lo += alpha * ones * v_l
                            logits_s = nns_pt.output.logits.save()
                        pred, _ = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                    except Exception:
                        pred = 0
                    total += 1; correct += (pred == lbl)
                acc = correct / max(total, 1) * 100
                da = acc - base_acc
                marker = " ***" if da > best_delta else ""
                if da > best_delta: best_delta = da; best_alpha = alpha
                oracle_marker = " [ORACLE]" if alpha == oracle_alpha else ""
                print(f"    α={alpha}: {acc:.2f}% (Δ={da:+.2f}%){marker}{oracle_marker}", flush=True)
                feat_results[str(alpha)] = {"acc": acc, "delta_acc": da}

            print(f"    BEST α={best_alpha}: Δ={best_delta:+.2f}%", flush=True)
            rel_results["features"][feat_key] = {
                "best_alpha": best_alpha, "best_delta": best_delta,
                "oracle_alpha": oracle_alpha,
                "oracle_delta": feat_results.get(str(oracle_alpha), {}).get("delta_acc"),
                "alphas": feat_results,
            }

        summary[rel] = rel_results
        with open(result_path, "w") as f:
            json.dump(rel_results, f, indent=2)
        print(f"  [SAVED] {result_path}", flush=True)

    # Summary table
    print("\n" + "=" * 80, flush=True)
    print(f"{'Relation':22s}  {'N':5s}  {'Base':7s}  {'L12(oracle)':11s}  {'L12(best)':9s}  {'L6_best':7s}  {'L15_best':8s}", flush=True)
    print("-" * 80, flush=True)
    for rel, r in summary.items():
        base = r.get("base_acc", 0)
        feats = r.get("features", {})
        f12 = feats.get("L12_F2257", {})
        f6  = feats.get("L6_F7539", {})
        f15 = feats.get("L15_F220", {})
        d12_ora  = f12.get("oracle_delta", 0) or 0
        d12_best = f12.get("best_delta", 0) or 0
        d6  = f6.get("best_delta", 0) or 0
        d15 = f15.get("best_delta", 0) or 0
        print(f"{rel:22s}  {r.get('n',0):5d}  {base:7.2f}%  {d12_ora:+11.2f}%  {d12_best:+9.2f}%  {d6:+7.2f}%  {d15:+8.2f}%", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
