#!/usr/bin/env python3
"""
Smart Oracle v4 — Updated alphas from per-feature own-relation subset sweeps (all 8 confirmed).

Changes from v3:
- L11/F12278: α=0.5 → α=0.45 (subset sweep on 'touching' N=1281: α=0.45→+6.01% vs α=0.5→+5.54%)
- L9/F387:    α=0.5 → α=0.4  (subset sweep on 'at the right side of' N=480: α=0.4→+4.17% vs α=0.5→+3.12%)
- All other configs same as v3.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_smart_oracle_v4/
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_smart_oracle_v4.py
"""

import os, sys, json, re, warnings, math
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_smart_oracle_v4")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# All alphas confirmed from per-feature own-relation subset sweeps
FEATURE_CONFIGS = {
    "L4_F14233":  {"layer": 4,  "feature": 14233, "start": 0,  "alpha": 0.9},   # subset: α=0.9 → +15.38%
    "L14_F10561": {"layer": 14, "feature": 10561, "start": 0,  "alpha": 2.0},   # subset: α=2.0 → +10.75%
    "L12_F2257":  {"layer": 12, "feature": 2257,  "start": 1,  "alpha": 0.75},  # subset: α=0.75 → +8.82%
    "L15_F220":   {"layer": 15, "feature": 220,   "start": 15, "alpha": 0.7},   # subset: α=0.7 → +8.55%
    "L11_F12278": {"layer": 11, "feature": 12278, "start": 5,  "alpha": 0.45},  # subset: α=0.45 → +6.01% (was 0.5→+5.54%)
    "L9_F387":    {"layer": 9,  "feature": 387,   "start": 1,  "alpha": 0.4},   # subset: α=0.4 → +4.17% (was 0.5→+3.12%)
    "L6_F7539":   {"layer": 6,  "feature": 7539,  "start": 1,  "alpha": 1.5},   # subset: α=1.5 → +3.10% (unchanged)
    "L9_F7540":   {"layer": 9,  "feature": 7540,  "start": 9,  "alpha": 0.25},  # prior best (subset TBD, N=35 too small)
}

SKIP_RELATIONS = frozenset([
    "against", "at the edge of", "far away from", "has as a part",
    "in", "in front of", "into", "perpendicular to"
])

# Best feature per relation (from per_relation_steer cross-feature matrix)
BEST_FEATURE_MAP = {
    "above":               "L15_F220",
    "across from":         "L6_F7539",
    "adjacent to":         "L9_F387",
    "ahead of":            "L4_F14233",
    "alongside":           "L6_F7539",
    "at the back of":      "L6_F7539",
    "at the left side of": "L15_F220",
    "at the right side of":"L9_F387",
    "at the side of":      "L9_F387",
    "attached to":         "L9_F387",
    "away from":           "L15_F220",
    "behind":              "L4_F14233",
    "below":               "L6_F7539",
    "beneath":             "L12_F2257",
    "beside":              "L15_F220",
    "beyond":              "L12_F2257",
    "by":                  "L14_F10561",
    "close to":            "L14_F10561",
    "connected to":        "L14_F10561",
    "consists of":         "L9_F7540",
    "contains":            "L15_F220",
    "enclosed by":         "L12_F2257",
    "facing":              "L12_F2257",
    "facing away from":    "L6_F7539",
    "far from":            "L9_F387",
    "in the middle of":    "L9_F7540",
    "inside":              "L15_F220",
    "left of":             "L6_F7539",
    "near":                "L12_F2257",
    "next to":             "L9_F7540",
    "off":                 "L12_F2257",
    "on":                  "L9_F7540",
    "on top of":           "L11_F12278",
    "opposite to":         "L9_F7540",
    "outside":             "L12_F2257",
    "over":                "L15_F220",
    "parallel to":         "L9_F7540",
    "part of":             "L15_F220",
    "right of":            "L6_F7539",
    "surrounding":         "L11_F12278",
    "touching":            "L11_F12278",
    "toward":              "L12_F2257",
    "under":               "L11_F12278",
    "within":              "L15_F220",
}

_ALL_RELATIONS_SORTED = sorted(
    [(r, 'skip') for r in SKIP_RELATIONS] + [(r, 'beneficial') for r in BEST_FEATURE_MAP],
    key=lambda x: len(x[0]), reverse=True
)


def parse_relation(caption: str):
    cap = caption.lower()
    for r, rtype in _ALL_RELATIONS_SORTED:
        if re.search(r'\b' + re.escape(r) + r'\b', cap):
            return '__SKIP__' if rtype == 'skip' else r
    return None


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"


def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]: toks = tok.encode(t, add_special_tokens=False); yes_ids.update(toks[:1] if toks else [])
    for t in [" No","No"," no","NO"]: toks = tok.encode(t, add_special_tokens=False); no_ids.update(toks[:1] if toks else [])
    ov = yes_ids & no_ids; yes_ids -= ov; no_ids -= ov
    return yes_ids, no_ids


def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n; p = max(y/d if d>0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1-p, 1e-7))


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

    print("[INFO] Loading pt-448...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    print("[INFO] Loading CAA vectors...", flush=True)
    caa_vectors = {}
    for feat_key, cfg in FEATURE_CONFIGS.items():
        path = CAA_DIR / f"caa_L{cfg['layer']}_F{cfg['feature']}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu")
            vecs = {}
            for l, ld in saved.get("caa_data", {}).items():
                v = ld.get("v_caa_norm")
                if v is not None:
                    vecs[int(l)] = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)
            if vecs:
                caa_vectors[feat_key] = vecs
                print(f"  [LOADED] {feat_key} ({len(vecs)} layers)", flush=True)

    print("[INFO] Parsing relations...", flush=True)
    parsed = [parse_relation(str(vsr_all[vi].get("caption", ""))) for vi in range(N)]
    stats = {"inject": sum(1 for r in parsed if r and r != "__SKIP__"),
             "skip":   sum(1 for r in parsed if r == "__SKIP__"),
             "unknown":sum(1 for r in parsed if r is None)}
    print(f"  inject={stats['inject']}, skip={stats['skip']}, unknown={stats['unknown']}", flush=True)

    result_path = OUT_DIR / "smart_oracle_v4_per_feature.json"
    if result_path.exists():
        print(f"[SKIP] {result_path} exists", flush=True)
        with open(result_path) as f:
            r = json.load(f)
        print(f"  Result: Δ={r['delta_acc']:+.2f}%  (v3 expected ~+4.1%)")
        return

    print(f"\n{'='*60}", flush=True)
    print(f"Smart Oracle v4 — all subset-confirmed alphas", flush=True)

    correct_base = correct_smart = total = 0
    mb_sum = ms_sum = 0.0
    by_action = defaultdict(lambda: [0, 0, 0])

    for vi in range(N):
        ex = vsr_all[vi]; lbl = int(ex.get("label", 0))
        img = _load_image(ex)
        if img is None: continue

        rel = parsed[vi]
        feat_key = BEST_FEATURE_MAP.get(rel) if rel and rel != "__SKIP__" else None
        do_inject = feat_key is not None and feat_key in caa_vectors

        try:
            iids, attn, pv = process_vlm_inputs(
                img, _build_vsr_prompt(str(ex.get("caption",""))),
                processor_pt, model_pt, device=device)
            _, img_end = get_image_token_positions(iids)

            with torch.inference_mode():
                out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
            pb, mb = _pm(out.logits[0,-1,:], yes_ids, no_ids)
            mb_sum += mb if lbl==1 else -mb
            correct_base += (pb == lbl)

            if do_inject:
                cfg = FEATURE_CONFIGS[feat_key]
                start = cfg["start"]
                alpha = cfg["alpha"]
                vecs = caa_vectors[feat_key]
                with nns_pt.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l in range(start, N_LAYERS):
                        if l not in vecs: continue
                        v_l = vecs[l]
                        v_col = v_l.unsqueeze(1)
                        lo = nns_pt.model.language_model.layers[l].output[0][0, img_end:]
                        ones = (lo @ v_col) * 0.0 + 1.0
                        lo += alpha * ones * v_l
                    logits_s = nns_pt.output.logits.save()
                ps, ms = _pm(logits_s[0,-1,:], yes_ids, no_ids)
                ms_sum += ms if lbl==1 else -ms
                correct_smart += (ps == lbl)
                action = "inject"
            else:
                ms_sum += mb if lbl==1 else -mb
                correct_smart += (pb == lbl)
                ps = pb
                action = "skip" if rel == "__SKIP__" else "unknown"

            d = by_action[action]
            d[0] += (pb == lbl); d[1] += (ps == lbl); d[2] += 1

        except Exception as e:
            print(f"  [ERR] vi={vi}: {e}", flush=True); continue
        total += 1
        if total % 2000 == 0:
            print(f"  {total} base={100*correct_base/total:.2f}% smart={100*correct_smart/total:.2f}%", flush=True)

    ba = correct_base / max(total, 1) * 100
    sa = correct_smart / max(total, 1) * 100
    print(f"\n[RESULT v4] base={ba:.2f}%  smart={sa:.2f}%  Δ={sa-ba:+.2f}%  N={total}", flush=True)
    print(f"  [vs v3: expected ~+4.1%;  v2: +3.96%]", flush=True)
    for act, d in sorted(by_action.items()):
        ab = d[0]/max(d[2],1)*100; as_ = d[1]/max(d[2],1)*100
        print(f"  [{act:10s}] N={d[2]:5d}  base={ab:.2f}%  smart={as_:.2f}%  Δ={as_-ab:+.2f}%", flush=True)

    res = {"version": "v4", "alpha_mode": "per_feature",
           "feature_alphas": {k: v["alpha"] for k,v in FEATURE_CONFIGS.items()},
           "base_acc": ba, "smart_acc": sa, "delta_acc": sa-ba,
           "base_margin": mb_sum/max(total,1), "smart_margin": ms_sum/max(total,1),
           "n_total": total, "parse_stats": stats,
           "by_action": {k: {"correct_base":d[0],"correct_smart":d[1],"total":d[2]} for k,d in by_action.items()}}
    with open(result_path, "w") as f: json.dump(res, f, indent=2)
    print(f"[SAVED] {result_path}", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
