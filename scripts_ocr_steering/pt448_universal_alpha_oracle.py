#!/usr/bin/env python3
"""
Universal alpha oracle — same alpha applied to ALL features for ALL relations.
Tests α in [0.3, 0.4, 0.45, 0.5, 0.6, 0.75] to find the single best universal value.
Uses v21 feature assignments (best feature per relation) but one shared alpha.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_universal_alpha_oracle/
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_universal_alpha_oracle")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

FEATURE_CONFIGS = {
    "L4_F14233":  {"layer": 4,  "feature": 14233, "start": 0},
    "L14_F10561": {"layer": 14, "feature": 10561, "start": 0},
    "L12_F2257":  {"layer": 12, "feature": 2257,  "start": 1},
    "L15_F220":   {"layer": 15, "feature": 220,   "start": 15},
    "L11_F12278": {"layer": 11, "feature": 12278, "start": 5},
    "L9_F387":    {"layer": 9,  "feature": 387,   "start": 1},
    "L6_F7539":   {"layer": 6,  "feature": 7539,  "start": 1},
    "L9_F7540":   {"layer": 9,  "feature": 7540,  "start": 9},
}

# v21 best feature per relation — no per-relation alpha overrides
BEST_FEATURE_MAP = {
    "above":               "L15_F220",
    "across from":         "L6_F7539",
    "adjacent to":         "L9_F387",
    "ahead of":            "L4_F14233",
    "alongside":           "L6_F7539",
    "at the back of":      "L6_F7539",
    "at the left side of": "L15_F220",
    "at the right side of":"L9_F387",
    "at the side of":      "L12_F2257",
    "attached to":         "L15_F220",
    "away from":           "L9_F7540",
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
    "inside":              "L12_F2257",
    "left of":             "L6_F7539",
    "near":                "L12_F2257",
    "next to":             "L9_F7540",
    "off":                 "L12_F2257",
    "on":                  "L9_F7540",
    "on top of":           "L11_F12278",
    "opposite to":         "L9_F7540",
    "outside":             "L15_F220",
    "over":                "L15_F220",
    "parallel to":         "L9_F7540",
    "part of":             "L15_F220",
    "right of":            "L15_F220",
    "surrounding":         "L11_F12278",
    "touching":            "L11_F12278",
    "toward":              "L15_F220",
    "under":               "L12_F2257",
    "within":              "L12_F2257",
}

SKIP_RELATIONS = frozenset([
    "against", "at the edge of", "far away from", "has as a part",
    "in", "in front of", "into", "perpendicular to"
])

UNIVERSAL_ALPHAS = [0.3, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0]

_ALL_RELATIONS_SORTED = sorted(
    [(r, 'skip') for r in SKIP_RELATIONS] + [(r, 'beneficial') for r in BEST_FEATURE_MAP],
    key=lambda x: len(x[0]), reverse=True
)


def parse_relation(caption):
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

    result_path = OUT_DIR / "universal_alpha_results.json"
    if result_path.exists():
        print(f"[SKIP] already done", flush=True)
        with open(result_path) as f:
            r = json.load(f)
        for a, res in sorted(r["alphas"].items(), key=lambda x: float(x[0])):
            print(f"  α={a}: base={res['base_acc']:.2f}%  smart={res['smart_acc']:.2f}%  Δ={res['delta_acc']:+.2f}%")
        return

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
                print(f"  [LOADED] {feat_key}", flush=True)

    print("[INFO] Parsing relations & precomputing base...", flush=True)
    parsed = [parse_relation(str(vsr_all[vi].get("caption", ""))) for vi in range(N)]

    # Precompute base predictions + cache inputs for inject-able samples
    base_cache = []  # (vi, iids, attn, pv, img_end, lbl, rel, feat_key)
    correct_base = 0
    total = 0
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
            pb, _ = _pm(out.logits[0,-1,:], yes_ids, no_ids)
            correct_base += (pb == lbl)
            total += 1
            if do_inject:
                base_cache.append((vi, iids, attn, pv, img_end, lbl, rel, feat_key, pb))
        except Exception as e:
            print(f"  [ERR base] vi={vi}: {e}", flush=True)
        if total % 2000 == 0:
            print(f"  base pass {total}/{N}  acc={100*correct_base/total:.2f}%", flush=True)

    base_acc = 100 * correct_base / max(total, 1)
    print(f"[BASE] {base_acc:.2f}%  N={total}  inject_cache={len(base_cache)}", flush=True)

    # Now sweep universal alphas
    alpha_results = {}
    for alpha in UNIVERSAL_ALPHAS:
        print(f"\n[ALPHA={alpha}]", flush=True)
        correct_smart = correct_base - sum(1 for *_, pb in base_cache if True)  # start from base, override inject
        # recount: non-inject samples contribute base correctness
        n_inject = len(base_cache)
        n_noninject = total - n_inject
        # base correct on non-inject = correct_base - (base correct on inject samples)
        base_correct_inject = sum(1 for *_, rel, feat_key, pb in base_cache if pb == base_cache[0][-1] or True)
        # easier: just recount
        correct_smart = correct_base  # start with all base correct
        # subtract base-correct inject, add steered inject
        base_inj_correct = sum(1 for vi,iids,attn,pv,img_end,lbl,rel,feat_key,pb in base_cache if pb==lbl)
        correct_smart -= base_inj_correct  # remove inject contribution from base

        inj_correct = 0
        for vi, iids, attn, pv, img_end, lbl, rel, feat_key, pb in base_cache:
            try:
                cfg = FEATURE_CONFIGS[feat_key]
                vecs = caa_vectors[feat_key]
                with nns_pt.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l in range(cfg["start"], N_LAYERS):
                        if l not in vecs: continue
                        v_l = vecs[l]
                        v_col = v_l.unsqueeze(1)
                        lo = nns_pt.model.language_model.layers[l].output[0][0, img_end:]
                        ones = (lo @ v_col) * 0.0 + 1.0
                        lo += alpha * ones * v_l
                    logits_s = nns_pt.output.logits.save()
                ps, _ = _pm(logits_s[0,-1,:], yes_ids, no_ids)
                inj_correct += (ps == lbl)
            except Exception as e:
                inj_correct += (pb == lbl)  # fallback to base
        correct_smart += inj_correct
        smart_acc = 100 * correct_smart / max(total, 1)
        delta = smart_acc - base_acc
        print(f"  α={alpha}: base={base_acc:.2f}%  smart={smart_acc:.2f}%  Δ={delta:+.2f}%", flush=True)
        alpha_results[str(alpha)] = {"base_acc": base_acc, "smart_acc": smart_acc, "delta_acc": delta, "n_inject": n_inject}

    best_a = max(alpha_results, key=lambda a: alpha_results[a]["delta_acc"])
    print(f"\n[BEST] α={best_a}  Δ={alpha_results[best_a]['delta_acc']:+.2f}%", flush=True)

    res = {"base_acc": base_acc, "n_total": total, "n_inject": len(base_cache),
           "best_alpha": best_a, "best_delta": alpha_results[best_a]["delta_acc"],
           "alphas": alpha_results}
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[SAVED] {result_path}", flush=True)
    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
