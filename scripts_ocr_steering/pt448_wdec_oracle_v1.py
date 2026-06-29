#!/usr/bin/env python3
"""
W_dec Oracle v1 — W_dec direction injection with per-feature best strategy.

Routes each VSR sample by caption keyword → best feature → inject W_dec[feature_id]
at home layer using per-feature best strategy (from Exp 8/12 results).

Analogous to CAA-based oracle (pt448_smart_oracle_v18.py) but uses the actual SAE
decoder direction at the correct layer with the best strategy found in prior experiments.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_wdec_oracle_v1/oracle_results.json
Usage: CUDA_VISIBLE_DEVICES=3 python3 pt448_wdec_oracle_v1.py
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

MODEL_PT       = "google/paligemma2-3b-pt-448"
N_LAYERS       = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_wdec_oracle_v1")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

DECAY = 0.85

# Per-feature configs: layer, feature, strategy, alpha
FEATURE_CONFIGS = {
    "L4_F14233":  {"layer": 4,  "feature": 14233, "strategy": "sae_only_down", "alpha": 4.0},
    "L6_F7539":   {"layer": 6,  "feature": 7539,  "strategy": "flat_all",      "alpha": 5.0},
    "L9_F387":    {"layer": 9,  "feature": 387,   "strategy": "decay_fwd",     "alpha": 2.0},
    "L9_F7540":   {"layer": 9,  "feature": 7540,  "strategy": "single",        "alpha": 10.0},
    "L11_F12278": {"layer": 11, "feature": 12278, "strategy": "single",        "alpha": 25.0},
    "L12_F2257":  {"layer": 12, "feature": 2257,  "strategy": "flat_all",      "alpha": 50.0},
    "L14_F10561": {"layer": 14, "feature": 10561, "strategy": "flat_all",      "alpha": 2.0},
    "L15_F220":   {"layer": 15, "feature": 220,   "strategy": "sae_only_up",   "alpha": 5.0},
}

SKIP_RELATIONS = frozenset([
    "against", "at the edge of", "far away from", "has as a part",
    "in", "in front of", "into", "perpendicular to"
])

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

_ALL_SORTED = sorted(
    [(r, 'skip') for r in SKIP_RELATIONS] + [(r, 'go') for r in BEST_FEATURE_MAP],
    key=lambda x: len(x[0]), reverse=True
)


def parse_relation(caption: str):
    cap = caption.lower()
    for r, rtype in _ALL_SORTED:
        if re.search(r'\b' + re.escape(r) + r'\b', cap):
            return '__SKIP__' if rtype == 'skip' else r
    return None


def _build_layer_weights(strategy: str, home_layer: int) -> dict:
    if strategy == "single":
        return {home_layer: 1.0}
    elif strategy == "flat_all":
        return {l: 1.0 for l in range(N_LAYERS)}
    elif strategy == "sae_only_down":
        return {l: 1.0 for l in range(home_layer, N_LAYERS)}
    elif strategy == "sae_only_up":
        return {l: 1.0 for l in range(0, home_layer + 1)}
    elif strategy == "decay_fwd":
        return {l: DECAY ** max(l - home_layer, 0) for l in range(N_LAYERS)}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"


def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tok.encode(t, add_special_tokens=False)
        yes_ids.update(toks[:1] if toks else [])
    for t in [" No", "No", " no", "NO"]:
        toks = tok.encode(t, add_special_tokens=False)
        no_ids.update(toks[:1] if toks else [])
    ov = yes_ids & no_ids
    yes_ids -= ov
    no_ids -= ov
    return yes_ids, no_ids


def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n
    p = max(y / d if d > 0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1 - p, 1e-7))


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    h = __import__("hashlib").md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    if cp.exists():
        try:
            return Image.open(cp).convert("RGB")
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        cp.parent.mkdir(parents=True, exist_ok=True)
        img.save(cp)
        return img
    except Exception:
        return None


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / "oracle_results.json"

    # Early exit if results already exist
    if result_path.exists():
        print(f"[SKIP] Result file already exists: {result_path}", flush=True)
        with open(result_path) as f:
            r = json.load(f)
        print(f"  base_acc={r['base_acc']:.2f}%  smart_acc={r['smart_acc']:.2f}%  "
              f"delta_acc={r['delta_acc']:+.2f}%  n_total={r['n_total']}", flush=True)
        return

    device = "cuda:0"

    print("[INFO] Loading VSR (train+dev+test)...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    print(f"  N={N}", flush=True)

    print("[INFO] Loading pt-448 model...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_model = NNsight(model_pt)

    print("[INFO] Loading W_dec vectors from SAE checkpoints...", flush=True)
    wdec_vectors = {}
    for feat_key, cfg in FEATURE_CONFIGS.items():
        layer = cfg["layer"]
        feat_id = cfg["feature"]
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer}.pt"
        if not ckpt_path.exists():
            print(f"  [WARN] Checkpoint not found: {ckpt_path}", flush=True)
            continue
        ck = torch.load(ckpt_path, map_location="cpu")
        # W_dec shape: [n_features, d_model]
        w_dec = ck["W_dec"]  # [n_features, 2304]
        fv = w_dec[feat_id].clone().to(dtype).to(device)
        # Unit-normalize
        fv = fv / fv.norm().clamp(min=1e-8)
        wdec_vectors[feat_key] = fv
        print(f"  [LOADED] {feat_key}: layer={layer}, feature={feat_id}, norm={fv.norm().item():.4f}", flush=True)

    print("[INFO] Precomputing layer_weights for each feature config...", flush=True)
    layer_weights_map = {}
    for feat_key, cfg in FEATURE_CONFIGS.items():
        layer_weights_map[feat_key] = _build_layer_weights(cfg["strategy"], cfg["layer"])
        print(f"  {feat_key}: strategy={cfg['strategy']}, layers={sorted(layer_weights_map[feat_key].keys())[:5]}...", flush=True)

    print("[INFO] Parsing relations...", flush=True)
    parsed = [parse_relation(str(vsr_all[vi].get("caption", ""))) for vi in range(N)]
    n_inject  = sum(1 for r in parsed if r and r != "__SKIP__")
    n_skip    = sum(1 for r in parsed if r == "__SKIP__")
    n_unknown = sum(1 for r in parsed if r is None)
    print(f"  inject={n_inject}, skip={n_skip}, unknown={n_unknown}", flush=True)

    correct_base = correct_smart = total = 0
    # Per-relation tracking: rel -> [base_correct, smart_correct, count]
    by_relation = defaultdict(lambda: [0, 0, 0])

    for vi in range(N):
        ex = vsr_all[vi]
        lbl = int(ex.get("label", 0))
        img = _load_image(ex)
        if img is None:
            continue

        rel = parsed[vi]
        feat_key = BEST_FEATURE_MAP.get(rel) if rel and rel != "__SKIP__" else None
        do_inject = feat_key is not None and feat_key in wdec_vectors

        try:
            iids, attn, pv = process_vlm_inputs(
                img, _build_vsr_prompt(str(ex.get("caption", ""))),
                processor_pt, model_pt, device=device)
            _, img_end = get_image_token_positions(iids)

            # Base prediction
            with torch.inference_mode():
                out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
            pb, _mb = _pm(out.logits[0, -1, :], yes_ids, no_ids)
            correct_base += (pb == lbl)

            if do_inject:
                cfg = FEATURE_CONFIGS[feat_key]
                alpha = cfg["alpha"]
                fv = wdec_vectors[feat_key]
                layer_weights = layer_weights_map[feat_key]

                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l, w in layer_weights.items():
                        lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        v_col = fv.unsqueeze(1)
                        ones = (lo @ v_col) * 0.0 + 1.0
                        lo += (alpha * w) * ones * fv
                    logits_s = nns_model.output.logits.save()

                ps, _ms = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                correct_smart += (ps == lbl)
                action_key = rel
            else:
                ps = pb
                correct_smart += (pb == lbl)
                action_key = "__SKIP__" if rel == "__SKIP__" else "__UNKNOWN__"

            d = by_relation[action_key]
            d[0] += (pb == lbl)
            d[1] += (ps == lbl)
            d[2] += 1

        except Exception as e:
            print(f"  [ERR] vi={vi}: {e}", flush=True)
            continue

        total += 1
        if total % 1000 == 0:
            ba_now = 100 * correct_base / total
            sa_now = 100 * correct_smart / total
            print(f"  [{total}/{N}] base={ba_now:.2f}%  smart={sa_now:.2f}%  "
                  f"delta={sa_now - ba_now:+.2f}%", flush=True)

    ba = correct_base / max(total, 1) * 100
    sa = correct_smart / max(total, 1) * 100
    delta = sa - ba

    print(f"\n[RESULT] base_acc={ba:.2f}%  smart_acc={sa:.2f}%  "
          f"delta_acc={delta:+.2f}%  n_total={total}", flush=True)

    # Build per_relation output (only actual relations, not __SKIP__/__UNKNOWN__)
    per_relation = {}
    for rel_key, d in sorted(by_relation.items()):
        rb = d[0] / max(d[2], 1) * 100
        rs = d[1] / max(d[2], 1) * 100
        per_relation[rel_key] = {
            "n": d[2],
            "base_acc": rb,
            "smart_acc": rs,
            "delta_acc": rs - rb,
        }
        print(f"  [{rel_key:25s}] N={d[2]:5d}  base={rb:.2f}%  smart={rs:.2f}%  "
              f"delta={rs - rb:+.2f}%", flush=True)

    results = {
        "base_acc":    ba,
        "smart_acc":   sa,
        "delta_acc":   delta,
        "n_total":     total,
        "n_inject":    n_inject,
        "n_skip":      n_skip,
        "per_relation": per_relation,
    }

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {result_path}", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
