#!/usr/bin/env python3
"""
Approach 1: W_dec single-layer injection.

For each of the 8 canonical spatial SAE features, injects
    alpha * W_dec[feature_id]   (unit-normalised)
at the feature's exact home layer only (text tokens only).

Sweeps alpha in ALPHAS, evaluates on the home-relation VSR subset.

Per-feature output: wdec_L{layer}_F{feature}.json
  {layer, feature, relations, n, base_acc, alphas: {str(alpha): {acc, delta_acc}},
   best_alpha, best_delta}

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 pt448_wdec_injection.py
"""

import os, sys, re, json, math, hashlib, warnings, gc
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# ------------------------------------------------------------------ env/paths
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MODEL_NAME     = "google/paligemma2-3b-pt-448"
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_wdec_injection")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

# ------------------------------------------------------------------ features
FEATURES = [
    {"layer": 4,  "feature": 14233, "relations": ["ahead of", "behind"]},
    {"layer": 6,  "feature": 7539,  "relations": ["left of", "right of", "across from", "alongside", "at the back of", "below", "facing away from"]},
    {"layer": 9,  "feature": 387,   "relations": ["at the right side of", "adjacent to", "far from", "attached to"]},
    {"layer": 9,  "feature": 7540,  "relations": ["on", "next to", "parallel to", "in the middle of", "opposite to", "away from", "consists of"]},
    {"layer": 11, "feature": 12278, "relations": ["touching", "on top of", "surrounding", "under"]},
    {"layer": 12, "feature": 2257,  "relations": ["facing", "beneath", "near", "off", "enclosed by", "inside", "within", "beyond", "at the side of"]},
    {"layer": 14, "feature": 10561, "relations": ["close to", "by", "connected to"]},
    {"layer": 15, "feature": 220,   "relations": ["above", "at the left side of", "beside", "contains", "over", "part of", "right of", "outside", "toward"]},
]

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# ------------------------------------------------------------------ relation list & parser
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


# ------------------------------------------------------------------ helpers
def _build_vsr_prompt(s):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {s.strip()}\nAnswer:"
    )


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids  -= overlap
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
    h  = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists():
            return Image.open(cp).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None


def _load_wdec_vec(layer, feature_id, model_dtype, device):
    """Load W_dec[feature_id] from text-only SAE checkpoint and unit-normalise."""
    ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer}.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # W_dec shape: [16384, 2304]
    vec = state["W_dec"][feature_id].to(model_dtype).to(device)
    vec = vec / vec.norm().clamp(min=1e-8)
    return vec


# ------------------------------------------------------------------ evaluation
def run_baseline(indices, vsr_all, processor, model_raw, yes_ids, no_ids, device):
    """Evaluate model without any injection."""
    from utils import process_vlm_inputs
    correct = total = 0
    for vi in indices:
        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            continue
        label  = int(ex.get("label", 0))
        prompt = _build_vsr_prompt(str(ex.get("caption", "")))
        try:
            iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
            with torch.inference_mode():
                out = model_raw(input_ids=iids, attention_mask=attn,
                                pixel_values=pv, use_cache=False)
            pred, _ = _pm(out.logits[0, -1, :], yes_ids, no_ids)
        except Exception:
            pred = 0
        total   += 1
        correct += (pred == label)
    return correct / max(total, 1) * 100, total


def run_injection(indices, vsr_all, wdec_vec, home_layer, alpha,
                  nns_model, processor, base_module, yes_ids, no_ids, device):
    """Inject alpha * wdec_vec at home_layer (text tokens), return (acc, n)."""
    from utils import process_vlm_inputs, get_image_token_positions
    correct = total = 0
    v_col   = wdec_vec.unsqueeze(1)   # (d, 1) for proxy matmul
    for vi in indices:
        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            continue
        label  = int(ex.get("label", 0))
        prompt = _build_vsr_prompt(str(ex.get("caption", "")))
        try:
            iids, attn, pv = process_vlm_inputs(img, prompt, processor, base_module, device=device)
            _, img_end = get_image_token_positions(iids)
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                lo   = nns_model.model.language_model.layers[home_layer].output[0][0, img_end:]
                ones = (lo @ v_col) * 0.0 + 1.0
                lo  += alpha * ones * wdec_vec
                logits_s = nns_model.output.logits.save()
            pred, _ = _pm(logits_s[0, -1, :], yes_ids, no_ids)
        except Exception:
            pred = 0
        total   += 1
        correct += (pred == label)
    return correct / max(total, 1) * 100, total


# ------------------------------------------------------------------ main
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions  # noqa: F401

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    # ---- load model
    print(f"[INFO] Loading {MODEL_NAME} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model  = NNsight(model_raw)
    tokenizer  = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # ---- load VSR
    print("[INFO] Loading VSR (train+dev+test) ...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    # build relation → sample index map
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    # ---- per-feature loop
    for feat in FEATURES:
        layer      = feat["layer"]
        feature_id = feat["feature"]
        relations  = feat["relations"]
        key        = f"L{layer}_F{feature_id}"
        out_path   = OUT_DIR / f"wdec_{key}.json"

        if out_path.exists():
            print(f"[SKIP] {key} — output already exists", flush=True)
            continue

        # gather VSR indices
        indices = []
        for r in relations:
            indices.extend(relation_indices.get(r, []))
        if not indices:
            print(f"[WARN] {key}: no VSR samples for relations {relations}", flush=True)
            continue

        print(f"\n[FEAT] {key}  relations={relations}  N={len(indices)}", flush=True)

        # baseline
        print(f"  [BASE] computing baseline ...", flush=True)
        base_acc, n = run_baseline(indices, vsr_all, processor, model_raw,
                                   yes_ids, no_ids, device)
        print(f"  [BASE] acc={base_acc:.2f}%  N={n}", flush=True)

        # load W_dec direction
        wdec_vec = _load_wdec_vec(layer, feature_id, model_dtype, device)

        # alpha sweep
        alpha_results = {}
        best_alpha = None
        best_delta = -1e9

        for alpha in ALPHAS:
            print(f"  [INJECT] {key} alpha={alpha} ...", flush=True)
            acc, n_inj = run_injection(
                indices, vsr_all, wdec_vec, layer, alpha,
                nns_model, processor, nns_model._module,
                yes_ids, no_ids, device
            )
            delta = acc - base_acc
            alpha_results[str(alpha)] = {"acc": acc, "delta_acc": delta}
            print(f"    alpha={alpha:>5}  acc={acc:.2f}%  delta={delta:+.2f}%", flush=True)
            if delta > best_delta:
                best_delta = delta
                best_alpha = alpha

        result = {
            "layer":      layer,
            "feature":    feature_id,
            "relations":  relations,
            "n":          n,
            "base_acc":   base_acc,
            "alphas":     alpha_results,
            "best_alpha": best_alpha,
            "best_delta": best_delta,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [SAVE] {out_path}", flush=True)

        # free GPU memory between features
        del wdec_vec
        torch.cuda.empty_cache()
        gc.collect()

    # ---- summary
    print(f"\n{'='*90}")
    print("W_dec Single-Layer Injection — Summary")
    print(f"{'='*90}")
    header = f"{'Feature':<16} {'Relations':<38} {'N':>5} {'Base':>7}"
    for a in ALPHAS:
        header += f"  {a:>5}"
    header += f"  {'BestA':>6}  {'BestΔ':>7}"
    print(header)
    print("-" * 90)

    for feat in FEATURES:
        layer      = feat["layer"]
        feature_id = feat["feature"]
        key        = f"L{layer}_F{feature_id}"
        out_path   = OUT_DIR / f"wdec_{key}.json"
        if not out_path.exists():
            continue
        with open(out_path) as f:
            r = json.load(f)
        rels_str = ", ".join(r["relations"])[:37]
        row = f"{key:<16} {rels_str:<38} {r['n']:>5} {r['base_acc']:>6.1f}%"
        for a in ALPHAS:
            d = r["alphas"].get(str(a), {}).get("delta_acc")
            row += f"  {d:>+5.1f}" if d is not None else f"  {'--':>5}"
        row += f"  {str(r.get('best_alpha', '--')):>6}  {r.get('best_delta', 0):>+7.2f}"
        print(row)

    print(f"\n[DONE] Results in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
