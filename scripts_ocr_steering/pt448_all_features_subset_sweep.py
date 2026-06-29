#!/usr/bin/env python3
"""
Fine alpha sweep for all 8 features on their own-relation subsets.

Correct methodology: evaluate each feature on its own-relation subset only.
Fine-grained alphas around known good values for each feature.

Skips features already handled by dedicated scripts (L11, L15).

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_all_features_subset_sweep/
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_all_features_subset_sweep.py
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_all_features_subset_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Feature configs with known rough best alpha from global sweep
# Fine alphas centered around known good values
FEATURE_CONFIGS = {
    "L4_F14233": {
        "layer": 4, "feature": 14233, "start": 0,
        "own_rels": ["ahead of"],
        "alphas": [0.5, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    },
    "L14_F10561": {
        "layer": 14, "feature": 10561, "start": 0,
        "own_rels": ["close to"],
        "alphas": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    },
    "L12_F2257": {
        "layer": 12, "feature": 2257, "start": 1,
        "own_rels": ["facing"],
        "alphas": [0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    },
    "L9_F387": {
        "layer": 9, "feature": 387, "start": 1,
        "own_rels": ["at the right side of"],
        "alphas": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0],
    },
    "L6_F7539": {
        "layer": 6, "feature": 7539, "start": 1,
        "own_rels": ["left of", "right of"],
        "alphas": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    },
    "L9_F7540": {
        "layer": 9, "feature": 7540, "start": 9,
        "own_rels": ["consists of"],
        "alphas": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0],
    },
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

    for feat_key, cfg in FEATURE_CONFIGS.items():
        result_path = OUT_DIR / f"subset_{feat_key}.json"
        if result_path.exists():
            print(f"[SKIP] {feat_key} already done", flush=True)
            with open(result_path) as f:
                r = json.load(f)
            summary[feat_key] = r
            continue

        vecs = all_vecs.get(feat_key)
        if vecs is None:
            print(f"[SKIP] {feat_key} no vectors", flush=True); continue

        own_rels = cfg["own_rels"]
        start = cfg["start"]
        alphas = cfg["alphas"]

        # Collect own-relation indices
        own_indices = []
        for rel in own_rels:
            own_indices.extend(rel_indices.get(rel, []))
        if not own_indices:
            print(f"[SKIP] {feat_key} no samples", flush=True); continue

        print(f"\n[FEATURE] {feat_key}  own_rels={own_rels}  N={len(own_indices)}  start={start}", flush=True)

        # Baseline
        correct = total = 0; margins = []
        for vi in own_indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            lbl = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, _build_vsr_prompt(str(ex.get("caption", ""))),
                    processor_pt, model_pt, device=device)
                with torch.inference_mode():
                    out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, m = _pm(out.logits[0, -1, :], yes_ids, no_ids)
                margins.append(m if lbl == 1 else -m)
            except Exception:
                pred = 0; margins.append(0.0)
            total += 1; correct += (pred == lbl)
        base_acc = correct / max(total, 1) * 100
        base_mg = sum(margins) / max(len(margins), 1)
        print(f"  BASE: {base_acc:.2f}%  N={total}", flush=True)

        # Alpha sweep
        alpha_results = {}
        best_alpha = None; best_delta = -999; best_acc = base_acc
        for alpha in alphas:
            correct = total = 0; margins = []
            for vi in own_indices:
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
                    pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                    margins.append(m if lbl == 1 else -m)
                except Exception:
                    pred = 0; margins.append(0.0)
                total += 1; correct += (pred == lbl)
            acc = correct / max(total, 1) * 100
            da = acc - base_acc
            mg = sum(margins) / max(len(margins), 1)
            marker = " *** NEW BEST ***" if da > best_delta else ""
            if da > best_delta: best_delta = da; best_alpha = alpha; best_acc = acc
            print(f"  α={alpha}: {acc:.2f}% (Δ={da:+.2f}%) mg={mg:.3f}{marker}", flush=True)
            alpha_results[str(alpha)] = {"acc": acc, "delta_acc": da, "margin": mg}

        print(f"\n  [RESULT] Best α={best_alpha}: {best_acc:.2f}% (Δ={best_delta:+.2f}%)", flush=True)
        result = {
            "feat_key": feat_key, "layer": cfg["layer"], "feature": cfg["feature"],
            "start": start, "own_rels": own_rels, "n_own_rel": len(own_indices),
            "base_acc": base_acc, "base_margin": base_mg, "n": total,
            "best_alpha": best_alpha, "best_acc": best_acc, "best_delta": best_delta,
            "alphas": alpha_results,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        summary[feat_key] = result
        print(f"  [SAVED] {result_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"{'Feature':14s}  {'N_ownrel':8s}  {'Base':6s}  {'Best Δ':8s}  {'Best α':7s}", flush=True)
    print("-" * 70, flush=True)
    for fk, r in sorted(summary.items()):
        print(f"{fk:14s}  {r.get('n_own_rel', 0):8d}  {r.get('base_acc', 0):6.2f}%  {r.get('best_delta', 0):+8.2f}%  {str(r.get('best_alpha', '?')):7s}", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
