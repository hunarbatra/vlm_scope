#!/usr/bin/env python3
"""
Fine alpha sweep for L15/F220 on own-relation subsets ("across from", "at the left side of").

Correct methodology: evaluate on per-relation subset only, not full VSR.
L15/F220 own_rels=["across from", "at the left side of"], natural start=15.

Alphas: 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L15_subset_sweep/
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_L15_subset_sweep.py
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
LAYER       = 15
FEATURE     = 220
START_LAYER = 15
OWN_RELS    = ["across from", "at the left side of"]
CAA_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L15_subset_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0]

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
    result_path = OUT_DIR / f"L{LAYER}_F{FEATURE}_subset_sweep.json"
    if result_path.exists():
        print(f"[SKIP] Results already at {result_path}", flush=True)
        with open(result_path) as f:
            r = json.load(f)
        for rel, rd in r.get("relations", {}).items():
            print(f"  {rel}: best α={rd.get('best_alpha')} → {rd.get('best_acc'):.2f}% (Δ={rd.get('best_delta'):+.2f}%)")
        return

    device = "cuda:0"

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)

    print(f"[INFO] Building subset indices for {OWN_RELS}...", flush=True)
    subset_indices = {}
    for vi in range(N):
        cap = str(vsr_all[vi].get("caption", ""))
        r = parse_relation(cap)
        if r in OWN_RELS:
            subset_indices.setdefault(r, []).append(vi)
    for rel, idxs in subset_indices.items():
        print(f"  {rel}: N={len(idxs)}", flush=True)

    print("[INFO] Loading pt-448...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    print(f"[INFO] Loading CAA vectors for L{LAYER}/F{FEATURE}...", flush=True)
    caa_path = CAA_DIR / f"caa_L{LAYER}_F{FEATURE}.pt"
    saved = torch.load(caa_path, map_location="cpu")
    caa_data = saved.get("caa_data", {})
    vecs = {}
    for l, ld in caa_data.items():
        v = ld.get("v_caa_norm")
        if v is not None:
            vecs[int(l)] = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)
    print(f"  Loaded {len(vecs)} layer vectors", flush=True)

    all_rel_results = {}

    for rel, indices in subset_indices.items():
        print(f"\n[RELATION] {rel}  N={len(indices)}", flush=True)

        # Baseline on this subset
        correct = total = 0; margins = []
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
        for alpha in ALPHAS:
            correct = total = 0; margins = []
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
                        for l in range(START_LAYER, N_LAYERS):
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
        all_rel_results[rel] = {
            "base_acc": base_acc, "base_margin": base_mg, "n": total,
            "best_alpha": best_alpha, "best_acc": best_acc, "best_delta": best_delta,
            "alphas": alpha_results,
        }

    result = {
        "layer": LAYER, "feature": FEATURE, "start_layer": START_LAYER,
        "own_rels": OWN_RELS, "relations": all_rel_results,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] {result_path}", flush=True)


if __name__ == "__main__":
    main()
