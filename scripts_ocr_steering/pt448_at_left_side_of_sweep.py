#!/usr/bin/env python3
"""
Complete at_left_side_of sweep for remaining features.
L6_F7539 and L15_F220 already done in l6_relations_sweep.log:
  L6_F7539: oracle α=1.5 → +1.66% | best α=0.75 → +8.31%
  L15_F220: oracle α=0.7 → +8.55% | best α=0.7 → +8.55% [ORACLE]

This script sweeps remaining features: L12_F2257, L9_F387, L11_F12278
to complete the picture for v20 planning.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_l6_relations_sweep/rel_at_the_left_side_of.json
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_l6_relations_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Only sweep features not yet done
FEATURE_CONFIGS = {
    "L12_F2257": {"layer": 12, "feature": 2257, "start": 1,  "oracle_alpha": 0.75,
                  "alphas": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]},
    "L9_F387":   {"layer": 9,  "feature": 387,  "start": 1,  "oracle_alpha": 0.4,
                  "alphas": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5]},
    "L11_F12278":{"layer": 11, "feature": 12278,"start": 5,  "oracle_alpha": 0.45,
                  "alphas": [0.2, 0.3, 0.45, 0.6, 0.75, 1.0]},
}

# Pre-known results from l6_relations_sweep.log
KNOWN_RESULTS = {
    "L6_F7539": {
        "best_alpha": 0.75, "best_delta": 8.31,
        "oracle_alpha": 1.5, "oracle_delta": 1.66,
        "alphas": {
            "0.5": {"acc": 57.96, "delta_acc": 6.18},
            "0.75": {"acc": 60.10, "delta_acc": 8.31},
            "1.0": {"acc": 57.72, "delta_acc": 5.94},
            "1.25": {"acc": 56.29, "delta_acc": 4.51},
            "1.5": {"acc": 53.44, "delta_acc": 1.66},
            "1.75": {"acc": 52.26, "delta_acc": 0.48},
            "2.0": {"acc": 50.59, "delta_acc": -1.19},
            "2.5": {"acc": 50.12, "delta_acc": -1.66},
            "3.0": {"acc": 50.12, "delta_acc": -1.66},
        }
    },
    "L15_F220": {
        "best_alpha": 0.7, "best_delta": 8.55,
        "oracle_alpha": 0.7, "oracle_delta": 8.55,
        "alphas": {
            "0.4": {"acc": 55.34, "delta_acc": 3.56},
            "0.5": {"acc": 56.29, "delta_acc": 4.51},
            "0.6": {"acc": 59.86, "delta_acc": 8.08},
            "0.7": {"acc": 60.33, "delta_acc": 8.55},
            "0.8": {"acc": 59.62, "delta_acc": 7.84},
            "0.9": {"acc": 57.96, "delta_acc": 6.18},
        }
    }
}

ALL_RELATIONS = [
    "above", "across from", "adjacent to", "against", "ahead of", "alongside",
    "at the back of", "at the edge of", "at the left side of", "at the right side of",
    "at the side of", "attached to", "away from", "behind", "below", "beneath",
    "beside", "beyond", "by", "close to", "connected to", "consists of", "contains",
    "enclosed by", "facing", "facing away from", "far away from", "far from",
    "has as a part", "in", "in front of", "in the middle of", "inside", "into",
    "left of", "near", "next to", "off", "on", "on top of", "opposite to",
    "outside", "over", "parallel to", "part of", "perpendicular to", "right of",
    "surrounding", "touching", "toward", "under", "within",
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


def sweep_relation(rel, indices, vsr_all, vecs_by_feat, processor_pt, model_pt, nns_pt, yes_ids, no_ids, device):
    base_correct = base_total = 0
    feat_results = {}

    imgs = []
    lbls = []
    for vi in indices:
        ex = vsr_all[vi]
        img = _load_image(ex)
        if img is None: continue
        imgs.append((vi, img))
        lbls.append(int(ex.get("label", 0)))

    print(f"  BASE: computing...", flush=True)
    base_preds = []
    for i, (vi, img) in enumerate(imgs):
        ex = vsr_all[vi]; lbl = lbls[i]
        try:
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                  processor_pt, model_pt, device=device)
            _, img_end = get_image_token_positions(iids)
            with torch.inference_mode():
                out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
            pb, _ = _pm(out.logits[0,-1,:], yes_ids, no_ids)
            base_correct += (pb == lbl)
            base_total += 1
            base_preds.append((vi, img, iids, attn, pv, img_end, lbl))
        except Exception as e:
            print(f"    [ERR base] vi={vi}: {e}", flush=True)

    base_acc = 100 * base_correct / max(base_total, 1)
    print(f"  BASE: {base_acc:.2f}%  N={base_total}", flush=True)

    for feat_key, cfg in FEATURE_CONFIGS.items():
        vecs = vecs_by_feat.get(feat_key, {})
        if not vecs:
            print(f"  [SKIP] {feat_key}: no vectors", flush=True)
            continue

        print(f"  [FEAT {feat_key}] oracle_α={cfg['oracle_alpha']}", flush=True)
        alphas_results = {}
        best_delta = -999; best_alpha = None

        for alpha in cfg["alphas"]:
            correct = 0
            for vi, img, iids, attn, pv, img_end, lbl in base_preds:
                try:
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
                    correct += (ps == lbl)
                except Exception as e:
                    print(f"      [ERR α={alpha}] vi={vi}: {e}", flush=True)

            acc = 100 * correct / max(base_total, 1)
            delta = acc - base_acc
            marker = "*** " if delta > 0 else ("[-] " if delta < 0 else "")
            oracle_mark = "[ORACLE]" if abs(alpha - cfg["oracle_alpha"]) < 1e-6 else ""
            print(f"    α={alpha}: {acc:.2f}% (Δ={delta:+.2f}%) {marker}{oracle_mark}", flush=True)
            alphas_results[str(alpha)] = {"acc": acc, "delta_acc": delta}
            if delta > best_delta:
                best_delta = delta; best_alpha = alpha

        print(f"    BEST α={best_alpha}: Δ={best_delta:+.2f}%", flush=True)
        feat_results[feat_key] = {
            "best_alpha": best_alpha, "best_delta": best_delta,
            "oracle_alpha": cfg["oracle_alpha"],
            "oracle_delta": alphas_results.get(str(cfg["oracle_alpha"]), {}).get("delta_acc", 0),
            "alphas": alphas_results
        }

    return base_acc, base_total, feat_results


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    globals()["process_vlm_inputs"] = process_vlm_inputs
    globals()["get_image_token_positions"] = get_image_token_positions

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    out_path = OUT_DIR / "rel_at_the_left_side_of.json"

    if out_path.exists():
        print(f"[SKIP] {out_path} exists", flush=True)
        return

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    print("[INFO] Loading pt-448...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    print("[INFO] Loading CAA vectors...", flush=True)
    vecs_by_feat = {}
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
                vecs_by_feat[feat_key] = vecs
                print(f"  [LOADED] {feat_key}", flush=True)

    print("[INFO] Finding at_left_side_of indices...", flush=True)
    rel_target = "at the left side of"
    indices = [vi for vi in range(len(vsr_all))
               if parse_relation(str(vsr_all[vi].get("caption", ""))) == rel_target]
    print(f"  Found {len(indices)} samples", flush=True)

    print(f"\n[RELATION] '{rel_target}'  N={len(indices)}", flush=True)
    base_acc, n, feat_results = sweep_relation(
        rel_target, indices, vsr_all, vecs_by_feat,
        processor_pt, model_pt, nns_pt, yes_ids, no_ids, device
    )

    # Merge with known results
    all_features = dict(KNOWN_RESULTS)
    all_features.update(feat_results)

    result = {
        "relation": rel_target,
        "n": n,
        "base_acc": base_acc,
        "features": all_features
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [SAVED] {out_path}", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
