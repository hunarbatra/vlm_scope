#!/usr/bin/env python3
"""
Cross-feature sweep for beside(N=188), facing_away_from(N=180), away_from(N=155), right_of(N=113).
All only ever tested with their single assigned feature. Full 8-feature cross-sweep.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_beside_facingaway_awayfrom_rightof_sweep/
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_beside_facingaway_awayfrom_rightof_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TARGET_RELATIONS = ["beside", "facing away from", "away from", "right of"]

FEATURE_CONFIGS = {
    "L4_F14233":  {"layer": 4,  "feature": 14233, "start": 1,  "oracle_alpha": 0.9,
                   "alphas": [0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0]},
    "L14_F10561": {"layer": 14, "feature": 10561, "start": 1,  "oracle_alpha": 2.0,
                   "alphas": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
    "L11_F12278": {"layer": 11, "feature": 12278, "start": 5,  "oracle_alpha": 0.45,
                   "alphas": [0.2, 0.3, 0.45, 0.5, 0.6, 0.75, 1.0]},
    "L12_F2257":  {"layer": 12, "feature": 2257,  "start": 1,  "oracle_alpha": 0.75,
                   "alphas": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]},
    "L15_F220":   {"layer": 15, "feature": 220,   "start": 15, "oracle_alpha": 0.7,
                   "alphas": [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25]},
    "L6_F7539":   {"layer": 6,  "feature": 7539,  "start": 1,  "oracle_alpha": 1.5,
                   "alphas": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]},
    "L9_F387":    {"layer": 9,  "feature": 387,   "start": 1,  "oracle_alpha": 0.4,
                   "alphas": [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]},
    "L9_F7540":   {"layer": 9,  "feature": 7540,  "start": 1,  "oracle_alpha": 0.25,
                   "alphas": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]},
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

    print("[INFO] Parsing relations...", flush=True)
    rel_indices = {r: [] for r in TARGET_RELATIONS}
    for vi in range(len(vsr_all)):
        r = parse_relation(str(vsr_all[vi].get("caption", "")))
        if r in rel_indices:
            rel_indices[r].append(vi)

    for rel in TARGET_RELATIONS:
        out_path = OUT_DIR / f"rel_{rel.replace(' ', '_')}.json"
        if out_path.exists():
            print(f"[SKIP] {rel} — already done", flush=True)
            continue

        indices = rel_indices[rel]
        print(f"\n[RELATION] '{rel}'  N={len(indices)}", flush=True)

        base_preds = []
        base_correct = 0
        for vi in indices:
            ex = vsr_all[vi]; lbl = int(ex.get("label", 0))
            img = _load_image(ex)
            if img is None: continue
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, _build_vsr_prompt(str(ex.get("caption",""))),
                    processor_pt, model_pt, device=device)
                _, img_end = get_image_token_positions(iids)
                with torch.inference_mode():
                    out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pb, _ = _pm(out.logits[0,-1,:], yes_ids, no_ids)
                base_correct += (pb == lbl)
                base_preds.append((vi, iids, attn, pv, img_end, lbl))
            except Exception as e:
                print(f"  [ERR base] vi={vi}: {e}", flush=True)

        n = len(base_preds)
        base_acc = 100 * base_correct / max(n, 1)
        print(f"  BASE: {base_acc:.2f}%  N={n}", flush=True)

        feat_results = {}
        for feat_key, cfg in FEATURE_CONFIGS.items():
            vecs = vecs_by_feat.get(feat_key, {})
            if not vecs: continue
            print(f"  [FEAT {feat_key}] oracle_α={cfg['oracle_alpha']}", flush=True)
            alphas_results = {}
            best_delta = -999; best_alpha = None
            for alpha in cfg["alphas"]:
                correct = 0
                for vi, iids, attn, pv, img_end, lbl in base_preds:
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
                        print(f"    [ERR α={alpha}] vi={vi}: {e}", flush=True)
                acc = 100 * correct / max(n, 1)
                delta = acc - base_acc
                marker = "***" if delta > 0 else ("[-]" if delta < 0 else "")
                oracle_mark = "[ORACLE]" if abs(alpha - cfg["oracle_alpha"]) < 1e-6 else ""
                print(f"    α={alpha}: {acc:.2f}% (Δ={delta:+.2f}%) {marker} {oracle_mark}", flush=True)
                alphas_results[str(alpha)] = {"acc": acc, "delta_acc": delta}
                if delta > best_delta:
                    best_delta = delta; best_alpha = alpha
            print(f"    BEST α={best_alpha}: Δ={best_delta:+.2f}%", flush=True)
            feat_results[feat_key] = {
                "best_alpha": best_alpha, "best_delta": best_delta,
                "oracle_alpha": cfg["oracle_alpha"],
                "oracle_delta": alphas_results.get(str(cfg["oracle_alpha"]), {}).get("delta_acc", 0.0),
                "alphas": alphas_results
            }

        result = {"relation": rel, "n": n, "base_acc": base_acc, "features": feat_results}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [SAVED] {out_path}", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
