#!/usr/bin/env python3
"""
Per-relation alpha sweep — proper subset-level ablation for caa_sae_down steering.

For each VSR relation with N>=20 samples, sweeps alpha for every feature on that
relation's subset. This is the correct evaluation methodology (matching original
ablation codebase): report per-feature steering results on own-relation subsets,
not diluted by full-VSR averaging.

Also evaluates FC-CAA vectors (full-dataset feature-conditioned) on own-relation
subsets for direct comparison.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_per_relation_alpha_sweep/
  - per_relation_{relation_slug}.json  — per alpha, per feature results
  - summary_own_relation.json          — own-relation best per feature
  - summary_cross_relation.json        — cross-relation matrix

Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_per_relation_alpha_sweep.py
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
FCCAA_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fc_caa/fc_caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_per_relation_alpha_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIN_SAMPLES = 20

FEATURE_CONFIGS = {
    "L4_F14233":  {"layer": 4,  "feature": 14233, "start": 0,  "own_rels": ["ahead of"]},
    "L14_F10561": {"layer": 14, "feature": 10561, "start": 0,  "own_rels": ["close to"]},
    "L12_F2257":  {"layer": 12, "feature": 2257,  "start": 1,  "own_rels": ["facing"]},
    "L15_F220":   {"layer": 15, "feature": 220,   "start": 15, "own_rels": ["across from", "at the left side of"]},
    "L11_F12278": {"layer": 11, "feature": 12278, "start": 5,  "own_rels": ["touching"]},
    "L9_F387":    {"layer": 9,  "feature": 387,   "start": 1,  "own_rels": ["at the right side of"]},
    "L6_F7539":   {"layer": 6,  "feature": 7539,  "start": 1,  "own_rels": ["left of", "right of"]},
    "L9_F7540":   {"layer": 9,  "feature": 7540,  "start": 9,  "own_rels": ["consists of"]},
}

# Alphas to sweep per feature per relation
ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# All VSR relations
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


def parse_relation(caption: str):
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


def run_subset_alpha_sweep(indices, vsr_all, nns_pt, model_pt, processor_pt,
                            yes_ids, no_ids, vecs, start, alpha, device):
    """Run one pass over a subset at a given alpha, return (acc, delta_acc, margin)."""
    correct = total = 0
    margins = []
    from utils import process_vlm_inputs, get_image_token_positions
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
            pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
            margins.append(m if lbl == 1 else -m)
        except Exception:
            pred = 0; margins.append(0.0)
        total += 1; correct += (pred == lbl)
    acc = correct / max(total, 1) * 100
    mg = sum(margins) / max(len(margins), 1)
    return acc, mg, total


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

    # Group indices by relation
    print("[INFO] Parsing relations...", flush=True)
    rel_indices = {}
    for vi in range(N):
        cap = str(vsr_all[vi].get("caption", ""))
        r = parse_relation(cap)
        if r is not None:
            rel_indices.setdefault(r, []).append(vi)
    for r, idxs in sorted(rel_indices.items()):
        print(f"  {r:<30} N={len(idxs)}", flush=True)

    print("[INFO] Loading pt-448...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    # Load caa_sae_down vectors
    print("[INFO] Loading caa_sae_down vectors...", flush=True)
    caa_vecs = {}
    for feat_key, cfg in FEATURE_CONFIGS.items():
        path = CAA_DIR / f"caa_L{cfg['layer']}_F{cfg['feature']}.pt"
        if not path.exists(): print(f"  [WARN] Missing {path}"); continue
        saved = torch.load(path, map_location="cpu")
        vecs = {}
        for l, ld in saved.get("caa_data", {}).items():
            v = ld.get("v_caa_norm")
            if v is not None:
                vecs[int(l)] = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)
        caa_vecs[feat_key] = vecs
        print(f"  [caa] {feat_key} ({len(vecs)} layers)", flush=True)

    # Load fc_caa vectors
    print("[INFO] Loading fc_caa vectors...", flush=True)
    fc_vecs = {}
    for feat_key, cfg in FEATURE_CONFIGS.items():
        path = FCCAA_DIR / f"fc_caa_L{cfg['layer']}_F{cfg['feature']}.pt"
        if not path.exists(): continue
        saved = torch.load(path, map_location="cpu")
        vecs = {}
        for l, ld in saved.get("fc_caa_data", {}).items():
            v = ld.get("v_fc_caa_norm")
            if v is not None:
                vecs[int(l)] = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)
        if vecs:
            fc_vecs[feat_key] = vecs
            print(f"  [fc]  {feat_key} ({len(vecs)} layers)", flush=True)

    # Own-relation summary accumulator
    own_rel_summary = {}

    # Process each relation
    for rel in sorted(rel_indices.keys(), key=lambda r: -len(rel_indices.get(r, []))):
        idxs = rel_indices.get(rel, [])
        if len(idxs) < MIN_SAMPLES:
            print(f"\n[SKIP] {rel} (N={len(idxs)} < {MIN_SAMPLES})", flush=True)
            continue

        slug = rel.replace(" ", "_").replace("/", "_")
        result_path = OUT_DIR / f"rel_{slug}.json"
        if result_path.exists():
            print(f"[SKIP] {rel} already done", flush=True)
            with open(result_path) as f:
                existing = json.load(f)
            # Update own_rel_summary
            for feat_key in FEATURE_CONFIGS:
                if rel in FEATURE_CONFIGS[feat_key]["own_rels"]:
                    if feat_key in existing.get("features", {}):
                        own_rel_summary.setdefault(feat_key, {})[rel] = existing["features"][feat_key]
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[RELATION] {rel}  N={len(idxs)}", flush=True)

        # Baseline
        correct_b = total_b = 0; marg_b = []
        for vi in idxs:
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
                marg_b.append(m if lbl == 1 else -m)
            except Exception:
                pred = 0; marg_b.append(0.0)
            total_b += 1; correct_b += (pred == lbl)
        base_acc = correct_b / max(total_b, 1) * 100
        base_mg = sum(marg_b) / max(len(marg_b), 1)
        print(f"  BASE: {base_acc:.2f}%  N={total_b}", flush=True)

        feat_results = {}

        # caa_sae_down alpha sweep per feature
        for feat_key, cfg in FEATURE_CONFIGS.items():
            if feat_key not in caa_vecs: continue
            vecs = caa_vecs[feat_key]
            start = cfg["start"]
            is_own = rel in cfg["own_rels"]
            alpha_res = {}
            best_delta = -999; best_alpha = None
            for alpha in ALPHAS:
                acc, mg, n = run_subset_alpha_sweep(
                    idxs, vsr_all, nns_pt, model_pt, processor_pt,
                    yes_ids, no_ids, vecs, start, alpha, device)
                da = acc - base_acc
                marker = " ***" if da > best_delta else ""
                if da > best_delta: best_delta = da; best_alpha = alpha
                alpha_res[str(alpha)] = {"acc": acc, "delta_acc": da, "margin": mg, "n": n}
                print(f"  [{feat_key}{'*' if is_own else ' '}] α={alpha}: {acc:.2f}% (Δ={da:+.2f}%){marker}", flush=True)
            feat_results[feat_key] = {
                "is_own_relation": is_own,
                "best_alpha": best_alpha, "best_delta": best_delta,
                "alphas": alpha_res,
            }
            if is_own:
                own_rel_summary.setdefault(feat_key, {})[rel] = feat_results[feat_key]

        # fc_caa alpha sweep per feature (same subset)
        for feat_key, cfg in FEATURE_CONFIGS.items():
            if feat_key not in fc_vecs: continue
            vecs = fc_vecs[feat_key]
            start = cfg["start"]
            is_own = rel in cfg["own_rels"]
            fk = f"fc_{feat_key}"
            alpha_res = {}
            best_delta = -999; best_alpha = None
            for alpha in ALPHAS:
                acc, mg, n = run_subset_alpha_sweep(
                    idxs, vsr_all, nns_pt, model_pt, processor_pt,
                    yes_ids, no_ids, vecs, start, alpha, device)
                da = acc - base_acc
                if da > best_delta: best_delta = da; best_alpha = alpha
                alpha_res[str(alpha)] = {"acc": acc, "delta_acc": da, "margin": mg, "n": n}
            feat_results[fk] = {
                "is_own_relation": is_own, "vector_type": "fc_caa",
                "best_alpha": best_alpha, "best_delta": best_delta,
                "alphas": alpha_res,
            }
            if is_own:
                own_rel_summary.setdefault(f"fc_{feat_key}", {})[rel] = feat_results[fk]

        result = {
            "relation": rel, "n": total_b, "base_acc": base_acc, "base_margin": base_mg,
            "features": feat_results,
        }
        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        print(f"  [SAVED] {result_path}", flush=True)

    # Save own-relation summary
    with open(OUT_DIR / "summary_own_relation.json", "w") as f:
        json.dump(own_rel_summary, f, indent=2)
    print("\n[OWN-RELATION SUMMARY]", flush=True)
    print(f"{'Feature':<14} {'Relation':<30} {'Best α':>7} {'Best Δ':>8}  caa vs fc_caa", flush=True)
    print("-" * 75, flush=True)
    for feat_key in FEATURE_CONFIGS:
        for rel, rd in own_rel_summary.get(feat_key, {}).items():
            fc_rd = own_rel_summary.get(f"fc_{feat_key}", {}).get(rel, {})
            fc_str = f"fc={fc_rd.get('best_delta', 0):+.2f}%" if fc_rd else "fc=n/a"
            print(f"{feat_key:<14} {rel:<30} {str(rd.get('best_alpha','-')):>7} {rd.get('best_delta',0):>+7.2f}%  {fc_str}", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
