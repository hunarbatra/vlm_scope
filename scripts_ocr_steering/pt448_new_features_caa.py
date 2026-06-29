#!/usr/bin/env python3
"""
Compute CAA vectors for 2 new features (L15/F8844 "behind", L15/F1149 "in front of")
and run per-relation subset steering to check if they can fill the coverage gap.

Both features are at layer 15 (same as L15/F220 which gave +2.09% universal best)
and have high odds_ratios (15.4 and 16.7) in mix-448.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_new_features_caa/
Usage:  CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_new_features_caa.py
"""

import os, sys, json, warnings, math, hashlib
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_MIX   = "google/paligemma2-3b-mix-448"
MODEL_PT    = "google/paligemma2-3b-pt-448"
N_LAYERS    = 26
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_new_features_caa")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

NEW_FEATURES = [
    {"key": "L15_F8844",  "layer": 15, "feature": 8844,  "relations": ["behind"],      "start": 15},
    {"key": "L15_F1149",  "layer": 15, "feature": 1149,  "relations": ["in front of"], "start": 15},
]

ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MIN_SAMPLES = 20

import re
ALL_RELATIONS = [
    "above","across from","adjacent to","against","ahead of","alongside",
    "at the back of","at the edge of","at the left side of","at the right side of",
    "at the side of","attached to","away from","behind","below","beneath",
    "beside","beyond","by","close to","connected to","consists of","contains",
    "enclosed by","facing","facing away from","far away from","far from",
    "has as a part","in","in front of","in the middle of","inside","into",
    "left of","near","next to","off","on","on top of","opposite to","outside",
    "over","parallel to","part of","perpendicular to","right of","surrounding",
    "touching","toward","under","within",
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
    url = ex.get("image_link","")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
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

    # Group by relation
    rel_indices = {}
    for vi in range(N):
        r = parse_relation(str(vsr_all[vi].get("caption","")))
        if r: rel_indices.setdefault(r, []).append(vi)

    # ---- PHASE 1: Extract CAA vectors from mix-448 ----
    vecs_path = OUT_DIR / "caa_vectors_new.pt"
    if vecs_path.exists():
        print("[SKIP] CAA vectors already extracted.", flush=True)
        all_vecs = torch.load(vecs_path, map_location="cpu")
    else:
        print("[INFO] Loading mix-448 for CAA extraction...", flush=True)
        proc_mix = AutoProcessor.from_pretrained(MODEL_MIX)
        model_mix = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_MIX, dtype=torch.bfloat16).to(device).eval()

        all_vecs = {}
        for cfg in NEW_FEATURES:
            feat_key = cfg["key"]
            relations = cfg["relations"]
            print(f"\n[PHASE 1] Extracting CAA for {feat_key} rels={relations}", flush=True)
            
            pos_idx = []
            neg_idx = []
            for rel in relations:
                idxs = rel_indices.get(rel, [])
                for vi in idxs:
                    lbl = int(vsr_all[vi].get("label", 0))
                    if lbl == 1: pos_idx.append(vi)
                    else: neg_idx.append(vi)
            print(f"  pos={len(pos_idx)}, neg={len(neg_idx)}", flush=True)

            # Collect hidden states at all layers
            pos_hs = {l: [] for l in range(N_LAYERS)}
            neg_hs = {l: [] for l in range(N_LAYERS)}

            for sign, idxs_list in [(1, pos_idx), (0, neg_idx)]:
                hs_dict = pos_hs if sign == 1 else neg_hs
                for vi in idxs_list:
                    ex = vsr_all[vi]; img = _load_image(ex)
                    if img is None: continue
                    try:
                        iids, attn, pv = process_vlm_inputs(
                            img, _build_vsr_prompt(str(ex.get("caption",""))),
                            proc_mix, model_mix, device=device)
                        _, img_end = get_image_token_positions(iids)
                        with torch.inference_mode():
                            out = model_mix(input_ids=iids, attention_mask=attn,
                                           pixel_values=pv, output_hidden_states=True, use_cache=False)
                        # Use last text token hidden state at each layer
                        for l in range(N_LAYERS):
                            h = out.hidden_states[l+1][0, -1, :].float().cpu()
                            hs_dict[l].append(h)
                    except Exception as e:
                        print(f"  [ERR] vi={vi}: {e}", flush=True)

            # Compute CAA = mean(pos) - mean(neg) at each layer
            feat_vecs = {}
            for l in range(N_LAYERS):
                if not pos_hs[l] or not neg_hs[l]: continue
                v_pos = torch.stack(pos_hs[l]).mean(0)
                v_neg = torch.stack(neg_hs[l]).mean(0)
                v_raw = v_pos - v_neg
                v_norm = v_raw / v_raw.norm().clamp(min=1e-8)
                feat_vecs[l] = {"v_caa_raw": v_raw, "v_caa_norm": v_norm,
                                "n_pos": len(pos_hs[l]), "n_neg": len(neg_hs[l])}
            all_vecs[feat_key] = feat_vecs
            print(f"  [DONE] {feat_key}: {len(feat_vecs)} layer vectors, "
                  f"n_pos={len(pos_hs[0])}, n_neg={len(neg_hs[0])}", flush=True)

        torch.save(all_vecs, vecs_path)
        print(f"[SAVED] {vecs_path}", flush=True)
        del model_mix, proc_mix
        torch.cuda.empty_cache()

    # ---- PHASE 2: Steer pt-448 with new vectors on per-relation subsets ----
    print("\n[INFO] Loading pt-448 for steering...", flush=True)
    proc_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(proc_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    # Convert vectors to device
    dev_vecs = {}
    for feat_key, fv in all_vecs.items():
        dev_vecs[feat_key] = {
            l: (ld["v_caa_norm"] / ld["v_caa_norm"].norm().clamp(min=1e-8)).to(dtype).to(device)
            for l, ld in fv.items()
        }

    # Evaluate on all relations (own + cross)
    for cfg in NEW_FEATURES:
        feat_key = cfg["key"]
        start = cfg["start"]
        own_rels = cfg["relations"]
        vecs = dev_vecs.get(feat_key, {})
        if not vecs:
            print(f"[SKIP] {feat_key} no vectors", flush=True); continue

        result_path = OUT_DIR / f"steer_{feat_key}.json"
        if result_path.exists():
            print(f"[SKIP] {feat_key} already done", flush=True); continue

        print(f"\n{'='*60}", flush=True)
        print(f"[STEERING] {feat_key}  own_rels={own_rels}  start={start}", flush=True)

        rel_results = {}
        # Focus on own-relation + related high-importance relations
        eval_rels = list(set(own_rels + ["behind", "in front of", "under", "on top of", "on", "above", "below"]))

        for rel in eval_rels:
            idxs = rel_indices.get(rel, [])
            if len(idxs) < MIN_SAMPLES: continue
            is_own = rel in own_rels

            # Baseline
            correct_b = total_b = 0; mb = []
            for vi in idxs:
                ex = vsr_all[vi]; img = _load_image(ex)
                if img is None: continue
                lbl = int(ex.get("label",0))
                try:
                    iids, attn, pv = process_vlm_inputs(
                        img, _build_vsr_prompt(str(ex.get("caption",""))),
                        proc_pt, model_pt, device=device)
                    with torch.inference_mode():
                        out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                    pred, m = _pm(out.logits[0,-1,:], yes_ids, no_ids)
                    mb.append(m if lbl==1 else -m)
                except Exception: pred=0; mb.append(0.0)
                total_b += 1; correct_b += (pred==lbl)
            base_acc = correct_b / max(total_b,1) * 100

            # Alpha sweep
            alpha_res = {}
            best_delta = -999; best_alpha = None
            for alpha in ALPHAS:
                correct = total = 0; margins = []
                for vi in idxs:
                    ex = vsr_all[vi]; img = _load_image(ex)
                    if img is None: continue
                    lbl = int(ex.get("label",0))
                    try:
                        iids, attn, pv = process_vlm_inputs(
                            img, _build_vsr_prompt(str(ex.get("caption",""))),
                            proc_pt, model_pt, device=device)
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
                        pred, m = _pm(logits_s[0,-1,:], yes_ids, no_ids)
                        margins.append(m if lbl==1 else -m)
                    except Exception: pred=0; margins.append(0.0)
                    total += 1; correct += (pred==lbl)
                acc = correct / max(total,1) * 100
                da = acc - base_acc
                mk = " ***" if da > best_delta else ""
                if da > best_delta: best_delta = da; best_alpha = alpha
                alpha_res[str(alpha)] = {"acc": acc, "delta_acc": da, "margin": sum(margins)/max(len(margins),1)}
                own_mark = "*" if is_own else " "
                print(f"  [{own_mark}{rel[:22]:<22}] α={alpha}: {acc:.2f}% (Δ={da:+.2f}%){mk}", flush=True)
            rel_results[rel] = {
                "n": total_b, "base_acc": base_acc, "is_own_relation": is_own,
                "best_alpha": best_alpha, "best_delta": best_delta, "alphas": alpha_res
            }

        with open(result_path, "w") as f: json.dump({"feat_key": feat_key, "relations": rel_results}, f, indent=2)
        print(f"  [SAVED] {result_path}", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
