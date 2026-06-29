#!/usr/bin/env python3
"""
Test L11/F12278 with late-start injection (start_layer=15) on all 4 oracle-assigned relations.

Motivation: Late-start sweep showed L11 with start=15, α=1.0 gives +7.84% on "touching"
vs natural start=5, α=0.45 giving +6.01% (+1.83pp improvement).

This tests whether late-start also helps (or hurts) on L11's other 3 assigned relations:
- on top of (N=?)
- surrounding (N=?)
- under (N=?)

Also tests natural-start on these relations for comparison.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L11_latestart_sweep/
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_L11_latestart_sweep.py
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
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L11_latestart_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

L11_LAYER    = 11
L11_FEATURE  = 12278

# Natural start is 5; late start is 15
NATURAL_START = 5
LATE_START    = 15

# Oracle-assigned relations for L11/F12278
TARGET_RELATIONS = ["on top of", "surrounding", "touching", "under"]

# Alphas to test for both natural-start and late-start
NATURAL_ALPHAS = [0.25, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0]
LATE_ALPHAS    = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

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


def eval_feature(indices, vsr_all, vecs, start_layer, alpha, nns_model,
                 yes_ids, no_ids, processor, model, device):
    correct = total = 0
    for vi in indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        lbl = int(ex.get("label", 0))
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from utils import process_vlm_inputs, get_image_token_positions
            iids, attn, pv = process_vlm_inputs(
                img, _build_vsr_prompt(str(ex.get("caption", ""))),
                processor, model, device=device)
            _, img_end = get_image_token_positions(iids)
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in range(start_layer, N_LAYERS):
                    if l not in vecs: continue
                    v_l = vecs[l]
                    v_col = v_l.unsqueeze(1)
                    lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                    ones = (lo @ v_col) * 0.0 + 1.0
                    lo += alpha * ones * v_l
                logits_s = nns_model.output.logits.save()
            pred, _ = _pm(logits_s[0, -1, :], yes_ids, no_ids)
        except Exception:
            pred = 0
        total += 1; correct += (pred == lbl)
    return correct / max(total, 1) * 100, total


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))

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
    for rel in TARGET_RELATIONS:
        print(f"  {rel}: N={len(rel_indices.get(rel, []))}", flush=True)

    print("[INFO] Loading pt-448...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    from utils import process_vlm_inputs, get_image_token_positions

    print("[INFO] Loading L11/F12278 CAA vectors...", flush=True)
    path = CAA_DIR / f"caa_L{L11_LAYER}_F{L11_FEATURE}.pt"
    saved = torch.load(path, map_location="cpu")
    vecs = {}
    for l, ld in saved.get("caa_data", {}).items():
        v = ld.get("v_caa_norm")
        if v is not None:
            vecs[int(l)] = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)
    print(f"  [LOADED] L11/F12278 ({len(vecs)} layers)", flush=True)

    summary = {}

    for rel in TARGET_RELATIONS:
        indices = rel_indices.get(rel, [])
        if not indices:
            print(f"\n[SKIP] {rel} — no samples", flush=True); continue

        result_path = OUT_DIR / f"rel_{rel.replace(' ', '_')}.json"
        if result_path.exists():
            print(f"[SKIP] {rel} already done", flush=True)
            with open(result_path) as f:
                r = json.load(f)
            summary[rel] = r
            continue

        print(f"\n[RELATION] '{rel}'  N={len(indices)}", flush=True)

        # Baseline
        correct = total = 0
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
                pred, _ = _pm(out.logits[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            total += 1; correct += (pred == lbl)
        base_acc = correct / max(total, 1) * 100
        print(f"  BASE: {base_acc:.2f}%  N={total}", flush=True)

        rel_result = {"relation": rel, "n": total, "base_acc": base_acc,
                      "natural_start": {}, "late_start": {}}

        # Natural-start sweep
        print(f"  [NATURAL START={NATURAL_START}]", flush=True)
        best_nat_delta = -999; best_nat_alpha = None
        for alpha in NATURAL_ALPHAS:
            acc, n = eval_feature(indices, vsr_all, vecs, NATURAL_START, alpha,
                                  nns_pt, yes_ids, no_ids, processor_pt, model_pt, device)
            da = acc - base_acc
            marker = " ***" if da > best_nat_delta else ""
            if da > best_nat_delta: best_nat_delta = da; best_nat_alpha = alpha
            oracle_m = " [ORACLE]" if alpha == 0.45 else ""
            print(f"    α={alpha}: {acc:.2f}% (Δ={da:+.2f}%){marker}{oracle_m}", flush=True)
            rel_result["natural_start"][str(alpha)] = {"acc": acc, "delta_acc": da}
        print(f"    BEST natural: α={best_nat_alpha}, Δ={best_nat_delta:+.2f}%", flush=True)
        rel_result["best_natural_alpha"] = best_nat_alpha
        rel_result["best_natural_delta"] = best_nat_delta

        # Late-start sweep
        print(f"  [LATE START={LATE_START}]", flush=True)
        best_late_delta = -999; best_late_alpha = None
        for alpha in LATE_ALPHAS:
            acc, n = eval_feature(indices, vsr_all, vecs, LATE_START, alpha,
                                  nns_pt, yes_ids, no_ids, processor_pt, model_pt, device)
            da = acc - base_acc
            marker = " ***" if da > best_late_delta else ""
            if da > best_late_delta: best_late_delta = da; best_late_alpha = alpha
            print(f"    α={alpha}: {acc:.2f}% (Δ={da:+.2f}%){marker}", flush=True)
            rel_result["late_start"][str(alpha)] = {"acc": acc, "delta_acc": da}
        print(f"    BEST late: α={best_late_alpha}, Δ={best_late_delta:+.2f}%  vs nat: {best_late_delta - best_nat_delta:+.2f}pp", flush=True)
        rel_result["best_late_alpha"] = best_late_alpha
        rel_result["best_late_delta"] = best_late_delta

        summary[rel] = rel_result
        with open(result_path, "w") as f:
            json.dump(rel_result, f, indent=2)
        print(f"  [SAVED] {result_path}", flush=True)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print(f"{'Relation':20s}  {'N':5s}  {'Base':7s}  {'Nat best':8s}  {'Late best':9s}  {'Late-Nat':8s}", flush=True)
    print("-" * 80, flush=True)
    for rel, r in summary.items():
        nat = r.get("best_natural_delta", 0)
        late = r.get("best_late_delta", 0)
        print(f"{rel:20s}  {r.get('n',0):5d}  {r.get('base_acc',0):7.2f}%  {nat:+8.2f}%  {late:+9.2f}%  {late-nat:+8.2f}pp", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
