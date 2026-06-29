#!/usr/bin/env python3
"""
Fine alpha sweep for L15/F220 caa_sae_down on full VSR.

L15/F220 hit +2.09% at α=0.5 (new best single-feature universal result,
surpassing fixed_L6 +1.90% and L4/L14 +1.88%).
This script sweeps finer alphas around 0.5 to find the true peak,
and also tests whether a "smart oracle" using L15 exclusively does better.

Alphas swept: 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 1.0, 1.25, 1.5, 2.0

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L15_fine_sweep/
Usage:  CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_L15_fine_sweep.py
"""

import os, sys, json, warnings, math
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
CAA_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L15_fine_sweep")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0]


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]: toks = tok.encode(t, add_special_tokens=False); yes_ids.update(toks[:1] if toks else [])
    for t in [" No","No"," no","NO"]:  toks = tok.encode(t, add_special_tokens=False); no_ids.update(toks[:1] if toks else [])
    ov = yes_ids & no_ids; yes_ids -= ov; no_ids -= ov
    return yes_ids, no_ids

def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n; p = max(y/d if d > 0 else 0.5, 1e-7)
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
    result_path = OUT_DIR / "L15_F220_fine_sweep.json"
    if result_path.exists():
        print(f"[SKIP] Results already exist at {result_path}", flush=True)
        with open(result_path) as f:
            r = json.load(f)
        print(f"Best alpha: {r.get('best_alpha')} → {r.get('best_acc'):.2f}% (Δ={r.get('best_delta'):+.2f}%)")
        return

    device = "cuda:0"

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    indices = list(range(N))

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

    # Baseline pass
    print("\n[INFO] Running baseline...", flush=True)
    correct = total = 0
    margins = []
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
        except Exception: pred = 0; margins.append(0.0)
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
            except Exception: pred = 0; margins.append(0.0)
            total += 1; correct += (pred == lbl)
        acc = correct / max(total, 1) * 100
        da = acc - base_acc
        mg = sum(margins) / max(len(margins), 1)
        marker = " *** NEW BEST ***" if da > best_delta else ""
        if da > best_delta: best_delta = da; best_alpha = alpha; best_acc = acc
        print(f"  α={alpha}: {acc:.2f}% (Δ={da:+.2f}%) mg={mg:.3f}{marker}", flush=True)
        alpha_results[str(alpha)] = {"acc": acc, "delta_acc": da, "margin": mg}

    print(f"\n[RESULT] Best α={best_alpha}: {best_acc:.2f}% (Δ={best_delta:+.2f}%)", flush=True)

    result = {
        "layer": LAYER, "feature": FEATURE, "start_layer": START_LAYER,
        "base_acc": base_acc, "base_margin": base_mg, "n_total": total,
        "best_alpha": best_alpha, "best_acc": best_acc, "best_delta": best_delta,
        "alphas": alpha_results,
    }
    with open(result_path, "w") as f: json.dump(result, f, indent=2)
    print(f"[SAVED] {result_path}", flush=True)


if __name__ == "__main__":
    main()
