#!/usr/bin/env python3
"""
Inject ALL 10 spatial features simultaneously with their individual best configs.

Ultimate test: if we stack all known spatial feature injections in one forward
pass, does the model's overall spatial reasoning improve across ALL VSR relations?

Each feature uses its best (strategy, alpha) from prior experiments.
Evaluates on each relation's examples AND on the full VSR test set.

Also tests a "top-5 only" combination and a "high-transfer only" combination.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_all10_combined/

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 pt448_all10_combined.py
"""

import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_NAME     = "google/paligemma2-3b-pt-448"
N_LAYERS       = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_all10_combined")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

DECAY_ML = 0.7
DECAY_RA = 0.85

# All 10 features with best known configs
ALL10 = [
    (4,  14233, "sae_only_down",  4.0),
    (12, 2257,  "all_ml",        50.0),
    (11, 12278, "single",        20.0),
    (9,  387,   "decay_fwd_ra",   2.0),
    (15, 220,   "sae_only_up",    5.0),
    (13, 15219, "downstream_ml", 30.0),
    (9,  7540,  "single",        10.0),
    (14, 10561, "all_ml",         2.0),
    (6,  7539,  "topK_ml",       20.0),
    (11, 9639,  "answer",        10.0),
]

# Top-5 by delta (excluding tiny N)
TOP5 = [(4,14233,"sae_only_down",4.0),(12,2257,"all_ml",50.0),(11,12278,"single",20.0),
        (9,387,"decay_fwd_ra",2.0),(15,220,"sae_only_up",5.0)]

# High-transfer only (xfer >= 0.40x)
HIGH_XFER = [(4,14233,"sae_only_down",4.0),(11,12278,"single",20.0),
             (14,10561,"all_ml",2.0),(6,7539,"topK_ml",20.0)]

TOP10_RELS = [
    (9,  387,   ["at the right side of"]),
    (14, 10561, ["close to"]),
    (11, 12278, ["touching"]),
    (9,  7540,  ["consists of"]),
    (4,  14233, ["ahead of"]),
    (6,  7539,  ["left of", "right of"]),
    (11, 9639,  ["in", "inside", "on"]),
    (13, 15219, ["behind"]),
    (15, 220,   ["across from", "at the left side of"]),
    (12, 2257,  ["facing"]),
]


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No","No"," no","NO"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    ov = yes_ids & no_ids; yes_ids -= ov; no_ids -= ov
    return yes_ids, no_ids

def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n; p = max(y/d if d > 0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1-p, 1e-7))

def _load_image(ex):
    url = ex.get("image_link","")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB"); img.save(cp,"JPEG")
        return img
    except Exception: return None

def _get_layer_weights(strategy, sae_layer):
    if strategy == "single":
        return {sae_layer: 1.0}
    elif strategy == "sae_only_down":
        return {l: 1.0 for l in range(sae_layer, N_LAYERS)}
    elif strategy == "sae_only_up":
        return {l: 1.0 for l in range(0, sae_layer + 1)}
    elif strategy == "decay_fwd_ra":
        return {l: DECAY_RA ** max(l - sae_layer, 0) for l in range(N_LAYERS)}
    elif strategy == "all_ml":
        return {l: DECAY_ML ** abs(l - sae_layer) for l in range(N_LAYERS)}
    elif strategy == "downstream_ml":
        return {l: DECAY_ML ** (l - sae_layer) for l in range(sae_layer, N_LAYERS)}
    elif strategy == "topK_ml":
        return {l: DECAY_ML ** abs(l - sae_layer) for l in range(max(0, sae_layer-2), min(N_LAYERS, sae_layer+3))}
    elif strategy == "answer":
        return {l: 1.0 for l in range(21, N_LAYERS)}
    return {sae_layer: 1.0}


def _run_eval(indices, vsr_all, fv_configs, processor, nns_model, model_raw,
              yes_ids, no_ids, device):
    """Run injection on indices; fv_configs = list of (fv, lw, alpha, fv_col)."""
    correct = total = 0; margins = []
    from utils import process_vlm_inputs, get_image_token_positions
    for vi in indices:
        ex = vsr_all[vi]; img = _load_image(ex)
        if img is None: continue
        label = int(ex.get("label", 0))
        try:
            iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                 processor, nns_model._module, device=device)
            _, img_end = get_image_token_positions(iids)
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in range(N_LAYERS):
                    for fv, lw, alpha, fv_col in fv_configs:
                        w = lw.get(l, 0.0)
                        if w < 1e-8: continue
                        lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        ones = (lo @ fv_col) * 0.0 + 1.0
                        lo += (alpha * w) * ones * fv
                logits_s = nns_model.output.logits.save()
            pred, m = _pm(logits_s[0,-1,:], yes_ids, no_ids)
            margins.append(m if label==1 else -m)
        except Exception: pred=0; margins.append(0.0)
        total+=1; correct+=(pred==label)
    acc = correct/max(total,1)*100
    mg = sum(margins)/max(len(margins),1)
    return acc, mg, total


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {MODEL_NAME}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train":"train.jsonl","dev":"dev.jsonl","test":"test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train","dev","test"]
    ])

    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation","")].append(vi)

    # Build per-relation index for top-10 only
    top10_indices = set()
    for _, _, rels in TOP10_RELS:
        for r in rels: top10_indices.update(relation_indices.get(r, []))
    top10_indices = list(top10_indices)
    all_indices = list(range(len(vsr_all)))

    print("[INFO] Loading all SAE feature vectors...", flush=True)
    fv_all = {}
    for layer_idx, feat_idx, strategy, alpha in ALL10:
        key = f"L{layer_idx}F{feat_idx}"
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feat_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        lw = _get_layer_weights(strategy, layer_idx)
        fv_all[key] = (fv, lw, alpha, fv.unsqueeze(1))
        del sae; torch.cuda.empty_cache()
        print(f"  {key} loaded ({strategy} α={alpha})", flush=True)

    # Run baseline on top-10 relation subset
    def run_baseline(indices, label=""):
        correct = total = 0; margins = []
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            lbl = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                     processor, model_raw, device=device)
                with torch.inference_mode():
                    out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, m = _pm(out.logits[0,-1,:], yes_ids, no_ids)
                margins.append(m if lbl==1 else -m)
            except Exception: pred=0; margins.append(0.0)
            total+=1; correct+=(pred==lbl)
        acc = correct/max(total,1)*100
        mg = sum(margins)/max(len(margins),1)
        print(f"[BASE{label}] {acc:.2f}% margin={mg:.3f} N={total}", flush=True)
        return acc, mg, total

    print("[INFO] Running baselines...", flush=True)
    base_top10_acc, base_top10_mg, base_top10_n = run_baseline(top10_indices, " top10-rels")

    all_results = []

    def run_combo(combo_name, feature_keys, indices, base_acc, base_mg):
        rp = OUT_DIR / f"combo_{combo_name}.json"
        if rp.exists():
            print(f"[SKIP] {combo_name}", flush=True)
            with open(rp) as f: r=json.load(f); all_results.append(r); return

        fv_configs = [fv_all[k] for k in feature_keys]
        print(f"[COMBO] {combo_name} ({len(feature_keys)} features, N={len(indices)})...", flush=True)
        acc, mg, total = _run_eval(indices, vsr_all, fv_configs, processor, nns_model,
                                   model_raw, yes_ids, no_ids, device)
        da = acc - base_acc; dm = mg - base_mg
        print(f"  {combo_name}: {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f} (Δ={dm:+.3f})", flush=True)
        r = {"combo": combo_name, "n_features": len(feature_keys), "feature_keys": feature_keys,
             "n_samples": total, "baseline_acc": base_acc, "baseline_margin": base_mg,
             "acc": acc, "delta_acc": da, "margin": mg, "delta_margin": dm}
        with open(rp,"w") as f: json.dump(r, f, indent=2)
        all_results.append(r)
        torch.cuda.empty_cache()

    # Build feature key lists
    all10_keys  = [f"L{l}F{f}" for l,f,_,_ in ALL10]
    top5_keys   = [f"L{l}F{f}" for l,f,_,_ in TOP5]
    hixfer_keys = [f"L{l}F{f}" for l,f,_,_ in HIGH_XFER]

    run_combo("all10_top10rels",  all10_keys,  top10_indices, base_top10_acc, base_top10_mg)
    run_combo("top5_top10rels",   top5_keys,   top10_indices, base_top10_acc, base_top10_mg)
    run_combo("hixfer_top10rels", hixfer_keys, top10_indices, base_top10_acc, base_top10_mg)

    # Also per-relation breakdown for all10 combo
    print("\n[PER-RELATION] all-10 combo breakdown...", flush=True)
    per_rel_results = {}
    for layer_idx, feat_idx, rels in TOP10_RELS:
        rel_indices = []
        for r in rels: rel_indices.extend(relation_indices.get(r, []))
        key = f"L{layer_idx}F{feat_idx}"
        # baseline for this relation
        correct=total=0; margins=[]
        for vi in rel_indices:
            ex=vsr_all[vi]; img=_load_image(ex)
            if img is None: continue
            lbl=int(ex.get("label",0))
            try:
                iids,attn,pv=process_vlm_inputs(img,_build_vsr_prompt(str(ex.get("caption",""))),processor,model_raw,device=device)
                with torch.inference_mode(): out=model_raw(input_ids=iids,attention_mask=attn,pixel_values=pv,use_cache=False)
                pred,m=_pm(out.logits[0,-1,:],yes_ids,no_ids); margins.append(m if lbl==1 else -m)
            except: pred=0; margins.append(0.0)
            total+=1; correct+=(pred==lbl)
        base_r=correct/max(total,1)*100; base_mr=sum(margins)/max(len(margins),1)
        acc_r, mg_r, n_r = _run_eval(rel_indices, vsr_all, [fv_all[k] for k in all10_keys],
                                      processor, nns_model, model_raw, yes_ids, no_ids, device)
        da_r = acc_r - base_r
        per_rel_results[str(rels)] = {"rels": rels, "n": n_r, "base": base_r, "acc": acc_r, "delta_acc": da_r}
        print(f"  {rels}: {base_r:.2f}% → {acc_r:.2f}% (Δ={da_r:+.2f}%)", flush=True)

    with open(OUT_DIR/"per_relation_all10.json","w") as f:
        json.dump(per_rel_results, f, indent=2)

    print(f"\n{'='*100}")
    print("All-10-Feature Combined Injection Summary")
    print(f"{'='*100}")
    for r in all_results:
        print(f"  {r['combo']}: base={r['baseline_acc']:.2f}% → {r['acc']:.2f}% (Δ={r['delta_acc']:+.2f}%)")
    print("\nPer-relation breakdown (all-10 combo):")
    for v in per_rel_results.values():
        print(f"  {v['rels']}: Δ={v['delta_acc']:+.2f}%")


if __name__ == "__main__":
    main()
