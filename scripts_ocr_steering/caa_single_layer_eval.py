#!/usr/bin/env python3
"""
Single (variant, layer) α-sweep on full VSR test — designed to parallelize
across GPUs. Writes into a per-layer results.json so multiple copies can run
without collision.

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_single_layer_eval.py --variant meanpool --layer 13
"""
import os, sys, json, gc, hashlib, warnings, argparse
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

PT_MODEL    = "google/paligemma2-3b-pt-448"
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
PAIRED_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken")

OUT_ROOT    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_single_layer_eval")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 8777
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]


def _build_vsr_prompt(statement):
    return ("Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
            f"Statement: {statement.strip()}\nAnswer:")

def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No","No"," no","NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap; no_ids -= overlap
    return yes_ids, no_ids

def _predict(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item()  if no_ids  else 1e-9
    d = y + n
    return 1 if (y/d if d > 0 else 0.5) > 0.5 else 0

def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(cp, "JPEG"); return img
    except Exception: return None


def compute_caa(variant, layer, vsr_labels):
    print(f"[STEP] Computing {variant} CAA L{layer} on train+dev (n<{TRAIN_END})...", flush=True)
    if variant == "meanpool":
        pos = neg = None; pn = nn = 0
        for vi in range(TRAIN_END):
            p = MEANPOOL_DIR / f"vi_{vi:05d}.pt"
            if not p.exists(): continue
            try: d = torch.load(p, map_location="cpu", weights_only=True)
            except Exception: continue
            if layer not in d: continue
            v = d[layer].float()
            label = int(vsr_labels[vi])
            if label == 1:
                pos = v.clone() if pos is None else pos + v; pn += 1
            else:
                neg = v.clone() if neg is None else neg + v; nn += 1
        if pos is None or neg is None: return None
        vec = pos/pn - neg/nn
        print(f"  meanpool L{layer}: norm={vec.norm():.3f}  pn={pn} nn={nn}", flush=True)
        return vec
    else:  # paired
        acc = None; n = 0
        for vi in range(TRAIN_END):
            p = PAIRED_DIR / f"vi_{vi:05d}.pt"
            if not p.exists(): continue
            try: d = torch.load(p, map_location="cpu", weights_only=True)
            except Exception: continue
            if "yes" not in d or "no" not in d: continue
            if layer not in d["yes"] or layer not in d["no"]: continue
            label = int(vsr_labels[vi])
            diff = (d["yes"][layer].float() - d["no"][layer].float()) if label == 1 else (d["no"][layer].float() - d["yes"][layer].float())
            acc = diff.clone() if acc is None else acc + diff
            n += 1
        if acc is None or n == 0: return None
        vec = acc / n
        print(f"  paired L{layer}: norm={vec.norm():.3f}  n={n}", flush=True)
        return vec


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["meanpool","paired"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()

    device = "cuda:0"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = OUT_ROOT / f"{args.variant}_L{args.layer}.json"

    print(f"[INFO] {args.variant}/L{args.layer}  → {results_path}", flush=True)

    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train","dev","test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label",0)) for vi in range(len(vsr_all))]
    test_vis = list(range(TRAIN_END, len(vsr_all)))
    test_labels = [vsr_labels[vi] for vi in test_vis]
    print(f"[INFO] test: {len(test_vis)} samples", flush=True)

    vec = compute_caa(args.variant, args.layer, vsr_labels)
    if vec is None:
        print("[FATAL] CAA compute failed", flush=True); return
    gc.collect()

    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    # Try to reuse shared baseline from caa_find_working_layer if available
    shared_base = None
    find_results = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_find_working_layer/results.json")
    if find_results.exists():
        try:
            d = json.load(open(find_results))
            if "base" in d: shared_base = d["base"]["acc"]
        except Exception: pass

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    if "base" not in all_results:
        if shared_base is not None:
            all_results["base"] = {"acc": shared_base, "n": 2195, "source": "caa_find_working_layer"}
            print(f"[BASE] reused from caa_find_working_layer: {shared_base:.2f}%", flush=True)
        else:
            print(f"\n[BASELINE] pt-448 full test (n={len(test_vis)})...", flush=True)
            bc = bt = 0
            for vi, lbl in zip(test_vis, test_labels):
                ex = vsr_all[vi]; img = _load_image(ex)
                if img is None: continue
                prompt = _build_vsr_prompt(str(ex.get("caption","")))
                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                    with torch.no_grad():
                        out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    pred = _predict(out.logits[0,-1,:], yes_ids, no_ids)
                    bt += 1; bc += int(pred == lbl)
                except Exception: continue
            all_results["base"] = {"acc": bc/max(bt,1)*100, "n": bt}
            print(f"[BASE] {all_results['base']['acc']:.2f}% (n={bt})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    base_acc = all_results["base"]["acc"]

    # α-sweep
    sv_norm = vec / vec.norm().clamp(min=1e-8)
    img_end_r = [0]
    for alpha in ALPHAS:
        akey = str(alpha)
        if akey in all_results and all_results[akey].get("n",0) > 0:
            r = all_results[akey]
            print(f"  [SKIP α={alpha}] acc={r['acc']:.2f}%  Δ={r['delta']:+.2f}%", flush=True); continue
        sv_gpu = (sv_norm * alpha).to(next(model.parameters()).dtype).to(device)
        def make_hook(sv_=sv_gpu):
            def hook_fn(module, input, output):
                ie = img_end_r[0]
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[0, ie:] = hidden[0, ie:] + sv_.unsqueeze(0)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook_fn
        correct = total = 0
        for vi, lbl in zip(test_vis, test_labels):
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            prompt = _build_vsr_prompt(str(ex.get("caption","")))
            hh = None
            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                hh = model.model.language_model.layers[args.layer].register_forward_hook(make_hook())
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                hh.remove(); hh = None
                pred = _predict(out.logits[0,-1,:], yes_ids, no_ids)
                total += 1; correct += int(pred == lbl)
            except Exception as e:
                if hh is not None:
                    try: hh.remove()
                    except Exception: pass
                if total < 3: print(f"  [WARN] vi={vi}: {e}", flush=True)
        if total == 0: continue
        acc = correct/total*100; delta = acc - base_acc
        all_results[akey] = {"acc": acc, "delta": delta, "n": total}
        print(f"  [α={alpha}] acc={acc:.2f}%  Δ={delta:+.2f}%  ({correct}/{total})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    print(f"\n=== {args.variant}/L{args.layer} done ===", flush=True)
    best_α = max([k for k in all_results if k != "base"], key=lambda k: all_results[k].get("delta",-999), default=None)
    if best_α: print(f"Best: α={best_α}  acc={all_results[best_α]['acc']:.2f}%  Δ={all_results[best_α]['delta']:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
