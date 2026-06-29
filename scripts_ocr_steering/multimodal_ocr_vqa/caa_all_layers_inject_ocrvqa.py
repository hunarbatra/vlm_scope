#!/usr/bin/env python3
"""
Multi-layer CAA injection at ALL 8 cached layers simultaneously.
mix-src → pt-448 full VSR test.

This is the aggressive Rimsky-style "inject at every layer from early to late"
approach that typically gives the biggest gains.

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_all_layers_inject.py [--variant meanpool|paired]
"""
import os, sys, json, gc, hashlib, warnings, argparse
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

PT_MODEL = "google/paligemma2-3b-pt-448"
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
PAIRED_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken")
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_all_layers_inject")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 8777
LAYERS = [4, 6, 9, 11, 12, 13, 14, 15]
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0]


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    y, n = set(), set()
    for t in [" Yes","Yes"," yes","YES"]:
        tt = tok.encode(t, add_special_tokens=False)
        if tt: y.add(tt[0])
    for t in [" No","No"," no","NO"]:
        tt = tok.encode(t, add_special_tokens=False)
        if tt: n.add(tt[0])
    o = y & n; y -= o; n -= o
    return y, n

def _predict(logits, yids, nids):
    p = torch.softmax(logits.float(), dim=-1)
    y = p[list(yids)].sum().item() if yids else 1e-9
    nn = p[list(nids)].sum().item() if nids else 1e-9
    d = y + nn
    return 1 if (y/d if d > 0 else 0.5) > 0.5 else 0

def _load_image(ex):
    url = ex.get("image_link","")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(cp, "JPEG"); return img
    except Exception: return None


def compute_caa(variant, layer, vsr_labels):
    if variant == "meanpool":
        pos = neg = None; pn = nn = 0
        for vi in range(TRAIN_END):
            p = MEANPOOL_DIR / f"vi_{vi:05d}.pt"
            if not p.exists(): continue
            try: d = torch.load(p, map_location="cpu", weights_only=True)
            except: continue
            if layer not in d: continue
            v = d[layer].float()
            if int(vsr_labels[vi]) == 1:
                pos = v.clone() if pos is None else pos+v; pn += 1
            else:
                neg = v.clone() if neg is None else neg+v; nn += 1
        if pos is None or neg is None: return None
        return pos/pn - neg/nn
    else:
        acc = None; n = 0
        for vi in range(TRAIN_END):
            p = PAIRED_DIR / f"vi_{vi:05d}.pt"
            if not p.exists(): continue
            try: d = torch.load(p, map_location="cpu", weights_only=True)
            except: continue
            if "yes" not in d or "no" not in d or layer not in d["yes"]: continue
            label = int(vsr_labels[vi])
            diff = (d["yes"][layer].float() - d["no"][layer].float()) if label == 1 else (d["no"][layer].float() - d["yes"][layer].float())
            acc = diff.clone() if acc is None else acc + diff
            n += 1
        return acc/n if acc is not None else None


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["meanpool","paired"], default="meanpool")
    args = ap.parse_args()

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / f"results_{args.variant}.json"

    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s:f"{s}.jsonl"}, split=s)
        for s in ["train","dev","test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label",0)) for vi in range(len(vsr_all))]
    test_vis = list(range(TRAIN_END, len(vsr_all)))
    test_labels = [vsr_labels[vi] for vi in test_vis]

    print(f"[INFO] {args.variant} CAA at all {len(LAYERS)} cached layers...", flush=True)
    vecs = {}
    for l in LAYERS:
        v = compute_caa(args.variant, l, vsr_labels)
        if v is not None:
            vecs[l] = v
            print(f"  L{l}: norm={v.norm():.3f}", flush=True)
    gc.collect()

    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)

    # Baseline shared
    base_acc = 53.53
    shared = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_find_working_layer/results.json")
    if shared.exists():
        try: base_acc = json.load(open(shared))["base"]["acc"]
        except: pass
    print(f"[BASE] {base_acc:.2f}% (reused)", flush=True)

    all_results = json.load(open(results_path)) if results_path.exists() else {}
    all_results["base"] = {"acc": base_acc, "n": 2195}

    # Unit-norm each vector
    unit_vecs = {l: v/v.norm().clamp(min=1e-8) for l,v in vecs.items()}
    img_end_r = [0]
    for alpha in ALPHAS:
        akey = str(alpha)
        if akey in all_results and all_results[akey].get("n",0) > 0:
            r = all_results[akey]
            print(f"  [SKIP α={alpha}] Δ={r['delta']:+.2f}%", flush=True); continue
        # Prepare per-layer α-scaled vectors
        sv_by_layer = {l: (uv*alpha).to(next(mdl.parameters()).dtype).to(device) for l, uv in unit_vecs.items()}
        handles = []

        def make_hook(sv_):
            def f(m,i,o):
                ie = img_end_r[0]
                h = o[0] if isinstance(o,tuple) else o
                h[0,ie:] = h[0,ie:] + sv_.unsqueeze(0)
                return (h,)+o[1:] if isinstance(o,tuple) else h
            return f

        c = t = 0
        for vi, lbl in zip(test_vis, test_labels):
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            pt = _build_vsr_prompt(str(ex.get("caption","")))
            local_hooks = []
            try:
                iids, attn, pv = process_vlm_inputs(img, pt, proc, mdl, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                for l, sv in sv_by_layer.items():
                    local_hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
                with torch.no_grad():
                    out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv)
                for h in local_hooks:
                    try: h.remove()
                    except: pass
                pred = _predict(out.logits[0,-1,:], yids, nids)
                t += 1; c += int(pred == lbl)
            except Exception:
                for h in local_hooks:
                    try: h.remove()
                    except: pass
        if t == 0: continue
        acc = c/t*100; d = acc - base_acc
        all_results[akey] = {"acc": acc, "delta": d, "n": t}
        print(f"  [α={alpha}] ALL-LAYER: {acc:.2f}%  Δ={d:+.2f}%  ({c}/{t})", flush=True)
        with open(results_path,"w") as f: json.dump(all_results, f, indent=2)

    print(f"\n=== DONE ({args.variant} all layers) ===")
    for a, r in all_results.items():
        if a == "base": continue
        print(f"  α={a}: {r['acc']:.2f}%  Δ={r['delta']:+.2f}%")


if __name__ == "__main__":
    main()
