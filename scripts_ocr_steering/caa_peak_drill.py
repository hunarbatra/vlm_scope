#!/usr/bin/env python3
"""Drill into α=0.8..1.3 on all-layers meanpool to find exact peak."""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

PT_MODEL = "google/paligemma2-3b-pt-448"
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_peak_drill")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
TRAIN_END = 8777
LAYERS = [4, 6, 9, 11, 12, 13, 14, 15]
ALPHAS = [0.8, 0.9, 1.1, 1.2, 1.3]

def _build_vsr_prompt(s): return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"
def _get_yes_no_ids(tok):
    y, n = set(), set()
    for t in [" Yes","Yes"," yes","YES"]:
        tt = tok.encode(t, add_special_tokens=False)
        if tt: y.add(tt[0])
    for t in [" No","No"," no","NO"]:
        tt = tok.encode(t, add_special_tokens=False)
        if tt: n.add(tt[0])
    o=y&n; y-=o; n-=o
    return y,n
def _predict(logits, yids, nids):
    p = torch.softmax(logits.float(), dim=-1)
    y = p[list(yids)].sum().item() if yids else 1e-9
    nn = p[list(nids)].sum().item() if nids else 1e-9
    return 1 if (y/(y+nn) if y+nn>0 else 0.5) > 0.5 else 0
def _load_image(ex):
    url=ex.get("image_link","")
    if not url.startswith("http"): return None
    h=hashlib.md5(url.encode()).hexdigest(); cp = IMAGE_CACHE/f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r=requests.get(url,timeout=10); r.raise_for_status()
        img=Image.open(BytesIO(r.content)).convert("RGB"); img.save(cp,"JPEG"); return img
    except Exception: return None

def compute_caa(layer, vsr_labels):
    pos=neg=None; pn=nn=0
    for vi in range(TRAIN_END):
        p=MEANPOOL_DIR/f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try: d=torch.load(p, map_location="cpu", weights_only=True)
        except: continue
        if layer not in d: continue
        v=d[layer].float()
        if int(vsr_labels[vi])==1: pos=v.clone() if pos is None else pos+v; pn+=1
        else: neg=v.clone() if neg is None else neg+v; nn+=1
    return pos/pn - neg/nn if pos is not None else None

def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions
    device="cuda:0"; OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR/"results.json"
    vsr_all = concatenate_datasets([load_dataset("cambridgeltl/vsr_random", data_files={s:f"{s}.jsonl"}, split=s) for s in ["train","dev","test"]])
    vsr_labels = [int(vsr_all[vi].get("label",0)) for vi in range(len(vsr_all))]
    test_vis = list(range(TRAIN_END, len(vsr_all)))
    test_labels = [vsr_labels[vi] for vi in test_vis]
    vecs = {l: (compute_caa(l, vsr_labels)) for l in LAYERS}
    vecs = {l: v/v.norm().clamp(min=1e-8) for l,v in vecs.items() if v is not None}
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)
    base_acc = 53.53
    shared = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_find_working_layer/results.json")
    if shared.exists():
        try: base_acc = json.load(open(shared))["base"]["acc"]
        except: pass
    all_results = json.load(open(results_path)) if results_path.exists() else {}
    all_results["base"] = {"acc": base_acc, "n": 2195}
    img_end_r = [0]
    for alpha in ALPHAS:
        akey = str(alpha)
        if akey in all_results and all_results[akey].get("n",0) > 0: continue
        sv_by_layer = {l: (uv*alpha).to(next(mdl.parameters()).dtype).to(device) for l,uv in vecs.items()}
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
            hooks = []
            try:
                iids, attn, pv = process_vlm_inputs(img, pt, proc, mdl, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                for l, sv in sv_by_layer.items():
                    hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
                with torch.no_grad():
                    out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv)
                for h in hooks:
                    try: h.remove()
                    except: pass
                pred = _predict(out.logits[0,-1,:], yids, nids)
                t += 1; c += int(pred == lbl)
            except Exception:
                for h in hooks:
                    try: h.remove()
                    except: pass
        if t == 0: continue
        acc = c/t*100; d = acc - base_acc
        all_results[akey] = {"acc": acc, "delta": d, "n": t}
        print(f"  [α={alpha}] PEAK: {acc:.2f}%  Δ={d:+.2f}%  ({c}/{t})", flush=True)
        with open(results_path,"w") as f: json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
