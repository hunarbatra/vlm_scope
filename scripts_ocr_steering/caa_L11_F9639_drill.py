#!/usr/bin/env python3
"""Drill into L11_F9639 with wider γ sweep: γ ∈ {2, 5, 7, 15, 20, 30} on top of BACKBONE α=1."""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

PT_MODEL = "google/paligemma2-3b-pt-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_L11_F9639_drill")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
TRAIN_END = 8777
LAYERS = [4, 6, 9, 11, 12, 13, 14, 15]
FEATURE_KEY = "L11_F9639"
FEATURE_LAYER = 11
FEATURE_IDX = 9639
GAMMAS = [2.0, 5.0, 7.0, 15.0, 20.0, 30.0]

def _build_vsr_prompt(s): return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"
def _get_yes_no_ids(tok):
    y,n=set(),set()
    for t in [" Yes","Yes"," yes","YES"]:
        tt=tok.encode(t,add_special_tokens=False)
        if tt: y.add(tt[0])
    for t in [" No","No"," no","NO"]:
        tt=tok.encode(t,add_special_tokens=False)
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
    h=hashlib.md5(url.encode()).hexdigest(); cp=IMAGE_CACHE/f"{h}.jpg"
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
    results_path=OUT_DIR/"results.json"
    vsr_all = concatenate_datasets([load_dataset("cambridgeltl/vsr_random", data_files={s:f"{s}.jsonl"}, split=s) for s in ["train","dev","test"]])
    vsr_labels = [int(vsr_all[vi].get("label",0)) for vi in range(len(vsr_all))]
    test_vis = list(range(TRAIN_END, len(vsr_all)))
    vsr_labels_test = [vsr_labels[vi] for vi in test_vis]
    caa = {l: compute_caa(l, vsr_labels) for l in LAYERS}
    caa = {l: v/v.norm().clamp(min=1e-8) for l,v in caa.items() if v is not None}
    # Load W_dec for L11_F9639
    wd = torch.load(SAE_CKPT_DIR/f"text-only_layer_{FEATURE_LAYER}.pt", map_location="cpu", weights_only=True)["W_dec"][FEATURE_IDX].float()
    # R(F)∩test
    ad = json.load(open(SAE_ACTS_DIR/f"acts_{FEATURE_KEY}.json"))
    ak = {int(k) for k in ad.get("acts",{}).keys()}
    rF_vis = [v for v in test_vis if v in ak]
    rF_labels = [vsr_labels[v] for v in rF_vis]
    print(f"R(F)∩test: n={len(rF_vis)}", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)
    # base
    bc=bt=0
    for vi, lbl in zip(rF_vis, rF_labels):
        ex=vsr_all[vi]; img=_load_image(ex)
        if img is None: continue
        pt=_build_vsr_prompt(str(ex.get("caption","")))
        try:
            iids,attn,pv=process_vlm_inputs(img,pt,proc,mdl,device=device)
            with torch.no_grad():
                out=mdl(input_ids=iids, attention_mask=attn, pixel_values=pv)
            pred=_predict(out.logits[0,-1,:], yids, nids)
            bt+=1; bc+=int(pred==lbl)
        except Exception: continue
    base=bc/max(bt,1)*100
    print(f"[BASE] R(F): {base:.2f}%", flush=True)
    dtype = next(mdl.parameters()).dtype
    # Backbone vectors on GPU
    backbone_sv = {l: (caa[l] * 1.0).to(dtype).to(device) for l in caa}
    wd_gpu = wd.to(dtype).to(device)
    img_end_r=[0]
    all_results = json.load(open(results_path)) if results_path.exists() else {}
    all_results["base"] = {"acc": base, "n": bt}
    for gamma in GAMMAS:
        gk = f"g{gamma}"
        if gk in all_results and all_results[gk].get("n",0) > 0: continue
        # inject backbone at all layers, with γ·W_dec added at lF only
        inject = []
        for l, sv in backbone_sv.items():
            if l == FEATURE_LAYER:
                inject.append((l, sv + wd_gpu * gamma))
            else:
                inject.append((l, sv))
        def make_hook(sv_):
            def f(m,i,o):
                ie=img_end_r[0]
                h = o[0] if isinstance(o,tuple) else o
                h[0,ie:] = h[0,ie:] + sv_.unsqueeze(0)
                return (h,)+o[1:] if isinstance(o,tuple) else h
            return f
        c=t=0
        for vi, lbl in zip(rF_vis, rF_labels):
            ex=vsr_all[vi]; img=_load_image(ex)
            if img is None: continue
            pt=_build_vsr_prompt(str(ex.get("caption","")))
            hooks=[]
            try:
                iids,attn,pv=process_vlm_inputs(img,pt,proc,mdl,device=device)
                _,img_end_r[0]=get_image_token_positions(iids)
                for l,sv in inject:
                    hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
                with torch.no_grad():
                    out=mdl(input_ids=iids,attention_mask=attn,pixel_values=pv)
                for h in hooks:
                    try: h.remove()
                    except: pass
                pred=_predict(out.logits[0,-1,:], yids, nids)
                t+=1; c+=int(pred==lbl)
            except Exception:
                for h in hooks:
                    try: h.remove()
                    except: pass
        if t==0: continue
        acc=c/t*100; d=acc-base
        all_results[gk] = {"acc":acc, "delta":d, "n":t}
        print(f"  [γ={gamma}] BB+{gamma}·W_dec: {acc:.2f}%  Δ={d:+.2f}%  ({c}/{t})", flush=True)
        with open(results_path,"w") as f: json.dump(all_results,f,indent=2)

if __name__ == "__main__":
    main()
