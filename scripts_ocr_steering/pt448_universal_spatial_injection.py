#!/usr/bin/env python3
"""
Universal spatial injection: inject the top spatial feature (L4/F14233 ahead of,
best individual result) into pt-448 on ALL VSR examples and measure overall VSR acc.
Also test L12/F2257 facing (best per N) and L11/F12278 touching.

This answers: does spatial feature injection help VSR OVERALL, or only for the
specific relation the feature was selected for?

CUDA_VISIBLE_DEVICES=7 python3 pt448_universal_spatial_injection.py
"""
import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image
from collections import defaultdict
warnings.filterwarnings("ignore", message=".*PaliGemma.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_NAME="google/paligemma2-3b-pt-448"; N_LAYERS=26
CHECKPOINT_DIR=Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR=Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_universal_inject")
IMAGE_CACHE=Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE="/data1/hf_cache/hub"; VSR_DATASET="cambridgeltl/vsr_random"
os.environ.update({"HF_DATASETS_CACHE":"/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache",
                    "HF_HOME":"/data1/hf_cache",
                    "HF_TOKEN":os.environ.get("HF_TOKEN","hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")})

# Features to test universally: (layer, feat, strategy, alpha, best_delta_on_subset)
UNIVERSAL_TESTS = [
    (4,  14233, "sae_only_down",  4.0,  +10.26, "ahead_of"),
    (12, 2257,  "all_ml",        50.0,   +3.92, "facing"),
    (11, 12278, "single",        20.0,   +3.20, "touching"),
]

DECAY_ML=0.7; DECAY_RA=0.85

def _prompt(s): return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"
def _yesno(tok):
    y,n=set(),set()
    for t in[" Yes","Yes"," yes","YES"]:
        toks=tok.encode(t,add_special_tokens=False)
        if toks: y.add(toks[0])
    for t in[" No","No"," no","NO"]:
        toks=tok.encode(t,add_special_tokens=False)
        if toks: n.add(toks[0])
    ov=y&n;y-=ov;n-=ov; return y,n
def _pm(logits,y,n):
    probs=torch.softmax(logits.float(),dim=-1)
    yp=probs[list(y)].sum().item() if y else 1e-9
    np_=probs[list(n)].sum().item() if n else 1e-9
    d=yp+np_; p=max(yp/d if d>0 else 0.5,1e-7)
    return (1 if p>0.5 else 0),math.log(p/max(1-p,1e-7))
def _img(ex):
    url=ex.get("image_link","")
    if not url.startswith("http"): return None
    h=hashlib.md5(url.encode()).hexdigest(); cp=IMAGE_CACHE/f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r=requests.get(url,timeout=10);r.raise_for_status()
        img=Image.open(BytesIO(r.content)).convert("RGB");img.save(cp,"JPEG");return img
    except: return None
def _lw(strategy, sae):
    if strategy=="single": return {sae:1.0}
    if strategy=="sae_only_down": return {l:1.0 for l in range(sae,N_LAYERS)}
    if strategy=="decay_fwd_ra": return {l:DECAY_RA**max(l-sae,0) for l in range(N_LAYERS)}
    if strategy=="all_ml": return {l:DECAY_ML**abs(l-sae) for l in range(N_LAYERS)}
    return {sae:1.0}

def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0,str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    device="cuda:0"; OUT_DIR.mkdir(parents=True,exist_ok=True)
    print(f"[INFO] Loading {MODEL_NAME}...",flush=True)
    processor=AutoProcessor.from_pretrained(MODEL_NAME)
    model=PaliGemmaForConditionalGeneration.from_pretrained(MODEL_NAME,torch_dtype=torch.bfloat16).to(device).eval()
    nns=NNsight(model); tokenizer=processor.tokenizer; yes_ids,no_ids=_yesno(tokenizer)
    model_dtype=next(model.parameters()).dtype
    vsr_all=concatenate_datasets([load_dataset(VSR_DATASET,data_files={"train":"train.jsonl","dev":"dev.jsonl","test":"test.jsonl"},split=s) for s in["train","dev","test"]])
    all_indices=list(range(len(vsr_all)))
    # Sample 2000 random for overall baseline (full run too slow)
    import random; random.seed(42)
    sample_idx=random.sample(all_indices,min(2000,len(all_indices)))
    print(f"[BASE] Running baseline on {len(sample_idx)} VSR examples...",flush=True)
    correct=total=0; margins=[]
    for vi in sample_idx:
        ex=vsr_all[vi]; img=_img(ex)
        if img is None: continue
        label=int(ex.get("label",0))
        try:
            iids,attn,pv=process_vlm_inputs(img,_prompt(str(ex.get("caption",""))),processor,model,device=device)
            with torch.inference_mode(): out=model(input_ids=iids,attention_mask=attn,pixel_values=pv,use_cache=False)
            pred,m=_pm(out.logits[0,-1,:],yes_ids,no_ids); margins.append(m if label==1 else -m)
        except: pred=0; margins.append(0.0)
        total+=1; correct+=(pred==label)
    base_acc=correct/max(total,1)*100; base_mg=sum(margins)/max(len(margins),1)
    print(f"[BASE] Overall VSR: {base_acc:.2f}% margin={base_mg:.3f}",flush=True)
    results={"baseline_acc":base_acc,"baseline_margin":base_mg,"n_sample":len(sample_idx),"tests":{}}
    for layer_idx,feat_idx,strategy,alpha,prior_best,name in UNIVERSAL_TESTS:
        rp=OUT_DIR/f"univ_{name}.json"
        if rp.exists():
            print(f"[SKIP] {name}",flush=True)
            with open(rp) as f: results["tests"][name]=json.load(f); continue
        ckpt=CHECKPOINT_DIR/f"text-only_layer_{layer_idx}.pt"
        sae=initialize_jumprelu_sae(layer_idx,checkpoint_path=str(ckpt),device=device,cache_dir=HF_CACHE); sae.eval()
        fv=sae.W_dec[feat_idx].detach().to(model_dtype).to(device); fv=fv/fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()
        lw=_lw(strategy,layer_idx); fv_col=fv.unsqueeze(1)
        print(f"[INJECT] {name} L{layer_idx}/F{feat_idx} {strategy} α={alpha} on ALL {len(sample_idx)} examples...",flush=True)
        correct=total=0; margins=[]
        for vi in sample_idx:
            ex=vsr_all[vi]; img=_img(ex)
            if img is None: continue
            label=int(ex.get("label",0))
            try:
                iids,attn,pv=process_vlm_inputs(img,_prompt(str(ex.get("caption",""))),processor,nns._module,device=device)
                _,img_end=get_image_token_positions(iids)
                with nns.trace(input_ids=iids,attention_mask=attn,pixel_values=pv):
                    for l,w in lw.items():
                        lo=nns.model.language_model.layers[l].output[0][0,img_end:]
                        ones=(lo@fv_col)*0.0+1.0; lo+=(alpha*w)*ones*fv
                    logits_s=nns.output.logits.save()
                pred,m=_pm(logits_s[0,-1,:],yes_ids,no_ids); margins.append(m if label==1 else -m)
            except: pred=0; margins.append(0.0)
            total+=1; correct+=(pred==label)
        acc=correct/max(total,1)*100; mg=sum(margins)/max(len(margins),1)
        da=acc-base_acc; dm=mg-base_mg
        print(f"  {name}: {acc:.2f}% (Δ={da:+.2f}%, prior_on_subset={prior_best:+.2f}%)",flush=True)
        r={"name":name,"layer":layer_idx,"feature":feat_idx,"strategy":strategy,"alpha":alpha,
           "prior_subset_delta":prior_best,"overall_acc":acc,"delta_acc":da,"margin":mg,"delta_margin":dm}
        with open(rp,"w") as f: json.dump(r,f,indent=2)
        results["tests"][name]=r; torch.cuda.empty_cache()
    with open(OUT_DIR/"universal_summary.json","w") as f: json.dump(results,f,indent=2)
    print("\n=== UNIVERSAL INJECTION SUMMARY (N=2000 random VSR) ===")
    print(f"  Baseline: {base_acc:.2f}%")
    for name,r in results["tests"].items():
        print(f"  {name}: Δ={r.get('delta_acc',0):+.2f}% (subset best was {r.get('prior_subset_delta',0):+.2f}%)")

if __name__=="__main__": main()
