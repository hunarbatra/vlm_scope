#!/usr/bin/env python3
"""
Extended alpha sweep for L11/F12278 "touching" — pushing beyond alpha=20.
Single-layer at L11 gave best +3.20% at alpha=20. Does it plateau or improve further?
Also tests "all_ml" strategy at extended alphas.
CUDA_VISIBLE_DEVICES=5 python3 pt448_touching_extended.py
"""
import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict
import torch, requests
from PIL import Image
warnings.filterwarnings("ignore", message=".*PaliGemma.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_NAME="google/paligemma2-3b-pt-448"; N_LAYERS=26
CHECKPOINT_DIR=Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR=Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_touching_ext")
IMAGE_CACHE=Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE="/data1/hf_cache/hub"; VSR_DATASET="cambridgeltl/vsr_random"
os.environ["HF_DATASETS_CACHE"]="/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]="/data1/hf_cache"
os.environ["HF_TOKEN"]=os.environ.get("HF_TOKEN","hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

LAYER, FEAT, RELS = 11, 12278, ["touching"]
CONFIGS = [
    ("single",   {11: 1.0},  [15.0,20.0,25.0,30.0,35.0,40.0,50.0,70.0,100.0]),
    ("all_ml",   {l: 0.7**abs(l-11) for l in range(26)}, [5.0,10.0,15.0,20.0,25.0,30.0,40.0,50.0]),
    ("sae_down", {l: 1.0 for l in range(11,26)}, [1.0,2.0,3.0,5.0,7.0,10.0,15.0,20.0]),
]

def _prompt(s): return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"
def _yesno(tok):
    y,n=set(),set()
    for t in[" Yes","Yes"," yes","YES"]: toks=tok.encode(t,add_special_tokens=False);(y if toks else set()).add(toks[0]) if toks else None
    for t in[" No","No"," no","NO"]: toks=tok.encode(t,add_special_tokens=False);(n if toks else set()).add(toks[0]) if toks else None
    ov=y&n;y-=ov;n-=ov; return y,n
def _pm(logits,y,n):
    probs=torch.softmax(logits.float(),dim=-1)
    yp=probs[list(y)].sum().item() if y else 1e-9; np_=probs[list(n)].sum().item() if n else 1e-9
    d=yp+np_; p=max(yp/d if d>0 else 0.5,1e-7); return (1 if p>0.5 else 0),math.log(p/max(1-p,1e-7))
def _img(ex):
    url=ex.get("image_link","")
    if not url.startswith("http"): return None
    h=hashlib.md5(url.encode()).hexdigest(); cp=IMAGE_CACHE/f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r=requests.get(url,timeout=10);r.raise_for_status();img=Image.open(BytesIO(r.content)).convert("RGB");img.save(cp,"JPEG");return img
    except: return None

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
    ridx=defaultdict(list)
    for vi in range(len(vsr_all)): ridx[vsr_all[vi].get("relation","")].append(vi)
    indices=ridx.get("touching",[])
    print(f"[BASE] N={len(indices)}...",flush=True)
    correct=total=0; margins=[]
    for vi in indices:
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
    print(f"[BASE] {base_acc:.2f}% margin={base_mg:.3f}",flush=True)
    ckpt=CHECKPOINT_DIR/f"text-only_layer_{LAYER}.pt"
    sae=initialize_jumprelu_sae(LAYER,checkpoint_path=str(ckpt),device=device,cache_dir=HF_CACHE); sae.eval()
    fv=sae.W_dec[FEAT].detach().to(model_dtype).to(device); fv=fv/fv.norm().clamp(min=1e-8)
    del sae; torch.cuda.empty_cache()
    fv_col=fv.unsqueeze(1)
    all_results=[]
    for strat_name,lw,alphas in CONFIGS:
        rp=OUT_DIR/f"touch_ext_{strat_name}.json"
        if rp.exists():
            print(f"[SKIP] {strat_name}",flush=True)
            with open(rp) as f: all_results.append(json.load(f)); continue
        res={"strategy":strat_name,"baseline_vsr_acc":base_acc,"baseline_margin":base_mg,"alphas":{}}
        print(f"\n[STRAT] touching {strat_name}",flush=True)
        for alpha in alphas:
            print(f"  α={alpha:+g}...",flush=True)
            correct=total=0; mgs=[]
            for vi in indices:
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
                    pred,m=_pm(logits_s[0,-1,:],yes_ids,no_ids); mgs.append(m if label==1 else -m)
                except: pred=0; mgs.append(0.0)
                total+=1; correct+=(pred==label)
            acc=correct/max(total,1)*100; mg=sum(mgs)/max(len(mgs),1)
            da=acc-base_acc; dm=mg-base_mg
            res["alphas"][str(alpha)]={"acc":acc,"delta_acc":da,"margin":mg,"delta_margin":dm}
            print(f"    α={alpha:+g}: {acc:.2f}% (Δ={da:+.2f}%) margin_Δ={dm:+.3f}",flush=True)
        with open(rp,"w") as f: json.dump(res,f,indent=2); all_results.append(res)
        torch.cuda.empty_cache()
    print("\n=== TOUCHING EXTENDED SWEEP SUMMARY ===")
    for r in all_results:
        best_a,best_v=max(r["alphas"].items(),key=lambda x:x[1]["delta_acc"])
        print(f"  {r['strategy']}: best Δ={best_v['delta_acc']:+.2f}% @ α={best_a}")

if __name__=="__main__": main()
