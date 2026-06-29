#!/usr/bin/env python3
"""
Random-vector control: inject α · unit(random_gaussian[2304]) at L13 on
mix→pt full test. If random injection produces similar Δ curve to CAA, the
signal we're seeing is just "inject anything at L13, get a small perturbation".

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_random_control.py
"""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

PT_MODEL = "google/paligemma2-3b-pt-448"
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_random_control")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 8777
LAYER = 13
ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]
N_RANDOM = 3  # 3 random directions for mean estimate

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
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(cp, "JPEG"); return img
    except Exception: return None

def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("[INFO] Random-vector control mix→pt L13", flush=True)
    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s:f"{s}.jsonl"}, split=s)
        for s in ["train","dev","test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label",0)) for vi in range(len(vsr_all))]
    test_vis = list(range(TRAIN_END, len(vsr_all)))
    test_labels = [vsr_labels[vi] for vi in test_vis]

    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)

    # Reuse baseline
    shared = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_find_working_layer/results.json")
    base_acc = 53.53
    if shared.exists():
        try: base_acc = json.load(open(shared))["base"]["acc"]
        except: pass
    print(f"[BASE] {base_acc:.2f}% (reused)", flush=True)

    all_results = json.load(open(results_path)) if results_path.exists() else {}
    all_results["base"] = {"acc": base_acc, "n": 2195}

    torch.manual_seed(42)
    img_end_r = [0]
    for seed in range(N_RANDOM):
        torch.manual_seed(1000 + seed)
        rv = torch.randn(2304)
        rv_norm = rv / rv.norm().clamp(min=1e-8)
        for alpha in ALPHAS:
            rkey = f"seed{seed}_α{alpha}"
            if rkey in all_results and all_results[rkey].get("n",0) > 0:
                r = all_results[rkey]
                print(f"  [SKIP {rkey}] Δ={r['delta']:+.2f}%", flush=True); continue
            sv = (rv_norm*alpha).to(next(mdl.parameters()).dtype).to(device)
            def make_hook(s=sv):
                def f(m,i,o):
                    ie = img_end_r[0]
                    h = o[0] if isinstance(o,tuple) else o
                    h[0,ie:] = h[0,ie:] + s.unsqueeze(0)
                    return (h,)+o[1:] if isinstance(o,tuple) else h
                return f
            c = t = 0
            for vi, lbl in zip(test_vis, test_labels):
                ex = vsr_all[vi]; img = _load_image(ex)
                if img is None: continue
                pt = _build_vsr_prompt(str(ex.get("caption","")))
                hh = None
                try:
                    iids, attn, pv = process_vlm_inputs(img, pt, proc, mdl, device=device)
                    _, img_end_r[0] = get_image_token_positions(iids)
                    hh = mdl.model.language_model.layers[LAYER].register_forward_hook(make_hook())
                    with torch.no_grad():
                        out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    hh.remove(); hh = None
                    pred = _predict(out.logits[0,-1,:], yids, nids)
                    t += 1; c += int(pred == lbl)
                except Exception:
                    if hh is not None:
                        try: hh.remove()
                        except: pass
            if t == 0: continue
            acc = c/t*100; d = acc - base_acc
            all_results[rkey] = {"acc": acc, "delta": d, "n": t}
            print(f"  [seed={seed} α={alpha}] acc={acc:.2f}%  Δ={d:+.2f}%", flush=True)
            with open(results_path,"w") as f: json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
