#!/usr/bin/env python3
"""
Simple Rimsky CAA on mix-448 using VQA-format prompt (what mix was trained on).

Prompt format: "answer en Is the following statement correct? <caption> Yes or No"
Extraction: append " Yes" / " No" continuation, take h at last text-token (-1) at L13.

Compared to prior `caa_rimsky_simple_mix.py` which used the verbose English format:
    "Is the following statement correct? Answer only with 'Yes' or 'No'.
     Statement: <caption>
     Answer:"

This simpler VQA format matches PaliGemma's instruction tuning and should give
a cleaner Yes/No distribution at baseline, which is necessary for CAA to find
a steerable truth direction.

Runs on-the-fly (no pre-built paired cache since prompt differs from what's cached).
GPU 0, ~1h for 8777 train+dev extraction × 2 + 2195 test × (baseline + 6 multipliers).
"""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL = "google/paligemma2-3b-mix-448"
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_mix_vqa_prompt")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 8777
LAYER = 13
MULTIPLIERS = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]


def _build_vqa_prompt(s):
    """PaliGemma VQA-native format."""
    return f"answer en Is the following statement correct? {s.strip()} Yes or No"

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
    return 1 if (y/(y+nn) if y+nn > 0 else 0.5) > 0.5 else 0

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


def extract_caa(model, processor, vsr_all, vsr_labels, device):
    """On-the-fly label-aware paired extraction at L13 last-token."""
    from utils import process_vlm_inputs
    acc_sum = None; n = 0
    captures = {"h": None}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures["h"] = h[0, -1, :].detach().float().cpu()
    hh = model.model.language_model.layers[LAYER].register_forward_hook(hook_fn)
    try:
        for vi in range(TRAIN_END):
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            caption = str(ex.get("caption", ""))
            label = int(vsr_labels[vi])

            h_yes = h_no = None
            for answer in ["Yes", "No"]:
                prompt = _build_vqa_prompt(caption) + f" {answer}"
                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                    with torch.no_grad():
                        model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    if answer == "Yes": h_yes = captures["h"].clone()
                    else:               h_no  = captures["h"].clone()
                except Exception:
                    pass
            if h_yes is None or h_no is None: continue
            diff = (h_yes - h_no) if label == 1 else (h_no - h_yes)
            acc_sum = diff.clone() if acc_sum is None else acc_sum + diff
            n += 1
            if (vi + 1) % 500 == 0:
                print(f"    extract {vi+1}/{TRAIN_END}  n={n}", flush=True)
    finally:
        hh.remove()
    return acc_sum / n if acc_sum is not None else None, n


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print(f"[INFO] Rimsky CAA on mix-448 with VQA prompt at L{LAYER}", flush=True)
    print(f"[INFO] Prompt: 'answer en Is the following statement correct? <caption> Yes or No'", flush=True)

    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s:f"{s}.jsonl"}, split=s)
        for s in ["train","dev","test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label",0)) for vi in range(len(vsr_all))]
    test_vis = list(range(TRAIN_END, len(vsr_all)))
    test_labels = [vsr_labels[vi] for vi in test_vis]

    print(f"\n[INFO] Loading {MIX_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(MIX_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Baseline
    if "base" not in all_results:
        print(f"\n[BASELINE] mix-448 VQA-prompt full test...", flush=True)
        bc = bt = 0; bias_y = bias_n = 0
        for i, (vi, lbl) in enumerate(zip(test_vis, test_labels)):
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            prompt = _build_vqa_prompt(str(ex.get("caption","")))
            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, proc, mdl, device=device)
                with torch.no_grad():
                    out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv)
                logits = out.logits[0, -1, :]
                p = torch.softmax(logits.float(), dim=-1)
                y_mass = p[list(yids)].sum().item()
                n_mass = p[list(nids)].sum().item()
                if y_mass + n_mass > 0:
                    pred = 1 if y_mass / (y_mass + n_mass) > 0.5 else 0
                else:
                    pred = 1
                bias_y += y_mass; bias_n += n_mass
                bt += 1; bc += int(pred == lbl)
            except Exception: continue
            if (i+1) % 500 == 0:
                print(f"  baseline progress {i+1}/{len(test_vis)}  acc={bc/max(bt,1)*100:.2f}%  avg P(Yes)={bias_y/max(bt,1):.3f}  P(No)={bias_n/max(bt,1):.3f}", flush=True)
        base = bc/max(bt,1)*100
        all_results["base"] = {"acc": base, "n": bt, "avg_P_yes": bias_y/max(bt,1), "avg_P_no": bias_n/max(bt,1)}
        print(f"[BASELINE] {base:.2f}% (n={bt})  avg P(Yes)={bias_y/bt:.3f}  P(No)={bias_n/bt:.3f}", flush=True)
        with open(results_path,"w") as f: json.dump(all_results, f, indent=2)
    else:
        base = all_results["base"]["acc"]
        print(f"[BASELINE] (cached) {base:.2f}%", flush=True)

    # Extract CAA vector
    if "v_caa" not in all_results:
        print(f"\n[EXTRACT] Building CAA vector from train+dev on-the-fly...", flush=True)
        v, n = extract_caa(mdl, proc, vsr_all, vsr_labels, device)
        if v is None:
            print("[FATAL] CAA extract failed"); return
        all_results["v_caa"] = {"norm": v.norm().item(), "n": n}
        torch.save(v, OUT_DIR / "v_caa_L13.pt")
        print(f"  v_CAA L{LAYER}: norm={v.norm():.3f}  n={n}", flush=True)
        with open(results_path,"w") as f: json.dump(all_results, f, indent=2)
    else:
        v = torch.load(OUT_DIR / "v_caa_L13.pt", map_location="cpu", weights_only=True)
        print(f"[EXTRACT] (cached) v_CAA norm={v.norm():.3f}", flush=True)

    # α-sweep
    img_end_r = [0]
    for mult in MULTIPLIERS:
        akey = f"m{mult}"
        if akey in all_results and all_results[akey].get("n",0) > 0:
            r = all_results[akey]
            print(f"  [SKIP m={mult}] Δ={r['delta']:+.2f}%", flush=True); continue
        sv = (v * mult).to(next(mdl.parameters()).dtype).to(device)
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
            pt = _build_vqa_prompt(str(ex.get("caption","")))
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
        acc = c/t*100; d = acc - base
        all_results[akey] = {"acc": acc, "delta": d, "n": t, "mult": mult}
        print(f"  [m={mult:+g}] {acc:.2f}%  Δ={d:+.2f}%  ({c}/{t})", flush=True)
        with open(results_path,"w") as f: json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
