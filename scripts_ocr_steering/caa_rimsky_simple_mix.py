#!/usr/bin/env python3
"""
Simple Rimsky/Tong CAA on mix-448, full VSR test split.

Literal recipe (nrimsky/CAA repo):
  1. For each train sample, run prompt+" Yes" and prompt+" No" forward passes,
     extract h at last (answer-token) position.
  2. For label=1 samples: pos = h_yes, neg = h_no
     For label=0 samples: pos = h_no,  neg = h_yes
     v = mean(pos - neg)  ← RAW VECTOR (no unit normalization)
  3. Inject α · v at single layer L13. Sweep α ∈ {-3, -2, -1, 1, 2, 3}.

Uses cache: /data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken/
"""
import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
import torch, requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL = "google/paligemma2-3b-mix-448"
PAIRED_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken")
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_rimsky_simple_mix")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 8777
LAYER = 13
MULTIPLIERS = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]


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


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print(f"[INFO] Simple Rimsky CAA on mix-448 at L{LAYER}", flush=True)

    vsr_all = concatenate_datasets([
        load_dataset("cambridgeltl/vsr_random", data_files={s:f"{s}.jsonl"}, split=s)
        for s in ["train","dev","test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label",0)) for vi in range(len(vsr_all))]
    test_vis = list(range(TRAIN_END, len(vsr_all)))
    test_labels = [vsr_labels[vi] for vi in test_vis]
    print(f"[INFO] train+dev: {TRAIN_END}, test: {len(test_vis)}", flush=True)

    # ---- Compute v = mean(h_correct - h_wrong) over train+dev ----
    print(f"[STEP] Building Rimsky CAA from paired cache at L{LAYER}...", flush=True)
    acc_sum = None; n = 0
    for vi in range(TRAIN_END):
        p = PAIRED_DIR / f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception: continue
        if "yes" not in d or "no" not in d: continue
        if LAYER not in d["yes"] or LAYER not in d["no"]: continue
        h_yes = d["yes"][LAYER].float()
        h_no  = d["no"][LAYER].float()
        label = vsr_labels[vi]
        if label == 1:
            diff = h_yes - h_no    # Yes is correct
        else:
            diff = h_no - h_yes    # No is correct
        acc_sum = diff.clone() if acc_sum is None else acc_sum + diff
        n += 1
    v = acc_sum / n
    print(f"  v_CAA L{LAYER}: norm={v.norm():.3f}  n={n}  (RAW, not normalized)", flush=True)

    # ---- Load model ----
    print(f"\n[INFO] Loading {MIX_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(MIX_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    yids, nids = _get_yes_no_ids(proc.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # ---- Baseline ----
    if "base" not in all_results:
        print("[BASELINE] mix-448 full VSR test...", flush=True)
        bc = bt = 0
        for vi, lbl in zip(test_vis, test_labels):
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            pt = _build_vsr_prompt(str(ex.get("caption","")))
            try:
                iids, attn, pv = process_vlm_inputs(img, pt, proc, mdl, device=device)
                with torch.no_grad():
                    out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv)
                pred = _predict(out.logits[0,-1,:], yids, nids)
                bt += 1; bc += int(pred == lbl)
            except Exception: continue
        base = bc/max(bt,1)*100
        all_results["base"] = {"acc": base, "n": bt}
        print(f"[BASELINE] mix-448: {base:.2f}% (n={bt})", flush=True)
        with open(results_path,"w") as f: json.dump(all_results, f, indent=2)
    else:
        base = all_results["base"]["acc"]
        print(f"[BASELINE] (cached) {base:.2f}%", flush=True)

    # ---- α-sweep with RAW vector × multiplier ----
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
        acc = c/t*100; d = acc - base
        all_results[akey] = {"acc": acc, "delta": d, "n": t, "mult": mult}
        print(f"  [m={mult:+g}] {acc:.2f}%  Δ={d:+.2f}%  ({c}/{t})", flush=True)
        with open(results_path,"w") as f: json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
