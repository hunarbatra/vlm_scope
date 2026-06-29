#!/usr/bin/env python3
"""
True CAA pt-src → steer mix-448 (cache-based).

Uses cached pt-448 mean-over-text-tokens hidden states (pt_hidden/*.pt) to
compute label-contrastive vectors, then steers mix-448 on each R(F) subset.

This is the reverse cross-model direction: pt hidden states → mix target.
Completing the 2×2 grid:
  mix→mix  (mix_to_mix script)    mix→pt  (mix_to_pt script)
  pt→mix   (this script)          pt→pt   (pt_src script)

Usage:
    CUDA_VISIBLE_DEVICES=5 python3 -B pt448_true_caa_v4_pt_to_mix.py
"""

import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# ─────────────────────── Config ───────────────────────────────
MIX_MODEL    = "google/paligemma2-3b-mix-448"
MIDDLE_LAYER = 13

PT_HIDDEN_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/pt_hidden")
SAE_ACTS_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR       = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_v4_pt_to_mix")
IMAGE_CACHE   = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET   = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

SPATIAL_FEATURES = [
    {"layer": 9,  "feature": 387,   "key": "L9_F387"},
    {"layer": 14, "feature": 10561, "key": "L14_F10561"},
    {"layer": 11, "feature": 12278, "key": "L11_F12278"},
    {"layer": 9,  "feature": 7540,  "key": "L9_F7540"},
    {"layer": 4,  "feature": 14233, "key": "L4_F14233"},
    {"layer": 6,  "feature": 7539,  "key": "L6_F7539"},
    {"layer": 11, "feature": 9639,  "key": "L11_F9639"},
    {"layer": 13, "feature": 15219, "key": "L13_F15219"},
    {"layer": 15, "feature": 220,   "key": "L15_F220"},
    {"layer": 12, "feature": 2257,  "key": "L12_F2257"},
]

# ─────────────────────── Helpers ──────────────────────────────
def _build_vsr_prompt(statement):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\nAnswer:"
    )

def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
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
    return 1 if (y / d if d > 0 else 0.5) > 0.5 else 0

def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h  = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None

def _load_pt_hidden_layer(vi, layer):
    path = PT_HIDDEN_DIR / f"vi_{vi:05d}.pt"
    if not path.exists(): return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d[layer].float() if layer in d else None
    except Exception:
        return None


# ─────────────────────── Vector computation ───────────────────
def compute_vectors(vsr_all):
    print("[STEP 1] Computing True CAA vectors from pt-448 hidden cache...", flush=True)

    print(f"  MIDDLE: L{MIDDLE_LAYER} over all {len(vsr_all)} samples...", flush=True)
    pos_sum = neg_sum = None
    pos_n = neg_n = 0
    for vi in range(len(vsr_all)):
        label = int(vsr_all[vi].get("label", 0))
        v = _load_pt_hidden_layer(vi, MIDDLE_LAYER)
        if v is None: continue
        if label == 1:
            pos_sum = v.clone() if pos_sum is None else pos_sum + v; pos_n += 1
        else:
            neg_sum = v.clone() if neg_sum is None else neg_sum + v; neg_n += 1
        if (vi + 1) % 2000 == 0:
            print(f"    {vi+1}/{len(vsr_all)}...", flush=True)
    v_middle = None
    if pos_n > 0 and neg_n > 0:
        v_middle = pos_sum / pos_n - neg_sum / neg_n
        print(f"  MIDDLE pos={pos_n}, neg={neg_n}, norm={v_middle.norm():.4f}", flush=True)

    v_feat = {}
    for sf in SPATIAL_FEATURES:
        key, layer = sf["key"], sf["layer"]
        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists(): continue
        acts_data = json.load(open(acts_path))
        acts      = acts_data.get("acts", {})
        relations = acts_data.get("relations", [])
        fire_vis  = [int(k) for k, c in acts.items() if c > 0]
        pos_s = neg_s = None; pn = nn = 0
        for vi in fire_vis:
            lbl = int(vsr_all[vi].get("label", 0))
            v = _load_pt_hidden_layer(vi, layer)
            if v is None: continue
            if lbl == 1:
                pos_s = v.clone() if pos_s is None else pos_s + v; pn += 1
            else:
                neg_s = v.clone() if neg_s is None else neg_s + v; nn += 1
        if pn == 0 or nn == 0:
            print(f"  [{key}] skip: pos={pn} neg={nn}", flush=True); continue
        vec = pos_s / pn - neg_s / nn
        v_feat[key] = {"layer": layer, "vec": vec, "pos_n": pn, "neg_n": nn, "relations": relations}
        print(f"  [{key}] L{layer} pos={pn}, neg={nn}, norm={vec.norm():.4f}", flush=True)

    return v_middle, v_feat


# ─────────────────────── Hook-based steering ──────────────────
def run_steer_sweep(cond_tag, steer_vec, steer_layer, rel_vis, rel_labels,
                    base_acc, result_key, all_results, results_path,
                    model, processor, yes_ids, no_ids, device, vsr_all):
    from utils import process_vlm_inputs, get_image_token_positions

    sv_norm   = steer_vec / steer_vec.norm().clamp(min=1e-8)
    img_end_r = [0]

    for alpha in ALPHAS:
        akey = str(alpha)
        if akey in all_results.get(result_key, {}) and \
                all_results[result_key][akey].get("n", 0) > 0:
            r = all_results[result_key][akey]
            print(f"  [SKIP {cond_tag}] α={alpha}: {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
            continue

        sv_gpu = (sv_norm * alpha).to(next(model.parameters()).dtype).to(device)

        def make_hook(sv_=sv_gpu):
            def hook_fn(module, input, output):
                ie = img_end_r[0]
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[0, ie:] = hidden[0, ie:] + sv_.unsqueeze(0)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook_fn

        correct = total = 0
        for vi, label in zip(rel_vis, rel_labels):
            ex  = vsr_all[vi]
            img = _load_image(ex)
            if img is None: continue
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            hook_h = None
            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                hook_h = model.model.language_model.layers[steer_layer].register_forward_hook(make_hook())
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                hook_h.remove(); hook_h = None
                pred    = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                total  += 1
                correct += int(pred == label)
            except Exception as e:
                if hook_h is not None:
                    try: hook_h.remove()
                    except Exception: pass
                if total < 3:
                    print(f"    [WARN] vi={vi}: {e}", flush=True)

        if total == 0: continue
        acc   = correct / total * 100
        delta = acc - base_acc
        all_results.setdefault(result_key, {})[akey] = {"acc": acc, "delta": delta, "n": total}
        print(f"  [{cond_tag}] α={alpha}: {acc:.2f}% Δ={delta:+.2f}% ({correct}/{total})", flush=True)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("True CAA pt-src → steer mix-448 (cache-based)")
    print(f"MIDDLE: L{MIDDLE_LAYER}, all VSR; FEATURE: lF, R(F)∩fire_F")
    print("Cache: pt_hidden (mean-over-text-tokens)")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    v_middle, v_feat = compute_vectors(vsr_all)
    gc.collect()

    print(f"\n[INFO] Loading {MIX_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    for sf in SPATIAL_FEATURES:
        key, layer = sf["key"], sf["layer"]

        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists():
            print(f"\n[{key}] acts missing — skip", flush=True); continue
        acts_data  = json.load(open(acts_path))
        acts       = acts_data.get("acts", {})
        relations  = acts_data.get("relations", [])
        rel_vis    = [int(k) for k in acts.keys()]
        rel_labels = [int(vsr_all[vi].get("label", 0)) for vi in rel_vis]

        base_key = f"{key}_base"
        if base_key not in all_results:
            print(f"\n[{key}] mix-448 baseline on R(F) (n={len(rel_vis)}, rel={relations})...", flush=True)
            bc = bt = 0
            for vi, lbl in zip(rel_vis, rel_labels):
                ex  = vsr_all[vi]
                img = _load_image(ex)
                if img is None: continue
                prompt = _build_vsr_prompt(str(ex.get("caption", "")))
                try:
                    from utils import process_vlm_inputs
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                    with torch.no_grad():
                        out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                    bt += 1; bc += int(pred == lbl)
                except Exception: continue
            base_acc = bc / max(bt, 1) * 100
            all_results[base_key] = {"acc": base_acc, "n": bt, "relations": relations}
            print(f"  [{key}] baseline: {base_acc:.2f}% (n={bt})", flush=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            base_acc = all_results[base_key]["acc"]
            print(f"\n[{key}] baseline (cached): {base_acc:.2f}%  rel={relations}", flush=True)

        if v_middle is not None:
            print(f"  [{key}] MIDDLE steer (L{MIDDLE_LAYER})...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/MIDDLE", v_middle, MIDDLE_LAYER,
                rel_vis, rel_labels, base_acc,
                f"{key}_middle", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )

        if key in v_feat:
            print(f"  [{key}] FEATURE steer (L{layer})...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/FEATURE", v_feat[key]["vec"], layer,
                rel_vis, rel_labels, base_acc,
                f"{key}_feature", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )

        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("True CAA pt-src → mix-448")
    print(f"{'='*70}")
    print(f"  {'Feature':<16} {'N':>5} {'Base':>7}  {'MIDDLE Δ':>10}  {'FEATURE Δ':>10}")
    print("  " + "-"*55)
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        base = all_results.get(f"{key}_base", {})
        if not base: continue
        ba = base["acc"]; n = base["n"]
        def bd(rkey):
            r = all_results.get(rkey, {})
            if not r: return "—"
            return f"{max(v.get('delta',-999) for v in r.values()):+.2f}%"
        print(f"  {key:<16} {n:>5} {ba:>6.2f}%  {bd(key+'_middle'):>10}  {bd(key+'_feature'):>10}")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
