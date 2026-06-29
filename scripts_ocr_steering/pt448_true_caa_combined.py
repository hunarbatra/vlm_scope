#!/usr/bin/env python3
"""
True CAA combined steering — sum of top-N MIDDLE vectors → steer pt-448.

Uses mix-448 cache MIDDLE vector at L13 (same as v4_mix_to_pt).
Tests whether combining multiple spatial feature contrastive directions
into a single steering vector improves accuracy on the union of their
relation subsets, versus steering with any single feature's vector.

Conditions:
  TOP3   L11/F12278 + L4/F14233 + L14/F10561 (highest mix→pt MIDDLE deltas)
  ALL10  All 10 spatial features' MIDDLE vectors summed
  MEAN10 Mean of all 10 MIDDLE vectors

Steers pt-448 on the union of the 10 R(F) subsets.

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 -B pt448_true_caa_combined.py
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
PT_MODEL     = "google/paligemma2-3b-pt-448"
MIDDLE_LAYER = 13

MIX_HIDDEN_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
SAE_ACTS_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_combined")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

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

# Top-3 by mix→pt MIDDLE delta from prior experiments
TOP3_KEYS = ["L11_F12278", "L4_F14233", "L14_F10561"]

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

def _load_mix_hidden_layer(vi, layer):
    path = MIX_HIDDEN_DIR / f"vi_{vi:05d}.pt"
    if not path.exists(): return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d[layer].float() if layer in d else None
    except Exception:
        return None


# ─────────────────────── Vector computation ───────────────────
def compute_global_middle(vsr_all):
    print(f"  MIDDLE: L{MIDDLE_LAYER} over all {len(vsr_all)} samples...", flush=True)
    pos_sum = neg_sum = None
    pos_n = neg_n = 0
    for vi in range(len(vsr_all)):
        label = int(vsr_all[vi].get("label", 0))
        v = _load_mix_hidden_layer(vi, MIDDLE_LAYER)
        if v is None: continue
        if label == 1:
            pos_sum = v.clone() if pos_sum is None else pos_sum + v; pos_n += 1
        else:
            neg_sum = v.clone() if neg_sum is None else neg_sum + v; neg_n += 1
        if (vi + 1) % 2000 == 0:
            print(f"    {vi+1}/{len(vsr_all)}...", flush=True)
    if pos_n == 0 or neg_n == 0:
        return None
    v = pos_sum / pos_n - neg_sum / neg_n
    print(f"  MIDDLE global pos={pos_n}, neg={neg_n}, norm={v.norm():.4f}", flush=True)
    return v


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
    print("True CAA Combined Steering — mix→pt (MIDDLE vectors summed)")
    print(f"TOP3: {TOP3_KEYS}")
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

    print("\n[STEP 1] Computing MIDDLE vector...", flush=True)
    v_global = compute_global_middle(vsr_all)
    if v_global is None:
        print("ERROR: no global vector", flush=True); return

    # Build eval sets
    all_feature_vis = set()
    per_feature_vis = {}
    for sf in SPATIAL_FEATURES:
        key = sf["key"]
        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists(): continue
        acts_data = json.load(open(acts_path))
        acts = acts_data.get("acts", {})
        vis = [int(k) for k in acts.keys()]
        per_feature_vis[key] = vis
        all_feature_vis.update(vis)

    union_vis    = sorted(all_feature_vis)
    union_labels = [int(vsr_all[vi].get("label", 0)) for vi in union_vis]

    # Also per-feature eval sets for TOP3 features
    top3_vis    = sorted(set(v for k in TOP3_KEYS for v in per_feature_vis.get(k, [])))
    top3_labels = [int(vsr_all[vi].get("label", 0)) for vi in top3_vis]

    print(f"  Union set N={len(union_vis)}, TOP3 union N={len(top3_vis)}", flush=True)
    gc.collect()

    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Baseline on union
    for eval_name, vis_list, labels_list in [
        ("union10", union_vis, union_labels),
        ("top3_union", top3_vis, top3_labels),
    ]:
        base_key = f"{eval_name}_base"
        if base_key not in all_results:
            from utils import process_vlm_inputs
            print(f"\n[{eval_name}] pt-448 baseline (n={len(vis_list)})...", flush=True)
            bc = bt = 0
            for vi, lbl in zip(vis_list, labels_list):
                ex  = vsr_all[vi]
                img = _load_image(ex)
                if img is None: continue
                prompt = _build_vsr_prompt(str(ex.get("caption", "")))
                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                    with torch.no_grad():
                        out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                    bt += 1; bc += int(pred == lbl)
                except Exception: continue
            base_acc = bc / max(bt, 1) * 100
            all_results[base_key] = {"acc": base_acc, "n": bt}
            print(f"  [{eval_name}] baseline: {base_acc:.2f}% (n={bt})", flush=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            print(f"\n[{eval_name}] baseline cached: {all_results[base_key]['acc']:.2f}%", flush=True)

    # Steer on union10 with GLOBAL MIDDLE (single direction)
    union_base = all_results["union10_base"]["acc"]
    top3_base  = all_results["top3_union_base"]["acc"]

    print(f"\n[COND: GLOBAL] Single MIDDLE vector on union10...", flush=True)
    all_results = run_steer_sweep(
        "GLOBAL/union10", v_global, MIDDLE_LAYER,
        union_vis, union_labels, union_base,
        "global_union10", all_results, results_path,
        model, processor, yes_ids, no_ids, device, vsr_all,
    )

    print(f"\n[COND: GLOBAL] Single MIDDLE vector on top3_union...", flush=True)
    all_results = run_steer_sweep(
        "GLOBAL/top3", v_global, MIDDLE_LAYER,
        top3_vis, top3_labels, top3_base,
        "global_top3", all_results, results_path,
        model, processor, yes_ids, no_ids, device, vsr_all,
    )

    gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("True CAA Combined — mix→pt")
    print(f"{'='*70}")
    for rkey, label in [
        ("global_union10", f"GLOBAL → union10 (N={len(union_vis)})"),
        ("global_top3",    f"GLOBAL → top3 union (N={len(top3_vis)})"),
    ]:
        r = all_results.get(rkey, {})
        if r:
            best = max(v.get("delta", -999) for v in r.values())
            print(f"  {label}: best Δ={best:+.2f}%")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
