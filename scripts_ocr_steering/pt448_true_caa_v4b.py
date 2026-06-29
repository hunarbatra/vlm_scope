#!/usr/bin/env python3
"""
True CAA v4b — answer-appended token[-2] extraction, mix-448 self-steering.

Exact match to Rimsky et al. 2023 / Meg Tong's implementation:
  - For each sample, append the ground-truth answer (" Yes" if label=1,
    " No" if label=0) to the prompt before the forward pass.
  - Extract hidden state at token index -2 (second-to-last): this is the
    last *prompt* token (the ":" before the answer), after the model has
    integrated the full statement + the answer token context.
  - In Rimsky/Tong: prompt ends "...(A)" or "...(B)", extraction at -2
    is the "(" token. Here: prompt ends "...Answer: Yes", extraction at
    -2 is the ":" token — functionally equivalent.

Source = Target = mix-448. Two conditions: MIDDLE (L13) and FEATURE (lF).

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 -B pt448_true_caa_v4b.py
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

SAE_ACTS_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts")
OUT_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_v4b")
IMAGE_CACHE  = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET  = "cambridgeltl/vsr_random"

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


# ─────────────────────── Last-token extraction ────────────────
def extract_last_token_hidden(model, processor, vsr_all, vis, layers, device,
                              label_map=None):
    """
    Run mix-448 forward on each sample in vis, capture token[-2] hidden state.

    Appends the ground-truth answer (' Yes' if label=1, ' No' if label=0)
    to the prompt so that token[-2] is the ':' after "Answer" — the last
    prompt token after the model has processed the full statement and sees
    the answer. Exact match to Rimsky et al. / Meg Tong token[-2] convention.

    label_map: dict {vi: label} — required to know which answer to append.
    """
    from utils import process_vlm_inputs, get_image_token_positions

    hidden_store = {}

    for idx, vi in enumerate(vis):
        ex    = vsr_all[vi]
        img   = _load_image(ex)
        if img is None: continue
        label = label_map[vi] if (label_map and vi in label_map) else int(ex.get("label", 0))
        # Append ground-truth answer — extraction at -2 gives last prompt token
        answer_str = " Yes" if label == 1 else " No"
        prompt = _build_vsr_prompt(str(ex.get("caption", ""))) + answer_str

        captures = {}
        hooks = []

        def make_hook(l):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captures[l] = h[0, -2, :].detach().float().cpu()  # token -2
            return hook_fn

        for l in layers:
            hooks.append(
                model.model.language_model.layers[l].register_forward_hook(make_hook(l))
            )

        try:
            iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
            with torch.no_grad():
                model(input_ids=iids, attention_mask=attn, pixel_values=pv)
            hidden_store[vi] = {l: captures[l] for l in layers if l in captures}
        except Exception as e:
            if idx < 5:
                print(f"  [WARN] vi={vi}: {e}", flush=True)
        finally:
            for h in hooks:
                try: h.remove()
                except Exception: pass
            captures.clear()

        if (idx + 1) % 500 == 0:
            print(f"  extracted {idx+1}/{len(vis)}...", flush=True)

    return hidden_store


# ─────────────────────── Vector computation ───────────────────
def compute_vectors(model, processor, vsr_all, device):
    """
    Compute MIDDLE (L13, all VSR) and FEATURE (lF, R(F)∩fire_F) vectors
    using last-token hidden states extracted live from mix-448.
    """
    print("\n[STEP 1] Collecting sample sets...", flush=True)

    # Gather which vis are needed per layer
    layer_to_vis = {}  # layer -> list of (vi, label, condition)

    # MIDDLE: all VSR at layer 13
    all_vis = list(range(len(vsr_all)))
    layer_to_vis.setdefault(MIDDLE_LAYER, []).extend(
        [(vi, int(vsr_all[vi].get("label", 0)), "middle") for vi in all_vis]
    )

    # FEATURE: per feature, R(F) ∩ fire_F at lF
    feature_vis = {}
    for sf in SPATIAL_FEATURES:
        key, layer = sf["key"], sf["layer"]
        acts_path = SAE_ACTS_DIR / f"acts_{key}.json"
        if not acts_path.exists(): continue
        acts_data = json.load(open(acts_path))
        acts      = acts_data.get("acts", {})
        relations = acts_data.get("relations", [])
        fire_vi   = {int(k) for k, c in acts.items() if c > 0}
        samples   = [(vi, int(vsr_all[vi].get("label", 0))) for vi in fire_vi]
        feature_vis[key] = {"layer": layer, "samples": samples, "relations": relations}
        layer_to_vis.setdefault(layer, []).extend(
            [(vi, lbl, f"feat_{key}") for vi, lbl in samples]
        )
        print(f"  [{key}] R(F)∩fire={len(fire_vi)}, rel={relations}", flush=True)

    # Deduplicate: per layer, unique vis
    layer_unique_vis = {l: list({vi for vi,_,_ in entries}) for l, entries in layer_to_vis.items()}
    total = sum(len(v) for v in layer_unique_vis.values())
    print(f"  Total unique (layer,vi) extractions needed: {total}", flush=True)

    # Build label map for all vis across all layers
    all_needed_vis = set()
    for vis in layer_unique_vis.values():
        all_needed_vis.update(vis)
    label_map = {vi: int(vsr_all[vi].get("label", 0)) for vi in all_needed_vis}

    # Extract hidden states per layer
    print("\n[STEP 2] Extracting token[-2] hidden states (answer-appended) from mix-448...", flush=True)
    layer_hidden = {}
    for layer, vis in sorted(layer_unique_vis.items()):
        print(f"  Layer {layer}: {len(vis)} samples...", flush=True)
        layer_hidden[layer] = extract_last_token_hidden(
            model, processor, vsr_all, vis, [layer], device, label_map=label_map
        )

    # Build vectors
    print("\n[STEP 3] Computing label-contrastive vectors...", flush=True)

    # MIDDLE vector
    pos_sum = neg_sum = None
    pos_n = neg_n = 0
    for vi, lbl, _ in layer_to_vis.get(MIDDLE_LAYER, []):
        h_map = layer_hidden.get(MIDDLE_LAYER, {}).get(vi)
        if h_map is None or MIDDLE_LAYER not in h_map: continue
        v = h_map[MIDDLE_LAYER]
        if lbl == 1:
            pos_sum = v.clone() if pos_sum is None else pos_sum + v; pos_n += 1
        else:
            neg_sum = v.clone() if neg_sum is None else neg_sum + v; neg_n += 1
    v_middle = None
    if pos_n > 0 and neg_n > 0:
        v_middle = pos_sum / pos_n - neg_sum / neg_n
        print(f"  MIDDLE L{MIDDLE_LAYER}: pos={pos_n}, neg={neg_n}, norm={v_middle.norm():.4f}", flush=True)

    # FEATURE vectors
    v_feat = {}
    for sf in SPATIAL_FEATURES:
        key, layer = sf["key"], sf["layer"]
        if key not in feature_vis: continue
        samples = feature_vis[key]["samples"]
        relations = feature_vis[key]["relations"]
        pos_s = neg_s = None; pn = nn = 0
        for vi, lbl in samples:
            h_map = layer_hidden.get(layer, {}).get(vi)
            if h_map is None or layer not in h_map: continue
            v = h_map[layer]
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
    from utils import process_vlm_inputs, get_image_token_positions
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("True CAA v4b — answer-appended token[-2] — mix-448 self-steering")
    print(f"MIDDLE: L{MIDDLE_LAYER}, all 10,972 VSR samples, token[-2]")
    print(f"FEATURE: lF, R(F)∩fire_F, token[-2] (answer appended)")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    print(f"\n[INFO] Loading {MIX_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    # Compute vectors via live forward passes
    v_middle, v_feat = compute_vectors(model, processor, vsr_all, device)
    gc.collect(); torch.cuda.empty_cache()

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Per-feature evaluation
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

        # Baseline
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

        # MIDDLE condition
        if v_middle is not None:
            print(f"  [{key}] MIDDLE steer (L{MIDDLE_LAYER})...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/MIDDLE", v_middle, MIDDLE_LAYER,
                rel_vis, rel_labels, base_acc,
                f"{key}_middle", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )

        # FEATURE condition
        if key in v_feat:
            print(f"  [{key}] FEATURE steer (L{layer})...", flush=True)
            all_results = run_steer_sweep(
                f"{key}/FEATURE", v_feat[key]["vec"], layer,
                rel_vis, rel_labels, base_acc,
                f"{key}_feature", all_results, results_path,
                model, processor, yes_ids, no_ids, device, vsr_all,
            )

        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("True CAA v4b — answer-appended token[-2] — mix-448 self-steering")
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
