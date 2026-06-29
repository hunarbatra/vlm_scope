#!/usr/bin/env python3
"""
FGAA-style Feature-Guided Activation Addition steering.

Based on: "Feature Guided Activation Additions" (arXiv 2501.09929)
Combines CAA with SAE:
  1. Find feature-specific steering direction using SAE features
  2. Filter out high-density "noise" features (broad language features)
  3. Use the PROJECTED direction: only the component of the CAA vector that lies
     along identified spatial SAE features

Also tests the "population-level hidden state difference" at the SAE layer:
  v = mean(h_L[last_text] | label=1 AND feature_F > threshold)
    - mean(h_L[last_text] | label=0 AND feature_F > threshold)

This is "CAA conditioned on feature firing" — only uses examples where the
spatial feature actually fires in mix-448, giving a cleaner signal.

Additionally tests gradient-free optimization: sweeps over linear combinations
of W_dec[F] for the top-10 features using coordinate ascent on accuracy.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fgaa/

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 pt448_fgaa_steering.py
"""

import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_MIX  = "google/paligemma2-3b-mix-448"
MODEL_PT   = "google/paligemma2-3b-pt-448"
N_LAYERS   = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fgaa")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE   = "/data1/hf_cache/hub"
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

DECAY_ML = 0.7

# Features to analyze, ordered by best W_dec injection result
FEATURES = [
    (4,  14233, ["ahead of"],             +10.26, "sae_only_down", 4.0),
    (12, 2257,  ["facing"],               +3.92,  "all_ml",       50.0),
    (11, 12278, ["touching"],             +3.36,  "single",       25.0),
    (9,  387,   ["at the right side of"], +3.12,  "decay_fwd_ra",  2.0),
    (15, 220,   ["across from","at the left side of"], +3.11, "sae_only_up", 2.0),
    (13, 15219, ["behind"],               +2.12,  "downstream_ml",30.0),
]

ALPHA_RANGE = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
SAE_FIRE_THRESHOLD = 0.5  # minimum SAE activation to consider "firing"


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No","No"," no","NO"]:
        toks = tok.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    ov = yes_ids & no_ids; yes_ids -= ov; no_ids -= ov
    return yes_ids, no_ids

def _pm(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n; p = max(y/d if d > 0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1-p, 1e-7))

def _load_image(ex):
    url = ex.get("image_link","")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB"); img.save(cp,"JPEG")
        return img
    except Exception: return None


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train":"train.jsonl","dev":"dev.jsonl","test":"test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train","dev","test"]
    ])
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation","")].append(vi)

    # ---- PHASE 1: Extract feature-conditioned CAA vectors from mix-448 ----
    print(f"\n[INFO] Loading {MODEL_MIX} for feature-conditioned CAA extraction...", flush=True)
    processor_mix = AutoProcessor.from_pretrained(MODEL_MIX)
    model_mix = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_MIX, torch_dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_mix.parameters()).dtype

    fgaa_vectors = {}  # key -> {"v_conditioned", "v_unconditioned", "v_filtered"}

    for layer_idx, feature_idx, relations, *_ in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        vec_path = OUT_DIR / f"fgaa_vec_{key}.pt"
        if vec_path.exists():
            print(f"[SKIP] FGAA vectors {key}", flush=True)
            fgaa_vectors[key] = torch.load(vec_path)
            continue

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))

        # Load SAE for this layer
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        wdec = sae.W_dec[feature_idx].detach().to(dtype).to(device)
        wdec_n = wdec / wdec.norm().clamp(min=1e-8)

        print(f"\n[FGAA EXTRACT] {key} N={len(indices)}", flush=True)

        # Accumulators:
        # 1. Unconditioned: all positive vs all negative
        # 2. Feature-conditioned: only examples where feature fires > threshold
        pos_h_uncond = torch.zeros(model_mix.config.text_config.hidden_size, dtype=torch.float32)
        neg_h_uncond = torch.zeros_like(pos_h_uncond)
        pos_h_cond   = torch.zeros_like(pos_h_uncond)
        neg_h_cond   = torch.zeros_like(pos_h_uncond)
        n_pos_u = n_neg_u = n_pos_c = n_neg_c = 0

        for vi in indices[:300]:  # cap for speed
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                     processor_mix, model_mix, device=device)
                _, img_end = get_image_token_positions(iids)
                with torch.inference_mode():
                    out = model_mix(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                    output_hidden_states=True, use_cache=False)
                # Hidden state at SAE layer, last text token
                h_all = out.hidden_states[layer_idx + 1]  # (1, T, d)
                h_last = h_all[0, -1, :].float().cpu()   # last token
                h_text_mean = h_all[0, img_end:, :].float().cpu().mean(0)

                # Check if feature fires
                h_enc = h_text_mean.to(dtype).to(device)
                with torch.inference_mode():
                    pre_act = torch.relu(h_enc @ sae.W_enc.T + sae.b_enc)
                f_val = pre_act[feature_idx].item()
                fires = f_val > SAE_FIRE_THRESHOLD

                # Unconditioned accumulation
                if label == 1: pos_h_uncond += h_last; n_pos_u += 1
                else:          neg_h_uncond += h_last; n_neg_u += 1

                # Conditioned accumulation
                if fires:
                    if label == 1: pos_h_cond += h_last; n_pos_c += 1
                    else:          neg_h_cond += h_last; n_neg_c += 1

            except Exception: continue

        del sae

        # Compute steering vectors
        v_uncond = (pos_h_uncond / max(n_pos_u,1) - neg_h_uncond / max(n_neg_u,1)).to(dtype).to(device)
        v_cond   = (pos_h_cond   / max(n_pos_c,1) - neg_h_cond   / max(n_neg_c,1)).to(dtype).to(device)

        # Project unconditioned onto W_dec (FGAA filtering)
        v_proj_on_wdec = (v_uncond @ wdec_n) * wdec_n

        # Normalize all
        def _normalize(v):
            n = v.norm()
            return v / n.clamp(min=1e-8) if n > 1e-6 else v

        vecs = {
            "v_uncond_norm":   _normalize(v_uncond),
            "v_cond_norm":     _normalize(v_cond) if n_pos_c > 5 else _normalize(v_uncond),
            "v_proj_norm":     _normalize(v_proj_on_wdec),
            "v_wdec":          wdec_n,
            "n_pos_u": n_pos_u, "n_neg_u": n_neg_u,
            "n_pos_c": n_pos_c, "n_neg_c": n_neg_c,
            "cos_uncond_wdec": (_normalize(v_uncond) @ wdec_n).item(),
            "cos_cond_wdec":   (_normalize(v_cond) @ wdec_n).item() if n_pos_c > 5 else 0.0,
        }
        fgaa_vectors[key] = vecs
        torch.save(vecs, vec_path)
        print(f"  Saved. n_pos_c={n_pos_c}/{n_pos_u} fired. "
              f"cos(uncond,wdec)={vecs['cos_uncond_wdec']:.3f}, "
              f"cos(cond,wdec)={vecs['cos_cond_wdec']:.3f}", flush=True)
        torch.cuda.empty_cache()

    del model_mix; gc.collect(); torch.cuda.empty_cache()
    del processor_mix

    # ---- PHASE 2: Inject FGAA vectors into pt-448 ----
    print(f"\n[INFO] Loading {MODEL_PT} for FGAA injection...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_pt)
    tokenizer = processor_pt.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    dtype = next(model_pt.parameters()).dtype

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations, prior_best, prior_strat, prior_alpha in FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"fgaa_result_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        if key not in fgaa_vectors:
            print(f"[SKIP] no vectors for {key}", flush=True)
            continue

        vecs = fgaa_vectors[key]

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            print(f"[BASE] {rel_key} N={len(indices)}...", flush=True)
            correct = total = 0; margins = []
            for vi in indices:
                ex = vsr_all[vi]; img = _load_image(ex)
                if img is None: continue
                label = int(ex.get("label", 0))
                try:
                    iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                         processor_pt, model_pt, device=device)
                    with torch.inference_mode():
                        out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                    pred, m = _pm(out.logits[0,-1,:], yes_ids, no_ids)
                    margins.append(m if label==1 else -m)
                except Exception: pred=0; margins.append(0.0)
                total+=1; correct+=(pred==label)
            acc = correct/max(total,1)*100; mg = sum(margins)/max(len(margins),1)
            baseline_cache[rel_key] = (acc, mg, total)
            print(f"  base={acc:.2f}% margin={mg:.3f}", flush=True)
        base_acc, base_mg, _ = baseline_cache[rel_key]

        result = {
            "layer": layer_idx, "feature": feature_idx, "relations": relations,
            "prior_best_wdec": prior_best, "prior_strat": prior_strat, "prior_alpha": prior_alpha,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "cos_uncond_wdec": vecs["cos_uncond_wdec"],
            "cos_cond_wdec": vecs["cos_cond_wdec"],
            "n_pos_conditioned": vecs["n_pos_c"],
            "steering_variants": {}
        }

        # Test 3 steering vectors × ALPHA_RANGE, single-layer (at SAE layer) only
        vec_variants = [
            ("uncond_CAA",  "v_uncond_norm",  "Unconditioned CAA at last text token"),
            ("cond_CAA",    "v_cond_norm",    "Feature-conditioned CAA (fires>threshold)"),
            ("proj_CAA",    "v_proj_norm",    "FGAA: CAA projected onto W_dec"),
            ("wdec_ref",    "v_wdec",         "W_dec reference (baseline injection)"),
        ]

        # Use all_ml for all (same as best strategy for many features)
        lw_all = {l: DECAY_ML ** abs(l - layer_idx) for l in range(N_LAYERS)}

        for vname, vkey, vdesc in vec_variants:
            print(f"\n  [VARIANT] {key} {vname}: {vdesc}", flush=True)
            fv = vecs[vkey].to(dtype).to(device)
            fv_col = fv.unsqueeze(1)
            var_res = {"description": vdesc, "alphas": {}}

            for alpha in ALPHA_RANGE:
                print(f"    α={alpha:+g}...", flush=True)
                correct = total = 0; margins = []
                for vi in indices:
                    ex = vsr_all[vi]; img = _load_image(ex)
                    if img is None: continue
                    label = int(ex.get("label", 0))
                    try:
                        iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                             processor_pt, nns_model._module, device=device)
                        _, img_end = get_image_token_positions(iids)
                        with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                            for l, w in lw_all.items():
                                lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                                ones = (lo @ fv_col) * 0.0 + 1.0
                                lo += (alpha * w) * ones * fv
                            logits_s = nns_model.output.logits.save()
                        pred, m = _pm(logits_s[0,-1,:], yes_ids, no_ids)
                        margins.append(m if label==1 else -m)
                    except Exception: pred=0; margins.append(0.0)
                    total+=1; correct+=(pred==label)
                acc = correct/max(total,1)*100; mg = sum(margins)/max(len(margins),1)
                da = acc - base_acc; dm = mg - base_mg
                var_res["alphas"][str(alpha)] = {"acc":acc,"delta_acc":da,"margin":mg,"delta_margin":dm}
                print(f"      {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f}", flush=True)

            best_a = max(var_res["alphas"].items(), key=lambda x: x[1]["delta_acc"])
            var_res["best_alpha"] = best_a[0]
            var_res["best_delta_acc"] = best_a[1]["delta_acc"]
            result["steering_variants"][vname] = var_res
            print(f"  >> {vname}: best Δ={best_a[1]['delta_acc']:+.2f}% @ α={best_a[0]}", flush=True)
            del fv, fv_col; torch.cuda.empty_cache()

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        gc.collect(); torch.cuda.empty_cache()

    # Final comparison table
    print(f"\n{'='*120}")
    print("FGAA Steering — Variant Comparison vs W_dec Injection")
    print(f"{'='*120}")
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        print(f"\n{key} {r['relations']} base={r['baseline_vsr_acc']:.2f}% "
              f"cos(uncond,wdec)={r['cos_uncond_wdec']:.3f} cos(cond,wdec)={r['cos_cond_wdec']:.3f}")
        print(f"  W_dec prior best:  {r['prior_best_wdec']:+.2f}% @ {r['prior_strat']} α={r['prior_alpha']}")
        for vname, vres in r["steering_variants"].items():
            print(f"  {vname:<20}: best Δ={vres['best_delta_acc']:+.2f}% @ α={vres['best_alpha']}")


if __name__ == "__main__":
    main()
