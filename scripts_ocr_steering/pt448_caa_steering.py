#!/usr/bin/env python3
"""
Contrastive Activation Addition (CAA) steering for pt-448.

Implements the CAA approach (Rimsky et al. 2023, arXiv 2312.06681):
  v_steer = mean(h_L | label=1) - mean(h_L | label=0)

using mix-448 as the SOURCE model to extract contrastive vectors,
then injecting those vectors into pt-448 as the TARGET model.

This is the ACTIVATION STEERING approach vs the prior FEATURE INJECTION approach.
Key difference:
  - Feature injection (W_dec): uses SAE dictionary atom — monosemantic, but ignores scale
  - CAA: uses actual hidden-state difference — full circuit info, natural scale, may be noisy

Four extraction variants:
  1. "last": extract at the last text token (the "Answer:" aggregation point)
  2. "mean_text": mean over all text tokens (img_end onward)
  3. "sae_layer": extract at the SAE feature's native layer only
  4. "proj_sae": project CAA onto the W_dec[F] direction (FGAA-style)

Phase 1: run mix-448 to extract and save CAA vectors per layer per variant
Phase 2: run pt-448 to inject CAA vectors and measure VSR accuracy

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 pt448_caa_steering.py
"""

import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_MIX  = "google/paligemma2-3b-mix-448"
MODEL_PT   = "google/paligemma2-3b-pt-448"
N_LAYERS   = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering")
CAA_DIR    = OUT_DIR / "caa_vectors"
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE   = "/data1/hf_cache/hub"
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Top-10 spatial features with best individual injection results for comparison
TOP10_FEATURES = [
    (4,  14233, ["ahead of"],                          "sae_only_down", 4.0,  +10.26),
    (12, 2257,  ["facing"],                            "all_ml",       50.0,  +3.92),
    (11, 12278, ["touching"],                          "single",       25.0,  +3.36),
    (9,  387,   ["at the right side of"],              "decay_fwd_ra",  2.0,  +3.12),
    (15, 220,   ["across from", "at the left side of"],"sae_only_up",   2.0,  +3.11),
    (9,  7540,  ["consists of"],                       "single",       10.0,  +2.86),
    (14, 10561, ["close to"],                          "all_ml",        2.0,  +2.15),
    (13, 15219, ["behind"],                            "downstream_ml",30.0,  +2.12),
    (6,  7539,  ["left of", "right of"],               "topK_ml",      20.0,  +1.24),
    (11, 9639,  ["in", "inside", "on"],                "answer",       10.0,  +0.73),
]

# Alpha sweep for CAA injection (CAA vectors have natural scale from mix-448, so different range)
ALPHA_RANGE = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

DECAY_ML = 0.7


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


def phase1_extract_caa_vectors(processor_mix, model_mix, vsr_all, relation_indices, device):
    """Phase 1: extract CAA vectors from mix-448 for all features, all layers."""
    dtype = next(model_mix.parameters()).dtype

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions, initialize_jumprelu_sae

    CAA_DIR.mkdir(parents=True, exist_ok=True)

    for layer_idx, feature_idx, relations, *_ in TOP10_FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        caa_path = CAA_DIR / f"caa_{key}.pt"
        if caa_path.exists():
            print(f"[SKIP] CAA {key}", flush=True)
            continue

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue

        print(f"\n[CAA EXTRACT] {key} {relations} N={len(indices)}...", flush=True)

        # Load SAE W_dec for projection variant
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        wdec = sae.W_dec[feature_idx].detach().to(dtype).to(device)
        wdec = wdec / wdec.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        # Collect hidden states per layer: pos_accum[l] = sum, pos_count
        pos_accum = {l: torch.zeros(model_mix.config.text_config.hidden_size, dtype=torch.float32, device="cpu")
                     for l in range(N_LAYERS)}
        neg_accum = {l: torch.zeros(model_mix.config.text_config.hidden_size, dtype=torch.float32, device="cpu")
                     for l in range(N_LAYERS)}
        pos_count = {l: 0 for l in range(N_LAYERS)}
        neg_count = {l: 0 for l in range(N_LAYERS)}

        n_skipped = 0
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: n_skipped += 1; continue
            label = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                     processor_mix, model_mix, device=device)
                _, img_end = get_image_token_positions(iids)
                seq_len = iids.shape[1]
                last_text_pos = seq_len - 1

                # Use output_hidden_states=True — returns all layer residual stream states
                with torch.inference_mode():
                    out = model_mix(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                    output_hidden_states=True, use_cache=False)

                # hidden_states[0]=embed, [1..N_LAYERS] = after each transformer block
                for l in range(N_LAYERS):
                    h_l = out.hidden_states[l + 1]  # (1, T, d)
                    h_last = h_l[0, last_text_pos, :].float().cpu()
                    if label == 1:
                        pos_accum[l] += h_last; pos_count[l] += 1
                    else:
                        neg_accum[l] += h_last; neg_count[l] += 1

                del out
                torch.cuda.empty_cache()
            except Exception as e:
                n_skipped += 1
                continue

        print(f"  Processed: pos={pos_count[0]}, neg={neg_count[0]}, skipped={n_skipped}", flush=True)

        # Compute CAA vectors for each layer
        caa_data = {}
        for l in range(N_LAYERS):
            pc = max(pos_count[l], 1); nc = max(neg_count[l], 1)
            v_pos = pos_accum[l] / pc
            v_neg = neg_accum[l] / nc
            v_caa = (v_pos - v_neg).to(dtype).to(device)

            # Norm and cosine to W_dec
            v_norm = v_caa.norm().item()
            cos_to_wdec = (v_caa / max(v_norm, 1e-8) @ wdec).item() if v_norm > 1e-6 else 0.0

            # Projection onto W_dec direction (FGAA-style)
            proj_coef = (v_caa @ wdec).item()
            v_proj = proj_coef * wdec  # component along W_dec

            # Orthogonal component (what CAA has beyond W_dec)
            v_orth = v_caa - v_proj

            caa_data[l] = {
                "v_caa_raw": v_caa.cpu(),           # full CAA vector
                "v_caa_norm": v_caa / max(v_norm, 1e-8),  # normalized
                "v_proj": v_proj.cpu(),              # CAA projected onto W_dec
                "v_orth": v_orth.cpu(),              # CAA orthogonal to W_dec
                "v_wdec": wdec.cpu(),                # the W_dec itself
                "norm": v_norm,
                "cos_to_wdec": cos_to_wdec,
                "proj_coef": proj_coef,
                "pos_count": pos_count[l],
                "neg_count": neg_count[l],
            }

            if l == layer_idx:
                print(f"  L{l} (SAE layer): norm={v_norm:.3f}, cos_to_wdec={cos_to_wdec:.3f}, "
                      f"proj_coef={proj_coef:.3f}", flush=True)

        torch.save({
            "layer_idx": layer_idx,
            "feature_idx": feature_idx,
            "relations": relations,
            "caa_data": caa_data,
        }, caa_path)
        print(f"  Saved: {caa_path}", flush=True)

        del wdec; gc.collect(); torch.cuda.empty_cache()

    print("\n[PHASE 1 DONE] All CAA vectors extracted.", flush=True)


def phase2_inject_caa(processor_pt, model_pt, vsr_all, relation_indices, device):
    """Phase 2: inject CAA vectors into pt-448 and measure accuracy."""
    from nnsight import NNsight
    nns_pt = NNsight(model_pt)
    tokenizer = processor_pt.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    dtype = next(model_pt.parameters()).dtype

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    all_results = []

    for layer_idx, feature_idx, relations, best_strat, best_alpha, best_wdec_delta in TOP10_FEATURES:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"caa_result_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        caa_path = CAA_DIR / f"caa_{key}.pt"
        if not caa_path.exists():
            print(f"[WAIT] {key} — no CAA vectors found", flush=True)
            continue

        caa_saved = torch.load(caa_path)
        caa_data = caa_saved["caa_data"]

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: continue

        # Compute baseline
        print(f"\n[BASE] {key} N={len(indices)}...", flush=True)
        correct = total = 0; margins = []
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            try:
                from utils import process_vlm_inputs
                iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                     processor_pt, model_pt, device=device)
                with torch.inference_mode():
                    out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, m = _pm(out.logits[0,-1,:], yes_ids, no_ids)
                margins.append(m if label==1 else -m)
            except Exception: pred=0; margins.append(0.0)
            total+=1; correct+=(pred==label)
        base_acc = correct/max(total,1)*100
        base_mg = sum(margins)/max(len(margins),1)
        print(f"  base={base_acc:.2f}% margin={base_mg:.3f}", flush=True)

        # Print CAA alignment stats at SAE layer
        cd = caa_data[layer_idx]
        print(f"  SAE layer L{layer_idx}: |v_caa|={cd['norm']:.3f}, "
              f"cos(caa,wdec)={cd['cos_to_wdec']:.3f}, proj_coef={cd['proj_coef']:.3f}", flush=True)

        result = {
            "layer": layer_idx, "feature": feature_idx, "relations": relations,
            "best_wdec_delta": best_wdec_delta, "best_wdec_strat": best_strat, "best_wdec_alpha": best_alpha,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "caa_norm_at_sae_layer": cd["norm"],
            "caa_cos_to_wdec_at_sae_layer": cd["cos_to_wdec"],
            "caa_proj_coef_at_sae_layer": cd["proj_coef"],
            "strategies": {}
        }

        # Strategy 1: inject v_caa_norm at SAE layer only (baseline CAA single-layer)
        # Strategy 2: inject v_caa_norm at all layers with 0.7 decay (CAA all-layer)
        # Strategy 3: inject v_caa_norm at SAE layer downstream (sae_only_down)
        # Strategy 4: inject v_proj (SAE-projected CAA) at all layers (FGAA-style)
        # Strategy 5: inject v_wdec (plain W_dec) at all layers — direct comparison

        strategies = [
            ("caa_single",      "single", "v_caa_norm"),
            ("caa_all_ml",      "all_ml", "v_caa_norm"),
            ("caa_sae_down",    "sae_only_down", "v_caa_norm"),
            ("caa_proj_all",    "all_ml", "v_proj"),  # FGAA-style
            ("wdec_all_ml",     "all_ml", "v_wdec"),  # W_dec reference
        ]

        for strat_name, injection_pattern, vec_key in strategies:
            print(f"\n  [STRAT] {key} {strat_name}", flush=True)
            strat_res = {"alphas": {}}

            for alpha in ALPHA_RANGE:
                print(f"    [α={alpha:+g}]...", flush=True)
                correct = total = 0; margins = []

                # Pre-compute the injection vector and layer weights
                if injection_pattern == "single":
                    lw = {layer_idx: 1.0}
                elif injection_pattern == "all_ml":
                    lw = {l: DECAY_ML ** abs(l - layer_idx) for l in range(N_LAYERS)}
                elif injection_pattern == "sae_only_down":
                    lw = {l: 1.0 for l in range(layer_idx, N_LAYERS)}
                else:
                    lw = {layer_idx: 1.0}

                # Build per-layer vectors (use the appropriate key from each layer's caa_data)
                layer_vecs = {}
                for l, w in lw.items():
                    if w < 1e-8: continue
                    if l in caa_data:
                        raw_vec = caa_data[l][vec_key]
                        if isinstance(raw_vec, torch.Tensor):
                            layer_vecs[l] = (raw_vec.to(dtype).to(device), w)
                    else:
                        # fallback to SAE layer vector
                        raw_vec = caa_data[layer_idx][vec_key]
                        layer_vecs[l] = (raw_vec.to(dtype).to(device), w)

                for vi in indices:
                    ex = vsr_all[vi]; img = _load_image(ex)
                    if img is None: continue
                    label = int(ex.get("label", 0))
                    try:
                        from utils import process_vlm_inputs, get_image_token_positions
                        iids, attn, pv = process_vlm_inputs(img, _build_vsr_prompt(str(ex.get("caption",""))),
                                                             processor_pt, nns_pt._module, device=device)
                        _, img_end = get_image_token_positions(iids)
                        with nns_pt.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                            for l, (fv, w) in layer_vecs.items():
                                fv_col = fv.unsqueeze(1)
                                lo = nns_pt.model.language_model.layers[l].output[0][0, img_end:]
                                ones = (lo @ fv_col) * 0.0 + 1.0
                                lo += (alpha * w) * ones * fv
                            logits_s = nns_pt.output.logits.save()
                        pred, m = _pm(logits_s[0,-1,:], yes_ids, no_ids)
                        margins.append(m if label==1 else -m)
                    except Exception: pred=0; margins.append(0.0)
                    total+=1; correct+=(pred==label)

                acc = correct/max(total,1)*100; mg = sum(margins)/max(len(margins),1)
                da = acc - base_acc; dm = mg - base_mg
                strat_res["alphas"][str(alpha)] = {"acc":acc,"delta_acc":da,"margin":mg,"delta_margin":dm}
                print(f"      {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f} (Δ={dm:+.3f})", flush=True)

                # Clean up layer_vecs to avoid memory accumulation
                for l in list(layer_vecs.keys()):
                    del layer_vecs[l]
                torch.cuda.empty_cache()

            best_a = max(strat_res["alphas"].items(), key=lambda x: x[1]["delta_acc"])
            strat_res["best_alpha"] = best_a[0]
            strat_res["best_delta_acc"] = best_a[1]["delta_acc"]
            result["strategies"][strat_name] = strat_res
            print(f"  >> {strat_name}: best Δ={best_a[1]['delta_acc']:+.2f}% @ α={best_a[0]}", flush=True)

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        gc.collect(); torch.cuda.empty_cache()

    # Print final comparison table
    print(f"\n{'='*120}")
    print("CAA Steering vs W_dec Injection — Comparison")
    print(f"{'='*120}")
    header = (f"{'L/F':<14} {'Relation':<25} {'N':>5} {'base':>7} "
              f"{'wdec_best':>10} {'caa_single':>10} {'caa_all':>9} {'caa_down':>9} {'caa_proj':>9}")
    print(header); print("-"*120)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rel = "; ".join(r["relations"])[:24]
        wdec_b = r.get("best_wdec_delta", 0.0)
        def _bst(sname):
            s = r.get("strategies", {}).get(sname, {})
            return s.get("best_delta_acc", 0.0)
        print(f"{key:<14} {rel:<25} {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}% "
              f"{wdec_b:>+9.2f}% {_bst('caa_single'):>+9.2f}% "
              f"{_bst('caa_all_ml'):>+8.2f}% {_bst('caa_sae_down'):>+8.2f}% "
              f"{_bst('caa_proj_all'):>+8.2f}%")


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

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

    # ---- PHASE 1: Extract CAA vectors from mix-448 ----
    all_done = all((CAA_DIR / f"caa_L{l}_F{f}.pt").exists()
                   for l, f, *_ in TOP10_FEATURES)
    if not all_done:
        print(f"\n{'='*80}", flush=True)
        print(f"[PHASE 1] Extracting CAA vectors from {MODEL_MIX}...", flush=True)
        print(f"{'='*80}", flush=True)
        processor_mix = AutoProcessor.from_pretrained(MODEL_MIX)
        model_mix = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_MIX, torch_dtype=torch.bfloat16).to(device).eval()
        phase1_extract_caa_vectors(processor_mix, model_mix, vsr_all, relation_indices, device)
        del model_mix; torch.cuda.empty_cache(); gc.collect()
        del processor_mix
    else:
        print("[PHASE 1 SKIP] All CAA vectors already extracted.", flush=True)

    # ---- PHASE 2: Inject into pt-448 ----
    print(f"\n{'='*80}", flush=True)
    print(f"[PHASE 2] Injecting into {MODEL_PT}...", flush=True)
    print(f"{'='*80}", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, torch_dtype=torch.bfloat16).to(device).eval()
    phase2_inject_caa(processor_pt, model_pt, vsr_all, relation_indices, device)


if __name__ == "__main__":
    main()
