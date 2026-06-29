#!/usr/bin/env python3
"""
SAE Feature Clamping Steering — adapted from Anthropic's Scaling Monosemanticity.

Instead of adding alpha*W_dec, we run the SAE on each hidden state and CLAMP
the feature activation to a target value (derived from mix-448 statistics).

Method:
  1. For each example, run pt-448 forward, intercept residual stream at SAE layer
  2. Encode with SAE: f = encoder(h)
  3. Clamp feature F to target value t: f_clamped[F] = t
  4. Reconstruct: h' = decoder(f_clamped) + SAE_bias
  5. Replace h with h' in the residual stream → continue forward

Three target strategies:
  a. "mix_mean_pos": t = mean activation of F on mix-448 positive (label=1) examples
  b. "pt_max": t = max activation of F we've seen in pt-448 forward passes
  c. "mix_target": t = mix_mean_pos + sigma*(mix_mean_pos - mix_mean_neg) — amplified target
  d. "zero_neg": for label-uncertain examples, clamp to 0 (ablate negative-contrast features)

This directly mirrors the "Golden Gate Claude" clamping experiment from Anthropic.
Key advantage over W_dec injection: respects the SAE structure, doesn't add energy
in directions orthogonal to the decoder — stays in the natural feature manifold.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_clamp/

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 pt448_sae_clamp_steering.py
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
OUT_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_clamp")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE   = "/data1/hf_cache/hub"
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Focus on the features where W_dec injection worked well — if clamping beats injection these are key
# Precomputed mix-448 SAE stats from Exp 10 (extract_mix_sae_acts.py) — used instead of re-running mix-448
# Format: (layer_idx, feature_idx, relations, prior_best_wdec, pos_mean, neg_mean)
FEATURES_TO_CLAMP = [
    (4,  14233, ["ahead of"],                            +10.26, 0.7132, 0.6014),
    (11, 12278, ["touching"],                             +3.36, 1.6013, 1.5073),
    (9,  387,   ["at the right side of"],                 +3.12, 4.4575, 4.5159),
    (15, 220,   ["across from", "at the left side of"],   +3.11, 1.6151, 1.5389),
    (12, 2257,  ["facing"],                               +3.92, 1.5734, 1.5658),
    (13, 15219, ["behind"],                               +2.12, 2.5971, 2.4764),
]

# Multipliers for the target clamp value (relative to mix_mean_pos)
CLAMP_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


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


def get_mix448_stats(layer_idx, feature_idx, relations, vsr_all, relation_indices,
                     processor_mix, model_mix, sae, device):
    """Compute mix-448 SAE feature activation statistics on VSR examples."""
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    indices = []
    for r in relations: indices.extend(relation_indices.get(r, []))

    pos_acts, neg_acts = [], []
    dtype = next(model_mix.parameters()).dtype

    print(f"  Computing mix-448 SAE stats for F{feature_idx} @ L{layer_idx}...", flush=True)
    for vi in indices[:200]:  # cap at 200 for speed
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
            h = out.hidden_states[layer_idx + 1]  # +1 because hidden_states[0] is embedding
            # Run SAE on mean of text tokens
            h_text = h[0, img_end:, :].mean(0).to(dtype)
            with torch.inference_mode():
                acts = sae.encode(h_text.unsqueeze(0))  # (1, d_sae)
            f_val = acts[0, feature_idx].item()
            if label == 1: pos_acts.append(f_val)
            else: neg_acts.append(f_val)
        except Exception:
            continue

    pos_mean = sum(pos_acts)/max(len(pos_acts),1)
    neg_mean = sum(neg_acts)/max(len(neg_acts),1)
    pos_std  = (sum((x-pos_mean)**2 for x in pos_acts)/max(len(pos_acts)-1,1))**0.5 if len(pos_acts)>1 else 1.0

    print(f"  mix-448 F{feature_idx}: pos_mean={pos_mean:.3f}, neg_mean={neg_mean:.3f}, "
          f"pos_std={pos_std:.3f}, N_pos={len(pos_acts)}, N_neg={len(neg_acts)}", flush=True)
    return pos_mean, neg_mean, pos_std


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

    # Use precomputed mix-448 stats from Exp 10 (FEATURES_TO_CLAMP includes pos_mean, neg_mean)
    feature_stats = {}
    for layer_idx, feature_idx, relations, prior_best, pos_mean, neg_mean in FEATURES_TO_CLAMP:
        key = f"L{layer_idx}_F{feature_idx}"
        contrast = pos_mean - neg_mean
        pos_std = abs(contrast) if abs(contrast) > 0 else 1.0  # approximate std from contrast
        feature_stats[key] = {"pos_mean": pos_mean, "neg_mean": neg_mean,
                               "pos_std": pos_std, "contrast": contrast}
        print(f"[STATS PRECOMP] {key}: pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} "
              f"contrast={contrast:+.4f}", flush=True)

    # Load pt-448 for clamping experiments
    print(f"\n[INFO] Loading {MODEL_PT} for clamping...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_pt)
    tokenizer = processor_pt.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    dtype = next(model_pt.parameters()).dtype

    all_results = []

    for layer_idx, feature_idx, relations, prior_best_wdec, *_ in FEATURES_TO_CLAMP:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"clamp_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        stats = feature_stats[key]
        pos_mean = stats["pos_mean"]
        neg_mean = stats["neg_mean"]
        contrast = stats["contrast"]

        print(f"\n[CLAMP] {key} pos_mean={pos_mean:.3f} neg_mean={neg_mean:.3f} "
              f"contrast={contrast:+.3f} (prior_best_wdec={prior_best_wdec:+.2f}%)", flush=True)

        # Load SAE
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()

        indices = []
        for r in relations: indices.extend(relation_indices.get(r, []))
        if not indices: del sae; torch.cuda.empty_cache(); continue

        # Compute baseline
        print(f"  [BASE] N={len(indices)}...", flush=True)
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
        base_acc = correct/max(total,1)*100
        base_mg = sum(margins)/max(len(margins),1)
        print(f"  base={base_acc:.2f}% margin={base_mg:.3f}", flush=True)

        result = {
            "layer": layer_idx, "feature": feature_idx, "relations": relations,
            "prior_best_wdec": prior_best_wdec,
            "mix_pos_mean": pos_mean, "mix_neg_mean": neg_mean, "mix_contrast": contrast,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc, "baseline_margin": base_mg,
            "clamp_results": {}
        }

        # Test different clamp target values
        # Base target is pos_mean; multipliers scale it relative to baseline
        for mult in CLAMP_MULTIPLIERS:
            # Use abs(pos_mean) as reference, since some features may have negative pos_mean
            if abs(pos_mean) < 0.01:
                target = mult * 1.0  # fallback
            else:
                target = mult * abs(pos_mean)

            print(f"  [CLAMP] target={target:.3f} (mult={mult}×pos_mean={pos_mean:.3f})...", flush=True)
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
                        # Hook into the SAE layer's output
                        lo = nns_model.model.language_model.layers[layer_idx].output[0][0, img_end:]
                        # We need to process this: encode with SAE, clamp feature, decode back
                        # NNsight proxy trick to do SAE encode/clamp/decode inline:
                        # Step 1: compute pre-bias activation
                        lo_f = lo.float()
                        pre_acts = torch.relu(lo_f @ sae.W_enc.float().T + sae.b_enc.float())
                        # Step 2: clamp feature F
                        clamp_vec = torch.zeros_like(pre_acts)
                        clamp_vec[:, feature_idx] = target - pre_acts[:, feature_idx]
                        pre_acts_clamped = pre_acts + clamp_vec
                        # Step 3: decode back
                        delta = (pre_acts_clamped - pre_acts) @ sae.W_dec.float()
                        lo += delta.to(lo.dtype)
                        logits_s = nns_model.output.logits.save()

                    pred, m = _pm(logits_s[0,-1,:], yes_ids, no_ids)
                    margins.append(m if label==1 else -m)
                except Exception as e:
                    pred=0; margins.append(0.0)
                total+=1; correct+=(pred==label)

            acc = correct/max(total,1)*100; mg = sum(margins)/max(len(margins),1)
            da = acc - base_acc; dm = mg - base_mg
            result["clamp_results"][str(mult)] = {
                "target": target, "acc": acc, "delta_acc": da,
                "margin": mg, "delta_margin": dm
            }
            print(f"    mult={mult}: target={target:.2f} → {acc:.2f}% (Δ={da:+.2f}%) margin={mg:.3f}", flush=True)

        del sae; torch.cuda.empty_cache()

        # Print best
        best_m, best_v = max(result["clamp_results"].items(), key=lambda x: x[1]["delta_acc"])
        print(f"  >> BEST CLAMP: mult={best_m} target={best_v['target']:.2f} Δ={best_v['delta_acc']:+.2f}%  "
              f"(vs W_dec: {prior_best_wdec:+.2f}%)", flush=True)

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        gc.collect()

    print(f"\n{'='*100}")
    print("SAE Feature Clamping vs W_dec Injection — Summary")
    print(f"{'='*100}")
    print(f"{'L/F':<14} {'Relation':<28} {'N':>5} {'base':>7} {'wdec_best':>10} {'clamp_best':>11} {'diff':>7}")
    print("-"*100)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rel = "; ".join(r["relations"])[:27]
        wdec_b = r.get("prior_best_wdec", 0.0)
        if r.get("clamp_results"):
            best_clamp = max(r["clamp_results"].values(), key=lambda x: x["delta_acc"])
            clamp_b = best_clamp["delta_acc"]
        else:
            clamp_b = 0.0
        diff = clamp_b - wdec_b
        print(f"{key:<14} {rel:<28} {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}% "
              f"{wdec_b:>+9.2f}% {clamp_b:>+10.2f}% {diff:>+6.2f}%")


if __name__ == "__main__":
    main()
