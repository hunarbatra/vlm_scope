#!/usr/bin/env python3
"""
Feature-Conditioned CAA (FC-CAA) steering for pt-448.

The key methodological difference from plain caa_sae_down:

  Plain CAA split: mean(h_l | VSR_label=1) - mean(h_l | VSR_label=0)
  → Uses ground-truth label to split; SAE feature only picks the relation subset.
  → Actual injection direction has cos(caa, wdec) ≈ 0.008 — almost orthogonal to feature.

  FC-CAA split: mean(h_l | F fires high) - mean(h_l | F fires low/silent)
  → Splits by the SAE feature's OWN activation on mix-448 residual stream.
  → Scans ALL VSR samples (not just own-relation), using the feature as a detector.
  → Direction genuinely encodes "what mix-448 looks like when feature F is active."

This properly leverages the monosemantic features found via SAE analysis.

Phase 1: Run mix-448 on ALL VSR samples, collect hidden states + SAE feature activations.
         Split samples into HIGH (top 33%) vs LOW (bottom 33%) firing on feature F.
         FC-CAA[l] = mean(h_l | HIGH) - mean(h_l | LOW), normalized.

Phase 2: Inject FC-CAA vectors into pt-448 at layers [start_layer..25], flat (no decay),
         at a range of alphas. Evaluate on own-relation VSR subset.

Also compare: FC-CAA injected on full VSR (with global α sweep).

Features tested (one per GPU, parallelized):
  This script takes FEATURE_IDX as command-line arg for parallelization.

  GPU2: L4/F14233   (ahead of)
  GPU5: L6/F7539    (left of / right of)
  GPU6: L12/F2257   (facing)
  GPU7: L11/F12278  (touching)

  Additional features (same script, different GPU):
  GPU??: L14/F10561, L15/F220, L9/F387, L9/F7540, L13/F15219, L11/F9639

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 pt448_fc_caa.py --layer 4  --feature 14233
    CUDA_VISIBLE_DEVICES=5 python3 pt448_fc_caa.py --layer 6  --feature 7539
    CUDA_VISIBLE_DEVICES=6 python3 pt448_fc_caa.py --layer 12 --feature 2257
    CUDA_VISIBLE_DEVICES=7 python3 pt448_fc_caa.py --layer 11 --feature 12278

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fc_caa/
"""

import os, sys, json, gc, hashlib, warnings, math, argparse
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_MIX   = "google/paligemma2-3b-mix-448"
MODEL_PT    = "google/paligemma2-3b-pt-448"
N_LAYERS    = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fc_caa")
FC_CAA_DIR  = OUT_DIR / "fc_caa_vectors"
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE    = "/data1/hf_cache/hub"
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# All features with own-relation info and start layer
ALL_FEATURES = {
    (4,  14233): {"relations": ["ahead of"],                            "start": 0},
    (6,  7539):  {"relations": ["left of", "right of"],                 "start": 1},
    (9,  387):   {"relations": ["at the right side of"],                "start": 1},
    (9,  7540):  {"relations": ["consists of"],                         "start": 9},
    (11, 12278): {"relations": ["touching"],                            "start": 5},
    (11, 9639):  {"relations": ["in", "inside", "on"],                  "start": 0},
    (12, 2257):  {"relations": ["facing"],                              "start": 1},
    (13, 15219): {"relations": ["behind"],                              "start": 0},
    (14, 10561): {"relations": ["close to"],                            "start": 0},
    (15, 220):   {"relations": ["across from", "at the left side of"],  "start": 15},
}

# Alpha sweep for injection
ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# Top quantile threshold: use top 33% as HIGH, bottom 33% as LOW
QUANTILE_HIGH = 0.67
QUANTILE_LOW  = 0.33


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
    url = ex.get("image_link", "")
    if not url.startswith("http"): return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists(): return Image.open(cp).convert("RGB")
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB"); img.save(cp, "JPEG")
        return img
    except Exception: return None


def phase1_extract_fc_caa(layer_idx, feature_idx, vsr_all, processor_mix, model_mix,
                           sae, device, dtype):
    """
    Phase 1: Extract FC-CAA vectors from mix-448.

    For every VSR sample:
      1. Run mix-448 forward pass with output_hidden_states=True
      2. Get the residual stream h_sae at the SAE layer
      3. Compute SAE activation: a = sae.encode(h_sae)[feature_idx]
      4. Store (activation, hidden_states_all_layers) for the last text token

    Then split by activation quantile:
      HIGH: top 33% firing samples
      LOW:  bottom 33% firing samples (silenced)

    FC-CAA[l] = mean(h_l | HIGH) - mean(h_l | LOW)
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    key = f"L{layer_idx}_F{feature_idx}"
    fc_caa_path = FC_CAA_DIR / f"fc_caa_{key}.pt"
    if fc_caa_path.exists():
        print(f"[SKIP] FC-CAA already extracted: {key}", flush=True)
        return torch.load(fc_caa_path)

    print(f"\n[PHASE 1] Extracting FC-CAA for {key} on ALL {len(vsr_all)} VSR samples...", flush=True)

    # Collect per-sample: peak activation + hidden state at peak-firing position
    # KEY FIX: use MAX activation over ALL text tokens (not just last token).
    # Spatial features fire at the preposition token (e.g. "inside", "ahead of"),
    # not at the final "Answer:" token — checking only the last token gave 0 activations.
    activations = []   # peak SAE feature activation for each sample (None if image failed)
    hiddens_all = []   # list of dicts: {l: tensor(d,)} at the peak-firing token position

    n_total = len(vsr_all)
    n_skipped = 0

    for vi in range(n_total):
        ex = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            n_skipped += 1
            activations.append(None)
            hiddens_all.append(None)
            continue

        try:
            iids, attn, pv = process_vlm_inputs(
                img, _build_vsr_prompt(str(ex.get("caption", ""))),
                processor_mix, model_mix, device=device)
            _, img_end = get_image_token_positions(iids)

            with torch.inference_mode():
                out = model_mix(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                output_hidden_states=True, use_cache=False)

            # Run SAE over ALL text tokens at the SAE layer, find peak-firing position
            h_text_all = out.hidden_states[layer_idx + 1][0, img_end:, :].float()  # (n_text, d)
            with torch.no_grad():
                acts_all = sae.encode(h_text_all)  # (n_text, d_sae)
                feature_acts_over_text = acts_all[:, feature_idx]  # (n_text,)
                act_val = feature_acts_over_text.max().item()
                peak_local = feature_acts_over_text.argmax().item()
                peak_pos = img_end + peak_local  # absolute sequence position

            # Store hidden state at the peak-firing position for every layer
            h_per_layer = {}
            for l in range(N_LAYERS):
                h_per_layer[l] = out.hidden_states[l + 1][0, peak_pos, :].float().cpu()

            activations.append(act_val)
            hiddens_all.append(h_per_layer)

            del out
            torch.cuda.empty_cache()

        except Exception as e:
            n_skipped += 1
            activations.append(None)
            hiddens_all.append(None)
            if n_skipped <= 3:
                import traceback; traceback.print_exc()

        if (vi + 1) % 1000 == 0:
            print(f"  {vi+1}/{n_total} (skipped={n_skipped})", flush=True)

    # Filter to valid samples
    valid = [(i, act, h) for i, (act, h) in enumerate(zip(activations, hiddens_all))
             if act is not None]
    act_vals = torch.tensor([v[1] for v in valid], dtype=torch.float32)

    print(f"  Valid: {len(valid)}/{n_total}, skipped={n_skipped}", flush=True)
    print(f"  Feature activation stats: min={act_vals.min():.4f}, max={act_vals.max():.4f}, "
          f"mean={act_vals.mean():.4f}, nonzero={( act_vals > 0).sum().item()}", flush=True)

    # Split strategy: for sparse features (>33% zeros), use nonzero vs zero.
    # For dense features, use top-33% vs bottom-33% quantile.
    # This handles JumpReLU sparsity — quantile collapses to 0 when >33% of samples are zero.
    n_nonzero = (act_vals > 0).sum().item()
    frac_nonzero = n_nonzero / max(len(valid), 1)

    if frac_nonzero < QUANTILE_HIGH:
        # Sparse: HIGH = all firing samples, LOW = all silent samples
        print(f"  Sparse feature ({n_nonzero}/{len(valid)} nonzero = {100*frac_nonzero:.1f}%): "
              f"using nonzero vs zero split", flush=True)
        high_samples = [(i, act, h) for (i, act, h) in valid if act > 0]
        low_samples  = [(i, act, h) for (i, act, h) in valid if act == 0.0]
        split_mode = "nonzero_vs_zero"
    else:
        # Dense: quantile split
        q_high = act_vals.quantile(QUANTILE_HIGH).item()
        q_low  = act_vals.quantile(QUANTILE_LOW).item()
        print(f"  Dense feature ({100*frac_nonzero:.1f}% nonzero): quantile split "
              f"HIGH≥{q_high:.4f} LOW≤{q_low:.4f}", flush=True)
        high_samples = [(i, act, h) for (i, act, h) in valid if act >= q_high]
        low_samples  = [(i, act, h) for (i, act, h) in valid if act <= q_low]
        split_mode = "quantile"

    print(f"  HIGH: {len(high_samples)} samples, LOW: {len(low_samples)} samples "
          f"(split={split_mode})", flush=True)

    if n_nonzero == 0:
        print(f"  [SKIP] Feature fires on 0/{len(valid)} VSR samples — no FC-CAA possible.", flush=True)
        # Save a null result and return early
        saved = {"layer_idx": layer_idx, "feature_idx": feature_idx, "n_high": 0, "n_low": 0,
                 "n_valid": len(valid), "n_nonzero": 0, "act_mean": 0.0, "act_max": 0.0,
                 "split_mode": "none", "fc_caa_data": {}, "skipped_no_firing": True}
        torch.save(saved, fc_caa_path)
        print(f"  [SAVED null] {fc_caa_path}", flush=True)
        return saved

    if len(high_samples) < 10 or len(low_samples) < 10:
        print(f"  [WARN] Few samples: HIGH={len(high_samples)}, LOW={len(low_samples)}", flush=True)

    # Compute FC-CAA[l] = mean(h_l | HIGH) - mean(h_l | LOW)
    d = model_mix.config.text_config.hidden_size
    high_accum = {l: torch.zeros(d, dtype=torch.float32) for l in range(N_LAYERS)}
    low_accum  = {l: torch.zeros(d, dtype=torch.float32) for l in range(N_LAYERS)}

    for _, _, h in high_samples:
        for l in range(N_LAYERS):
            high_accum[l] += h[l]
    for _, _, h in low_samples:
        for l in range(N_LAYERS):
            low_accum[l] += h[l]

    # Load W_dec for reference
    ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
    from utils import initialize_jumprelu_sae
    sae_ref = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                       device="cpu", cache_dir=HF_CACHE)
    wdec = sae_ref.W_dec[feature_idx].detach()
    wdec_norm = wdec / wdec.norm().clamp(min=1e-8)
    del sae_ref; gc.collect()

    fc_caa_data = {}
    for l in range(N_LAYERS):
        v_high = high_accum[l] / max(len(high_samples), 1)
        v_low  = low_accum[l]  / max(len(low_samples),  1)
        v_fc   = (v_high - v_low).float().to(device)

        v_norm = v_fc.norm().item()
        cos_wdec = (v_fc / max(v_norm, 1e-8) @ wdec_norm.float().to(device)).item() if v_norm > 1e-6 else 0.0
        v_fc_norm = v_fc / max(v_norm, 1e-8)

        fc_caa_data[l] = {
            "v_fc_caa":      v_fc.cpu(),
            "v_fc_caa_norm": v_fc_norm.cpu(),
            "norm":           v_norm,
            "cos_to_wdec":   cos_wdec,
        }
        if l == layer_idx:
            print(f"  SAE layer L{l}: |fc_caa|={v_norm:.3f}, cos(fc_caa,wdec)={cos_wdec:.3f}", flush=True)

    saved = {
        "layer_idx": layer_idx,
        "feature_idx": feature_idx,
        "n_high": len(high_samples),
        "n_low": len(low_samples),
        "n_valid": len(valid),
        "split_mode": split_mode,
        "act_mean": act_vals.mean().item(),
        "act_max": act_vals.max().item(),
        "n_nonzero": (act_vals > 0).sum().item(),
        "fc_caa_data": fc_caa_data,
    }
    torch.save(saved, fc_caa_path)
    print(f"  [SAVED] {fc_caa_path}", flush=True)
    return saved


def phase2_inject_fc_caa(layer_idx, feature_idx, fc_caa_saved, vsr_all, relation_indices,
                          processor_pt, model_pt, device, dtype):
    """
    Phase 2: Inject FC-CAA into pt-448 and measure accuracy.
    Tests on own-relation subset AND full VSR.
    """
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    key = f"L{layer_idx}_F{feature_idx}"
    result_path = OUT_DIR / f"fc_caa_result_{key}.json"
    if result_path.exists():
        print(f"[SKIP] Results exist: {key}", flush=True)
        with open(result_path) as f: return json.load(f)

    nns_pt = NNsight(model_pt)
    tokenizer = processor_pt.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)

    fc_caa_data = fc_caa_saved["fc_caa_data"]
    if not fc_caa_data:
        print(f"[SKIP Phase 2] fc_caa_data is empty for L{layer_idx}/F{feature_idx} — null feature.", flush=True)
        return None
    feature_info = ALL_FEATURES[(layer_idx, feature_idx)]
    start_layer  = feature_info["start"]
    relations    = feature_info["relations"]

    # Load normalized FC-CAA vectors to device
    layer_vecs = {}
    for l in range(N_LAYERS):
        layer_vecs[l] = fc_caa_data[l]["v_fc_caa_norm"].to(dtype).to(device)

    def eval_subset(indices, label):
        if not indices:
            return {"acc": 0.0, "margin": 0.0, "n": 0, "base_acc": 0.0, "alphas": {}}

        # Baseline
        correct = total = 0; margins = []
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            lbl = int(ex.get("label", 0))
            try:
                iids, attn, pv = process_vlm_inputs(
                    img, _build_vsr_prompt(str(ex.get("caption", ""))),
                    processor_pt, model_pt, device=device)
                with torch.inference_mode():
                    out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                pred, m = _pm(out.logits[0, -1, :], yes_ids, no_ids)
                margins.append(m if lbl == 1 else -m)
            except Exception: pred = 0; margins.append(0.0)
            total += 1; correct += (pred == lbl)
        base_acc = correct / max(total, 1) * 100
        base_mg  = sum(margins) / max(len(margins), 1)
        print(f"  BASE [{label}]: {base_acc:.2f}% N={total}", flush=True)

        # Alpha sweep
        alpha_results = {}
        best_da = -999; best_a = None
        for alpha in ALPHAS:
            correct = total = 0; margins = []
            for vi in indices:
                ex = vsr_all[vi]; img = _load_image(ex)
                if img is None: continue
                lbl = int(ex.get("label", 0))
                try:
                    iids, attn, pv = process_vlm_inputs(
                        img, _build_vsr_prompt(str(ex.get("caption", ""))),
                        processor_pt, model_pt, device=device)
                    _, img_end = get_image_token_positions(iids)
                    with nns_pt.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                        for l in range(start_layer, N_LAYERS):
                            v_l = layer_vecs[l]
                            v_col = v_l.unsqueeze(1)
                            lo = nns_pt.model.language_model.layers[l].output[0][0, img_end:]
                            ones = (lo @ v_col) * 0.0 + 1.0
                            lo += alpha * ones * v_l
                        logits_s = nns_pt.output.logits.save()
                    pred, m = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                    margins.append(m if lbl == 1 else -m)
                except Exception: pred = 0; margins.append(0.0)
                total += 1; correct += (pred == lbl)
            acc = correct / max(total, 1) * 100
            da  = acc - base_acc
            mg  = sum(margins) / max(len(margins), 1)
            marker = " *** BEST ***" if da > best_da else ""
            if da > best_da: best_da = da; best_a = alpha
            print(f"    α={alpha}: {acc:.2f}% (Δ={da:+.2f}%) mg={mg:.3f}{marker}", flush=True)
            alpha_results[str(alpha)] = {"acc": acc, "delta_acc": da, "margin": mg}

        return {"base_acc": base_acc, "base_margin": base_mg, "n": total,
                "alphas": alpha_results, "best_alpha": best_a, "best_delta": best_da}

    print(f"\n[PHASE 2] Injecting FC-CAA for {key}", flush=True)

    # Own-relation subset
    own_indices = []
    for r in relations: own_indices.extend(relation_indices.get(r, []))
    print(f"\n--- Own-relation subset: {relations} N={len(own_indices)} ---", flush=True)
    own_results = eval_subset(own_indices, f"own:{relations}")

    # Full VSR
    full_indices = list(range(len(vsr_all)))
    print(f"\n--- Full VSR N={len(full_indices)} ---", flush=True)
    full_results = eval_subset(full_indices, "full_vsr")

    result = {
        "layer": layer_idx, "feature": feature_idx, "relations": relations,
        "start_layer": start_layer,
        "n_high": fc_caa_saved["n_high"],
        "n_low": fc_caa_saved["n_low"],
        "n_valid": fc_caa_saved["n_valid"],
        "act_mean": fc_caa_saved["act_mean"],
        "act_max": fc_caa_saved["act_max"],
        "n_nonzero": fc_caa_saved["n_nonzero"],
        "cos_wdec_at_sae_layer": fc_caa_data[layer_idx]["cos_to_wdec"],
        "norm_at_sae_layer": fc_caa_data[layer_idx]["norm"],
        "own_relation": own_results,
        "full_vsr": full_results,
    }
    with open(result_path, "w") as f: json.dump(result, f, indent=2)
    print(f"\n[SAVED] {result_path}", flush=True)
    _print_result(result)
    return result


def _print_result(r):
    key = f"L{r['layer']}_F{r['feature']}"
    own = r["own_relation"]
    full = r["full_vsr"]
    print(f"\n{'='*60}")
    print(f"FC-CAA Results: {key} | cos(fc_caa,wdec)={r['cos_wdec_at_sae_layer']:.3f}")
    print(f"  High-firing: {r['n_high']} samples, Low: {r['n_low']}, Nonzero acts: {r['n_nonzero']}/{r['n_valid']}")
    print(f"  Own-relation [{r['relations']}]: base={own.get('base_acc',0):.2f}%  best Δ={own.get('best_delta',0):+.2f}% @ α={own.get('best_alpha','-')}")
    print(f"  Full VSR:                         base={full.get('base_acc',0):.2f}%  best Δ={full.get('best_delta',0):+.2f}% @ α={full.get('best_alpha','-')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",   type=int, required=True,  help="SAE layer index")
    parser.add_argument("--feature", type=int, required=True,  help="SAE feature index")
    args = parser.parse_args()

    layer_idx   = args.layer
    feature_idx = args.feature
    key = (layer_idx, feature_idx)

    if key not in ALL_FEATURES:
        print(f"[ERROR] Unknown feature {key}. Available: {list(ALL_FEATURES.keys())}", flush=True)
        sys.exit(1)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FC_CAA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, initialize_jumprelu_sae

    print(f"[INFO] FC-CAA for L{layer_idx}/F{feature_idx}", flush=True)

    # Load VSR
    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    fc_caa_path = FC_CAA_DIR / f"fc_caa_L{layer_idx}_F{feature_idx}.pt"

    if not fc_caa_path.exists():
        # --- Phase 1: need mix-448 + SAE ---
        print(f"[INFO] Loading mix-448 for FC-CAA extraction...", flush=True)
        processor_mix = AutoProcessor.from_pretrained(MODEL_MIX)
        model_mix = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_MIX, torch_dtype=torch.bfloat16).to(device).eval()
        dtype_mix = next(model_mix.parameters()).dtype

        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        print(f"[INFO] Loading SAE L{layer_idx} from {ckpt}...", flush=True)
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()

        fc_caa_saved = phase1_extract_fc_caa(
            layer_idx, feature_idx, vsr_all, processor_mix, model_mix,
            sae, device, dtype_mix)

        # Free mix-448 + SAE before loading pt-448
        del model_mix, processor_mix, sae
        gc.collect(); torch.cuda.empty_cache()
        print("[INFO] Freed mix-448 + SAE", flush=True)
    else:
        print(f"[INFO] Loading cached FC-CAA vectors from {fc_caa_path}", flush=True)
        fc_caa_saved = torch.load(fc_caa_path)

    # Skip Phase 2 if Phase 1 returned null (feature never fires on VSR)
    if fc_caa_saved.get("skipped_no_firing"):
        print(f"[SKIP Phase 2] Feature L{layer_idx}/F{feature_idx} has 0 nonzero activations — no FC-CAA vector.", flush=True)
        return

    # --- Phase 2: need pt-448 ---
    print(f"[INFO] Loading pt-448 for injection...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, torch_dtype=torch.bfloat16).to(device).eval()
    dtype_pt = next(model_pt.parameters()).dtype

    # Move fc_caa vectors to pt dtype
    for l in fc_caa_saved["fc_caa_data"]:
        fc_caa_saved["fc_caa_data"][l]["v_fc_caa_norm"] = \
            fc_caa_saved["fc_caa_data"][l]["v_fc_caa_norm"].to(dtype_pt)

    phase2_inject_fc_caa(
        layer_idx, feature_idx, fc_caa_saved, vsr_all, relation_indices,
        processor_pt, model_pt, device, dtype_pt)


if __name__ == "__main__":
    main()
