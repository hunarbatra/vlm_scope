#!/usr/bin/env python3
"""
Adaptive SAE steering — Options A+C combined.

Option C — SAE-activation-based feature selection:
    Instead of caption keyword parsing, run the mix-448 SAE on each pt-448 sample's
    hidden states at inference time. Measure each feature's activation deficit
    (threshold - activation, clamped to ≥0). Select the feature with the largest
    deficit — the one the model needs most that it's currently failing to activate.
    No text parsing needed; grounded in what the representations say is missing.

Option A — Activation-adaptive alpha:
    Once a feature is selected, scale the injection strength by the deficit:
        alpha_i = base_alpha * clip(threshold_f - act_f_i, 0, max_deficit) / mean_deficit
    If the feature is already firing at threshold, inject little/nothing.
    If it's far below threshold, inject hard. Personalises to each sample.

Why this is mechanistically principled:
    - The SAE threshold IS the model's own "minimum needed" signal.
    - Deficit = how far below the threshold the current representation is.
    - We're nudging the model toward activating the most spatially-relevant feature.

Implementation:
    1. Load mix-448 SAE for each of the 8 feature layers.
    2. At inference: run pt-448 forward with output_hidden_states=True.
    3. At the SAE layer for each feature, compute activation at the peak text token.
    4. Compute deficit for all 8 features. Pick the one with highest deficit.
    5. Inject caa_sae_down vector for that feature, scaled by deficit.

Two modes tested:
    - "deficit_select"     : select by deficit, fixed base_alpha per feature (Option C only)
    - "deficit_select_ada" : select by deficit, alpha scaled by deficit (A+C combined)

Also compares against smart oracle v2 (caption parsing) on the same samples.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_adaptive_steer/
Usage: CUDA_VISIBLE_DEVICES=<gpu> python3 pt448_adaptive_steer.py
"""

import os, sys, json, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_PT      = "google/paligemma2-3b-pt-448"
MODEL_MIX     = "google/paligemma2-3b-mix-448"
N_LAYERS      = 26
CKPT_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
CAA_DIR       = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/caa_vectors")
OUT_DIR       = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_adaptive_steer")
IMAGE_CACHE   = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET   = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

FEATURE_CONFIGS = {
    "L4_F14233":  {"layer": 4,  "feature": 14233, "start": 0,  "alpha": 1.0},
    "L14_F10561": {"layer": 14, "feature": 10561, "start": 0,  "alpha": 2.0},
    "L12_F2257":  {"layer": 12, "feature": 2257,  "start": 1,  "alpha": 1.0},
    "L15_F220":   {"layer": 15, "feature": 220,   "start": 15, "alpha": 0.75},
    "L11_F12278": {"layer": 11, "feature": 12278, "start": 5,  "alpha": 0.5},
    "L9_F387":    {"layer": 9,  "feature": 387,   "start": 1,  "alpha": 0.5},
    "L6_F7539":   {"layer": 6,  "feature": 7539,  "start": 1,  "alpha": 1.5},
    "L9_F7540":   {"layer": 9,  "feature": 7540,  "start": 9,  "alpha": 0.25},
}

# Features grouped by their SAE layer so we only run each SAE once
# L9 has two features (387 and 7540) — run a single SAE pass at layer 9
SAE_LAYERS_NEEDED = sorted(set(cfg["layer"] for cfg in FEATURE_CONFIGS.values()))


def _build_vsr_prompt(s):
    return f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s.strip()}\nAnswer:"

def _get_yes_no_ids(tok):
    yes_ids, no_ids = set(), set()
    for t in [" Yes","Yes"," yes","YES"]: toks = tok.encode(t, add_special_tokens=False); yes_ids.update(toks[:1] if toks else [])
    for t in [" No","No"," no","NO"]:  toks = tok.encode(t, add_special_tokens=False); no_ids.update(toks[:1] if toks else [])
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
    h = __import__("hashlib").md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    if cp.exists():
        try: return Image.open(cp).convert("RGB")
        except: pass
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        cp.parent.mkdir(parents=True, exist_ok=True); img.save(cp)
        return img
    except: return None


def _load_saes(device, dtype):
    """Load one JumpReLU SAE per unique SAE layer. Returns dict {layer: sae}."""
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae
    saes = {}
    for l in SAE_LAYERS_NEEDED:
        ckpt = CKPT_DIR / f"text-only_layer_{l}.pt"
        if not ckpt.exists():
            print(f"  [WARN] SAE checkpoint missing: {ckpt}", flush=True)
            continue
        sae = initialize_jumprelu_sae(l, checkpoint_path=str(ckpt), device=device)
        sae = sae.to(dtype).eval()
        saes[l] = sae
        print(f"  [SAE] Loaded layer {l}", flush=True)
    return saes


def _load_caa_vecs(dtype, device):
    """Load caa_sae_down vectors (same as smart oracle v2)."""
    caa_vectors = {}
    for feat_key, cfg in FEATURE_CONFIGS.items():
        layer = cfg["layer"]
        path = CAA_DIR / f"caa_L{layer}_F{cfg['feature']}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu")
            caa_data = saved.get("caa_data", {})
            vecs = {}
            for l, ld in caa_data.items():
                v = ld.get("v_caa_norm")
                if v is not None:
                    vecs[int(l)] = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)
            if vecs:
                caa_vectors[feat_key] = vecs
                print(f"  [CAA] {feat_key} ({len(vecs)} layers)", flush=True)
    return caa_vectors


def compute_deficits(hidden_states, saes, img_end, dtype, device):
    """
    For each feature in FEATURE_CONFIGS, compute:
        deficit = max(0, threshold_f - max_text_activation_f)

    hidden_states: tuple from output_hidden_states=True, shape (n_layers+1, 1, seq, d)
    Returns dict: feat_key -> deficit (float)
    """
    deficits = {}
    for feat_key, cfg in FEATURE_CONFIGS.items():
        layer = cfg["layer"]
        feature_idx = cfg["feature"]
        sae = saes.get(layer)
        if sae is None:
            deficits[feat_key] = 0.0
            continue

        # Hidden state at SAE layer, all text tokens
        h_text = hidden_states[layer + 1][0, img_end:, :].to(dtype)  # (n_text, d)

        with torch.no_grad():
            # SAE pre-activation: x @ W_enc + b_enc
            pre = h_text @ sae.W_enc + sae.b_enc       # (n_text, d_sae)
            act_f = pre[:, feature_idx]                  # (n_text,)
            max_act = act_f.max().item()
            threshold = sae.threshold[feature_idx].item()

        deficit = max(0.0, threshold - max_act)
        deficits[feat_key] = deficit

    return deficits


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)

    print("[INFO] Loading pt-448...", flush=True)
    processor_pt = AutoProcessor.from_pretrained(MODEL_PT)
    model_pt = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PT, dtype=torch.bfloat16).to(device).eval()
    dtype = next(model_pt.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(processor_pt.tokenizer)
    nns_pt = NNsight(model_pt)

    print("[INFO] Loading SAEs...", flush=True)
    saes = _load_saes(device, dtype)

    print("[INFO] Loading CAA vectors...", flush=True)
    caa_vectors = _load_caa_vecs(dtype, device)

    # Collect mean deficit per feature across all samples for adaptive alpha normalisation.
    # We'll estimate this during the first pass (deficit_select run) and reuse it.
    mean_deficits = {k: 1.0 for k in FEATURE_CONFIGS}  # will be updated after first pass

    for mode in ["deficit_select", "deficit_select_ada"]:
        result_path = OUT_DIR / f"adaptive_steer_{mode}.json"
        if result_path.exists():
            print(f"[SKIP] {result_path} exists", flush=True)
            # Load mean_deficits from saved result for adaptive mode
            if mode == "deficit_select":
                with open(result_path) as f:
                    saved_res = json.load(f)
                mean_deficits = saved_res.get("mean_deficits", mean_deficits)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Mode = {mode}", flush=True)

        correct_base = correct_smart = total = 0
        mb_sum = ms_sum = 0.0
        by_selected = defaultdict(lambda: [0, 0, 0])   # feat_key -> [base_correct, smart_correct, n]
        by_action = defaultdict(lambda: [0, 0, 0])      # "inject"/"no_deficit" -> counts
        deficit_accum = defaultdict(float)
        deficit_count = defaultdict(int)

        for vi in range(N):
            ex = vsr_all[vi]; lbl = int(ex.get("label", 0))
            img = _load_image(ex)
            if img is None: continue

            try:
                iids, attn, pv = process_vlm_inputs(
                    img, _build_vsr_prompt(str(ex.get("caption", ""))),
                    processor_pt, model_pt, device=device)
                _, img_end = get_image_token_positions(iids)

                # Forward pass with hidden states for SAE deficit computation
                with torch.inference_mode():
                    out = model_pt(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                   output_hidden_states=True, use_cache=False)
                pb, mb = _pm(out.logits[0, -1, :], yes_ids, no_ids)
                mb_sum += mb if lbl == 1 else -mb
                correct_base += (pb == lbl)

                # Compute deficits for all 8 features
                deficits = compute_deficits(out.hidden_states, saes, img_end, dtype, device)
                del out

                # Track for mean_deficit normalisation
                for k, dv in deficits.items():
                    deficit_accum[k] += dv
                    deficit_count[k] += 1

                # Select feature with largest deficit
                best_feat = max(deficits, key=lambda k: deficits[k])
                best_deficit = deficits[best_feat]

                do_inject = best_deficit > 0 and best_feat in caa_vectors

                if do_inject:
                    cfg = FEATURE_CONFIGS[best_feat]
                    start = cfg["start"]
                    base_alpha = cfg["alpha"]

                    if mode == "deficit_select_ada":
                        # Scale alpha by deficit relative to mean deficit for this feature
                        md = mean_deficits.get(best_feat, 1.0)
                        scale = min(best_deficit / max(md, 1e-6), 3.0)  # cap at 3× base
                        alpha = base_alpha * scale
                    else:
                        alpha = base_alpha

                    vecs = caa_vectors[best_feat]
                    with nns_pt.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                        for l in range(start, N_LAYERS):
                            if l not in vecs: continue
                            v_l = vecs[l]
                            v_col = v_l.unsqueeze(1)
                            lo = nns_pt.model.language_model.layers[l].output[0][0, img_end:]
                            ones = (lo @ v_col) * 0.0 + 1.0
                            lo += alpha * ones * v_l
                        logits_s = nns_pt.output.logits.save()
                    ps, ms = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                    ms_sum += ms if lbl == 1 else -ms
                    correct_smart += (ps == lbl)
                    action = "inject"
                    by_selected[best_feat][0] += (pb == lbl)
                    by_selected[best_feat][1] += (ps == lbl)
                    by_selected[best_feat][2] += 1
                else:
                    ms_sum += mb if lbl == 1 else -mb
                    correct_smart += (pb == lbl)
                    ps = pb
                    action = "no_deficit"

                d = by_action[action]
                d[0] += (pb == lbl); d[1] += (ps == lbl); d[2] += 1

            except Exception as e:
                print(f"  [ERR] vi={vi}: {e}", flush=True); continue
            total += 1
            if total % 2000 == 0:
                print(f"  {total} base={100*correct_base/total:.2f}% adaptive={100*correct_smart/total:.2f}%", flush=True)

        ba = correct_base / max(total, 1) * 100
        sa = correct_smart / max(total, 1) * 100
        print(f"\n[RESULT {mode}] base={ba:.2f}%  adaptive={sa:.2f}%  Δ={sa-ba:+.2f}%  N={total}", flush=True)
        for act, d in sorted(by_action.items()):
            ab = d[0]/max(d[2],1)*100; as_ = d[1]/max(d[2],1)*100
            print(f"  [{act:14s}] N={d[2]:5d}  base={ab:.2f}%  adaptive={as_:.2f}%  Δ={as_-ab:+.2f}%", flush=True)
        print("  Feature selection breakdown:", flush=True)
        for fk, d in sorted(by_selected.items(), key=lambda x: -x[1][2]):
            ab = d[0]/max(d[2],1)*100; as_ = d[1]/max(d[2],1)*100
            print(f"    {fk:14s} N={d[2]:5d}  base={ab:.2f}%  Δ={as_-ab:+.2f}%", flush=True)

        # Update mean_deficits from this pass
        mean_deficits = {k: deficit_accum[k] / max(deficit_count[k], 1)
                         for k in FEATURE_CONFIGS}
        print("  Mean deficits per feature:", flush=True)
        for k, v in sorted(mean_deficits.items()):
            print(f"    {k}: {v:.4f}", flush=True)

        res = {
            "mode": mode,
            "base_acc": ba, "adaptive_acc": sa, "delta_acc": sa - ba,
            "base_margin": mb_sum/max(total,1), "adaptive_margin": ms_sum/max(total,1),
            "n_total": total,
            "mean_deficits": mean_deficits,
            "by_action": {k: {"correct_base":d[0],"correct_adaptive":d[1],"total":d[2]}
                          for k,d in by_action.items()},
            "by_selected": {k: {"correct_base":d[0],"correct_adaptive":d[1],"total":d[2]}
                            for k,d in by_selected.items()},
        }
        with open(result_path, "w") as f: json.dump(res, f, indent=2)
        print(f"[SAVED] {result_path}", flush=True)

    print("\n[DONE]", flush=True)


if __name__ == "__main__":
    main()
