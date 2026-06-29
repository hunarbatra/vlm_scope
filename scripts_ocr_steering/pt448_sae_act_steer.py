#!/usr/bin/env python3
"""
SAE-Activation-Based Feature Steering for pt-448.

For each spatial feature F at layer l with relations R(F):
  1. Approximate SAE activation of feature F per sample via dot-product projection
     on saved SAE reconstructions (computed per-token-then-averaged in Phase 1/2):
       act_F_mix(vi) ≈ recon_mix_all26[l][vi] @ W_dec[F]
       act_F_pt(vi)  ≈ recon_pt_all26[l][vi]  @ W_dec[F]
  2. Per-sample injection scalar:
       delta_F(vi) = act_F_mix(vi) - act_F_pt(vi)
  3. Steer pt-448 at layer l via forward hook:
       inject alpha * delta_F(vi) * W_dec[F]
  4. Evaluate on R(F) subset only; report Δ vs baseline.

NOTE: We use SAE recon files (not mean hidden states) because JumpReLU features
fire at specific sparse tokens (prepositions). Mean-pooling the hidden state before
the SAE suppresses the signal. The recon files average post-activation reconstructions
(computed per-token), preserving the sparse firing information.

Two sub-modes (INJECT_MODE env var):
  PER_SAMPLE  inject alpha * delta_F(vi) * W_dec[F]  (adaptive per sample)
  MEAN        inject alpha * mean(delta_F) * W_dec[F] (constant vector)

Usage:
    CUDA_VISIBLE_DEVICES=0 FEATURE_IDX=0 python3 -B pt448_sae_act_steer.py
    CUDA_VISIBLE_DEVICES=1 FEATURE_IDX=1,2 python3 -B pt448_sae_act_steer.py
    FEATURE_IDX=ALL  runs all 8 sequentially

ENV:
    FEATURE_IDX   int, comma-list, or ALL (default: ALL)
    INJECT_MODE   PER_SAMPLE | MEAN | BOTH (default: BOTH)
    ALPHAS        comma-separated floats (default: 0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0)
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
PT_MODEL       = "google/paligemma2-3b-pt-448"
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
# SAE recon files from the all-26-layer run (per-token activations averaged correctly)
MIX_RECON_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta_all26/mix_reconstructions")
PT_RECON_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta_all26/pt_reconstructions")
PT_H_DIR       = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/pt_hidden")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_act_steer")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

_default_alphas = "0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0"
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", _default_alphas).split(",")]

INJECT_MODE = os.environ.get("INJECT_MODE", "BOTH")   # PER_SAMPLE | MEAN | BOTH

FEATURES = [
    {"layer": 4,  "feature": 14233, "relations": ["ahead of", "behind"]},
    {"layer": 6,  "feature": 7539,  "relations": ["left of", "right of", "across from", "alongside", "at the back of", "below", "facing away from"]},
    {"layer": 9,  "feature": 387,   "relations": ["at the right side of", "adjacent to", "far from", "attached to"]},
    {"layer": 9,  "feature": 7540,  "relations": ["on", "next to", "parallel to", "in the middle of", "opposite to", "away from", "consists of"]},
    {"layer": 11, "feature": 12278, "relations": ["touching", "on top of", "surrounding", "under"]},
    {"layer": 12, "feature": 2257,  "relations": ["facing", "beneath", "near", "off", "enclosed by", "inside", "within", "beyond", "at the side of"]},
    {"layer": 14, "feature": 10561, "relations": ["close to", "by", "connected to"]},
    {"layer": 15, "feature": 220,   "relations": ["above", "at the left side of", "beside", "contains", "over", "part of", "right of", "outside", "toward"]},
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
    p_yes = max(y / d if d > 0 else 0.5, 1e-7)
    return 1 if p_yes > 0.5 else 0

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
        img.save(cp, "JPEG"); return img
    except Exception:
        return None

def _recon_act_F(recon, W_dec_F):
    """Approximate feature F activation via dot-product projection on the SAE reconstruction.
    recon: [2304] float32 mean reconstruction over text tokens
    W_dec_F: [2304] float32 unit-normed W_dec column for feature F
    Returns scalar float (proxy for mean feature activation across text positions).
    """
    return float(recon @ W_dec_F)


# ─────────────────────── Per-feature runner ───────────────────
def run_feature(feat_idx, feat, modes, vsr_all, base_preds, model_raw,
                processor, yes_ids, no_ids, model_dtype, device):
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    l    = feat["layer"]
    F    = feat["feature"]
    rels = set(r.strip().lower() for r in feat["relations"])
    tag  = f"L{l}_F{F}"

    # ── Load SAE W_dec for feature F only (no encoder needed) ──
    ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{l}.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    W_dec_F = state["W_dec"][F].float()   # [2304], typically unit-normed
    del state

    W_dec_F_gpu = W_dec_F.to(model_dtype).to(device)
    print(f"\n{'='*70}", flush=True)
    print(f"Feature {feat_idx}: {tag}  |W_dec_F|={W_dec_F.norm():.3f}", flush=True)
    print(f"  Relations: {sorted(rels)}", flush=True)

    # ── Build R(F) subset + precompute act deltas from recon projections ──
    # act_F_mix(vi) ≈ recon_mix[l] @ W_dec_F  (mean-over-text-positions reconstruction)
    # This correctly reflects per-token SAE firings (averaged after activation).
    N = len(vsr_all)
    subset_data = []  # list of (vi, delta_F_scalar)

    for vi in range(N):
        ex  = vsr_all[vi]
        rel = str(ex.get("relation", "")).strip().lower()
        if rel not in rels:
            continue
        mix_p = MIX_RECON_DIR / f"vi_{vi:05d}.pt"
        pt_p  = PT_RECON_DIR  / f"vi_{vi:05d}.pt"
        if not mix_p.exists() or not pt_p.exists():
            continue

        try:
            recon_mix = torch.load(mix_p, map_location="cpu", weights_only=True)
            recon_pt  = torch.load(pt_p,  map_location="cpu", weights_only=True)
        except Exception:
            continue

        if l not in recon_mix or l not in recon_pt:
            continue

        act_mix = _recon_act_F(recon_mix[l].float(), W_dec_F)
        act_pt  = _recon_act_F(recon_pt[l].float(),  W_dec_F)
        delta_F = act_mix - act_pt
        subset_data.append((vi, delta_F, act_mix))

    if not subset_data:
        print(f"  [{tag}] No valid R(F) samples!", flush=True)
        return {}

    subset_vis = [d[0] for d in subset_data]
    deltas     = [d[1] for d in subset_data]
    acts_mix   = [d[2] for d in subset_data]

    n_mix_fires = sum(1 for a in acts_mix if a > 0)

    mean_delta = sum(deltas) / len(deltas)
    print(f"  R(F) subset: {len(subset_vis)} samples  mix_fires={n_mix_fires} ({100*n_mix_fires/len(subset_vis):.1f}%)", flush=True)
    print(f"  mean(act_mix-act_pt)={mean_delta:.4f}  (positive → mix fires more)", flush=True)

    sub_correct = sum(base_preds[str(vi)]["correct"] for vi in subset_vis if str(vi) in base_preds)
    sub_n       = sum(1 for vi in subset_vis if str(vi) in base_preds)
    sub_base    = sub_correct / max(sub_n, 1) * 100
    print(f"  Baseline on R(F): {sub_base:.2f}%  ({sub_correct}/{sub_n})", flush=True)

    vi_to_delta = {d[0]: d[1] for d in subset_data}
    img_end_ref = [0]
    all_results = {}

    for mode in modes:
        out_path = OUT_DIR / f"{tag}_{mode}.json"
        alpha_results = {}
        if out_path.exists():
            with open(out_path) as f:
                saved = json.load(f)
            alpha_results = saved.get("alphas", {})
            if len(alpha_results) >= len(ALPHAS):
                print(f"  [SKIP] {tag} {mode}: all alphas done.", flush=True)
                all_results[mode] = saved
                continue

        print(f"\n  Mode: {mode}", flush=True)

        for alpha in ALPHAS:
            akey = str(alpha)
            if akey in alpha_results:
                r = alpha_results[akey]
                print(f"    [SKIP] α={alpha}: {r['acc']:.2f}%  Δ={r['delta']:+.2f}%", flush=True)
                continue

            print(f"    [α={alpha}] Running {len(subset_vis)} samples...", flush=True)
            correct = total = 0

            for vi in subset_vis:
                delta_scalar = vi_to_delta[vi]

                if mode == "PER_SAMPLE":
                    if delta_scalar == 0.0:
                        # No difference — use base prediction
                        if str(vi) in base_preds:
                            total += 1
                            correct += base_preds[str(vi)]["correct"]
                        continue
                    inject_scale = alpha * delta_scalar
                else:  # MEAN
                    inject_scale = alpha * mean_delta

                ex    = vsr_all[vi]
                img   = _load_image(ex)
                if img is None:
                    continue
                label  = int(ex.get("label", 0))
                prompt = _build_vsr_prompt(str(ex.get("caption", "")))

                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                    _, img_end = get_image_token_positions(iids)
                    img_end_ref[0] = img_end

                    scale_gpu = torch.tensor(inject_scale, dtype=model_dtype, device=device)

                    def make_hook(s=scale_gpu, w=W_dec_F_gpu):
                        def hook_fn(module, input, output):
                            ie = img_end_ref[0]
                            hidden = output[0]
                            hidden[0, ie:] = hidden[0, ie:] + s * w.unsqueeze(0)
                            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                        return hook_fn

                    hook = model_raw.model.language_model.layers[l].register_forward_hook(make_hook())
                    with torch.no_grad():
                        out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    hook.remove()

                    pred   = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                    total += 1
                    correct += int(pred == label)

                except Exception as e:
                    try: hook.remove()
                    except Exception: pass
                    if total < 3:
                        print(f"      [WARN] vi={vi}: {e}", flush=True)
                    continue

            if total == 0:
                print(f"    α={alpha}: no valid samples!", flush=True)
                continue

            acc   = correct / total * 100
            delta = acc - sub_base
            alpha_results[akey] = {"acc": acc, "delta": delta, "n": total}
            print(f"    α={alpha:>6.1f}: {acc:.2f}%  Δ={delta:+.2f}%  ({correct}/{total})", flush=True)

            result = {
                "layer": l, "feature": F, "mode": mode,
                "relations": sorted(rels),
                "base_acc": sub_base, "subset_n": len(subset_vis),
                "mean_delta_act": mean_delta,
                "n_mix_fires": n_mix_fires, "frac_fires": n_mix_fires / max(len(subset_vis), 1),
                "alphas": alpha_results,
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        result = {
            "layer": l, "feature": F, "mode": mode,
            "relations": sorted(rels),
            "base_acc": sub_base, "subset_n": len(subset_vis),
            "mean_delta_act": mean_delta,
            "n_mix_fires": n_mix_fires, "frac_fires": n_mix_fires / max(len(subset_vis), 1),
            "alphas": alpha_results,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results[mode] = result
        print(f"    Saved: {out_path}", flush=True)

    del W_dec_F_gpu
    torch.cuda.empty_cache(); gc.collect()
    return all_results


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    feat_idx_str = os.environ.get("FEATURE_IDX", "ALL")
    feat_indices = list(range(len(FEATURES))) if feat_idx_str.upper() == "ALL" \
                   else [int(x) for x in feat_idx_str.split(",")]

    if INJECT_MODE == "BOTH":
        modes = ["PER_SAMPLE", "MEAN"]
    else:
        modes = [INJECT_MODE]

    print("=" * 70)
    print(f"SAE-Activation Feature Steering  modes={modes}")
    print(f"Features: {feat_indices}  Alphas: {ALPHAS}")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PT_H_DIR / "base_predictions.json") as f:
        base_preds = json.load(f)
    base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
    print(f"[INFO] Global baseline: {base_acc:.2f}% over {len(base_preds)} samples", flush=True)

    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    print(f"[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    all_results = {}
    for fi in feat_indices:
        feat = FEATURES[fi]
        res  = run_feature(fi, feat, modes, vsr_all, base_preds, model_raw,
                           processor, yes_ids, no_ids, model_dtype, device)
        all_results[f"L{feat['layer']}_F{feat['feature']}"] = res

    # ── Summary ──
    print(f"\n{'='*90}", flush=True)
    print(f"SUMMARY — SAE-Act Steering  modes={modes}")
    print(f"{'='*90}", flush=True)
    for mode in modes:
        print(f"\n--- {mode} ---")
        hdr = f"  {'Feature':>10}  {'N':>5}  {'Base':>6}  {'fires%':>7}"
        for a in ALPHAS: hdr += f"  {str(a):>6}"
        print(hdr)
        for fi in feat_indices:
            feat = FEATURES[fi]
            tag  = f"L{feat['layer']}_F{feat['feature']}"
            r    = all_results.get(tag, {}).get(mode, {})
            if not r: continue
            row = f"  {tag:>10}  {r.get('subset_n',0):>5}  {r.get('base_acc',0):>5.1f}%  {100*r.get('frac_fires',0):>6.1f}%"
            for a in ALPHAS:
                ar = r.get("alphas", {}).get(str(a), {})
                row += f"  {ar['delta']:>+5.1f}%" if ar else f"  {'--':>6}"
            print(row)


if __name__ == "__main__":
    main()
