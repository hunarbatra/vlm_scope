#!/usr/bin/env python3
"""
Spatial Feature Steering Ablation — comparing 4 vector construction methods.

For each of the 8 spatial features (layer l, feature F, relation subset R(F)):

  CAA-1  Layer-only, all samples:
         v = mean(h_mix[l] - h_pt[l])  over ALL VSR samples
         Inject alpha * norm(v) at layer l for every sample in R(F)

  CAA-2  Layer + relation-subset (= per_relation_steer baseline):
         v = mean(h_mix[l] - h_pt[l])  over R(F) samples only
         Inject alpha * norm(v) at layer l for every sample in R(F)

  SAE-B  Feature-specific recon delta, always:
         delta[vi] = proj(recon_delta[vi][l], W_dec[F]) * W_dec[F]
         where recon_delta[vi][l] = recon_mix[vi][l] - recon_pt[vi][l]
         This is the F-th feature's contribution to the full SAE recon delta.
         Inject alpha * delta[vi] at layer l for every sample in R(F).

  SAE-C  Feature-specific recon delta, fire-conditioned:
         Same as SAE-B but only when proj > 0  (feature contributes positively in mix vs pt)

All 4 evaluated on R(F) subset (same denominator → directly comparable).

NOTE on SAE-B/C: The precomputed recon_delta files store
  recon_delta[vi][l] = mean_over_text_tokens(recon_mix - recon_pt)  [2304]
Projecting onto W_dec[F] (unit norm) extracts the F-th feature's contribution:
  coeff_F[vi] = recon_delta[vi][l] · W_dec[F]
  delta_F[vi] = coeff_F[vi] * W_dec[F]
This is mathematically equivalent to (mean_act_mix_F - mean_act_pt_F) * W_dec[F]
averaged over text tokens, which is the correct mean-token SAE feature gap.

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 -B pt448_spatial_feature_ablation.py
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
PT_MODEL      = "google/paligemma2-3b-pt-448"
MIX_H_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
PT_H_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/pt_hidden")
DELTA_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta_all26/deltas")
CKPT_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR       = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_spatial_feature_ablation")
IMAGE_CACHE   = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET   = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# CAA alphas (unit-normed vectors)
CAA_ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
# SAE alphas (projection-scaled vectors — natural scale of the feature gap)
SAE_ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

SPATIAL_FEATURES = [
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


# ─────────────────────── Phase 1: Precompute vectors ──────────
def precompute(vsr_all, base_preds, features=None):
    N = len(vsr_all)
    if features is None:
        features = SPATIAL_FEATURES
    print(f"[PHASE 1] Precomputing for {len(features)} features...", flush=True)

    results = {}

    for feat in features:
        l = feat["layer"]; F = feat["feature"]
        rels = set(r.strip().lower() for r in feat["relations"])
        feat_key = f"L{l}_F{F}"
        print(f"\n  [{feat_key}] relations: {sorted(rels)}", flush=True)

        # R(F): samples with this relation that have hidden states cached
        subset_vis = [
            vi for vi in range(N)
            if str(vsr_all[vi].get("relation", "")).strip().lower() in rels
            and (MIX_H_DIR / f"vi_{vi:05d}.pt").exists()
            and (PT_H_DIR  / f"vi_{vi:05d}.pt").exists()
        ]
        print(f"    R(F) subset: {len(subset_vis)} samples", flush=True)

        # Baseline acc on subset
        n_correct = sum(base_preds[str(vi)]["correct"] for vi in subset_vis if str(vi) in base_preds)
        n_with_pred = sum(1 for vi in subset_vis if str(vi) in base_preds)
        base_acc = n_correct / max(n_with_pred, 1) * 100

        # Load W_dec[F] for this layer (unit norm by construction in our SAE)
        ckpt = torch.load(CKPT_DIR / f"text-only_layer_{l}.pt", map_location="cpu", weights_only=True)
        W_dec_F = ckpt["W_dec"][F].float()  # [2304]
        W_dec_F_norm = W_dec_F / W_dec_F.norm().clamp(min=1e-8)
        del ckpt

        # ── CAA-1: mean(h_mix[l] - h_pt[l]) over ALL samples ──
        caa1_sum = None; caa1_n = 0
        for vi in range(N):
            mix_p = MIX_H_DIR / f"vi_{vi:05d}.pt"
            pt_p  = PT_H_DIR  / f"vi_{vi:05d}.pt"
            if not mix_p.exists() or not pt_p.exists(): continue
            try:
                mix_h = torch.load(mix_p, map_location="cpu", weights_only=True)
                pt_h  = torch.load(pt_p,  map_location="cpu", weights_only=True)
            except Exception: continue
            if l not in mix_h or l not in pt_h: continue
            delta = mix_h[l].float() - pt_h[l].float()
            if caa1_sum is None: caa1_sum = delta.clone()
            else:                caa1_sum += delta
            caa1_n += 1
        caa1_raw = caa1_sum / caa1_n if caa1_sum is not None else torch.zeros(2304)
        caa1_vec = caa1_raw / caa1_raw.norm().clamp(min=1e-8)
        print(f"    CAA-1: raw_norm={caa1_raw.norm():.3f}  n={caa1_n}", flush=True)

        # ── CAA-2: mean(h_mix[l] - h_pt[l]) over R(F) only ──
        caa2_sum = None; caa2_n = 0
        for vi in subset_vis:
            try:
                mix_h = torch.load(MIX_H_DIR / f"vi_{vi:05d}.pt", map_location="cpu", weights_only=True)
                pt_h  = torch.load(PT_H_DIR  / f"vi_{vi:05d}.pt", map_location="cpu", weights_only=True)
            except Exception: continue
            if l not in mix_h or l not in pt_h: continue
            delta = mix_h[l].float() - pt_h[l].float()
            if caa2_sum is None: caa2_sum = delta.clone()
            else:                caa2_sum += delta
            caa2_n += 1
        caa2_raw = caa2_sum / caa2_n if caa2_sum is not None else torch.zeros(2304)
        caa2_vec = caa2_raw / caa2_raw.norm().clamp(min=1e-8)
        print(f"    CAA-2: raw_norm={caa2_raw.norm():.3f}  n={caa2_n}", flush=True)

        # ── SAE-B/C: per-sample feature projection from precomputed recon deltas ──
        # delta_vi = recon_mix[vi][l] - recon_pt[vi][l]  [2304]
        # coeff_F  = delta_vi · W_dec_F_norm              scalar
        # inject   = coeff_F * W_dec_F_norm               [2304]
        sae_coeffs = {}  # vi -> coeff_F (scalar)
        n_pos = n_neg = n_missing = 0
        for vi in subset_vis:
            dp = DELTA_DIR / f"vi_{vi:05d}.pt"
            if not dp.exists():
                n_missing += 1; continue
            try:
                delta_dict = torch.load(dp, map_location="cpu", weights_only=True)
            except Exception:
                n_missing += 1; continue
            if l not in delta_dict:
                n_missing += 1; continue
            coeff = (delta_dict[l].float() @ W_dec_F_norm).item()
            sae_coeffs[vi] = coeff
            if coeff > 0: n_pos += 1
            else: n_neg += 1
        mean_coeff = sum(sae_coeffs.values()) / max(len(sae_coeffs), 1)
        print(f"    SAE proj: n={len(sae_coeffs)}  mean_coeff={mean_coeff:.4f}  pos={n_pos}  neg={n_neg}  missing={n_missing}", flush=True)

        results[feat_key] = {
            "layer": l, "feature": F, "relations": sorted(rels),
            "subset_vis": subset_vis, "subset_n": n_with_pred,
            "base_acc": base_acc,
            "caa1_vec": caa1_vec,      # unit-normed [2304]
            "caa2_vec": caa2_vec,      # unit-normed [2304]
            "sae_coeffs": sae_coeffs,  # vi -> scalar (projection of recon delta onto W_dec[F])
            "W_dec_F_norm": W_dec_F_norm,  # [2304] unit-normed decoder direction
        }

    return results


# ─────────────────────── Phase 2: Inference ───────────────────
def run_feature(feat_key, feat_data, vsr_all, model_raw, processor, yes_ids, no_ids, device):
    from utils import process_vlm_inputs, get_image_token_positions

    l          = feat_data["layer"]
    subset_vis = feat_data["subset_vis"]
    base_acc   = feat_data["base_acc"]
    caa1_vec   = feat_data["caa1_vec"].to(next(model_raw.parameters()).dtype).to(device)
    caa2_vec   = feat_data["caa2_vec"].to(next(model_raw.parameters()).dtype).to(device)
    sae_coeffs = feat_data["sae_coeffs"]
    W_dec_F    = feat_data["W_dec_F_norm"].to(next(model_raw.parameters()).dtype).to(device)
    model_dtype = next(model_raw.parameters()).dtype

    print(f"\n{'='*60}", flush=True)
    print(f"[{feat_key}]  n={feat_data['subset_n']}  base={base_acc:.2f}%", flush=True)

    out_path = OUT_DIR / f"{feat_key}.json"
    if out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        # Check if already complete
        total_alphas = len(CAA_ALPHAS) * 2 + len(SAE_ALPHAS) * 2
        done = sum(len(v) for v in [saved.get("caa1",{}), saved.get("caa2",{}), saved.get("sae_b",{}), saved.get("sae_c",{})])
        if done >= total_alphas:
            print(f"  [SKIP] already complete ({done} alpha results)", flush=True)
            return saved

    feat_results = {
        "layer": l, "feature": feat_data["feature"],
        "relations": feat_data["relations"],
        "base_acc": base_acc, "subset_n": feat_data["subset_n"],
        "caa1": {}, "caa2": {}, "sae_b": {}, "sae_c": {},
    }
    if out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        for k in ["caa1", "caa2", "sae_b", "sae_c"]:
            feat_results[k] = saved.get(k, {})

    img_end_ref = [0]

    for method, alphas in [("caa1", CAA_ALPHAS), ("caa2", CAA_ALPHAS), ("sae_b", SAE_ALPHAS), ("sae_c", SAE_ALPHAS)]:
        print(f"\n  Method={method}", flush=True)
        for alpha in alphas:
            akey = str(alpha)
            if akey in feat_results[method]:
                r = feat_results[method][akey]
                print(f"    [SKIP] α={alpha}: Δ={r['delta']:+.2f}%", flush=True)
                continue

            correct = total = injected = 0
            for vi in subset_vis:
                ex    = vsr_all[vi]
                label = int(ex.get("label", 0))
                img   = _load_image(ex)
                if img is None: continue

                # Determine injection vector for this sample+method
                if method == "caa1":
                    vec = alpha * caa1_vec
                elif method == "caa2":
                    vec = alpha * caa2_vec
                elif method == "sae_b":
                    coeff = sae_coeffs.get(vi)
                    if coeff is None: vec = None
                    else: vec = torch.tensor(alpha * coeff, dtype=model_dtype, device=device) * W_dec_F
                elif method == "sae_c":
                    coeff = sae_coeffs.get(vi)
                    if coeff is None or coeff <= 0: vec = None
                    else: vec = torch.tensor(alpha * coeff, dtype=model_dtype, device=device) * W_dec_F

                prompt = _build_vsr_prompt(str(ex.get("caption", "")))
                hook = None
                try:
                    iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                    _, img_end = get_image_token_positions(iids)
                    img_end_ref[0] = img_end

                    if vec is not None:
                        def make_hook(ie=img_end, v=vec):
                            def hook_fn(module, input, output):
                                hidden = output[0]
                                hidden[0, ie:] = hidden[0, ie:] + v.unsqueeze(0)
                                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                            return hook_fn
                        hook = model_raw.model.language_model.layers[l].register_forward_hook(make_hook())
                        injected += 1

                    with torch.no_grad():
                        out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    if hook is not None: hook.remove()

                    pred   = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                    total += 1
                    correct += int(pred == label)
                except Exception as e:
                    if hook is not None:
                        try: hook.remove()
                        except: pass

            if total == 0: continue
            acc   = correct / total * 100
            delta = acc - base_acc
            feat_results[method][akey] = {"acc": acc, "delta": delta, "n": total, "n_injected": injected}
            print(f"    α={alpha:>5}: acc={acc:.2f}%  Δ={delta:+.2f}%  inj={injected}/{total}", flush=True)

            with open(out_path, "w") as f:
                json.dump(feat_results, f, indent=2)

    gc.collect(); torch.cuda.empty_cache()
    return feat_results


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("Spatial Feature Steering Ablation: CAA-1 vs CAA-2 vs SAE-B vs SAE-C")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    with open(PT_H_DIR / "base_predictions.json") as f:
        base_preds = json.load(f)

    # Determine which features this process should run
    feat_idx_str = os.environ.get("FEATURE_IDX", "ALL")
    if feat_idx_str.upper() == "ALL":
        assigned_indices = list(range(len(SPATIAL_FEATURES)))
    else:
        assigned_indices = [int(x) for x in feat_idx_str.split(",")]
    assigned_features = [SPATIAL_FEATURES[i] for i in assigned_indices]
    assigned_keys = {f"L{f['layer']}_F{f['feature']}" for f in assigned_features}
    print(f"[INFO] Assigned features: {sorted(assigned_keys)}", flush=True)

    # Phase 1: CPU precomputation — only for assigned features
    precomp = precompute(vsr_all, base_preds, assigned_features)

    # Phase 2: Load model and run inference
    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    all_results = {}
    for feat_key, feat_data in precomp.items():
        res = run_feature(feat_key, feat_data, vsr_all, model_raw, processor, yes_ids, no_ids, device)
        all_results[feat_key] = res

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY — Spatial Feature Ablation")
    print(f"{'Feature':<14} {'N':>5} {'Base':>6} | {'CAA-1 best':>11} {'CAA-2 best':>11} {'SAE-B best':>11} {'SAE-C best':>11}")
    print("-" * 80)
    for feat_key, res in all_results.items():
        def best(d):
            if not d: return "  --"
            b = max(d.items(), key=lambda x: x[1]["delta"])
            return f"{b[1]['delta']:+.2f}% (α={b[0]})"
        print(f"{feat_key:<14} {res['subset_n']:>5} {res['base_acc']:>5.1f}% | "
              f"{best(res['caa1']):>11} {best(res['caa2']):>11} {best(res['sae_b']):>11} {best(res['sae_c']):>11}")

    print(f"\nResults in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
