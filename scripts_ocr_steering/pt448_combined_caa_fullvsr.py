#!/usr/bin/env python3
"""
Combined Per-Relation CAA on Full VSR (hook-based).

For each sample in full VSR:
  1. Identify its spatial relation
  2. Look up which spatial feature F covers that relation → (layer l, v[F], alpha)
  3. Inject alpha * v[F] at layer l via register_forward_hook
  4. If no feature covers the relation, use base model prediction

Steering vectors v[F] are computed identically to pt448_per_relation_steer.py:
  v[F] = mean(h_mix[l] - h_pt[l]) over R(F) samples, then unit-normalized (SCALE_MODE=NORM).

Optimal alphas per feature (from per_relation subset sweeps):
  L4/F14233: α=10  (+2.41% on subset)
  L6/F7539:  α=2   (+0.29%)
  L9/F387:   α=1   (+0.40%)
  L9/F7540:  α=5   (+1.38%)
  L11/F12278: α=50 (+3.41%)
  L12/F2257:  α=50 (+5.24%)
  L14/F10561: α=20 (+7.14%)
  L15/F220:   α=10 (+4.79%)

For overlapping relations (e.g. "right of" in L6 and L15), priority goes to the
feature with higher subset Δacc (L15 > L6 for "right of").

ENV:
    ALPHA_SCALE  float multiplier on all optimal alphas (default: 1.0)
    ALPHAS_OVERRIDE  if set, replaces ALL optimal alphas with this single float
    OUT_SUFFIX  suffix for results file (default: "")
    SCALE_MODE  NORM (default) | RAW

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 -B pt448_combined_caa_fullvsr.py
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
PT_MODEL    = "google/paligemma2-3b-pt-448"
MIX_H_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
PT_H_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/pt_hidden")
OUT_DIR     = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_combined_caa_fullvsr")
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHA_SCALE      = float(os.environ.get("ALPHA_SCALE", "1.0"))
ALPHAS_OVERRIDE  = os.environ.get("ALPHAS_OVERRIDE", "")
SCALE_MODE       = os.environ.get("SCALE_MODE", "NORM")
OUT_SUFFIX       = os.environ.get("OUT_SUFFIX", "")

# Feature definitions with optimal alphas from per_relation subset sweeps
# Priority: higher subset_delta wins for overlapping relations
FEATURES = [
    {"layer": 4,  "feature": 14233, "subset_delta": 2.41, "opt_alpha": 10.0,
     "relations": ["ahead of", "behind"]},
    {"layer": 6,  "feature": 7539,  "subset_delta": 0.29, "opt_alpha": 2.0,
     "relations": ["left of", "right of", "across from", "alongside", "at the back of", "below", "facing away from"]},
    {"layer": 9,  "feature": 387,   "subset_delta": 0.40, "opt_alpha": 1.0,
     "relations": ["at the right side of", "adjacent to", "far from", "attached to"]},
    {"layer": 9,  "feature": 7540,  "subset_delta": 1.38, "opt_alpha": 5.0,
     "relations": ["on", "next to", "parallel to", "in the middle of", "opposite to", "away from", "consists of"]},
    {"layer": 11, "feature": 12278, "subset_delta": 3.41, "opt_alpha": 50.0,
     "relations": ["touching", "on top of", "surrounding", "under"]},
    {"layer": 12, "feature": 2257,  "subset_delta": 5.24, "opt_alpha": 50.0,
     "relations": ["facing", "beneath", "near", "off", "enclosed by", "inside", "within", "beyond", "at the side of"]},
    {"layer": 14, "feature": 10561, "subset_delta": 7.14, "opt_alpha": 20.0,
     "relations": ["close to", "by", "connected to"]},
    {"layer": 15, "feature": 220,   "subset_delta": 4.79, "opt_alpha": 10.0,
     "relations": ["above", "at the left side of", "beside", "contains", "over", "part of", "right of", "outside", "toward"]},
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


def _compute_steering_vector(layer, vis, dtype="float32"):
    """Compute v[F] = mean(h_mix[l] - h_pt[l]) over subset vis, optionally unit-norm."""
    steer_sum   = None
    steer_count = 0
    for vi in vis:
        mix_p = MIX_H_DIR / f"vi_{vi:05d}.pt"
        pt_p  = PT_H_DIR  / f"vi_{vi:05d}.pt"
        if not mix_p.exists() or not pt_p.exists():
            continue
        try:
            mix_h = torch.load(mix_p, map_location="cpu", weights_only=True)
            pt_h  = torch.load(pt_p,  map_location="cpu", weights_only=True)
        except Exception:
            continue
        if layer not in mix_h or layer not in pt_h:
            continue
        delta = mix_h[layer].float() - pt_h[layer].float()
        if steer_sum is None:
            steer_sum = delta.clone()
        else:
            steer_sum += delta
        steer_count += 1
    if steer_sum is None or steer_count == 0:
        return None, 0
    v_raw = steer_sum / steer_count
    if SCALE_MODE == "NORM":
        v = v_raw / v_raw.norm().clamp(min=1e-8)
    else:
        v = v_raw
    return v, steer_count


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    print("=" * 70)
    print(f"Combined Per-Relation CAA — Full VSR")
    print(f"SCALE_MODE={SCALE_MODE}  ALPHA_SCALE={ALPHA_SCALE}")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load VSR ──
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    print(f"[INFO] VSR total: {N}", flush=True)

    # ── Load base predictions ──
    base_preds_path = PT_H_DIR / "base_predictions.json"
    with open(base_preds_path) as f:
        base_preds = json.load(f)
    base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
    print(f"[INFO] Baseline: {base_acc:.2f}% over {len(base_preds)} samples", flush=True)

    # ── Build relation → feature routing table ──
    # For each relation, pick the feature with highest subset_delta
    rel_to_feat = {}  # relation → feat dict
    for feat in sorted(FEATURES, key=lambda x: x["subset_delta"]):
        for rel in feat["relations"]:
            rel_to_feat[rel.strip().lower()] = feat

    print(f"[INFO] Relations covered: {len(rel_to_feat)}", flush=True)

    # ── Compute steering vectors for all 8 features ──
    print("[INFO] Computing steering vectors...", flush=True)
    feat_vectors = {}  # (layer, feature) → v_steer tensor
    for feat in FEATURES:
        l = feat["layer"]; F = feat["feature"]
        rels = set(r.strip().lower() for r in feat["relations"])
        # Collect sample indices for this feature's relations
        vis = [vi for vi in range(N)
               if str(vsr_all[vi].get("relation","")).strip().lower() in rels
               and (MIX_H_DIR / f"vi_{vi:05d}.pt").exists()
               and (PT_H_DIR  / f"vi_{vi:05d}.pt").exists()]
        v, n = _compute_steering_vector(l, vis)
        if v is None:
            print(f"  [WARN] L{l}/F{F}: no valid hidden states!", flush=True)
            continue
        feat_vectors[(l, F)] = v
        tag = f"L{l}_F{F}"
        alpha = float(ALPHAS_OVERRIDE) if ALPHAS_OVERRIDE else feat["opt_alpha"] * ALPHA_SCALE
        print(f"  {tag}: v_norm={v.norm():.4f}  n={n}  alpha={alpha}", flush=True)

    # ── Load pt-448 model ──
    print(f"[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer    = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # Move all steering vectors to GPU
    v_gpu = {k: v.to(model_dtype).to(device) for k, v in feat_vectors.items()}

    # ── Run full VSR with routing ──
    results_path = OUT_DIR / f"combined_caa_fullvsr{OUT_SUFFIX}.json"
    if results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)
        if saved.get("n_total", 0) >= N * 0.99:
            print(f"[SKIP] Results already complete: {results_path}", flush=True)
            print(f"  acc={saved['acc']:.2f}%  Δ={saved['delta']:+.2f}%  n={saved['n_total']}", flush=True)
            return

    img_end_ref = [0]
    correct = total = steered = 0
    relation_stats = {}  # rel → {n, correct, steered}

    for vi in range(N):
        ex  = vsr_all[vi]
        rel = str(ex.get("relation", "")).strip().lower()
        label = int(ex.get("label", 0))

        # Check if this relation is covered
        matched_feat = rel_to_feat.get(rel)

        img = _load_image(ex)
        if img is None:
            continue

        prompt = _build_vsr_prompt(str(ex.get("caption", "")))

        try:
            iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
            _, img_end = get_image_token_positions(iids)
            img_end_ref[0] = img_end

            hook = None
            if matched_feat is not None:
                l = matched_feat["layer"]
                F = matched_feat["feature"]
                alpha = float(ALPHAS_OVERRIDE) if ALPHAS_OVERRIDE else matched_feat["opt_alpha"] * ALPHA_SCALE
                v = v_gpu.get((l, F))
                if v is not None:
                    def make_hook(ie_ref=img_end_ref, a=alpha, vec=v):
                        def hook_fn(module, input, output):
                            ie = ie_ref[0]
                            hidden = output[0]
                            hidden[0, ie:] = hidden[0, ie:] + a * vec.unsqueeze(0)
                            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                        return hook_fn
                    hook = model_raw.model.language_model.layers[l].register_forward_hook(make_hook())
                    steered += 1

            with torch.no_grad():
                out = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv)

            if hook is not None:
                hook.remove()

            pred   = _predict(out.logits[0, -1, :], yes_ids, no_ids)
            total += 1
            correct += int(pred == label)

            # Track per-relation stats
            if rel not in relation_stats:
                relation_stats[rel] = {"n": 0, "correct": 0, "steered": hook is not None}
            relation_stats[rel]["n"] += 1
            relation_stats[rel]["correct"] += int(pred == label)

        except Exception as e:
            if hook is not None:
                try: hook.remove()
                except Exception: pass
            if total < 10:
                print(f"  [WARN] vi={vi}: {e}", flush=True)

        if (vi + 1) % 1000 == 0:
            acc = correct / max(total, 1) * 100
            print(f"  [{vi+1}/{N}] acc={acc:.2f}%  Δ={acc-base_acc:+.2f}%  steered={steered}", flush=True)

    acc = correct / max(total, 1) * 100
    delta = acc - base_acc

    print(f"\n{'='*60}", flush=True)
    print(f"Combined CAA Full VSR Result:", flush=True)
    print(f"  acc={acc:.2f}%  Δ={delta:+.2f}%  ({correct}/{total})", flush=True)
    print(f"  steered={steered}/{total} samples", flush=True)
    print(f"  baseline={base_acc:.2f}%", flush=True)
    print(f"{'='*60}", flush=True)

    result = {
        "method": "combined_per_relation_caa",
        "scale_mode": SCALE_MODE,
        "alpha_scale": ALPHA_SCALE,
        "base_acc": base_acc,
        "acc": acc,
        "delta": delta,
        "n_total": total,
        "n_steered": steered,
        "relation_stats": relation_stats,
        "features": [
            {
                "layer": f["layer"], "feature": f["feature"],
                "opt_alpha": float(ALPHAS_OVERRIDE) if ALPHAS_OVERRIDE else f["opt_alpha"] * ALPHA_SCALE,
                "subset_delta": f["subset_delta"]
            } for f in FEATURES
        ]
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Saved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
