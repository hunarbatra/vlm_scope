#!/usr/bin/env python3
"""
Per-Relation CAA Steering for pt-448 (hook-based, no nnsight overhead).

For each of the 8 spatial features F at layer l with relations R(F):
  1. Load saved hidden states (mix_hidden/, pt_hidden/) for R(F) samples only
  2. Compute relation-specific steering vector: v[F] = mean(h_mix[l] - h_pt[l]) over R(F)
  3. Inject alpha * v[F] into pt-448 text positions at layer l, for R(F) samples only
  4. Evaluate accuracy on R(F) subset; report Δ vs baseline

Uses PyTorch register_forward_hook (not nnsight) to avoid the ~17 GB proxy overhead.

Usage:
    CUDA_VISIBLE_DEVICES=0 FEATURE_IDX=0 python3 -B pt448_per_relation_steer.py
    CUDA_VISIBLE_DEVICES=1 FEATURE_IDX=1 python3 -B pt448_per_relation_steer.py
    FEATURE_IDX=0,1,2  — run specified features sequentially
    FEATURE_IDX=ALL    — run all 8 (default)

ENV:
    FEATURE_IDX  int or comma-list or ALL (default: ALL)
    SCALE_MODE   NORM (unit-normalize v) | RAW (use raw mean delta) (default: NORM)
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
OUT_DIR     = Path(os.environ.get("OUT_DIR", "/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_per_relation_steer"))
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

_base_alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
# EXTRA_ALPHAS="100,200,500" adds to base list; ALPHAS="..." overrides fully
if os.environ.get("ALPHAS"):
    ALPHAS = [float(x) for x in os.environ["ALPHAS"].split(",")]
else:
    _extra = [float(x) for x in os.environ.get("EXTRA_ALPHAS", "").split(",") if x.strip()]
    ALPHAS = sorted(set(_base_alphas + _extra))

SCALE_MODE = os.environ.get("SCALE_MODE", "NORM")

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
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None

def _make_hook(img_end_ref, alpha, v_gpu):
    """Create a forward hook that adds alpha*v_gpu to text token positions."""
    def hook_fn(module, input, output):
        ie = img_end_ref[0]
        hidden = output[0]           # [1, T, 2304]
        hidden[0, ie:] = hidden[0, ie:] + alpha * v_gpu
        # Return modified output tuple
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook_fn


# ─────────────────────── Per-feature runner ───────────────────
def run_feature(feat_idx, feat, vsr_all, base_preds, model_raw, processor,
                yes_ids, no_ids, model_dtype, device):
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    l       = feat["layer"]
    F       = feat["feature"]
    rels    = set(r.strip().lower() for r in feat["relations"])
    tag     = f"L{l}_F{F}"

    out_path = OUT_DIR / f"{tag}.json"
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing.get("alphas", {})) >= len(ALPHAS):
            print(f"[SKIP] {tag}: all {len(ALPHAS)} alphas done.", flush=True)
            return existing

    print(f"\n{'='*70}", flush=True)
    print(f"Feature {feat_idx}: {tag}  scale_mode={SCALE_MODE}", flush=True)
    print(f"  Layer: {l}  Spatial feature: {F}", flush=True)
    print(f"  Relations: {sorted(rels)}", flush=True)

    # ── Step 1: Collect R(F) sample indices ──
    N = len(vsr_all)
    subset_vis = []
    for vi in range(N):
        ex  = vsr_all[vi]
        rel = str(ex.get("relation", "")).strip().lower()
        if rel not in rels:
            continue
        mix_path = MIX_H_DIR / f"vi_{vi:05d}.pt"
        pt_path  = PT_H_DIR  / f"vi_{vi:05d}.pt"
        if not mix_path.exists() or not pt_path.exists():
            continue
        subset_vis.append(vi)

    if not subset_vis:
        print(f"  [{tag}] No valid R(F) samples found!", flush=True)
        return None

    print(f"  R(F) sample count: {len(subset_vis)}", flush=True)

    # ── Step 2: Compute relation-specific steering vector ──
    print(f"  Computing v[F] = mean(h_mix[l] - h_pt[l]) over R(F)...", flush=True)
    steer_sum   = None
    steer_count = 0

    for vi in subset_vis:
        mix_path = MIX_H_DIR / f"vi_{vi:05d}.pt"
        pt_path  = PT_H_DIR  / f"vi_{vi:05d}.pt"
        try:
            mix_h = torch.load(mix_path, map_location="cpu", weights_only=True)
            pt_h  = torch.load(pt_path,  map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"    [WARN] vi={vi}: load error: {e}", flush=True)
            continue

        if l not in mix_h or l not in pt_h:
            continue

        delta = mix_h[l].float() - pt_h[l].float()  # [2304]
        if steer_sum is None:
            steer_sum = delta.clone()
        else:
            steer_sum += delta
        steer_count += 1

    if steer_sum is None or steer_count == 0:
        print(f"  [{tag}] No valid hidden state deltas!", flush=True)
        return None

    v_raw  = steer_sum / steer_count   # [2304] mean delta
    v_norm = v_raw.norm().item()
    print(f"  Steering vector: raw_norm={v_norm:.4f}  over {steer_count} samples", flush=True)

    if SCALE_MODE == "NORM":
        v_steer = v_raw / v_raw.norm().clamp(min=1e-8)
        print(f"  Mode NORM: unit-normalized; alpha scales magnitude", flush=True)
    else:
        v_steer = v_raw
        print(f"  Mode RAW: raw mean delta (norm={v_norm:.4f})", flush=True)

    v_gpu = v_steer.to(model_dtype).to(device)

    # ── Step 3: Baseline accuracy on R(F) subset ──
    sub_correct = sum(base_preds[str(vi)]["correct"] for vi in subset_vis if str(vi) in base_preds)
    sub_n       = sum(1 for vi in subset_vis if str(vi) in base_preds)
    sub_base    = sub_correct / max(sub_n, 1) * 100
    print(f"  Baseline on R(F): {sub_base:.2f}%  ({sub_correct}/{sub_n})", flush=True)

    # Load existing partial results
    alpha_results = {}
    if out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        alpha_results = saved.get("alphas", {})

    # Target layer module
    layer_module = model_raw.model.language_model.layers[l]

    # ── Step 4: Alpha sweep ──
    img_end_ref = [0]  # mutable ref captured by hook closure

    for alpha in ALPHAS:
        alpha_key = str(alpha)
        if alpha_key in alpha_results:
            r = alpha_results[alpha_key]
            print(f"  [SKIP] alpha={alpha}: acc={r['acc']:.2f}%  Δ={r['delta']:+.2f}%", flush=True)
            continue

        print(f"\n  [alpha={alpha}] Running {len(subset_vis)} samples...", flush=True)
        correct = total = 0
        hook_fn = _make_hook(img_end_ref, alpha, v_gpu)

        for vi in subset_vis:
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

                hook = layer_module.register_forward_hook(hook_fn)
                with torch.no_grad():
                    outputs = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv)
                hook.remove()

                pred   = _predict(outputs.logits[0, -1, :], yes_ids, no_ids)
                total += 1
                correct += int(pred == label)

            except Exception as e:
                if total < 5:
                    print(f"    [WARN] vi={vi}: {e}", flush=True)
                try:
                    hook.remove()
                except Exception:
                    pass
                continue

        if total == 0:
            print(f"    alpha={alpha}: no valid samples!", flush=True)
            continue

        acc   = correct / total * 100
        delta = acc - sub_base
        alpha_results[alpha_key] = {"acc": acc, "delta": delta, "n": total}
        print(f"    alpha={alpha:>5.1f}: {acc:.2f}%  Δ={delta:+.2f}%  ({correct}/{total})", flush=True)

        result = {
            "layer": l, "feature": F, "relations": sorted(rels),
            "base_acc": sub_base, "subset_n": len(subset_vis),
            "steer_vec_norm": v_norm, "scale_mode": SCALE_MODE,
            "alphas": alpha_results
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    del v_gpu
    torch.cuda.empty_cache(); gc.collect()

    result = {
        "layer": l, "feature": F, "relations": sorted(rels),
        "base_acc": sub_base, "subset_n": len(subset_vis),
        "steer_vec_norm": v_norm, "scale_mode": SCALE_MODE,
        "alphas": alpha_results
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)
    return result


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    feat_idx_str = os.environ.get("FEATURE_IDX", "ALL")
    if feat_idx_str.upper() == "ALL":
        feat_indices = list(range(len(FEATURES)))
    else:
        feat_indices = [int(x) for x in feat_idx_str.split(",")]

    print("=" * 70)
    print(f"Per-Relation CAA Steering (hook-based)  scale_mode={SCALE_MODE}")
    print(f"Features: {feat_indices}  Alphas: {ALPHAS}")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load base predictions ──
    base_preds_path = PT_H_DIR / "base_predictions.json"
    if not base_preds_path.exists():
        base_preds_path = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta/pt_reconstructions/base_predictions.json")
    if not base_preds_path.exists():
        print("[ERROR] base_predictions.json not found.", flush=True)
        sys.exit(1)
    with open(base_preds_path) as f:
        base_preds = json.load(f)
    base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
    print(f"[INFO] Global baseline: {base_acc:.2f}% over {len(base_preds)} samples", flush=True)

    # ── Load VSR ──
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    # ── Load pt-448 ──
    print(f"[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer   = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # ── Run per feature ──
    all_results = {}
    for fi in feat_indices:
        feat = FEATURES[fi]
        tag  = f"L{feat['layer']}_F{feat['feature']}"
        res  = run_feature(fi, feat, vsr_all, base_preds, model_raw, processor,
                           yes_ids, no_ids, model_dtype, device)
        if res:
            all_results[tag] = res

    # ── Summary table ──
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY — Per-Relation CAA Steering  (scale_mode={SCALE_MODE})")
    print(f"{'='*80}", flush=True)
    hdr = f"{'Feature':>12}  {'N':>5}  {'Base':>6}"
    for a in ALPHAS:
        hdr += f"  {str(a):>6}"
    print(hdr)
    print("-" * len(hdr))

    for fi in feat_indices:
        feat = FEATURES[fi]
        tag  = f"L{feat['layer']}_F{feat['feature']}"
        res  = all_results.get(tag)
        if not res:
            continue
        row = f"  L{feat['layer']}/F{feat['feature']:>6}  {res.get('subset_n',0):>5}  {res.get('base_acc',0):>5.1f}%"
        best_delta = -999
        best_alpha = None
        for a in ALPHAS:
            r = res.get("alphas", {}).get(str(a), {})
            if r:
                d = r["delta"]
                row += f"  {d:>+5.1f}%"
                if d > best_delta:
                    best_delta = d
                    best_alpha = a
            else:
                row += f"  {'--':>6}"
        if best_alpha is not None:
            row += f"  <- best α={best_alpha}"
        print(row)

    print(f"\nResults in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
