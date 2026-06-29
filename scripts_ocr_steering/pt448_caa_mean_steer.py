#!/usr/bin/env python3
"""
CAA-style Mean Reconstruction Steering for pt-448.

Analogous to Contrast-Consistent Activation Addition (CAA):
  1. Compute dataset-mean steering vectors per SAE layer:
        v_steer[l] = mean over N samples of (recon_mix[l] - recon_pt[l])
  2. At inference: hidden[l] += alpha * v_steer[l]  (same vector, same alpha, all samples)

No per-sample recon loading at inference. One global alpha. Fully universal.

Two modes (MODE env var):
  FULL      Use the raw mean delta vector (all 16k features contribute)
  TARGETED  Project mean delta onto the 8 spatial W_dec directions only
              v_targeted[l] = sum_F( (v_steer[l] @ W_dec[F]) * W_dec[F] )

Usage:
    CUDA_VISIBLE_DEVICES=1 MODE=FULL     python3 -B pt448_caa_mean_steer.py
    CUDA_VISIBLE_DEVICES=2 MODE=TARGETED python3 -B pt448_caa_mean_steer.py
"""

import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# ─────────────────────── Config ───────────────────────────────
PT_MODEL       = "google/paligemma2-3b-pt-448"
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
MIX_RECON_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta/mix_reconstructions")
PT_RECON_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta/pt_reconstructions")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_mean_steer")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MODE   = os.environ.get("MODE", "FULL")   # FULL or TARGETED
# SAE layers used in Phase 1/2 of pt448_sae_recon_delta.py
SAE_LAYERS = [4, 6, 9, 11, 12, 14, 15]

ALPHAS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# 8 spatial features for TARGETED mode
SPATIAL_FEATURES = [
    {"layer": 4,  "feature": 14233},
    {"layer": 6,  "feature": 7539},
    {"layer": 9,  "feature": 387},
    {"layer": 9,  "feature": 7540},
    {"layer": 11, "feature": 12278},
    {"layer": 12, "feature": 2257},
    {"layer": 14, "feature": 10561},
    {"layer": 15, "feature": 220},
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


# ─────────────────────── Step 1: Compute mean steering vectors ─
def compute_mean_steering_vectors():
    """Average (recon_mix[l] - recon_pt[l]) over all N samples per SAE layer."""
    print("[STEP 1] Computing dataset-mean steering vectors...", flush=True)

    mix_files = sorted(MIX_RECON_DIR.glob("vi_*.pt"))
    N = len(mix_files)
    print(f"  Found {N} mix recon files.", flush=True)

    # Accumulate sums per layer
    sums  = {l: None for l in SAE_LAYERS}
    counts = {l: 0   for l in SAE_LAYERS}

    for i, mix_path in enumerate(mix_files):
        vi_str = mix_path.stem  # vi_00000
        pt_path = PT_RECON_DIR / f"{vi_str}.pt"
        if not pt_path.exists():
            continue

        try:
            recon_mix = torch.load(mix_path, map_location="cpu", weights_only=True)
            recon_pt  = torch.load(pt_path,  map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  [WARN] {vi_str}: {e}", flush=True)
            continue

        for l in SAE_LAYERS:
            if l not in recon_mix or l not in recon_pt:
                continue
            delta = recon_mix[l].float() - recon_pt[l].float()  # [2304]
            if sums[l] is None:
                sums[l] = delta.clone()
            else:
                sums[l] += delta
            counts[l] += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N} files processed...", flush=True)

    mean_vecs = {}
    for l in SAE_LAYERS:
        if sums[l] is not None and counts[l] > 0:
            mean_vecs[l] = sums[l] / counts[l]  # [2304] float32
            norm = mean_vecs[l].norm().item()
            print(f"  Layer {l:2d}: mean_delta norm={norm:.4f}  (n={counts[l]})", flush=True)
        else:
            print(f"  Layer {l:2d}: SKIPPED (no valid data)", flush=True)

    return mean_vecs


def project_onto_spatial_features(mean_vecs):
    """Project mean_vecs[l] onto the 8 spatial W_dec directions."""
    print("\n[STEP 1b] Projecting onto 8 spatial W_dec directions...", flush=True)
    layer_to_feats = {}
    for sf in SPATIAL_FEATURES:
        layer_to_feats.setdefault(sf["layer"], []).append(sf["feature"])

    targeted_vecs = {}
    for l, feat_ids in layer_to_feats.items():
        if l not in mean_vecs:
            continue
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{l}.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        W_dec = state["W_dec"].float()  # [16384, 2304]
        del state

        v = mean_vecs[l]  # [2304]
        projected = torch.zeros(2304)
        for F in feat_ids:
            wdec = W_dec[F]
            wdec = wdec / wdec.norm().clamp(min=1e-8)
            coeff = (v @ wdec).item()
            projected += coeff * wdec
            print(f"  L{l}/F{F}: projection coeff={coeff:.4f}", flush=True)
        del W_dec

        if targeted_vecs.get(l) is None:
            targeted_vecs[l] = projected
        else:
            targeted_vecs[l] += projected

        norm = targeted_vecs[l].norm().item()
        print(f"  Layer {l}: targeted vec norm={norm:.4f}", flush=True)

    return targeted_vecs


# ─────────────────────── Step 2: Inference with universal alpha ─
def run_inference(steering_vecs, mode_tag, vsr_all, base_preds, nns_model, model_raw,
                  processor, yes_ids, no_ids, model_dtype, device):
    from utils import process_vlm_inputs, get_image_token_positions

    base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
    N = sum(1 for v in base_preds.values())
    print(f"\n[INFO] Baseline: {base_acc:.2f}% over {N} samples", flush=True)

    # Move steering vectors to device + cast to model dtype
    steer_gpu = {}
    for l, v in steering_vecs.items():
        v_norm = v / v.norm().clamp(min=1e-8)  # unit-normalize like CAA
        steer_gpu[l] = v_norm.to(model_dtype).to(device)
        print(f"  L{l}: steering vec on device (norm=1.0, original_norm={v.norm():.4f})", flush=True)

    results_path = OUT_DIR / f"results_{mode_tag}.json"
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)

    alpha_summary = existing.get("alphas", {})

    for alpha in ALPHAS:
        alpha_key = str(alpha)
        if alpha_key in alpha_summary and alpha_summary[alpha_key].get("n", 0) > 0:
            r = alpha_summary[alpha_key]
            print(f"[SKIP] alpha={alpha}: acc={r['acc']:.2f}% Δ={r['delta']:+.2f}% n={r['n']}", flush=True)
            continue

        print(f"\n[ALPHA={alpha}] Running over {len(vsr_all)} samples...", flush=True)
        correct = total = 0

        for vi in range(len(vsr_all)):
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label  = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))

            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                _, img_end = get_image_token_positions(iids)

                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l, sv in steer_gpu.items():
                        lo    = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        sv_col = sv.unsqueeze(1)               # [2304, 1]
                        ones  = (lo @ sv_col) * 0.0 + 1.0     # (T, 1) proxy ones
                        lo   += alpha * ones * sv.unsqueeze(0)
                    logits_s = nns_model.output.logits.save()

                pred   = _predict(logits_s[0, -1, :], yes_ids, no_ids)
                total += 1
                correct += int(pred == label)

            except Exception as e:
                if total < 5:
                    print(f"  [WARN] vi={vi}: {e}", flush=True)
                continue

            if (vi + 1) % 1000 == 0:
                cur_acc = correct / max(total, 1) * 100
                print(f"  [{vi+1}/{len(vsr_all)}] acc={cur_acc:.2f}% Δ={cur_acc-base_acc:+.2f}%", flush=True)

        acc   = correct / max(total, 1) * 100
        delta = acc - base_acc
        alpha_summary[alpha_key] = {"acc": acc, "delta": delta, "n": total}
        print(f"[RESULT] alpha={alpha}: {acc:.2f}%  Δ={delta:+.2f}%  ({correct}/{total})", flush=True)

        with open(results_path, "w") as f:
            json.dump({"mode": mode_tag, "base_acc": base_acc, "alphas": alpha_summary}, f, indent=2)

        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print(f"CAA Mean Steer — {mode_tag}")
    print(f"{'='*60}")
    print(f"{'alpha':>8}  {'acc':>7}  {'Δ acc':>8}  {'N':>6}")
    print("-" * 40)
    print(f"{'base':>8}  {base_acc:>6.2f}%  {'--':>8}  {N:>6}")
    for a in ALPHAS:
        r = alpha_summary.get(str(a), {})
        if r:
            print(f"{a:>8.2f}  {r['acc']:>6.2f}%  {r['delta']:>+7.2f}%  {r['n']:>6}")
    print(f"\nSaved: {results_path}", flush=True)


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))

    if MODE not in ("FULL", "TARGETED"):
        print(f"[ERROR] Unknown MODE={MODE!r}. Use FULL or TARGETED.", flush=True)
        sys.exit(1)

    print("=" * 70)
    print(f"CAA-Style Mean Reconstruction Steering  MODE={MODE}")
    print(f"Analogous to CAA: dataset-mean delta vector, universal alpha sweep")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: compute mean steering vectors ──
    mean_vecs = compute_mean_steering_vectors()

    if MODE == "TARGETED":
        steering_vecs = project_onto_spatial_features(mean_vecs)
        mode_tag = "TARGETED"
    else:
        steering_vecs = mean_vecs
        mode_tag = "FULL"

    print(f"\n[INFO] Steering layers: {sorted(steering_vecs.keys())}", flush=True)

    # ── Load base predictions ──
    base_preds_path = PT_RECON_DIR / "base_predictions.json"
    if not base_preds_path.exists():
        print("[ERROR] base_predictions.json not found.", flush=True)
        sys.exit(1)
    with open(base_preds_path) as f:
        base_preds = json.load(f)

    # ── Load VSR ──
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    # ── Load pt-448 ──
    print(f"[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model   = NNsight(model_raw)
    tokenizer   = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # ── Step 2: inference ──
    run_inference(steering_vecs, mode_tag, vsr_all, base_preds, nns_model, model_raw,
                  processor, yes_ids, no_ids, model_dtype, device)


if __name__ == "__main__":
    main()
