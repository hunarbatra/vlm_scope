#!/usr/bin/env python3
"""
Spatial Feature Injection: inject only the 8 identified spatial features into pt-448.

Core idea:
  The mix-448 SAE reconstruction files (saved in Phase 1 of pt448_sae_recon_delta.py)
  store the full SAE reconstruction vector recon = sum_F(act_F * W_dec[F]) averaged
  over text tokens.  We cannot recover individual act_F from this, but we can
  approximate the contribution of a single feature F via projection:

      mix_act_F ≈ recon_mix[l] @ W_dec[F]   (exact iff features orthogonal)

  Then we inject only those 8 feature contributions, not all 16,384.

Two modes (set via MODE env var):
  MODE=A  Inject mix-448 spatial feature contributions into pt-448 (unconditional).
          For each feature F at layer l:
              if mix_act_F > 0:
                  hidden[img_end:] += alpha * mix_act_F * W_dec[F]

  MODE=B  Inject the delta (mix - pt) for each spatial feature only.
          For each feature F at layer l:
              delta_F = mix_act_F - pt_act_F
              hidden[img_end:] += alpha * delta_F * W_dec[F]

Usage:
    CUDA_VISIBLE_DEVICES=7 MODE=A python3 -B pt448_spatial_feature_inject.py
    CUDA_VISIBLE_DEVICES=7 MODE=B python3 -B pt448_spatial_feature_inject.py
"""

import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# ─────────────────────────── Config ───────────────────────────

PT_MODEL       = "google/paligemma2-3b-pt-448"
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
MIX_RECON_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta/mix_reconstructions")
PT_RECON_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta/pt_reconstructions")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_spatial_feature_inject")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MODE   = os.environ.get("MODE", "A")   # "A" = inject mix only, "B" = inject delta
ALPHAS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# 8 canonical spatial features (layer, feature_id)
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

# Unique layers needed
_FEAT_LAYERS = sorted({f["layer"] for f in SPATIAL_FEATURES})


# ─────────────────────────── Helpers ───────────────────────────

def _build_vsr_prompt(statement):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\nAnswer:"
    )


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids  -= overlap
    return yes_ids, no_ids


def _predict_and_margin(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item()  if no_ids  else 1e-9
    d = y + n
    p_yes = max(y / d if d > 0 else 0.5, 1e-7)
    p_no  = max(1.0 - p_yes, 1e-7)
    return (1 if p_yes > 0.5 else 0), math.log(p_yes / p_no)


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    h  = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists():
            return Image.open(cp).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None


def _load_wdec_vectors(device):
    """Load W_dec[feature_id] for each of the 8 spatial features.

    Returns a nested dict: wdec[layer][feature_id] → [2304] float32 tensor on device,
    unit-normalised (W_dec rows are already unit norm from training, but we
    re-normalise to be safe).
    """
    # Group features by layer to avoid loading the same checkpoint twice
    layer_to_feats = {}
    for sf in SPATIAL_FEATURES:
        l, F = sf["layer"], sf["feature"]
        layer_to_feats.setdefault(l, []).append(F)

    wdec = {}
    for l, feat_ids in layer_to_feats.items():
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{l}.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        W_dec = state["W_dec"].float()  # [16384, 2304]
        wdec[l] = {}
        for F in feat_ids:
            vec = W_dec[F].clone()                             # [2304]
            vec = vec / vec.norm().clamp(min=1e-8)             # unit-normalise
            wdec[l][F] = vec.to(device)
        del state, W_dec
        print(f"  [W_dec] Loaded layer {l}: features {feat_ids}", flush=True)

    return wdec


# ─────────────────────────── Main ───────────────────────────

def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    if MODE not in ("A", "B"):
        print(f"[ERROR] Unknown MODE={MODE!r}. Use MODE=A or MODE=B.", flush=True)
        sys.exit(1)

    print("=" * 80)
    print(f"Spatial Feature Injection — MODE={MODE}")
    print("=" * 80, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Load W_dec vectors ──
    print("[INFO] Loading W_dec vectors for 8 spatial features...", flush=True)
    wdec = _load_wdec_vectors(device)  # wdec[layer][feature] = [2304] float32 on device

    # ── Load pt-448 ──
    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model   = NNsight(model_raw)
    tokenizer   = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # ── Load VSR dataset ──
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    print(f"[INFO] VSR total samples: {N}", flush=True)

    # ── Baseline accuracy from saved pt_reconstructions/base_predictions.json ──
    base_preds_path = PT_RECON_DIR / "base_predictions.json"
    base_preds = {}
    if base_preds_path.exists():
        with open(base_preds_path) as f:
            base_preds = json.load(f)
        base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
        print(f"[INFO] Baseline acc (from saved base_predictions): {base_acc:.2f}% over {len(base_preds)} samples.", flush=True)
    else:
        print("[WARN] base_predictions.json not found — will compute baseline on the fly.", flush=True)
        base_acc = None

    # ── Determine which samples to run ──
    # Only process samples where mix reconstruction file exists
    # (and pt reconstruction file too, if MODE=B)
    valid_vis = []
    for vi in range(N):
        mix_path = MIX_RECON_DIR / f"vi_{vi:05d}.pt"
        if not mix_path.exists():
            continue
        if MODE == "B":
            pt_path = PT_RECON_DIR / f"vi_{vi:05d}.pt"
            if not pt_path.exists():
                continue
        valid_vis.append(vi)

    print(f"[INFO] Valid samples (mix recon present{', pt recon present' if MODE == 'B' else ''}): {len(valid_vis)}", flush=True)

    if not valid_vis:
        print("[ERROR] No valid samples found. Run PHASE=1 (and PHASE=2 for MODE=B) of pt448_sae_recon_delta.py first.", flush=True)
        sys.exit(1)

    # Compute baseline over the valid_vis subset (use saved preds if available)
    if base_preds:
        sub_correct = sum(base_preds[str(vi)]["correct"] for vi in valid_vis if str(vi) in base_preds)
        sub_n       = sum(1 for vi in valid_vis if str(vi) in base_preds)
        sub_base_acc = sub_correct / max(sub_n, 1) * 100
        print(f"[INFO] Baseline acc on valid subset: {sub_base_acc:.2f}% over {sub_n} samples.", flush=True)
    else:
        sub_base_acc = None
        sub_n = len(valid_vis)

    # ── Intermediate results file ──
    results_path = OUT_DIR / f"results_MODE{MODE}.json"
    existing_results = {}
    if results_path.exists():
        with open(results_path) as f:
            existing_results = json.load(f)
        print(f"[INFO] Loaded existing results from {results_path}", flush=True)

    # ── Alpha sweep ──
    alpha_summary = existing_results.get("alphas", {})

    for alpha in ALPHAS:
        alpha_key = str(alpha)
        if alpha_key in alpha_summary and alpha_summary[alpha_key].get("n", 0) > 0:
            r = alpha_summary[alpha_key]
            print(
                f"[SKIP] alpha={alpha}: acc={r['acc']:.2f}% Δ={r['delta_acc']:+.2f}% n={r['n']}",
                flush=True,
            )
            continue

        print(f"\n[INJECT] MODE={MODE} alpha={alpha} over {len(valid_vis)} samples...", flush=True)
        correct = total = 0

        for step_i, vi in enumerate(valid_vis):
            # Load reconstruction files
            mix_path = MIX_RECON_DIR / f"vi_{vi:05d}.pt"
            if not mix_path.exists():
                continue
            try:
                recon_mix = torch.load(mix_path, map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"  [WARN] vi={vi}: failed loading mix recon: {e}", flush=True)
                continue

            recon_pt = None
            if MODE == "B":
                pt_path = PT_RECON_DIR / f"vi_{vi:05d}.pt"
                if not pt_path.exists():
                    continue
                try:
                    recon_pt = torch.load(pt_path, map_location="cpu", weights_only=True)
                except Exception as e:
                    print(f"  [WARN] vi={vi}: failed loading pt recon: {e}", flush=True)
                    continue

            # Compute per-feature activation scalars (approximate projection)
            # mix_act_F ≈ recon_mix[l] @ W_dec[F]
            # delta_F   = mix_act_F - pt_act_F
            feat_acts = {}  # key: (layer, feature) → scalar injection amount
            for sf in SPATIAL_FEATURES:
                l, F = sf["layer"], sf["feature"]
                if l not in recon_mix:
                    continue
                w = wdec[l][F]  # [2304] float32 on device

                mix_act = recon_mix[l].to(device).float() @ w  # scalar tensor

                if MODE == "A":
                    inj_scalar = mix_act.item()
                    if inj_scalar <= 0:
                        continue  # only inject when mix-448 was actually activating
                else:  # MODE == "B"
                    if l not in recon_pt:
                        continue
                    pt_act = recon_pt[l].to(device).float() @ w
                    inj_scalar = (mix_act - pt_act).item()
                    if inj_scalar == 0:
                        continue

                feat_acts[(l, F)] = inj_scalar

            if not feat_acts:
                # Nothing to inject for this sample — use base prediction if available
                if str(vi) in base_preds:
                    pred = base_preds[str(vi)]["pred"]
                    label = int(vsr_all[vi].get("label", 0))
                    total += 1
                    correct += int(pred == label)
                continue

            ex    = vsr_all[vi]
            img   = _load_image(ex)
            if img is None:
                continue
            label  = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))

            try:
                iids, attn, pv = process_vlm_inputs(
                    img, prompt, processor, model_raw, device=device
                )
                _, img_end = get_image_token_positions(iids)

                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for (l, F), inj_scalar in feat_acts.items():
                        w     = wdec[l][F]              # [2304] float32 on device
                        w_col = w.unsqueeze(1)          # [2304, 1]
                        lo    = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        # Proxy trick: (T, 2304) @ (2304, 1) → (T, 1); zero it; +1 → ones proxy
                        ones  = (lo @ w_col) * 0.0 + 1.0   # (T, 1) proxy ones
                        # Inject: add alpha * inj_scalar * w to every text token
                        lo    += (alpha * inj_scalar) * ones * w.unsqueeze(0)
                    logits_s = nns_model.output.logits.save()

                pred, _ = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
                total   += 1
                correct += int(pred == label)

            except Exception as e:
                if total < 5:
                    print(f"  [WARN] vi={vi} alpha={alpha}: {e}", flush=True)
                continue

            if (step_i + 1) % 500 == 0:
                cur_acc = correct / max(total, 1) * 100
                ref_acc = sub_base_acc if sub_base_acc is not None else 0.0
                print(
                    f"  [{step_i+1}/{len(valid_vis)}] acc={cur_acc:.2f}%"
                    f" ({correct}/{total}) Δ={cur_acc - ref_acc:+.2f}%",
                    flush=True,
                )

        inj_acc   = correct / max(total, 1) * 100
        ref_acc   = sub_base_acc if sub_base_acc is not None else inj_acc
        delta_acc = inj_acc - ref_acc
        alpha_summary[alpha_key] = {
            "acc":       inj_acc,
            "delta_acc": delta_acc,
            "n":         total,
        }

        print(
            f"[RESULT] alpha={alpha}: acc={inj_acc:.2f}% Δ={delta_acc:+.2f}%"
            f" ({correct}/{total})",
            flush=True,
        )

        # Save after each alpha
        save_obj = {
            "mode":     MODE,
            "base_acc": sub_base_acc if sub_base_acc is not None else base_acc,
            "n_base":   sub_n,
            "alphas":   alpha_summary,
        }
        with open(results_path, "w") as f:
            json.dump(save_obj, f, indent=2)

        torch.cuda.empty_cache()
        gc.collect()

    # ── Final summary table ──
    print(f"\n{'='*80}")
    print(f"Spatial Feature Injection Results — MODE={MODE}")
    print(f"{'='*80}")
    ref_acc = sub_base_acc if sub_base_acc is not None else base_acc
    print(f"{'Alpha':>8}  {'Acc':>7}  {'Δ Acc':>8}  {'N':>6}")
    print("-" * 40)
    if ref_acc is not None:
        print(f"{'base':>8}  {ref_acc:>6.2f}%  {'--':>8}  {sub_n:>6}")
    for alpha in ALPHAS:
        r = alpha_summary.get(str(alpha), {})
        if r:
            print(
                f"{alpha:>8.2f}  {r['acc']:>6.2f}%  {r['delta_acc']:>+7.2f}%  "
                f"{r['n']:>6}"
            )

    print(f"\nResults saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
