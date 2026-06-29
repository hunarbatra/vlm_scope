#!/usr/bin/env python3
"""
Direct Hidden State Delta Injection: mix-448 → pt-448.

Same principle as SAE recon delta, but uses RAW hidden states instead of
SAE reconstructions. Since mix-448 was fine-tuned FROM pt-448, their hidden
state spaces are directly comparable at each layer.

  injection[l] = alpha * (h_mix_saved[l] - h_pt_saved[l])

This captures 100% of the representational gap (vs. SAE recon which captures
only the SAE-reconstructable ~50-70%). Both approaches are per-sample adaptive.

Three phases (PHASE env var):
  PHASE=1  Extract mix-448 hidden states → mix_hidden/
  PHASE=2  Extract pt-448 hidden states + base preds → pt_hidden/
  PHASE=3  Delta injection sweep

Usage:
  CUDA_VISIBLE_DEVICES=4 PHASE=1 python3 -B pt448_hidden_delta.py
  CUDA_VISIBLE_DEVICES=5 PHASE=2 python3 -B pt448_hidden_delta.py
  CUDA_VISIBLE_DEVICES=6 PHASE=3 python3 -B pt448_hidden_delta.py
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
MIX_MODEL  = "google/paligemma2-3b-mix-448"
PT_MODEL   = "google/paligemma2-3b-pt-448"
BASE_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta")
MIX_DIR    = BASE_DIR / "mix_hidden"
PT_DIR     = BASE_DIR / "pt_hidden"
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# All 26 layers for extraction (Phases 1+2), but inject only at the 7 SAE spatial layers
# to avoid OOM during nnsight trace (26-layer trace exhausts 24GB GPU)
LAYERS        = list(range(26))          # extraction layers
INJECT_LAYERS = [4, 6, 9, 11, 12, 14, 15]  # injection layers (same as SAE recon delta)
# Fine sweep around the known SAE recon delta sweet-spot (alpha=0.5)
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]

PHASE = os.environ.get("PHASE", "1")


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

def _predict_and_margin(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item()  if no_ids  else 1e-9
    d = y + n
    p_yes = max(y / d if d > 0 else 0.5, 1e-7)
    return (1 if p_yes > 0.5 else 0), math.log(p_yes / (1 - p_yes + 1e-9))

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
    except Exception: return None


def _extract_hidden_states(model_name, out_dir, save_base_preds=False):
    """Run model forward passes, save mean-over-text-tokens hidden state per layer per sample."""
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    out_dir.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {model_name}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer) if save_base_preds else (set(), set())

    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    print(f"[INFO] VSR total: {N}  Saving to: {out_dir}", flush=True)

    base_preds = {}
    base_preds_path = out_dir / "base_predictions.json"
    if save_base_preds and base_preds_path.exists():
        with open(base_preds_path) as f:
            base_preds = json.load(f)

    n_done = n_skipped = n_failed = 0

    for vi in range(N):
        out_path = out_dir / f"vi_{vi:05d}.pt"
        if out_path.exists() and (not save_base_preds or str(vi) in base_preds):
            n_skipped += 1
            continue

        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            n_failed += 1; continue

        label  = int(ex.get("label", 0))
        prompt = _build_vsr_prompt(str(ex.get("caption", "")))

        try:
            iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
            _, img_end = get_image_token_positions(iids)

            saved_list = []
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in LAYERS:
                    saved_list.append(nns_model.model.language_model.layers[l].output[0].save())
                if save_base_preds:
                    logits_s = nns_model.output.logits.save()

            # Save mean-over-text-tokens hidden state for each layer
            h_dict = {}
            for idx, l in enumerate(LAYERS):
                h_gpu = saved_list[idx]   # [1, T, 2304]
                # mean over text positions only (after image tokens)
                h_dict[l] = h_gpu[0, img_end:, :].mean(0).to(torch.bfloat16).cpu()  # [2304]
                del h_gpu

            if not out_path.exists():
                torch.save(h_dict, out_path)

            if save_base_preds:
                pred, margin = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
                base_preds[str(vi)] = {"pred": pred, "margin": margin,
                                       "label": label, "correct": int(pred == label)}
            n_done += 1

        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"  [WARN] vi={vi}: {e}", flush=True)

        if (vi + 1) % 500 == 0 or vi == N - 1:
            print(f"  [{vi+1}/{N}] done={n_done} skipped={n_skipped} failed={n_failed}", flush=True)
            if save_base_preds and base_preds:
                with open(base_preds_path, "w") as f:
                    json.dump(base_preds, f)

    if save_base_preds:
        with open(base_preds_path, "w") as f:
            json.dump(base_preds, f)
        base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
        print(f"[DONE] Base acc = {base_acc:.2f}% over {len(base_preds)} samples", flush=True)

    print(f"[DONE] {model_name}: done={n_done} skipped={n_skipped} failed={n_failed}", flush=True)


def phase3_inject():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    print("=" * 70)
    print("PHASE 3: Hidden State Delta Injection Sweep")
    print("=" * 70, flush=True)

    device = "cuda:0"
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Load base predictions
    base_preds_path = PT_DIR / "base_predictions.json"
    if not base_preds_path.exists():
        print("[ERROR] Run PHASE=2 first.", flush=True); sys.exit(1)
    with open(base_preds_path) as f:
        base_preds = json.load(f)
    base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
    print(f"[INFO] Baseline: {base_acc:.2f}% over {len(base_preds)} samples", flush=True)

    # Identify samples where both mix and pt hidden states are saved
    valid_vis = []
    for vi in range(11000):
        mix_p = MIX_DIR / f"vi_{vi:05d}.pt"
        pt_p  = PT_DIR  / f"vi_{vi:05d}.pt"
        if mix_p.exists() and pt_p.exists() and str(vi) in base_preds:
            valid_vis.append(vi)
    print(f"[INFO] Valid samples (both mix+pt saved): {len(valid_vis)}", flush=True)

    if not valid_vis:
        print("[ERROR] No valid samples. Run PHASE=1 and PHASE=2 first.", flush=True); sys.exit(1)

    # Pre-compute and cache delta files to speed up Phase 3 sweeps
    DELTA_DIR = BASE_DIR / "deltas"
    DELTA_DIR.mkdir(exist_ok=True)
    print(f"[INFO] Pre-computing deltas (inject layers only: {INJECT_LAYERS}) for {len(valid_vis)} samples...", flush=True)
    n_delta = 0
    for vi in valid_vis:
        dp = DELTA_DIR / f"vi_{vi:05d}.pt"
        if dp.exists():
            # Verify it has all inject layers; recompute if stale (e.g. previously had 26 layers)
            try:
                existing = torch.load(dp, map_location="cpu", weights_only=True)
                if all(l in existing for l in INJECT_LAYERS):
                    n_delta += 1; continue
            except Exception:
                pass
        mix_h = torch.load(MIX_DIR / f"vi_{vi:05d}.pt", map_location="cpu", weights_only=True)
        pt_h  = torch.load(PT_DIR  / f"vi_{vi:05d}.pt", map_location="cpu", weights_only=True)
        delta = {l: (mix_h[l].float() - pt_h[l].float()).to(torch.bfloat16)
                 for l in INJECT_LAYERS if l in mix_h and l in pt_h}
        torch.save(delta, dp)
        n_delta += 1
    print(f"[INFO] Deltas ready: {n_delta}", flush=True)

    inject_vis = [vi for vi in valid_vis if (DELTA_DIR / f"vi_{vi:05d}.pt").exists()]
    print(f"[INFO] Injecting at layers: {INJECT_LAYERS}", flush=True)

    # Load VSR dataset
    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    # Load pt-448
    print(f"[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model   = NNsight(model_raw)
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    results_path = BASE_DIR / "results.json"
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
    alpha_summary = existing.get("alphas", {})

    for alpha in ALPHAS:
        akey = str(alpha)
        if akey in alpha_summary and alpha_summary[akey].get("n", 0) >= len(inject_vis) * 0.9:
            r = alpha_summary[akey]
            print(f"[SKIP] alpha={alpha}: {r['acc']:.2f}% Δ={r['delta']:+.2f}% n={r['n']}", flush=True)
            continue

        print(f"\n[ALPHA={alpha}] Injecting over {len(inject_vis)} samples...", flush=True)
        correct = total = 0

        for step_i, vi in enumerate(inject_vis):
            delta_path = DELTA_DIR / f"vi_{vi:05d}.pt"
            try:
                delta = torch.load(delta_path, map_location="cpu", weights_only=True)
            except Exception:
                continue

            # Move only inject-layer deltas to device
            delta_gpu = {l: v.to(model_dtype).to(device)
                         for l, v in delta.items() if l in INJECT_LAYERS}

            ex    = vsr_all[vi]
            img   = _load_image(ex)
            if img is None:
                del delta_gpu; continue
            label  = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))

            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                _, img_end = get_image_token_positions(iids)

                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l, dv in delta_gpu.items():
                        lo    = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        dv_col = dv.unsqueeze(1)                    # [2304, 1]
                        ones  = (lo @ dv_col) * 0.0 + 1.0          # (T, 1) proxy ones
                        lo   += alpha * ones * dv.unsqueeze(0)
                    logits_s = nns_model.output.logits.save()

                pred, _ = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
                total  += 1
                correct += int(pred == label)

            except Exception as e:
                if total < 5:
                    print(f"  [WARN] vi={vi}: {e}", flush=True)
            finally:
                del delta_gpu

            if (step_i + 1) % 1000 == 0:
                cur = correct / max(total, 1) * 100
                print(f"  [{step_i+1}/{len(inject_vis)}] acc={cur:.2f}% Δ={cur-base_acc:+.2f}%", flush=True)

        acc   = correct / max(total, 1) * 100
        delta_val = acc - base_acc
        alpha_summary[akey] = {"acc": acc, "delta": delta_val, "n": total}
        print(f"[RESULT] alpha={alpha}: {acc:.2f}%  Δ={delta_val:+.2f}%  ({correct}/{total})", flush=True)

        with open(results_path, "w") as f:
            json.dump({"base_acc": base_acc, "alphas": alpha_summary}, f, indent=2)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*60}\nHidden State Delta Injection Results\n{'='*60}")
    print(f"{'alpha':>8}  {'acc':>7}  {'Δ acc':>8}  {'N':>6}")
    print("-" * 40)
    print(f"{'base':>8}  {base_acc:>6.2f}%  {'--':>8}  {len(base_preds):>6}")
    for a in ALPHAS:
        r = alpha_summary.get(str(a), {})
        if r:
            print(f"{a:>8.2f}  {r['acc']:>6.2f}%  {r['delta']:>+7.2f}%  {r['n']:>6}")
    print(f"\nResults: {results_path}", flush=True)


def main():
    if PHASE == "1":
        print("=" * 70)
        print("PHASE 1: Extracting mix-448 hidden states (all 26 layers)")
        print("=" * 70, flush=True)
        MIX_DIR.mkdir(parents=True, exist_ok=True)
        _extract_hidden_states(MIX_MODEL, MIX_DIR, save_base_preds=False)
    elif PHASE == "2":
        print("=" * 70)
        print("PHASE 2: Extracting pt-448 hidden states + base predictions")
        print("=" * 70, flush=True)
        PT_DIR.mkdir(parents=True, exist_ok=True)
        _extract_hidden_states(PT_MODEL, PT_DIR, save_base_preds=True)
    elif PHASE == "3":
        phase3_inject()
    else:
        print(f"[ERROR] Unknown PHASE={PHASE}. Use 1, 2, or 3.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
