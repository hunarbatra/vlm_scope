#!/usr/bin/env python3
"""
Hidden State Delta Injection — Phase 3 (hook-based, no nnsight).

Uses pre-saved hidden states from pt448_hidden_delta.py Phase 1+2.
Injects alpha * (h_mix[l] - h_pt[l]) per sample at INJECT_LAYERS using
register_forward_hook — avoids the ~17 GB nnsight proxy overhead.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -B pt448_hidden_delta_phase3.py

ENV:
    INJECT_LAYERS  comma-separated layer indices (default: 4,6,9,11,12,14,15)
    ALPHAS         comma-separated floats (default: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.5,2.0)
    OUT_SUFFIX     suffix for results file (default: "")
"""

import os, sys, json, gc, hashlib, math, warnings
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
BASE_DIR    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta")
MIX_DIR     = BASE_DIR / "mix_hidden"
PT_DIR      = BASE_DIR / "pt_hidden"
DELTA_DIR   = BASE_DIR / "deltas"
IMAGE_CACHE = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

_default_layers = "4,6,9,11,12,14,15"
INJECT_LAYERS = [int(x) for x in os.environ.get("INJECT_LAYERS", _default_layers).split(",")]

_default_alphas = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.5,2.0"
ALPHAS = [float(x) for x in os.environ.get("ALPHAS", _default_alphas).split(",")]

OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")

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


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    print("=" * 70)
    print(f"Hidden State Delta — Phase 3 (hook-based)")
    print(f"Inject layers: {INJECT_LAYERS}")
    print(f"Alphas: {ALPHAS}")
    print("=" * 70, flush=True)

    device = "cuda:0"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    DELTA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load base predictions ──
    base_preds_path = PT_DIR / "base_predictions.json"
    if not base_preds_path.exists():
        print("[ERROR] Run Phase 2 first.", flush=True); sys.exit(1)
    with open(base_preds_path) as f:
        base_preds = json.load(f)
    base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
    print(f"[INFO] Baseline: {base_acc:.2f}% over {len(base_preds)} samples", flush=True)

    # ── Find valid samples ──
    valid_vis = []
    for vi in range(11000):
        mix_p = MIX_DIR / f"vi_{vi:05d}.pt"
        pt_p  = PT_DIR  / f"vi_{vi:05d}.pt"
        if mix_p.exists() and pt_p.exists() and str(vi) in base_preds:
            valid_vis.append(vi)
    print(f"[INFO] Valid samples: {len(valid_vis)}", flush=True)

    # ── Pre-compute delta files ──
    print(f"[INFO] Pre-computing deltas (inject layers: {INJECT_LAYERS})...", flush=True)
    n_delta = 0
    for vi in valid_vis:
        dp = DELTA_DIR / f"vi_{vi:05d}.pt"
        if dp.exists():
            try:
                ex = torch.load(dp, map_location="cpu", weights_only=True)
                if all(l in ex for l in INJECT_LAYERS):
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
        PT_MODEL, dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer   = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    results_path = BASE_DIR / f"results_hooks{OUT_SUFFIX}.json"
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
    alpha_summary = existing.get("alphas", {})

    img_end_ref = [0]  # mutable, captured by hook closures

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
                delta_cpu = torch.load(delta_path, map_location="cpu", weights_only=True)
            except Exception:
                continue

            delta_gpu = {l: v.to(model_dtype).to(device)
                         for l, v in delta_cpu.items() if l in INJECT_LAYERS}

            ex    = vsr_all[vi]
            img   = _load_image(ex)
            if img is None:
                del delta_gpu; continue
            label  = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))

            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                _, img_end = get_image_token_positions(iids)
                img_end_ref[0] = img_end

                # Register per-layer hooks
                hooks = []
                for l, dv in delta_gpu.items():
                    def make_hook(dv_=dv, alpha_=alpha):
                        def hook_fn(module, input, output):
                            ie = img_end_ref[0]
                            hidden = output[0]
                            hidden[0, ie:] = hidden[0, ie:] + alpha_ * dv_.unsqueeze(0)
                            if isinstance(output, tuple):
                                return (hidden,) + output[1:]
                            return hidden
                        return hook_fn
                    h = model_raw.model.language_model.layers[l].register_forward_hook(make_hook())
                    hooks.append(h)

                with torch.no_grad():
                    outputs = model_raw(input_ids=iids, attention_mask=attn, pixel_values=pv)

                for h in hooks:
                    h.remove()
                hooks.clear()

                pred   = _predict(outputs.logits[0, -1, :], yes_ids, no_ids)
                total += 1
                correct += int(pred == label)

            except Exception as e:
                for h in hooks:
                    try: h.remove()
                    except Exception: pass
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
            json.dump({"base_acc": base_acc, "inject_layers": INJECT_LAYERS,
                       "alphas": alpha_summary}, f, indent=2)
        torch.cuda.empty_cache(); gc.collect()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Hidden State Delta (hook-based) Results")
    print(f"Inject layers: {INJECT_LAYERS}")
    print(f"{'='*60}")
    print(f"{'alpha':>8}  {'acc':>7}  {'Δ acc':>8}  {'N':>6}")
    print("-" * 40)
    print(f"{'base':>8}  {base_acc:>6.2f}%  {'--':>8}  {len(base_preds):>6}")
    for a in ALPHAS:
        r = alpha_summary.get(str(a), {})
        if r:
            print(f"{a:>8.2f}  {r['acc']:>6.2f}%  {r['delta']:>+7.2f}%  {r['n']:>6}")
    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
