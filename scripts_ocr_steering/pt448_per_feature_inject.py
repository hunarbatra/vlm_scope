#!/usr/bin/env python3
"""
Per-Feature Spatial Injection: for each of the 8 spatial features, test independently
on that feature's relation-subset only.

Mode A: inject mix-448's activation for feature F  (act_F from mix recon projection)
Mode B: inject delta (mix_act_F - pt_act_F)

Usage:
    CUDA_VISIBLE_DEVICES=0 MODE=A FEATURE_IDX=0 python3 -B pt448_per_feature_inject.py
    CUDA_VISIBLE_DEVICES=1 MODE=B FEATURE_IDX=0 python3 -B pt448_per_feature_inject.py
    # FEATURE_IDX: 0-7 (index into FEATURES list)
    # MODE=ALL_A to run all 8 features sequentially in mode A
    # MODE=ALL_B to run all 8 features sequentially in mode B
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
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_per_feature_inject")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

ALPHAS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

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


# ─────────────────────── Per-feature runner ───────────────────
def run_feature(feat, mode, vsr_all, base_preds, nns_model, model_raw, processor,
                yes_ids, no_ids, model_dtype, device):
    l       = feat["layer"]
    F       = feat["feature"]
    rels    = set(feat["relations"])
    tag     = f"L{l}_F{F}"

    # Load W_dec vector for this feature
    ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{l}.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    wdec_cpu = state["W_dec"][F].float().clone()   # [2304]
    del state
    wdec_vec = (wdec_cpu / wdec_cpu.norm().clamp(min=1e-8)).to(model_dtype).to(device)
    wdec_cpu_f32 = wdec_cpu / wdec_cpu.norm().clamp(min=1e-8)  # for CPU dot product

    # Build subset of VSR: samples whose relation is in this feature's set AND recon files exist
    N = len(vsr_all)
    subset_vis = []
    for vi in range(N):
        ex = vsr_all[vi]
        rel = str(ex.get("relation", "")).strip().lower()
        if rel not in rels:
            continue
        mix_path = MIX_RECON_DIR / f"vi_{vi:05d}.pt"
        if not mix_path.exists():
            continue
        if mode == "B":
            pt_path = PT_RECON_DIR / f"vi_{vi:05d}.pt"
            if not pt_path.exists():
                continue
        subset_vis.append(vi)

    if not subset_vis:
        print(f"  [{tag}] No valid subset samples found for mode={mode}!", flush=True)
        return None

    # Baseline accuracy on this subset from saved preds
    sub_correct = sum(base_preds[str(vi)]["correct"] for vi in subset_vis if str(vi) in base_preds)
    sub_n       = sum(1 for vi in subset_vis if str(vi) in base_preds)
    sub_base    = sub_correct / max(sub_n, 1) * 100
    print(f"\n  [{tag}] mode={mode}  subset_n={len(subset_vis)}  base={sub_base:.2f}%", flush=True)

    results = {"layer": l, "feature": F, "mode": mode,
               "relations": list(rels), "base_acc": sub_base,
               "subset_n": len(subset_vis), "alphas": {}}

    from utils import process_vlm_inputs, get_image_token_positions

    for alpha in ALPHAS:
        correct = total = 0

        for vi in subset_vis:
            mix_path = MIX_RECON_DIR / f"vi_{vi:05d}.pt"
            try:
                recon_mix = torch.load(mix_path, map_location="cpu", weights_only=True)
            except Exception:
                continue

            if l not in recon_mix:
                continue

            # Approximate per-feature activation via projection
            mix_act_scalar = (recon_mix[l].float() @ wdec_cpu_f32).item()

            if mode == "A":
                inj_scalar = mix_act_scalar
                if inj_scalar <= 0:
                    # Nothing to inject — use base pred
                    if str(vi) in base_preds:
                        total += 1
                        correct += base_preds[str(vi)]["correct"]
                    continue
            else:  # mode == "B"
                pt_path = PT_RECON_DIR / f"vi_{vi:05d}.pt"
                try:
                    recon_pt = torch.load(pt_path, map_location="cpu", weights_only=True)
                except Exception:
                    continue
                if l not in recon_pt:
                    continue
                pt_act_scalar = (recon_pt[l].float() @ wdec_cpu_f32).item()
                inj_scalar = mix_act_scalar - pt_act_scalar
                if inj_scalar == 0:
                    if str(vi) in base_preds:
                        total += 1
                        correct += base_preds[str(vi)]["correct"]
                    continue

            ex    = vsr_all[vi]
            img   = _load_image(ex)
            if img is None:
                continue
            label  = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))

            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                _, img_end = get_image_token_positions(iids)

                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    lo = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                    w_col = wdec_vec.unsqueeze(1)   # [2304, 1]
                    ones  = (lo @ w_col) * 0.0 + 1.0
                    lo   += (alpha * inj_scalar) * ones * wdec_vec.unsqueeze(0)
                    logits_s = nns_model.output.logits.save()

                pred   = _predict(logits_s[0, -1, :], yes_ids, no_ids)
                total += 1
                correct += int(pred == label)
            except Exception as e:
                if total < 3:
                    print(f"    [WARN] vi={vi}: {e}", flush=True)
                continue

        if total == 0:
            print(f"    alpha={alpha}: no valid samples!", flush=True)
            continue

        acc   = correct / total * 100
        delta = acc - sub_base
        results["alphas"][str(alpha)] = {"acc": acc, "delta": delta, "n": total}
        print(f"    alpha={alpha:>5.2f}: {acc:.2f}%  Δ={delta:+.2f}%  ({correct}/{total})", flush=True)

    del wdec_vec, wdec_cpu, wdec_cpu_f32
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ─────────────────────── Main ─────────────────────────────────
def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    sys.path.insert(0, str(Path(__file__).parent))

    raw_mode = os.environ.get("MODE", "A")
    feat_idx_str = os.environ.get("FEATURE_IDX", "ALL")

    if raw_mode in ("A", "ALL_A"):
        modes = ["A"]
    elif raw_mode in ("B", "ALL_B"):
        modes = ["B"]
    elif raw_mode == "BOTH":
        modes = ["A", "B"]
    else:
        print(f"[ERROR] Unknown MODE={raw_mode!r}. Use A/B/BOTH/ALL_A/ALL_B.", flush=True)
        sys.exit(1)

    if feat_idx_str == "ALL":
        feat_indices = list(range(len(FEATURES)))
    else:
        feat_indices = [int(x) for x in feat_idx_str.split(",")]

    print("=" * 70)
    print(f"Per-Feature Spatial Injection  modes={modes}  features={feat_indices}")
    print("=" * 70, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load base predictions ──
    base_preds_path = PT_RECON_DIR / "base_predictions.json"
    if not base_preds_path.exists():
        print("[ERROR] base_predictions.json not found. Run PHASE=2 of pt448_sae_recon_delta.py first.", flush=True)
        sys.exit(1)
    with open(base_preds_path) as f:
        base_preds = json.load(f)
    base_acc = sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
    print(f"[INFO] Global base accuracy: {base_acc:.2f}% over {len(base_preds)} samples", flush=True)

    # ── Load VSR dataset ──
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
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model   = NNsight(model_raw)
    tokenizer   = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    # ── Run per feature ──
    all_results = {}
    for mode in modes:
        for fi in feat_indices:
            feat = FEATURES[fi]
            tag  = f"mode{mode}_L{feat['layer']}_F{feat['feature']}"
            out_path = OUT_DIR / f"{tag}.json"

            if out_path.exists():
                with open(out_path) as f:
                    existing = json.load(f)
                n_done = len(existing.get("alphas", {}))
                if n_done == len(ALPHAS):
                    print(f"[SKIP] {tag}: already complete ({n_done} alphas done)", flush=True)
                    all_results[tag] = existing
                    continue

            print(f"\n{'─'*70}", flush=True)
            print(f"Feature {fi}: L{feat['layer']}/F{feat['feature']}  mode={mode}", flush=True)
            print(f"  Relations: {feat['relations']}", flush=True)

            res = run_feature(feat, mode, vsr_all, base_preds, nns_model, model_raw,
                              processor, yes_ids, no_ids, model_dtype, device)
            if res is None:
                continue

            all_results[tag] = res
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Saved: {out_path}", flush=True)

    # ── Summary table ──
    print(f"\n{'='*80}")
    print(f"Summary: Per-Feature Injection (modes={modes})")
    print(f"{'='*80}")
    hdr = f"{'L/F':>12}  {'Mode':>5}  {'N':>5}  {'Base':>6}"
    for a in ALPHAS:
        hdr += f"  {a:>6}"
    print(hdr)
    print("-" * (len(hdr) + 5))
    for mode in modes:
        for fi in feat_indices:
            feat = FEATURES[fi]
            tag  = f"mode{mode}_L{feat['layer']}_F{feat['feature']}"
            res  = all_results.get(tag, {})
            if not res:
                continue
            row = f"  L{feat['layer']}/F{feat['feature']:>6}  {mode:>5}  {res.get('subset_n', 0):>5}  {res.get('base_acc', 0):>5.1f}%"
            for a in ALPHAS:
                r = res.get("alphas", {}).get(str(a), {})
                if r:
                    row += f"  {r['delta']:>+5.1f}%"
                else:
                    row += f"  {'--':>6}"
            print(row)

    print(f"\nAll results in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
