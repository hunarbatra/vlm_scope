#!/usr/bin/env python3
"""
MathVerse Step 6: Causal ablation to find top 3-5 math features.

Loads fisher_ranked.json, takes top N candidates, ablates each (zero out
the SAE feature's contribution in the hidden state) on the test split,
measures performance drop. Top features by drop become the steering targets.

Ablation method: for each feature F at layer L,
  h_ablated = h - proj(h, W_dec[F])  (project out the feature direction)

Output:
  analysis_mathverse/ablation_results.json
    per-feature: {base_acc, ablated_acc, drop, layer, feature}

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 -u mathverse_step3_ablation.py
"""
import os, sys, json, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL  = "google/paligemma2-3b-mix-448"
SAE_ROOT   = Path("/data1/vlm_scope_sae_mix448_textonly")
RANKED_F   = SAE_ROOT / "analysis_mathverse/fisher_ranked.json"
CORR_PATH  = SAE_ROOT / "analysis_mathverse/correctness.json"
CKPT_DIR   = SAE_ROOT / "checkpoints"
OUT_PATH   = SAE_ROOT / "analysis_mathverse/ablation_results.json"

TRAIN_END  = 344
TOP_N      = 40    # ablate top-40 Fisher features → pick best 3-5 by drop
MIN_RF     = 3     # min R(F)∩test size

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def _get_choice_ids(tok):
    choice_ids = {}
    for letter in "ABCD":
        ids_ = set()
        for form in [letter, f" {letter}", f"({letter})", f" ({letter})"]:
            try:
                t = tok.encode(form, add_special_tokens=False)
                if t: ids_.add(t[0])
            except: pass
        choice_ids[letter] = ids_
    return choice_ids


def _predict_mcq(logits, choice_ids):
    p = torch.softmax(logits.float(), dim=-1)
    scores = {l: sum(p[i].item() for i in ids_) for l, ids_ in choice_ids.items()}
    return max(scores, key=scores.get)


def _parse_gt(s):
    import re
    m = re.search(r'([A-D])', str(s))
    return m.group(1) if m else None


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions, initialize_jumprelu_sae

    device = "cuda:0"

    ranked = json.load(open(RANKED_F))
    corr_data = json.load(open(CORR_PATH))
    correct = {int(k): v for k, v in corr_data["correct"].items()}

    N = corr_data["N"]
    test_idx = list(range(TRAIN_END, N))

    print(f"[INFO] Loading MathVerse...", flush=True)
    ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")

    print(f"[INFO] Loading mix-448...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    choice_ids = _get_choice_ids(proc.tokenizer)
    dtype = next(model.parameters()).dtype

    # Load existing results
    all_results = json.load(open(OUT_PATH)) if OUT_PATH.exists() else {}

    # Global base accuracy on test
    if "base" not in all_results:
        print("[INFO] Computing global test baseline...", flush=True)
        c = t = 0
        for si in test_idx:
            ex = ds[si]
            img = ex.get("image")
            if img is None: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(img, f"answer en {ex['prompt']}", proc, model, device=device)
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                pred = _predict_mcq(out.logits[0, -1, :], choice_ids)
                gt   = _parse_gt(ex.get("answer", ""))
                if gt:
                    t += 1; c += int(pred == gt)
            except Exception: pass
        base_acc = c / max(t, 1) * 100
        all_results["base"] = {"acc": base_acc, "n": t}
        with open(OUT_PATH, "w") as f: json.dump(all_results, f, indent=2)
        print(f"  Base test acc: {base_acc:.2f}% (n={t})", flush=True)
    else:
        base_acc = all_results["base"]["acc"]
        print(f"  [SKIP] Base={base_acc:.2f}%", flush=True)

    # Ablate top-N features from Fisher ranking
    candidates = ranked[:TOP_N]
    print(f"[INFO] Ablating top {len(candidates)} Fisher candidates on test...", flush=True)

    for rank_i, feat in enumerate(candidates):
        layer, feature = feat["layer"], feat["feature"]
        key = feat["key"]

        if key in all_results:
            r = all_results[key]
            print(f"  [SKIP] {key}: drop={r.get('drop', 0):+.2f}%", flush=True)
            continue

        ckpt = CKPT_DIR / f"text-only_layer_{layer}.pt"
        if not ckpt.exists():
            print(f"  [SKIP] {key}: no checkpoint", flush=True)
            continue

        # Load W_dec direction for this feature
        ckpt_d = torch.load(ckpt, map_location="cpu", weights_only=True)
        w_dec = ckpt_d["W_dec"][feature].float().to(device)  # [2304]
        w_dec_unit = w_dec / w_dec.norm().clamp(min=1e-8)
        del ckpt_d

        # Hook: project out feature direction at this layer
        captured_ie = [0]
        def make_ablate_hook(w_unit):
            def _hook(mod, inp, out):
                ie = captured_ie[0]
                h = out[0] if isinstance(out, tuple) else out
                # project out on text tokens only
                txt = h[0, ie:, :].float()
                proj = (txt @ w_unit) .unsqueeze(1) * w_unit.unsqueeze(0)
                txt_new = (txt - proj).to(h.dtype)
                h[0, ie:, :] = txt_new
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return _hook

        handle = model.model.language_model.layers[layer].register_forward_hook(
            make_ablate_hook(w_dec_unit))

        c = t = 0
        rf_size = 0
        try:
            for si in test_idx:
                ex = ds[si]
                img = ex.get("image")
                if img is None: continue
                try:
                    img = img.convert("RGB")
                    iids, attn, pv = process_vlm_inputs(
                        img, f"answer en {ex['prompt']}", proc, model, device=device)
                    _, captured_ie[0] = get_image_token_positions(iids)
                    with torch.no_grad():
                        out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                    pred = _predict_mcq(out.logits[0, -1, :], choice_ids)
                    gt   = _parse_gt(ex.get("answer", ""))
                    if gt:
                        t += 1; c += int(pred == gt)
                    rf_size += 1
                except Exception: pass
        finally:
            handle.remove()
            del w_dec, w_dec_unit
            torch.cuda.empty_cache()

        acc  = c / max(t, 1) * 100
        drop = acc - base_acc
        all_results[key] = {
            "layer": layer, "feature": feature,
            "acc": acc, "drop": drop, "n": t,
            "fire_corr": feat["fire_corr"], "fire_incorr": feat["fire_incorr"],
            "fisher_score": feat["fisher_score"], "or": feat["or"],
            "rank": rank_i
        }
        with open(OUT_PATH, "w") as f: json.dump(all_results, f, indent=2)
        print(f"  [{rank_i+1:02d}/{len(candidates)}] {key}: ablated={acc:.2f}%  drop={drop:+.2f}%", flush=True)

    # Summary — top features by performance drop
    drops = [(k, v) for k, v in all_results.items() if k != "base" and "drop" in v]
    drops.sort(key=lambda x: x[1]["drop"])  # most negative drop = most impactful ablation

    print(f"\n{'='*70}")
    print(f"TOP FEATURES BY ABLATION DROP  (base={base_acc:.2f}%)")
    print(f"{'='*70}")
    print(f"  {'Key':<14} {'Layer':>5} {'Drop':>8}  {'OR':>6}  {'Fisher':>7}  fires(c/i)")
    for key, r in drops[:10]:
        print(f"  {key:<14} L{r['layer']:<4} {r['drop']:+7.2f}%  {r['or']:6.3f}  {r['fisher_score']:7.2f}  "
              f"{r['fire_corr']}c/{r['fire_incorr']}i")
    print(f"\nTop 5 steering targets (biggest drop):")
    for key, r in drops[:5]:
        print(f"  {key}")
    print(f"\nResults: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
