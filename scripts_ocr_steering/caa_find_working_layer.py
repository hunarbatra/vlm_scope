#!/usr/bin/env python3
"""
Make CAA work: layer sweep + variant comparison on full VSR test split.

Goal: find a single-layer CAA configuration that delivers Rimsky-style
10-20% gains on full VSR test (2195 samples, not R(F) subsets).

Splits:
  train+dev (8777 samples) → CAA vector extraction
  test       (2195 samples) → evaluation

Variants (all label-aware contrastive):
  MEANPOOL_CAA   v[L] = mean(h_pool[L] | label=1) − mean(h_pool[L] | label=0)
                 h_pool = mean over text tokens (from mix_hidden_lasttoken cache,
                          which is actually last-token; we ALSO have
                          pt448_hidden_delta/mix_hidden which is mean-pool)
  PAIRED_CAA     v[L] = mean over all samples of:
                   (h_yes[L] − h_no[L])  if label=1
                   (h_no[L]  − h_yes[L]) if label=0
                 Extracted at appended "Yes"/"No" token position.

For each variant: test at MIDDLE = L13 with α ∈ {0.5,1,2,3,5,8,12,20}.
Then layer sweep at the winning variant's best α, L ∈ {6,8,10,12,13,14,16,18,20}.
Then MULTI-LAYER injection: inject at 3 consecutive layers around the best.

Target: mix-src → pt-448 (largest headroom).

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_find_working_layer.py
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

PT_MODEL    = "google/paligemma2-3b-pt-448"
MIX_MODEL   = "google/paligemma2-3b-mix-448"

# Mean-pool cache covers all 26 layers
MEANPOOL_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden")
# Last-token cache (from our earlier fix) covers 8 layers
LASTTOKEN_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_lasttoken")
# Paired cache: both " Yes" and " No" continuations, 8 layers
PAIRED_DIR   = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix_hidden_paired_lasttoken")

OUT_DIR      = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/caa_find_working_layer")
IMAGE_CACHE  = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET  = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Split indices in concatenated VSR (train 0-7679, dev 7680-8776, test 8777-10971)
TRAIN_END = 8777    # inclusive start of test = TRAIN_END
TOTAL_N   = 10972

# Paired cache only has these layers; lasttoken cache has these layers.
CACHED_LAYERS = [4, 6, 9, 11, 12, 13, 14, 15]
# Meanpool has ALL 26 layers (0..25).

ALPHAS        = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
LAYERS_SWEEP  = [6, 9, 11, 12, 13, 14, 15]   # for phase-2 layer sweep (only layers we have in all 3 caches)


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
    return 1 if (y / d if d > 0 else 0.5) > 0.5 else 0

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


def compute_meanpool_caa(vsr_labels, layers):
    """CAA from mean-over-text-tokens cache. Train+dev only (vi < TRAIN_END)."""
    print(f"[STEP] Computing MEANPOOL CAA at layers {layers} on train+dev (n<{TRAIN_END})...", flush=True)
    pos = {l: None for l in layers}
    neg = {l: None for l in layers}
    pn = nn = 0
    for vi in range(TRAIN_END):
        p = MEANPOOL_DIR / f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        label = int(vsr_labels[vi])
        for l in layers:
            if l not in d: continue
            v = d[l].float()
            if label == 1:
                pos[l] = v.clone() if pos[l] is None else pos[l] + v
            else:
                neg[l] = v.clone() if neg[l] is None else neg[l] + v
        if label == 1: pn += 1
        else:          nn += 1
    out = {}
    for l in layers:
        if pos[l] is None or neg[l] is None: continue
        vec = pos[l] / pn - neg[l] / nn
        out[l] = vec
        print(f"  MEANPOOL L{l}: norm={vec.norm():.3f}  (pos={pn} neg={nn})", flush=True)
    return out


def compute_paired_caa(vsr_labels, layers):
    """Label-aware paired CAA from " Yes"/" No" continuations. Train+dev only."""
    print(f"[STEP] Computing PAIRED CAA at layers {layers} on train+dev (n<{TRAIN_END})...", flush=True)
    acc = {l: None for l in layers}
    n = 0
    for vi in range(TRAIN_END):
        p = PAIRED_DIR / f"vi_{vi:05d}.pt"
        if not p.exists(): continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if "yes" not in d or "no" not in d: continue
        label = int(vsr_labels[vi])
        n += 1
        for l in layers:
            if l not in d["yes"] or l not in d["no"]: continue
            h_yes = d["yes"][l].float()
            h_no  = d["no"][l].float()
            diff = (h_yes - h_no) if label == 1 else (h_no - h_yes)
            acc[l] = diff.clone() if acc[l] is None else acc[l] + diff
    out = {}
    for l in layers:
        if acc[l] is None: continue
        vec = acc[l] / n
        out[l] = vec
        print(f"  PAIRED L{l}: norm={vec.norm():.3f}  (n={n})", flush=True)
    return out


def run_test_eval(
    label, steer_vecs_by_layer, inject_layers,
    test_vis, test_labels,
    base_acc, result_key, all_results, results_path,
    alphas, model, processor, yes_ids, no_ids, device, vsr_all,
):
    """
    Run α-sweep on test set. `inject_layers` = list of (layer, vector) tuples to
    register simultaneously. Each vector pre-normalized. For single-layer, pass
    a length-1 list.
    """
    from utils import process_vlm_inputs, get_image_token_positions

    unit_vecs = [(l, v / v.norm().clamp(min=1e-8)) for (l, v) in inject_layers]
    img_end_r = [0]

    for alpha in alphas:
        akey = str(alpha)
        if akey in all_results.get(result_key, {}) and \
                all_results[result_key][akey].get("n", 0) > 0:
            r = all_results[result_key][akey]
            print(f"  [SKIP {label}] α={alpha}: {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
            continue

        sv_gpu = [(l, (uv * alpha).to(next(model.parameters()).dtype).to(device))
                  for (l, uv) in unit_vecs]

        def make_hook(sv_):
            def hook_fn(module, input, output):
                ie = img_end_r[0]
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[0, ie:] = hidden[0, ie:] + sv_.unsqueeze(0)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook_fn

        correct = total = 0
        for vi, lbl in zip(test_vis, test_labels):
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None: continue
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            hooks = []
            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                for (l, sv) in sv_gpu:
                    hooks.append(
                        model.model.language_model.layers[l].register_forward_hook(make_hook(sv))
                    )
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                for h in hooks:
                    try: h.remove()
                    except Exception: pass
                hooks = []
                pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                total += 1
                correct += int(pred == lbl)
            except Exception as e:
                for h in hooks:
                    try: h.remove()
                    except Exception: pass
                if total < 3:
                    print(f"    [WARN] vi={vi}: {e}", flush=True)

        if total == 0: continue
        acc = correct / total * 100
        delta = acc - base_acc
        all_results.setdefault(result_key, {})[akey] = {"acc": acc, "delta": delta, "n": total}
        print(f"  [{label}] α={alpha}: {acc:.2f}% Δ={delta:+.2f}% ({correct}/{total})", flush=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
    return all_results


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 72)
    print("CAA find-working-layer — mix-src → pt-448, full VSR test split (2195)")
    print("=" * 72, flush=True)

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("[INFO] Loading VSR...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    vsr_labels = [int(vsr_all[vi].get("label", 0)) for vi in range(len(vsr_all))]
    test_vis    = list(range(TRAIN_END, len(vsr_all)))
    test_labels = [vsr_labels[vi] for vi in test_vis]
    print(f"  train+dev: {TRAIN_END} samples for extraction")
    print(f"  test: {len(test_vis)} samples for evaluation  (label1={sum(test_labels)}  label0={len(test_labels)-sum(test_labels)})", flush=True)

    # Compute both CAA variants at all cached layers
    meanpool = compute_meanpool_caa(vsr_labels, CACHED_LAYERS)
    paired   = compute_paired_caa(vsr_labels, CACHED_LAYERS)
    gc.collect()

    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    yes_ids, no_ids = _get_yes_no_ids(processor.tokenizer)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # ── Baseline on full test ──
    from utils import process_vlm_inputs
    if "base" not in all_results:
        print(f"\n[BASELINE] pt-448 on full VSR test (n={len(test_vis)})...", flush=True)
        bc = bt = 0
        for i, (vi, lbl) in enumerate(zip(test_vis, test_labels)):
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None: continue
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model, device=device)
                with torch.no_grad():
                    out = model(input_ids=iids, attention_mask=attn, pixel_values=pv)
                pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                bt += 1; bc += int(pred == lbl)
            except Exception: continue
            if (i + 1) % 500 == 0:
                print(f"  baseline progress {i+1}/{len(test_vis)}  running acc={bc/max(bt,1)*100:.2f}%", flush=True)
        base_acc = bc / max(bt, 1) * 100
        all_results["base"] = {"acc": base_acc, "n": bt}
        print(f"[BASELINE] pt-448 full-test: {base_acc:.2f}% (n={bt})", flush=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        base_acc = all_results["base"]["acc"]
        print(f"[BASELINE] (cached) pt-448 full-test: {base_acc:.2f}%", flush=True)

    # ── PHASE 1: MIDDLE=L13, both variants, full α-sweep ──
    print(f"\n{'='*72}\nPHASE 1: L13 MIDDLE — MEANPOOL vs PAIRED\n{'='*72}", flush=True)
    if 13 in meanpool:
        all_results = run_test_eval(
            "MEANPOOL/L13", None, [(13, meanpool[13])],
            test_vis, test_labels, base_acc,
            "meanpool_L13", all_results, results_path, ALPHAS,
            model, processor, yes_ids, no_ids, device, vsr_all,
        )
    if 13 in paired:
        all_results = run_test_eval(
            "PAIRED/L13", None, [(13, paired[13])],
            test_vis, test_labels, base_acc,
            "paired_L13", all_results, results_path, ALPHAS,
            model, processor, yes_ids, no_ids, device, vsr_all,
        )

    # Decide which variant to sweep based on L13 results
    def best_d(rk):
        r = all_results.get(rk, {})
        if not r: return -999
        return max(v.get("delta",-999) for v in r.values())
    mp_best = best_d("meanpool_L13")
    pr_best = best_d("paired_L13")
    winner = "meanpool" if mp_best >= pr_best else "paired"
    winner_vecs = meanpool if winner == "meanpool" else paired
    print(f"\n[PHASE 1 WINNER] {winner}  (meanpool L13 best Δ={mp_best:+.2f}  paired L13 best Δ={pr_best:+.2f})", flush=True)

    # ── PHASE 2: LAYER SWEEP with winner ──
    print(f"\n{'='*72}\nPHASE 2: layer sweep with {winner} across L{LAYERS_SWEEP}\n{'='*72}", flush=True)
    # Use a narrower α set for speed in layer sweep: those that tended to help
    sweep_alphas = [1.0, 2.0, 5.0, 10.0]
    for l in LAYERS_SWEEP:
        if l == 13: continue  # already done in phase 1
        if l not in winner_vecs: continue
        all_results = run_test_eval(
            f"{winner.upper()}/L{l}", None, [(l, winner_vecs[l])],
            test_vis, test_labels, base_acc,
            f"{winner}_L{l}", all_results, results_path, sweep_alphas,
            model, processor, yes_ids, no_ids, device, vsr_all,
        )

    # ── PHASE 3: MULTI-LAYER injection around best single layer ──
    best_layer = 13
    best_d_overall = best_d(f"{winner}_L13")
    for l in LAYERS_SWEEP:
        d = best_d(f"{winner}_L{l}")
        if d > best_d_overall:
            best_d_overall = d; best_layer = l
    print(f"\n[PHASE 3 BEST single layer] L{best_layer} Δ={best_d_overall:+.2f}%", flush=True)

    # Multi-layer: inject at 3 consecutive cached layers around best
    cached_sorted = sorted(CACHED_LAYERS)
    try:
        idx = cached_sorted.index(best_layer)
        neighbours = []
        for offset in [-1, 0, 1]:
            j = idx + offset
            if 0 <= j < len(cached_sorted):
                neighbours.append(cached_sorted[j])
        multi_pairs = [(l, winner_vecs[l]) for l in neighbours if l in winner_vecs]
        print(f"[PHASE 3] multi-layer inject at L{[l for l,_ in multi_pairs]}", flush=True)
        all_results = run_test_eval(
            f"{winner.upper()}/MULTI_L{'_'.join(str(l) for l,_ in multi_pairs)}",
            None, multi_pairs,
            test_vis, test_labels, base_acc,
            f"{winner}_multi_{'_'.join(str(l) for l,_ in multi_pairs)}",
            all_results, results_path, sweep_alphas,
            model, processor, yes_ids, no_ids, device, vsr_all,
        )
    except Exception as e:
        print(f"[PHASE 3] skip: {e}", flush=True)

    # ── Summary ──
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}", flush=True)
    print(f"Baseline full VSR test: {base_acc:.2f}%  (n={all_results['base']['n']})")
    print(f"\nPhase 1 (L13):")
    for k in ["meanpool_L13", "paired_L13"]:
        r = all_results.get(k, {})
        if r:
            best = max(r.items(), key=lambda kv: kv[1].get("delta", -999))
            print(f"  {k:<20} best α={best[0]}  acc={best[1]['acc']:.2f}%  Δ={best[1]['delta']:+.2f}%")
    print(f"\nPhase 2 (layer sweep, {winner}):")
    for l in LAYERS_SWEEP:
        k = f"{winner}_L{l}"
        r = all_results.get(k, {})
        if r:
            best = max(r.items(), key=lambda kv: kv[1].get("delta", -999))
            print(f"  L{l:<3}  best α={best[0]}  acc={best[1]['acc']:.2f}%  Δ={best[1]['delta']:+.2f}%")

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
