#!/usr/bin/env python3
"""
Repair the paired contrast cache:

Problem: The original build used a lenient substring metric for `correct`:
   `g in r or r in g`
That gives BOTH false positives (pt='k' counts as right for GT='415kJ') and
false negatives (mix='12,721' counts as wrong for GT='12721').

The false-negative case is what pollutes the CAA contrast: when the substring
metric flagged mix as wrong, we used mix's actual response as the negative.
But ~11% of those "wrong" responses are semantically correct (just formatted
differently), so we contrasted GT vs another correct answer — polluting the
"right vs wrong" direction.

Fix:
  1. Apply a STRICT correctness metric (normalized exact match: lowercase,
     strip whitespace/commas/dollar signs, then ==).
  2. Recompute `correct` for each cache entry (no model rerun — we already
     have mix's `resp` saved).
  3. For samples where strict says CORRECT but lenient said wrong (these are
     the "polluted" ones), regenerate `neg_answer = synthetic_distort(gt)` and
     re-run a single forward pass to capture clean negative hidden states.
  4. Save updated cache.

Usage:
  CUDA_VISIBLE_DEVICES=X python3 -B scripts/repair_paired_cache_ocr.py
"""
import os, sys, json, gc, random, re, string, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL  = "google/paligemma2-3b-mix-448"
SAE_ROOT   = Path("/data1/vlm_scope_sae_mix448_textonly")
PAIR_CACHE = SAE_ROOT / "analysis_ocr/paired_contrast_cache"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

NUM_LAYERS = 26


def _normalize(s):
    """Strip case, whitespace, commas, dollar signs. Used for strict equality."""
    s = str(s).lower().strip()
    s = re.sub(r"[,\$\s]+", "", s)
    return s


def _correct_strict(resp, gt):
    """Normalized exact match. No substring."""
    if not resp: return False
    return bool(gt) and _normalize(resp) == _normalize(gt)


def _correct_lenient(resp, gt):
    """Original substring metric (for diagnosing flips)."""
    if not resp: return False
    r = str(resp).strip().lower(); g = str(gt).strip().lower()
    return bool(g) and (g in r or r in g)


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def distort_answer(gt, rng):
    """Same distortion logic as build_paired_cache_ocr.py."""
    s = list(str(gt).strip())
    if len(s) == 0: return "X"
    n_subs = 1 if len(s) <= 4 else 2
    n_subs = min(n_subs, len(s))
    idxs = rng.sample(range(len(s)), n_subs)
    for idx in idxs:
        c = s[idx]
        if c.isdigit():
            pool = [d for d in "0123456789" if d != c]
        elif c.isalpha():
            pool_lower = [d for d in string.ascii_lowercase if d != c.lower()]
            new = rng.choice(pool_lower)
            s[idx] = new.upper() if c.isupper() else new
            continue
        else:
            continue
        s[idx] = rng.choice(pool)
    out = "".join(s)
    if out == str(gt).strip(): out = out + "x"
    return out


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    rng = random.Random(0)

    print("=" * 80)
    print("REPAIR PAIRED CACHE — strict correctness + clean negatives")
    print("=" * 80, flush=True)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")

    # ---------- Pass 1: audit ----------
    print("\n[PASS 1] Auditing cache with strict metric...", flush=True)
    flips_LtoS = []   # lenient=True → strict=False (false positives)
    flips_StoL = []   # lenient=False → strict=True (false negatives - POLLUTING)
    n_total = 0
    for si in range(1000):
        p = PAIR_CACHE / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        gt = d.get("gt", ""); resp = d.get("resp", "")
        if not gt: continue
        n_total += 1
        old = bool(d.get("correct", False))
        new = _correct_strict(resp, gt)
        if old and not new:
            flips_LtoS.append(si)
        elif (not old) and new:
            flips_StoL.append(si)

    print(f"  total entries audited: {n_total}", flush=True)
    print(f"  flips (lenient=correct → strict=incorrect, FALSE POSITIVES): {len(flips_LtoS)}", flush=True)
    print(f"  flips (lenient=incorrect → strict=correct, FALSE NEGATIVES — POLLUTING): {len(flips_StoL)}", flush=True)
    print(f"  → polluted samples to repair: {len(flips_StoL)}", flush=True)

    # The false-positives don't pollute the CAA contrast (we used synthetic
    # distortion as neg, which is still "wrong"). We just update the flag.
    # The false-negatives DID pollute — we used a semantically-correct response
    # as neg. We need to:
    #   1. Flip the `correct` flag to True
    #   2. Generate a fresh synthetic distortion as new neg_answer
    #   3. Rerun the negative forward pass to overwrite neg hiddens

    if len(flips_StoL) == 0 and len(flips_LtoS) == 0:
        print("\n[DONE] No flips. Cache is already clean.", flush=True)
        return

    # ---------- Pass 2: update flags only (no model needed) ----------
    print("\n[PASS 2] Updating `correct` flags...", flush=True)
    n_flag_updates = 0
    for si in flips_LtoS:
        p = PAIR_CACHE / f"vi_{si:05d}.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        d["correct"] = False
        d["correct_lenient"] = True       # bookkeeping
        torch.save(d, p)
        n_flag_updates += 1
    # Don't update flips_StoL flags here — we'll do that after recomputing negs
    print(f"  updated {n_flag_updates} false-positive flags (no negs touched)", flush=True)

    # ---------- Pass 3: rebuild polluted negatives ----------
    if len(flips_StoL) == 0:
        print("\n[DONE] No false-negatives. Cache repaired.", flush=True)
        return

    print(f"\n[PASS 3] Loading {MIX_MODEL} to rebuild {len(flips_StoL)} polluted negatives...", flush=True)
    proc = AutoProcessor.from_pretrained(MIX_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()

    captured = {}
    def make_hook(l):
        def f(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                captured[l] = h.detach()
        return f

    hooks = []
    for l in range(NUM_LAYERS):
        hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(l)))

    n_repaired = 0
    n_err = 0
    for si in flips_StoL:
        p = PAIR_CACHE / f"vi_{si:05d}.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        gt = d.get("gt", "")
        if not gt:
            n_err += 1; continue
        ex = ds[si]
        img = ex.get("image"); q = str(ex.get("question","")).strip()
        if img is None or not q:
            n_err += 1; continue

        new_neg_answer = distort_answer(gt, rng)
        try:
            img = img.convert("RGB")
            prompt = f"answer en {q}\n{new_neg_answer}"
            captured.clear()
            iids, attn, pv = process_vlm_inputs(img, prompt, proc, mdl, device=device)
            _, img_end = get_image_token_positions(iids)
            with torch.no_grad():
                _ = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv,
                        use_cache=False)
            new_neg_dict = {}
            for l, h in captured.items():
                new_neg_dict[l] = h[0, img_end:, :].mean(0).to(torch.bfloat16).cpu()

            d["neg"] = new_neg_dict
            d["neg_answer"]      = new_neg_answer
            d["neg_answer_old"]  = d.get("neg_answer", "?")   # save what was wrong
            d["correct"]         = True
            d["correct_lenient"] = False                       # lenient said wrong
            d["repaired"]        = True
            torch.save(d, p)
            n_repaired += 1
        except Exception as e:
            print(f"  [si={si} ERROR] {type(e).__name__}: {str(e)[:120]}", flush=True)
            n_err += 1
            continue

        if n_repaired % 10 == 0:
            print(f"  repaired {n_repaired}/{len(flips_StoL)} (err={n_err})", flush=True)

    for h in hooks:
        try: h.remove()
        except: pass

    print(f"\n[DONE] repaired {n_repaired} polluted negatives  err={n_err}", flush=True)
    print(f"  cache dir: {PAIR_CACHE}", flush=True)


if __name__ == "__main__":
    main()
