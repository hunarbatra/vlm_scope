#!/usr/bin/env python3
"""
Build paired contrastive hidden-state cache for OCR-Bench (mix-448).

For each train sample (0..799):
  1. Run mix-448 generate to get its actual response
  2. is_correct = (gt in resp or resp in gt) (lowercase substring)
  3. neg_answer:
       if mix correct  -> synthetic char-substitution distortion of GT
       if mix incorrect -> mix's actual wrong response
  4. Forward "answer en {q}\\n{GT}"        -> pos hiddens (mean over text tokens)
  5. Forward "answer en {q}\\n{neg_answer}" -> neg hiddens (mean over text tokens)
  6. Save per-sample paired cache.

Output schema:
  /data1/vlm_scope_sae_mix448_textonly/analysis_ocr/paired_contrast_cache/vi_{si:05d}.pt
    {
      "pos": {layer_int: tensor[2304], ...},   # bf16
      "neg": {layer_int: tensor[2304], ...},
      "gt":  str, "neg_answer": str, "resp": str, "correct": bool,
    }

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -B scripts/build_paired_cache_ocr.py
"""
import os, sys, json, gc, random, warnings, string
from pathlib import Path
import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL  = "google/paligemma2-3b-mix-448"
SAE_ROOT   = Path("/data1/vlm_scope_sae_mix448_textonly")
PAIR_CACHE = SAE_ROOT / "analysis_ocr/paired_contrast_cache"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END      = 1000   # build paired cache for all 1000 OCR-Bench samples
NUM_LAYERS     = 26
MAX_NEW_TOKENS = 64


def _parse_gt(raw):
    """OCR-Bench 'answer' is a list of acceptable strings. Return first non-empty."""
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip():
                return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def _correct_ocr(resp, gt):
    if resp is None: return False
    r = resp.strip().lower()
    g = str(gt).strip().lower()
    return bool(g) and (g in r or r in g)


def distort_answer(gt, rng):
    """1-2 char substitutions, preserving char class. Always different from gt."""
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
    if out == str(gt).strip():
        out = out + "x"   # ensure differs
    return out


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    PAIR_CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    print("=" * 80)
    print("PAIRED CONTRAST CACHE — OCR-Bench, mix-448")
    print("=" * 80, flush=True)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")
    print(f"  N={len(ds)}, building cache for train indices 0..{TRAIN_END-1}", flush=True)

    print(f"\n[INFO] Loading {MIX_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(MIX_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tok = proc.tokenizer

    # Hook all layers to capture prefill hidden states
    captured = {}
    def make_hook(l):
        def f(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:   # prefill only
                captured[l] = h.detach()
        return f

    hooks = []
    for l in range(NUM_LAYERS):
        hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(l)))

    n_built = n_skip = n_err = 0
    n_correct_pool = n_wrong_pool = 0

    for si in range(TRAIN_END):
        out_path = PAIR_CACHE / f"vi_{si:05d}.pt"
        if out_path.exists():
            n_skip += 1
            continue

        ex = ds[si]
        img = ex.get("image")
        q   = str(ex.get("question", "")).strip()
        gt  = _parse_gt(ex.get("answer"))
        if img is None or not q or not gt:
            n_err += 1
            continue

        try:
            img = img.convert("RGB")

            # 1) Generate mix-448 response (no answer in prompt)
            iids, attn, pv = process_vlm_inputs(
                img, f"answer en {q}", proc, mdl, device=device)
            with torch.no_grad():
                out_ids = mdl.generate(
                    input_ids=iids, attention_mask=attn, pixel_values=pv,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True).strip()
            is_correct = _correct_ocr(resp, gt)

            # 2) Determine negative answer
            if is_correct:
                neg_ans = distort_answer(gt, rng)
                n_correct_pool += 1
            else:
                neg_ans = resp if resp.strip() else distort_answer(gt, rng)
                n_wrong_pool += 1

            sample_data = {"gt": gt, "neg_answer": neg_ans, "resp": resp,
                           "correct": is_correct}

            # 3) Forward pos and neg variants, capture mean-pooled hiddens
            for variant, ans in [("pos", gt), ("neg", neg_ans)]:
                prompt = f"answer en {q}\n{ans}"
                captured.clear()
                iids, attn, pv = process_vlm_inputs(
                    img, prompt, proc, mdl, device=device)
                _, img_end = get_image_token_positions(iids)
                with torch.no_grad():
                    _ = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv,
                            use_cache=False)
                # Mean pool over text tokens (after image)
                v_dict = {}
                for l, h in captured.items():
                    v_dict[l] = h[0, img_end:, :].mean(0).to(torch.bfloat16).cpu()
                sample_data[variant] = v_dict

            torch.save(sample_data, out_path)
            n_built += 1

            if (n_built + n_skip) % 25 == 0:
                print(f"  [si={si:4d}] built={n_built} skip={n_skip} err={n_err}  "
                      f"pos_correct={n_correct_pool} pos_wrong={n_wrong_pool}  "
                      f"resp='{resp[:25]}' gt='{gt[:25]}' correct={is_correct}",
                      flush=True)

        except Exception as e:
            n_err += 1
            print(f"  [si={si} ERROR] {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue

    for h in hooks:
        try: h.remove()
        except: pass

    print(f"\n[DONE] built={n_built} skip(existed)={n_skip} err={n_err}", flush=True)
    print(f"  pool: synthetic_distort={n_correct_pool} mix_wrong_resp={n_wrong_pool}", flush=True)
    print(f"  cache dir: {PAIR_CACHE}", flush=True)


if __name__ == "__main__":
    main()
