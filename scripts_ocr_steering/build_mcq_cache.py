#!/usr/bin/env python3
"""Build MCQ-specific paired cache: pos = forward(prompt + 'A' if GT at A, etc),
neg = forward(prompt + wrong_letter). Captures the residual stream's
'I just answered correctly' vs 'I just answered wrong' direction.
"""
import os, sys, json, gc, random, string, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

MIX_MODEL  = "google/paligemma2-3b-mix-448"
SAE_ROOT   = Path("/data1/vlm_scope_sae_mix448_textonly")
PAIR_CACHE = SAE_ROOT / "analysis_ocr/paired_cache_mcq"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

NUM_LAYERS = 26


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def distort_answer(gt, rng, n_subs):
    s = list(str(gt).strip())
    if len(s) == 0: return "X"
    n_subs = min(n_subs, len(s))
    if n_subs == 0: return "Y"
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
        else: continue
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
    PAIR_CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")

    print("[INFO] Loading mix-448...", flush=True)
    proc = AutoProcessor.from_pretrained(MIX_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok = proc.tokenizer

    captured = {}
    def make_hook(l):
        def f(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1: captured[l] = h.detach()
        return f

    hooks = []
    for l in range(NUM_LAYERS):
        hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(l)))

    n_built = 0
    for si in range(1000):
        out_path = PAIR_CACHE / f"vi_{si:05d}.pt"
        if out_path.exists():
            n_built += 1; continue

        ex = ds[si]
        img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
        q = str(ex.get("question", "")).strip() or "What is the text?"
        if img is None or not gt: continue

        d1 = distort_answer(gt, rng, 2)
        d2 = distort_answer(gt, rng, 3)
        d3 = distort_answer(gt, rng, 2)
        gt_pos = rng.randrange(4)
        non_gt = [d1, d2, d3]
        choices = list(non_gt); choices.insert(gt_pos, gt); choices = choices[:4]

        letters = ["A", "B", "C", "D"]
        body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        gt_letter = letters[gt_pos]
        # Pick a random WRONG letter
        wrong_letter = rng.choice([L for L in letters if L != gt_letter])

        # pos: prompt + correct letter; neg: prompt + wrong letter
        prompts = {
            "pos": f"ocr {q}\n{body}\nAnswer: ({gt_letter}",
            "neg": f"ocr {q}\n{body}\nAnswer: ({wrong_letter}",
        }

        sample_data = {
            "gt": gt, "gt_pos": gt_pos, "gt_letter": gt_letter,
            "wrong_letter": wrong_letter, "choices": choices, "q": q,
        }
        try:
            img = img.convert("RGB")
            for variant, prompt in prompts.items():
                captured.clear()
                iids, attn, pv = process_vlm_inputs(img, prompt, proc, mdl, device=device)
                _, img_end = get_image_token_positions(iids)
                with torch.no_grad():
                    _ = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv,
                            use_cache=False)
                v_dict = {l: h[0, img_end:, :].mean(0).to(torch.bfloat16).cpu()
                          for l, h in captured.items()}
                sample_data[variant] = v_dict

            torch.save(sample_data, out_path)
            n_built += 1
            if (n_built % 25) == 0:
                print(f"  built {n_built}/1000  ({gt_letter} vs {wrong_letter})", flush=True)
        except Exception as e:
            print(f"  [si={si} ERROR] {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue

    for h in hooks:
        try: h.remove()
        except: pass
    print(f"[DONE] {n_built}/1000", flush=True)


if __name__ == "__main__":
    main()
