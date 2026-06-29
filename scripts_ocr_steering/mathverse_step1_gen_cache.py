#!/usr/bin/env python3
"""
MathVerse Steps 1-2: Run mix-448 on all 430 testmini samples.

For each sample:
- Forward pass with hooks at all 26 layers during prefill
- Evaluate correctness via MCQ logit comparison (A/B/C/D)
- Save mean-over-text-tokens hidden states + correctness label

Output:
  analysis_mathverse/mix_hidden/vi_{si:05d}.pt
    {0: tensor([2304], bfloat16), ..., 25: ..., "correct": bool}
  analysis_mathverse/correctness.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -u mathverse_step1_gen_cache.py
"""
import os, sys, json, re, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MIX_MODEL  = "google/paligemma2-3b-mix-448"
SAE_ROOT   = Path("/data1/vlm_scope_sae_mix448_textonly")
OUT_DIR    = SAE_ROOT / "analysis_mathverse/mix_hidden"
CORR_PATH  = SAE_ROOT / "analysis_mathverse/correctness.json"
NUM_LAYERS = 26
TRAIN_END  = 344  # 0..343 train, 344..429 test

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def _parse_gt(s):
    m = re.search(r'([A-D])', str(s))
    return m.group(1) if m else None


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


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CORR_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading MathVerse testmini...", flush=True)
    ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")
    N = len(ds)
    print(f"  {N} samples (train=0..{TRAIN_END-1}, test={TRAIN_END}..{N-1})", flush=True)

    print("[INFO] Loading mix-448...", flush=True)
    proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    choice_ids = _get_choice_ids(proc.tokenizer)
    print(f"  mix-448 loaded. Choice IDs: { {l: sorted(v) for l,v in choice_ids.items()} }", flush=True)

    # Load existing correctness if resuming
    correctness = {}
    if CORR_PATH.exists():
        try:
            d = json.load(open(CORR_PATH))
            correctness = {int(k): v for k, v in d.get("correct", {}).items()}
            print(f"  Resuming: {len(correctness)} existing labels found", flush=True)
        except: pass

    for si in range(N):
        out_path = OUT_DIR / f"vi_{si:05d}.pt"
        if out_path.exists() and si in correctness:
            continue

        ex = ds[si]
        img = ex.get("image")
        if img is None:
            correctness[si] = False
            continue

        try:
            img = img.convert("RGB")
            prompt = f"answer en {ex['prompt']}"
            iids, attn, pv = process_vlm_inputs(img, prompt, proc, model, device=device)
            _, img_end = get_image_token_positions(iids)

            captured = {}
            handles = []
            for l in range(NUM_LAYERS):
                def make_hook(lid):
                    def _hook(mod, inp, out):
                        x = out[0] if isinstance(out, tuple) else out
                        if x.shape[1] > 1:
                            captured[lid] = x.detach()
                    return _hook
                handles.append(model.model.language_model.layers[l].register_forward_hook(make_hook(l)))

            with torch.inference_mode():
                out_fwd = model(input_ids=iids, attention_mask=attn, pixel_values=pv)

            for h in handles:
                h.remove()

            pred    = _predict_mcq(out_fwd.logits[0, -1, :], choice_ids)
            gt      = _parse_gt(ex.get("answer", ""))
            correct = (pred == gt) if gt else False
            correctness[si] = correct

            save_dict = {"correct": correct}
            for l in range(NUM_LAYERS):
                if l in captured:
                    save_dict[l] = captured[l][0, img_end:, :].mean(0).to(torch.bfloat16).cpu()
            torch.save(save_dict, out_path)

        except Exception as e:
            print(f"  [WARN] si={si}: {e}", flush=True)
            correctness[si] = False

        if (si + 1) % 50 == 0:
            n_c = sum(1 for v in correctness.values() if v)
            print(f"  {si+1}/{N}  correct={n_c}/{len(correctness)} ({100*n_c/max(len(correctness),1):.1f}%)", flush=True)
            # Save interim correctness
            _save_correctness(correctness, N, TRAIN_END, CORR_PATH)

    _save_correctness(correctness, N, TRAIN_END, CORR_PATH)
    n_ct = sum(1 for si, v in correctness.items() if si < TRAIN_END and v)
    n_ce = sum(1 for si, v in correctness.items() if si >= TRAIN_END and v)
    print(f"\n[DONE] train={n_ct}/{TRAIN_END} ({100*n_ct/TRAIN_END:.1f}%)"
          f"  test={n_ce}/{N-TRAIN_END} ({100*n_ce/(N-TRAIN_END):.1f}%)", flush=True)


def _save_correctness(correctness, N, TRAIN_END, path):
    n_ct = sum(1 for si, v in correctness.items() if si < TRAIN_END and v)
    n_ce = sum(1 for si, v in correctness.items() if si >= TRAIN_END and v)
    n_test = N - TRAIN_END
    out = {
        "train_end": TRAIN_END, "N": N,
        "n_correct_train": n_ct, "n_correct_test": n_ce,
        "acc_train": n_ct / TRAIN_END if TRAIN_END > 0 else 0,
        "acc_test":  n_ce / n_test if n_test > 0 else 0,
        "correct": {str(si): v for si, v in correctness.items()},
    }
    with open(path, "w") as f:
        import json; json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
