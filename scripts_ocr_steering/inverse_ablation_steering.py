#!/usr/bin/env python3
"""
Inverse-ablation steering: pure γ·W_dec[F] injection at feature's home layer.

Hypothesis: ablating W_dec[F] dropped accuracy by N%; adding γ·W_dec[F] should
increase accuracy by some fraction of that.

Eval on R(F)∩all-1000 from the original sae_acts/ files (answer-en derived).
Use "ocr" prompt + lenient substring match (OCR-Bench official).
Wide γ sweep including negative.

Usage: CUDA_VISIBLE_DEVICES=N python3 -B inverse_ablation_steering.py --layer L --feature F
"""
import os, sys, json, argparse, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
SAE_CKPT_DIR = SAE_ROOT / "checkpoints"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts"   # original (answer-en) acts
OUT_DIR      = SAE_ROOT / "analysis_ocr/inverse_ablation"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Wide γ sweep, both signs
GAMMAS = [-30.0, -10.0, -3.0, -1.0, -0.5, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
MAX_NEW_TOKENS = 64


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def _correct(resp, gt):
    if not resp: return False
    r = str(resp).strip().lower(); g = str(gt).strip().lower()
    return bool(g) and (g in r or r in g)


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][fi].float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--feature", type=int, required=True)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layer, feature = args.layer, args.feature
    key = f"L{layer}_F{feature}"
    results_path = OUT_DIR / f"results_{key}.json"

    print(f"=== INVERSE-ABLATION {key} ===", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")

    ap = SAE_ACTS_DIR / f"acts_{key}.json"
    if not ap.exists(): print(f"[ERROR] {ap} missing"); return
    ad = json.load(open(ap))
    rF = sorted({int(x) for x, v in ad.get("acts", {}).items() if v > 0})
    print(f"R({key}) (answer-en R(F)) over all 1000: n={len(rF)}", flush=True)

    if len(rF) < 20:
        print(f"[WARN] R(F) too small. Using all 1000."); rF = list(range(1000))

    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok = proc.tokenizer
    dtype = next(mdl.parameters()).dtype

    w_dec = _load_wdec(layer, feature)
    print(f"  W_dec[L{layer}/F{feature}] norm={w_dec.norm():.3f}", flush=True)

    img_end_r = [0]
    def make_hook(sv_):
        def f(m, inp, out):
            ie = img_end_r[0]
            h = out[0] if isinstance(out, tuple) else out
            h[0, ie:] = h[0, ie:] + sv_.unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return f

    def eval_at(inject_pairs):
        c = t = 0
        for si in rF:
            ex = ds[si]; img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            if img is None or not gt: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                hooks = []
                for (l, sv) in inject_pairs:
                    hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
                with torch.no_grad():
                    out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                           pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False, use_cache=True)
                for h in hooks:
                    try: h.remove()
                    except: pass
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                t += 1; c += int(_correct(resp, gt))
            except: continue
        return c, t

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    bk = "base"
    if bk not in all_results:
        c, t = eval_at([])
        all_results[bk] = {"acc": c/max(t,1)*100, "n": t}
        print(f"  Baseline (no steering): {all_results[bk]['acc']:.2f}% (n={t})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
    base = all_results[bk]["acc"]

    # Pure γ·W_dec[F] at lF only — no CAA
    for g in GAMMAS:
        rk = f"wdec_only_g{g}"
        if rk in all_results: continue
        sv = (w_dec * g).to(dtype).to(device)
        c, t = eval_at([(layer, sv)])
        if t == 0: continue
        acc = c/t*100; delta = acc - base
        all_results[rk] = {"acc": acc, "delta": delta, "n": t}
        print(f"  [{key}/W_dec γ={g}] {acc:.2f}% Δ={delta:+.2f}% ({c}/{t})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # Also try W_dec at MULTIPLE layers (broadcast feature direction)
    for g in [3.0, 10.0, 30.0]:
        rk = f"wdec_alllayers_g{g}"
        if rk in all_results: continue
        # Inject W_dec at this layer + adjacent layers (lF-1, lF, lF+1)
        target_layers = [l for l in [layer-1, layer, layer+1] if 0 <= l < 26]
        sv_ = (w_dec * g).to(dtype).to(device)
        inject = [(l, sv_) for l in target_layers]
        c, t = eval_at(inject)
        if t == 0: continue
        acc = c/t*100; delta = acc - base
        all_results[rk] = {"acc": acc, "delta": delta, "n": t}
        print(f"  [{key}/W_dec_3layers γ={g}] {acc:.2f}% Δ={delta:+.2f}% ({c}/{t})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    print(f"\n[DONE] {key}", flush=True)


if __name__ == "__main__":
    main()
