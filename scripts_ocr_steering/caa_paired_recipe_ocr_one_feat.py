#!/usr/bin/env python3
"""Recipe sweep for ONE feature (parallel across GPUs).
Usage: CUDA_VISIBLE_DEVICES=N python3 -B caa_paired_recipe_ocr_one_feat.py --layer L --feature F
"""
import os, sys, json, gc, argparse, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
SAE_CKPT_DIR = SAE_ROOT / "checkpoints"
PAIR_CACHE   = SAE_ROOT / "analysis_ocr/paired_cache_ocrprompt"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts_ocrprompt"
OUT_DIR      = SAE_ROOT / "analysis_ocr/caa_paired_recipe_ocrwinners"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER  = 13
BACKBONE_LAYERS = [17, 19, 20, 21]
ALPHAS        = [0.5, 1.0, 2.0, 5.0, 10.0]
GAMMAS        = [1.0, 3.0, 10.0]
MAX_NEW_TOKENS = 64


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def _correct(resp, gt):
    if resp is None: return False
    r = str(resp).strip().lower(); g = str(gt).strip().lower()
    return bool(g) and (g in r or r in g)


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][fi].float()


def compute_paired_caa(layer, indices):
    pos = neg = None; n = 0
    for si in indices:
        p = PAIR_CACHE / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception: continue
        if "pos" not in d or "neg" not in d: continue
        if layer not in d["pos"] or layer not in d["neg"]: continue
        hp = d["pos"][layer].float(); hn = d["neg"][layer].float()
        pos = hp.clone() if pos is None else pos + hp
        neg = hn.clone() if neg is None else neg + hn
        n += 1
    if n == 0 or pos is None: return None, 0
    return (pos - neg) / n, n


def run_eval(tag, inject_pairs, test_indices, ds, model, processor, tok, device,
             base_acc, result_key, all_results, results_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    if result_key in all_results and all_results[result_key].get("n", 0) > 0:
        r = all_results[result_key]
        print(f"  [SKIP {tag}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
        return all_results

    img_end_r = [0]
    def make_hook(sv_):
        def f(m, inp, out):
            ie = img_end_r[0]
            h = out[0] if isinstance(out, tuple) else out
            h[0, ie:] = h[0, ie:] + sv_.unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return f

    c = t = 0
    for si in test_indices:
        ex  = ds[si]
        img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
        if img is None or not gt: continue
        hooks = []
        try:
            img = img.convert("RGB")
            iids, attn, pv = process_vlm_inputs(img, "ocr", processor, model, device=device)
            _, img_end_r[0] = get_image_token_positions(iids)
            for (l, sv) in inject_pairs:
                hooks.append(model.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
            with torch.no_grad():
                out_ids = model.generate(input_ids=iids, attention_mask=attn,
                                         pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=False, use_cache=True)
            for h in hooks:
                try: h.remove()
                except: pass
            resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
            t += 1
            c += int(_correct(resp, gt))
        except Exception:
            for h in hooks:
                try: h.remove()
                except: pass

    if t == 0: return all_results
    acc = c / t * 100
    delta = acc - base_acc
    all_results[result_key] = {"acc": acc, "delta": delta, "n": t}
    print(f"  [{tag}] {acc:.2f}%  Δ={delta:+.2f}%  ({c}/{t})", flush=True)
    with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
    return all_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--feature", type=int, required=True)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layer, feature = args.layer, args.feature
    key = f"L{layer}_F{feature}"
    results_path = OUT_DIR / f"results_{key}.json"   # per-feature file → no race

    print(f"=== {key} (single-feature run) ===", flush=True)

    ds = load_dataset("echo840/OCRBench", split="test")
    test_indices = list(range(len(ds)))

    ap = SAE_ACTS_DIR / f"acts_{key}.json"
    if not ap.exists():
        print(f"[ERROR] acts file missing: {ap}"); return
    ad = json.load(open(ap))
    ak = {int(x) for x, v in ad.get("acts", {}).items() if v > 0}
    rF = [v for v in test_indices if v in ak]
    print(f"R({key}) = n={len(rF)}", flush=True)

    needed_layers = sorted(set(BACKBONE_LAYERS) | {MIDDLE_LAYER, layer})
    caa_unit = {}
    for l in needed_layers:
        v, n = compute_paired_caa(l, indices=rF)
        if v is not None:
            caa_unit[l] = v / v.norm().clamp(min=1e-8)
            print(f"  L{l}: paired n={n} raw_norm={v.norm():.3f}", flush=True)

    print(f"\n[Loading {PT_MODEL}]", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok = proc.tokenizer
    dtype = next(mdl.parameters()).dtype

    # File locking required if multiple processes write simultaneously — rely on
    # write being short and atomic for now (write-after-read risk minimal)
    all_results = json.load(open(results_path)) if results_path.exists() else {}

    bk = f"{key}_rF_base"
    if bk not in all_results:
        bc = bt = 0
        for si in rF:
            ex = ds[si]; img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            if img is None or not gt: continue
            try:
                img = img.convert("RGB")
                from utils import process_vlm_inputs
                iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
                with torch.no_grad():
                    out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                           pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False, use_cache=True)
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                bt += 1; bc += int(_correct(resp, gt))
            except Exception: continue
        all_results[bk] = {"acc": bc/max(bt,1)*100, "n": bt}
        print(f"[{key}] rF base: {all_results[bk]['acc']:.2f}% (n={bt})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    base_rF = all_results[bk]["acc"]
    print(f"\n--- Recipe sweep: {key} n={len(rF)} base={base_rF:.2f}% ---", flush=True)

    if MIDDLE_LAYER in caa_unit:
        for a in ALPHAS:
            sv = (caa_unit[MIDDLE_LAYER] * a).to(dtype).to(device)
            rk = f"{key}_A_middle_a{a}"
            all_results = run_eval(
                f"{key}/A_MIDDLE α={a}", [(MIDDLE_LAYER, sv)],
                rF, ds, mdl, proc, tok, device,
                base_rF, rk, all_results, results_path)

    w_dec = _load_wdec(layer, feature)
    for a in ALPHAS:
        for g in GAMMAS:
            inject = []
            for l in BACKBONE_LAYERS:
                if l not in caa_unit: continue
                if l == layer:
                    sv = (caa_unit[l] * a + w_dec * g).to(dtype).to(device)
                else:
                    sv = (caa_unit[l] * a).to(dtype).to(device)
                inject.append((l, sv))
            rk = f"{key}_D_bb_wdec_a{a}_g{g}"
            all_results = run_eval(
                f"{key}/D_BB+WDEC α={a} γ={g}", inject,
                rF, ds, mdl, proc, tok, device,
                base_rF, rk, all_results, results_path)

    print(f"\n[DONE] {key}", flush=True)


if __name__ == "__main__":
    main()
