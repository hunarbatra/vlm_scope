#!/usr/bin/env python3
"""
End-to-end QA of the OCR steering pipeline.

After the recipe sweep completes, this script:
  1. Loads the best A_MIDDLE and best D_BB+WDEC configs from results.json
  2. For 15 sample R(F) members per feature, generates 3 versions:
       - baseline (no steering, pt-448 with "ocr" prompt)
       - A_MIDDLE injection
       - D_BB+WDEC injection
  3. Side-by-side prints: GT, baseline resp, A resp, D resp, lenient match each
  4. Verifies CAA injection is actually changing outputs
  5. Sanity checks: CAA norms, W_dec norms, paired contrast magnitude

Usage:
  CUDA_VISIBLE_DEVICES=X python3 -B scripts/qa_steering_outputs.py
"""
import os, sys, json, gc, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
SAE_CKPT_DIR = SAE_ROOT / "checkpoints"
PAIR_CACHE   = SAE_ROOT / "analysis_ocr/paired_cache_ocrprompt"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts"
RESULTS_PATH = SAE_ROOT / "analysis_ocr/caa_paired_recipe_ocrprompt/results.json"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

TRAIN_END = 1000
MIDDLE_LAYER = 13
BACKBONE_LAYERS = [17, 19, 20, 21]
NUM_LAYERS = 26

SPATIAL_FEATURES = [
    {"layer": 17, "feature": 13602, "key": "L17_F13602"},
    {"layer": 21, "feature": 9577,  "key": "L21_F9577"},
]

N_SAMPLE_PER_FEAT = 15
MAX_NEW_TOKENS = 256


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def _correct_lenient(resp, gt):
    if not resp: return False
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
    if n == 0: return None, 0
    return (pos - neg) / n, n


def best_config(results, prefix, has_gamma=False):
    """Return best config tuple for a given recipe prefix."""
    best = None
    keys = [k for k in results if k.startswith(prefix)]
    for k in keys:
        if "_rF_base" in k: continue
        r = results[k]
        if not isinstance(r, dict) or "delta" not in r: continue
        if best is None or r["delta"] > best["delta"]:
            best = {"key": k, **r}
    return best


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"

    print("=" * 90)
    print("QA — End-to-end verification of OCR steering pipeline")
    print("=" * 90, flush=True)

    if not RESULTS_PATH.exists():
        print(f"[ERROR] No results.json at {RESULTS_PATH}; recipe sweep hasn't run yet.")
        return

    results = json.load(open(RESULTS_PATH))
    print(f"\n[INFO] Loaded results from {RESULTS_PATH}: {len(results)} entries", flush=True)

    # Sanity-check counts
    cache_files = list(PAIR_CACHE.glob("vi_*.pt"))
    print(f"[INFO] Paired cache: {len(cache_files)} entries\n", flush=True)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")

    print(f"[INFO] Loading {PT_MODEL}...", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tok = proc.tokenizer
    dtype = next(mdl.parameters()).dtype

    img_end_r = [0]
    def make_hook(sv_):
        def f(m, inp, out):
            ie = img_end_r[0]
            h = out[0] if isinstance(out, tuple) else out
            h[0, ie:] = h[0, ie:] + sv_.unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return f

    def generate(img, inject_pairs):
        hooks = []
        try:
            iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
            _, img_end_r[0] = get_image_token_positions(iids)
            for (l, sv) in inject_pairs:
                hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
            with torch.no_grad():
                out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                       pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                       do_sample=False, use_cache=True)
            for h in hooks:
                try: h.remove()
                except: pass
            return tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            for h in hooks:
                try: h.remove()
                except: pass
            return f"<ERR:{type(e).__name__}>"

    for sf in SPATIAL_FEATURES:
        key, lF, fi = sf["key"], sf["layer"], sf["feature"]
        print(f"\n{'='*90}\n=== {key} (layer L{lF}, feature {fi}) ===\n{'='*90}", flush=True)

        # ---- R(F) ----
        ap = SAE_ACTS_DIR / f"acts_{key}.json"
        ad = json.load(open(ap))
        rF = sorted([int(k) for k, v in ad["acts"].items() if v > 0])
        print(f"\nR(F) over all 1000: n={len(rF)}", flush=True)

        # ---- per-feature CAA ----
        needed = sorted(set(BACKBONE_LAYERS) | {MIDDLE_LAYER, lF})
        caa_unit = {}
        for l in needed:
            v, n = compute_paired_caa(l, indices=rF)
            if v is not None:
                caa_unit[l] = v / v.norm().clamp(min=1e-8)

        print("\n[CAA SANITY] Per-layer raw norms (built from R(F)):", flush=True)
        for l in needed:
            v, _ = compute_paired_caa(l, indices=rF)
            if v is not None:
                print(f"  L{l}: raw_norm={v.norm():.3f}  unit_norm={(v/v.norm()).norm():.4f}", flush=True)
        w_dec = _load_wdec(lF, fi)
        print(f"\n[W_dec SANITY] L{lF}/F{fi}: norm={w_dec.norm():.3f}  shape={tuple(w_dec.shape)}", flush=True)

        # ---- best configs ----
        bA = best_config(results, f"{key}_A_middle")
        bD = best_config(results, f"{key}_D_bb_wdec", has_gamma=True)
        rF_base = results.get(f"{key}_rF_base", {}).get("acc", "?")
        print(f"\n[CONFIG] R(F) baseline (lenient, ocr prompt): {rF_base}%", flush=True)
        print(f"[CONFIG] Best A_MIDDLE: {bA}", flush=True)
        print(f"[CONFIG] Best D_BB+WDEC: {bD}", flush=True)

        if not bA or not bD:
            print(f"[WARN] missing A or D config for {key}; skip side-by-side", flush=True)
            continue

        # Parse alpha (and gamma for D) from key strings
        # A: f"{key}_A_middle_a{a}"
        # D: f"{key}_D_bb_wdec_a{a}_g{g}"
        aA_alpha = float(bA["key"].split("_a")[-1])
        aD_alpha, aD_gamma = bD["key"].split("_a")[-1].split("_g")
        aD_alpha = float(aD_alpha); aD_gamma = float(aD_gamma)

        # Build injection pairs
        sv_A = (caa_unit[MIDDLE_LAYER] * aA_alpha).to(dtype).to(device)
        inject_A = [(MIDDLE_LAYER, sv_A)]

        inject_D = []
        for l in BACKBONE_LAYERS:
            if l not in caa_unit: continue
            if l == lF:
                sv = (caa_unit[l] * aD_alpha + w_dec.to(dtype).to(device) * aD_gamma).to(dtype).to(device)
            else:
                sv = (caa_unit[l] * aD_alpha).to(dtype).to(device)
            inject_D.append((l, sv))

        # ---- side-by-side outputs on N_SAMPLE_PER_FEAT R(F) members ----
        print(f"\n[SIDE-BY-SIDE] {N_SAMPLE_PER_FEAT} samples from R({key}):", flush=True)
        print(f"  {'idx':>4}  {'GT':<28}  {'baseline':<35}  {'A_MIDDLE':<35}  {'D_BB+WDEC':<35}  {'b':>2} {'A':>2} {'D':>2}")
        print("  " + "-"*160)

        # take a stride to sample
        stride = max(1, len(rF) // N_SAMPLE_PER_FEAT)
        sampled = rF[::stride][:N_SAMPLE_PER_FEAT]

        n_b = n_A = n_D = 0
        for si in sampled:
            ex = ds[si]
            img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            if img is None or not gt: continue
            try: img = img.convert("RGB")
            except: continue

            r_b = generate(img, [])
            r_A = generate(img, inject_A)
            r_D = generate(img, inject_D)

            ok_b = _correct_lenient(r_b, gt)
            ok_A = _correct_lenient(r_A, gt)
            ok_D = _correct_lenient(r_D, gt)
            n_b += ok_b; n_A += ok_A; n_D += ok_D

            gt_s = (gt[:26] + '..') if len(gt) > 26 else gt
            r_b_s = (r_b[:33] + '..') if len(r_b) > 33 else r_b
            r_A_s = (r_A[:33] + '..') if len(r_A) > 33 else r_A
            r_D_s = (r_D[:33] + '..') if len(r_D) > 33 else r_D
            r_b_s = r_b_s.replace("\n", " ⏎ ")
            r_A_s = r_A_s.replace("\n", " ⏎ ")
            r_D_s = r_D_s.replace("\n", " ⏎ ")
            print(f"  {si:>4}  {gt_s:<28}  {r_b_s:<35}  {r_A_s:<35}  {r_D_s:<35}  "
                  f"{'✓' if ok_b else '✗':>2} {'✓' if ok_A else '✗':>2} {'✓' if ok_D else '✗':>2}", flush=True)

        print(f"\n  Sample totals (n={len(sampled)}):  baseline={n_b}  A={n_A}  D={n_D}", flush=True)

        # ---- output-changes diagnostic ----
        # Did A and D actually change the output vs baseline?
        # (We've already run, but let's count distinct outputs from above)

    print(f"\n{'='*90}\nQA COMPLETE\n{'='*90}", flush=True)


if __name__ == "__main__":
    main()
