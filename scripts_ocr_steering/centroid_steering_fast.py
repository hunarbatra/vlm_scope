#!/usr/bin/env python3
"""
FAST centroid steering: single forward pass, captures + injects in same pass.

Strategy:
  1. Pre-compute K=8 centroids per layer from mix-448 'correct' samples (offline)
  2. At inference: hook captures pt-448 prefill state, computes nearest centroid,
     and INJECTS direction in the SAME forward pass (modifies output of that layer).
  3. Generation steps then proceed with the modified residual.

Single forward pass per sample (vs 2 in the slow version).
"""
import os, sys, json, gc, argparse, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
PAIR_CACHE   = SAE_ROOT / "analysis_ocr/paired_cache_ocrprompt"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts_ocrprompt"
OUT_DIR      = SAE_ROOT / "analysis_ocr/centroid_fast"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER = 13
BACKBONE_LAYERS = [17, 19, 20, 21]
K_CLUSTERS = 8
ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]   # smaller because direction = centroid - h
MAX_NEW_TOKENS = 64

WINNERS = ["L21_F13072", "L19_F9893", "L21_F677", "L19_F8866",
           "L17_F9368", "L17_F12336", "L21_F10675", "L19_F89"]


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


def kmeans_torch(X, K, n_iter=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    idx = torch.randperm(N, generator=g)[:K]
    centroids = X[idx].clone()
    for _ in range(n_iter):
        d = torch.cdist(X, centroids)
        assign = d.argmin(dim=1)
        new_c = torch.zeros_like(centroids)
        for k in range(K):
            mask = assign == k
            new_c[k] = X[mask].mean(dim=0) if mask.sum() > 0 else centroids[k]
        if torch.allclose(new_c, centroids, atol=1e-5): break
        centroids = new_c
    return centroids


def build_centroids():
    """Per layer, k-means on correct mix samples' pos hidden states."""
    centroids = {}
    for L in [MIDDLE_LAYER] + BACKBONE_LAYERS:
        h_correct = []
        for si in range(1000):
            p = PAIR_CACHE / f"vi_{si:05d}.pt"
            if not p.exists(): continue
            try: d = torch.load(p, map_location="cpu", weights_only=False)
            except: continue
            if not d.get("correct", False): continue
            if L not in d.get("pos", {}): continue
            h_correct.append(d["pos"][L].float())
        if not h_correct: continue
        H = torch.stack(h_correct)
        centroids[L] = kmeans_torch(H, K=K_CLUSTERS)
        print(f"  L{L}: {H.shape[0]} correct samples → {K_CLUSTERS} centroids "
              f"(mean ||c||={centroids[L].norm(dim=1).mean():.2f})", flush=True)
    return centroids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=None,
                   help="Single alpha to test (skips sweep)")
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("="*80)
    print("CENTROID STEERING (FAST) — single forward pass")
    print("="*80, flush=True)

    centroids = build_centroids()

    ds = load_dataset("echo840/OCRBench", split="test")

    # Eval set: union of winners
    rF_union = set()
    for k in WINNERS:
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists(): continue
        ad = json.load(open(ap))
        rF_union |= {int(x) for x, v in ad.get("acts", {}).items() if v > 0}
    rF_union = sorted(rF_union)
    print(f"\nEval set: union of winners = {len(rF_union)}", flush=True)

    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok = proc.tokenizer
    dtype = next(mdl.parameters()).dtype

    centroids_dev = {L: c.to(dtype).to(device) for L, c in centroids.items()}

    img_end_r = [0]
    # State: per-sample direction computed during prefill, applied during generation
    inject_state = {}   # {layer: tensor or None}

    def make_centroid_hook(L, alpha, mode):
        """Hook captures during prefill, computes centroid direction, applies to current
        and all subsequent forwards.

        mode: 'middle' (only L13) or 'backbone' (only backbone) or 'all'
        """
        target_layers = []
        if mode == "middle": target_layers = [MIDDLE_LAYER]
        elif mode == "backbone": target_layers = BACKBONE_LAYERS
        elif mode == "all": target_layers = [MIDDLE_LAYER] + BACKBONE_LAYERS

        def f(m, inp, out):
            if L not in target_layers: return out
            ie = img_end_r[0]
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:  # prefill: compute direction
                h_text = h[0, ie:, :].float()
                h_mean = h_text.mean(dim=0)
                cents = centroids_dev[L].float()
                d = torch.cdist(h_mean.unsqueeze(0), cents)
                nearest = d.argmin().item()
                delta = (cents[nearest] - h_mean) * alpha
                inject_state[L] = delta.to(dtype)
            # Apply
            sv = inject_state.get(L)
            if sv is not None:
                h[0, ie:] = h[0, ie:] + sv.unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return f

    def eval_centroid(alpha, mode):
        c = t = 0
        for si in rF_union:
            ex = ds[si]; img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            if img is None or not gt: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                inject_state.clear()
                hooks = []
                for L in centroids_dev:
                    hooks.append(mdl.model.language_model.layers[L].register_forward_hook(
                        make_centroid_hook(L, alpha, mode)))
                with torch.no_grad():
                    out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                           pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False, use_cache=True)
                for h in hooks:
                    try: h.remove()
                    except: pass
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                t += 1; c += int(_correct(resp, gt))
            except Exception:
                continue
        return c, t

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    bk = "centroid_base"
    if bk not in all_results:
        c, t = eval_centroid(0.0, "middle")  # alpha=0 = no change, but still has overhead
        # Actually do clean baseline
        c0 = t0 = 0
        for si in rF_union:
            ex = ds[si]; img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            if img is None or not gt: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
                with torch.no_grad():
                    out_ids = mdl.generate(input_ids=iids, attention_mask=attn,
                                           pixel_values=pv, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False, use_cache=True)
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                t0 += 1; c0 += int(_correct(resp, gt))
            except: continue
        all_results[bk] = {"acc": c0/max(t0,1)*100, "n": t0}
        print(f"  Baseline: {all_results[bk]['acc']:.2f}% (n={t0})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
    base = all_results[bk]["acc"]

    alphas = [args.alpha] if args.alpha else ALPHAS
    for mode in ["middle", "backbone", "all"]:
        for alpha in alphas:
            rk = f"centroid_{mode}_a{alpha}"
            if rk in all_results: continue
            c, t = eval_centroid(alpha, mode)
            if t == 0: continue
            acc = c/t*100; delta = acc - base
            all_results[rk] = {"acc": acc, "delta": delta, "n": t}
            print(f"  [centroid_{mode} α={alpha}] {acc:.2f}% Δ={delta:+.2f}% ({c}/{t})", flush=True)
            with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    print(f"\n[DONE] base={base:.2f}%", flush=True)


if __name__ == "__main__":
    main()
