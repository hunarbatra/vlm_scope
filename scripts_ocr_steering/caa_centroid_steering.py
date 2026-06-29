#!/usr/bin/env python3
"""
SDS-inspired centroid steering.

Approach:
  1. Cluster mix-448's "correct" hidden states (pos[L] from cache) into K=8 centroids per layer
  2. At pt-448 inference: capture current state, find nearest CORRECT centroid,
     push direction = α · (centroid - current)
  3. Input-dependent steering — adapts per sample

Eval on UNION of all winner R(F) sets, lenient metric.
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
OUT_DIR      = SAE_ROOT / "analysis_ocr/caa_centroid"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER = 13
BACKBONE_LAYERS = [17, 19, 20, 21]
K_CLUSTERS = 8
ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]
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


def kmeans_torch(X, K, n_iter=30, seed=0):
    """Simple k-means in torch. X: [N, D]."""
    g = torch.Generator().manual_seed(seed)
    N, D = X.shape
    idx = torch.randperm(N, generator=g)[:K]
    centroids = X[idx].clone()
    for _ in range(n_iter):
        # Assign each point to nearest centroid
        d = torch.cdist(X, centroids)  # [N, K]
        assign = d.argmin(dim=1)
        # Update centroids
        new_c = torch.zeros_like(centroids)
        for k in range(K):
            mask = assign == k
            if mask.sum() > 0:
                new_c[k] = X[mask].mean(dim=0)
            else:
                new_c[k] = centroids[k]
        if torch.allclose(new_c, centroids, atol=1e-5): break
        centroids = new_c
    return centroids, assign


def build_centroids():
    """Per backbone layer, cluster correct-sample hidden states into K centroids."""
    print("[INFO] Loading paired cache and building centroids...", flush=True)
    centroids_by_layer = {}
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
        H = torch.stack(h_correct)  # [N_correct, D]
        cents, _ = kmeans_torch(H, K=K_CLUSTERS, n_iter=50)
        centroids_by_layer[L] = cents
        print(f"  L{L}: {H.shape[0]} correct samples → {K_CLUSTERS} centroids "
              f"(mean ||c||={cents.norm(dim=1).mean():.2f})", flush=True)
    return centroids_by_layer


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print("="*80)
    print("CENTROID STEERING — SDS-inspired, mix→pt, ocr prompt")
    print("="*80, flush=True)

    centroids = build_centroids()
    if not centroids:
        print("[ERROR] No centroids built"); return

    ds = load_dataset("echo840/OCRBench", split="test")

    # Eval set: UNION of winner R(F)
    rF_union = set()
    for k in WINNERS:
        ap = SAE_ACTS_DIR / f"acts_{k}.json"
        if not ap.exists(): continue
        ad = json.load(open(ap))
        rF_union |= {int(x) for x, v in ad.get("acts", {}).items() if v > 0}
    rF_union = sorted(rF_union)
    print(f"\n[INFO] Eval set: union of {len(WINNERS)} winners = {len(rF_union)} samples", flush=True)

    print(f"\n[Loading {PT_MODEL}]", flush=True)
    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok = proc.tokenizer
    dtype = next(mdl.parameters()).dtype

    # Move centroids to device
    centroids_dev = {L: c.to(dtype).to(device) for L, c in centroids.items()}

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Baseline
    bk = "centroid_rF_base"
    if bk not in all_results:
        bc = bt = 0
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
                bt += 1; bc += int(_correct(resp, gt))
            except: continue
        all_results[bk] = {"acc": bc/max(bt,1)*100, "n": bt}
        print(f"\n  Baseline: {all_results[bk]['acc']:.2f}% (n={bt})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    base = all_results[bk]["acc"]

    # Steering: at prefill, compute direction = (nearest_correct_centroid - mean_h),
    # then inject α · direction at the same layer for ALL forward passes
    img_end_r = [0]
    inject_dirs = {L: None for L in [MIDDLE_LAYER] + BACKBONE_LAYERS}
    captured_pre = {}

    def capture_hook(L):
        def f(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:  # prefill only
                captured_pre[L] = h.detach()
        return f

    def inject_hook(L):
        def f(m, inp, out):
            ie = img_end_r[0]
            h = out[0] if isinstance(out, tuple) else out
            d = inject_dirs.get(L)
            if d is not None:
                h[0, ie:] = h[0, ie:] + d.unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return f

    # We register both: capture during prefill (hook reads), inject during all passes
    # But capture needs to be a separate forward pass. Strategy:
    #   1) First forward with capture-only → get current state at each layer
    #   2) Compute injection directions
    #   3) Run generate with injection hooks active
    # This doubles compute but enables per-sample direction.

    def steer_eval(alpha, mode):
        """mode: 'all_layers' or 'middle_only' or 'backbone_only'"""
        target_layers = []
        if mode == "all_layers": target_layers = [MIDDLE_LAYER] + BACKBONE_LAYERS
        elif mode == "middle_only": target_layers = [MIDDLE_LAYER]
        elif mode == "backbone_only": target_layers = BACKBONE_LAYERS
        else: raise ValueError(mode)

        c = t = 0
        for si in rF_union:
            ex = ds[si]; img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            if img is None or not gt: continue
            try:
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(img, "ocr", proc, mdl, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)

                # 1) Capture-only forward (no injection)
                captured_pre.clear()
                for L in target_layers: inject_dirs[L] = None
                cap_hooks = []
                for L in target_layers:
                    cap_hooks.append(mdl.model.language_model.layers[L].register_forward_hook(capture_hook(L)))
                with torch.no_grad():
                    _ = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                for h in cap_hooks:
                    try: h.remove()
                    except: pass

                # 2) Compute injection direction for each layer
                for L in target_layers:
                    if L not in captured_pre: continue
                    h_mean = captured_pre[L][0, img_end_r[0]:, :].float().mean(dim=0)
                    cents = centroids_dev[L].float()
                    d = torch.cdist(h_mean.unsqueeze(0), cents)
                    nearest = d.argmin().item()
                    delta = (cents[nearest] - h_mean) * alpha
                    inject_dirs[L] = delta.to(dtype).to(device)

                # 3) Generate with injection hooks
                inj_hooks = []
                for L in target_layers:
                    if inject_dirs[L] is not None:
                        inj_hooks.append(mdl.model.language_model.layers[L].register_forward_hook(inject_hook(L)))
                with torch.no_grad():
                    out_ids = mdl.generate(input_ids=iids, attention_mask=attn, pixel_values=pv,
                                           max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
                for h in inj_hooks:
                    try: h.remove()
                    except: pass
                resp = tok.decode(out_ids[0, iids.shape[1]:], skip_special_tokens=True)
                t += 1; c += int(_correct(resp, gt))
            except Exception:
                continue
        return c, t

    for mode in ["middle_only", "backbone_only", "all_layers"]:
        for alpha in ALPHAS:
            rk = f"centroid_{mode}_a{alpha}"
            if rk in all_results and all_results[rk].get("n", 0) > 0:
                r = all_results[rk]
                print(f"  [SKIP {mode} α={alpha}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
                continue
            c, t = steer_eval(alpha, mode)
            if t == 0: continue
            acc = c / t * 100
            delta = acc - base
            all_results[rk] = {"acc": acc, "delta": delta, "n": t, "mode": mode, "alpha": alpha}
            print(f"  [centroid_{mode} α={alpha}] {acc:.2f}% Δ={delta:+.2f}% ({c}/{t})", flush=True)
            with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    print(f"\n[DONE] base={base:.2f}%", flush=True)


if __name__ == "__main__":
    main()
