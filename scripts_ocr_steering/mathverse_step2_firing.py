#!/usr/bin/env python3
"""
MathVerse Steps 3-5: Per-layer SAE firing stats + Fisher/OR feature ranking.

For every layer L in 0..25, runs SAE encode on cached hidden states,
computes per-feature firing rates split by correct/incorrect.

Outputs:
  analysis_mathverse/firing/firing_L{L}.json
    {"layer": L, "features": {str(f): {"fire_corr": int, "fire_incorr": int,
                                        "n_corr": int, "n_incorr": int}}}
  analysis_mathverse/fisher_ranked.json
    Top features sorted by Fisher score and OR magnitude.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 -u mathverse_step2_firing.py
"""
import os, sys, json, math
from pathlib import Path
import torch
from scipy.stats import fisher_exact

SAE_ROOT   = Path("/data1/vlm_scope_sae_mix448_textonly")
CACHE_DIR  = SAE_ROOT / "analysis_mathverse/mix_hidden"
CORR_PATH  = SAE_ROOT / "analysis_mathverse/correctness.json"
FIRE_DIR   = SAE_ROOT / "analysis_mathverse/firing"
CKPT_DIR   = SAE_ROOT / "checkpoints"
RANKED_OUT = SAE_ROOT / "analysis_mathverse/fisher_ranked.json"

TRAIN_END  = 344
NUM_LAYERS = 26
MIN_FIRE   = 3    # minimum total fires across corr+incorr to consider

os.environ["HF_HOME"] = "/data1/hf_cache"


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae

    device = "cuda:0"
    FIRE_DIR.mkdir(parents=True, exist_ok=True)

    corr_data = json.load(open(CORR_PATH))
    correct   = {int(k): v for k, v in corr_data["correct"].items()}
    train_idx = [si for si in correct if si < TRAIN_END]

    corr_ids   = {si for si in train_idx if correct[si]}
    incorr_ids = {si for si in train_idx if not correct[si]}
    n_corr, n_incorr = len(corr_ids), len(incorr_ids)
    print(f"[INFO] train={len(train_idx)}: correct={n_corr}, incorrect={n_incorr}", flush=True)

    all_features = {}  # key: (L, F) -> stats

    for layer in range(NUM_LAYERS):
        out_path = FIRE_DIR / f"firing_L{layer}.json"
        if out_path.exists():
            d = json.load(open(out_path))
            for f_str, stats in d["features"].items():
                f = int(f_str)
                all_features[(layer, f)] = stats
            print(f"  [SKIP] L{layer} already done", flush=True)
            continue

        ckpt = CKPT_DIR / f"text-only_layer_{layer}.pt"
        if not ckpt.exists():
            print(f"  [SKIP] L{layer}: no checkpoint", flush=True)
            continue

        sae = initialize_jumprelu_sae(layer, checkpoint_path=str(ckpt), device=device)
        sae.eval()
        n_features = sae.W_dec.shape[0]

        fire_corr   = torch.zeros(n_features, dtype=torch.long)
        fire_incorr = torch.zeros(n_features, dtype=torch.long)

        loaded = 0
        for si in train_idx:
            p = CACHE_DIR / f"vi_{si:05d}.pt"
            if not p.exists(): continue
            try:
                d = torch.load(p, map_location="cpu", weights_only=True)
                if layer not in d: continue
                h = d[layer].float().unsqueeze(0).to(device)  # [1, 2304]
                with torch.no_grad():
                    acts = sae.encode(h)[0]  # [n_features]
                fires = (acts > 0).cpu()
                if correct[si]:
                    fire_corr   += fires.long()
                else:
                    fire_incorr += fires.long()
                loaded += 1
            except Exception as e:
                print(f"    [WARN] si={si} L{layer}: {e}", flush=True)

        del sae; torch.cuda.empty_cache()

        layer_feats = {}
        for f in range(n_features):
            fc, fi = fire_corr[f].item(), fire_incorr[f].item()
            if fc + fi < MIN_FIRE: continue
            layer_feats[str(f)] = {
                "fire_corr": fc, "fire_incorr": fi,
                "n_corr": n_corr, "n_incorr": n_incorr
            }

        all_features.update({(layer, int(f)): v for f, v in layer_feats.items()})
        out = {"layer": layer, "n_loaded": loaded, "features": layer_feats}
        with open(out_path, "w") as fp:
            json.dump(out, fp)
        print(f"  L{layer}: loaded={loaded}, features_with_fire={len(layer_feats)}", flush=True)

    # Compute Fisher + OR for all features
    print("\n[INFO] Computing Fisher scores...", flush=True)
    ranked = []
    for (layer, f), stats in all_features.items():
        fc, fi = stats["fire_corr"], stats["fire_incorr"]
        nc, ni = stats["n_corr"], stats["n_incorr"]
        nfc, nfi = nc - fc, ni - fi

        table = [[fc, fi], [nfc, nfi]]
        try:
            _, pval = fisher_exact(table, alternative="two-sided")
        except: pval = 1.0

        if fc > 0 and nfi > 0 and fi > 0 and nfc > 0:
            or_ = (fc / max(nfc, 1)) / (fi / max(nfi, 1))
        elif fc > 0 and fi == 0:
            or_ = float("inf")
        elif fc == 0 and fi > 0:
            or_ = 0.0
        else:
            or_ = 1.0

        rate_corr   = fc / nc   if nc  > 0 else 0.0
        rate_incorr = fi / ni   if ni  > 0 else 0.0
        diff        = rate_corr - rate_incorr

        ranked.append({
            "layer": layer, "feature": f,
            "key": f"L{layer}_F{f}",
            "fire_corr": fc, "fire_incorr": fi,
            "n_corr": nc, "n_incorr": ni,
            "rate_corr": rate_corr, "rate_incorr": rate_incorr,
            "diff": diff,
            "or": or_,
            "pval": pval,
            "fisher_score": -math.log10(max(pval, 1e-300)),
        })

    # Sort by Fisher score descending
    ranked.sort(key=lambda x: x["fisher_score"], reverse=True)

    with open(RANKED_OUT, "w") as f:
        json.dump(ranked[:500], f, indent=2)
    print(f"\n[DONE] Top 10 features by Fisher score:", flush=True)
    for r in ranked[:10]:
        print(f"  {r['key']:<14} OR={r['or']:.3f}  diff={r['diff']:+.3f}"
              f"  fire={r['fire_corr']}c/{r['fire_incorr']}i  p={r['pval']:.2e}", flush=True)
    print(f"Full ranking saved → {RANKED_OUT}", flush=True)


if __name__ == "__main__":
    main()
