#!/usr/bin/env python3
"""Summarize all per-feature OCR steering results."""
import json
from pathlib import Path

OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/caa_paired_recipe_ocrwinners")

print("="*100)
print("OCR STEERING — full summary across all features")
print("="*100)
print()
print(f"{'Feature':<14} {'L':>3} {'N':>5} {'Base':>8}  "
      f"{'A_MIDDLE best':>22}  {'D_BB+WDEC best':>28}  {'D > A?':>8}")
print("-"*100)

best_overall = None
for fp in sorted(OUT_DIR.glob("results_L*.json")):
    d = json.load(open(fp))
    key = fp.stem.replace("results_", "")
    bk = f"{key}_rF_base"
    if bk not in d: continue
    base = d[bk]["acc"]
    n = d[bk]["n"]
    layer = key.split("_")[0][1:]

    # Best A_MIDDLE
    aA = None
    for k, v in d.items():
        if not k.startswith(f"{key}_A_middle_a"): continue
        if "delta" not in v: continue
        if aA is None or v["delta"] > aA["delta"]:
            alpha = float(k.split("_a")[-1])
            aA = {**v, "alpha": alpha}

    # Best D_BB+WDEC
    aD = None
    for k, v in d.items():
        if not k.startswith(f"{key}_D_bb_wdec_a"): continue
        if "delta" not in v: continue
        if aD is None or v["delta"] > aD["delta"]:
            ag = k.split("_a")[-1].split("_g")
            alpha, gamma = float(ag[0]), float(ag[1])
            aD = {**v, "alpha": alpha, "gamma": gamma}

    fmt_a = lambda x: f"{x['acc']:.2f}% Δ={x['delta']:+.2f} (a{x['alpha']:g})" if x else "—"
    fmt_d = lambda x: f"{x['acc']:.2f}% Δ={x['delta']:+.2f} (a{x['alpha']:g}/g{x['gamma']:g})" if x else "—"
    diff = (aD["delta"] - aA["delta"]) if (aA and aD) else 0
    win = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"

    print(f"{key:<14} L{layer:>2} {n:>5} {base:>7.2f}%  {fmt_a(aA):>22}  {fmt_d(aD):>28}  {win:>8}")

    if aD and (best_overall is None or aD["delta"] > best_overall[1]["delta"]):
        best_overall = (key, aD)

print()
if best_overall:
    k, r = best_overall
    print(f"BEST D_BB+WDEC RESULT: {k} α={r['alpha']:g} γ={r['gamma']:g}: "
          f"{r['acc']:.2f}% (Δ={r['delta']:+.2f}pp)")

print()
print("="*100)
