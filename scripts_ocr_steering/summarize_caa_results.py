#!/usr/bin/env python3
"""Summarize all CAA experiment results into one table."""
import json
from pathlib import Path

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly/analysis")

find_path = ROOT / "caa_find_working_layer" / "results.json"
single_dir = ROOT / "caa_single_layer_eval"
boost_path = ROOT / "caa_spatial_feature_boost" / "results.json"

# Parse find-working-layer
print("=" * 80)
print("CAA FIND WORKING LAYER")
print("=" * 80)
if find_path.exists():
    d = json.load(open(find_path))
    base = d.get("base", {}).get("acc")
    if base:
        print(f"Baseline full VSR test: {base:.2f}% (n={d['base']['n']})")
    print(f"\n{'Condition':<30} {'best α':>8} {'best acc':>9} {'best Δ':>9}  {'Curve':<50}")
    for k, v in sorted(d.items()):
        if k == "base" or not isinstance(v, dict): continue
        recs = [(float(a), r) for a, r in v.items() if isinstance(r, dict) and "delta" in r]
        if not recs: continue
        best = max(recs, key=lambda x: x[1]["delta"])
        curve = " ".join(f"{a:g}:{r['delta']:+.1f}" for a,r in sorted(recs))
        print(f"{k:<30} {best[0]:>8g} {best[1]['acc']:>8.2f}% {best[1]['delta']:>+8.2f}%  {curve:<50}")
else:
    print("(no results yet)")

# Parse parallel single-layer evals
print("\n" + "=" * 80)
print("PARALLEL SINGLE-LAYER EVALS")
print("=" * 80)
if single_dir.exists():
    files = sorted(single_dir.glob("*.json"))
    for f in files:
        d = json.load(open(f))
        base = d.get("base", {}).get("acc")
        tag = f.stem
        recs = [(float(a), r) for a, r in d.items() if a != "base" and isinstance(r, dict) and "delta" in r]
        if not recs:
            print(f"{tag:<25} base={base} (no α results yet)")
            continue
        best = max(recs, key=lambda x: x[1]["delta"])
        curve = " ".join(f"{a:g}:{r['delta']:+.1f}" for a,r in sorted(recs))
        print(f"{tag:<25} base={base:.2f}% best α={best[0]:g} Δ={best[1]['delta']:+.2f}% ({best[1]['acc']:.2f}%)  curve: {curve}")
else:
    print("(no results yet)")

# Parse spatial-feature-boost
print("\n" + "=" * 80)
print("SPATIAL FEATURE BOOST")
print("=" * 80)
if boost_path.exists():
    d = json.load(open(boost_path))
    fb = d.get("full_base", {}).get("acc")
    if fb: print(f"Full test baseline: {fb:.2f}%")
    FEATS = ["L9_F387","L14_F10561","L11_F12278","L9_F7540","L4_F14233","L6_F7539","L11_F9639","L13_F15219","L15_F220","L12_F2257"]
    print(f"\n{'Feature':<14} {'rF n':>5} {'rF base':>8}  {'SPAT Δ_full':>12} {'SPAT Δ_rF':>10} {'W.5 Δ_rF':>9} {'W1 Δ_rF':>9} {'W3 Δ_rF':>9}")
    for ft in FEATS:
        bk = f"{ft}_rF_base"
        if bk not in d: continue
        b = d[bk]
        def bd(k):
            r = d.get(k, {})
            if not r: return "—"
            return f"{max(v.get('delta',-999) for v in r.values()):+.1f}"
        print(f"  {ft:<12} {b['n']:>5} {b['acc']:>7.2f}%  {bd(ft+'_spatial_full'):>12} {bd(ft+'_spatial_rF'):>10} {bd(ft+'_wdec_g0.5_rF'):>9} {bd(ft+'_wdec_g1.0_rF'):>9} {bd(ft+'_wdec_g3.0_rF'):>9}")
else:
    print("(no boost results yet)")
