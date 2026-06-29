#!/usr/bin/env python3
"""
MathVerse pipeline monitor — prints current status of all 5 steps.
Run anytime: python3 mathverse_monitor.py
"""
import json, os, subprocess
from pathlib import Path

SAE_ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
LOG_DIR  = Path("/tmp/mathverse_logs")

def check_file(p): return "✓" if p.exists() else "…"

def tail_log(path, n=3):
    if not path.exists(): return "  (no log)"
    try:
        lines = path.read_text().strip().split("\n")
        return "\n".join(f"  {l}" for l in lines[-n:])
    except: return "  (unreadable)"

def main():
    print("=" * 70)
    print("MATHVERSE PIPELINE STATUS")
    print("=" * 70)

    # Step 1: hidden cache
    cache_dir = SAE_ROOT / "analysis_mathverse/mix_hidden"
    corr_path = SAE_ROOT / "analysis_mathverse/correctness.json"
    n_cached = len(list(cache_dir.glob("vi_*.pt"))) if cache_dir.exists() else 0
    print(f"\n[Step 1] Hidden cache: {n_cached}/430 files {check_file(corr_path)}")
    if corr_path.exists():
        try:
            d = json.load(open(corr_path))
            print(f"  train_acc={d.get('acc_train',0):.1%}  test_acc={d.get('acc_test',0):.1%}"
                  f"  ({d.get('n_correct_train',0)}c/{d.get('train_end',0)}t train,"
                  f"  {d.get('n_correct_test',0)}c/{430-d.get('train_end',344)}t test)")
        except: pass
    print(tail_log(LOG_DIR / "step1.log"))

    # Step 2: firing
    fire_dir = SAE_ROOT / "analysis_mathverse/firing"
    n_fire = len(list(fire_dir.glob("firing_L*.json"))) if fire_dir.exists() else 0
    ranked = SAE_ROOT / "analysis_mathverse/fisher_ranked.json"
    print(f"\n[Step 2] Firing stats: {n_fire}/26 layers {check_file(ranked)}")
    if ranked.exists():
        try:
            r = json.load(open(ranked))
            print(f"  Top 5 by Fisher:")
            for x in r[:5]:
                print(f"    {x['key']:<14} OR={x['or']:.3f}  diff={x['diff']:+.3f}"
                      f"  fires={x['fire_corr']}c/{x['fire_incorr']}i  p={x['pval']:.2e}")
        except: pass
    print(tail_log(LOG_DIR / "step2.log"))

    # Step 3: ablation
    abl_path = SAE_ROOT / "analysis_mathverse/ablation_results.json"
    print(f"\n[Step 3] Causal ablation: {check_file(abl_path)}")
    if abl_path.exists():
        try:
            abl = json.load(open(abl_path))
            base = abl.get("base", {}).get("acc", 0)
            drops = [(k,v) for k,v in abl.items() if k!="base" and "drop" in v]
            drops.sort(key=lambda x: x[1]["drop"])
            print(f"  Base test acc: {base:.2f}%  ({len(drops)} features ablated)")
            print(f"  Top 5 by drop:")
            for k, v in drops[:5]:
                print(f"    {k:<14} drop={v['drop']:+.2f}%  fisher={v['fisher_score']:.2f}")
        except: pass
    print(tail_log(LOG_DIR / "step3.log"))

    # Step 4: sae acts
    sae_acts_dir = SAE_ROOT / "analysis_mathverse/sae_acts"
    n_acts = len(list(sae_acts_dir.glob("acts_*.json"))) if sae_acts_dir.exists() else 0
    print(f"\n[Step 4] SAE acts: {n_acts}/5 files")
    if sae_acts_dir.exists():
        for f in sorted(sae_acts_dir.glob("acts_*.json")):
            try:
                d = json.load(open(f))
                print(f"  {f.stem}: train_firing={d.get('n_firing_train','?')}  "
                      f"test_firing={d.get('n_firing_test','?')}")
            except: pass
    print(tail_log(LOG_DIR / "step4.log"))

    # Step 5: CAA recipe results
    res_path = SAE_ROOT / "analysis_mathverse/caa_recipe_results/results.json"
    print(f"\n[Step 5] CAA recipe eval: {check_file(res_path)}")
    if res_path.exists():
        try:
            res = json.load(open(res_path))
            bases = {k: v for k, v in res.items() if k.endswith("_rF_base")}
            print(f"  {len(bases)} feature baselines computed")
            # Find best deltas
            best = []
            for k, v in res.items():
                if "delta" in v and v["delta"] is not None:
                    best.append((k, v["delta"]))
            best.sort(key=lambda x: -x[1])
            print(f"  Best deltas:")
            for k, d in best[:5]:
                print(f"    {k}: Δ={d:+.2f}%")
        except: pass
    print(tail_log(LOG_DIR / "step5.log"))

    # Running processes
    print(f"\n[Procs] Running mathverse jobs:")
    try:
        out = subprocess.check_output(
            "ps aux | grep 'mathverse_step' | grep -v grep | awk '{print $2, substr($0,index($0,$11))}'",
            shell=True, text=True).strip()
        print("  " + out if out else "  (none)")
    except: pass

    # GPU usage
    print(f"\n[GPUs]")
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits",
            shell=True, text=True).strip()
        for line in out.split("\n"):
            parts = [x.strip() for x in line.split(",")]
            used = int(parts[1]); free = int(parts[2])
            busy = "BUSY" if used > 1000 else "free"
            print(f"  GPU{parts[0]}: {used}MB used ({busy})")
    except: pass

    print("=" * 70)

if __name__ == "__main__":
    main()
