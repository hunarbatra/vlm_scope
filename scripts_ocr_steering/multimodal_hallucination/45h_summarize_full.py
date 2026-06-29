#!/usr/bin/env python3
"""
Step 11 (hallucination) — build the final table from the ablation sweep.

For each feature F in features_to_ablate.csv:
  1. Read analysis_hallucination/ablation_results/responses_L{L}_F{F}.jsonl
  2. Parse each ablated yes/no response with regex → CORRECT / INCORRECT (vs gt_answer).
     (NO EXTERNAL LLM JUDGE — binary ground truth lives in HallusionBench.)
  3. Compute ablated HallusionBench accuracy on the (originally-INCORRECT) eval set.
     Baseline = 0% correct by construction (every sample was INCORRECT at baseline).
     delta_halluc_acc = ablated_acc − 0  (positive = features whose removal restores accuracy)
  4. Read vqa_L{L}_F{F}.json for ΔVQA.

Output:
  analysis_hallucination/final_ablation_table.csv
  analysis_hallucination/final_best_per_subcategory.csv
"""
import os, json, re, warnings
from pathlib import Path
from collections import defaultdict
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
ANALYSIS_DIR = ROOT / "analysis_hallucination"
FEATURES_CSV = ANALYSIS_DIR / "ablation_input" / "features_to_ablate.csv"
ABLATION_DIR = ANALYSIS_DIR / "ablation_results"
OUT_CSV      = ANALYSIS_DIR / "final_ablation_table.csv"
OUT_BEST_SC  = ANALYSIS_DIR / "final_best_per_subcategory.csv"

YES_RE = re.compile(r"^\s*(?:yes|y|true|correct|right|affirmative|1\b)", re.IGNORECASE)
NO_RE  = re.compile(r"^\s*(?:no|n|false|incorrect|wrong|negative|0\b)", re.IGNORECASE)


def parse_yesno(text: str):
    if not text: return None
    t = text.strip().split("\n")[0]
    if YES_RE.search(t): return "1"
    if NO_RE.search(t):  return "0"
    has_yes = bool(re.search(r"\byes\b", t, re.IGNORECASE))
    has_no  = bool(re.search(r"\bno\b",  t, re.IGNORECASE))
    if has_yes and not has_no: return "1"
    if has_no and not has_yes: return "0"
    return None


def compute_ablated_acc(resp_path: Path):
    n_correct = n_judged = 0
    per_sc = defaultdict(lambda: {"c": 0, "t": 0})
    for line in open(resp_path):
        r = json.loads(line)
        if r.get("status") != "ok" or not r.get("response"): continue
        pred = parse_yesno(r["response"])
        if pred is None: continue
        gt = str(r.get("gt_answer", "")).strip()
        n_judged += 1
        sc = r.get("subcategory", "?")
        per_sc[sc]["t"] += 1
        if pred == gt:
            n_correct += 1
            per_sc[sc]["c"] += 1
    acc = n_correct / max(n_judged, 1) * 100
    per_sc_acc = {s: d["c"] / max(d["t"],1) * 100 for s, d in per_sc.items()}
    return acc, n_correct, n_judged, per_sc_acc


def main():
    if not FEATURES_CSV.exists():
        print(f"[FATAL] {FEATURES_CSV} missing"); return
    df_feat = pd.read_csv(FEATURES_CSV)
    print(f"[MAIN] {len(df_feat)} features")

    rows = []
    n_skipped = 0
    for _, row in df_feat.iterrows():
        L = int(row["layer"]); F = int(row["feature"])
        resp_path      = ABLATION_DIR / f"responses_L{L}_F{F}.jsonl"
        resp_ctrl_path = ABLATION_DIR / f"responses_correct_L{L}_F{F}.jsonl"
        vqa_path       = ABLATION_DIR / f"vqa_L{L}_F{F}.json"
        if not resp_path.exists():
            n_skipped += 1; continue

        # INCORRECT eval (target) — baseline acc = 0% by construction
        acc, n_c, n_j, per_sc_acc = compute_ablated_acc(resp_path)

        # CORRECT control eval — baseline acc = 100% by construction
        ctrl_acc = None; ctrl_n = 0; delta_retention = None
        if resp_ctrl_path.exists():
            ctrl_acc, _, ctrl_n, _ = compute_ablated_acc(resp_ctrl_path)
            delta_retention = ctrl_acc - 100.0  # negative = ablation broke formerly-correct answers

        vqa_base = vqa_abl = delta_vqa = None
        if vqa_path.exists():
            vj = json.load(open(vqa_path))
            vqa_base = vj.get("baseline_vqa_acc")
            vqa_abl  = vj.get("vqa_acc")
            delta_vqa = vj.get("delta_vqa")

        rows.append({
            "layer": L, "feature": F,
            "subcategory": row.get("primary_subcategory", row.get("selected_for_subcategory", "?")),
            "is_control": int(row["is_control"]),
            "odds_ratio": float(row.get("odds_ratio", 0.0)),
            "baseline_acc_on_incorrect_set": 0.0,
            "ablated_acc": acc,
            "delta_halluc_acc": acc,                    # ablated_acc − 0  (+ = flipped INCORRECT→CORRECT)
            "n_judged": n_j,
            "per_subcat_acc": json.dumps(per_sc_acc),
            "ablated_correct_acc": ctrl_acc,            # ablated acc on samples that were baseline-CORRECT
            "delta_correct_retention": delta_retention, # ablated_correct_acc − 100 (≈0 = good, negative = specificity hit)
            "n_correct_control": ctrl_n,
            "baseline_vqa_acc": vqa_base,
            "ablated_vqa_acc": vqa_abl,
            "delta_vqa": delta_vqa,
        })
    if not rows:
        print("[WARN] no ablation results found"); return
    df = pd.DataFrame(rows).sort_values(["is_control", "delta_halluc_acc"], ascending=[True, False])
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] wrote {OUT_CSV} ({len(df)} rows)")

    test = df[df.is_control == 0]
    ctrl = df[df.is_control == 1]

    print("\n" + "="*82)
    print("BASELINES — HallusionBench accuracy on INCORRECT-eval set = 0% (by construction)")
    print("           VQA yes/no ≈ 86.5%")
    print("="*82)
    print(f"\n{'Group':>8} | {'n':>4} | {'Δ Halluc_Acc':>14} | {'Δ Correct_Ret':>15} | {'Δ VQA':>8}")
    print("-"*80)
    for lbl, sub in [("test", test), ("control", ctrl)]:
        if len(sub) == 0: continue
        dv  = sub.delta_vqa.mean() if sub.delta_vqa.notna().any() else 0.0
        dcr = sub.delta_correct_retention.mean() if "delta_correct_retention" in sub.columns and sub.delta_correct_retention.notna().any() else 0.0
        print(f"{lbl:>8} | {len(sub):>4} | "
              f"{sub.delta_halluc_acc.mean():>+13.2f}% | "
              f"{dcr:>+14.2f}% | "
              f"{dv:>+7.2f}%")

    print("\n=== Top-30 features by Δ Halluc_Acc (most hallucination reduction) ===")
    top = test.sort_values("delta_halluc_acc", ascending=False).head(30).copy()
    for col in ["odds_ratio","delta_halluc_acc","delta_vqa","ablated_acc","delta_correct_retention"]:
        if col in top.columns: top[col] = top[col].astype(float).round(2)
    cols_to_show = [c for c in ["layer","feature","subcategory","ablated_acc",
                                "delta_halluc_acc","delta_correct_retention",
                                "delta_vqa","odds_ratio"] if c in top.columns]
    print(top[cols_to_show].to_string(index=False))

    print("\n=== Best feature per subcategory (test only) ===")
    best = (test.sort_values("delta_halluc_acc", ascending=False)
                .groupby("subcategory").head(1))
    best_cols = [c for c in ["subcategory","layer","feature","ablated_acc",
                             "delta_halluc_acc","delta_correct_retention",
                             "delta_vqa","odds_ratio"] if c in best.columns]
    best[best_cols].to_csv(OUT_BEST_SC, index=False)
    print(best[best_cols].to_string(index=False))


if __name__ == "__main__":
    main()
