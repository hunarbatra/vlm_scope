#!/usr/bin/env python3
"""
Phase A.2 — Judge correctness on HallusionBench.

HallusionBench is yes/no with binary ground truth (gt_answer = '0' or '1').
We parse mix-448's first-line response with a yes/no regex, then compare vs gt.
No LLM judge needed.

Output fields per record:
  pred_label ∈ {CORRECT, INCORRECT, PARSE_FAIL}
  judge_status

Output file: analysis_hallucination/judgments/mix448_hallusionbench_judgments.jsonl
Summary:     analysis_hallucination/judgments/hallusionbench_summary.json

Usage: python3 -B 23h_judge_correctness.py
"""
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
INP = ROOT / "analysis_hallucination" / "responses" / "mix448_hallusionbench_responses.jsonl"
OUT = ROOT / "analysis_hallucination" / "judgments" / "mix448_hallusionbench_judgments.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

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


def main():
    if not INP.exists():
        print(f"[FATAL] {INP} missing — run 21h first"); return
    records = [json.loads(l) for l in open(INP) if l.strip()]
    print(f"[INFO] {len(records)} responses to judge", flush=True)

    by_cat = defaultdict(lambda: {"correct": 0, "incorrect": 0, "parse_fail": 0, "n": 0})
    by_subcat = defaultdict(lambda: {"correct": 0, "incorrect": 0, "parse_fail": 0, "n": 0})
    n_ok = n_inc = n_pf = 0

    out = open(OUT, "w")
    for r in records:
        if r.get("status") != "ok" or not r.get("response"):
            label, status = None, "no_response"
        else:
            pred = parse_yesno(r["response"])
            gt   = str(r.get("gt_answer", "")).strip()
            if pred is None:
                label, status = "PARSE_FAIL", "parse_fail"; n_pf += 1
            elif pred == gt:
                label, status = "CORRECT", "ok"; n_ok += 1
            else:
                label, status = "INCORRECT", "ok"; n_inc += 1
            cat = r.get("category", "?"); sc = r.get("subcategory", "?")
            by_cat[cat]["n"] += 1; by_subcat[sc]["n"] += 1
            if label == "CORRECT":
                by_cat[cat]["correct"] += 1; by_subcat[sc]["correct"] += 1
            elif label == "INCORRECT":
                by_cat[cat]["incorrect"] += 1; by_subcat[sc]["incorrect"] += 1
            else:
                by_cat[cat]["parse_fail"] += 1; by_subcat[sc]["parse_fail"] += 1
        rec2 = dict(r); rec2["pred_label"] = label; rec2["judge_status"] = status
        out.write(json.dumps(rec2, ensure_ascii=False) + "\n")
    out.close()

    total_judged = n_ok + n_inc
    acc = n_ok / max(total_judged, 1) * 100
    print(f"\n[HEADLINE] correct={n_ok} incorrect={n_inc} parse_fail={n_pf}")
    print(f"           accuracy = {acc:.2f}%   (hallucination rate = {100-acc:.2f}%)")
    print(f"           parse-fail rate = {n_pf/len(records)*100:.2f}%")

    print("\n[PER-CATEGORY]")
    for cat, d in sorted(by_cat.items()):
        a = d["correct"] / max(d["correct"]+d["incorrect"], 1) * 100
        print(f"  {cat:>3}  n={d['n']:>4}  correct={d['correct']:>4}  incorrect={d['incorrect']:>4}  parse_fail={d['parse_fail']:>3}  acc={a:.2f}%")
    print("\n[PER-SUB-CATEGORY]")
    for sc, d in sorted(by_subcat.items()):
        a = d["correct"] / max(d["correct"]+d["incorrect"], 1) * 100
        print(f"  {sc:>10}  n={d['n']:>4}  correct={d['correct']:>4}  incorrect={d['incorrect']:>4}  parse_fail={d['parse_fail']:>3}  acc={a:.2f}%")

    summary = {
        "total": len(records), "correct": n_ok, "incorrect": n_inc, "parse_fail": n_pf,
        "accuracy_pct": acc, "hallucination_rate_pct": 100 - acc,
        "by_category": {k: dict(v) for k, v in by_cat.items()},
        "by_subcategory": {k: dict(v) for k, v in by_subcat.items()},
    }
    with open(OUT.parent / "hallusionbench_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] wrote {OUT}")


if __name__ == "__main__":
    main()
