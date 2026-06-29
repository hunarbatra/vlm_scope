#!/usr/bin/env python3
"""
Step 0.5 — Aggregate 5 sampled runs of mix-448 on HallusionBench and split by
answer-consistency.

Output splits:
  - **robustly-CORRECT**   (all 5 runs correct)    → sample-level specificity control
  - **robustly-INCORRECT** (all 5 runs incorrect)  → target eval set (target of ablation)
  - **borderline**         (anything else)          → discarded; model isn't consistent

Writes:
  analysis_hallucination/consistency/consistency_summary.json
  analysis_hallucination/consistency/consistency_per_sample.jsonl
  analysis_hallucination/judgments/mix448_hallusionbench_judgments.jsonl  (REWRITTEN with
    consistency-aware pred_label = ROBUST_CORRECT | ROBUST_INCORRECT | BORDERLINE)

Downstream scripts (30h, 32h, 35h, 45h) already look at pred_label==INCORRECT and
CORRECT; they'll need a small patch to accept ROBUST_* but we'll handle that in the
orchestration — for now we write the judgments file with raw labels CORRECT / INCORRECT /
BORDERLINE so downstream works unchanged: CORRECT ≡ robust-correct, INCORRECT ≡ robust-incorrect,
BORDERLINE is excluded.
"""
import json, re
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
RUNS_DIR = ROOT / "analysis_hallucination" / "consistency" / "runs"
OUT_DIR  = ROOT / "analysis_hallucination" / "consistency"
JUDGE_FILE = ROOT / "analysis_hallucination" / "judgments" / "mix448_hallusionbench_judgments.jsonl"
OLD_JUDGE_BACKUP = JUDGE_FILE.with_suffix(".jsonl.greedy_backup")

YES_RE = re.compile(r"^\s*(?:yes|y|true|correct|right|affirmative|1\b)", re.IGNORECASE)
NO_RE  = re.compile(r"^\s*(?:no|n|false|incorrect|wrong|negative|0\b)", re.IGNORECASE)


def parse_yesno(text):
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
    # Gather every per-run JSONL file across all 8 GPUs × 5 runs
    # (filenames: run_{i}_mix448_hb_gpu{g}.jsonl)
    per_sample = defaultdict(lambda: {"meta": None, "runs": []})
    for p in sorted(RUNS_DIR.glob("run_*_mix448_hb_gpu*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            uid = r.get("uid")
            if not uid: continue
            if per_sample[uid]["meta"] is None:
                per_sample[uid]["meta"] = {
                    "category": r.get("category"),
                    "subcategory": r.get("subcategory"),
                    "question": r.get("question"),
                    "gt_answer": r.get("gt_answer"),
                }
            pred = parse_yesno(r.get("response"))
            per_sample[uid]["runs"].append({
                "run_idx": r.get("run_idx"),
                "response": r.get("response"),
                "pred": pred,
            })

    # Classify each sample
    label_counts = Counter()
    out_rows = []
    judgments = []
    by_subcat = defaultdict(lambda: {"ROBUST_CORRECT": 0, "ROBUST_INCORRECT": 0, "BORDERLINE": 0})

    for uid, d in per_sample.items():
        meta = d["meta"]; runs = d["runs"]
        gt = str(meta.get("gt_answer", "")).strip()
        n_correct = sum(1 for r in runs if r["pred"] is not None and r["pred"] == gt)
        n_incorrect = sum(1 for r in runs if r["pred"] is not None and r["pred"] != gt)
        n_parsed = n_correct + n_incorrect
        n_total = len(runs)

        if n_parsed == n_total and n_correct == n_total:
            label = "ROBUST_CORRECT"
        elif n_parsed == n_total and n_incorrect == n_total:
            label = "ROBUST_INCORRECT"
        else:
            label = "BORDERLINE"
        label_counts[label] += 1
        by_subcat[meta.get("subcategory", "?")][label] += 1

        out_rows.append({
            "uid": uid, **meta,
            "n_runs": n_total, "n_correct": n_correct, "n_incorrect": n_incorrect,
            "consistency_label": label,
            "run_preds": [r["pred"] for r in runs],
            "run_responses": [r["response"] for r in runs],
        })
        judgments.append({
            "uid": uid, **meta,
            "response": runs[0]["response"],   # keep first-run response as canonical
            "pred_label": {"ROBUST_CORRECT": "CORRECT",
                           "ROBUST_INCORRECT": "INCORRECT",
                           "BORDERLINE": "BORDERLINE"}[label],
            "judge_status": "consistency",
            "n_runs": n_total, "n_correct": n_correct,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "consistency_per_sample.jsonl", "w") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Backup greedy-pass judgments and overwrite with consistency-aware ones
    if JUDGE_FILE.exists() and not OLD_JUDGE_BACKUP.exists():
        JUDGE_FILE.rename(OLD_JUDGE_BACKUP)
        print(f"[INFO] backed up greedy judgments to {OLD_JUDGE_BACKUP.name}")
    with open(JUDGE_FILE, "w") as f:
        for r in judgments:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] rewrote {JUDGE_FILE} with {len(judgments)} consistency-aware judgments")

    summary = {
        "n_total": sum(label_counts.values()),
        "label_counts": dict(label_counts),
        "by_subcategory": {k: dict(v) for k, v in by_subcat.items()},
    }
    with open(OUT_DIR / "consistency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total = summary["n_total"]
    print(f"\n[HEADLINE]  total={total}")
    for label, n in label_counts.most_common():
        print(f"  {label:>16}: {n:>4}  ({n/total*100:.2f}%)")
    print(f"\n[PER-SUBCAT]")
    for sc, d in sorted(by_subcat.items()):
        n = sum(d.values())
        rc = d["ROBUST_CORRECT"]; ri = d["ROBUST_INCORRECT"]; bl = d["BORDERLINE"]
        print(f"  {sc:>10}  n={n:>4}  robust_correct={rc:>4}  robust_incorrect={ri:>4}  borderline={bl:>4}")
    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
