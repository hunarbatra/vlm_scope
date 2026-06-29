#!/usr/bin/env python3
"""
Step 12 (safety) — prepare 100 truly-benign control samples from MSSBench
(kzhou35/mssbench). Two sources:

  (a) embodied split (76 rows) — each has explicit safe_instruction +
      safe image. Robotic-task QA, but most explicitly "safe" by design.
  (b) chat split (top-up to 100) — each has a query that's used for both
      safe and unsafe images; in the safe-image context the (image, query)
      pair is benign.

We take all 76 embodied + 24 chat = 100 control samples.

Downloads images to analysis_safety/mssbench_safe/{chat,embodied}/ and writes
analysis_safety/mssbench_safe/safe_eval.jsonl.
"""
import json, random, time, urllib.request
from pathlib import Path

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
COMBINED = Path("/tmp/mssbench_combined.json")
OUT_DIR = ROOT / "analysis_safety" / "mssbench_safe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_PATH = OUT_DIR / "safe_eval.jsonl"

N_TARGET = 100
SEED = 7
HF_BASE = "https://huggingface.co/datasets/kzhou35/mssbench/resolve/main/"


def download_one(url, out_path):
    if out_path.exists() and out_path.stat().st_size > 1000:
        return "skip"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vlm-scope-bot/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out_path.write_bytes(r.read())
            return "ok"
        except Exception as e:
            if attempt == 2:
                print(f"  [ERR] {url}: {e}")
                return "err"
            time.sleep(0.5 * (attempt + 1))


def main():
    if not COMBINED.exists():
        print(f"[FATAL] {COMBINED} missing"); return
    d = json.load(open(COMBINED))
    random.seed(SEED)

    # --- (a) embodied split: 76 explicit safe_instruction rows ---
    emb_dir = OUT_DIR / "embodied"; emb_dir.mkdir(exist_ok=True)
    emb_records = []
    for i, r in enumerate(d.get("embodied", [])):
        instr = r.get("safe_instruction", "").strip()
        img = r.get("safe", "").strip()
        if not instr or not img: continue
        emb_records.append({
            "instruction_id": f"mssbench_emb_{i:03d}",
            "category": "MSSBench-safe-embodied",
            "sub_category": r.get("category", ""),
            "task": r.get("task", ""),
            "instruction": instr,
            "image_relpath": f"embodied/{img}",
            "image_description": r.get("observation_safe", ""),
            "safety_reason": "Benign embodied control: explicit safe_instruction with safe image.",
        })
    print(f"[INFO] embodied safe samples: {len(emb_records)}")

    # --- (b) chat split: top up to N_TARGET ---
    chat_dir = OUT_DIR / "chat"; chat_dir.mkdir(exist_ok=True)
    chat_pool = []
    for i, r in enumerate(d.get("chat", [])):
        qs = r.get("queries", []) or []
        if not qs: continue
        chat_pool.append({
            "instruction_id": f"mssbench_chat_{i:03d}",
            "category": "MSSBench-safe-chat",
            "sub_category": r.get("Type", ""),
            "instruction": qs[0].strip(),
            "image_relpath": f"chat/{r['safe_image_path']}",
            "image_description": "",
            "safety_reason": f"Benign chat control. Safe-pair intent: {r.get('intent','')}",
        })
    random.shuffle(chat_pool)
    needed = max(0, N_TARGET - len(emb_records))
    chat_records = chat_pool[:needed]
    print(f"[INFO] chat safe samples added: {len(chat_records)} (target total {N_TARGET})")

    all_records = emb_records + chat_records

    # --- download required images ---
    needed_paths = sorted({r["image_relpath"] for r in all_records})
    print(f"[INFO] downloading {len(needed_paths)} images...")
    counts = {"ok": 0, "skip": 0, "err": 0}
    for relp in needed_paths:
        out = OUT_DIR / relp
        out.parent.mkdir(parents=True, exist_ok=True)
        st = download_one(HF_BASE + relp, out)
        counts[st] = counts.get(st, 0) + 1
    print(f"[INFO] download tally: {counts}")

    # finalize records with absolute image paths
    with open(EVAL_PATH, "w") as f:
        for r in all_records:
            r2 = dict(r)
            r2["image_path"] = str(OUT_DIR / r["image_relpath"])
            r2.pop("image_relpath", None)
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    print(f"[DONE] wrote {EVAL_PATH} ({len(all_records)} rows)")


if __name__ == "__main__":
    main()
