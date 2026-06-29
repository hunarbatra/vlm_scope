#!/usr/bin/env python3
"""
Build a balanced multimodal subset of JailBreakV-28K that excludes Template
(text-only jailbreak wrappers) and focuses on the 7,664 genuinely-multimodal
attacks (figstep, SD, SD_typo, typo, Persuade, Logic).

Downloads the required image files from HF to
/data1/vlm_scope_sae_mix448_textonly/analysis_safety/jailbreakv_images/

Output: /data1/vlm_scope_sae_mix448_textonly/analysis_safety/jailbreakv_subset.jsonl
  {id, format, policy, jailbreak_query, redteam_query, image_path, image_local_path}
"""
import csv, json, os
from pathlib import Path
from huggingface_hub import hf_hub_download

os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR"
HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "JailbreakV-28K/JailBreakV-28k"

CSV_PATH = "/data1/hf_cache/hub/datasets--JailbreakV-28K--JailBreakV-28k/snapshots/f949ca582fff13d396ac8fce59596afafb2b78d3/JailBreakV_28K/JailBreakV_28K.csv"
OUT_FILE = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_safety/jailbreakv_subset.jsonl")
IMG_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_safety/jailbreakv_images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Multimodal formats only — skip 'Template' since its image is decorative
KEEP_FORMATS = {"figstep", "SD", "SD_typo", "typo", "Persuade", "Logic"}


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    subset = [r for r in rows if r["format"] in KEEP_FORMATS]
    print(f"[INFO] total rows: {len(rows)}, multimodal subset: {len(subset)}", flush=True)

    # Download images (resume-safe)
    n_need = len(subset)
    n_have = 0
    n_fail = 0
    records = []
    for i, r in enumerate(subset):
        img_remote = f"JailBreakV_28K/{r['image_path']}"
        img_local  = IMG_DIR / r["image_path"]
        img_local.parent.mkdir(parents=True, exist_ok=True)
        if not img_local.exists():
            try:
                p = hf_hub_download(
                    repo_id=REPO_ID, filename=img_remote,
                    repo_type="dataset", token=HF_TOKEN,
                    local_dir=None,   # use default hub cache
                )
                # symlink to our flat dir
                import shutil
                shutil.copy2(p, img_local)
            except Exception as e:
                n_fail += 1
                if n_fail <= 10: print(f"  [FAIL] {img_remote}: {e}", flush=True)
                continue
        n_have += 1
        records.append({
            "id": r["id"],
            "format": r["format"],
            "policy": r["policy"],
            "jailbreak_query": r["jailbreak_query"],
            "redteam_query": r["redteam_query"],
            "image_path": r["image_path"],
            "image_local_path": str(img_local),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_need}  ok={n_have} fail={n_fail}", flush=True)

    with open(OUT_FILE, "w") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[DONE] wrote {len(records)} records to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
