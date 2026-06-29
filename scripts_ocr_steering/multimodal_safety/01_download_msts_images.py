#!/usr/bin/env python3
"""
Phase A, Step 1 — Download the 200 MSTS unsafe images to local disk.
Images are cached by unsafe_image_id -> <ID>.<ext>.

Usage:  python3 01_download_msts_images.py
"""
import csv, hashlib, os, sys, time
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests

MSTS_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_safety/msts")
IMG_DIR   = MSTS_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH  = MSTS_DIR / "unsafe_images.csv"

def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    print(f"[INFO] MSTS unsafe_images.csv: {len(rows)} rows", flush=True)
    n_ok = n_skip = n_fail = 0
    for i, r in enumerate(rows):
        img_id = r["unsafe_image_id"]
        url    = r["unsafe_image_url"]
        out_p  = IMG_DIR / f"{img_id}.jpg"
        if out_p.exists() and out_p.stat().st_size > 1024:
            n_skip += 1; continue
        # Wikimedia blocks generic UAs + thumbnail URLs rate-limit aggressively; rewrite to original
        candidates = [url]
        if "upload.wikimedia.org/wikipedia/commons/thumb/" in url:
            import re
            stripped = re.sub(r"/thumb(/[a-z0-9]/[a-z0-9]{2}/[^/]+)/[^/?]+(\?.*)?$", r"\1", url)
            stripped = stripped.split("?")[0]
            candidates.insert(0, stripped)
        ua = "VLMSafetyResearch/1.0 (https://github.com; contact@research.edu) requests/2.32"
        got = False
        last_err = None
        for attempt in range(3):
            for u in candidates:
                try:
                    resp = requests.get(u, timeout=25,
                                        headers={"User-Agent": ua, "Accept": "image/*,*/*",
                                                 "Accept-Language": "en-US,en;q=0.9"})
                    if resp.status_code == 429:
                        last_err = f"429 on {u[-60:]}"
                        time.sleep(5 + attempt * 5)  # backoff
                        continue
                    resp.raise_for_status()
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    img.thumbnail((1024, 1024))
                    img.save(out_p, "JPEG", quality=90)
                    got = True; break
                except Exception as e:
                    last_err = e; continue
            if got: break
            time.sleep(2 + attempt * 2)
        if got:
            n_ok += 1
            if (i+1) % 20 == 0:
                print(f"  downloaded {i+1}/{len(rows)}  ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
        else:
            n_fail += 1
            if n_fail <= 20:
                print(f"  [FAIL] {img_id} ({r.get('unsafe_image_cw','')}): {last_err}", flush=True)
        time.sleep(0.8)   # heavier throttle to avoid wikimedia bot policy
    print(f"[DONE] ok={n_ok} skip={n_skip} fail={n_fail} total={len(rows)}", flush=True)

if __name__ == "__main__":
    main()
