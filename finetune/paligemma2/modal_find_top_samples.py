"""
Step 1 of auto-interp: Find top-activating samples per feature on Modal (GPU).
Saves results to volume, then we run GPT-4o-mini API calls locally.

Usage:
    export HF_TOKEN=hf_...
    MODAL_PROFILE=hunar-oxford modal run modal_find_top_samples.py
"""
import os
import json
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("vlm-scope-top-samples")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers>=4.44", "sae-lens>=4.0",
        "nnsight>=0.3", "datasets", "h5py", "tqdm", "huggingface-hub",
        "Pillow", "numpy", "accelerate",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_MODE": "disabled",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

RESULTS_BASE = "/vol/results/paligemma2"
SAE_TYPE = "jumprelu"
TOP_K = 10  # 5 for interp + 5 for validation


@app.function(image=image, gpu="A100", volumes={"/vol": volume}, timeout=7200)
def find_top_samples():
    """Scan cached activations, encode through SAE, find top-activating samples."""
    import sys
    import torch
    import h5py
    import pandas as pd
    from pathlib import Path
    from collections import defaultdict
    from tqdm import tqdm

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_jumprelu_sae

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    ckpt_dir = Path(RESULTS_BASE) / f"run{sae_suffix}" / "checkpoints"
    act_dir = Path(RESULTS_BASE) / "run" / "activations"

    # Load feature list
    csv_path = Path(RESULTS_BASE) / "analysis" / f"final_features{sae_suffix}" / "final_spatial_visual_features.csv"
    if not csv_path.exists():
        print(f"[ERROR] No CSV at {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    features = [(int(row["layer"]), int(row["feature"])) for _, row in df.iterrows()]
    print(f"[INFO] {len(features)} features to process")

    layer_features = defaultdict(list)
    for layer, feat in features:
        layer_features[layer].append(feat)

    results = {}

    # H5 structure: chunk_X_Y.h5 / layer_{idx} / sample_{idx} -> (seq, 2304)
    # Only scan first 5 chunks (5K samples) — enough for top-10 per feature
    chunk_files = sorted(act_dir.glob("chunk_*.h5"))[:5]
    print(f"[INFO] Scanning {len(chunk_files)} activation chunks (5K samples)")

    for layer_idx in sorted(layer_features.keys()):
        feats = layer_features[layer_idx]
        print(f"\n[L{layer_idx}] {len(feats)} features")

        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP — no checkpoint")
            continue

        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device="cuda", cache_dir="/vol/cache/huggingface")
        sae.eval()

        # Per-feature top-k tracker
        feat_top = {f: [] for f in feats}
        layer_key = f"layer_{layer_idx}"

        for cf in tqdm(chunk_files, desc=f"L{layer_idx} chunks"):
            try:
                with h5py.File(str(cf), "r") as hf:
                    if layer_key not in hf:
                        continue
                    layer_grp = hf[layer_key]
                    for sample_key in layer_grp.keys():
                        if not sample_key.startswith("sample_"):
                            continue
                        sample_idx = int(sample_key.split("_")[1])
                        act = torch.from_numpy(layer_grp[sample_key][:]).float().to("cuda")

                        with torch.no_grad():
                            codes = sae.encode(act)  # (seq, d_sae)

                        for fi in feats:
                            max_act = codes[:, fi].max().item()
                            if max_act > 0:
                                heap = feat_top[fi]
                                entry = (max_act, sample_idx, "vqa")
                                if len(heap) < TOP_K:
                                    heap.append(entry)
                                    heap.sort(key=lambda x: x[0])
                                elif max_act > heap[0][0]:
                                    heap[0] = entry
                                    heap.sort(key=lambda x: x[0])
            except Exception as e:
                print(f"  [WARN] {cf.name}: {e}")

        del sae
        torch.cuda.empty_cache()

        for fi in feats:
            key = f"L{layer_idx}_F{fi}"
            top = sorted(feat_top[fi], key=lambda x: -x[0])
            results[key] = [{"activation": act, "sample_idx": sidx, "dataset": ds}
                           for act, sidx, ds in top]
            if top:
                print(f"  {key}: top={top[0][0]:.4f}, n={len(top)}")
            else:
                print(f"  {key}: NO activating samples")

    # Save
    out_dir = Path(RESULTS_BASE) / "analysis" / "autointerp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "top_samples.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    print(f"\n[DONE] Saved {len(results)} features to {out_path}")
    return results


@app.local_entrypoint()
def main():
    results = find_top_samples.remote()
    # Also save locally for the local interp script
    local_path = Path("top_samples.json")
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved locally to {local_path}")
    print(f"Features with samples: {sum(1 for v in results.values() if v)}/{len(results)}")
