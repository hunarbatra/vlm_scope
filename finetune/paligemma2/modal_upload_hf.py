"""
Upload SAE checkpoints and cached activations from Modal Volume to HuggingFace.

Usage:
    cd finetune/paligemma2
    MODAL_PROFILE=hunar-oxford modal run modal_upload_hf.py

Creates two HF repos:
  - hunarbatra/paligemma2-sae-checkpoints  (SAE weights, ~15GB)
  - hunarbatra/paligemma2-vqav2-activations (cached H5 activations, large)
"""

import os
import modal

VOLUME_NAME = "vlm-scope-data-v2"
RESULTS_BASE = "/vol/results/paligemma2"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = "hunarbatra"

app = modal.App("vlm-scope-upload-hf")
volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub", "tqdm")
    .env({
        "HF_TOKEN": HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
    })
)


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=86400,
)
def upload_checkpoints():
    """Upload SAE checkpoints (pretrained + random) to HuggingFace."""
    from pathlib import Path
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=HF_TOKEN)
    repo_id = f"{HF_USERNAME}/vlm_scope_paligemma2_sae"

    # Create repo if needed
    create_repo(repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN)

    checkpoint_dir = Path(RESULTS_BASE) / "run" / "checkpoints"
    log_dir = Path(RESULTS_BASE) / "run" / "logs"

    # Upload final checkpoints (skip optimizer states)
    print("[INFO] Uploading SAE checkpoints...")
    for method in ["pretrained", "random"]:
        for pt_file in sorted(checkpoint_dir.glob(f"{method}_layer_*.pt")):
            if pt_file.name.endswith("_optim.pt"):
                continue
            remote_path = f"{method}/{pt_file.name}"
            print(f"  Uploading {pt_file.name} -> {remote_path}")
            api.upload_file(
                path_or_fileobj=str(pt_file),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )

    # Upload intermediate checkpoints (25/50/75%)
    run_dir = Path(RESULTS_BASE) / "run"
    for pct in [25, 50, 75]:
        pct_dir = run_dir / f"checkpoint_{pct}pct"
        if not pct_dir.exists():
            continue
        for pt_file in sorted(pct_dir.glob("*.pt")):
            remote_path = f"intermediate/{pct}pct/{pt_file.name}"
            print(f"  Uploading {pt_file.name} -> {remote_path}")
            api.upload_file(
                path_or_fileobj=str(pt_file),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )

    # Upload training logs
    print("[INFO] Uploading training logs...")
    for log_file in sorted(log_dir.glob("*.csv")):
        remote_path = f"logs/{log_file.name}"
        print(f"  Uploading {log_file.name} -> {remote_path}")
        api.upload_file(
            path_or_fileobj=str(log_file),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )

    # Upload a README
    readme = """# PaliGemma2-3B SAE Checkpoints

Sparse Autoencoders (TopK, k=50) trained on PaliGemma2-3B residual stream activations using VQAv2.

## Architecture
- **Base model**: google/paligemma2-3b-pt-224 (Gemma 2B backbone, 26 layers)
- **SAE type**: TopK with k=50, width 16,384, d_in=2,304
- **Training data**: 50,000 VQAv2 validation samples
- **LR**: 2e-4 / sqrt(d_sae / 16384)

## Methods
- `pretrained/`: Initialized from Gemma Scope 2B residual SAEs, then fine-tuned on VQAv2
- `random/`: Randomly initialized, trained on VQAv2
- `intermediate/`: Checkpoints at 25%, 50%, 75% training (pretrained only)

## Files
- `{method}_layer_{i}.pt`: SAE state_dict for layer i
- `logs/metrics_{method}_layer_{i}.csv`: Training metrics (tokens, FVU, recon_loss, sparsity)
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"[SUCCESS] Checkpoints uploaded to https://huggingface.co/{repo_id}")
    return f"Checkpoints uploaded to {repo_id}"


@app.local_entrypoint()
def main():
    print("Starting upload to HuggingFace...")
    result = upload_checkpoints.remote()
    print(result)
    print("\n[DONE] Upload complete!")
