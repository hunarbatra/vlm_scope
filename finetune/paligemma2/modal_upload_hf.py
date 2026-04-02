"""
Upload SAE checkpoints and training logs from Modal Volume to HuggingFace.

Uploads both TopK and JumpReLU checkpoints, intermediate checkpoints,
and training logs.

Usage:
    cd finetune/paligemma2
    MODAL_PROFILE=hunar-oxford modal run modal_upload_hf.py
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


def _upload_run(api, repo_id, run_name, run_dir, methods, hf_prefix):
    """Upload checkpoints, intermediate checkpoints, and logs from a run directory."""
    from pathlib import Path

    checkpoint_dir = Path(run_dir) / "checkpoints"
    log_dir = Path(run_dir) / "logs"

    # Upload final checkpoints (skip optimizer states)
    if checkpoint_dir.exists():
        print(f"[INFO] Uploading {run_name} checkpoints...")
        for method in methods:
            for pt_file in sorted(checkpoint_dir.glob(f"{method}_layer_*.pt")):
                if pt_file.name.endswith("_optim.pt"):
                    continue
                remote_path = f"{hf_prefix}/{method}/{pt_file.name}"
                print(f"  {pt_file.name} -> {remote_path}")
                api.upload_file(
                    path_or_fileobj=str(pt_file),
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="model",
                )

    # Upload intermediate checkpoints (25/50/75%)
    for pct in [25, 50, 75]:
        pct_dir = Path(run_dir) / f"checkpoint_{pct}pct"
        if not pct_dir.exists():
            continue
        for pt_file in sorted(pct_dir.glob("*.pt")):
            remote_path = f"{hf_prefix}/intermediate/{pct}pct/{pt_file.name}"
            print(f"  {pt_file.name} -> {remote_path}")
            api.upload_file(
                path_or_fileobj=str(pt_file),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )

    # Upload training logs
    if log_dir.exists():
        print(f"[INFO] Uploading {run_name} training logs...")
        for log_file in sorted(log_dir.glob("*.csv")):
            remote_path = f"{hf_prefix}/logs/{log_file.name}"
            print(f"  {log_file.name} -> {remote_path}")
            api.upload_file(
                path_or_fileobj=str(log_file),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=86400,
)
def upload_checkpoints():
    """Upload all SAE checkpoints (TopK + JumpReLU) to HuggingFace."""
    from pathlib import Path
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=HF_TOKEN)
    repo_id = f"{HF_USERNAME}/vlm_scope_paligemma2_sae"

    # Create repo if needed
    create_repo(repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN)

    # Upload TopK SAE run
    topk_dir = Path(RESULTS_BASE) / "run"
    if topk_dir.exists():
        _upload_run(api, repo_id, "TopK", str(topk_dir),
                    methods=["pretrained", "random"], hf_prefix="topk")

    # Upload JumpReLU SAE run
    jumprelu_dir = Path(RESULTS_BASE) / "run_jumprelu"
    if jumprelu_dir.exists():
        _upload_run(api, repo_id, "JumpReLU", str(jumprelu_dir),
                    methods=["pretrained"], hf_prefix="jumprelu")

    # Upload a README
    readme = """# PaliGemma2-3B SAE Checkpoints

Sparse Autoencoders trained on PaliGemma2-3B residual stream activations using VQAv2.

## Architecture
- **Base model**: google/paligemma2-3b-pt-224 (Gemma 2B backbone, 26 layers)
- **SAE width**: 16,384 features, d_in=2,304
- **Training data**: 50,000 VQAv2 validation samples

## TopK SAE (`topk/`)
- **Activation**: top-k selection, k=50
- **LR**: 2e-4 / sqrt(d_sae / 16384)
- **Init**: Gemma Scope 2B (`pretrained/`) or random (`random/`)
- Note: Architecture mismatch — Gemma Scope natively uses JumpReLU

## JumpReLU SAE (`jumprelu/`) — Recommended
- **Activation**: JumpReLU with learnable per-feature threshold
- **LR**: 7e-5, Adam betas=(0.0, 0.999)
- **Target L0**: 50, bandwidth=0.001, sparsity_coeff=1.0
- **Warmup**: 1000-step LR warmup, 2000-step sparsity warmup
- **Init**: Gemma Scope 2B (`pretrained/`) — architecture-matched, includes threshold
- Based on: github.com/saprmarks/dictionary_learning JumpReLU trainer

## Files
- `{type}/{method}/pretrained_layer_{i}.pt`: SAE state_dict for layer i
- `{type}/intermediate/{25,50,75}pct/`: Training dynamics checkpoints
- `{type}/logs/metrics_{method}_layer_{i}.csv`: Training metrics
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
