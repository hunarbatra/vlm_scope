#!/usr/bin/env python3
"""
Upload mix-448 JumpReLU SAE checkpoints to HuggingFace.
Deletes remaining old pt-224 files, then uploads new checkpoints.

Structure on HF:
  jumprelu_mix448/pretrained/          - 26 final model checkpoints
  jumprelu_mix448/intermediate/25pct/  - 26 intermediate (25% training)
  jumprelu_mix448/intermediate/50pct/  - 26 intermediate (50% training)
  jumprelu_mix448/intermediate/75pct/  - 26 intermediate (75% training)

Usage: python3 upload_to_hf.py
"""

import os
import time
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationDelete, CommitOperationAdd

REPO_ID = "hunarbatra/vlm_scope_paligemma2_sae"
TOKEN = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

CKPT_DIR = Path("/data1/vlm_scope_sae_mix448/checkpoints")
CKPT_25 = Path("/data1/vlm_scope_sae_mix448/checkpoint_25pct")
CKPT_50 = Path("/data1/vlm_scope_sae_mix448/checkpoint_50pct")
CKPT_75 = Path("/data1/vlm_scope_sae_mix448/checkpoint_75pct")

api = HfApi(token=TOKEN)


def delete_old_files():
    """Delete remaining old pt-224 files in a single commit."""
    files = api.list_repo_files(REPO_ID)
    to_delete = [f for f in files if f not in ['.gitattributes', 'README.md']
                 and not f.startswith('jumprelu_mix448/')]
    if not to_delete:
        print("No old files to delete.")
        return
    print(f"Deleting {len(to_delete)} old files in one commit...")
    ops = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
    api.create_commit(
        repo_id=REPO_ID,
        operations=ops,
        commit_message="Remove old pt-224 base model SAE checkpoints",
    )
    print("Old files deleted.")


def upload_folder(local_dir, remote_prefix):
    """Upload all .pt files (excluding optim) from local_dir to remote_prefix/."""
    files = sorted(local_dir.glob("pretrained_layer_*.pt"))
    # Exclude optimizer files
    files = [f for f in files if "_optim" not in f.name]
    print(f"\nUploading {len(files)} files from {local_dir} -> {remote_prefix}/")

    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(local_dir),
        path_in_repo=remote_prefix,
        allow_patterns=["pretrained_layer_*.pt"],
        ignore_patterns=["*_optim*"],
        commit_message=f"Add mix-448 JumpReLU SAE: {remote_prefix}",
    )
    print(f"  Uploaded {remote_prefix}/")


def main():
    print(f"Repository: {REPO_ID}")
    print(f"Checkpoints:")
    for label, d in [("final", CKPT_DIR), ("25%", CKPT_25), ("50%", CKPT_50), ("75%", CKPT_75)]:
        n = len(list(d.glob("pretrained_layer_*.pt"))) - len(list(d.glob("*_optim*")))
        print(f"  {label}: {n} files in {d}")

    # Step 1: Delete old files
    print("\n=== Step 1: Delete old pt-224 files ===")
    try:
        delete_old_files()
    except Exception as e:
        if "429" in str(e):
            print(f"Rate limited. Waiting 10 minutes and retrying...")
            time.sleep(600)
            delete_old_files()
        else:
            raise

    # Step 2: Upload new checkpoints
    print("\n=== Step 2: Upload mix-448 JumpReLU SAE checkpoints ===")

    uploads = [
        (CKPT_DIR, "jumprelu_mix448/pretrained"),
        (CKPT_25, "jumprelu_mix448/intermediate/25pct"),
        (CKPT_50, "jumprelu_mix448/intermediate/50pct"),
        (CKPT_75, "jumprelu_mix448/intermediate/75pct"),
    ]

    for local_dir, remote_prefix in uploads:
        try:
            upload_folder(local_dir, remote_prefix)
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limited. Waiting 10 minutes and retrying...")
                time.sleep(600)
                upload_folder(local_dir, remote_prefix)
            else:
                raise

    # Verify
    print("\n=== Verification ===")
    files = api.list_repo_files(REPO_ID)
    print(f"Total files in repo: {len(files)}")
    prefixes = {}
    for f in files:
        p = '/'.join(f.split('/')[:-1]) or '(root)'
        prefixes[p] = prefixes.get(p, 0) + 1
    for p in sorted(prefixes):
        print(f"  {p}: {prefixes[p]} files")

    print("\nDone!")


if __name__ == "__main__":
    main()
