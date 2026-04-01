#!/usr/bin/env python3
"""
Orchestration script for SAE fine-tuning pipeline.
This script manages the workflow between activation caching and SAE training.
"""
import dotenv
dotenv.load_dotenv(".env")

import subprocess
import sys
import argparse
import shutil
from pathlib import Path
import wandb

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"[SUCCESS] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Orchestrate SAE fine-tuning pipeline")
    parser.add_argument("--from-layer", type=int, default=0, help="First layer index (inclusive)")
    parser.add_argument("--to-layer", type=int, default=25, help="Last layer index (exclusive)")
    parser.add_argument("--n-training-samples", type=int, default=500, help="Number of training samples")
    parser.add_argument("--methods", nargs='+', default=["pretrained"], help="List of fine-tuning methods to run for each layer. Supported: 'pretrained', 'random', 'text-only', 'image-only'")
    parser.add_argument("--caching-batch-size", type=int, default=100, help="Batch size for caching activations")
    parser.add_argument("--caching-chunk-size", type=int, default=100, help="Number of samples to cache at once")
    parser.add_argument("--training-batch-size", type=int, default=32, help="Batch size for SAE training")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of samples to use for validation")
    parser.add_argument("--val-every-n-batches", type=int, default=100, help="Run validation every N training batches")
    parser.add_argument("--resume-from-sample", type=int, default=0, help="Sample index to resume training from (must be multiple of caching_chunk_size)")
    parser.add_argument("--resume-wandb-run-id", type=str, default=None, help="Weights & Biases run ID to resume (optional)")
    
    args = parser.parse_args()

    RESULTS_DIR = Path("results")
    RUN_DIR = RESULTS_DIR / "run"
    OUTPUT_DIR = RESULTS_DIR / "output"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # If this is a fresh run, clear any previous artifacts. If resuming, keep the directory intact.
    if RUN_DIR.exists():
        if args.resume_from_sample == 0:
            shutil.rmtree(RUN_DIR)
            RUN_DIR.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[INFO] Resuming run using existing directory at {RUN_DIR}")
    else:
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize wandb *after* preparing directories so that run id is available
    wandb_kwargs = dict(project="sae-fine-tune", config=vars(args))
    if args.resume_wandb_run_id:
        wandb_kwargs.update({"id": args.resume_wandb_run_id, "resume": "allow"})
    wandb_run = wandb.init(**wandb_kwargs)
    wandb_run_id = wandb_run.id
    
    # Calculate validation set
    val_start = args.n_training_samples
    val_end = args.n_training_samples + int(args.n_training_samples * args.val_frac)
    
    print(f"[INFO] Starting SAE fine-tuning pipeline")
    print(f"[INFO] Training samples: 0 → {args.n_training_samples}")
    print(f"[INFO] Validation samples: {val_start} → {val_end}")
    print(f"[INFO] Layers: {args.from_layer} → {args.to_layer}")
    print(f"[INFO] Chunk size: {args.caching_chunk_size}")

    # Step 1: Cache validation set first
    print(f"\n[STEP 1] Preparing validation set cache ...")
    cache_val_cmd = [
        "python", "cache_activations.py",
        "--from-layer", str(args.from_layer),
        "--to-layer", str(args.to_layer),
        "--start-sample", str(val_start),
        "--end-sample", str(val_end),
        "--caching-batch-size", str(args.caching_batch_size),
        "--output-dir", str(RUN_DIR / "val_temp")
    ]
    
    # Skip caching if it already exists when resuming
    if (RUN_DIR / "val_temp").exists():
        print("[INFO] Existing validation cache found. Skipping caching step.")
    else:
        if not run_command(cache_val_cmd, "Caching validation set"):
            print("[ERROR] Failed to cache validation set. Exiting.")
            sys.exit(1)

    # Step 2: Process training data chunk by chunk
    # Initialise chunk counter based on resume offset so numbering remains consistent
    chunk_count = args.resume_from_sample // args.caching_chunk_size
    
    for chunk_start in range(args.resume_from_sample, args.n_training_samples, args.caching_chunk_size):
        chunk_end = min(chunk_start + args.caching_chunk_size, args.n_training_samples)
        chunk_count += 1
        
        print(f"\n[STEP 2.{chunk_count}] Processing chunk {chunk_start} → {chunk_end}")
        
        # Step 2a: Cache the chunk
        cache_cmd = [
            "python", "cache_activations.py",
            "--from-layer", str(args.from_layer),
            "--to-layer", str(args.to_layer),
            "--start-sample", str(chunk_start),
            "--end-sample", str(chunk_end),
            "--caching-batch-size", str(args.caching_batch_size),
            "--output-dir", str(RUN_DIR / "temp")
        ]
        
        if not run_command(cache_cmd, f"Caching chunk {chunk_start} → {chunk_end}"):
            print(f"[ERROR] Failed to cache chunk {chunk_start} → {chunk_end}. Skipping to next chunk.")
            continue

        # Step 2b: Train SAEs on the chunk
        train_cmd = [
            "python", "train_sae.py",
            "--from-layer", str(args.from_layer),
            "--to-layer", str(args.to_layer),
            "--chunk-start", str(chunk_start),
            "--chunk-end", str(chunk_end),
            "--training-batch-size", str(args.training_batch_size),
            "--data-dir", str(RUN_DIR / "temp"),
            "--run-dir", str(RUN_DIR),
            "--wandb-run-id", str(wandb_run_id),
        ]
        
        # Always pass validation parameters; evaluation frequency is handled inside train_sae.py
        train_cmd.extend([
            "--val-start", str(val_start),
            "--val-end", str(val_end),
            "--val-data-dir", str(RUN_DIR / "val_temp"),
            "--val-every-n-batches", str(args.val_every_n_batches)
        ])
        
        # Forward the methods list to train_sae.py
        train_cmd.append("--methods")
        train_cmd.extend(args.methods)
        
        # Run train_sae.py
        try:
            subprocess.run(train_cmd, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to train SAEs on chunk {chunk_start} → {chunk_end}. Continuing to next chunk.")
        
        # Step 2c: Move chunk file to persistent activations dir (keep for FVU / experiments)
        chunk_file = RUN_DIR / "temp" / f"chunk_{chunk_start}_{chunk_end}.h5"
        act_dir = RUN_DIR / "activations"
        act_dir.mkdir(parents=True, exist_ok=True)
        if chunk_file.exists():
            try:
                shutil.move(str(chunk_file), str(act_dir / chunk_file.name))
                print(f"[INFO] Moved {chunk_file} -> {act_dir}")
            except Exception as e:
                print(f"[WARN] Could not move {chunk_file}: {e}")

        print(f"[INFO] Completed chunk {chunk_count}: {chunk_start} → {chunk_end}")

    # Step 3: Move validation data to persistent activations dir
    print(f"\n[STEP 3] Preserving validation data...")
    val_chunk_file = RUN_DIR / "val_temp" / f"chunk_{val_start}_{val_end}.h5"
    act_dir = RUN_DIR / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    if val_chunk_file.exists():
        try:
            shutil.move(str(val_chunk_file), str(act_dir / val_chunk_file.name))
            print(f"[INFO] Moved validation data -> {act_dir}")
        except Exception as e:
            print(f"[WARN] Could not move validation data: {e}")

    # Copy only model weight files (exclude *_optim.pt) and organise by method
    src_ckpt_dir = RUN_DIR / "checkpoints"
    dest_ckpt_dir = OUTPUT_DIR / wandb_run_id

    try:
        # Refresh destination directory
        if dest_ckpt_dir.exists():
            shutil.rmtree(dest_ckpt_dir)
        dest_ckpt_dir.mkdir(parents=True, exist_ok=True)

        for ckpt_file in src_ckpt_dir.iterdir():
            # Skip non-pt files and optimiser checkpoints
            if ckpt_file.suffix != ".pt" or ckpt_file.name.endswith("_optim.pt"):
                continue

            # Derive method name from filename pattern "{method}_layer_{idx}.pt"
            try:
                method_name = ckpt_file.name.split("_layer_")[0]
            except Exception:
                method_name = "unknown"

            method_dir = dest_ckpt_dir / method_name
            method_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(ckpt_file, method_dir / ckpt_file.name)

        print(f"[INFO] Final weight-only checkpoints organised in {dest_ckpt_dir}")
    except Exception as e:
        print(f"[WARN] Could not copy organised checkpoints: {e}")

    print(f"\n[SUCCESS] SAE fine-tuning pipeline completed!")
    print(f"[INFO] Processed {chunk_count} chunks")
    print(f"[INFO] Checkpoints saved in {dest_ckpt_dir}")
    wandb_run.finish()

if __name__ == "__main__":
    main() 