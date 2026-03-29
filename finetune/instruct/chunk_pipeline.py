#!/usr/bin/env python3
"""
Chunked pipeline with checkpoint resuming for both pretrained and random SAEs
"""

import subprocess
import shutil
import pathlib
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Configuration
TOTAL = 15000        # target samples
CHUNK = 1000           # whatever fits on disk
EXTRACT = "python finetune/extract_llava_activations.py"
FINETUNE = "python finetune/finetune_improved.py"

# Use SSD for temporary chunk data, keep results local
DATA_ROOT = pathlib.Path("/scratch/local/ssd/lachin/chunks")
DATA_ROOT.mkdir(exist_ok=True, parents=True)  # Create the chunks directory
CKPT_DIR = pathlib.Path(f"sae_ckpts_chunked_{TOTAL}")
CKPT_DIR.mkdir(exist_ok=True)

def create_holdout_dataset(chunk_dir, holdout_path, take_samples=200):
    """Create a holdout dataset from the first chunk for consistent evaluation."""
    print(f"Creating holdout dataset from {chunk_dir}...")
    
    # Collect all activation files from the chunk using glob
    activation_files = sorted(chunk_dir.glob("all_activations_sample_*.npy"))
    
    if not activation_files:
        print(f"   Warning: No activation files found in {chunk_dir}")
        return False
    
    # Take all available files (or up to take_samples)
    take = min(take_samples, len(activation_files))
    selected_files = activation_files[:take]
    
    print(f"   Found {len(activation_files)} files, using {take} for holdout")
    
    # Load and combine activations
    all_activations = []
    for file_path in selected_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            all_activations.append(data)
        except Exception as e:
            print(f"   Warning: Could not load {file_path}: {e}")
    
    if not all_activations:
        print(f"   Warning: No valid activations found")
        return False
    
    # Save as pickle
    with open(holdout_path, 'wb') as f:
        pickle.dump(all_activations, f)
    
    print(f"   Holdout dataset created: {holdout_path} ({len(all_activations)} samples)")
    return True

def create_per_chunk_plots(ckpt_dir, chunk_id):
    """Create the standard comparison plots for this specific chunk."""
    try:
        # Find all result files for this chunk
        result_files = list(ckpt_dir.glob("layer*_result.json"))
        if not result_files:
            print(f"   No result files found for chunk {chunk_id}")
            return
        
        # Collect data for this chunk
        layers = []
        fvu_finetuned = []
        fvu_vanilla = []
        fvu_random = []
        fvu_random_finetuned = []
        l0_finetuned = []
        l0_vanilla = []
        l0_random = []
        l0_random_finetuned = []
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    layers.append(result['layer_idx'])
                    fvu_finetuned.append(result['fvu_finetuned'])
                    fvu_vanilla.append(result['fvu_vanilla'])
                    fvu_random.append(result['fvu_random'])
                    fvu_random_finetuned.append(result['fvu_random_finetuned'])
                    l0_finetuned.append(result.get('l0_sparsity_finetuned', 0))
                    l0_vanilla.append(result.get('l0_sparsity_vanilla', 0))
                    l0_random.append(result.get('l0_sparsity_random', 0))
                    l0_random_finetuned.append(result.get('l0_sparsity_random_finetuned', 0))
            except Exception as e:
                print(f"   Warning: Could not read {result_file}: {e}")
        
        if not layers:
            print(f"   No valid results for chunk {chunk_id}")
            return
        
        # Sort by layer index
        sorted_data = sorted(zip(layers, fvu_finetuned, fvu_vanilla, fvu_random, fvu_random_finetuned,
                                l0_finetuned, l0_vanilla, l0_random, l0_random_finetuned))
        layers, fvu_finetuned, fvu_vanilla, fvu_random, fvu_random_finetuned, \
        l0_finetuned, l0_vanilla, l0_random, l0_random_finetuned = zip(*sorted_data)
        
        # Create FVU plot
        plt.figure(figsize=(12, 8))
        plt.plot(layers, fvu_finetuned, 'ro-', label='Finetuned', linewidth=2, markersize=6)
        plt.plot(layers, fvu_vanilla, 'bs-', label='Vanilla', linewidth=2, markersize=6)
        plt.plot(layers, fvu_random, 'g^-', label='Random', linewidth=2, markersize=6)
        plt.plot(layers, fvu_random_finetuned, 'm*-', label='Random Finetuned', linewidth=2, markersize=6)
        
        plt.xlabel('Layer Index')
        plt.ylabel('FVU')
        plt.title(f'FVU per Layer - Chunk {chunk_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        fvu_plot_path = ckpt_dir / f"fvu_per_layer_chunk_{chunk_id:03d}.png"
        plt.savefig(fvu_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create L0 sparsity plot
        plt.figure(figsize=(12, 8))
        plt.plot(layers, l0_finetuned, 'ro-', label='Finetuned', linewidth=2, markersize=6)
        plt.plot(layers, l0_vanilla, 'bs-', label='Vanilla', linewidth=2, markersize=6)
        plt.plot(layers, l0_random, 'g^-', label='Random', linewidth=2, markersize=6)
        plt.plot(layers, l0_random_finetuned, 'm*-', label='Random Finetuned', linewidth=2, markersize=6)
        
        plt.xlabel('Layer Index')
        plt.ylabel('L0 Sparsity (Average Features Active)')
        plt.title(f'L0 Sparsity per Layer - Chunk {chunk_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        l0_plot_path = ckpt_dir / f"l0_sparsity_per_layer_chunk_{chunk_id:03d}.png"
        plt.savefig(l0_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Chunk {chunk_id} plots saved:")
        print(f"     - FVU: {fvu_plot_path}")
        print(f"     - L0: {l0_plot_path}")
        
    except Exception as e:
        print(f"   Error creating plots for chunk {chunk_id}: {e}")

# Track checkpoints for both types - now layer-specific
pretrained_ckpts = {}  # layer_idx -> checkpoint_path
random_ckpts = {}      # layer_idx -> checkpoint_path
holdout_created = False

for start in range(0, TOTAL, CHUNK):
    chunk_id = start // CHUNK
    chunk_dir = DATA_ROOT / f"chunk_{chunk_id:03d}"
    chunk_dir.mkdir(exist_ok=True)

    print(f"\n--- Processing chunk {chunk_id} (samples {start}-{start + CHUNK - 1}) ---")

    # ---- 1. extract ----
    print("1. Extracting activations...")
    subprocess.run([
        *EXTRACT.split(),
        "--max_samples", str(CHUNK),
        "--start_index", str(start),
        "--save_dir", str(chunk_dir),
        "--no_combine"
    ], check=True)

    # ---- 1.5. create holdout dataset from first chunk ----
    if chunk_id == 0 and not holdout_created:
        print("1.5. Creating holdout dataset for consistent evaluation...")
        holdout_path = CKPT_DIR / "heldout.pkl"
        created = create_holdout_dataset(chunk_dir, holdout_path, take_samples=200)
        if created:
            # Check if we actually have enough samples
            try:
                with open(holdout_path, 'rb') as f:
                    samples = pickle.load(f)
                if len(samples) >= 200:
                    holdout_created = True
                    print(f"   Holdout dataset complete with {len(samples)} samples")
                else:
                    print(f"   Holdout dataset created with {len(samples)} samples, will expand")
            except Exception as e:
                print(f"   Warning: Could not verify holdout size: {e}")
        else:
            print("   Warning: Could not create holdout dataset, will use chunk test sets")
    
    # ---- 1.6. expand holdout dataset if needed (from multiple chunks) ----
    elif not holdout_created:  # Keep adding until we have enough
        print(f"1.6. Expanding holdout dataset from chunk {chunk_id}...")
        holdout_path = CKPT_DIR / "heldout.pkl"
        
        # Load existing holdout if it exists
        existing_activations = []
        if holdout_path.exists():
            try:
                with open(holdout_path, 'rb') as f:
                    existing_activations = pickle.load(f)
                print(f"   Loaded {len(existing_activations)} existing samples")
            except Exception as e:
                print(f"   Warning: Could not load existing holdout: {e}")
        
        # Add samples from current chunk
        activation_files = sorted(chunk_dir.glob("all_activations_sample_*.npy"))
        take = min(50, len(activation_files))  # Take up to 50 from this chunk
        selected_files = activation_files[:take]
        
        new_activations = []
        for file_path in selected_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                new_activations.append(data)
            except Exception as e:
                print(f"   Warning: Could not load {file_path}: {e}")
        
        # Combine and save
        all_activations = existing_activations + new_activations
        if all_activations:
            with open(holdout_path, 'wb') as f:
                pickle.dump(all_activations, f)
            print(f"   Holdout dataset expanded: {holdout_path} ({len(all_activations)} total samples)")
            
            # Stop collecting after we have enough samples
            if len(all_activations) >= 150:
                holdout_created = True
                print("   Holdout dataset complete")
        else:
            print("   Warning: Could not add samples to holdout dataset")

    # ---- 2. fine-tune on this chunk ----
    print("2. Fine-tuning SAEs...")
    
    # Use lower learning rate for resumed training to prevent catastrophic forgetting
    lr_args = []
    if chunk_id > 0:  # If not first chunk, use lower learning rate
        lr_args = ["--learning_rate", "0.00001"]  # 10x lower
        print("   Using reduced learning rate to prevent catastrophic forgetting")
    
    # Add holdout dataset if available
    holdout_path = CKPT_DIR / "heldout.pkl"
    if holdout_path.exists():
        holdout_args = ["--holdout_path", str(holdout_path)]
        print("   Using (possibly partial) holdout dataset for evaluation")
    else:
        holdout_args = []
    
    subprocess.run([
        *FINETUNE.split(),
        "--data_path", str(chunk_dir),
        "--output_dir", str(CKPT_DIR),
        "--run_name", f"chunk_{chunk_id:03d}",
        "--num_epochs", "1",            # one epoch per chunk
        *lr_args,
        *holdout_args
    ], check=True)

    # ---- 3. verify checkpoints were created ----
    print("3. Verifying checkpoints were created...")
    
    # Count checkpoints created
    pretrained_count = len(list(CKPT_DIR.glob("layer*_text_finetuned_*.pt")))
    random_count = len(list(CKPT_DIR.glob("layer*_random_finetuned_*.pt")))
    
    print(f"   Pretrained checkpoints: {pretrained_count}")
    print(f"   Random checkpoints: {random_count}")
    
    if pretrained_count == 0 and random_count == 0:
        print("   Warning: No checkpoints found - this may indicate an error")

    # ---- 4. create per-chunk plots ----
    print("4. Creating per-chunk plots...")
    create_per_chunk_plots(CKPT_DIR, chunk_id)

    # ---- 5. clean up this chunk's activations ----
    print("5. Cleaning up chunk files...")
    shutil.rmtree(chunk_dir)
    print(f"✅ Chunk {chunk_id} completed and cleaned up")

print(f"\n🎉 Pipeline completed!")
print(f"Final checkpoints in: {CKPT_DIR}")
print(f"📊 Summary:")
print(f"   - Processed {TOTAL} samples in {chunk_id + 1} chunks")
print(f"   - Each chunk had {CHUNK} samples")
print(f"   - Both pretrained and random SAEs trained incrementally")
print(f"   - Per-chunk plots saved for each chunk")
print(f"   - Reduced learning rate used for resumed training to prevent catastrophic forgetting")
if holdout_created:
    print(f"   - Holdout dataset created for consistent evaluation across chunks")
