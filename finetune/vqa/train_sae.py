# %%
import dotenv
dotenv.load_dotenv(".env")

import os
import json
from pathlib import Path
from utils import initialize_sae, get_image_token_positions
import torch
import gc
from tqdm import tqdm
from dataclasses import dataclass
from torch.optim import Adam
from typing import List
import datetime
import argparse
import h5py
import wandb

# %%
def load_chunk(start_idx, end_idx, layer_idx, data_dir="temp", full_path=None):
    """Load a chunk of activations. Returns list of tuples (tensor, img_start, img_end)."""

    chunk = []
    h5_path = Path(data_dir) / f"chunk_{start_idx}_{end_idx}.h5"
    if full_path is not None:
        h5_path = full_path
    
    # if not h5_path.exists():
    #     print(f"[WARN] Expected file {h5_path} not found. Skipping.")
    #     return chunk
    
    with h5py.File(h5_path, 'r') as f:
        layer_group = f.get(f'layer_{layer_idx}')
        if layer_group is None:
            print(f"[WARN] Layer {layer_idx} not found in {h5_path}. Skipping.")
            return chunk
        
        # Pre-allocate list for faster appending
        chunk = [None] * (end_idx - start_idx)
        valid_count = 0
        
        pbar = tqdm(range(start_idx, end_idx), desc=f"Loading chunk (layer {layer_idx})")
        for i in pbar:
            sample_key = f'sample_{i}'
            if sample_key in layer_group:
                ds = layer_group[sample_key]
                activation_data = ds[:]  # numpy array

                # Retrieve stored image span (may be missing in older caches)
                img_start = ds.attrs.get('img_start', None)
                img_end = ds.attrs.get('img_end', None)

                if img_start is not None:
                    img_start = int(img_start)
                if img_end is not None:
                    img_end = int(img_end)

                chunk[valid_count] = (torch.from_numpy(activation_data), img_start, img_end)
                valid_count += 1
                del activation_data
            else:
                print(f"[WARN] Sample {i} not found in layer {layer_idx}. Skipping.")
        pbar.close()
        
        # Return only valid entries
        return chunk[:valid_count]

# %%
def evaluate_sae(sae, activations, layer_idx, train_cfg, method="full"):
    """Evaluate an SAE on validation activations in batches. Supports 'image-only'/'text-only' masking like fine_tune_sae."""
    sae.to("cuda")

    total_recon_loss = 0.0
    total_fvu = 0.0
    total_sparsity = 0.0
    n_samples = len(activations)
    device = sae.device
    dtype = sae.dtype
    batch_size = train_cfg.model_batch_size

    with torch.no_grad():
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_entries = activations[batch_start:batch_end]
            batch_activations_raw = [entry[0] for entry in batch_entries]
            batch_max_seq_len = max(act.shape[0] for act in batch_activations_raw)

            padded_activations = []
            padded_masks = []
            seq_lens = []

            is_image_only = method.lower() == "image-only"
            is_text_only = method.lower() == "text-only"

            for (act, img_start, img_end) in batch_entries:
                seq_len = act.shape[0]
                seq_lens.append(seq_len)
                act = act.to(device).to(dtype)
                pad_len = batch_max_seq_len - seq_len
                if pad_len > 0:
                    act = torch.nn.functional.pad(act, (0, 0, 0, pad_len))
                    mask = torch.ones(batch_max_seq_len, dtype=torch.bool, device=device)
                    mask[seq_len:] = False
                else:
                    mask = torch.ones(batch_max_seq_len, dtype=torch.bool, device=device)

                if (is_image_only or is_text_only) and img_start is not None and img_end is not None:
                    img_start = int(img_start)
                    img_end = int(img_end)

                    if is_image_only:
                        specific_mask = torch.zeros(batch_max_seq_len, dtype=torch.bool, device=device)
                        specific_mask[img_start:img_end + 1] = True
                    else:
                        specific_mask = torch.ones(batch_max_seq_len, dtype=torch.bool, device=device)
                        specific_mask[img_start:img_end + 1] = False
                    mask = mask & specific_mask

                if mask.sum() == 0:
                    continue

                padded_activations.append(act)
                padded_masks.append(mask)

            if len(padded_activations) == 0:
                continue

            batch_activations = torch.stack(padded_activations)  # (batch, seq, d_in)
            batch_masks = torch.stack(padded_masks)  # (batch, seq)

            pred = sae(batch_activations)  # (batch, seq, d_in)
            error = pred - batch_activations
            masked_error = error * batch_masks.unsqueeze(-1)
            num_valid_tokens = batch_masks.sum()
            recon_loss = (masked_error ** 2).sum() / (num_valid_tokens + 1e-8)

            masked_activations = batch_activations * batch_masks.unsqueeze(-1)
            mean_activation = masked_activations.sum(dim=(0, 1)) / (num_valid_tokens + 1e-8)
            variance = ((masked_activations - mean_activation) ** 2).sum() / (num_valid_tokens + 1e-8)
            fvu = recon_loss / (variance + 1e-8)

            top_acts = sae.encode(batch_activations)  # (batch, seq, num_latents)
            masked_top_acts = top_acts * batch_masks.unsqueeze(-1)
            sparsity = (masked_top_acts != 0).float().sum() / (num_valid_tokens * top_acts.shape[-1] + 1e-8)

            total_recon_loss += recon_loss.item()
            total_fvu += fvu.item()
            total_sparsity += sparsity.item()

            del batch_activations, batch_masks, pred, error, masked_error, masked_activations, mean_activation, variance, fvu, top_acts, masked_top_acts, sparsity
            del batch_activations_raw, batch_max_seq_len, padded_activations, padded_masks, pad_len, seq_lens, mask, act
            torch.cuda.empty_cache()
            gc.collect()

    sae.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    num_batches = (n_samples + batch_size - 1) // batch_size
    return {
        f"val_recon_loss/{method}/layer_{layer_idx}": total_recon_loss / num_batches,
        f"val_fvu/{method}/layer_{layer_idx}": total_fvu / num_batches,
        f"val_sparsity/{method}/layer_{layer_idx}": total_sparsity / num_batches,
    }

# %%
@dataclass
class TrainConfig:
    """Minimal configuration for SAE fine-tuning.

    • model_batch_size:      Number of cached activation samples per optimisation batch.
    """
    model_batch_size: int = 8


def fine_tune_sae(
    sae: torch.nn.Module,
    activations: List[tuple],  # Each entry: (tensor, img_start, img_end)
    train_cfg: TrainConfig,
    optimizer: Adam,
    layer_idx: int,
    chunk_start: int,
    chunk_end: int,
    token_counter: List[int],  # Mutable single-element list holding cumulative token count
    method: str,
    val_activations=None,
    val_every_n_batches: int = 0,
    wandb_run=None,
):
    """Fine-tune an SAE on a list of cached activations.

    activations: list of tensors with shape (seq_len, d_in)
    The function processes activations in batches, padding each batch to its own max sequence length.
    """
    sae.to("cuda")

    # Ensure optimizer state tensors live on the same device / dtype as the SAE parameters
    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=sae_device, dtype=sae_dtype, non_blocking=True)

    n_samples = len(activations)
    pbar = tqdm(range(0, n_samples, train_cfg.model_batch_size), desc="Fine-tuning SAE")

    local_token_counter = 0

    for batch_idx, batch_start in enumerate(pbar):
        batch_end = min(batch_start + train_cfg.model_batch_size, n_samples)

        # Get batch activations and pad to this batch's max length
        batch_entries = activations[batch_start:batch_end]
        batch_activations_raw = [entry[0] for entry in batch_entries]
        batch_max_seq_len = max(act.shape[0] for act in batch_activations_raw)

        padded_activations = []
        padded_masks = []

        is_image_only = method.lower() == "image-only"
        is_text_only = method.lower() == "text-only"

        for (act, img_start, img_end) in batch_entries:
            original_seq_len = act.shape[0]
            act = act.to(sae.device)
            pad_len = batch_max_seq_len - original_seq_len
            if pad_len > 0:
                act = torch.nn.functional.pad(act, (0, 0, 0, pad_len))
                mask = torch.ones(batch_max_seq_len, dtype=torch.bool, device=sae.device)
                mask[original_seq_len:] = False  # Mask the padded portion
            else:
                mask = torch.ones(batch_max_seq_len, dtype=torch.bool, device=sae.device)

            # Apply method-specific masking
            if (is_image_only or is_text_only) and img_start is not None and img_end is not None:
                img_start = int(img_start)
                img_end = int(img_end)

                if is_image_only:
                    specific_mask = torch.zeros(batch_max_seq_len, dtype=torch.bool, device=sae.device)
                    specific_mask[img_start:img_end + 1] = True  # inclusive
                else:  # text-only
                    specific_mask = torch.ones(batch_max_seq_len, dtype=torch.bool, device=sae.device)
                    specific_mask[img_start:img_end + 1] = False
                mask = mask & specific_mask

            # If no valid tokens remain (edge-case), skip sample
            if mask.sum() == 0:
                continue

            padded_activations.append(act)
            padded_masks.append(mask)

        if len(padded_activations) == 0:
            # Nothing to train on in this batch (should rarely happen)
            continue

        batch_activations = torch.stack(padded_activations)  # (effective_batch, batch_max_seq_len, d_in)
        batch_masks = torch.stack(padded_masks)  # (effective_batch, batch_max_seq_len)

        batch_activations = batch_activations.to(sae.dtype)

        pred = sae(batch_activations)  # (batch_size, max_seq_len, d_in)
        error = pred - batch_activations  # (batch_size, max_seq_len, d_in)
        masked_error = error * batch_masks.unsqueeze(-1)  # (batch_size, max_seq_len, d_in)
        num_valid_tokens = batch_masks.sum()
        recon_loss = (masked_error ** 2).sum() / (num_valid_tokens + 1e-8)

        masked_activations = batch_activations * batch_masks.unsqueeze(-1)
        mean_activation = masked_activations.sum(dim=(0, 1)) / (num_valid_tokens + 1e-8)
        variance = ((masked_activations - mean_activation) ** 2).sum() / (num_valid_tokens + 1e-8)

        fvu = recon_loss / (variance + 1e-8)

        local_token_counter += int(num_valid_tokens.item())

        current_total_tokens = token_counter[method.lower()] + local_token_counter

        fvu.backward()

        with torch.no_grad():
            top_acts = sae.encode(batch_activations)  # (batch_size, max_seq_len, num_latents)
            masked_top_acts = top_acts * batch_masks.unsqueeze(-1)
            sparsity = (masked_top_acts != 0).float().sum() / (num_valid_tokens * top_acts.shape[-1] + 1e-8)

            # Log to wandb if run is provided
            if wandb_run is not None:
                wandb_run.log({
                    f"total_tokens/{method}": current_total_tokens,
                    f"train/fvu/{method}/layer_{layer_idx}": fvu.item(),
                    f"train/recon_loss/{method}/layer_{layer_idx}": recon_loss.item(),
                    f"train/sparsity/{method}/layer_{layer_idx}": sparsity.item(),
                })

        if val_activations is not None and val_every_n_batches > 0 and ((batch_idx + 1) % val_every_n_batches == 0):
            sae.eval()
            val_metrics = evaluate_sae(sae, val_activations, layer_idx, train_cfg, method=method)
            sae.to("cuda")
            sae.train()
            if wandb_run is not None:
                # Merge dictionaries and log
                wandb_run.log(val_metrics | {f"total_tokens/{method}": current_total_tokens})

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(sae, "set_decoder_norm_to_unit_norm"):
            sae.set_decoder_norm_to_unit_norm()

        pbar.set_postfix(fvu=fvu.item())

        del batch_activations, batch_masks, pred, error, masked_error, masked_activations, mean_activation, variance, fvu, top_acts, masked_top_acts, sparsity
        del batch_activations_raw, batch_max_seq_len, padded_activations, padded_masks, pad_len, original_seq_len, act, mask
        torch.cuda.empty_cache()
        gc.collect()

    sae.to("cpu")
    del pbar
    torch.cuda.empty_cache()
    gc.collect()

    return local_token_counter

# %%
def train_sae_on_chunk(
    from_layer,
    to_layer,
    chunk_start,
    chunk_end,
    methods,
    training_batch_size,
    val_start=None,
    val_end=None,
    data_dir="temp",
    val_data_dir="val_temp",
    run_dir=".",
    wandb_run_id=None,
    val_every_n_batches: int = 0,
):
    """Train SAEs on a single chunk of cached activations."""
    
    run_path = Path(run_dir)
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    token_counter_file = run_path / "total_tokens.json"

    # Initialise per-method counters (load if file exists, else zeros)
    if token_counter_file.exists():
        try:
            token_counter = json.loads(token_counter_file.read_text())
        except Exception:
            token_counter = {}
    else:
        token_counter = {}

    # Ensure every requested method has an entry
    for m in methods:
        token_counter.setdefault(m.lower(), 0)

    train_cfg = TrainConfig(model_batch_size=training_batch_size)

    wandb_run = None
    if wandb_run_id is not None:
        wandb_run = wandb.init(id=wandb_run_id, project="sae-fine-tune", resume="allow")

    for layer_idx in tqdm(range(from_layer, to_layer), desc="Processing layers"):
        # Load activations once per layer to reuse across methods
        activations = load_chunk(chunk_start, chunk_end, layer_idx, data_dir)
        val_activations = None
        if val_start is not None and val_end is not None:
            val_activations = load_chunk(val_start, val_end, layer_idx, val_data_dir)

        for method in methods:
            initialize_random = (method.lower() == "random")
            run_type = method.lower()
            checkpoint_path = checkpoint_dir / f"{run_type}_layer_{layer_idx}.pt"
            optimizer_ckpt_path = checkpoint_dir / f"{run_type}_layer_{layer_idx}_optim.pt"
            
            # Initialize SAE according to method
            if checkpoint_path.exists():
                sae = initialize_sae(layer_idx=layer_idx, checkpoint_path=checkpoint_path, device="cpu")
            else:
                # No checkpoint yet → initialise. Use random weights only for the 'random' method.
                sae = initialize_sae(layer_idx=layer_idx, initialize_random=initialize_random, device="cpu")

            if hasattr(sae, "cfg") and hasattr(sae.cfg, "num_latents"):
                lr = 2e-4 / (sae.cfg.num_latents / (2 ** 14)) ** 0.5
            else:
                lr = 2e-4

            torch.cuda.empty_cache()
            gc.collect()

            optimizer = Adam(sae.parameters(), lr=lr)

            # --- Restore previous optimizer state if available ---
            if optimizer_ckpt_path.exists():
                try:
                    optimizer_state = torch.load(optimizer_ckpt_path, map_location="cpu")
                    optimizer.load_state_dict(optimizer_state)
                    print(f"[INFO] Loaded optimizer state from {optimizer_ckpt_path}")
                except Exception as e:
                    print(f"[WARN] Could not load optimizer state from {optimizer_ckpt_path}: {e}")

            local_token_counter = fine_tune_sae(
                sae,
                activations,
                train_cfg,
                optimizer,
                layer_idx,
                chunk_start,
                chunk_end,
                token_counter,
                method,
                val_activations=val_activations,
                val_every_n_batches=val_every_n_batches,
                wandb_run=wandb_run,
            )

            # Accumulate token count for each method on the final processed layer
            if layer_idx == to_layer - 1:
                token_counter[method.lower()] += local_token_counter

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if hasattr(sae, "set_decoder_norm_to_unit_norm"):
                sae.set_decoder_norm_to_unit_norm()

            # --- Persist model & optimizer ---
            torch.save(sae.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)

            del sae, optimizer, lr
            torch.cuda.empty_cache()
            gc.collect()

        # Finished with this layer; release activations
        del activations, val_activations
        torch.cuda.empty_cache()
        gc.collect()

    # Ensure all CUDA memory is reclaimed at the end of the chunk
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        token_counter_file.write_text(json.dumps(token_counter))
    except Exception as e:
        print(f"[WARN] Could not write token counter file: {e}")

    return


def main():
    parser = argparse.ArgumentParser(description="Train SAEs on cached activations")
    parser.add_argument("--from-layer", type=int, default=24, help="First layer index (inclusive)")
    parser.add_argument("--to-layer", type=int, default=25, help="Last layer index (exclusive)")
    parser.add_argument("--chunk-start", type=int, required=True, help="Starting sample index for chunk")
    parser.add_argument("--chunk-end", type=int, required=True, help="Ending sample index for chunk")
    parser.add_argument("--methods", nargs='+', default=["pretrained"], help="List of fine-tuning methods to run for each layer. Supported: 'pretrained', 'random', 'text-only', 'image-only'")
    parser.add_argument("--training-batch-size", type=int, default=32, help="Batch size for SAE training")
    parser.add_argument("--val-start", type=int, help="Starting sample index for validation")
    parser.add_argument("--val-end", type=int, help="Ending sample index for validation")
    parser.add_argument("--data-dir", type=str, default="temp", help="Directory containing cached activations")
    parser.add_argument("--val-data-dir", type=str, default="val_temp", help="Directory containing validation activations")
    parser.add_argument("--run-dir", type=str, default=".", help="Directory for run-scoped temporary files (checkpoints, counters)")
    parser.add_argument("--wandb-run-id", type=str, help="wandb run id for logging")
    parser.add_argument("--val-every-n-batches", type=int, default=0, help="Evaluate every N training batches (0 disables periodic eval)")

    args = parser.parse_args()

    train_sae_on_chunk(
        from_layer=args.from_layer,
        to_layer=args.to_layer,
        chunk_start=args.chunk_start,
        chunk_end=args.chunk_end,
        methods=args.methods,
        training_batch_size=args.training_batch_size,
        val_start=args.val_start,
        val_end=args.val_end,
        data_dir=args.data_dir,
        val_data_dir=args.val_data_dir,
        run_dir=args.run_dir,
        wandb_run_id=args.wandb_run_id,
        val_every_n_batches=args.val_every_n_batches,
    )

if __name__ == "__main__":
    main() 