#!/usr/bin/env python3
"""
Improved SAE Fine-tuning Script

Enhanced version with better configuration, error handling, and resume functionality.
Keeps all original functionality from finetune.py but adds improvements.

Usage:
    python finetune_improved.py --config config.json
    python finetune_improved.py --data_path local_data/dataset_all_activations_llava_128.pkl --num_epochs 15
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import multiprocessing as mp
from multiprocessing import Pool
import traceback
import signal

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
import wandb
import pickle
import bisect
import copy
from sae_lens import SAE
from sae_lens.sae import SAEConfig
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetune_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for SAE training."""
    # Data settings
    data_path: str = "local_data/dataset_all_activations_llava_128.pkl"
    output_dir: str = "sae-ckpts_improved"
    run_name: str = "finetune_improved"
    
    # Training settings
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-4
    learning_rate_finetuned: float = 1e-4  # Different LR for finetuned vs random
    l1_coefficient: float = 3e-4
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    
    # Model settings
    num_layers: int = 32
    topk_k: int = 50
    
    # Hardware settings
    num_gpus: Optional[int] = None  # Auto-detect if None
    layers_to_train: Optional[List[int]] = None  # Train specific layers only
    max_samples: Optional[int] = None  # Limit number of samples (for testing)
    
    # Optimization settings
    optimize_every_n_tokens: int = 8192
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "sae-finetune"
    log_level: str = "INFO"
    
    # Resume settings
    resume_from_checkpoint: Optional[str] = None
    resume_random_checkpoint: Optional[str] = None  # Separate checkpoint for random SAE
    skip_completed_layers: bool = True
    holdout_path: Optional[str] = None  # Path to holdout dataset for consistent evaluation

    @classmethod
    def from_json(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def save_to_json(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def validate(self):
        """Validate configuration settings."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if self.train_split_ratio + self.val_split_ratio >= 1.0:
            raise ValueError("train_split_ratio + val_split_ratio must be < 1.0")
        
        if self.num_gpus and self.num_gpus > torch.cuda.device_count():
            logger.warning(f"Requested {self.num_gpus} GPUs but only {torch.cuda.device_count()} available")
            self.num_gpus = torch.cuda.device_count()

class LazyActivationDataset(Dataset):
    """Memory-efficient dataset for loading activations."""
    
    def __init__(self, data_path: str, layer_idx: int, dtype=torch.bfloat16, device='cpu'):
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                all_samples = pickle.load(f)
        else:
            all_samples = np.load(data_path, allow_pickle=True)
        
        self.layerwise = []
        self.lengths = []
        
        for sample in all_samples:
            arr = sample[layer_idx]
            if arr.ndim == 3:  # (1, T, D)
                arr = arr.squeeze(0)
            elif arr.ndim == 2:  # (T, D)
                pass
            else:
                raise ValueError(f"Unexpected activation shape: {arr.shape}")
            
            self.layerwise.append(arr)       
            self.lengths.append(arr.shape[0])
        
        # Build prefix sums for efficient indexing
        self.prefix_sums = [0]
        for length in self.lengths:
            self.prefix_sums.append(self.prefix_sums[-1] + length)
        
        self.total_tokens = self.prefix_sums[-1]
        self.dtype = dtype
        self.device = device
        
        logger.debug(f"Dataset loaded: {len(self.layerwise)} samples, {self.total_tokens} total tokens")

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx):
        sample_idx = bisect.bisect_right(self.prefix_sums, idx) - 1
        offset = idx - self.prefix_sums[sample_idx]
        vec = self.layerwise[sample_idx][offset]  
        # Keep on CPU - we'll move it to the right GPU in the training loop
        return torch.tensor(vec, dtype=self.dtype)

class IndividualFileDataset(Dataset):
    """Memory-efficient dataset for loading from individual .npy files."""
    
    def __init__(self, data_dir: str, layer_idx: int, max_samples: int = 15000, 
                 dtype=torch.bfloat16, device='cpu'):
        self.data_dir = data_dir
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.device = device
        
        # Find all existing files
        self.existing_files = []
        self.sample_lengths = []
        self.prefix_sums = [0]
        
        total_tokens = 0
        logger.info(f"Scanning {max_samples} files to build index...")
        
        for i in tqdm(range(max_samples), desc="Building file index"):
            file_path = os.path.join(data_dir, f"all_activations_sample_{i}.npy")
            if os.path.exists(file_path):
                self.existing_files.append(i)
                
                # Load ONLY to get token count, then immediately free memory
                try:
                    data = np.load(file_path, allow_pickle=True)
                    arr = data[layer_idx]
                    if arr.ndim == 3:
                        arr = arr.squeeze(0)
                    
                    length = arr.shape[0]
                    self.sample_lengths.append(length)
                    total_tokens += length
                    self.prefix_sums.append(total_tokens)
                    
                    # Explicitly free memory
                    del data, arr
                    
                except Exception as e:
                    logger.warning(f"Could not process file {i}: {e}")
                    continue
        
        self.total_tokens = total_tokens
        logger.info(f"Found {len(self.existing_files)} files with {total_tokens} total tokens")

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx):
        # Find which sample contains this token
        sample_idx = bisect.bisect_right(self.prefix_sums, idx) - 1
        token_offset = idx - self.prefix_sums[sample_idx]
        
        # Load the sample file
        file_idx = self.existing_files[sample_idx]
        file_path = os.path.join(self.data_dir, f"all_activations_sample_{file_idx}.npy")
        
        data = np.load(file_path, allow_pickle=True)
        arr = data[self.layer_idx]
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        
        vec = arr[token_offset]
        # Keep on CPU - we'll move it to the right GPU in the training loop
        return torch.tensor(vec, dtype=self.dtype)

def finetune_sae_on_activations(
    sae,
    dataloader,
    num_epochs: int = 10,
    device: str = "cuda",
    use_wandb: bool = True,
    wandb_project: str = "sae-finetune",
    wandb_name: str = "layer5_text",
    val_loader = None,
    train_loss_list=None,
    val_loss_list=None,
    lr: float = 2e-4,
    l1_coefficient: float = 3e-4,
    optimize_every_n_tokens: int = 8192,
    gradient_clip_norm: float = 0.0,
):
    """Improved version of the original finetune function with better monitoring."""
    sae.to(device).train()
    num_latents = sae.cfg.d_sae if hasattr(sae.cfg, 'd_sae') else sae.cfg.num_latents
    lr = lr / (num_latents / (2**14)) ** 0.5
    optimizer = Adam(sae.parameters(), lr=lr)
    tokens_since_last_step = 0
    
    if train_loss_list is None:
        train_loss_list = []
    if val_loss_list is None:
        val_loss_list = []
    
    if use_wandb:
        try:
            wandb.init(
                name=wandb_name,
                project=wandb_project,
                config={"sae_config": str(sae.cfg) if hasattr(sae, "cfg") else "custom"},
                save_code=True,
                reinit=True
            )
        except Exception as e:
            logger.warning(f"Wandb initialization failed: {e}")
            use_wandb = False
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_fvu = 0.0
        epoch_ev = 0.0
        epoch_mse = 0.0
        epoch_l0 = 0.0
        n_batches = 0
        total_train_tokens = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch = batch.to(device).bfloat16()
            pred = sae(batch)
            error = pred - batch
            loss = (error ** 2).sum() / ((batch - batch.mean(dim=1, keepdim=True)) ** 2).sum() # FVU
            loss.backward()
            tokens_since_last_step += batch.size(0)
            
            if tokens_since_last_step >= optimize_every_n_tokens:
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), gradient_clip_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                tokens_since_last_step = 0
                if hasattr(sae, 'set_decoder_norm_to_unit_norm'):
                    sae.set_decoder_norm_to_unit_norm()

            # For logging (keep FVU/EV for monitoring)
            mse_val = loss.item()
            denom = ((batch - batch.mean(dim=1, keepdim=True)) ** 2).sum().item()
            fvu = mse_val / denom if denom != 0 else float('nan')
            ev = 1.0 - fvu if denom != 0 else float('nan')
            mse_per_token = mse_val / batch.size(0)
            l0 = float("nan")
            
            if hasattr(sae, "encode"):
                encoded = sae.encode(batch)
                if isinstance(encoded, torch.Tensor):
                    l0 = (encoded > 0).float().sum(dim=1).mean().item()
                elif isinstance(encoded, tuple) and len(encoded) == 2:
                    top_acts = encoded[0]
                    l0 = (top_acts > 0).float().sum(dim=1).mean().item()
            
            epoch_loss += mse_val * batch.size(0)
            total_train_tokens += batch.size(0)
            epoch_fvu += fvu
            epoch_ev += ev
            epoch_mse += mse_per_token
            if not np.isnan(l0):
                epoch_l0 += l0
            n_batches += 1
        
        avg_train_loss = epoch_loss / total_train_tokens if total_train_tokens > 0 else float('nan')
        avg_fvu = epoch_fvu / n_batches
        avg_ev = epoch_ev / n_batches
        avg_mse = epoch_mse / n_batches
        avg_l0 = epoch_l0 / n_batches if n_batches > 0 else float('nan')
        train_loss_list.append(avg_train_loss)
        
        # Validation loss
        if val_loader is not None:
            sae.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device).bfloat16()
                    pred = sae(val_batch)
                    error = pred - val_batch
                    mse = (error ** 2).sum() / ((val_batch - val_batch.mean(dim=1, keepdim=True)) ** 2).sum()
                    val_loss += mse * val_batch.size(0)
                    n_val += val_batch.size(0)
            avg_val_loss = val_loss / n_val if n_val > 0 else 0.0
            val_loss_list.append(avg_val_loss)
            sae.train()
        else:
            avg_val_loss = None
            val_loss_list.append(None)
        
        # Log all metrics to wandb per epoch
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "fvu": avg_fvu,
                "explained_variance": avg_ev,
                "reconstruction_mse": avg_mse,
                "l0_sparsity": avg_l0
            }, step=epoch)
        
        # Final optimizer step if any gradients remain
        if tokens_since_last_step > 0:
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            if hasattr(sae, 'set_decoder_norm_to_unit_norm'):
                sae.set_decoder_norm_to_unit_norm()
    
    if use_wandb:
        try:
            wandb.finish()
        except:
            pass
    
    return sae, train_loss_list, val_loss_list

def evaluate_sae(sae, dataloader, device="cuda"):
    """Evaluate SAE performance."""
    sae.eval()
    total_fvu = 0.0
    total_tokens = 0
    total_l0 = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device).bfloat16()
            recon = sae(batch)

            mse = ((recon - batch) ** 2).sum()
            denom = ((batch - batch.mean(dim=1, keepdim=True)) ** 2).sum()
            fvu = mse / denom

            # Calculate L0 sparsity
            if hasattr(sae, "encode"):
                encoded = sae.encode(batch)
                if isinstance(encoded, torch.Tensor):
                    batch_l0 = (encoded > 0).float().sum(dim=1).mean().item()
                    total_l0 += batch_l0 * batch.size(0)
                elif isinstance(encoded, tuple) and len(encoded) == 2:
                    top_acts = encoded[0]
                    batch_l0 = (top_acts > 0).float().sum(dim=1).mean().item()
                    total_l0 += batch_l0 * batch.size(0)
                else:
                    batch_l0 = float("nan")
                    total_l0 = float("nan")
            else:
                batch_l0 = float("nan")
                total_l0 = float("nan")

            total_fvu += fvu.item() * batch.size(0)
            total_tokens += batch.size(0)

    avg_l0 = total_l0 / total_tokens if not isinstance(total_l0, float) or not np.isnan(total_l0) else float("nan")
    return total_fvu / total_tokens, avg_l0

def train_one_layer(layer_idx, config_dict):
    """Train all SAE variants for one layer with improved error handling."""
    try:
        # Reconstruct config
        config = TrainingConfig(**config_dict)
        gpu_id = layer_idx % config.num_gpus
        device = f"cuda:{gpu_id}"
        
        # Clear GPU memory at the start of each process
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Setup logging for this process
        layer_logger = logging.getLogger(f"layer_{layer_idx}")
        handler = logging.FileHandler(f"{config.output_dir}/layer_{layer_idx}.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        layer_logger.addHandler(handler)
        layer_logger.setLevel(getattr(logging, config.log_level))
        
        layer_logger.info(f"Starting training for layer {layer_idx} on GPU {gpu_id}")
        
        # Check if already completed
        result_file = f"{config.output_dir}/layer{layer_idx}_result.json"
        if config.skip_completed_layers and Path(result_file).exists():
            layer_logger.info(f"Layer {layer_idx} already completed, skipping")
            return {"layer_idx": layer_idx, "status": "skipped"}
        
        # --- Load the SAE weights -----------------------------------------
        sae, cfg, sparsity = SAE.from_pretrained(
            release="llama_scope_lxr_8x",
            sae_id=f"l{layer_idx}r_8x",
            device=device,
        )
        
        # Create TopK config
        topk_cfg = dict(cfg)
        for key in ['architecture', 'jump_relu_threshold', 'neuronpedia_id', 'activation_fn_str']:
            topk_cfg.pop(key, None)

        new_topk_cfg = SAEConfig(
            architecture="topk",
            activation_fn_kwargs={"k": config.topk_k},
            activation_fn_str="topk",
            **topk_cfg
        )
        
        # Create new SAE and random SAE
        new_sae = SAE(new_topk_cfg)
        sae_rand = copy.deepcopy(new_sae)
        
        # Load original weights for baseline evaluation
        og_weights = sae.state_dict().copy()
        og_weights.pop("threshold", None)
        new_sae.load_state_dict(og_weights)
        
        # Setup datasets - detect if using individual files or pickle file
        if os.path.isdir(config.data_path):
            # Directory with individual .npy files
            max_samples = config.max_samples if config.max_samples else 15000
            dataset = IndividualFileDataset(config.data_path,
                                    layer_idx=layer_idx,
                                    max_samples=max_samples,
                                    dtype=torch.bfloat16,
                                    device="cpu")
        else:
            # Single pickle file
            dataset = LazyActivationDataset(config.data_path,
                                layer_idx=layer_idx,
                                dtype=torch.bfloat16,
                                device="cpu")
        
        num_total = len(dataset)
        num_train = int(config.train_split_ratio * num_total)
        num_test = num_total - num_train
        num_val = max(1, int(config.val_split_ratio * num_total))
        num_train_adj = num_train - num_val
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train_adj, num_val, num_test])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # --- 1. Vanilla SAE baseline ---------------------------------
        if config.resume_from_checkpoint:
            # We already evaluated and logged vanilla/random during chunk 0
            # Load the stored values so the JSON schema stays unchanged
            with open(result_file) as f:
                prev = json.load(f)
            fvu_vanilla = prev["fvu_vanilla"]
            l0_vanilla = prev["l0_sparsity_vanilla"]
            layer_logger.info("Baseline metrics loaded from previous run")
        else:
            # Use holdout dataset for evaluation if available, otherwise use test set
            if config.holdout_path and os.path.exists(config.holdout_path):
                layer_logger.info(f"Evaluating vanilla SAE on holdout dataset: {config.holdout_path}")
                holdout_dataset = LazyActivationDataset(config.holdout_path, layer_idx, dtype=torch.bfloat16, device="cpu")
                holdout_loader = DataLoader(holdout_dataset, batch_size=config.batch_size, shuffle=False)
                fvu_vanilla, l0_vanilla = evaluate_sae(new_sae, holdout_loader, device=device)
            else:
                fvu_vanilla, l0_vanilla = evaluate_sae(new_sae, test_loader, device=device)
            layer_logger.info(
                f"Layer {layer_idx:02d} | Vanilla FVU: {fvu_vanilla:.4f}, L0: {l0_vanilla:.2f}"
            )
        
        # --- 2. Random SAE baseline ---
        if config.resume_from_checkpoint:
            # Load random baseline from previous run
            with open(result_file) as f:
                prev = json.load(f)
            fvu_rand = prev["fvu_random"]
            l0_rand = prev["l0_sparsity_random"]
            layer_logger.info("Random baseline metrics loaded from previous run")
        else:
            if hasattr(sae_rand, 'initialize_weights_basic'):
                sae_rand.initialize_weights_basic()
            # Use holdout dataset for evaluation if available, otherwise use test set
            if config.holdout_path and os.path.exists(config.holdout_path):
                layer_logger.info(f"Evaluating random SAE on holdout dataset: {config.holdout_path}")
                holdout_dataset = LazyActivationDataset(config.holdout_path, layer_idx, dtype=torch.bfloat16, device="cpu")
                holdout_loader = DataLoader(holdout_dataset, batch_size=config.batch_size, shuffle=False)
                fvu_rand, l0_rand = evaluate_sae(sae_rand, holdout_loader, device=device)
            else:
                fvu_rand, l0_rand = evaluate_sae(sae_rand, test_loader, device=device)
            layer_logger.info(f"Layer {layer_idx:02d} | Random FVU: {fvu_rand:.4f}, L0: {l0_rand:.2f}")
        
        # NOW load checkpoints for training (after baseline evaluation)
        # Load weights: (1) resume ckpt if provided, else (2) original SAE
        if config.resume_from_checkpoint:
            # Check if checkpoint matches this layer
            if f"layer{layer_idx}_" in Path(config.resume_from_checkpoint).name:
                layer_logger.info(f"Resuming from {config.resume_from_checkpoint}")
                new_sae.load_state_dict(
                    torch.load(config.resume_from_checkpoint, map_location="cpu")
                )
            else:
                layer_logger.warning(f"Checkpoint layer mismatch – ignored: {config.resume_from_checkpoint}")
        else:
            # auto-resume from prior layer-specific checkpoint if present
            for gpu in range(config.num_gpus or 8):
                ckpt_path = Path(config.output_dir) / f"layer{layer_idx}_text_finetuned_gpu{gpu}.pt"
                if ckpt_path.exists():
                    layer_logger.info(f"Resuming pretrained SAE from {ckpt_path}")
                    new_sae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    break
        
        # Load random SAE from checkpoint if provided
        if config.resume_random_checkpoint:
            # Check if checkpoint matches this layer
            if f"layer{layer_idx}_" in Path(config.resume_random_checkpoint).name:
                layer_logger.info(f"Resuming random SAE from {config.resume_random_checkpoint}")
                sae_rand.load_state_dict(
                    torch.load(config.resume_random_checkpoint, map_location="cpu")
                )
            else:
                layer_logger.warning(f"Random checkpoint layer mismatch – ignored: {config.resume_random_checkpoint}")
        else:
            # Try to find layer-specific random checkpoint
            layer_specific_random_ckpt = None
            for gpu_id in range(8):  # Check all possible GPUs
                ckpt_path = f"{config.output_dir}/layer{layer_idx}_random_finetuned_gpu{gpu_id}.pt"
                if os.path.exists(ckpt_path):
                    layer_specific_random_ckpt = ckpt_path
                    break
            
            if layer_specific_random_ckpt:
                layer_logger.info(f"Found layer-specific random checkpoint: {layer_specific_random_ckpt}")
                sae_rand.load_state_dict(
                    torch.load(layer_specific_random_ckpt, map_location="cpu")
                )
        
        wandb_name_rand = f"layer{layer_idx}_random_gpu{gpu_id}_{config.run_name}"
        finetuned_rand_sae, train_loss_list_rand, val_loss_list_rand = finetune_sae_on_activations(
            sae_rand,
            train_loader,
            num_epochs=config.num_epochs,
            device=device,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            wandb_name=wandb_name_rand,
            val_loader=val_loader,
            lr=config.learning_rate,
            l1_coefficient=config.l1_coefficient,
            optimize_every_n_tokens=config.optimize_every_n_tokens,
            gradient_clip_norm=config.gradient_clip_norm
        )
        # Use holdout dataset for evaluation if available, otherwise use test set
        if config.holdout_path and os.path.exists(config.holdout_path):
            layer_logger.info(f"Evaluating finetuned random SAE on holdout dataset: {config.holdout_path}")
            holdout_dataset = LazyActivationDataset(config.holdout_path, layer_idx, dtype=torch.bfloat16, device="cpu")
            holdout_loader = DataLoader(holdout_dataset, batch_size=config.batch_size, shuffle=False)
            fvu_rand_finetuned, l0_rand_finetuned = evaluate_sae(finetuned_rand_sae, holdout_loader, device=device)
        else:
            fvu_rand_finetuned, l0_rand_finetuned = evaluate_sae(finetuned_rand_sae, test_loader, device=device)
        layer_logger.info(f"Layer {layer_idx:02d} | Random SAE (trained) FVU: {fvu_rand_finetuned:.4f}, L0: {l0_rand_finetuned:.2f}")
        
        # Save random finetuned SAE
        sae_rand_save_path = f"{config.output_dir}/layer{layer_idx}_random_finetuned_gpu{gpu_id}.pt"
        torch.save(finetuned_rand_sae.state_dict(), sae_rand_save_path)
        
        # --- 4. Train Pretrained SAE ---
        wandb_name = f"layer{layer_idx}_text_gpu{gpu_id}_{config.run_name}"
        finetuned_sae, train_loss_list, val_loss_list = finetune_sae_on_activations(
            new_sae,
            train_loader,
            num_epochs=config.num_epochs,
            device=device,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            wandb_name=wandb_name,
            val_loader=val_loader,
            lr=config.learning_rate_finetuned,
            l1_coefficient=config.l1_coefficient,
            optimize_every_n_tokens=config.optimize_every_n_tokens,
            gradient_clip_norm=config.gradient_clip_norm
        )
        # Use holdout dataset for evaluation if available, otherwise use test set
        if config.holdout_path and os.path.exists(config.holdout_path):
            layer_logger.info(f"Evaluating finetuned SAE on holdout dataset: {config.holdout_path}")
            holdout_dataset = LazyActivationDataset(config.holdout_path, layer_idx, dtype=torch.bfloat16, device="cpu")
            holdout_loader = DataLoader(holdout_dataset, batch_size=config.batch_size, shuffle=False)
            fvu_after, l0_after = evaluate_sae(finetuned_sae, holdout_loader, device=device)
        else:
            fvu_after, l0_after = evaluate_sae(finetuned_sae, test_loader, device=device)
        
        # Save finetuned SAE
        sae_save_path = f"{config.output_dir}/layer{layer_idx}_text_finetuned_gpu{gpu_id}.pt"
        torch.save(finetuned_sae.state_dict(), sae_save_path)
        
        # Log final results
        layer_logger.info(f"Layer {layer_idx:02d} | Final Results:")
        layer_logger.info(f"  Vanilla      FVU: {fvu_vanilla:.4f}, L0: {l0_vanilla:.2f}")
        layer_logger.info(f"  Finetuned    FVU: {fvu_after:.4f}, L0: {l0_after:.2f}")
        layer_logger.info(f"  Random       FVU: {fvu_rand:.4f}, L0: {l0_rand:.2f}")
        layer_logger.info(f"  Random Tune  FVU: {fvu_rand_finetuned:.4f}, L0: {l0_rand_finetuned:.2f}")
        
        # Save results JSON (matching original format)
        with open(result_file, "w") as f:
            json.dump({
                'run_name': config.run_name,
                'layer_idx': layer_idx,
                'gpu_id': gpu_id,
                'fvu_vanilla': fvu_vanilla,
                'l0_sparsity_vanilla': l0_vanilla,
                'fvu_finetuned': fvu_after,
                'l0_sparsity_finetuned': l0_after,
                'fvu_random': fvu_rand,
                'l0_sparsity_random': l0_rand,
                'fvu_random_finetuned': fvu_rand_finetuned,
                'l0_sparsity_random_finetuned': l0_rand_finetuned
            }, f, indent=2)
        
        # Clean up memory
        del sae, finetuned_sae, sae_rand, finetuned_rand_sae, new_sae
        torch.cuda.empty_cache()
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        layer_logger.info(f"Layer {layer_idx} training completed successfully")
        
        return {
            "layer_idx": layer_idx,
            "status": "completed",
            "fvu_vanilla": fvu_vanilla,
            "fvu_finetuned": fvu_after,
            "fvu_random": fvu_rand,
            "fvu_random_finetuned": fvu_rand_finetuned
        }
        
    except Exception as e:
        error_msg = f"Layer {layer_idx} failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"layer_idx": layer_idx, "status": "failed", "error": error_msg}

def train_one_layer_wrapper(args):
    """Wrapper for multiprocessing."""
    return train_one_layer(*args)

def create_comparison_plots(config: TrainingConfig):
    """Create the original comparison plots (fvu_per_layer_all_types.png, etc.)."""
    try:
        metrics = [
            ("fvu", "FVU"),
            ("l0_sparsity", "L0 Sparsity (Average Features Active)")
        ]
        
        for metric_key, metric_label in metrics:
            finetuned = []
            vanilla = []
            random = []
            random_finetuned = []
            layers = []
            
            for layer_idx in range(config.num_layers):
                result_file = f"{config.output_dir}/layer{layer_idx}_result.json"
                if Path(result_file).exists():
                    with open(result_file, "r") as f:
                        result = json.load(f)
                        layers.append(layer_idx)
                        finetuned.append(result[f"{metric_key}_finetuned"])
                        vanilla.append(result[f"{metric_key}_vanilla"])
                        random.append(result[f"{metric_key}_random"])
                        random_finetuned.append(result.get(f"{metric_key}_random_finetuned", None))
            
            if not layers:
                logger.warning(f"No results found for {metric_key} plot")
                continue
            
            plt.figure(figsize=(10, 6))
            plt.plot(layers, finetuned, label="Finetuned", marker="o")
            plt.plot(layers, vanilla, label="Vanilla", marker="o")
            plt.plot(layers, random, label="Random", marker="o")
            if any(x is not None for x in random_finetuned):
                plt.plot(layers, random_finetuned, label="Random Finetuned", marker="o")
            
            plt.xlabel("Layer Index")
            plt.ylabel(metric_label)
            plt.title(f"{metric_label} per Layer for Finetuned, Vanilla, Random, and Random Finetuned SAEs")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.yscale('log')
            
            plot_path = f"{config.output_dir}/{metric_key}_per_layer_all_types.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved plot: {plot_path}")
            
    except Exception as e:
        logger.error(f"Failed to create comparison plots: {e}")

def create_training_summary_plots(results: List[Dict], config: TrainingConfig):
    """Create additional summary plots."""
    try:
        successful = [r for r in results if r['status'] == 'completed']
        if not successful:
            return
        
        layers = [r['layer_idx'] for r in successful]
        fvu_vanilla = [r['fvu_vanilla'] for r in successful]
        fvu_finetuned = [r['fvu_finetuned'] for r in successful]
        fvu_random = [r['fvu_random'] for r in successful]
        fvu_random_finetuned = [r['fvu_random_finetuned'] for r in successful]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # FVU comparison
        axes[0, 0].plot(layers, fvu_vanilla, 'b-o', label='Vanilla')
        axes[0, 0].plot(layers, fvu_finetuned, 'r-o', label='Finetuned')
        axes[0, 0].plot(layers, fvu_random, 'g-o', label='Random')
        axes[0, 0].plot(layers, fvu_random_finetuned, 'm-o', label='Random Finetuned')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('FVU')
        axes[0, 0].set_title('FVU by Layer')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Improvement over vanilla
        improvement = [(v - f) / v for v, f in zip(fvu_vanilla, fvu_finetuned)]
        axes[0, 1].plot(layers, improvement, 'r-o')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Relative Improvement')
        axes[0, 1].set_title('FVU Improvement over Vanilla')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of improvements
        axes[1, 0].hist(improvement, bins=20, alpha=0.7)
        axes[1, 0].set_xlabel('Relative Improvement')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of FVU Improvements')
        
        # Summary statistics
        axes[1, 1].bar(['Vanilla', 'Finetuned', 'Random', 'Random Fine'], 
                      [np.mean(fvu_vanilla), np.mean(fvu_finetuned), 
                       np.mean(fvu_random), np.mean(fvu_random_finetuned)])
        axes[1, 1].set_ylabel('Mean FVU')
        axes[1, 1].set_title('Average Performance')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(Path(config.output_dir) / "training_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary plots saved to {config.output_dir}/training_summary.png")
        
    except Exception as e:
        logger.error(f"Failed to create summary plots: {e}")

def main():
    parser = argparse.ArgumentParser(description="Improved SAE Fine-tuning")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--data_path", type=str, help="Path to activation data")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to use")
    parser.add_argument("--layers", type=str, help="Comma-separated layer indices to train")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--skip_completed", action="store_true", help="Skip already completed layers")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_random_checkpoint", type=str, help="Path to random SAE checkpoint to resume from")
    parser.add_argument("--holdout_path", type=str, help="Path to holdout dataset for consistent evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = TrainingConfig.from_json(args.config)
    else:
        config = TrainingConfig()
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            if key == 'layers':
                config.layers_to_train = [int(x.strip()) for x in value.split(',')]
            elif key == 'no_wandb':
                config.use_wandb = not value
            elif key == 'skip_completed':
                config.skip_completed_layers = value
            elif hasattr(config, key):
                setattr(config, key, value)
    
    # Validate and setup
    config.validate()
    
    if config.num_gpus is None:
        config.num_gpus = torch.cuda.device_count()
    
    if config.layers_to_train is None:
        config.layers_to_train = list(range(config.num_layers))
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config.save_to_json(Path(config.output_dir) / "config.json")
    
    logger.info(f"Starting training with {config.num_gpus} GPUs")
    logger.info(f"Training layers: {config.layers_to_train}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Set CUDA memory management to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Prepare arguments for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    args_list = [
        (layer_idx, asdict(config))
        for layer_idx in config.layers_to_train
    ]
    
    # Train with multiprocessing - use exactly num_gpus processes to avoid conflicts
    try:
        with mp.Pool(processes=config.num_gpus) as pool:
            results = pool.map(train_one_layer_wrapper, args_list)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return
    
    # Process results
    successful = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    logger.info(f"Training completed: {len(successful)} successful, {len(failed)} failed, {len(skipped)} skipped")
    
    # Save results summary
    summary = {
        "config": asdict(config),
        "results": results,
        "summary": {
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped)
        }
    }
    
    with open(Path(config.output_dir) / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create all plots
    logger.info("Creating comparison plots...")
    create_comparison_plots(config)
    
    if successful:
        logger.info("Creating training summary plots...")
        create_training_summary_plots(results, config)
    
    if failed:
        logger.error(f"Failed layers: {[r['layer_idx'] for r in failed]}")
        for failure in failed:
            logger.error(f"Layer {failure['layer_idx']}: {failure['error']}")
    
    logger.info("✅ Training completed successfully!")

if __name__ == "__main__":
    main() 