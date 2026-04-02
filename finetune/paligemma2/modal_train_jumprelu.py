"""
Modal deployment for PaliGemma2 JumpReLU SAE training (8-GPU parallel).

Uses the same cached activations from the TopK training run.
Trains JumpReLU SAEs initialized from Gemma Scope pretrained weights,
which are natively JumpReLU — so no architecture mismatch.

Methods:
  - pretrained: init from Gemma Scope JumpReLU weights (threshold included)
  - random: random init JumpReLU

Architecture:
  Phase 2 only — reuses cached activations from the TopK run.
  8 GPUs, each trains ~3-4 layers across all 50 training chunks.

Usage:
    cd finetune/paligemma2
    MODAL_PROFILE=hunar-oxford modal run modal_train_jumprelu.py
"""

import os
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
GPU_TYPE = "H100"
TIMEOUT = 86400

app = modal.App("vlm-scope-jumprelu-train")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.44",
        "sae-lens>=4.0",
        "h5py",
        "tqdm",
        "huggingface-hub",
        "numpy",
        "accelerate",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

# --------------- Training parameters ---------------

N_TRAINING_SAMPLES = 50_000
CHUNK_SIZE = 1_000
FROM_LAYER = 0
TO_LAYER = 26
METHODS = ["pretrained"]
TRAINING_BATCH_SIZE = 8
RESULTS_BASE = "/vol/results/paligemma2"
N_GPUS = 8

# JumpReLU-specific hyperparameters
TARGET_L0 = 50.0         # Target sparsity (matching TopK k=50)
BANDWIDTH = 0.001         # STE bandwidth for threshold gradients
SPARSITY_COEFF = 1.0     # Sparsity penalty coefficient
LR = 7e-5                # Learning rate (from JumpReLU paper)
LR_WARMUP_STEPS = 1000   # LR warmup period
SPARSITY_WARMUP_STEPS = 2000  # Ramp up sparsity penalty over this many steps

# Checkpoint percentages for intermediate saves (pretrained only)
CHECKPOINT_PCTS = {25: 12, 50: 25, 75: 37}


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def train_worker(worker_id: int, layer_indices: list):
    """Train JumpReLU SAEs for assigned layers across all cached chunks."""
    import sys
    import gc
    import shutil

    import torch
    import h5py
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_jumprelu_sae

    run_dir = Path(RESULTS_BASE) / "run_jumprelu"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    activations_dir = Path(RESULTS_BASE) / "run" / "activations"  # Reuse from TopK run
    for d in [checkpoint_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total_chunks = N_TRAINING_SAMPLES // CHUNK_SIZE  # 50

    def load_chunk(start_idx, end_idx, layer_idx):
        h5_path = activations_dir / f"chunk_{start_idx}_{end_idx}.h5"
        if not h5_path.exists():
            print(f"  [WARN] Missing {h5_path}")
            return []
        chunk = []
        with h5py.File(h5_path, "r") as f:
            grp = f.get(f"layer_{layer_idx}")
            if grp is None:
                return chunk
            for i in range(start_idx, end_idx):
                key = f"sample_{i}"
                if key in grp:
                    ds = grp[key]
                    act = torch.from_numpy(ds[:])
                    s = ds.attrs.get("img_start")
                    e = ds.attrs.get("img_end")
                    chunk.append((
                        act,
                        int(s) if s is not None else None,
                        int(e) if e is not None else None,
                    ))
        return chunk

    print(f"[JR Train W{worker_id}] Layers: {layer_indices}, "
          f"Methods: {METHODS}, Chunks: {total_chunks}")

    for layer_idx in layer_indices:
        for method in METHODS:
            print(f"\n[JR Train W{worker_id}] === Layer {layer_idx} / {method} ===")

            init_random = method.lower() == "random"
            ckpt_path = checkpoint_dir / f"{method}_layer_{layer_idx}.pt"
            optim_path = checkpoint_dir / f"{method}_layer_{layer_idx}_optim.pt"
            log_path = log_dir / f"metrics_{method}_layer_{layer_idx}.csv"

            # Init log file
            if not log_path.exists():
                with open(log_path, "w") as f:
                    f.write("total_tokens,fvu,recon_loss,l0,sparsity_loss,"
                            "chunk_start,chunk_end,batch_idx\n")

            # Init or resume SAE
            if ckpt_path.exists():
                sae = initialize_jumprelu_sae(
                    layer_idx, checkpoint_path=str(ckpt_path),
                    device="cpu", cache_dir="/vol/cache/huggingface",
                )
            else:
                sae = initialize_jumprelu_sae(
                    layer_idx, initialize_random=init_random,
                    device="cpu", cache_dir="/vol/cache/huggingface",
                )

            # Optimizer: Adam with betas=(0.0, 0.999) as in JumpReLU paper
            optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.0, 0.999), eps=1e-8)

            # LR warmup scheduler
            def lr_lambda(step):
                if step < LR_WARMUP_STEPS:
                    return step / max(LR_WARMUP_STEPS, 1)
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            if optim_path.exists():
                try:
                    optimizer.load_state_dict(
                        torch.load(optim_path, map_location="cpu")
                    )
                except Exception:
                    pass

            total_tokens = 0
            batch_idx = 0

            for chunk_idx in tqdm(range(total_chunks),
                                  desc=f"W{worker_id} L{layer_idx}/{method}"):
                chunk_start = chunk_idx * CHUNK_SIZE
                chunk_end = min(chunk_start + CHUNK_SIZE, N_TRAINING_SAMPLES)

                acts = load_chunk(chunk_start, chunk_end, layer_idx)
                if not acts:
                    continue

                sae.to("cuda")
                sae.train()
                sae_dev = "cuda"
                sae_dt = next(sae.parameters()).dtype

                # Move optimizer states to GPU
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device=sae_dev, dtype=sae_dt)

                bs = TRAINING_BATCH_SIZE
                last_fvu, last_recon, last_l0 = 0.0, 0.0, 0.0
                local_tokens = 0

                for b_start in range(0, len(acts), bs):
                    b_end = min(b_start + bs, len(acts))
                    entries = acts[b_start:b_end]
                    max_seq = max(e[0].shape[0] for e in entries)

                    acts_list, masks_list = [], []
                    for (act, img_s, img_e) in entries:
                        seq = act.shape[0]
                        act = act.to(sae_dev).to(sae_dt)
                        if seq < max_seq:
                            act = torch.nn.functional.pad(
                                act, (0, 0, 0, max_seq - seq)
                            )
                            mask = torch.ones(max_seq, dtype=torch.bool, device=sae_dev)
                            mask[seq:] = False
                        else:
                            mask = torch.ones(max_seq, dtype=torch.bool, device=sae_dev)

                        if mask.sum() == 0:
                            continue
                        acts_list.append(act)
                        masks_list.append(mask)

                    if not acts_list:
                        continue

                    batch_acts = torch.stack(acts_list)   # (B, seq, d_in)
                    batch_masks = torch.stack(masks_list)  # (B, seq)

                    # Flatten valid tokens for JumpReLU loss
                    valid_acts = batch_acts[batch_masks]  # (N_valid, d_in)
                    nv = valid_acts.shape[0]
                    if nv == 0:
                        continue

                    local_tokens += nv

                    # Sparsity warmup: ramp up over first SPARSITY_WARMUP_STEPS
                    sparsity_scale = min(1.0, batch_idx / max(SPARSITY_WARMUP_STEPS, 1))

                    # Compute JumpReLU loss
                    loss, recon_loss, l0, fvu = sae.compute_loss(
                        valid_acts,
                        bandwidth=BANDWIDTH,
                        target_l0=TARGET_L0,
                        sparsity_coeff=SPARSITY_COEFF * sparsity_scale,
                    )

                    loss.backward()

                    # Remove gradient parallel to decoder directions
                    sae.remove_gradient_parallel_to_decoder_directions()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Normalize decoder weights to unit norm
                    sae.set_decoder_norm_to_unit_norm()

                    last_fvu = fvu
                    last_recon = recon_loss
                    last_l0 = l0
                    batch_idx += 1

                    del valid_acts, batch_acts, batch_masks
                    torch.cuda.empty_cache()

                total_tokens += local_tokens

                # Log one entry per chunk
                with open(log_path, "a") as lf:
                    lf.write(
                        f"{total_tokens},{last_fvu:.6f},{last_recon:.4f},"
                        f"{last_l0:.2f},{0.0:.4f},"
                        f"{chunk_start},{chunk_end},{batch_idx}\n"
                    )

                # Move SAE back to CPU between chunks
                sae.to("cpu")
                torch.cuda.empty_cache()
                del acts
                gc.collect()

                # Save checkpoint every chunk
                torch.save(sae.state_dict(), ckpt_path)
                torch.save(optimizer.state_dict(), optim_path)

                # Intermediate checkpoints at 25/50/75% (pretrained only)
                if method.lower() == "pretrained":
                    for pct, target_ci in CHECKPOINT_PCTS.items():
                        if chunk_idx == target_ci:
                            pct_dir = run_dir / f"checkpoint_{pct}pct"
                            pct_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(
                                ckpt_path,
                                pct_dir / f"{method}_layer_{layer_idx}.pt",
                            )
                            print(f"  [SAVE] {pct}% checkpoint for "
                                  f"{method}_layer_{layer_idx}")

                # Commit to volume every 5 chunks
                if chunk_idx % 5 == 0:
                    volume.commit()

            # End of all chunks for this (layer, method)
            print(f"[JR Train W{worker_id}] Done L{layer_idx}/{method}: "
                  f"{total_tokens} tokens, final FVU={last_fvu:.4f}, L0={last_l0:.1f}")

            del sae, optimizer
            torch.cuda.empty_cache()
            gc.collect()
            volume.commit()

    return f"JR Train W{worker_id}: layers {layer_indices} done"


@app.local_entrypoint()
def main():
    import math

    n_layers = TO_LAYER - FROM_LAYER  # 26
    layers_per_worker = math.ceil(n_layers / N_GPUS)
    assignments = []
    for w in range(N_GPUS):
        start_layer = FROM_LAYER + w * layers_per_worker
        end_layer = min(start_layer + layers_per_worker, TO_LAYER)
        worker_layers = list(range(start_layer, end_layer))
        if worker_layers:
            assignments.append((w, worker_layers))

    print(f"{'=' * 60}")
    print(f"[JumpReLU Training] {n_layers} layers x {len(METHODS)} methods "
          f"across {len(assignments)} GPUs")
    print(f"  Target L0: {TARGET_L0}, Bandwidth: {BANDWIDTH}")
    print(f"  LR: {LR}, Sparsity coeff: {SPARSITY_COEFF}")
    print(f"  Activations: {RESULTS_BASE}/run/activations/ (reused from TopK)")
    print(f"  Output: {RESULTS_BASE}/run_jumprelu/")
    print(f"{'=' * 60}")
    for w, layers in assignments:
        print(f"  GPU {w}: layers {layers}")

    results = list(train_worker.starmap(assignments))
    for r in results:
        print(r)

    print(f"\n{'=' * 60}")
    print("[SUCCESS] JumpReLU training complete!")
    print(f"  Checkpoints: {RESULTS_BASE}/run_jumprelu/checkpoints/")
    print(f"  Logs: {RESULTS_BASE}/run_jumprelu/logs/")
    print(f"{'=' * 60}")
