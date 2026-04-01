"""
Modal deployment for PaliGemma2 SAE training pipeline (8-GPU parallel).

Architecture:
  Phase 1 — Cache activations: 8 GPUs each cache ~7 chunks of VQAv2 activations
  Phase 2 — Train SAEs: 8 GPUs each train ~3-4 layers (both methods) across all 50 chunks

Usage:
    cd finetune/paligemma2
    MODAL_PROFILE=hunar-oxford modal run modal_train.py

Matches the paper's recipe:
- 50,000 VQAv2 training samples + 5,000 validation
- Chunk size 1,000 (50 training + 5 validation chunks)
- All 26 layers
- Methods: pretrained (Gemma Scope init) + random
- TopK SAE with k=50, width 16,384
- Training batch size 8
- LR: 2e-4 / sqrt(d_sae / 16384)
"""

import os
import modal
from pathlib import Path

# --------------- Modal configuration ---------------

VOLUME_NAME = "vlm-scope-data-v2"
GPU_TYPE = "A100"
TIMEOUT = 86400  # 24h (Modal max)

app = modal.App("vlm-scope-sae-train-v2")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.44",
        "sae-lens>=4.0",
        "nnsight>=0.3",
        "datasets",
        "h5py",
        "tqdm",
        "huggingface-hub",
        "Pillow",
        "numpy",
        "accelerate",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        # Disable wandb to prevent timeout/auth delays
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
N_VAL_SAMPLES = 5_000
CHUNK_SIZE = 1_000
FROM_LAYER = 0
TO_LAYER = 26
METHODS = ["pretrained", "random"]
TRAINING_BATCH_SIZE = 8
CACHING_BATCH_SIZE = 4
MODEL_NAME = "google/paligemma2-3b-pt-224"
RESULTS_BASE = "/vol/results/paligemma2"
N_CACHING_GPUS = 8
N_TRAINING_GPUS = 8

# Checkpoint percentages for intermediate saves (pct -> chunk_idx 0-indexed)
CHECKPOINT_PCTS = {25: 12, 50: 25, 75: 37}


# ================================================================
#  Phase 1: Cache activations (one function per GPU)
# ================================================================

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def cache_worker(worker_id: int, chunk_indices: list):
    """Cache PaliGemma2 residual-stream activations for assigned chunks."""
    import sys
    import gc

    import torch
    import h5py
    from pathlib import Path
    from tqdm import tqdm
    from datasets import load_dataset
    from nnsight import NNsight

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions

    activations_dir = Path(RESULTS_BASE) / "run" / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Cache W{worker_id}] Loading PaliGemma2-3B...")
    processor, model_raw = initialize_vlm_model(MODEL_NAME, device="cpu")
    model_raw = model_raw.to("cuda")
    nns_model = NNsight(model_raw)

    print(f"[Cache W{worker_id}] Loading VQAv2...")
    vqa_dataset = load_dataset("lmms-lab/VQAv2", split="validation")
    print(f"[Cache W{worker_id}] VQAv2 size: {len(vqa_dataset)}")
    print(f"[Cache W{worker_id}] Assigned chunks: {chunk_indices}")

    total_samples = N_TRAINING_SAMPLES + N_VAL_SAMPLES

    for ci_num, chunk_idx in enumerate(chunk_indices):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, total_samples)
        h5_path = activations_dir / f"chunk_{chunk_start}_{chunk_end}.h5"

        if h5_path.exists():
            print(f"[Cache W{worker_id}] SKIP chunk {chunk_idx} ({chunk_start}->{chunk_end}) — already cached")
            continue

        print(f"[Cache W{worker_id}] Chunk {ci_num+1}/{len(chunk_indices)}: "
              f"samples {chunk_start}->{chunk_end}")

        for i in tqdm(range(chunk_start, chunk_end, CACHING_BATCH_SIZE),
                      desc=f"W{worker_id} chunk {chunk_idx}"):
            actual_end = min(i + CACHING_BATCH_SIZE, chunk_end)

            batch_ids, batch_masks, batch_pv, img_pos = [], [], [], []
            for j in range(i, actual_end):
                sample = vqa_dataset[j]
                image = sample["image"].convert("RGB")
                prompt = sample["question"]
                ids, mask, pv = process_vlm_inputs(
                    image, prompt, processor, model_raw, device="cuda"
                )
                batch_ids.append(ids.squeeze(0))
                batch_masks.append(mask.squeeze(0))
                batch_pv.append(pv)
                img_pos.append(get_image_token_positions(ids))

            pad_id = processor.tokenizer.pad_token_id or 0
            ids_pad = torch.nn.utils.rnn.pad_sequence(
                batch_ids, batch_first=True, padding_value=pad_id
            ).to("cuda")
            mask_pad = torch.nn.utils.rnn.pad_sequence(
                batch_masks, batch_first=True, padding_value=0
            ).to("cuda")
            pv_cat = torch.cat(batch_pv, dim=0).to("cuda")

            layer_outs = []
            with torch.no_grad():
                with nns_model.trace(
                    input_ids=ids_pad,
                    attention_mask=mask_pad,
                    pixel_values=pv_cat,
                ) as tr:
                    for li in range(FROM_LAYER, TO_LAYER):
                        layer_outs.append(
                            nns_model.model.language_model.layers[li].output.save()
                        )

            actual_batch = actual_end - i

            # Debug on first batch of first chunk
            if i == chunk_start and ci_num == 0:
                raw0 = layer_outs[0]
                print(f"[Cache W{worker_id}] DEBUG: ids_pad={ids_pad.shape}, "
                      f"mask_pad={mask_pad.shape}")
                print(f"[Cache W{worker_id}] DEBUG: layer_outs[0] type={type(raw0)}")
                if isinstance(raw0, tuple):
                    for ti, t in enumerate(raw0):
                        print(f"  [{ti}] type={type(t)}, "
                              f"shape={t.shape if hasattr(t, 'shape') else 'N/A'}")

            with h5py.File(h5_path, "a") as f:
                for li, layer_idx in enumerate(range(FROM_LAYER, TO_LAYER)):
                    grp = f.require_group(f"layer_{layer_idx}")
                    raw = layer_outs[li]
                    if isinstance(raw, tuple):
                        out_tensor = raw[0].detach().cpu()
                    elif hasattr(raw, "detach"):
                        out_tensor = raw.detach().cpu()
                    else:
                        out_tensor = raw
                    if out_tensor.ndim == 2:
                        out_tensor = out_tensor.unsqueeze(0)
                    for si in range(actual_batch):
                        seq_len = int(mask_pad[si].sum().item())
                        act = out_tensor[si, :seq_len].float().contiguous().numpy()
                        ds = grp.create_dataset(
                            f"sample_{i + si}", data=act, compression=None
                        )
                        ds.attrs["img_start"] = int(img_pos[si][0])
                        ds.attrs["img_end"] = int(img_pos[si][1])

            del ids_pad, mask_pad, pv_cat, layer_outs
            del batch_ids, batch_masks, batch_pv
            torch.cuda.empty_cache()

        volume.commit()
        print(f"[Cache W{worker_id}] Done chunk {chunk_idx} ({chunk_start}->{chunk_end})")

    # Cleanup
    del nns_model, model_raw, processor
    torch.cuda.empty_cache()
    gc.collect()
    return f"Cache W{worker_id}: {len(chunk_indices)} chunks done"


# ================================================================
#  Phase 2: Train SAEs (one function per GPU, subset of layers)
# ================================================================

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def train_worker(worker_id: int, layer_indices: list):
    """Train SAEs for assigned layers across all cached chunks."""
    import sys
    import json
    import gc
    import shutil

    import torch
    import h5py
    from pathlib import Path
    from tqdm import tqdm
    from torch.optim import Adam

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_sae

    run_dir = Path(RESULTS_BASE) / "run"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    activations_dir = run_dir / "activations"
    for d in [checkpoint_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total_chunks = (N_TRAINING_SAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE  # 50

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

    print(f"[Train W{worker_id}] Layers: {layer_indices}, "
          f"Methods: {METHODS}, Chunks: {total_chunks}")

    for layer_idx in layer_indices:
        for method in METHODS:
            print(f"\n[Train W{worker_id}] === Layer {layer_idx} / {method} ===")

            init_random = method.lower() == "random"
            ckpt_path = checkpoint_dir / f"{method}_layer_{layer_idx}.pt"
            optim_path = checkpoint_dir / f"{method}_layer_{layer_idx}_optim.pt"
            log_path = log_dir / f"metrics_{method}_layer_{layer_idx}.csv"

            # Init log file
            if not log_path.exists():
                with open(log_path, "w") as f:
                    f.write("total_tokens,fvu,recon_loss,sparsity,"
                            "chunk_start,chunk_end,batch_idx\n")

            # Init or resume SAE
            if ckpt_path.exists():
                sae = initialize_sae(
                    layer_idx, checkpoint_path=str(ckpt_path),
                    device="cpu", cache_dir="/vol/cache/huggingface",
                )
            else:
                sae = initialize_sae(
                    layer_idx, initialize_random=init_random,
                    device="cpu", cache_dir="/vol/cache/huggingface",
                )

            d_sae = getattr(sae.cfg, "d_sae", 16384)
            lr = 2e-4 / (d_sae / (2 ** 14)) ** 0.5

            optimizer = Adam(sae.parameters(), lr=lr)
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
                sae_dev = "cuda"
                sae_dt = next(sae.parameters()).dtype

                # Move optimizer states to GPU
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device=sae_dev, dtype=sae_dt)

                bs = TRAINING_BATCH_SIZE
                is_text = method.lower() == "text-only"
                is_img = method.lower() == "image-only"
                last_fvu, last_recon, last_sparsity = 0.0, 0.0, 0.0
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
                            mask = torch.ones(
                                max_seq, dtype=torch.bool, device=sae_dev
                            )
                            mask[seq:] = False
                        else:
                            mask = torch.ones(
                                max_seq, dtype=torch.bool, device=sae_dev
                            )

                        if ((is_text or is_img)
                                and img_s is not None and img_e is not None):
                            if is_img:
                                sp = torch.zeros(
                                    max_seq, dtype=torch.bool, device=sae_dev
                                )
                                sp[img_s:img_e + 1] = True
                            else:
                                sp = torch.ones(
                                    max_seq, dtype=torch.bool, device=sae_dev
                                )
                                sp[img_s:img_e + 1] = False
                            mask = mask & sp

                        if mask.sum() == 0:
                            continue
                        acts_list.append(act)
                        masks_list.append(mask)

                    if not acts_list:
                        continue

                    batch_acts = torch.stack(acts_list)
                    batch_masks = torch.stack(masks_list)

                    pred = sae(batch_acts)
                    err = pred - batch_acts
                    m_err = err * batch_masks.unsqueeze(-1)
                    nv = batch_masks.sum()
                    recon = (m_err ** 2).sum() / (nv + 1e-8)

                    m_acts = batch_acts * batch_masks.unsqueeze(-1)
                    mu = m_acts.sum(dim=(0, 1)) / (nv + 1e-8)
                    var = ((m_acts - mu) ** 2).sum() / (nv + 1e-8)
                    fvu = recon / (var + 1e-8)

                    local_tokens += int(nv.item())
                    fvu.backward()

                    with torch.no_grad():
                        codes = sae.encode(batch_acts)
                        m_codes = codes * batch_masks.unsqueeze(-1)
                        sparsity = (m_codes != 0).float().sum() / (
                            nv * codes.shape[-1] + 1e-8
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Normalize decoder weights
                    if hasattr(sae, "set_decoder_norm_to_unit_norm"):
                        sae.set_decoder_norm_to_unit_norm()
                    else:
                        with torch.no_grad():
                            norms = sae.W_dec.norm(
                                dim=-1, keepdim=True
                            ).clamp(min=1e-8)
                            sae.W_dec.div_(norms)

                    last_fvu = fvu.item()
                    last_recon = recon.item()
                    last_sparsity = sparsity.item()
                    batch_idx += 1

                    del batch_acts, batch_masks, pred, err, m_err
                    del m_acts, codes, m_codes
                    torch.cuda.empty_cache()

                total_tokens += local_tokens

                # Log one entry per chunk
                with open(log_path, "a") as lf:
                    lf.write(
                        f"{total_tokens},{last_fvu:.6f},{last_recon:.4f},"
                        f"{last_sparsity:.10f},{chunk_start},{chunk_end},"
                        f"{batch_idx}\n"
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
            print(f"[Train W{worker_id}] Done L{layer_idx}/{method}: "
                  f"{total_tokens} tokens, final FVU={last_fvu:.4f}")

            del sae, optimizer
            torch.cuda.empty_cache()
            gc.collect()
            volume.commit()

    return f"Train W{worker_id}: layers {layer_indices} done"


# ================================================================
#  Entrypoint: orchestrate Phase 1 + Phase 2
# ================================================================

@app.local_entrypoint()
def main():
    import math

    total_samples = N_TRAINING_SAMPLES + N_VAL_SAMPLES  # 55000
    total_all_chunks = (total_samples + CHUNK_SIZE - 1) // CHUNK_SIZE  # 55

    # === Phase 1: Parallel caching ===
    chunks_per_worker = math.ceil(total_all_chunks / N_CACHING_GPUS)
    cache_assignments = []
    for w in range(N_CACHING_GPUS):
        start = w * chunks_per_worker
        end = min(start + chunks_per_worker, total_all_chunks)
        worker_chunks = list(range(start, end))
        if worker_chunks:
            cache_assignments.append((w, worker_chunks))

    print(f"{'='*60}")
    print(f"[Phase 1] Caching {total_all_chunks} chunks "
          f"across {len(cache_assignments)} GPUs")
    print(f"{'='*60}")
    for w, chunks in cache_assignments:
        sample_ranges = [(c * CHUNK_SIZE,
                          min((c + 1) * CHUNK_SIZE, total_samples))
                         for c in chunks]
        print(f"  GPU {w}: chunks {chunks[0]}-{chunks[-1]} "
              f"(samples {sample_ranges[0][0]}->{sample_ranges[-1][1]})")

    cache_results = list(cache_worker.starmap(cache_assignments))
    for r in cache_results:
        print(r)

    # === Phase 2: Parallel training ===
    n_layers = TO_LAYER - FROM_LAYER  # 26
    layers_per_worker = math.ceil(n_layers / N_TRAINING_GPUS)
    train_assignments = []
    for w in range(N_TRAINING_GPUS):
        start_layer = FROM_LAYER + w * layers_per_worker
        end_layer = min(start_layer + layers_per_worker, TO_LAYER)
        worker_layers = list(range(start_layer, end_layer))
        if worker_layers:
            train_assignments.append((w, worker_layers))

    print(f"\n{'='*60}")
    print(f"[Phase 2] Training {n_layers} layers x {len(METHODS)} methods "
          f"across {len(train_assignments)} GPUs")
    print(f"{'='*60}")
    for w, layers in train_assignments:
        print(f"  GPU {w}: layers {layers}")

    train_results = list(train_worker.starmap(train_assignments))
    for r in train_results:
        print(r)

    print(f"\n{'='*60}")
    print("[SUCCESS] All training complete!")
    print(f"  Checkpoints: {RESULTS_BASE}/run/checkpoints/")
    print(f"  Activations: {RESULTS_BASE}/run/activations/")
    print(f"  Logs: {RESULTS_BASE}/run/logs/")
    print(f"{'='*60}")
