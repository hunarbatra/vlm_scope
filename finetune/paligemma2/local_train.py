"""
Local 8-GPU JumpReLU SAE training for PaliGemma2-3B.

Port of modal_train_jumprelu.py to local 8x A5000 GPUs.
Online training: extracts activations on-the-fly (no disk caching).

Training loop matches saprmarks/dictionary_learning JumpReluTrainer:
  - Same JumpReLU loss with STE gradients
  - Same optimizer (Adam, beta1=0, beta2=0.999, lr=7e-5)
  - Same hyperparams (bandwidth=0.001, target_l0=50, sparsity_coeff=1.0)
  - Same batching (pad, mask, flatten valid tokens)
  - Same checkpoint logic (every chunk, intermediate at 25/50/75%)

Usage:
    cd vlm_scope/finetune/paligemma2
    HF_TOKEN=hf_... python3 local_train.py
    HF_TOKEN=hf_... python3 local_train.py --layers 0 1 2 3
    HF_TOKEN=hf_... python3 local_train.py --model google/paligemma2-3b-mix-448
"""

import os
import sys
import gc
import math
import shutil
import argparse
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# ======================== Configuration ========================
# Mirrors modal_train_jumprelu.py exactly

MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "google/paligemma2-3b-mix-448")
RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR", "/data1/vlm_scope_sae_mix448"))
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"

HF_CACHE = str(RESULTS_DIR / "hf_cache")
HF_DATASETS_CACHE = str(RESULTS_DIR / "hf_datasets_cache")

# Set env vars at module level so spawned workers inherit them
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ.setdefault("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

N_TRAINING_SAMPLES = 50_000
CHUNK_SIZE = 1_000
FROM_LAYER = 0
TO_LAYER = 26
METHODS = ["pretrained"]
TRAINING_BATCH_SIZE = 8
N_GPUS = 8

# JumpReLU hyperparameters — matches saprmarks/dictionary_learning JumpReluTrainer
TARGET_L0 = 50.0
BANDWIDTH = 0.001
SPARSITY_COEFF = 1.0
LR = 7e-5
LR_WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 2000

# VLM forward batch size (smaller for 448px — more tokens per sample)
VLM_BATCH_SIZE = 2

# Checkpoint percentages (pct -> chunk_idx, same as modal)
CHECKPOINT_PCTS = {25: 12, 50: 25, 75: 37}


# ======================== Helper: extract activations ========================

def extract_chunk_activations(
    nns_model, processor, model_raw, vqa_dataset,
    chunk_start, chunk_end, layer_idx, device,
):
    """Extract activations for one chunk of VQAv2 samples at a single layer.

    Returns list of (act_tensor, img_start, img_end) — same format as
    modal_train_jumprelu.py load_chunk().
    """
    from utils import process_vlm_inputs, get_image_token_positions

    chunk = []
    for i in range(chunk_start, chunk_end, VLM_BATCH_SIZE):
        actual_end = min(i + VLM_BATCH_SIZE, chunk_end)

        batch_ids, batch_masks, batch_pv, img_pos = [], [], [], []
        for j in range(i, actual_end):
            sample = vqa_dataset[j]
            image = sample["image"].convert("RGB")
            # PaliGemma uses task-prefix prompts: "answer en {question}"
            prompt = f"answer en {sample['question']}"
            ids, mask, pv = process_vlm_inputs(
                image, prompt, processor, model_raw, device=device
            )
            batch_ids.append(ids.squeeze(0))
            batch_masks.append(mask.squeeze(0))
            batch_pv.append(pv)
            img_pos.append(get_image_token_positions(ids))

        pad_id = processor.tokenizer.pad_token_id or 0
        ids_pad = torch.nn.utils.rnn.pad_sequence(
            batch_ids, batch_first=True, padding_value=pad_id
        ).to(device)
        mask_pad = torch.nn.utils.rnn.pad_sequence(
            batch_masks, batch_first=True, padding_value=0
        ).to(device)
        pv_cat = torch.cat(batch_pv, dim=0).to(device)

        with torch.no_grad():
            with nns_model.trace(
                input_ids=ids_pad,
                attention_mask=mask_pad,
                pixel_values=pv_cat,
            ):
                layer_out = nns_model.model.language_model.layers[
                    layer_idx
                ].output.save()

        # Unpack
        if isinstance(layer_out, tuple):
            out_tensor = layer_out[0].detach().cpu()
        elif hasattr(layer_out, "detach"):
            out_tensor = layer_out.detach().cpu()
        else:
            out_tensor = layer_out
        if out_tensor.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0)

        actual_batch = actual_end - i
        for si in range(actual_batch):
            seq_len = int(mask_pad[si].sum().item())
            act = out_tensor[si, :seq_len].float()  # (seq, d_in)
            chunk.append((
                act,
                int(img_pos[si][0]),
                int(img_pos[si][1]),
            ))

        del ids_pad, mask_pad, pv_cat, layer_out, out_tensor
        del batch_ids, batch_masks, batch_pv
        torch.cuda.empty_cache()

    return chunk


# ======================== Training ========================

def train_worker(gpu_id: int, layer_indices: list):
    """Train JumpReLU SAEs for assigned layers.

    Mirrors modal_train_jumprelu.py train_worker() exactly,
    except activations are extracted online instead of loaded from H5.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    from nnsight import NNsight
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae

    for d in [CHECKPOINT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    total_chunks = N_TRAINING_SAMPLES // CHUNK_SIZE  # 50

    # Load VLM once, reuse for all layers
    print(f"[GPU{gpu_id}] Loading {MODEL_NAME}...")
    hf_token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, token=hf_token)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, token=hf_token,
    ).to(device).eval()
    nns_model = NNsight(model_raw)

    print(f"[GPU{gpu_id}] Loading VQAv2...")
    vqa_dataset = load_dataset("lmms-lab/VQAv2", split="validation")
    print(f"[GPU{gpu_id}] VQAv2: {len(vqa_dataset)}, layers: {layer_indices}")

    for layer_idx in layer_indices:
        for method in METHODS:
            print(f"\n[GPU{gpu_id}] === Layer {layer_idx} / {method} ===")

            init_random = method.lower() == "random"
            ckpt_path = CHECKPOINT_DIR / f"{method}_layer_{layer_idx}.pt"
            optim_path = CHECKPOINT_DIR / f"{method}_layer_{layer_idx}_optim.pt"
            log_path = LOG_DIR / f"metrics_{method}_layer_{layer_idx}.csv"

            # Init log file — same columns as modal
            if not log_path.exists():
                with open(log_path, "w") as f:
                    f.write("total_tokens,fvu,recon_loss,l0,sparsity_loss,"
                            "chunk_start,chunk_end,batch_idx\n")

            # Init or resume SAE
            if ckpt_path.exists():
                sae = initialize_jumprelu_sae(
                    layer_idx, checkpoint_path=str(ckpt_path),
                    device="cpu",
                )
                print(f"  Resumed from checkpoint")
            else:
                sae = initialize_jumprelu_sae(
                    layer_idx, initialize_random=init_random,
                    device="cpu",
                )

            # Optimizer — same as modal
            optimizer = torch.optim.Adam(
                sae.parameters(), lr=LR, betas=(0.0, 0.999), eps=1e-8
            )

            def lr_lambda(step):
                if step < LR_WARMUP_STEPS:
                    return step / max(LR_WARMUP_STEPS, 1)
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )

            if optim_path.exists():
                try:
                    optimizer.load_state_dict(
                        torch.load(optim_path, map_location="cpu",
                                   weights_only=True)
                    )
                except Exception:
                    pass

            total_tokens = 0
            batch_idx = 0

            for chunk_idx in tqdm(range(total_chunks),
                                  desc=f"GPU{gpu_id} L{layer_idx}/{method}",
                                  disable=(gpu_id != 0)):
                chunk_start = chunk_idx * CHUNK_SIZE
                chunk_end = min(chunk_start + CHUNK_SIZE, N_TRAINING_SAMPLES)

                # --- Only difference from modal: extract online instead of H5 ---
                acts = extract_chunk_activations(
                    nns_model, processor, model_raw, vqa_dataset,
                    chunk_start, chunk_end, layer_idx, device,
                )
                if not acts:
                    continue

                # --- From here: IDENTICAL to modal_train_jumprelu.py ---
                sae.to(device)
                sae.train()
                sae_dev = device
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
                            mask = torch.ones(
                                max_seq, dtype=torch.bool, device=sae_dev
                            )
                            mask[seq:] = False
                        else:
                            mask = torch.ones(
                                max_seq, dtype=torch.bool, device=sae_dev
                            )

                        if mask.sum() == 0:
                            continue
                        acts_list.append(act)
                        masks_list.append(mask)

                    if not acts_list:
                        continue

                    batch_acts = torch.stack(acts_list)    # (B, seq, d_in)
                    batch_masks = torch.stack(masks_list)  # (B, seq)

                    # Flatten valid tokens for JumpReLU loss
                    valid_acts = batch_acts[batch_masks]   # (N_valid, d_in)
                    nv = valid_acts.shape[0]
                    if nv == 0:
                        continue

                    local_tokens += nv

                    # Sparsity warmup
                    sparsity_scale = min(
                        1.0, batch_idx / max(SPARSITY_WARMUP_STEPS, 1)
                    )

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

                # Log one entry per chunk — same format as modal
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

                # Intermediate checkpoints at 25/50/75%
                if method.lower() == "pretrained":
                    for pct, target_ci in CHECKPOINT_PCTS.items():
                        if chunk_idx == target_ci:
                            pct_dir = RESULTS_DIR / f"checkpoint_{pct}pct"
                            pct_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(
                                ckpt_path,
                                pct_dir / f"{method}_layer_{layer_idx}.pt",
                            )
                            print(f"  [SAVE] {pct}% checkpoint for "
                                  f"{method}_layer_{layer_idx}")

            # End of all chunks for this (layer, method)
            print(f"[GPU{gpu_id}] Done L{layer_idx}/{method}: "
                  f"{total_tokens:,} tokens, FVU={last_fvu:.4f}, L0={last_l0:.1f}")

            del sae, optimizer
            torch.cuda.empty_cache()
            gc.collect()

    # Cleanup VLM
    del nns_model, model_raw, processor
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[GPU{gpu_id}] All layers complete")


# ======================== Main ========================

def main():
    global MODEL_NAME, RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR

    parser = argparse.ArgumentParser(
        description="Local JumpReLU SAE training (online, no caching)"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers (default: all 26)")
    parser.add_argument("--gpus", type=int, default=N_GPUS)
    parser.add_argument("--model", type=str, default=None,
                        help="Override VLM model name")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    args = parser.parse_args()

    if args.model:
        MODEL_NAME = args.model
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
        LOG_DIR = RESULTS_DIR / "logs"

    layer_list = args.layers if args.layers else list(range(FROM_LAYER, TO_LAYER))
    n_gpus = args.gpus
    n_layers = len(layer_list)
    layers_per_worker = math.ceil(n_layers / n_gpus)
    assignments = []
    for w in range(n_gpus):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, n_layers)
        worker_layers = layer_list[start:end]
        if worker_layers:
            assignments.append((w, worker_layers))

    print(f"{'=' * 60}")
    print(f"JumpReLU SAE Training (local, online)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layers: {n_layers}, Methods: {METHODS}")
    print(f"  Target L0: {TARGET_L0}, Bandwidth: {BANDWIDTH}")
    print(f"  LR: {LR}, Sparsity coeff: {SPARSITY_COEFF}")
    print(f"  VLM batch: {VLM_BATCH_SIZE}, SAE batch: {TRAINING_BATCH_SIZE}")
    print(f"  Training samples: {N_TRAINING_SAMPLES:,}")
    print(f"  Output: {RESULTS_DIR}")
    print(f"{'=' * 60}")
    for gpu_id, layers in assignments:
        print(f"  GPU {gpu_id}: layers {layers}")

    t0 = time.time()

    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id, layers in assignments:
        p = mp.Process(target=train_worker, args=(gpu_id, layers))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    elapsed = time.time() - t0

    if failed:
        print(f"\n[ERROR] Workers {failed} failed!")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"[DONE] Total: {elapsed / 3600:.1f}h")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Logs: {LOG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
