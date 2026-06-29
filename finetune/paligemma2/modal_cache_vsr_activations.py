"""
Cache PaliGemma2 residual-stream activations for VSR dataset on Modal.

Same H5 format as VQA activations: layer_{idx}/sample_{idx} -> (seq, 2304)
with img_start/img_end attrs.

VSR has ~10K samples → 10 chunks of 1000.

Usage:
    export HF_TOKEN=hf_...
    MODAL_PROFILE=hunar-oxford modal run modal_cache_vsr_activations.py
"""
import os
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
GPU_TYPE = "A100"
TIMEOUT = 86400

app = modal.App("vlm-scope-cache-vsr")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers>=4.44", "sae-lens>=4.0",
        "nnsight>=0.3", "datasets", "h5py", "tqdm", "huggingface-hub",
        "Pillow", "numpy", "accelerate", "requests",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR"),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR"),
        "WANDB_MODE": "disabled",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

RESULTS_BASE = "/vol/results/paligemma2"
MODEL_NAME = "google/paligemma2-3b-pt-224"
FROM_LAYER = 0
TO_LAYER = 26
CHUNK_SIZE = 1000
CACHING_BATCH_SIZE = 1  # VSR images are URL-loaded, process one at a time
N_CACHING_GPUS = 4  # 4 GPUs for ~10K samples

VSR_DATASET = "cambridgeltl/vsr_random"
VSR_SPLITS = ["train", "validation", "test"]
IMAGE_CACHE_DIR = "/vol/cache/vsr_images"


@app.function(image=image, gpu=GPU_TYPE, volumes={"/vol": volume}, timeout=TIMEOUT)
def cache_vsr_worker(worker_id: int, sample_range: tuple):
    """Cache PaliGemma2 activations for assigned VSR sample range."""
    import sys
    import gc
    import hashlib
    import io

    import torch
    import h5py
    import requests as req
    from pathlib import Path
    from PIL import Image
    from tqdm import tqdm
    from datasets import load_dataset, concatenate_datasets
    from nnsight import NNsight

    sys.path.insert(0, "/root/paligemma2")
    from utils import process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    activations_dir = Path(RESULTS_BASE) / "run" / "activations_vsr"
    activations_dir.mkdir(parents=True, exist_ok=True)
    img_cache = Path(IMAGE_CACHE_DIR)
    img_cache.mkdir(parents=True, exist_ok=True)

    start_idx, end_idx = sample_range
    print(f"[VSR Cache W{worker_id}] Samples {start_idx}->{end_idx}")

    # Load model from cache (model already downloaded to volume)
    print(f"[VSR Cache W{worker_id}] Loading PaliGemma2-3B from cache...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model_raw = model_raw.to("cuda")
    model_raw.eval()
    nns_model = NNsight(model_raw)

    # Load VSR dataset (fresh download to avoid cache hash mismatch)
    print(f"[VSR Cache W{worker_id}] Loading VSR dataset...")
    vsr_splits = []
    for split in VSR_SPLITS:
        vsr_splits.append(load_dataset(VSR_DATASET, split=split,
                                        cache_dir="/vol/cache/datasets_vsr",
                                        download_mode="reuse_cache_if_exists"))
    vsr = concatenate_datasets(vsr_splits)
    total_vsr = len(vsr)
    print(f"[VSR Cache W{worker_id}] VSR total: {total_vsr} samples")

    # Process samples in chunks
    actual_end = min(end_idx, total_vsr)
    n_chunks = (actual_end - start_idx + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        chunk_start = start_idx + chunk_idx * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, actual_end)
        h5_path = activations_dir / f"vsr_chunk_{chunk_start}_{chunk_end}.h5"

        if h5_path.exists():
            print(f"[VSR Cache W{worker_id}] SKIP chunk {chunk_start}->{chunk_end} — already cached")
            continue

        print(f"[VSR Cache W{worker_id}] Chunk {chunk_start}->{chunk_end}")
        skipped = 0

        for i in tqdm(range(chunk_start, chunk_end),
                      desc=f"W{worker_id} vsr {chunk_start}-{chunk_end}"):
            ex = vsr[i]
            url = ex.get("image_link", "")
            caption = str(ex.get("caption", "")).strip()
            label = int(ex.get("label", 0))
            relation = ex.get("relation", "")

            if not caption or not url:
                skipped += 1
                continue

            # Load/cache image
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_path = img_cache / f"{url_hash}.jpg"
            try:
                if cache_path.exists():
                    pil_img = Image.open(cache_path).convert("RGB")
                else:
                    resp = req.get(url, timeout=10)
                    resp.raise_for_status()
                    pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    pil_img.save(str(cache_path), "JPEG", quality=95)
            except Exception as e:
                skipped += 1
                continue

            # Process through model
            try:
                prompt = f"Is this statement true or false? {caption}"
                ids, mask, pv = process_vlm_inputs(
                    pil_img, prompt, processor, model_raw, device="cuda"
                )
                img_start, img_end = get_image_token_positions(ids)

                layer_outs = []
                with torch.no_grad():
                    with nns_model.trace(
                        input_ids=ids, attention_mask=mask, pixel_values=pv,
                    ) as tr:
                        for li in range(FROM_LAYER, TO_LAYER):
                            layer_outs.append(
                                nns_model.model.language_model.layers[li].output.save()
                            )

                with h5py.File(str(h5_path), "a") as f:
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
                        seq_len = int(mask[0].sum().item()) if mask.ndim == 2 else int(mask.sum().item())
                        act = out_tensor[0, :seq_len].float().contiguous().numpy()
                        ds = grp.create_dataset(
                            f"sample_{i}", data=act, compression=None
                        )
                        ds.attrs["img_start"] = int(img_start)
                        ds.attrs["img_end"] = int(img_end)
                        ds.attrs["label"] = label
                        ds.attrs["relation"] = relation
                        ds.attrs["caption"] = caption

                del ids, mask, pv, layer_outs
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [WARN] W{worker_id} sample {i}: {e}")
                skipped += 1
                continue

        volume.commit()
        print(f"[VSR Cache W{worker_id}] Done chunk {chunk_start}->{chunk_end} (skipped {skipped})")

    del nns_model, model_raw, processor
    torch.cuda.empty_cache()
    gc.collect()
    return f"VSR Cache W{worker_id}: done {start_idx}->{actual_end}"


@app.local_entrypoint()
def main():
    import math

    # VSR has ~10K samples total across train+dev+test
    # We'll estimate and distribute evenly
    total_vsr = 10972  # from prior runs
    samples_per_worker = math.ceil(total_vsr / N_CACHING_GPUS)

    assignments = []
    for w in range(N_CACHING_GPUS):
        start = w * samples_per_worker
        end = min(start + samples_per_worker, total_vsr)
        if start < total_vsr:
            assignments.append((w, (start, end)))

    print(f"Caching VSR activations: {total_vsr} samples across {len(assignments)} GPUs")
    for w, (s, e) in assignments:
        print(f"  GPU {w}: samples {s}->{e}")

    results = list(cache_vsr_worker.starmap(assignments))
    for r in results:
        print(r)

    print("\n[DONE] VSR activations cached!")
