"""
Cache PaliGemma2-3B intermediate activations on VQAv2 for SAE training.

Mirrors finetune/vqa/cache_activations.py but adapted for PaliGemma2.

Usage:
    python cache_activations.py \
        --from-layer 0 --to-layer 26 \
        --start-sample 0 --end-sample 1000 \
        --caching-batch-size 8 \
        --output-dir /vol/results/paligemma2/run/temp
"""

import argparse
import gc
from pathlib import Path

import h5py
import torch
from datasets import load_dataset
from nnsight import NNsight
from tqdm import tqdm

from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions

NUM_LAYERS = 26


class VQAv2Dataset:
    """Thin wrapper around the VQAv2 validation split."""

    def __init__(self):
        self.dataset = load_dataset("lmms-lab/VQAv2", split="validation")

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        prompt = sample["question"]
        return image, prompt

    def __len__(self):
        return len(self.dataset)


def cache_chunk(
    processor,
    model,
    dataset,
    start_idx,
    end_idx,
    from_layer,
    to_layer,
    batch_size,
    output_dir="temp",
):
    """Cache residual-stream activations for a chunk of VQAv2 samples."""

    device = next(model._module.parameters()).device
    temp_dir = Path(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    h5_path = temp_dir / f"chunk_{start_idx}_{end_idx}.h5"

    for i in tqdm(range(start_idx, end_idx, batch_size), desc="Caching"):
        actual_end = min(i + batch_size, end_idx)

        # Collect inputs one-by-one (variable image sizes)
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        img_positions = []

        for j in range(i, actual_end):
            image, prompt = dataset[j]
            input_ids, attention_mask, pixel_values = process_vlm_inputs(
                image, prompt, processor, model._module, device=device
            )
            all_input_ids.append(input_ids.squeeze(0))
            all_attention_masks.append(attention_mask.squeeze(0))
            all_pixel_values.append(pixel_values)

            img_start, img_end = get_image_token_positions(input_ids)
            img_positions.append((img_start, img_end))

        # Pad to same length
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id or 0
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            all_attention_masks, batch_first=True, padding_value=0
        )
        pixel_values_batched = torch.cat(all_pixel_values, dim=0)

        input_ids_padded = input_ids_padded.to(device)
        attention_mask_padded = attention_mask_padded.to(device)
        pixel_values_batched = pixel_values_batched.to(device)

        # Extract layer activations via NNsight
        layer_outputs = []
        with torch.no_grad():
            with model.trace(
                input_ids=input_ids_padded,
                attention_mask=attention_mask_padded,
                pixel_values=pixel_values_batched,
            ) as tr:
                for layer_idx in range(from_layer, to_layer):
                    # PaliGemma2 decoder layers: model.model.language_model.layers[i]
                    out = model.model.language_model.layers[layer_idx].output[0].detach().cpu().save()
                    layer_outputs.append(out)

        # Save to HDF5
        with h5py.File(h5_path, "a") as f:
            for li, layer_idx in enumerate(range(from_layer, to_layer)):
                layer_group = f.require_group(f"layer_{layer_idx}")
                layer_out = layer_outputs[li]

                actual_batch = actual_end - i
                for si in range(actual_batch):
                    # Use attention mask to determine actual sequence length
                    seq_len = int(attention_mask_padded[si].sum().item())
                    activation = layer_out[si, :seq_len].contiguous().numpy()

                    sample_key = f"sample_{i + si}"
                    ds = layer_group.create_dataset(sample_key, data=activation, compression=None)

                    img_start, img_end = img_positions[si]
                    ds.attrs["img_start"] = int(img_start)
                    ds.attrs["img_end"] = int(img_end)

        del input_ids_padded, attention_mask_padded, pixel_values_batched, layer_outputs
        del all_input_ids, all_attention_masks, all_pixel_values
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Cache PaliGemma2 activations")
    parser.add_argument("--from-layer", type=int, default=0)
    parser.add_argument("--to-layer", type=int, default=26)
    parser.add_argument("--start-sample", type=int, required=True)
    parser.add_argument("--end-sample", type=int, required=True)
    parser.add_argument("--caching-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="temp")
    parser.add_argument("--model-name", type=str, default="google/paligemma2-3b-pt-224")
    args = parser.parse_args()

    print(f"[INFO] Caching samples {args.start_sample} -> {args.end_sample}")
    print(f"[INFO] Layers {args.from_layer} -> {args.to_layer}")

    processor, model_raw = initialize_vlm_model(args.model_name, device="cpu")
    model = NNsight(model_raw)
    model._module.to("cuda")

    dataset = VQAv2Dataset()

    cache_chunk(
        processor,
        model,
        dataset,
        args.start_sample,
        args.end_sample,
        args.from_layer,
        args.to_layer,
        args.caching_batch_size,
        args.output_dir,
    )

    print(f"[INFO] Done -> {args.output_dir}/chunk_{args.start_sample}_{args.end_sample}.h5")


if __name__ == "__main__":
    main()
