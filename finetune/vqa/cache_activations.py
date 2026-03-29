# %%
"""
CUDA_VISIBLE_DEVICES=3 
python vqa/cache_activations.py --from-layer 0 --to-layer 32 --start-sample 50000 --end-sample 51024 --caching-batch-size 16 --output-dir /scratch/local/ssd/lachin/activations/validation_50k
"""
# %%
import dotenv
dotenv.load_dotenv(".env")

import os
from pathlib import Path
from datasets import load_dataset
from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions
import torch
from nnsight import NNsight
import gc
from tqdm import tqdm
import argparse
import h5py

# %%
NUM_VIS_TOKENS = 575

# %%
class DatasetWrapper:

    def __init__(self):
        self.dataset = load_dataset("lmms-lab/VQAv2", split="validation")

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        prompt = sample["question"]
        return image, prompt

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)


def cache_chunk(vlm_tokenizer, vlm_model, vlm_image_processor, dataset, start_idx, end_idx, from_layer, to_layer, caching_batch_size, output_dir="temp"):

    vlm_model.to("cuda")

    temp_dir = Path(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate HDF5 file for efficient batched saving
    h5_path = temp_dir / f"chunk_{start_idx}_{end_idx}.h5"
    
    for i in tqdm(range(start_idx, end_idx, caching_batch_size), desc="Caching chunk"):
        batch_input_ids = []
        batch_attention_mask = []
        batch_image_tensors = []

        img_positions = []

        for j in range(i, min(i + caching_batch_size, end_idx)):
            image, prompt = dataset[j]
            input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(image, prompt, vlm_image_processor, vlm_model, vlm_tokenizer)
            # Store tensors
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_image_tensors.append(image_tensor)

            img_start, img_end = get_image_token_positions(input_ids)
            img_positions.append((img_start - 1, img_end - 1))
            del input_ids, attention_mask, image_tensor

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            [ids.squeeze(0) for ids in batch_input_ids],  # remove leading dim (1, seq_len)
            batch_first=True,
            padding_value=vlm_tokenizer.pad_token_id,
        )

        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [mask.squeeze(0) for mask in batch_attention_mask],
            batch_first=True,
            padding_value=0,
        )

        batch_image_tensors = torch.cat(batch_image_tensors, dim=0)
        batch_input_ids = batch_input_ids.to(vlm_model.device)
        batch_attention_mask = batch_attention_mask.to(vlm_model.device)
        batch_image_tensors = batch_image_tensors.to(vlm_model.device)
        
        layer_outputs = []
        
        with torch.no_grad():
            with vlm_model.trace(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                images=batch_image_tensors,
                image_sizes=image_sizes,
            ) as tr:
                for layer_idx in range(from_layer, to_layer):
                    layer_outputs.append(vlm_model.model.layers[layer_idx].output[0][:, 1:].detach().cpu().save())

            # Ultra-fast batch save using memory-mapped arrays
            pbar = tqdm(total=(to_layer - from_layer) * layer_outputs[0].shape[0], desc="Saving activations")
            
            all_layer_data = {}
            all_layer_lengths = {}
            
            for layer_idx in range(from_layer, to_layer):
                layer_output = layer_outputs[layer_idx - from_layer]
                layer_activations = []
                layer_lengths = []
                
                for sample_idx in range(layer_output.shape[0]):
                    actual_seq_len = batch_attention_mask[sample_idx, 1:].sum().item() + NUM_VIS_TOKENS
                    activation_data = layer_output[sample_idx, :actual_seq_len].contiguous().numpy()
                    layer_activations.append(activation_data)
                    layer_lengths.append(actual_seq_len)
                
                all_layer_data[layer_idx] = layer_activations
                all_layer_lengths[layer_idx] = layer_lengths
            
            with h5py.File(h5_path, 'a') as f:
                for layer_idx in range(from_layer, to_layer):
                    layer_group = f.require_group(f'layer_{layer_idx}')
                    layer_activations = all_layer_data[layer_idx]
                    layer_lengths = all_layer_lengths[layer_idx]
                    
                    for sample_idx, (activation_data, seq_len) in enumerate(zip(layer_activations, layer_lengths)):
                        sample_key = f'sample_{i + sample_idx}'
                        ds = layer_group.create_dataset(sample_key, data=activation_data, 
                                                 compression=None,  # No compression for speed
                                                 shuffle=False)     # No shuffle for speed

                        # Attach image token span attributes
                        img_start_adj, img_end_adj = img_positions[sample_idx]
                        ds.attrs['img_start'] = int(img_start_adj)
                        ds.attrs['img_end'] = int(img_end_adj)
                        pbar.update(1)
            pbar.close()

        del batch_input_ids, batch_attention_mask, batch_image_tensors, layer_outputs, all_layer_data, all_layer_lengths
        del layer_output, layer_activations, layer_lengths, activation_data, pbar
        torch.cuda.empty_cache()

    vlm_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Cache VLM activations")
    parser.add_argument("--from-layer", type=int, default=24, help="First layer index (inclusive)")
    parser.add_argument("--to-layer", type=int, default=25, help="Last layer index (exclusive)")
    parser.add_argument("--start-sample", type=int, required=True, help="Starting sample index")
    parser.add_argument("--end-sample", type=int, required=True, help="Ending sample index")
    parser.add_argument("--caching-batch-size", type=int, default=8, help="Batch size for caching activations")
    parser.add_argument("--output-dir", type=str, default="temp", help="Output directory for cached activations")

    args = parser.parse_args()

    print(f"[INFO] Caching activations for samples {args.start_sample} → {args.end_sample}")
    print(f"[INFO] Layers {args.from_layer} → {args.to_layer}")
    print(f"[INFO] Output directory: {args.output_dir}")

    # Initialize model and dataset
    vlm_tokenizer, vlm_model, vlm_image_processor = initialize_vlm_model("llava-more", device="cpu")
    vlm_model = NNsight(vlm_model)
    dataset = DatasetWrapper()

    # Cache the chunk
    cache_chunk(
        vlm_tokenizer, 
        vlm_model, 
        vlm_image_processor, 
        dataset, 
        args.start_sample, 
        args.end_sample, 
        args.from_layer, 
        args.to_layer, 
        args.caching_batch_size, 
        args.output_dir
    )

    print(f"[INFO] Successfully cached activations to {args.output_dir}/chunk_{args.start_sample}_{args.end_sample}.h5")


if __name__ == "__main__":
    main() 