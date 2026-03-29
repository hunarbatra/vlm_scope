"""
CUDA_VISIBLE_DEVICES=4 python vqa/cache_activations_with_text.py --from-layer 0 --to-layer 32 --start-sample 50000 --end-sample 54096 --caching-batch-size 16 --output-dir /scratch/local/ssd/lachin/activations/validation_50k --with-text

"""
# %%
import os
import dotenv
dotenv.load_dotenv(".env")
from pathlib import Path
from datasets import load_dataset
from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions, initialize_language_model, process_llm_inputs
import torch
from nnsight import NNsight
import gc
from tqdm import tqdm
import argparse
import h5py
import json

# %%
NUM_VIS_TOKENS = 575

# %%
def load_coco_captions():
    """Load COCO captions from local annotation file."""
    annotation_path = Path(__file__).parent.parent / "local_data" / "coco_annotations" / "annotations" / "captions_train2017.json"
    print(f"Loading COCO captions from: {annotation_path}")
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    # Build mapping from image_id to filename
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # Build mapping from filename to first caption
    caption_map = {}
    for ann in data['annotations']:
        filename = id_to_filename[ann['image_id']]
        if filename not in caption_map:
            caption_map[filename] = ann['caption']
    
    print(f"Loaded {len(caption_map)} COCO captions")
    return caption_map

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


def cache_chunk(vlm_tokenizer, vlm_model, vlm_image_processor, dataset, start_idx, end_idx, from_layer, to_layer, caching_batch_size, output_dir="temp", 
                with_text=False, llm_tokenizer=None, llm_model=None, coco_captions=None):

    vlm_model.to("cuda")
    if with_text and llm_model is not None:
        llm_model.to("cuda")

    temp_dir = Path(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate HDF5 files for efficient batched saving
    vlm_h5_path = temp_dir / f"vlm_chunk_{start_idx}_{end_idx}.h5"
    if with_text:
        text_h5_path = temp_dir / f"text_chunk_{start_idx}_{end_idx}.h5"
    
    # Sample info for reproducibility
    sample_info = {}
    
    for i in tqdm(range(start_idx, end_idx, caching_batch_size), desc="Caching chunk"):
        batch_input_ids = []
        batch_attention_mask = []
        batch_image_tensors = []
        
        # Text batch data
        text_batch_input_ids = []
        text_batch_attention_mask = []

        img_positions = []

        for j in range(i, min(i + caching_batch_size, end_idx)):
            image, prompt = dataset[j]
            
            # Get VQAv2 sample data for image_id
            vqa_sample = dataset.dataset[j]
            image_id = vqa_sample['image_id']
            
            # Store sample info
            sample_info[j] = {
                'index': j,
                'question': prompt,
                'image_id': image_id,
                'image_filename': f'{image_id:012d}.jpg'  # Standard COCO format
            }
            
            # Process VLM inputs
            input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(image, prompt, vlm_image_processor, vlm_model, vlm_tokenizer)
            
            # Store VLM tensors
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_image_tensors.append(image_tensor)

            img_start, img_end = get_image_token_positions(input_ids)
            img_positions.append((img_start - 1, img_end - 1))
            
            # Process text-only inputs if requested
            if with_text and llm_model is not None and coco_captions is not None:
                image_filename = sample_info[j]['image_filename']
                caption = coco_captions.get(image_filename, f"A photo containing various objects and scenes.")
                
                sample_info[j]['caption'] = caption
                
                text_prompt = f"Caption: {caption}\nInstruction: {prompt}"
                
                # Process text inputs
                text_input_ids, text_attention_mask = process_llm_inputs(text_prompt, llm_tokenizer, json_mode=False)
                text_batch_input_ids.append(text_input_ids)
                text_batch_attention_mask.append(text_attention_mask)
            
            del input_ids, attention_mask, image_tensor

        # Process VLM batch
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
        
        # Extract VLM activations
        vlm_layer_outputs = []
        
        with torch.no_grad():
            with vlm_model.trace(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                images=batch_image_tensors,
                image_sizes=image_sizes,
            ) as tr:
                for layer_idx in range(from_layer, to_layer):
                    vlm_layer_outputs.append(vlm_model.model.layers[layer_idx].output[0][:, 1:].detach().cpu().save())

            # Save VLM activations
            save_activations_to_h5(vlm_h5_path, vlm_layer_outputs, batch_attention_mask, img_positions, 
                                 from_layer, to_layer, i, "VLM")

        # Process text batch if requested (sequentially, no device moving)
        if with_text and llm_model is not None and text_batch_input_ids:
            # Pad text sequences
            text_batch_input_ids = torch.nn.utils.rnn.pad_sequence(
                [ids.squeeze(0) for ids in text_batch_input_ids],
                batch_first=True,
                padding_value=llm_tokenizer.pad_token_id,
            )
            
            text_batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
                [mask.squeeze(0) for mask in text_batch_attention_mask],
                batch_first=True,
                padding_value=0,
            )
            
            text_batch_input_ids = text_batch_input_ids.to(llm_model.device)
            text_batch_attention_mask = text_batch_attention_mask.to(llm_model.device)
            
            # Extract text activations
            text_layer_outputs = []
            
            with torch.no_grad():
                with llm_model.trace(
                    input_ids=text_batch_input_ids,
                    attention_mask=text_batch_attention_mask,
                ) as tr:
                    for layer_idx in range(from_layer, to_layer):
                        text_layer_outputs.append(llm_model.model.layers[layer_idx].output[0][:, 1:].detach().cpu().save())

                # Save text activations (no image positions for text)
                save_activations_to_h5(text_h5_path, text_layer_outputs, text_batch_attention_mask, 
                                     None, from_layer, to_layer, i, "Text")

        del batch_input_ids, batch_attention_mask, batch_image_tensors, vlm_layer_outputs
        if with_text and 'text_layer_outputs' in locals():
            del text_batch_input_ids, text_batch_attention_mask, text_layer_outputs
        torch.cuda.empty_cache()

    # Save sample info
    sample_info_path = temp_dir / f"sample_info_{start_idx}_{end_idx}.json"
    with open(sample_info_path, 'w') as f:
        json.dump(sample_info, f, indent=2)

    vlm_model.to("cpu")
    if with_text and llm_model is not None:
        llm_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()


def save_activations_to_h5(h5_path, layer_outputs, attention_mask, img_positions, 
                          from_layer, to_layer, batch_start_idx, data_type):
    """Save activations to HDF5 file."""
    pbar = tqdm(total=(to_layer - from_layer) * layer_outputs[0].shape[0], desc=f"Saving {data_type} activations")
    
    all_layer_data = {}
    all_layer_lengths = {}
    
    for layer_idx in range(from_layer, to_layer):
        layer_output = layer_outputs[layer_idx - from_layer]
        layer_activations = []
        layer_lengths = []
        
        for sample_idx in range(layer_output.shape[0]):
            if data_type == "VLM":
                actual_seq_len = attention_mask[sample_idx, 1:].sum().item() + NUM_VIS_TOKENS
            else:  # Text
                actual_seq_len = attention_mask[sample_idx, 1:].sum().item()
            
            activation_data = layer_output[sample_idx, :actual_seq_len].contiguous().numpy()
            layer_activations.append(activation_data)
            layer_lengths.append(actual_seq_len)
        
        all_layer_data[layer_idx] = layer_activations
        all_layer_lengths[layer_idx] = layer_lengths
    
    # Use append mode if file exists, write mode if new
    mode = 'a' if h5_path.exists() else 'w'
    
    with h5py.File(h5_path, mode) as f:
        for layer_idx in range(from_layer, to_layer):
            layer_group = f.require_group(f'layer_{layer_idx}')
            layer_activations = all_layer_data[layer_idx]
            layer_lengths = all_layer_lengths[layer_idx]
            
            for sample_idx, (activation_data, seq_len) in enumerate(zip(layer_activations, layer_lengths)):
                sample_key = f'sample_{batch_start_idx + sample_idx}'
                
                # # Skip if dataset already exists
                # if sample_key in layer_group:
                #     continue
                    
                ds = layer_group.create_dataset(sample_key, data=activation_data, 
                                         compression=None,  # No compression for speed
                                         shuffle=False)     # No shuffle for speed

                # Attach image token span attributes (only for VLM)
                if data_type == "VLM" and img_positions is not None:
                    img_start_adj, img_end_adj = img_positions[sample_idx]
                    ds.attrs['img_start'] = int(img_start_adj)
                    ds.attrs['img_end'] = int(img_end_adj)
                
                pbar.update(1)
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Cache VLM activations")
    parser.add_argument("--from-layer", type=int, default=24, help="First layer index (inclusive)")
    parser.add_argument("--to-layer", type=int, default=25, help="Last layer index (exclusive)")
    parser.add_argument("--start-sample", type=int, required=True, help="Starting sample index")
    parser.add_argument("--end-sample", type=int, required=True, help="Ending sample index")
    parser.add_argument("--caching-batch-size", type=int, default=2, help="Batch size for caching activations")
    parser.add_argument("--output-dir", type=str, default="temp", help="Output directory for cached activations")
    
    # New arguments for text processing
    parser.add_argument("--with-text", action="store_true", help="Also extract text-only activations")
    parser.add_argument("--language-model", type=str, default="llama-3.1-8b-it", help="Language model to use for text processing")

    args = parser.parse_args()

    print(f"[INFO] Caching activations for samples {args.start_sample} → {args.end_sample}")
    print(f"[INFO] Layers {args.from_layer} → {args.to_layer}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] With text processing: {args.with_text}")

    # Initialize VLM model and dataset
    vlm_tokenizer, vlm_model, vlm_image_processor = initialize_vlm_model("llava-more", device="cpu")
    vlm_model = NNsight(vlm_model)
    dataset = DatasetWrapper()

    # Initialize text processing if requested
    llm_tokenizer = None
    llm_model = None
    coco_captions = None
    
    if args.with_text:
        print(f"[INFO] Initializing language model: {args.language_model}")
        llm_tokenizer, llm_model = initialize_language_model(args.language_model, device="cpu")
        llm_model = NNsight(llm_model)
        
        # Set pad token if not set
        if llm_tokenizer.pad_token_id is None:
            llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
        
        print("[INFO] Loading COCO captions for text processing")
        try:
            coco_captions = load_coco_captions()
        except Exception as e:
            print(f"[WARNING] Could not load COCO captions: {e}")
            print("[INFO] Will proceed with placeholder captions")
            coco_captions = {}

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
        args.output_dir,
        with_text=args.with_text,
        llm_tokenizer=llm_tokenizer,
        llm_model=llm_model,
        coco_captions=coco_captions
    )

    print(f"[INFO] Successfully cached VLM activations to {args.output_dir}/vlm_chunk_{args.start_sample}_{args.end_sample}.h5")
    if args.with_text:
        print(f"[INFO] Successfully cached text activations to {args.output_dir}/text_chunk_{args.start_sample}_{args.end_sample}.h5")
    print(f"[INFO] Sample info saved to {args.output_dir}/sample_info_{args.start_sample}_{args.end_sample}.json")


if __name__ == "__main__":
    main() 