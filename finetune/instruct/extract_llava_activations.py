#!/usr/bin/env python3
"""
VLM Activation Extraction

This script extracts activations from LLaVA-MORE vision-language model
with error handling, retry logic, and configurable parameters.

Usage:
    python finetune/extract_llava_activations.py \
  --max_samples 2048 \
  --batch_size 4 \
  --save_dir /scratch/local/ssd/lachin/activations \
  --save_sample_order \
  --device cuda:3 \
  --with_llama \
  --save_llama_dataset
"""

import os
import sys
import argparse
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import pickle
import glob
import random
import logging
import time
import json
from typing import List, Tuple, Optional

# Setup paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
LLAVA_MORE_PATH = os.getenv('LLAVA_MORE_PATH', ROOT_DIR / 'LLaVA-MORE')
sys.path.append(str(ROOT_DIR))
sys.path.insert(0, str(LLAVA_MORE_PATH))
os.environ['TOKENIZER_PATH'] = 'aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning'

# Clean llava modules
if 'llava' in sys.modules:
    for key in list(sys.modules.keys()):
        if key.startswith('llava'):
            del sys.modules[key]

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nnsight import NNsight

# Add LLaMA imports
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.utils import (
    initialize_vlm_model,
    process_vlm_inputs,
    get_image_token_positions,
    get_text_token_positions,
    apply_vlm_template,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
COCO_URL = "http://images.cocodataset.org/train2017/"
MAX_RETRIES = 3
RETRY_DELAY = 1.0


def left_pad_sequences(sequences, pad_token_id):
    """Left-pad sequences to the same length."""
    max_len = max(seq.size(-1) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padded_seq = torch.cat([torch.full((pad_len,), pad_token_id, dtype=seq.dtype, device=seq.device), seq], dim=0)
        padded.append(padded_seq)
    return torch.stack(padded, dim=0)

def left_pad_masks(masks):
    """Left-pad attention masks to the same length."""
    max_len = max(mask.size(-1) for mask in masks)
    padded = []
    for mask in masks:
        pad_len = max_len - mask.size(-1)
        padded_mask = torch.cat([torch.zeros(pad_len, dtype=torch.bool, device=mask.device), mask], dim=0)
        padded.append(padded_mask)
    return torch.stack(padded, dim=0)

def download_image_with_retry(image_url: str, max_retries: int = MAX_RETRIES) -> Optional[Image.Image]:
    """Download image with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {image_url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            continue
    
    logger.error(f"Failed to download image after {max_retries} attempts: {image_url}")
    return None

def clean_existing_files(save_dir: str):
    """Clean existing activation files."""
    logger.info("Cleaning existing activation files...")
    for npy_file in glob.glob(os.path.join(save_dir, "all_activations_sample_*.npy")):
        try:
            os.remove(npy_file)
            logger.debug(f"Removed {npy_file}")
        except Exception as e:
            logger.warning(f"Could not delete {npy_file}: {e}")

def load_and_shuffle_dataset(max_samples: int, seed: int = 42, no_shuffle: bool = False) -> List:
    """Load and shuffle the LLaVA-Instruct dataset efficiently."""
    logger.info(f"Loading LLaVA-Instruct dataset (max {max_samples} samples)...")
    
    # Load dataset
    ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True)
    
    if no_shuffle:
        logger.info("Skipping shuffle for faster processing")
    else:
        # Efficient streaming shuffle (RECOMMENDED for large runs)
        # Uses a reasonable buffer size for good randomness without excessive memory
        ds = ds.shuffle(seed=seed, buffer_size=10000)  # 10K buffer = good balance
        logger.info("Using shuffled dataset for diverse sampling")
    
    # Alternative options (uncomment to use):
    
    # Option 2: No shuffle (fastest, but less diverse)
    # ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True)
    
    # Option 3: Random skip + take (memory efficient, good diversity)  
    # import random
    # random.seed(seed)
    # skip_samples = random.randint(0, 150000 - max_samples)
    # ds = ds.skip(skip_samples).take(max_samples)
    
    # Option 4: Multiple random chunks (best diversity, more complex)
    # chunk_size = max_samples // 5  # Get 5 random chunks
    # buffer = []
    # for _ in range(5):
    #     skip = random.randint(0, 150000 - chunk_size)
    #     chunk = ds.skip(skip).take(chunk_size)
    #     buffer.extend(list(chunk))
    # return buffer
    
    # Take only what we need
    buffer = []
    for i, sample in enumerate(ds):
        buffer.append(sample)
        if len(buffer) >= max_samples:
            break
    
    logger.info(f"Loaded {len(buffer)} shuffled samples")
    return buffer

def process_sample(sample: dict, image_processor, model, tokenizer) -> Optional[Tuple]:
    """Process a single sample into model inputs."""
    try:
        image_filename = sample["image"]
        image_url = COCO_URL + image_filename
        
        # Download image with retry
        image = download_image_with_retry(image_url)
        if image is None:
            return None
        
        # Process conversations
        conv = sample["conversations"]
        if len(conv) < 2:
            logger.warning(f"Sample has insufficient conversations: {len(conv)}")
            return None
        
        human_message = conv[0]["value"].replace("<image>", "").strip()
        assistant_message = conv[1]["value"]
        
        # Process VLM inputs
        input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(
            image, human_message, image_processor, model, tokenizer
        )
        
        # Add assistant response
        assistant_tokens = tokenizer.encode(assistant_message, add_special_tokens=False, return_tensors="pt").to(input_ids.device)
        input_ids = torch.cat([input_ids, assistant_tokens], dim=1)
        
        assistant_mask = torch.ones_like(assistant_tokens, dtype=torch.bool)
        attention_mask = torch.cat([attention_mask, assistant_mask], dim=1)
        
        # Ensure proper image tensor shape
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        
        # Construct image sizes from tensor
        image_sizes_tensor = torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]], 
                                        dtype=torch.long, device=image_tensor.device)
        
        return (input_ids.squeeze(0), attention_mask.squeeze(0), image_tensor, image_sizes_tensor, human_message)
        
    except Exception as e:
        logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
        return None

def extract_activations_batch(input_ids, attention_mask, image_tensor, image_sizes, 
                            model, tokenizer, batch_input_ids: List) -> List:
    """Extract activations for a batch."""
    start_idx = 1
    batch_activations = []
    num_layers = len(model.model.layers)
    
    with torch.no_grad():
        with model.trace(input_ids, attention_mask=attention_mask, 
                        images=image_tensor, image_sizes=image_sizes, use_cache=False):
            layer_activations = []
            for l_idx in range(num_layers):
                saved_activation = model.model.layers[l_idx].output[0].save()
                layer_activations.append(saved_activation)
    
    # Process activations
    for l_idx, saved_activation in enumerate(layer_activations):
        activations = saved_activation.value.detach().cpu().numpy()
        
        for b in range(activations.shape[0]):
            # Handle padding correctly
            max_len = input_ids.size(1)
            orig_len = batch_input_ids[b].size(0)
            pad_len = max_len - orig_len
            
            # Extract activations for actual sequence (skip first token)
            act = activations[b, pad_len + start_idx : pad_len + orig_len]
            
            if len(batch_activations) <= b:
                batch_activations.append([])
            batch_activations[b].append(act)
        
        # Clean up
        del activations
        torch.cuda.empty_cache()
    
    return batch_activations

def load_coco_captions():
    annotation_path = ROOT_DIR / "local_data" / "coco_annotations" / "annotations" / "captions_train2017.json"
    logger.info(f"Loading COCO captions from local file: {annotation_path}")
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    caption_map = {}
    for ann in data['annotations']:
        filename = id_to_filename[ann['image_id']]
        if filename not in caption_map:
            caption_map[filename] = ann['caption']
    logger.info(f"Loaded {len(caption_map)} COCO captions from local file.")
    return caption_map

def main():
    parser = argparse.ArgumentParser(description="Extract VLM activations with improved reliability")
    parser.add_argument("--max_samples", type=int, default=15000, help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="local_data/llava_activations", help="Directory to save activations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_sample_order", action="store_true", help="Save sample order for reproducible text-only dataset")
    parser.add_argument("--no_shuffle", action="store_true", help="Skip shuffling for faster processing (use for small test runs)")

    parser.add_argument("--start_index", type=int, default=0,
                        help="Global index of the first sample in this chunk "
                             "(so file names stay unique across chunks)")
    parser.add_argument("--no_combine", action="store_true",
                        help="Skip combining individual .npy files at the end")
    parser.add_argument("--device", type=str, default="cuda",
                        help="GPU device to use (e.g., cuda:0, cuda:1, cuda:2)")
    # Add llama device option (optional, default to same as VLM)
    parser.add_argument("--llama_save_dir", type=str, default=None, help="Directory to save LLaMA activations (default: save_dir/llama_activations)")
    parser.add_argument("--llama_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="LLaMA model name from HuggingFace")
    parser.add_argument("--with_llama", action="store_true", help="Also extract LLaMA activations using text-only prompts.")
    parser.add_argument("--save_llama_dataset", action="store_true", help="Save the LLaMA dataset as JSONL file.")

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting VLM activation extraction with config:")
    logger.info(f"  Max samples: {args.max_samples}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Save directory: {args.save_dir}")
    logger.info(f"  Save sample order: {args.save_sample_order}")
    logger.info(f"  Device: {args.device}")

    # Initialize VLM model
    logger.info("Initializing VLM model...")
    if args.device != "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
        logger.info(f"Set CUDA_VISIBLE_DEVICES to GPU {args.device.split(':')[-1]}")
    tokenizer, model, image_processor = initialize_vlm_model()
    model = NNsight(model)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id >= tokenizer.vocab_size or tokenizer.pad_token_id < 0:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize LLaMA model/tokenizer only if requested
    if args.with_llama:
        llama_save_dir = args.llama_save_dir if args.llama_save_dir is not None else os.path.join(args.save_dir, "llama_activations")
        os.makedirs(llama_save_dir, exist_ok=True)
        logger.info(f"Initializing LLaMA model on device {args.device}...")
        llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_name)
        llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        llama_model = NNsight(llama_model)
        llama_model.eval()
        if llama_tokenizer.pad_token_id is None or llama_tokenizer.pad_token_id >= llama_tokenizer.vocab_size or llama_tokenizer.pad_token_id < 0:
            llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
        llama_num_layers = len(llama_model.model.layers)
        logger.info(f"Model has {len(model.model.layers)} VLM layers and {llama_num_layers} LLaMA layers")
    else:
        llama_save_dir = None
        llama_tokenizer = None
        llama_model = None
        llama_num_layers = None

    # Setup save directory for VLM (LLaVA) activations as a subfolder
    llava_save_dir = os.path.join(args.save_dir, "llava_activations")
    os.makedirs(llava_save_dir, exist_ok=True)
    clean_existing_files(llava_save_dir)

    # Setup save directory for LLaMA activations as a subfolder
    llama_save_dir = os.path.join(args.save_dir, "llama_activations")
    if args.with_llama:
        os.makedirs(llama_save_dir, exist_ok=True)

    # Load dataset
    buffer = load_and_shuffle_dataset(args.max_samples, args.seed, args.no_shuffle)

    # Load COCO captions for LLaMA prompt construction
    coco_captions = load_coco_captions()

    # Save sample order for reproducibility if requested
    if args.save_sample_order:
        sample_order = []
        for i, sample in enumerate(buffer):
            sample_order.append({
                'index': i,
                'id': sample.get('id', f'sample_{i}'),
                'image': sample.get('image', ''),
                'conversations': sample.get('conversations', [])
            })
        order_file = os.path.join(llava_save_dir, f"vlm_sample_order_{args.max_samples}_seed{args.seed}.json")
        with open(order_file, 'w') as f:
            json.dump(sample_order, f, indent=2)
        logger.info(f"Saved sample order to: {order_file}")
    
    # Also save sample info in a format that's easier to load with activations
    sample_info = {}
    for i, sample in enumerate(buffer):
        sample_info[i] = {
            'id': sample.get('id', f'sample_{i}'),
            'image': sample.get('image', ''),
            'conversations': sample.get('conversations', [])
        }
    sample_info_file = os.path.join(llava_save_dir, f"sample_info_{args.max_samples}_seed{args.seed}.json")
    with open(sample_info_file, 'w') as f:
        json.dump(sample_info, f, indent=2)
    logger.info(f"Saved sample info to: {sample_info_file}")

    processed_count = 0
    batch_inputs = []
    llama_batch_prompts = []
    llama_batch_samples = []
    llama_dataset_samples = []  # For saving LLaMA dataset
    torch.cuda.empty_cache()

    for i, sample in enumerate(tqdm(buffer, desc="Processing samples")):
        if processed_count >= args.max_samples:
            break

        # Process VLM sample
        result = process_sample(sample, image_processor, model, tokenizer)
        if result is None:
            continue
        input_ids, attention_mask, image_tensor, image_sizes, message = result
        batch_inputs.append((input_ids, attention_mask, image_tensor, image_sizes, message))

        # Prepare LLaMA prompt for this sample only if requested
        if args.with_llama:
            image_filename = sample.get('image', '')
            caption = sample.get('caption', None)
            conversations = sample.get('conversations', [])
            # Try to get COCO caption if missing
            if not caption and image_filename:
                caption = coco_captions.get(image_filename)
            if not caption:
                caption = f"[NO CAPTION for {image_filename}]"
            # print(f"DEBUG: Sample {i}: caption={caption}, conversations={conversations}")
            if caption is not None and len(conversations) > 0:
                # Use the first human message as instruction
                instruction = conversations[0]['value'].replace('<image>', '').strip()
                llama_prompt = f"Caption: {caption}\nInstruction: {instruction}"
                llama_batch_prompts.append(llama_prompt)
                llama_batch_samples.append(sample)
                
                # Save LLaMA dataset sample if requested
                if args.save_llama_dataset:
                    # Find the assistant response (second message in conversation)
                    assistant_response = ""
                    if len(conversations) > 1:
                        assistant_response = conversations[1]['value']
                    
                    llama_sample = {
                        "sample_id": sample.get('id', f'sample_{i}'),
                        "original_image": image_filename,
                        "prompt": llama_prompt,
                        "instruction": instruction,
                        "caption": caption,
                        "response": assistant_response,
                        "conversations": conversations
                    }
                    llama_dataset_samples.append(llama_sample)
            else:
                llama_batch_prompts.append(None)
                llama_batch_samples.append(None)

        # Process batch when full or at end
        if len(batch_inputs) == args.batch_size or processed_count + len(batch_inputs) == len(buffer):
            logger.info(f"\nProcessing batch of {len(batch_inputs)} samples...")
            try:
                # VLM batch processing (existing code)
                batch_input_ids = [x[0] for x in batch_inputs]
                batch_attention_masks = [x[1] for x in batch_inputs]
                batch_image_tensors = [x[2] for x in batch_inputs]
                batch_image_sizes = [x[3] for x in batch_inputs]
                batch_messages = [x[4] for x in batch_inputs]
                sequence_lengths = [seq.size(0) for seq in batch_input_ids]
                logger.info(f"Sequence lengths: {sequence_lengths}")
                logger.info(f"Max length: {max(sequence_lengths)}")
                input_ids_padded = left_pad_sequences(batch_input_ids, tokenizer.pad_token_id)
                attention_mask_padded = left_pad_masks(batch_attention_masks)
                image_tensor_batch = torch.stack(batch_image_tensors)
                image_sizes_batch = torch.stack(batch_image_sizes)
                logger.info(f"Batch shape: {input_ids_padded.shape}")
                batch_activations = extract_activations_batch(
                    input_ids_padded, attention_mask_padded, image_tensor_batch, 
                    image_sizes_batch, model, tokenizer, batch_input_ids
                )
                for j, sample_acts in enumerate(batch_activations):
                    sample_idx = args.start_index + processed_count + j
                    output_path = os.path.join(llava_save_dir, f"all_activations_sample_{sample_idx}.npy")
                    np.save(output_path, sample_acts)
                    
                    # Save input IDs for accurate token position detection (exclude BOS token to match activations)
                    input_ids_no_bos = batch_input_ids[j][1:]  # Skip BOS token to match activations
                    input_ids_path = os.path.join(llava_save_dir, f"input_ids_sample_{sample_idx}.npy")
                    np.save(input_ids_path, input_ids_no_bos.cpu().numpy())
                    logger.debug(f"Saved activations and input IDs for sample {sample_idx}: {len(sample_acts)} layers")
                # LLaMA batch processing (new code)
                if args.with_llama:
                    # print(f"DEBUG: Entering LLaMA batch processing for batch {processed_count} to {processed_count + len(batch_inputs)}")
                    llama_valid_indices = [k for k, p in enumerate(llama_batch_prompts) if p is not None]
                    # print(f"DEBUG: LLaMA valid indices: {llama_valid_indices}")
                    if llama_valid_indices:
                        llama_prompts = [llama_batch_prompts[k] for k in llama_valid_indices]
                        llama_samples = [llama_batch_samples[k] for k in llama_valid_indices]
                        llama_inputs = llama_tokenizer(llama_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
                        # Get the device of the LLaMA model
                        model_device = next(llama_model.parameters()).device
                        llama_input_ids = llama_inputs.input_ids.to(model_device)
                        llama_attention_mask = llama_inputs.attention_mask.to(model_device)
                        try:
                            with torch.no_grad():
                                with llama_model.trace(llama_input_ids, attention_mask=llama_attention_mask, use_cache=False):
                                    llama_layer_activations = []
                                    for l_idx in range(llama_num_layers):
                                        saved_activation = llama_model.model.layers[l_idx].output[0].save()
                                        llama_layer_activations.append(saved_activation)
                            # Process and save LLaMA activations for each sample
                            for b in range(llama_input_ids.size(0)):
                                sample_idx = args.start_index + processed_count + llama_valid_indices[b]
                                sample_acts = []
                                for l_idx in range(llama_num_layers):
                                    activations = llama_layer_activations[l_idx].value.detach().cpu().numpy()
                                    orig_len = (llama_attention_mask[b] > 0).sum().item()
                                    # Skip BOS token (first token)
                                    act = activations[b, 1:orig_len]
                                    sample_acts.append(act)
                                llama_output_path = os.path.join(llama_save_dir, f"llama_activations_sample_{sample_idx}.npy")
                                print(f"DEBUG: About to save LLaMA activation {llama_output_path}")
                                np.save(llama_output_path, sample_acts)
                                # Optionally save input_ids for LLaMA
                                llama_input_ids_path = os.path.join(llama_save_dir, f"llama_input_ids_sample_{sample_idx}.npy")
                                np.save(llama_input_ids_path, llama_input_ids[b, 1:orig_len].cpu().numpy())
                                print(f"DEBUG: Saved LLaMA activation {llama_output_path}")
                        except Exception as e:
                            logger.error(f"Error during LLaMA batch processing: {e}")
                            import traceback; traceback.print_exc()
                            torch.cuda.empty_cache()
                processed_count += len(batch_inputs)
                logger.info(f"Successfully processed batch, total: {processed_count}/{args.max_samples}")
            except Exception as e:
                logger.error(f"Error during batch processing: {e}")
                torch.cuda.empty_cache()
            batch_inputs = []
            llama_batch_prompts = []
            llama_batch_samples = []
    if args.no_combine:
        logger.info("Skipping combine step (--no_combine)")
        return
    # Combine VLM activations (existing code)
    logger.info("Combining saved activation files...")
    all_activations = []
    for i in range(processed_count):
        file_path = os.path.join(llava_save_dir, f"all_activations_sample_{args.start_index + i}.npy")
        if os.path.exists(file_path):
            all_activations.append(np.load(file_path, allow_pickle=True))
    if all_activations:
        output_file = os.path.join(llava_save_dir, f"dataset_all_activations_llava_{processed_count}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(all_activations, f)
        logger.info(f"LLaVA activation dataset saved: {output_file}")
        # Clean up individual activation files
        for i in range(processed_count):
            file_path = os.path.join(llava_save_dir, f"all_activations_sample_{args.start_index + i}.npy")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
        # Combine all input IDs into final file
        logger.info("Combining saved input IDs files...")
        all_input_ids = []
        for i in range(processed_count):
            file_path = os.path.join(llava_save_dir, f"input_ids_sample_{args.start_index + i}.npy")
            if os.path.exists(file_path):
                all_input_ids.append(np.load(file_path, allow_pickle=True))
        input_ids_file = os.path.join(llava_save_dir, f"input_ids_llava_{processed_count}.pkl")
        with open(input_ids_file, 'wb') as f:
            pickle.dump(all_input_ids, f)
        logger.info(f"VLM input IDs dataset saved: {input_ids_file}")
        for i in range(processed_count):
            file_path = os.path.join(llava_save_dir, f"input_ids_sample_{args.start_index + i}.npy")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
    else:
        logger.info("No VLM (LLaVA) activations to combine or delete.")
    
    # Combine LLaMA activations (new code)
    if args.with_llama:
        logger.info("Combining saved LLaMA activation files...")
        all_llama_activations = []
        for i in range(processed_count):
            file_path = os.path.join(llama_save_dir, f"llama_activations_sample_{args.start_index + i}.npy")
            if os.path.exists(file_path):
                all_llama_activations.append(np.load(file_path, allow_pickle=True))
        llama_output_file = os.path.join(llama_save_dir, f"dataset_all_activations_llama_{processed_count}.pkl")
        with open(llama_output_file, 'wb') as f:
            pickle.dump(all_llama_activations, f)
        logger.info(f"LLaMA activation dataset saved: {llama_output_file}")
        # Combine and save all LLaMA input_ids
        logger.info("Combining saved LLaMA input IDs files...")
        all_llama_input_ids = []
        for i in range(processed_count):
            file_path = os.path.join(llama_save_dir, f"llama_input_ids_sample_{args.start_index + i}.npy")
            if os.path.exists(file_path):
                all_llama_input_ids.append(np.load(file_path, allow_pickle=True))
        llama_input_ids_file = os.path.join(llama_save_dir, f"llama_input_ids_{processed_count}.pkl")
        with open(llama_input_ids_file, 'wb') as f:
            pickle.dump(all_llama_input_ids, f)
        logger.info(f"LLaMA input IDs dataset saved: {llama_input_ids_file}")
        # Clean up individual activation files (LLaMA) but NOT input_ids
        for i in range(processed_count):
            file_path = os.path.join(llama_save_dir, f"llama_activations_sample_{args.start_index + i}.npy")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
        # Clean up individual input ID files (LLaMA)
        for i in range(processed_count):
            file_path = os.path.join(llama_save_dir, f"llama_input_ids_sample_{args.start_index + i}.npy")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
    # Clean up individual activation files (VLM) but NOT input_ids
    for i in range(processed_count):
        file_path = os.path.join(llava_save_dir, f"all_activations_sample_{args.start_index + i}.npy")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
    # Save LLaMA dataset if requested
    if args.save_llama_dataset and llama_dataset_samples:
        llama_dataset_path = os.path.join(args.save_dir, f"llama_dataset_{processed_count}.jsonl")
        with open(llama_dataset_path, 'w') as f:
            for sample in llama_dataset_samples:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"LLaMA dataset saved to: {llama_dataset_path}")
    
    logger.info("✅ VLM{}activation extraction completed successfully!".format(" + LLaMA " if args.with_llama else " "))

if __name__ == "__main__":
    main() 