#!/usr/bin/env python3
"""
Extract LLaMA Activations from Text-Only Dataset

This script extracts activations from LLaMA using the text-only dataset generated
by generate_text_only_dataset.py. This enables comparison with VLM activations
for mechanistic interpretability analysis.

Usage:
    python extract_llama_activations.py [--dataset_path local_data/llama_dataset.jsonl] [--max_samples 15000]
    python finetune/extract_llama_activations.py --max_samples 200 
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any
import pickle

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

import torch
import numpy as np
from tqdm import tqdm
from nnsight import NNsight
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def initialize_llama_model(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda:0"):
    """
    Initialize LLaMA model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading LLaMA model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        model.eval()
        
        # Wrap with NNsight for activation extraction
        model = NNsight(model)
        
        logger.info(f"Successfully loaded model with {len(model.model.layers)} layers")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

def load_text_dataset(dataset_path: str, max_samples: int = None, sample_order_file: str = None) -> List[Dict[str, Any]]:
    """
    Load the text-only dataset.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        max_samples: Maximum number of samples to load
        sample_order_file: Path to VLM sample order file for verification
        
    Returns:
        List of dataset samples
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Load sample order for verification if provided
    expected_order = None
    if sample_order_file and Path(sample_order_file).exists():
        logger.info(f"Loading expected sample order from: {sample_order_file}")
        with open(sample_order_file, 'r') as f:
            vlm_order = json.load(f)
        expected_order = [sample['id'] for sample in vlm_order]
        logger.info(f"Expected {len(expected_order)} samples in specific order")
    
    samples = []
    actual_order = []
    
    with open(dataset_path, 'r') as f:
        for line_num, line in enumerate(f):
            if max_samples and len(samples) >= max_samples:
                break
            
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
                actual_order.append(sample.get('sample_id'))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num + 1}: {e}")
                continue
    
    # Verify order if expected order is provided
    if expected_order is not None:
        if len(actual_order) != len(expected_order):
            logger.warning(f"Sample count mismatch: expected {len(expected_order)}, got {len(actual_order)}")
        
        mismatches = 0
        for i, (expected, actual) in enumerate(zip(expected_order, actual_order)):
            if expected != actual:
                mismatches += 1
                if mismatches <= 5:  # Only log first few mismatches
                    logger.warning(f"Order mismatch at position {i}: expected {expected}, got {actual}")
        
        if mismatches == 0:
            logger.info("✅ Sample order matches VLM extraction perfectly!")
        else:
            logger.warning(f"⚠️  {mismatches} sample order mismatches detected")
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples

def format_llama_prompt(sample: Dict[str, Any]) -> str:
    """
    Format the prompt for LLaMA using Llama 3.1 chat template.
    
    Args:
        sample: Dataset sample containing prompt and response
        
    Returns:
        Formatted prompt string
    """
    # Get the text prompt (Caption: ... \nInstruction: ...)
    user_prompt = sample['prompt']
    response = sample.get('response', '')
    
    # Create messages in chat format
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    # Add response if available (for complete sequence processing)
    if response:
        messages.append({"role": "assistant", "content": response})
    
    return messages

def process_llama_inputs(sample: Dict[str, Any], tokenizer, device: str = "cuda:0") -> tuple:
    """
    Process a sample into LLaMA inputs.
    
    Args:
        sample: Dataset sample
        tokenizer: LLaMA tokenizer
        
    Returns:
        Tuple of (input_ids, attention_mask, prompt_text)
    """
    # Format the prompt
    messages = format_llama_prompt(sample)
    
    # Apply chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        # Use the built-in chat template
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=len(messages) == 1  # Add generation prompt if only user message
        )
    else:
        # Fallback formatting
        prompt_text = f"User: {messages[0]['content']}\n"
        if len(messages) > 1:
            prompt_text += f"Assistant: {messages[1]['content']}"
        else:
            prompt_text += "Assistant: "
    
    # Tokenize
    encoding = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048  # Reasonable max length
    )
    
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    
    return input_ids, attention_mask, prompt_text

def extract_activations(
    dataset_path: str,
    max_samples: int = 15000,
    batch_size: int = 4,  # Reduced default batch size
    save_dir: str = "local_data/llama_activations",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    sample_order_file: str = None,
    device: str = "cuda:0"
):
    """
    Extract activations from LLaMA on the text-only dataset.
    
    Args:
        dataset_path: Path to the text-only dataset
        max_samples: Maximum number of samples to process
        batch_size: Batch size for processing
        save_dir: Directory to save activations
        model_name: LLaMA model name
        sample_order_file: Path to VLM sample order file for verification
    """
    # Initialize model
    tokenizer, model = initialize_llama_model(model_name, device)
    
    # Clear GPU memory at start
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Load dataset
    samples = load_text_dataset(dataset_path, max_samples, sample_order_file)
    
    # Prepare save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up existing activation files
    for npy_file in glob.glob(str(save_dir / "llama_activations_sample_*.npy")):
        try:
            os.remove(npy_file)
            logger.debug(f"Removed existing file: {npy_file}")
        except Exception as e:
            logger.warning(f"Could not delete {npy_file}: {e}")
    
    # Initialize lists to collect all activations and metadata
    all_activations = []
    all_metadata = []
    
    num_layers = len(model.model.layers)
    logger.info(f"Model has {num_layers} layers")
    
    # Process in batches
    processed_count = 0
    batch_input_ids = []
    batch_attention_masks = []
    batch_samples = []
    
    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        if processed_count >= max_samples:
            break
        
        try:
            # Process sample
            input_ids, attention_mask, prompt_text = process_llama_inputs(sample, tokenizer, device)
            
            # Add to batch
            batch_input_ids.append(input_ids.squeeze(0))
            batch_attention_masks.append(attention_mask.squeeze(0))
            batch_samples.append(sample)
            
            # Process batch when full or at end
            if len(batch_input_ids) == batch_size or processed_count + len(batch_input_ids) == len(samples):
                logger.info(f"\nProcessing batch of {len(batch_input_ids)} samples...")
                
                # Log batch info
                sequence_lengths = [seq.size(0) for seq in batch_input_ids]
                logger.info(f"Sequence lengths: {sequence_lengths}")
                logger.info(f"Max length: {max(sequence_lengths)}")
                
                # Pad sequences
                input_ids = left_pad_sequences(batch_input_ids, tokenizer.pad_token_id)
                attention_mask = left_pad_masks(batch_attention_masks)
                
                logger.info(f"Batch shape: {input_ids.shape}")
                
                try:
                    # Extract activations
                    batch_activations = []
                    
                    with torch.no_grad():
                        with model.trace(input_ids, attention_mask=attention_mask, use_cache=False):
                            layer_activations = []
                            for l_idx in range(num_layers):
                                # Extract the output of each transformer layer
                                saved_activation = model.model.layers[l_idx].output[0].save()
                                layer_activations.append(saved_activation)
                    
                    # Process and save activations for each sample
                    for l_idx, saved_activation in enumerate(layer_activations):
                        activations = saved_activation.value.detach().cpu().numpy()
                        # activations.shape: (batch, seq_len, hidden_size)
                        
                        for b in range(activations.shape[0]):
                            # Get original sequence length
                            orig_len = batch_input_ids[b].size(0)
                            max_len = input_ids.size(1)
                            pad_len = max_len - orig_len
                            
                            # Extract activations for original sequence (skip padding)
                            # Skip first token (typically BOS) like in the VLM version
                            start_idx = 1
                            act = activations[b, pad_len + start_idx : pad_len + orig_len]
                            
                            if len(batch_activations) <= b:
                                batch_activations.append([])
                            batch_activations[b].append(act)
                        
                        # Clear layer activations immediately
                        del activations, saved_activation
                        torch.cuda.empty_cache()
                    
                    # Clear all layer activations
                    del layer_activations
                    torch.cuda.empty_cache()
                    
                    # Collect activations for batch
                    for j, sample_acts in enumerate(batch_activations):
                        sample_idx = processed_count + j
                        all_activations.append(sample_acts)
                        
                        # Also save sample metadata with verification info
                        metadata = {
                            'sample_id': batch_samples[j]['sample_id'],
                            'original_image': batch_samples[j]['original_image'],
                            'prompt': batch_samples[j]['prompt'],
                            'sequence_length': batch_input_ids[j].size(0),
                            'activation_shape': [act.shape for act in sample_acts],
                            'sample_order_verified': sample_order_file is not None
                        }
                        all_metadata.append(metadata)
                    
                    processed_count += len(batch_input_ids)
                    logger.info(f"Successfully processed batch, total: {processed_count}/{max_samples}")
                    
                except Exception as e:
                    logger.error(f"Error during activation extraction for batch: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Clear batch and memory
                del batch_input_ids, batch_attention_masks, batch_samples
                batch_input_ids = []
                batch_attention_masks = []
                batch_samples = []
                
                # Force memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
        
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue
    
    # Save all activations as a single pickle file
    output_file = save_dir / f"llama_activations_{processed_count}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_activations, f)
    
    # Save all metadata as a single JSON file
    metadata_file = save_dir / f"llama_metadata_{processed_count}.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    logger.info(f"Activation extraction complete! Processed {processed_count} samples")
    logger.info(f"Activations saved to: {output_file}")
    logger.info(f"Metadata saved to: {metadata_file}")
    
    # Print summary
    if all_activations:
        logger.info(f"Number of samples: {len(all_activations)}")
        logger.info(f"Layers per sample: {len(all_activations[0])}")
        if all_activations[0]:
            logger.info(f"First layer shape: {all_activations[0][0].shape}")
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Extract LLaMA activations from text-only dataset")
    parser.add_argument("--dataset_path", type=str, default="local_data/llama_dataset.jsonl",
                       help="Path to the text-only dataset file")
    parser.add_argument("--max_samples", type=int, default=15000,
                       help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing (reduced for memory)")
    parser.add_argument("--save_dir", type=str, default="local_data/llama_activations",
                       help="Directory to save activations")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="LLaMA model name from HuggingFace")
    parser.add_argument("--sample_order_file", type=str, default=None,
                       help="Path to VLM sample order file for verification")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="GPU device to use (e.g., cuda:0, cuda:1, cuda:2)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    extract_activations(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        model_name=args.model_name,
        sample_order_file=args.sample_order_file,
        device=args.device
    )

if __name__ == "__main__":
    main() 