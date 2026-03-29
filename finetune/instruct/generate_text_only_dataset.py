#!/usr/bin/env python3
"""
Generate Text-Only Dataset for LLaMA

This script converts the LLaVA-Instruct-150K multimodal dataset into a text-only equivalent
by replacing images with their COCO captions. This enables fair comparison between 
vision-language model (VLM) and language-only model activations.

Usage:
    python generate_text_only_dataset.py [--max_samples 15000] [--output_path llama_dataset.jsonl]

    # python finetune/generate_text_only_dataset.py --max_samples 100
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm
import logging

# Setup paths  
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_coco_captions() -> Dict[str, str]:
    """
    Load COCO captions from local JSON annotation file.
    Returns:
        Dict mapping COCO image filenames to their captions
    """
    import json
    annotation_path = ROOT_DIR / "local_data" / "coco_annotations" / "annotations" / "captions_train2017.json"
    logger.info(f"Loading COCO captions from local file: {annotation_path}")
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
    logger.info(f"Loaded {len(caption_map)} COCO captions from local file.")
    return caption_map

def download_coco_captions_api() -> Dict[str, str]:
    """
    Backup method to download COCO captions using the COCO API.
    """
    try:
        # This is a simplified backup - in practice you might want to use pycocotools
        # For now, return empty dict and we'll handle missing captions gracefully
        logger.warning("COCO API backup not implemented. Will use placeholder captions.")
        return {}
    except Exception as e:
        logger.error(f"Backup COCO API also failed: {e}")
        return {}

def process_conversation(conversations: List[Dict], sample_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract human instruction and assistant response from conversation.
    
    Args:
        conversations: List of conversation messages
        sample_id: Sample identifier for logging
        
    Returns:
        Tuple of (instruction, response) or (None, None) if parsing fails
    """
    if len(conversations) < 2:
        logger.warning(f"Sample {sample_id}: Conversation has less than 2 messages")
        return None, None
    
    human_msg = None
    assistant_msg = None
    
    for msg in conversations:
        if msg.get('from') == 'human':
            human_msg = msg.get('value', '').strip()
            # Remove image tokens
            human_msg = human_msg.replace('<image>', '').strip()
        elif msg.get('from') == 'gpt':
            assistant_msg = msg.get('value', '').strip()
    
    if not human_msg:
        logger.warning(f"Sample {sample_id}: No human message found")
        return None, None
    
    return human_msg, assistant_msg

def create_text_prompt(caption: str, instruction: str) -> str:
    """
    Create text-only prompt combining caption and instruction.
    
    Args:
        caption: Image caption
        instruction: Human instruction/question
        
    Returns:
        Formatted prompt string
    """
    return f"Caption: {caption}\nInstruction: {instruction}"

def generate_text_dataset(max_samples: int = 15000, output_path: str = "llama_dataset.jsonl", 
                         sample_order_file: str = None, seed: int = 42):
    """
    Generate the text-only dataset.
    
    Args:
        max_samples: Maximum number of samples to process
        output_path: Output file path
        sample_order_file: Path to VLM sample order file for exact replication
        seed: Random seed for shuffling (if not using sample order file)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return
    
    logger.info(f"Loading LLaVA-Instruct-150K dataset...")
    
    # Load the dataset
    ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True)
    # No shuffling: preserve original order
    
    # Load COCO captions
    logger.info("Loading COCO captions...")
    coco_captions = load_coco_captions()
    
    # Prepare output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load sample order if provided
    target_samples = None
    if sample_order_file and Path(sample_order_file).exists():
        logger.info(f"Loading sample order from: {sample_order_file}")
        with open(sample_order_file, 'r') as f:
            sample_order = json.load(f)
        
        # Create mapping of sample IDs to preserve order
        target_samples = {sample['id']: i for i, sample in enumerate(sample_order)}
        logger.info(f"Will replicate exact order of {len(target_samples)} samples from VLM extraction")
        
        # Initialize results list to maintain order
        ordered_results = [None] * len(sample_order)
    # No shuffling: preserve original order if no sample order file is provided
    
    # Process samples
    processed_count = 0
    skipped_count = 0
    missing_caption_count = 0
    found_count = 0
    
    logger.info(f"Processing up to {max_samples} samples...")
    
    for i, sample in enumerate(tqdm(ds, total=max_samples, desc="Processing samples")):
        if target_samples is None and processed_count >= max_samples:
            break
        
        try:
            sample_id = sample.get('id', f'sample_{i}')
            image_filename = sample.get('image', '')
            conversations = sample.get('conversations', [])
            
            # If using target samples, check if this sample is in our target list
            if target_samples is not None:
                if sample_id not in target_samples:
                    continue  # Skip samples not in VLM extraction
                target_idx = target_samples[sample_id]
                found_count += 1
                logger.debug(f"Found target sample {sample_id} at position {target_idx}")
            
            # Process conversation
            instruction, response = process_conversation(conversations, sample_id)
            if instruction is None:
                instruction = "[MISSING INSTRUCTION]"
            if response is None:
                response = "[MISSING RESPONSE]"
            # Get caption
            caption = coco_captions.get(image_filename)
            if caption is None:
                base_filename = os.path.basename(image_filename)
                caption = coco_captions.get(base_filename)
            if caption is None:
                # Use a placeholder caption
                caption = f"An image from the COCO dataset (filename: {image_filename})"
                missing_caption_count += 1
                logger.debug(f"Sample {sample_id}: Using placeholder caption for {image_filename}")
            
            # Create text prompt
            prompt = create_text_prompt(caption, instruction)
            
            # Create output sample
            output_sample = {
                "sample_id": sample_id,
                "original_image": image_filename,
                "prompt": prompt,
                "instruction": instruction,
                "caption": caption,
                "response": response or "",
                "conversations": conversations  # Keep original for reference
            }
            
            if target_samples is not None:
                # Store in correct position for ordered output
                ordered_results[target_idx] = output_sample
                
                # Check if we've found all target samples
                if found_count >= len(target_samples):
                    logger.info(f"Found all {len(target_samples)} target samples")
                    break
            else:
                # Direct writing for non-ordered processing
                if processed_count == 0:
                    # Open file for writing
                    outfile = open(output_path, 'w')
                outfile.write(json.dumps(output_sample) + '\n')
                processed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            skipped_count += 1
            continue
    
    # Write ordered results if using target samples
    if target_samples is not None:
        logger.info("Writing samples in exact VLM order...")
        with open(output_path, 'w') as outfile:
            for result in ordered_results:
                if result is not None:
                    outfile.write(json.dumps(result) + '\n')
                    processed_count += 1
                else:
                    logger.warning("Missing sample in ordered results")
    else:
        # Close file for non-ordered processing
        outfile.close()
    
    # Summary
    logger.info(f"Dataset generation complete!")
    logger.info(f"Processed: {processed_count} samples")
    logger.info(f"Skipped: {skipped_count} samples") 
    logger.info(f"Missing captions: {missing_caption_count} samples")
    if target_samples is not None:
        logger.info(f"Target samples found: {found_count}/{len(target_samples)}")
    logger.info(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate text-only dataset for LLaMA")
    parser.add_argument("--max_samples", type=int, default=15000, 
                       help="Maximum number of samples to process")
    parser.add_argument("--output_path", type=str, default="local_data/llama_dataset.jsonl",
                       help="Output file path")
    parser.add_argument("--sample_order_file", type=str, default=None,
                       help="JSON file with VLM sample order for exact replication")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for shuffling (if not using sample order file)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    generate_text_dataset(args.max_samples, args.output_path, args.sample_order_file, args.seed)

if __name__ == "__main__":
    main() 