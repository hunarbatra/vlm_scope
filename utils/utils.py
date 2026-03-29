# %%
import sys
import os
import argparse
import dotenv
import base64
from openai import OpenAI
from io import BytesIO
from tqdm import tqdm
import requests
import aiohttp
import json
import pathlib
import numpy as np
from datasets import load_dataset
from sae_lens import SAE, SAEConfig
import torch
dotenv.load_dotenv("../.env")

# Get the absolute path to the local LLaVA-MORE directory
llava_more_path = "/homes/55/lachin/llama-scope-finetune-3/temp_llava_more"
sys.path.insert(0, llava_more_path)  # Put it at the beginning of sys.path to ensure it's used first
os.environ['PYTHONPATH'] = '.'
os.environ['TOKENIZER_PATH'] = 'aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning'

from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForImageTextToText

import torch
from PIL import Image
import requests
from io import BytesIO


# %% Initialize model
def initialize_vlm_model(vlm_model="llava-more"):
    tokenizer, model, image_processor, _ = load_pretrained_model(
        "aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning", 
        None, 
        'llava',
        device_map="auto" 
    )

    model.generation_config.temperature=None
    model.generation_config.top_p=None  

    return tokenizer, model, image_processor

def process_image(image, image_processor, model):
    images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    return images_tensor

def apply_vlm_template(query):
    image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    prompt = image_token + '\n' + query

    # Setup conversation
    conv = conv_templates["llama_3_1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt

def get_vlm_input_ids(prompt, tokenizer):
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    attention_mask = torch.ones_like(input_ids).cuda()
    return input_ids, attention_mask

def process_vlm_inputs(image, prompt, image_processor, vlm_model, vlm_tokenizer):    
    template_prompt = apply_vlm_template(prompt)
    image_tensor = process_image(image, image_processor, vlm_model)[0].unsqueeze(0).to('cuda')
    
    input_ids, attention_mask = get_vlm_input_ids(template_prompt, vlm_tokenizer)
    input_ids = input_ids.to(vlm_model.device)
    attention_mask = attention_mask.to(vlm_model.device)
  
    # Get actual dimensions from the image tensor
    image_sizes = torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]], dtype=torch.long, device=image_tensor.device)

    return input_ids, attention_mask, image_tensor, image_sizes

def get_image_token_positions(input_ids):

    start_pos = input_ids[0].tolist().index(-200)
    end_pos = start_pos + 575 + 1

    return (start_pos, end_pos)


def get_text_token_positions(input_ids):
    if torch.is_tensor(input_ids):
        input_ids = input_ids.tolist()
    for i in range(len(input_ids) - 3):
        if input_ids[i:i+4] == [128006, 9125, 128007, 271]:
            start_pos = i + 4
            break
    for i in range(start_pos, len(input_ids) - 4):
        if input_ids[i:i+5] == [128009, 128006, 78191, 128007, 271]:
            end_pos = i
            break   
    return (start_pos, end_pos)