# %%
import sys
import os

sys.path.append("/homes/55/lachin/llama-scope-finetune-3/LLaVA-MORE")
os.environ['PYTHONPATH'] = '.'
os.environ['TOKENIZER_PATH'] = 'aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning'

from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE, SAEConfig

import torch

template = (
    "{% for message in messages %}"
    "{% if message['role'] != 'system' %}"
    "{{ message['role'].upper() + ': '}}"
    "{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] + ' '}}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}"
    "{{ content['text'] + ' '}}"
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'ASSISTANT:' }}"
    "{% endif %}"
)

# %% Initialize model
def initialize_vlm_model(vlm_model="llava-more", device="cuda"):
    if vlm_model == "llava-more":
        tokenizer, model, image_processor, _ = load_pretrained_model("aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning", None, 'llava')
    elif vlm_model == "llava-1.5-13b":
        tokenizer, model, image_processor, _ = load_pretrained_model("liuhaotian/llava-v1.5-13b", None, 'llava')
    elif vlm_model == "llava-1.5-7b":
        tokenizer, model, image_processor, _ = load_pretrained_model("liuhaotian/llava-v1.5-7b", None, 'llava')
    
    # Don't try to move the model if it's using accelerate hooks
    # The model will automatically use the appropriate device
    try:
        model = model.to(device)
    except RuntimeError as e:
        if "offloaded" in str(e) or "dispatched" in str(e):
            print(f"[WARN] Model is using accelerate hooks, keeping on current device: {model.device}")
        else:
            raise e

    model.generation_config.temperature=None
    model.generation_config.top_p=None  

    return tokenizer, model, image_processor

def initialize_language_model(language_model="llama-3.1-8b-it", device="cuda"):
    if language_model == "llama-3.1-8b-it":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.float16)
    elif language_model == "llama-2-13b-it":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16)
    elif language_model == "llama-2-7b-it":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16)
  
    model = model.to(device)

    model.generation_config.temperature=None
    model.generation_config.top_p=None  

    return tokenizer, model

def initialize_sae(layer_idx=0, checkpoint_path=None, initialize_random=False, device="cpu"):

    sae, cfg, sparsity = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{layer_idx}r_8x",
        device="cpu"  # Always load to CPU first
    )

    topk_cfg = dict(cfg)

    del topk_cfg['architecture']
    del topk_cfg['jump_relu_threshold']
    del topk_cfg['neuronpedia_id']
    del topk_cfg['activation_fn_str']

    new_topk_cfg = SAEConfig(
        architecture="topk",
        activation_fn_kwargs={
            "k": 50
        },
        activation_fn_str='topk',
        **topk_cfg
    )

    new_sae = SAE(
        new_topk_cfg
    ).to(device)

    if initialize_random:
        del sae

        if checkpoint_path is not None:
            print(f"[DEBUG] loading {checkpoint_path}")
            new_sae.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        else:
            print(f"[DEBUG] no checkpoint found, initializing...")

        return new_sae

    og_weights = sae.state_dict().copy()
    del og_weights['threshold']

    if checkpoint_path is not None:
        print(f"[DEBUG] loading {checkpoint_path}")
        new_sae.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    else:
        print(f"[DEBUG] no checkpoint found, initializing...")
        new_sae.load_state_dict(og_weights)

    del sae

    return new_sae

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

def get_vlm_input_ids(prompt, tokenizer, device="cuda"):
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    return input_ids, attention_mask

def instantiate_vlm_template(prompt, vlm_model, vlm_tokenizer, json_mode=True):
    model_name = vlm_model.config._name_or_path.lower()
    if "llava_more" in model_name:
        template_prompt = apply_vlm_template(prompt)
    elif "llava-v1.5-13b" in model_name or "llava-v1.5-7b" in model_name:
        if json_mode:
            system_prompt = "You are a helpful assistant. You respond only in json format."
        else:
            system_prompt = "You are a helpful assistant."

        template_prompt = vlm_tokenizer.apply_chat_template([
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": f"{prompt}"}],
            }
        ], 
        chat_template=template,
        add_generation_prompt=True,
        tokenize=False
        )

    return template_prompt

def process_vlm_inputs(image, prompt, image_processor, vlm_model, vlm_tokenizer, json_mode=True, use_black_image=False):    
    template_prompt = instantiate_vlm_template(prompt, vlm_model, vlm_tokenizer, json_mode)
    device = vlm_model.device
    
    if use_black_image:
        from PIL import Image
        # Get expected size from model config (default 224)
        size = getattr(vlm_model.config, 'image_size', 224)
        black_image = Image.new("RGB", (size, size), (0, 0, 0))
        image_tensor = process_image(black_image, image_processor, vlm_model)[0].unsqueeze(0).to(device)
    else:
        image_tensor = process_image(image, image_processor, vlm_model)[0].unsqueeze(0).to(device)
    
    input_ids, attention_mask = get_vlm_input_ids(template_prompt, vlm_tokenizer, device)
  
    image_sizes = [img.size for img in image_tensor]

    return input_ids, attention_mask, image_tensor, image_sizes

def generate_vlm_response(image, prompt, image_processor, vlm_model, vlm_tokenizer, max_new_tokens=100, json_mode=True):    
    input_ids, attention_mask, image_tensor, image_sizes = process_vlm_inputs(image, prompt, image_processor, vlm_model, vlm_tokenizer, json_mode)
   
    outputs = vlm_model.generate(
        input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        image_sizes=image_sizes,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=vlm_tokenizer.eos_token_id
    )
    
    response = vlm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def process_llm_inputs(prompt, tokenizer, json_mode=False, device="cuda"):
    if json_mode:
        system_prompt = "You are a helpful assistant. You respond only in json format."
    else:
        system_prompt = "You are a helpful assistant."

    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}"}
        ], 
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    return input_ids, attention_mask

def generate_llm_response(prompt, tokenizer, model, max_new_tokens=100, json_mode=True):

    input_ids, attention_mask = process_llm_inputs(prompt, tokenizer, json_mode, device=model.device)
    
    output = model.generate(input_ids, 
                           attention_mask=attention_mask, 
                           max_new_tokens=max_new_tokens, 
                           do_sample=False, 
                           pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)

def get_image_token_positions(input_ids):

    start_pos = input_ids[0].tolist().index(-200)
    end_pos = start_pos + 575 + 1
            
    return (start_pos, end_pos)