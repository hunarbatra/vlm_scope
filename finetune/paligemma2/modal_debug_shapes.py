"""Debug: print shapes of model layer outputs to understand dimensions."""
import os
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("debug-shapes")
volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers>=4.44", "sae-lens>=4.0",
        "nnsight>=0.3", "Pillow", "numpy", "accelerate",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "TRANSFORMERS_OFFLINE": "1",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)


@app.function(image=image, gpu="A100", volumes={"/vol": volume}, timeout=300)
def debug_shapes():
    import sys, torch
    from nnsight import NNsight
    from PIL import Image

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions

    processor, model_raw = initialize_vlm_model("google/paligemma2-3b-pt-224", device="cuda")
    model_raw.eval()
    nns_model = NNsight(model_raw)
    n_layers = model_raw.config.text_config.num_hidden_layers

    print(f"Model config:")
    print(f"  hidden_size: {model_raw.config.text_config.hidden_size}")
    print(f"  num_attention_heads: {model_raw.config.text_config.num_attention_heads}")
    print(f"  head_dim: {model_raw.config.text_config.head_dim}")
    print(f"  num_hidden_layers: {n_layers}")

    img = Image.new("RGB", (224, 224), (100, 150, 200))
    prompt = 'Is this true? "The cat is on the table"'
    input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, model_raw)
    _, img_end = get_image_token_positions(input_ids)
    print(f"\nInput: seq_len={input_ids.shape[1]}, img_end={img_end}")

    # Trace to capture shapes of layer 0 outputs
    with nns_model.trace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values) as tr:
        # Layer 0 self_attn output
        attn_out_raw = nns_model.model.language_model.layers[0].self_attn.output
        attn_shape = attn_out_raw[0].shape  # save shape
        print(f"  layers[0].self_attn.output[0].shape = {attn_shape}")

        # Layer 0 mlp output
        mlp_out_raw = nns_model.model.language_model.layers[0].mlp.output
        mlp_shape = mlp_out_raw.shape
        print(f"  layers[0].mlp.output.shape = {mlp_shape}")

        # Layer 0 full output
        layer_out_raw = nns_model.model.language_model.layers[0].output
        layer_shape = layer_out_raw[0].shape
        print(f"  layers[0].output[0].shape = {layer_shape}")

        logits = nns_model.output.logits.save()

    print(f"\nOutput logits shape: {logits.shape}")

    # Also check: what does layer.output actually contain?
    print("\n--- Detailed layer output inspection ---")
    with nns_model.trace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values) as tr:
        layer_out = nns_model.model.language_model.layers[0].output
        n_elements = len(layer_out)
        print(f"  layers[0].output has {n_elements} elements")
        for i in range(min(n_elements, 4)):
            elem = layer_out[i]
            if elem is not None:
                print(f"  layers[0].output[{i}].shape = {elem.shape}")
            else:
                print(f"  layers[0].output[{i}] = None")

        logits2 = nns_model.output.logits.save()

    return "Shape debug complete"


@app.local_entrypoint()
def main():
    print(debug_shapes.remote())
