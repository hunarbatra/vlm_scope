"""Debug: verify NNsight ablation with corrected indexing.

Found bugs:
1. layer.output[0] has shape (seq, d_in) — NO batch dim. Using [0, img_end:]
   slices [row=0, col=img_end:] = 2048 values, not text tokens!
2. In NNsight 0.6.x, accessing .output twice on same module may deadlock.
   Use single access + in-place modify on the view.
"""
import os
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("debug-ablation")
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


@app.function(image=image, gpu="A100", volumes={"/vol": volume}, timeout=600)
def debug_trace():
    import sys, torch, numpy as np, traceback
    from nnsight import NNsight
    from PIL import Image

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions, initialize_jumprelu_sae

    print(f"NNsight version: {__import__('nnsight').__version__}")

    processor, model_raw = initialize_vlm_model("google/paligemma2-3b-pt-224", device="cuda")
    model_raw.eval()
    nns_model = NNsight(model_raw)
    n_layers = model_raw.config.text_config.num_hidden_layers
    model_dtype = next(model_raw.parameters()).dtype

    img = Image.new("RGB", (224, 224), (100, 150, 200))
    prompt = 'Is this true or false? "The cat is on the table"'
    input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, model_raw)
    _, img_end = get_image_token_positions(input_ids)
    print(f"seq_len={input_ids.shape[1]}, img_end={img_end}, text_tokens={input_ids.shape[1] - img_end}")

    # Load SAE, get two feature vectors
    ckpt_path = f"/vol/results/paligemma2/run_jumprelu/checkpoints/pretrained_layer_17.pt"
    sae = initialize_jumprelu_sae(17, checkpoint_path=ckpt_path, device="cuda",
                                   cache_dir="/vol/cache/huggingface")
    feat_a, feat_b = 14475, 12345
    fv_a = sae.W_dec[feat_a].detach().to(model_dtype).to("cuda")
    fv_a = fv_a / fv_a.norm().clamp(min=1e-8)
    fv_b = sae.W_dec[feat_b].detach().to(model_dtype).to("cuda")
    fv_b = fv_b / fv_b.norm().clamp(min=1e-8)
    del sae
    print(f"Feat A ({feat_a}): first3={fv_a[:3].tolist()}, dim={fv_a.shape}")
    print(f"Feat B ({feat_b}): first3={fv_b[:3].tolist()}, dim={fv_b.shape}")
    print(f"cos(A,B) = {(fv_a @ fv_b).item():.6f}")

    def run_ablation_fixed(fv_vec, label):
        """Fixed pattern: correct indexing + single .output access."""
        fv = fv_vec.unsqueeze(0)  # (1, d_in)
        try:
            with nns_model.trace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values) as tr:
                for l in range(n_layers):
                    # self_attn.output[0]: (batch, seq, d_in) — index [0, img_end:]
                    attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                    attn_proj = (attn_out @ fv.T) * fv
                    attn_out -= attn_proj

                    # mlp.output: (batch, seq, d_in) — index [0, img_end:]
                    mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                    mlp_proj = (mlp_out @ fv.T) * fv
                    mlp_out -= mlp_proj

                    # layer.output[0]: (seq, d_in) — NO batch dim! index [img_end:]
                    layer_out = nns_model.model.language_model.layers[l].output[0][img_end:]
                    layer_proj = (layer_out @ fv.T) * fv
                    layer_out -= layer_proj

                logits_saved = nns_model.output.logits.save()
            last5 = logits_saved[:, -1, :5].detach().float().cpu().numpy()
            print(f"  {label}: logits[:5] = {last5}")
            return last5
        except Exception as e:
            print(f"  {label}: EXCEPTION:\n{traceback.format_exc()}")
            return None

    # Baseline
    print("\n--- Baseline ---")
    with nns_model.trace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values) as tr:
        logits_base = nns_model.output.logits.save()
    base_val = logits_base[:, -1, :5].detach().float().cpu().numpy()
    print(f"  Baseline: logits[:5] = {base_val}")

    # Fixed pattern
    print("\n--- Fixed ablation (correct indexing) ---")
    a_fixed = run_ablation_fixed(fv_a, "Feat A")
    b_fixed = run_ablation_fixed(fv_b, "Feat B")

    fv_rand = torch.randn_like(fv_a)
    fv_rand = fv_rand / fv_rand.norm()
    rand_fixed = run_ablation_fixed(fv_rand, "Random")

    print("\n=== SUMMARY ===")
    if a_fixed is not None and b_fixed is not None:
        print(f"A==B? {np.allclose(a_fixed, b_fixed, atol=1e-4)}")
        print(f"A==baseline? {np.allclose(a_fixed, base_val, atol=1e-4)}")
        print(f"B==baseline? {np.allclose(b_fixed, base_val, atol=1e-4)}")
    if rand_fixed is not None and a_fixed is not None:
        print(f"A==rand? {np.allclose(a_fixed, rand_fixed, atol=1e-4)}")

    return "Debug complete"


@app.local_entrypoint()
def main():
    print(debug_trace.remote())
