"""
Utility functions for PaliGemma2-3B SAE pipeline.
Mirrors finetune/vqa/utils.py but adapted for PaliGemma2 + Gemma Scope SAEs.

Key differences from LLaVA-MORE version:
- Model: PaliGemma2-3B (google/paligemma2-3b-pt-224) — 26 Gemma2 decoder layers
- SAE base: gemma-scope-2b-pt-res (width 16k) from HuggingFace
- Image tokens: PaliGemma2 prepends image tokens (256 tokens for 224x224)
- No LLaVA-specific template — PaliGemma2 uses simple text after image tokens

Supports two SAE architectures:
- TopK SAE (from sae-lens): used by Llama Scope, top-k activation selection
- JumpReLU SAE (custom): used by Gemma Scope, threshold-based gating with STE gradients
"""

import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig
from sae_lens.saes.sae import SAEMetadata
from huggingface_hub import hf_hub_download


# ---------- Model ----------

NUM_IMAGE_TOKENS = 256  # PaliGemma2-3B with 224x224 input

def initialize_vlm_model(model_name="google/paligemma2-3b-pt-224", device="cpu"):
    """Load PaliGemma2-3B model and processor."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.eval()
    return processor, model


def process_vlm_inputs(image, prompt, processor, model, device=None):
    """Process image + text into model inputs for PaliGemma2.

    Returns (input_ids, attention_mask, pixel_values) — all on device.
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    return inputs["input_ids"], inputs["attention_mask"], inputs.get("pixel_values")


def get_image_token_positions(input_ids):
    """Return (start, end) of image token span.

    PaliGemma2 prepends image tokens at the beginning of the sequence.
    The image token id is typically 257152 (the <image> placeholder).
    We detect the contiguous block of image tokens at the start.
    """
    ids = input_ids[0].tolist()
    # PaliGemma2 uses token id 257152 for image placeholders
    IMAGE_TOKEN_ID = 257152

    start = None
    end = None
    for i, tok in enumerate(ids):
        if tok == IMAGE_TOKEN_ID:
            if start is None:
                start = i
            end = i + 1
        elif start is not None:
            break

    if start is None:
        # Fallback: assume first NUM_IMAGE_TOKENS tokens are image
        start = 0
        end = min(NUM_IMAGE_TOKENS, len(ids))

    return (start, end)


# ---------- SAE ----------

# Gemma Scope 2B residual SAEs: width 16k, one per layer
# Each layer has multiple L0 variants; pick the one closest to the paper's sparsity
GEMMA_SCOPE_REPO = "google/gemma-scope-2b-pt-res"

# Average L0 values per layer (from the Gemma Scope release)
# We pick width_16k with the available average_l0 variant
# Picked the L0 variant closest to k=50 for each layer
LAYER_L0_MAP = {
    0: 46,  1: 40,  2: 53,  3: 59,  4: 60,  5: 34,
    6: 36,  7: 36,  8: 37,  9: 37,  10: 39, 11: 41,
    12: 41, 13: 43, 14: 43, 15: 41, 16: 42, 17: 42,
    18: 40, 19: 40, 20: 38, 21: 38, 22: 38, 23: 38,
    24: 38, 25: 55,
}


def _download_gemma_scope_params(layer_idx: int, cache_dir: str = None) -> str:
    """Download Gemma Scope SAE params.npz for a given layer. Returns local path."""
    l0 = LAYER_L0_MAP.get(layer_idx)
    if l0 is None:
        raise ValueError(f"No L0 mapping for layer {layer_idx}")

    subfolder = f"layer_{layer_idx}/width_16k/average_l0_{l0}"
    path = hf_hub_download(
        repo_id=GEMMA_SCOPE_REPO,
        filename="params.npz",
        subfolder=subfolder,
        cache_dir=cache_dir,
    )
    return path


def _load_gemma_scope_weights(params_path: str) -> dict:
    """Load Gemma Scope npz into a state_dict compatible with SAE Lens TopK SAE.

    Gemma Scope npz keys (already in SAE Lens naming):
        W_enc: (d_in=2304, d_sae=16384)
        W_dec: (d_sae=16384, d_in=2304)
        b_enc: (d_sae=16384,)
        b_dec: (d_in=2304,)
        threshold: (d_sae=16384,) — JumpReLU threshold, not used for TopK
    """
    data = np.load(params_path)

    return {
        "W_enc": torch.from_numpy(data["W_enc"]).float(),   # (d_in, d_sae)
        "W_dec": torch.from_numpy(data["W_dec"]).float(),    # (d_sae, d_in)
        "b_enc": torch.from_numpy(data["b_enc"]).float(),    # (d_sae,)
        "b_dec": torch.from_numpy(data["b_dec"]).float(),    # (d_in,)
    }


def initialize_sae(layer_idx: int = 0, checkpoint_path=None, initialize_random=False,
                    device="cpu", cache_dir=None, k=50):
    """Initialize a TopK SAE for PaliGemma2 (Gemma 2B backbone).

    - If checkpoint_path is given, loads fine-tuned weights from it
    - If initialize_random=True and no checkpoint, random init
    - Otherwise, loads Gemma Scope pretrained weights

    Architecture: TopK with k=50, width 16384, d_in=2304 (Gemma 2B hidden size)
    """
    d_in = 2304       # Gemma 2B hidden dimension
    d_sae = 16384     # Gemma Scope width_16k

    meta = SAEMetadata()
    meta["model_name"] = "google/gemma-2-2b"
    meta["hook_name"] = f"blocks.{layer_idx}.hook_resid_post"
    meta["hook_layer"] = layer_idx
    meta["context_size"] = 1024
    meta["dataset_path"] = ""
    meta["prepend_bos"] = False

    cfg = TopKSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        normalize_activations="none",
        dtype="float32",
        device=str(device),
        apply_b_dec_to_input=False,
        metadata=meta,
    )

    sae = TopKSAE(cfg)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sae.load_state_dict(state)
    elif not initialize_random:
        # Load Gemma Scope pretrained weights
        print(f"[INFO] Loading Gemma Scope pretrained weights for layer {layer_idx}")
        params_path = _download_gemma_scope_params(layer_idx, cache_dir=cache_dir)
        pretrained_weights = _load_gemma_scope_weights(params_path)
        sae.load_state_dict(pretrained_weights, strict=False)
    else:
        print(f"[INFO] Random initialization for layer {layer_idx}")

    sae = sae.to(device)
    return sae


# ---------- JumpReLU SAE ----------

class _RectangleFunction(autograd.Function):
    """STE approximation for indicator function derivative."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class _JumpReLUFunction(autograd.Function):
    """JumpReLU activation with STE gradient for threshold."""
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * _RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class _StepFunction(autograd.Function):
    """Step function with STE gradient for threshold (used in L0 computation)."""
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * _RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder matching Gemma Scope architecture.

    Architecture:
        encode: x @ W_enc + b_enc -> JumpReLU(pre_act, threshold)
        decode: f @ W_dec + b_dec

    Parameters:
        W_enc: (d_in, d_sae) — encoder weights
        W_dec: (d_sae, d_in) — decoder weights (unit-norm rows)
        b_enc: (d_sae,) — encoder bias
        b_dec: (d_in,) — decoder bias
        threshold: (d_sae,) — per-feature JumpReLU threshold (learnable)
    """

    def __init__(self, d_in: int, d_sae: int, device="cpu"):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae, device=device))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(d_sae, d_in, device=device))
        )
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device))
        self.threshold = nn.Parameter(torch.ones(d_sae, device=device) * 0.001)

        # Initialize W_dec to unit norm, W_enc = W_dec^T
        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x, bandwidth=0.001):
        """Encode activations through JumpReLU."""
        pre_jump = x @ self.W_enc + self.b_enc
        return _JumpReLUFunction.apply(pre_jump, self.threshold, bandwidth)

    def decode(self, f):
        """Decode sparse features back to activation space."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x, bandwidth=0.001):
        """Full forward pass: encode then decode."""
        f = self.encode(x, bandwidth=bandwidth)
        return self.decode(f)

    def compute_loss(self, x, bandwidth=0.001, target_l0=50.0, sparsity_coeff=1.0):
        """Compute reconstruction + sparsity loss.

        Returns (loss, recon_loss, l0, fvu) for logging.
        """
        pre_jump = x @ self.W_enc + self.b_enc
        f = _JumpReLUFunction.apply(pre_jump, self.threshold, bandwidth)
        recon = self.decode(f)

        # Reconstruction loss
        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()

        # L0 sparsity via step function with STE gradient
        l0 = _StepFunction.apply(pre_jump, self.threshold, bandwidth).sum(dim=-1).mean()

        # Sparsity penalty targeting l0 ≈ target_l0
        sparsity_loss = sparsity_coeff * ((l0 / target_l0) - 1).pow(2)

        loss = recon_loss + sparsity_loss

        # FVU for logging
        with torch.no_grad():
            var = (x - x.mean(dim=0)).pow(2).sum(dim=-1).mean()
            fvu = recon_loss / (var + 1e-8)

        return loss, recon_loss.item(), l0.item(), fvu.item()

    def set_decoder_norm_to_unit_norm(self):
        """Normalize decoder weight rows to unit norm."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.W_dec.div_(norms)

    def remove_gradient_parallel_to_decoder_directions(self):
        """Remove component of W_dec gradient parallel to W_dec (keeps perpendicular)."""
        if self.W_dec.grad is not None:
            with torch.no_grad():
                # W_dec: (d_sae, d_in), each row is a unit vector
                parallel = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True) * self.W_dec
                self.W_dec.grad -= parallel


def _load_gemma_scope_weights_jumprelu(params_path: str) -> dict:
    """Load Gemma Scope npz into JumpReLU SAE state_dict (includes threshold)."""
    data = np.load(params_path)

    state = {
        "W_enc": torch.from_numpy(data["W_enc"]).float(),     # (d_in, d_sae)
        "W_dec": torch.from_numpy(data["W_dec"]).float(),      # (d_sae, d_in)
        "b_enc": torch.from_numpy(data["b_enc"]).float(),      # (d_sae,)
        "b_dec": torch.from_numpy(data["b_dec"]).float(),      # (d_in,)
    }
    # Load threshold if present (Gemma Scope JumpReLU SAEs have this)
    if "threshold" in data:
        state["threshold"] = torch.from_numpy(data["threshold"]).float()  # (d_sae,)

    return state


def initialize_jumprelu_sae(layer_idx: int = 0, checkpoint_path=None,
                             initialize_random=False, device="cpu", cache_dir=None):
    """Initialize a JumpReLU SAE for PaliGemma2 (Gemma 2B backbone).

    - If checkpoint_path is given, loads fine-tuned weights
    - If initialize_random=True, random init
    - Otherwise, loads Gemma Scope pretrained weights (including threshold)

    Architecture: JumpReLU, width 16384, d_in=2304
    """
    d_in = 2304
    d_sae = 16384

    sae = JumpReLUSAE(d_in=d_in, d_sae=d_sae, device="cpu")

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"[INFO] Loading JumpReLU checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sae.load_state_dict(state)
    elif not initialize_random:
        print(f"[INFO] Loading Gemma Scope JumpReLU weights for layer {layer_idx}")
        params_path = _download_gemma_scope_params(layer_idx, cache_dir=cache_dir)
        pretrained_weights = _load_gemma_scope_weights_jumprelu(params_path)
        sae.load_state_dict(pretrained_weights, strict=False)
    else:
        print(f"[INFO] Random JumpReLU initialization for layer {layer_idx}")

    sae = sae.to(device)
    return sae
