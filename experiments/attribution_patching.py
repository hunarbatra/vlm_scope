"""
Attribution patching with corrupted inputs using mean image embeddings.

This implements two attribution variants for a chosen SAE feature on a
Visual-Language Model (LLaVA-MORE) using pre-extracted VQA feature samples:

- Method A ("optionA"): (corrupt - clean) · clean_grad
- Method B ("optionB"): (clean - corrupt) · corrupt_grad

Core idea:
- Load pre-extracted VQA feature samples from JSON files created by extract_feature_samples_vqa_spatial.py
- Extract top-K VQA sample indices and load the corresponding dataset samples
- Compute the mean image embedding once from layer-0 inputs over image token
  positions on a subset of examples.
- For each sample, run two traces:
  • Clean: original inputs.
  • Corrupt: replace layer-0 image embeddings (for image token span) with the
    computed mean embedding. This keeps the model on-distribution and avoids
    illegal in-graph setitem of intermediate tensors.
- Use layer outputs and attention outputs (and their gradients) from both runs

Dataset support: VQAv2 using pre-extracted feature samples from
extract_feature_samples_vqa_spatial.py. The script directly loads VQA samples
using indices stored in the JSON feature files.

Outputs include per-layer and per-head scores averaged over the dataset, saved
as JSON, plus optional plots.



CUDA_VISIBLE_DEVICES=1 \
CUDA_VISIBLE_DEVICES=0 python experiments/attribution_patching.py \
  --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
  --layer 9 \
  --feature-idx 29983 \
  --dataset vqa \
  --vqa-use-topk \
  --mean-samples 256 \
  --vqa-mean-from-mixed \
  --vqa-topk-dir /homes/55/lachin/llama-scope-finetune-3/results/stage_4/feature_samples/vqa_spatial_all_spatial \
  --vqa-spatial-cache-dir /homes/55/lachin/llama-scope-finetune-3/.cache/vqa_spatial_filter \
  --scoring both

CUDA_VISIBLE_DEVICES=7 python experiments/attribution_patching.py \
  --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
  --layer 26 \
  --proj-layer 20 \
  --feature-idx 807 \
  --dataset vqa \
  --vqa-use-topk \
  --mean-samples 256 \
  --vqa-mean-from-mixed \
  --vqa-topk-dir /homes/55/lachin/llama-scope-finetune-3/results/stage_4/feature_samples/vqa_spatial_all_spatial \
  --vqa-spatial-cache-dir /homes/55/lachin/llama-scope-finetune-3/.cache/vqa_spatial_filter \
  --scoring both   \
    --num-examples 200

CUDA_VISIBLE_DEVICES=0 python experiments/attribution_patching.py \
  --sae-checkpoint-dir /scratch/local/ssd/lachin/checkpoints_50k/text-only \
  --layer 10 \
  --proj-layer 10 \
  --feature-idx 24644 \
  --dataset vqa \
  --vqa-use-topk \
  --mean-samples 256 \
  --vqa-mean-from-mixed \
  --vqa-topk-dir /homes/55/lachin/llama-scope-finetune-3/results/stage_4/feature_samples/full/vqa_ocr \
  --scoring both   \
    --num-examples 200    
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, List, Tuple, Optional

import dotenv
dotenv.load_dotenv(".env")

import torch
import einops
from datasets import load_dataset
from nnsight import NNsight

import sys
import os
from pathlib import Path
workspace_root = Path(__file__).parent.parent
sys.path.append(str(workspace_root))  # enable importing the top-level `utils` package

from utils.datasets import load_vqa

import importlib.util
_vlm_utils_path = workspace_root / "finetune" / "vqa" / "utils.py"
_spec = importlib.util.spec_from_file_location("vlm_vqa_utils", _vlm_utils_path)
vlm_utils = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(vlm_utils)



def load_top_vqa_sample_indices(
    vqa_feature_data_dir: str,
    layer_idx: int,
    feature_idx: int,
    top_k: int,
) -> List[int]:
    """Load top-K VQA sample indices that most strongly activate a feature.

    Expects extracted feature samples at:
      {vqa_feature_data_dir}/text-only_layer_{layer_idx}_feature_{feature_idx}/sample_info.json

    This function reads from the JSON files created by extract_feature_samples_vqa_spatial.py
    instead of the original PT files, providing direct access to pre-extracted samples.

    Returns a list of integer indices into the VQA split.
    """
    feature_dir = Path(vqa_feature_data_dir)
    preferred = feature_dir / f"text-only_layer_{layer_idx}_feature_{feature_idx}" / "sample_info.json"
    fallback = feature_dir / f"layer_{layer_idx}_feature_{feature_idx}" / "sample_info.json"
    feature_file = preferred if preferred.exists() else fallback

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found. Tried: {preferred} and {fallback}"
        )

    try:
        with open(feature_file, "r") as f:
            samples = json.load(f)
        
        if not samples:
            raise ValueError(f"No samples found in {feature_file}")

        top_indices = [sample["sample_idx"] for sample in samples[:top_k] if "sample_idx" in sample]
        
        if not top_indices:
            raise ValueError(f"No VQA indices found in top {len(samples)} samples from {feature_file}")

        print(f"[INFO] Feature {feature_idx} (layer {layer_idx}) has {len(samples)} total samples")
        print(f"[INFO] Using top {len(top_indices)} samples by magnitude from {feature_file}")
        return top_indices
    except Exception as e:
        raise RuntimeError(f"Failed to load/parse feature data: {e}") from e





def load_vqa_spatial_indices(
    cache_dir: Optional[str] = ".cache/vqa_spatial_filter",
    cache_file: Optional[str] = None,
) -> Optional[List[int]]:
    """Load cached mapping from spatial subset indices to base VQA indices.

    Returns list such that list[i] = base_index for spatial subset index i.
    Returns None if no cache could be found or parsed.
    """
    candidates: List[Path] = []
    try:
        if cache_file:
            p = Path(cache_file)
            if p.exists():
                candidates.append(p)

        search_dirs: List[Path] = []
        if cache_dir:
            search_dirs.append(Path(cache_dir))
        search_dirs.append(Path(".cache/vqa_spatial_filter"))

        for d in search_dirs:
            if d.exists():
                for f in sorted(d.glob("indices_validation_*.json")):
                    candidates.append(f)

        for f in candidates:
            try:
                payload = json.loads(Path(f).read_text())
                indices = payload.get("indices") or payload.get("filtered_indices")
                if indices and isinstance(indices, list):
                    return [int(x) for x in indices]
            except Exception:
                continue
    except Exception:
        pass
    return None




def compute_mean_image_embedding(
    nnsight_model: NNsight,
    tokenizer,
    image_processor,
    dataset,
    num_mean_examples: int,
) -> torch.Tensor:
    """Compute mean layer-0 input embedding over image token positions across examples.

    Returns a 1D tensor of shape (d_model,) on the model device, dtype=float16.
    """
    n = min(len(dataset), num_mean_examples)
    
    batch_size = min(100, n)  # Process max 100 samples at a time
    all_embeddings = []
    
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_embeddings = []
        
        for i in range(batch_start, batch_end):
            sample = dataset[i]
            image = sample.get("pil_image") if "pil_image" in sample else sample["image"].convert("RGB")
            prompt = (sample.get("caption") or sample.get("question", "")).strip()

            input_ids, attention_mask, image_tensor, image_sizes = vlm_utils.process_vlm_inputs(
                image, prompt, image_processor, nnsight_model._module, tokenizer
            )
            img_start, img_end = vlm_utils.get_image_token_positions(input_ids)

            with torch.no_grad():
                with nnsight_model.trace(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=image_sizes,
                ) as tr_mean:
                    l0_img = nnsight_model.model.layers[0].input[0, img_start:img_end].save()

            l0_img_val = l0_img.cpu()  # (num_img_tokens, d_model)
            batch_embeddings.append(l0_img_val)

            if (i + 1) % 50 == 0 or i == batch_end - 1:
                print(f"  mean embed: processed {i + 1}/{n} samples")

        if batch_embeddings:
            batch_concat = torch.cat(batch_embeddings, dim=0)  # (total_tokens, d_model)
            all_embeddings.append(batch_concat)
        
        del batch_embeddings
        torch.cuda.empty_cache()

    if not all_embeddings:
        raise RuntimeError("Failed to compute mean image embedding: no tokens captured")

    all_embeddings_concat = torch.cat(all_embeddings, dim=0)  # (total_tokens, d_model)
    mean_image_embedding = all_embeddings_concat.mean(dim=0).to(torch.float16)
    
    del all_embeddings, all_embeddings_concat
    torch.cuda.empty_cache()
    
    return mean_image_embedding.to(getattr(nnsight_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")))


def compute_scores_for_sample(
    nnsight_model: NNsight,
    tokenizer,
    image_processor,
    sample,
    feature_vec: torch.Tensor,
    feature_layer: int,
    projection_layer: int,
    num_heads: int,
    head_dim: int,
    mean_image_embedding: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Method A and Method B scores for one sample.

    Returns:
      - method_A_layer: (projection_layer,)
      - method_A_heads: (projection_layer, num_heads)
      - method_B_layer: (projection_layer,)
      - method_B_heads: (projection_layer, num_heads)
    """
    image = sample.get("pil_image") if "pil_image" in sample else sample["image"].convert("RGB")
    prompt = (sample.get("caption") or sample.get("question", "")).strip()

    input_ids, attention_mask, image_tensor, image_sizes = vlm_utils.process_vlm_inputs(
        image, prompt, image_processor, nnsight_model._module, tokenizer
    )

    img_start, img_end = vlm_utils.get_image_token_positions(input_ids)

    layer_acts_clean, layer_grads_clean = [], []
    attn_acts_clean, attn_grads_clean = [], []
    with nnsight_model.trace(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        image_sizes=image_sizes,
    ) as tr:
        loss = (nnsight_model.model.layers[projection_layer].output[0][0, img_end:] @ feature_vec).sum()
        loss.backward()
        for l in range(projection_layer):
            layer_acts_clean.append(
                nnsight_model.model.layers[l].output[0][0, img_end:].detach().cpu().save()
            )
            layer_grads_clean.append(
                nnsight_model.model.layers[l].output[0].grad[0, img_end:].detach().cpu().save()
            )
            attn_acts_clean.append(
                nnsight_model.model.layers[l].self_attn.o_proj.input[0, img_end:].detach().cpu().save()
            )
            attn_grads_clean.append(
                nnsight_model.model.layers[l].self_attn.o_proj.input.grad[0, img_end:].detach().cpu().save()
            )

    layer_acts_corr, layer_grads_corr = [], []
    attn_acts_corr, attn_grads_corr = [], []
    with nnsight_model.trace(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        image_sizes=image_sizes,
    ) as tr_corr:
        num_img_tokens = int((img_end - img_start).item() if torch.is_tensor(img_end) else (img_end - img_start))
        patch_tensor = mean_image_embedding.to(getattr(nnsight_model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))).unsqueeze(0).repeat(num_img_tokens, 1)
        nnsight_model.model.layers[0].input[0, img_start:img_end] = patch_tensor

        corr_loss = (nnsight_model.model.layers[projection_layer].output[0][0, img_end:] @ feature_vec).sum()
        corr_loss.backward()
        for l in range(projection_layer):
            layer_acts_corr.append(
                nnsight_model.model.layers[l].output[0][0, img_end:].detach().cpu().save()
            )
            layer_grads_corr.append(
                nnsight_model.model.layers[l].output[0].grad[0, img_end:].detach().cpu().save()
            )
            attn_acts_corr.append(
                nnsight_model.model.layers[l].self_attn.o_proj.input[0, img_end:].detach().cpu().save()
            )
            attn_grads_corr.append(
                nnsight_model.model.layers[l].self_attn.o_proj.input.grad[0, img_end:].detach().cpu().save()
            )

    method_A_layer = torch.zeros(projection_layer)
    method_B_layer = torch.zeros(projection_layer)
    method_A_heads = torch.zeros(projection_layer, num_heads)
    method_B_heads = torch.zeros(projection_layer, num_heads)

    for l in range(projection_layer):
        diff = layer_acts_corr[l].value - layer_acts_clean[l].value
        score = einops.einsum(diff, layer_grads_clean[l].value, "tokens dim, tokens dim -> tokens").abs().mean()
        method_A_layer[l] = score

    for l in range(projection_layer):
        for h in range(num_heads):
            clean_act = attn_acts_clean[l].value[:, h * head_dim:(h + 1) * head_dim]
            corr_act = attn_acts_corr[l].value[:, h * head_dim:(h + 1) * head_dim]
            clean_grad = attn_grads_clean[l].value[:, h * head_dim:(h + 1) * head_dim]
            diff = corr_act - clean_act
            score = einops.einsum(diff, clean_grad, "tokens dim, tokens dim -> tokens").abs().mean()
            method_A_heads[l, h] = score

    for l in range(projection_layer):
        diff = layer_acts_clean[l].value - layer_acts_corr[l].value
        score = einops.einsum(diff, layer_grads_corr[l].value, "tokens dim, tokens dim -> tokens").abs().mean()
        method_B_layer[l] = score

    for l in range(projection_layer):
        for h in range(num_heads):
            clean_act = attn_acts_clean[l].value[:, h * head_dim:(h + 1) * head_dim]
            corr_act = attn_acts_corr[l].value[:, h * head_dim:(h + 1) * head_dim]
            corr_grad = attn_grads_corr[l].value[:, h * head_dim:(h + 1) * head_dim]
            diff = clean_act - corr_act
            score = einops.einsum(diff, corr_grad, "tokens dim, tokens dim -> tokens").abs().mean()
            method_B_heads[l, h] = score

    torch.cuda.empty_cache()
    return method_A_layer, method_A_heads, method_B_layer, method_B_heads




def main(args):
    if args.output_dir is None:
        args.output_dir = f"results/organized_spatial_features/layer_{args.layer}_feature_{args.feature_idx}"
        print(f"[INFO] Auto-generated output directory: {args.output_dir}")
    
    tokenizer, vlm_model, image_processor = vlm_utils.initialize_vlm_model("llava-more", device="cuda")
    vlm_model = NNsight(vlm_model)

    ckpt_candidates = list(Path(args.sae_checkpoint_dir).glob(f"*layer_{args.layer}*.pt"))
    if not ckpt_candidates:
        raise FileNotFoundError(
            f"No checkpoint matching *layer_{args.layer}*.pt found in {args.sae_checkpoint_dir}"
        )
    sae_ckpt = ckpt_candidates[0]
    sae = vlm_utils.initialize_sae(layer_idx=args.layer, checkpoint_path=sae_ckpt, device="cpu")
    sae.eval()

    if args.feature_idx >= sae.W_dec.shape[0]:
        raise ValueError(
            f"Feature-idx {args.feature_idx} exceeds decoder dim {sae.W_dec.shape[0]}"
        )
    feature_vec = sae.W_dec[args.feature_idx].to(dtype=torch.float16, device=vlm_model.device)

    if args.dataset != "vqa":
        raise ValueError(f"Only dataset 'vqa' is supported, got: {args.dataset}")
    
    if not args.vqa_use_topk:
        raise ValueError("--vqa-use-topk must be set for VQA dataset")
    
    spatial_subset_indices = load_top_vqa_sample_indices(
        vqa_feature_data_dir=args.vqa_topk_dir,
        layer_idx=args.layer,
        feature_idx=args.feature_idx,
        top_k=min(args.vqa_topk or args.num_examples, args.num_examples),
    )
    
    if len(spatial_subset_indices) == 0:
        raise ValueError("Top-K selection returned 0 indices; check your feature data args")
    
    if getattr(args, "vqa_topk_indices_are_base", False):
        print("[INFO] Treating top-k indices as base VQAv2 indices (skipping spatial mapping)")
        base_indices = spatial_subset_indices
    else:
        spatial_map = load_vqa_spatial_indices(
            cache_dir=args.vqa_spatial_cache_dir,
            cache_file=args.vqa_spatial_cache_file,
        )
        if spatial_map is None:
            print("[WARN] Spatial VQA indices cache not found. Assuming sample_idx are base VQAv2 indices.")
            base_indices = spatial_subset_indices
        else:
            base_indices = []
            for sidx in spatial_subset_indices:
                if sidx < 0 or sidx >= len(spatial_map):
                    print(f"[WARN] Spatial index {sidx} out of range {len(spatial_map)}; skipping")
                    continue
                base_indices.append(int(spatial_map[sidx]))

    if len(base_indices) == 0:
        raise ValueError("After mapping spatial indices, no valid base VQAv2 indices remained")

    full_ds = load_vqa(split=args.vqa_split)
    dataset = full_ds.select(base_indices[: args.num_examples])
    print(f"[INFO] Loaded {len(dataset)} VQAv2 samples directly using indices from feature samples")
    
    print(f"\n[DEBUG] First 5 spatial questions being analyzed:")
    for i in range(min(5, len(dataset))):
        question = dataset[i].get("question", "N/A")
        print(f"  {i+1}. {question[:100]}{'...' if len(question) > 100 else ''}")
    
    if len(dataset) > 10:
        print(f"  ... and {len(dataset) - 10} more spatial questions ...")
        print(f"\n[DEBUG] Last 5 spatial questions being analyzed:")
        for i in range(max(0, len(dataset) - 5), len(dataset)):
            question = dataset[i].get("question", "N/A")
            print(f"  {i+1}. {question[:100]}{'...' if len(question) > 100 else ''}")
    elif len(dataset) > 5:
        print(f"\n[DEBUG] Last {len(dataset) - 5} spatial questions being analyzed:")
        for i in range(5, len(dataset)):
            question = dataset[i].get("question", "N/A")
            print(f"  {i+1}. {question[:100]}{'...' if len(question) > 100 else ''}")
    print()

    print(f"[INFO] Running attribution patching on {len(dataset)} samples …")

    cache_dir = Path(args.output_dir) / "samples_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    from PIL import Image
    for i in range(len(dataset)):
        ex = dataset[i]
        img = ex.get("pil_image") if "pil_image" in ex else ex["image"].convert("RGB")
        caption = (ex.get("caption") or ex.get("question", "")).strip()
        img_path = cache_dir / f"sample{i}.png"
        try:
            img.save(img_path)
        except Exception:
            Image.new("RGB", (224, 224), (128, 128, 128)).save(img_path)
        manifest.append({"idx": i, "image_path": str(img_path), "caption": caption})
    manifest_path = Path(args.output_dir) / "samples_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Cached {len(manifest)} samples → {manifest_path}")

    if args.vqa_use_topk and args.vqa_mean_from_mixed:
        print(f"[INFO] Computing mean image embedding from {args.mean_samples} mixed VQA samples (not filtered)...")
        full_vqa_dataset = load_vqa(split=args.vqa_split)
        mean_image_embedding = compute_mean_image_embedding(
            vlm_model, tokenizer, image_processor, full_vqa_dataset, args.mean_samples
        )
        del full_vqa_dataset
    else:
        print(f"[INFO] Computing mean image embedding from first {args.mean_samples} samples …")
        mean_image_embedding = compute_mean_image_embedding(
            vlm_model, tokenizer, image_processor, dataset, args.mean_samples
        )

    head_dim = vlm_model.model.config.head_dim
    num_heads = vlm_model.model.config.num_attention_heads
    feature_layer = int(args.layer)
    total_layers = len(vlm_model.model.layers)
    if args.proj_layer is None:
        projection_layer = feature_layer
    else:
        projection_layer = int(args.proj_layer)
    if projection_layer < 1 or projection_layer >= total_layers:
        raise ValueError(f"--proj-layer must be in [1, {total_layers - 1}] (got {projection_layer})")

    A_layer_sum = torch.zeros(projection_layer)
    A_heads_sum = torch.zeros(projection_layer, num_heads)
    B_layer_sum = torch.zeros(projection_layer)
    B_heads_sum = torch.zeros(projection_layer, num_heads)

    for idx in range(len(dataset)):
        A_layer, A_heads, B_layer, B_heads = compute_scores_for_sample(
            vlm_model,
            tokenizer,
            image_processor,
            dataset[idx],
            feature_vec,
            feature_layer,
            projection_layer,
            num_heads,
            head_dim,
            mean_image_embedding,
        )
        A_layer_sum += A_layer
        A_heads_sum += A_heads
        B_layer_sum += B_layer
        B_heads_sum += B_heads

        if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
            print(f"  processed {idx + 1}/{len(dataset)} samples")

    A_layer_avg = A_layer_sum / len(dataset)
    A_heads_avg = A_heads_sum / len(dataset)
    B_layer_avg = B_layer_sum / len(dataset)
    B_heads_avg = B_heads_sum / len(dataset)

    def _compute_top_k_heads(head_scores_tensor: torch.Tensor, k: int) -> list:
        if head_scores_tensor.numel() == 0:
            return []
        k = min(k, head_scores_tensor.numel())
        flat_scores = head_scores_tensor.reshape(-1)
        topk_vals, topk_idx = torch.topk(flat_scores, k)
        results = []
        num_heads_local = head_scores_tensor.shape[1]
        for rank_idx in range(k):
            flat_index = int(topk_idx[rank_idx].item())
            layer_index = flat_index // num_heads_local
            head_index = flat_index % num_heads_local
            results.append({
                "layer": int(layer_index),
                "head": int(head_index),
                "name": f"L{int(layer_index)}H{int(head_index)}",
                "score": float(topk_vals[rank_idx].item()),
            })
        return results

    top_heads_A = _compute_top_k_heads(A_heads_avg, k=5)
    top_heads_B = _compute_top_k_heads(B_heads_avg, k=5)

    if args.scoring in ("optionA", "both"):
        print("\n=== Average layer scores (Method A) ===")
        for l, s in enumerate(A_layer_avg):
            print(f"Layer {l:02d}: {s:.4f}")
    if args.scoring in ("optionB", "both"):
        print("\n=== Average layer scores (Method B) ===")
        for l, s in enumerate(B_layer_avg):
            print(f"Layer {l:02d}: {s:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "layer_scores_method_A": A_layer_avg.tolist(),
        "head_scores_method_A": A_heads_avg.tolist(),
        "layer_scores_method_B": B_layer_avg.tolist(),
        "head_scores_method_B": B_heads_avg.tolist(),
        "top_heads_method_A": top_heads_A,
        "top_heads_method_B": top_heads_B,
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "layer_limit": int(projection_layer),
        "feature_layer": int(feature_layer),
        "projection_layer": int(projection_layer),
        "mean_samples": int(args.mean_samples),
    }
    with open(out_dir / "dla_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved summary → {out_dir / 'dla_summary.json'}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'legend.frameon': False,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

        if args.scoring in ("optionA", "both"):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(range(projection_layer), A_layer_avg.numpy(), 
                   color='#E84D8A', linewidth=3, marker='o', markersize=6, 
                   alpha=0.85, markerfacecolor='white', markeredgewidth=2)
            ax.set_xlabel("Layer Index", fontsize=14, fontweight='bold')
            ax.set_ylabel("Attribution Score", fontsize=14, fontweight='bold')
            ax.set_title("Layer-wise Attribution (Method A)", fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)  # Start y-axis from 0 for better visualization
            
            ax.set_xticks(range(0, projection_layer, max(1, max(1, projection_layer // 10))))
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            plt.tight_layout()
            plt.savefig(out_dir / "layer_scores_method_A.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved → {out_dir / 'layer_scores_method_A.png'}")

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            heatmap = sns.heatmap(
                A_heads_avg.numpy(),
                cmap="magma",
                xticklabels=range(num_heads),
                yticklabels=range(projection_layer),
                cbar_kws={"label": "Attribution Score", "shrink": 0.8},
                ax=ax,
                square=True,
                linewidths=0,
                cbar=True
            )
            ax.set_xlabel("Attention Head", fontsize=14, fontweight='bold')
            ax.set_ylabel("Layer Index", fontsize=14, fontweight='bold')
            ax.set_title("Attention Head Attribution (Method A)", fontsize=16, fontweight='bold')
            
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=11)
            cbar.ax.set_ylabel("Attribution Score", fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(out_dir / "head_scores_method_A.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved → {out_dir / 'head_scores_method_A.png'}")

        if args.scoring in ("optionB", "both"):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(range(projection_layer), B_layer_avg.numpy(), 
                   color='#FEB326', linewidth=3, marker='o', markersize=6, 
                   alpha=0.85, markerfacecolor='white', markeredgewidth=2)
            ax.set_xlabel("Layer Index", fontsize=14, fontweight='bold')
            ax.set_ylabel("Attribution Score", fontsize=14, fontweight='bold')
            ax.set_title("Layer-wise Attribution (Method B)", fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)  # Start y-axis from 0 for better visualization
            
            ax.set_xticks(range(0, projection_layer, max(1, max(1, projection_layer // 10))))
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            plt.tight_layout()
            plt.savefig(out_dir / "layer_scores_method_B.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved → {out_dir / 'layer_scores_method_B.png'}")

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            heatmap = sns.heatmap(
                B_heads_avg.numpy(),
                cmap="cividis",
                xticklabels=range(num_heads),
                yticklabels=range(projection_layer),
                cbar_kws={"label": "Attribution Score", "shrink": 0.8},
                ax=ax,
                square=True,
                linewidths=0,
                cbar=True
            )
            ax.set_xlabel("Attention Head", fontsize=14, fontweight='bold')
            ax.set_ylabel("Layer Index", fontsize=14, fontweight='bold')
            ax.set_title("Attention Head Attribution (Method B)", fontsize=16, fontweight='bold')
            
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=11)
            cbar.ax.set_ylabel("Attribution Score", fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(out_dir / "head_scores_method_B.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved → {out_dir / 'head_scores_method_B.png'}")

    print("[INFO] Attribution patching (mean-embedding corruption) finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attribution patching with mean image embeddings")
    parser.add_argument("--sae-checkpoint-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True, help="SAE/VLM layer index (feature layer)")
    parser.add_argument("--feature-idx", type=int, required=True, help="SAE feature index")
    parser.add_argument("--proj-layer", type=int, default=None, help="Layer index at which to project the feature vector (defaults to --layer)")

    parser.add_argument("--num-examples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--mean-samples", type=int, default=128, help="Samples used to compute mean image embedding")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for plots and processed data (auto-generated if not specified)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--dataset",
        type=str,
        default="vqa",
        choices=["vqa"],
        help="Dataset to use: 'vqa' (lmms-lab/VQAv2 spatially filtered)",
    )

    parser.add_argument(
        "--vqa-split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="VQAv2 split to use when dataset=='vqa'",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="both",
        choices=["optionA", "optionB", "both"],
        help="Which scores to print/plot (JSON always stores both)",
    )

    parser.add_argument("--vqa-use-topk", action="store_true", help="Use top-K VQA samples by feature activation")
    parser.add_argument("--vqa-topk", type=int, default=0, help="Top-K samples to select (defaults to --num-examples when 0)")
    parser.add_argument("--vqa-topk-dir", type=str, default="results/stage_4/feature_samples/vqa_spatial_all_spatial", help="Directory containing extracted feature samples")


    parser.add_argument("--vqa-mean-from-mixed", action="store_true", help="When using --vqa-use-topk, compute mean embedding from mixed VQA samples instead of filtered ones")

    parser.add_argument("--vqa-spatial-cache-dir", type=str, default=None,
                       help="Directory containing cached VQA spatial indices (indices_validation_*.json)")
    parser.add_argument("--vqa-spatial-cache-file", type=str, default=None,
                       help="Explicit path to a cached spatial indices JSON file")

    parser.add_argument("--vqa-topk-indices-are-base", action="store_true",
                       help="Interpret top-k sample_idx as base VQAv2 indices (skip spatial mapping)")

    args = parser.parse_args()

    main(args)


