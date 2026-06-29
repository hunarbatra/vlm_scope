"""
Local 8-GPU analysis pipeline for PaliGemma2 text-only JumpReLU SAEs.

Matches the original vlm_scope pipeline (LLaVA-MORE codebase):
  Step 1: FVU table (from training logs)
  Step 2: Cosine similarity (Gemma Scope base vs fine-tuned W_dec)
  Step 3: Visual energy Ev (VLM activations vs text-only LLM activations)
  Step 4: Select adapted features (Ev > epsilon AND cosine < threshold)
  Step 5: Firing frequencies — VQA baseline (50K) + VSR spatial (all splits)
          Uses SAMPLE-LEVEL firing (feature fires in ANY token of sample)
          matching modal_vsr_firing.py approach.
  Step 6: Spatial features (Fisher exact test: VQA vs VSR, sample-level)
  Step 7: Lexical artifact filtering
  Step 8: Intersection (adapted ∩ spatial ∩ lexical)

Step 5 uses VSR dataset (cambridgeltl/vsr_random) for spatial comparison,
NOT VQA-spatial-keyword subset. This matches the original paper's approach
that produces 7-15% ablation drops.

Usage:
    cd vlm_scope/finetune/paligemma2
    python3 local_analysis_textonly.py
    python3 local_analysis_textonly.py --step 5 6    # Run steps 5 and 6
"""

import os
import sys
import gc
import re
import ast
import csv
import json
import math
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import numpy as np

# ======================== Configuration ========================

MODEL_NAME = "google/paligemma2-3b-mix-448"
TEXT_MODEL_NAME = "google/gemma-2-2b"  # Base LLM for text-only activations (matches gemma-scope-2b-pt-res)
N_LAYERS = 26
D_SAE = 16384
N_GPUS = 8

# Paths — overridable via env vars (set by main() so spawned workers inherit them)
CHECKPOINT_DIR = Path(os.environ.get("VLMSCOPE_CKPT_DIR", "/data1/vlm_scope_sae_mix448_textonly/checkpoints"))
LOG_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/logs")
ANALYSIS_DIR = Path(os.environ.get("VLMSCOPE_ANALYSIS_DIR", "/data1/vlm_scope_sae_mix448_textonly/analysis_ocrvqa"))
HF_CACHE = "/data1/hf_cache/hub"
HF_DATASETS_CACHE = "/data1/hf_cache/datasets"

# Set env before any HF imports in workers — critical for spawned processes
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HOME"] = "/data1/hf_cache"

# OCR-VQA firing index file (built by build_ocrvqa_indices.py)
OCRVQA_INDICES = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocrvqa/ocrvqa_indices_train.json")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

# Dataset
N_TRAINING_SAMPLES = 50_000
N_FIRING_SAMPLES = 50_000  # Match original: 50K samples for firing frequencies
N_ENERGY_SAMPLES = 5_000   # VQAv2 validation indices 50000-54999

# Full spatial keyword list from original vlm_scope codebase (90+ keywords with regex matching)
SPATIAL_KEYWORDS = [
    # Basic directions and movement
    "left", "right", "front", "back", "ahead", "behind", "forward", "backward",
    "forwards", "backwards", "up", "down", "upward", "downward",
    # Corners, sides, and extremes
    "top", "bottom", "upper", "lower", "leftmost", "rightmost", "topmost", "bottommost",
    "uppermost", "lowermost", "corner", "edge", "border", "side",
    "left side", "right side", "top side", "bottom side",
    # Multi-axis quadrant phrases
    "top left", "top right", "bottom left", "bottom right",
    "upper left", "upper right", "lower left", "lower right",
    "middle left", "middle right", "center left", "center right",
    # Relative spatial relations
    "above", "over", "overhead", "atop", "on top", "on top of",
    "below", "under", "underneath", "beneath",
    "in front", "in front of", "at the front", "at the back",
    "next to", "beside", "alongside", "near", "nearby", "close to",
    "adjacent", "adjacent to", "across from", "opposite", "opposite to", "facing",
    "around", "surrounding", "encircling", "between", "in between", "among", "amid",
    "inside", "inside of", "outside", "outside of", "within",
    "to the left", "to the right", "to the left of", "to the right of",
    # Distance and extent
    "distance", "closer", "closest", "nearest", "nearer",
    "far", "farther", "farthest", "further", "furthest",
    "height", "width",
    # Orientation and axes
    "vertical", "horizontal", "diagonal", "direction", "oriented", "orientation",
    "rotated", "rotation",
    # Compass directions
    "north", "south", "east", "west",
    "north east", "north west", "south east", "south west",
    "northeast", "northwest", "southeast", "southwest",
    # Locative cues
    "position", "positioned", "located", "location", "placement", "placed",
    # Foreground/background
    "foreground", "background", "frontmost", "backmost", "background of",
]

# Generic prompts for lexical artifact filtering
GENERIC_PROMPTS = [
    "Describe how the items are arranged.",
    "Comment on the overall layout and organization of the scene.",
    "Summarize the structure in terms of grouping or separation.",
    "Explain the relative positioning of objects without naming directions.",
    "Describe patterns of arrangement, such as order or symmetry.",
]

# Adapted feature selection
EPSILON = 0.01
COSINE_PERCENTILE = 25.0

# Spatial feature identification (match original find_spatial_features.py VSR example)
# Original uses --min-diff 0.05 --odds-thr 3 for VSR-based spatial detection
# (NOT 0.005 which is for the less-strict VQA-spatial-text-only variant)
ODDS_THR = 3.0
MIN_FREQ_DIFF = 0.05


def compile_spatial_regex():
    """Compile spatial keywords into a single regex pattern (matches original codebase)."""
    escaped_variants = []
    for kw in SPATIAL_KEYWORDS:
        kw = kw.strip()
        if not kw:
            continue
        parts = [re.escape(p) for p in kw.split()]
        if len(parts) == 1:
            pattern = parts[0]
        else:
            joiner = r"(?:\s+|-)"
            pattern = joiner.join(parts)
        escaped_variants.append(rf"\b{pattern}\b")
    combined = "|".join(escaped_variants)
    return re.compile(combined, flags=re.IGNORECASE)


# ======================== Step 1: FVU Table ========================

def step1_fvu_table():
    """Read training logs and extract final FVU per layer."""
    print("\n" + "=" * 60)
    print("[Step 1] FVU Table")
    print("=" * 60)

    out_dir = ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for layer_idx in range(N_LAYERS):
        csv_path = LOG_DIR / f"metrics_text-only_layer_{layer_idx}.csv"
        if not csv_path.exists():
            results[layer_idx] = None
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            last = rows[-1]
            results[layer_idx] = {
                "fvu": float(last["fvu"]),
                "l0": float(last["l0"]),
                "total_tokens": int(last["total_tokens"]),
            }

    out_path = out_dir / "fvu_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "fvu", "l0", "total_tokens"])
        for li in range(N_LAYERS):
            e = results.get(li)
            if e:
                writer.writerow([li, f"{e['fvu']:.6f}", f"{e['l0']:.1f}", e["total_tokens"]])
            else:
                writer.writerow([li, "N/A", "N/A", "N/A"])

    print(f"Saved: {out_path}")
    for li in range(N_LAYERS):
        e = results.get(li)
        if e:
            print(f"  L{li:2d}: FVU={e['fvu']:.4f}, L0={e['l0']:.1f}")


# ======================== Step 2: Cosine Similarity ========================

def step2_cosine():
    """Cosine similarity between Gemma Scope base and fine-tuned W_dec. No GPU needed."""
    print("\n" + "=" * 60)
    print("[Step 2] Cosine Similarity (base vs fine-tuned W_dec)")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import _download_gemma_scope_params, _load_gemma_scope_weights

    out_dir = ANALYSIS_DIR / "cosines"
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(N_LAYERS):
        out_path = out_dir / f"cosines_layer_{layer_idx}.npy"
        if out_path.exists():
            cos = np.load(out_path)
            print(f"  L{layer_idx}: SKIP (exists) mean={cos.mean():.4f}")
            continue

        # Load base Gemma Scope W_dec
        params_path = _download_gemma_scope_params(layer_idx, cache_dir=HF_CACHE)
        base_weights = _load_gemma_scope_weights(params_path)
        base_W_dec = base_weights["W_dec"]  # (d_sae, d_in)

        # Load fine-tuned W_dec
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"  L{layer_idx}: SKIP (no checkpoint)")
            continue
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ft_W_dec = state["W_dec"]

        # Cosine similarity per feature (row-wise)
        base_norm = base_W_dec / base_W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        ft_norm = ft_W_dec / ft_W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cosines = (base_norm * ft_norm).sum(dim=1).numpy()

        np.save(out_path, cosines)
        print(f"  L{layer_idx}: mean={cosines.mean():.4f}, min={cosines.min():.4f}, max={cosines.max():.4f}")


# ======================== Step 3: Visual Energy Ev ========================
# Matches original vlm_scope: compare VLM (PaliGemma2 with images) vs text-only LLM (Gemma-2-2B).
# Ev = mean_sq_vlm / n_vlm_tokens, Et = mean_sq_text / n_text_tokens (raw energies, not ratios).

def _energy_worker(gpu_id, layer_indices):
    """Compute visual energy Ev: VLM activations vs text-only LLM activations."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
    from nnsight import NNsight
    from datasets import load_dataset

    out_dir = ANALYSIS_DIR / "energy"
    out_dir.mkdir(parents=True, exist_ok=True)

    remaining = [l for l in layer_indices if not (out_dir / f"Ev_layer_{l}.npy").exists()]
    if not remaining:
        print(f"[Energy GPU{gpu_id}] All layers done, skipping")
        return

    # Load VLM (PaliGemma2 mix-448)
    print(f"[Energy GPU{gpu_id}] Loading VLM: {MODEL_NAME}...")
    vlm_processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=False)
    vlm_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=False
    ).to(device).eval()
    vlm_nns = NNsight(vlm_raw)

    # Load text-only LLM (Gemma-2-2B — matches gemma-scope-2b-pt-res SAE base)
    print(f"[Energy GPU{gpu_id}] Loading text LLM: {TEXT_MODEL_NAME}...")
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, local_files_only=False)
    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token_id = text_tokenizer.eos_token_id
    text_raw = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=False
    ).to(device).eval()
    text_nns = NNsight(text_raw)

    print(f"[Energy GPU{gpu_id}] Loading VQAv2...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")

    val_start, val_end = 50_000, min(50_000 + N_ENERGY_SAMPLES, len(vqa))
    print(f"[Energy GPU{gpu_id}] Layers: {remaining}, samples: {val_start}-{val_end}")

    for layer_idx in remaining:
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()

        sum_sq_vlm = np.zeros(D_SAE, dtype=np.float64)
        sum_sq_text = np.zeros(D_SAE, dtype=np.float64)
        n_vlm, n_text, n_errors = 0, 0, 0

        for si in range(val_start, val_end):
            try:
                sample = vqa[si]
                image = sample["image"].convert("RGB")
                question = sample["question"]
                vlm_prompt = f"answer en {question}"

                # --- VLM forward pass (image + text) ---
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    image, vlm_prompt, vlm_processor, vlm_raw, device=device
                )
                with torch.no_grad():
                    with vlm_nns.trace(
                        input_ids=input_ids, attention_mask=attn_mask,
                        pixel_values=pixel_values, use_cache=False,
                    ) as tr:
                        vlm_act = vlm_nns.model.language_model.layers[layer_idx].output[0].save()

                vlm_codes = sae.encode(vlm_act.detach().squeeze(0).float()).detach().cpu().numpy().astype(np.float64)
                sum_sq_vlm += (vlm_codes ** 2).sum(axis=0)
                n_vlm += vlm_codes.shape[0]

                # --- Text-only LLM forward pass (question only, no image) ---
                text_inputs = text_tokenizer(question, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    with text_nns.trace(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                    ) as tr:
                        text_act = text_nns.model.layers[layer_idx].output[0].save()

                text_codes = sae.encode(text_act.detach().squeeze(0).float()).detach().cpu().numpy().astype(np.float64)
                sum_sq_text += (text_codes ** 2).sum(axis=0)
                n_text += text_codes.shape[0]

            except Exception as e:
                n_errors += 1
                if n_errors <= 10:
                    print(f"[Energy GPU{gpu_id}] Error sample {si}: {e}", flush=True)
                    import traceback; traceback.print_exc()
                continue

            if (si - val_start) % 500 == 0 and gpu_id == 0:
                print(f"  L{layer_idx}: {si - val_start}/{val_end - val_start} samples", flush=True)

        if n_errors > 0:
            print(f"[Energy GPU{gpu_id}] L{layer_idx}: {n_errors} errors out of {val_end - val_start} samples", flush=True)

        # Raw per-feature energies (match original compute_feature_metrics.py)
        Ev = (sum_sq_vlm / max(n_vlm, 1)).astype(np.float32)
        Et = (sum_sq_text / max(n_text, 1)).astype(np.float32)

        np.save(out_dir / f"Ev_layer_{layer_idx}.npy", Ev)
        np.save(out_dir / f"Et_layer_{layer_idx}.npy", Et)
        print(f"[Energy GPU{gpu_id}] L{layer_idx}: mean Ev={Ev.mean():.6f}, mean Et={Et.mean():.6f}, "
              f"n_vlm_tokens={n_vlm}, n_text_tokens={n_text}, Ev>{EPSILON}: {(Ev > EPSILON).sum()}", flush=True)

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    del vlm_raw, vlm_nns, text_raw, text_nns
    torch.cuda.empty_cache()
    gc.collect()


def step3_energy(n_gpus=N_GPUS):
    """Run visual energy computation across GPUs."""
    print("\n" + "=" * 60)
    print("[Step 3] Visual Energy Ev (VLM vs text-only LLM)")
    print("=" * 60)

    (ANALYSIS_DIR / "energy").mkdir(parents=True, exist_ok=True)

    # Use fewer GPUs since each worker loads 2 models (VLM + text LLM)
    # Each PaliGemma2-3b ~6GB + Gemma-2-2B ~4GB ≈ 10GB per worker → fits in 24GB A5000
    layers_per_worker = math.ceil(N_LAYERS / n_gpus)
    assignments = []
    for w in range(n_gpus):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, N_LAYERS)
        worker_layers = list(range(start, end))
        if worker_layers:
            assignments.append((w, worker_layers))

    for g, layers in assignments:
        print(f"  GPU {g}: layers {layers}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for gpu_id, layers in assignments:
        p = mp.Process(target=_energy_worker, args=(gpu_id, layers))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        print(f"[ERROR] Energy workers {failed} failed!")


# ======================== Step 4: Adapted Features ========================

def step4_adapted():
    """Select adapted features: Ev > epsilon AND cosine in bottom percentile."""
    print("\n" + "=" * 60)
    print("[Step 4] Select Adapted Features")
    print("=" * 60)

    cos_dir = ANALYSIS_DIR / "cosines"
    ev_dir = ANALYSIS_DIR / "energy"
    out_dir = ANALYSIS_DIR / "adapted"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_Ev, all_cosines, layer_indices = [], [], []
    for layer_idx in range(N_LAYERS):
        ev_path = ev_dir / f"Ev_layer_{layer_idx}.npy"
        cos_path = cos_dir / f"cosines_layer_{layer_idx}.npy"
        if not ev_path.exists() or not cos_path.exists():
            print(f"  [WARN] Missing data for layer {layer_idx}")
            continue
        ev = np.load(ev_path)
        cos = np.load(cos_path)
        all_Ev.append(ev)
        all_cosines.append(cos)
        layer_indices.extend([layer_idx] * len(ev))

    global_Ev = np.concatenate(all_Ev)
    global_cosines = np.concatenate(all_cosines)
    layer_arr = np.array(layer_indices)

    # Select: Ev > epsilon AND cosine in bottom percentile
    ev_mask = global_Ev > EPSILON
    cos_threshold = float(np.percentile(global_cosines, COSINE_PERCENTILE))
    cos_mask = global_cosines <= cos_threshold
    adapted_mask = ev_mask & cos_mask
    adapted_global_indices = set(np.where(adapted_mask)[0].tolist())

    print(f"  Ev > {EPSILON}: {ev_mask.sum()}")
    print(f"  Cosine <= {cos_threshold:.4f} ({COSINE_PERCENTILE}th pctile): {cos_mask.sum()}")
    print(f"  Intersection: {len(adapted_global_indices)}")

    # Group by layer
    results = []
    for layer_idx in range(N_LAYERS):
        layer_mask = layer_arr == layer_idx
        layer_adapted = adapted_mask & layer_mask
        feature_indices = np.where(layer_adapted)[0] - (layer_arr < layer_idx).sum()
        feature_indices = feature_indices.tolist()

        layer_ev = global_Ev[layer_mask]
        layer_cos = global_cosines[layer_mask]

        results.append({
            "layer": layer_idx,
            "n_adapted": len(feature_indices),
            "adapted_indices": feature_indices,
            "mean_cosine": float(layer_cos.mean()) if len(layer_cos) > 0 else 0,
            "mean_Ev": float(layer_ev.mean()) if len(layer_ev) > 0 else 0,
        })
        if feature_indices:
            print(f"  L{layer_idx}: {len(feature_indices)} adapted features")

    out_path = out_dir / "adapted_features_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "n_adapted", "adapted_indices", "mean_cosine", "mean_Ev"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "layer": r["layer"],
                "n_adapted": r["n_adapted"],
                "adapted_indices": str(r["adapted_indices"]),
                "mean_cosine": f"{r['mean_cosine']:.6f}",
                "mean_Ev": f"{r['mean_Ev']:.6f}",
            })

    summary = {
        "total_adapted": len(adapted_global_indices),
        "epsilon": EPSILON,
        "cosine_percentile": COSINE_PERCENTILE,
        "cosine_threshold": cos_threshold,
        "per_layer": {r["layer"]: r["n_adapted"] for r in results},
    }
    with open(out_dir / "adapted_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Total adapted: {len(adapted_global_indices)}")
    return summary


# ======================== Step 5: Firing Frequencies ========================
# Per-TOKEN firing counts (matching original LLaVA-MORE pipeline).
# Pass 1: 50K VQA baseline — count tokens where each feature fires > 0
# Pass 2: VSR dataset (cambridgeltl/vsr_random, all splits) — same metric.
# Fisher test then compares per-token firing rates.

def _firing_worker(gpu_id, layer_indices):
    """Track per-token firing frequencies for assigned layers.

    - Pass 1: 50K VQA samples for baseline
    - Pass 2: All VSR samples (cambridgeltl/vsr_random, train+val+test)
    - Per-token: count how many text tokens each feature fires on
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset
    from PIL import Image
    import io

    vqa_dir = ANALYSIS_DIR / "firing_vqa_pertoken"
    ocrvqa_dir = ANALYSIS_DIR / "firing_ocrvqa_pertoken"
    vqa_dir.mkdir(parents=True, exist_ok=True)
    ocrvqa_dir.mkdir(parents=True, exist_ok=True)

    # Check which passes still need to run
    remaining_vqa = [l for l in layer_indices if not (vqa_dir / f"firing_vqa_layer_{l}.json").exists()]
    remaining_ocrvqa = [l for l in layer_indices if not (ocrvqa_dir / f"firing_ocrvqa_layer_{l}.json").exists()]
    if not remaining_vqa and not remaining_ocrvqa:
        print(f"[Firing GPU{gpu_id}] All layers done, skipping")
        return

    print(f"[Firing GPU{gpu_id}] Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE
    ).to(device).eval()
    nns_model = NNsight(model_raw)

    # --- Pass 1: VQA baseline (sample-level firing) ---
    if remaining_vqa:
        print(f"[Firing GPU{gpu_id}] Loading VQAv2 for baseline...")
        vqa = load_dataset("lmms-lab/VQAv2", split="validation")
        n_baseline = min(N_FIRING_SAMPLES, len(vqa))
        print(f"[Firing GPU{gpu_id}] VQA Pass: {n_baseline} samples, layers: {remaining_vqa}", flush=True)

        for layer_idx in remaining_vqa:
            ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
            sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                           device=device, cache_dir=HF_CACHE)
            sae.eval()

            # Per-token: count text tokens where each feature fires > 0
            token_fire_count = np.zeros(D_SAE, dtype=np.int64)
            total_text_tokens = 0
            n_samples = 0
            n_errors = 0

            for si in range(n_baseline):
                try:
                    sample = vqa[si]
                    image = sample["image"].convert("RGB")
                    prompt = f"answer en {sample['question']}"

                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        image, prompt, processor, model_raw, device=device
                    )
                    img_s, img_e = get_image_token_positions(input_ids)

                    with torch.no_grad():
                        with nns_model.trace(
                            input_ids=input_ids, attention_mask=attn_mask,
                            pixel_values=pixel_values, use_cache=False,
                        ) as tr:
                            layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()

                    act = layer_out.detach().squeeze(0).float()
                    with torch.no_grad():
                        codes = sae.encode(act).detach()

                    # Text-token mask
                    seq_len = codes.shape[0]
                    txt_mask = torch.ones(seq_len, dtype=torch.bool, device=codes.device)
                    if img_e > img_s:
                        txt_mask[img_s:img_e] = False

                    # Per-token: count tokens where each feature fires > 0
                    txt_codes = codes[txt_mask]  # (n_text_tokens, d_sae)
                    if txt_codes.shape[0] > 0:
                        fired_tokens = (txt_codes > 0).sum(dim=0).cpu().numpy()  # (d_sae,)
                        token_fire_count += fired_tokens.astype(np.int64)
                        total_text_tokens += txt_codes.shape[0]

                    n_samples += 1

                except Exception as e:
                    n_errors += 1
                    if n_errors <= 5:
                        print(f"[Firing GPU{gpu_id}] VQA error sample {si}: {e}", flush=True)
                    continue

                if si % 5000 == 0 and gpu_id == 0:
                    print(f"  L{layer_idx} VQA: {si}/{n_baseline}", flush=True)

            layer_data = {
                "layer": int(layer_idx),
                "n_samples": int(n_samples),
                "n_tokens": int(total_text_tokens),
                "n_errors": int(n_errors),
                "dataset": "vqa",
                "fire_count": token_fire_count.tolist(),
            }
            with open(vqa_dir / f"firing_vqa_layer_{layer_idx}.json", "w") as f:
                json.dump(layer_data, f)

            print(f"[Firing GPU{gpu_id}] VQA L{layer_idx}: {n_samples} samples, {total_text_tokens} tokens, "
                  f"features firing >1%: {(token_fire_count > total_text_tokens * 0.01).sum()}", flush=True)

            del sae
            torch.cuda.empty_cache()
            gc.collect()

    # --- Pass 2: OCR-VQA (per-token firing) — random 10K sampled rows from train ---
    if remaining_ocrvqa:
        print(f"[Firing GPU{gpu_id}] Loading OCR-VQA train + indices...")
        ocrvqa = load_dataset("howard-hou/OCR-VQA", split="train")
        with open(OCRVQA_INDICES) as f:
            ocrvqa_idx = json.load(f)   # list of {row_idx, q_idx, question, answer}
        print(f"[Firing GPU{gpu_id}] OCR-VQA Pass: {len(ocrvqa_idx)} samples, layers: {remaining_ocrvqa}", flush=True)

        for layer_idx in remaining_ocrvqa:
            ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
            sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                           device=device, cache_dir=HF_CACHE)
            sae.eval()

            token_fire_count = np.zeros(D_SAE, dtype=np.int64)
            total_text_tokens = 0
            n_samples = 0
            n_failed = 0

            for si, meta in enumerate(ocrvqa_idx):
                question = meta["question"]
                row = ocrvqa[meta["row_idx"]]
                img = row.get("image")
                if img is None or not question:
                    n_failed += 1
                    continue
                try:
                    img = img.convert("RGB")
                    prompt = f"answer en {question}"
                    input_ids, attn_mask, pixel_values = process_vlm_inputs(
                        img, prompt, processor, model_raw, device=device
                    )
                    img_s, img_e = get_image_token_positions(input_ids)

                    with torch.no_grad():
                        with nns_model.trace(
                            input_ids=input_ids, attention_mask=attn_mask,
                            pixel_values=pixel_values, use_cache=False,
                        ) as tr:
                            layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()

                    act = layer_out.detach().squeeze(0).float()
                    with torch.no_grad():
                        codes = sae.encode(act).detach()

                    seq_len = codes.shape[0]
                    txt_mask = torch.ones(seq_len, dtype=torch.bool, device=codes.device)
                    if img_e > img_s:
                        txt_mask[img_s:img_e] = False

                    txt_codes = codes[txt_mask]
                    if txt_codes.shape[0] > 0:
                        fired_tokens = (txt_codes > 0).sum(dim=0).cpu().numpy()
                        token_fire_count += fired_tokens.astype(np.int64)
                        total_text_tokens += txt_codes.shape[0]

                    n_samples += 1
                except Exception as e:
                    n_failed += 1
                    if n_failed <= 3:
                        print(f"  [GPU{gpu_id}] OCR-VQA err L{layer_idx} si={si}: {e}", flush=True)
                    continue

                if (si + 1) % 1000 == 0 and gpu_id == 0:
                    print(f"  L{layer_idx} OCR-VQA: {si+1}/{len(ocrvqa_idx)} ({n_failed} failed)", flush=True)

            layer_data = {
                "layer": int(layer_idx),
                "n_samples": int(n_samples),
                "n_tokens": int(total_text_tokens),
                "n_failed": int(n_failed),
                "dataset": "ocrvqa",
                "fire_count": token_fire_count.tolist(),
            }
            with open(ocrvqa_dir / f"firing_ocrvqa_layer_{layer_idx}.json", "w") as f:
                json.dump(layer_data, f)

            print(f"[Firing GPU{gpu_id}] OCR-VQA L{layer_idx}: {n_samples} samples, {total_text_tokens} tokens ({n_failed} failed), "
                  f"features firing >1%: {(token_fire_count > total_text_tokens * 0.01).sum()}", flush=True)

            del sae
            torch.cuda.empty_cache()
            gc.collect()

    del nns_model, model_raw, processor
    torch.cuda.empty_cache()
    gc.collect()


def step5_firing(n_gpus=N_GPUS):
    """Run VQA + VSR firing frequency computation across GPUs."""
    print("\n" + "=" * 60)
    print(f"[Step 5] Firing Frequencies (per-token, VQA {N_FIRING_SAMPLES} + VSR all)")
    print("=" * 60)

    layers_per_worker = math.ceil(N_LAYERS / n_gpus)
    assignments = []
    for w in range(n_gpus):
        start = w * layers_per_worker
        end = min(start + layers_per_worker, N_LAYERS)
        worker_layers = list(range(start, end))
        if worker_layers:
            assignments.append((w, worker_layers))

    for g, layers in assignments:
        print(f"  GPU {g}: layers {layers}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for gpu_id, layers in assignments:
        p = mp.Process(target=_firing_worker, args=(gpu_id, layers))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        print(f"[ERROR] Firing workers {failed} failed!")


# ======================== Step 6: Spatial Features ========================
# Fisher exact test on PER-TOKEN firing (VQA vs VSR).
# Contingency table per feature:
#   [c_vsr, n_vsr - c_vsr]   (VSR tokens where feature fires / doesn't)
#   [c_vqa, n_vqa - c_vqa]   (VQA tokens where feature fires / doesn't)

def step6_spatial():
    """Fisher exact test: VQA vs VSR per-token firing."""
    print("\n" + "=" * 60)
    print("[Step 6] Spatial Features (Fisher Test — VQA vs VSR, per-token)")
    print("=" * 60)

    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests
    import pandas as pd

    vqa_dir = ANALYSIS_DIR / "firing_vqa_pertoken"
    ocrvqa_dir = ANALYSIS_DIR / "firing_ocrvqa_pertoken"
    out_dir = ANALYSIS_DIR / "ocrvqa_pertoken"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer_idx in range(N_LAYERS):
        vqa_path = vqa_dir / f"firing_vqa_layer_{layer_idx}.json"
        ocrvqa_path = ocrvqa_dir / f"firing_ocrvqa_layer_{layer_idx}.json"

        if not vqa_path.exists() or not ocrvqa_path.exists():
            print(f"  L{layer_idx}: SKIP (missing data)")
            continue

        with open(vqa_path) as f:
            vqa_data = json.load(f)
        with open(ocrvqa_path) as f:
            ocrvqa_data = json.load(f)

        n_vqa = vqa_data["n_tokens"]
        fire_vqa = np.array(vqa_data["fire_count"])

        n_ocr = ocrvqa_data["n_tokens"]
        fire_ocr = np.array(ocrvqa_data["fire_count"])

        if n_vqa == 0 or n_ocr == 0:
            continue

        print(f"  Layer {layer_idx}: n_vqa_tokens={n_vqa}, n_ocrvqa_tokens={n_ocr}")

        for fi in range(D_SAE):
            c_vqa = int(fire_vqa[fi])
            c_ocr = int(fire_ocr[fi])
            if c_vqa == 0 and c_ocr == 0:
                continue
            rows.append({
                "layer": layer_idx,
                "feature": fi,
                "c_vqa": c_vqa,
                "n_vqa": n_vqa,
                "c_ocr": c_ocr,
                "n_ocr": n_ocr,
            })

    if not rows:
        print("  No features with nonzero firing counts found")
        return {"total_ocrvqa": 0}

    df = pd.DataFrame(rows)
    print(f"  {len(df)} features with nonzero firing across {df['layer'].nunique()} layers")

    # Compute frequencies and Fisher exact test (OCR-VQA vs VQA, alt='greater')
    pvals, odds = [], []
    for _, r in df.iterrows():
        c_ocr = min(int(r.c_ocr), int(r.n_ocr))
        c_vqa = min(int(r.c_vqa), int(r.n_vqa))
        table = [[c_ocr, max(0, int(r.n_ocr) - c_ocr)],
                 [c_vqa, max(0, int(r.n_vqa) - c_vqa)]]
        try:
            o, p = fisher_exact(table, alternative="greater")
            odds.append(o if not math.isinf(o) else 1e9)
            pvals.append(p)
        except ValueError:
            odds.append(1.0)
            pvals.append(1.0)

    df["odds_ratio"] = odds
    df["p_raw"] = pvals
    df["freq_ocr"] = df.c_ocr / df.n_ocr
    df["freq_vqa"] = df.c_vqa / df.n_vqa
    df["freq_diff"] = df.freq_ocr - df.freq_vqa

    # FDR-BH adjustment (matches original)
    df["p_adj"] = multipletests(df.p_raw, method="fdr_bh")[1]

    # Filter: odds_ratio >= threshold AND freq_diff >= min_diff
    keep = (df.odds_ratio >= ODDS_THR) & (df.freq_diff >= MIN_FREQ_DIFF)
    ocrvqa_feat = df.loc[keep].sort_values("odds_ratio", ascending=False).copy()

    print(f"  OCR-VQA features found: {len(ocrvqa_feat)} out of {len(df)} "
          f"({100 * len(ocrvqa_feat) / max(len(df), 1):.1f}%)")
    if len(ocrvqa_feat) > 0:
        print(f"  Per layer:")
        for layer_idx in sorted(ocrvqa_feat["layer"].unique()):
            n = (ocrvqa_feat["layer"] == layer_idx).sum()
            print(f"    L{layer_idx}: {n} features")

    ocrvqa_feat.to_csv(out_dir / "ocrvqa_features_pertoken.csv", index=False)
    df.to_csv(out_dir / "all_features_stats_pertoken.csv", index=False)

    summary = {
        "total_ocrvqa": len(ocrvqa_feat),
        "odds_threshold": ODDS_THR,
        "min_freq_diff": MIN_FREQ_DIFF,
        "total_features_tested": len(df),
        "method": "VQA vs OCR-VQA per-token (matching original LLaVA-MORE pipeline)",
    }
    with open(out_dir / "ocrvqa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ======================== Step 7: Lexical Filtering ========================

def _lexical_worker(gpu_id, feature_assignments):
    """Test candidate features with generic prompts to filter lexical artifacts.

    Matches original probe_vqa_spatial_features.py:
    1. Find top-activating samples for each feature (scan VQA spatial subset)
    2. Replace original questions with generic spatial prompts
    3. Feature passes if ALL top samples still activate (max over all tokens > threshold)
    4. Feature fails if ANY top sample drops below threshold → lexical artifact
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset

    out_dir = ANALYSIS_DIR / "lexical"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments:
        return

    print(f"[Lexical GPU{gpu_id}] {len(feature_assignments)} features to test", flush=True)
    print(f"[Lexical GPU{gpu_id}] Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE
    ).to(device).eval()
    nns_model = NNsight(model_raw)

    # For OCR-VQA, scan our 10K-sample slice of OCR-VQA (the same set used in firing).
    # This is the analog of VSR's spatial-VQA-subset scan.
    ocrvqa = load_dataset("howard-hou/OCR-VQA", split="train")
    with open(OCRVQA_INDICES) as f:
        ocrvqa_idx = json.load(f)
    print(f"[Lexical GPU{gpu_id}] {len(ocrvqa_idx)} OCR-VQA samples available", flush=True)

    # Group by layer
    layer_features = defaultdict(list)
    for (layer_idx, feature_idx) in feature_assignments:
        layer_features[layer_idx].append(feature_idx)

    results = []
    top_k = 5             # number of top-activating samples to test (matches original default)
    scan_samples = 500    # scan this many spatial samples to find top activators
    activation_threshold = 0.01

    for layer_idx in sorted(layer_features.keys()):
        features = layer_features[layer_idx]

        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()

        # --- Phase 1: Scan OCR-VQA samples to find top activators per feature ---
        feature_set = set(features)
        feature_cols = {f: [] for f in features}  # feature -> list of (max_act, scan_i)

        scan_count = min(scan_samples, len(ocrvqa_idx))
        print(f"[Lexical GPU{gpu_id}] L{layer_idx}: scanning {scan_count} OCR-VQA samples "
              f"for {len(features)} features...", flush=True)

        for scan_i in range(scan_count):
            try:
                meta = ocrvqa_idx[scan_i]
                sample = ocrvqa[meta["row_idx"]]
                image = sample["image"].convert("RGB")
                question = meta["question"]
                prompt = f"answer en {question}"

                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    image, prompt, processor, model_raw, device=device
                )
                with torch.no_grad():
                    with nns_model.trace(
                        input_ids=input_ids, attention_mask=attn_mask,
                        pixel_values=pixel_values, use_cache=False,
                    ) as tr:
                        layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()

                act = layer_out.detach().squeeze(0).float()
                with torch.no_grad():
                    codes = sae.encode(act.to(device)).detach().cpu()

                for f in features:
                    max_act = float(codes[:, f].max().item())
                    if max_act > 0:
                        feature_cols[f].append((max_act, scan_i))
            except Exception:
                continue

        # --- Phase 2: Test top-activating samples with generic prompts ---
        for feature_idx in features:
            # Sort by activation magnitude, take top-k
            candidates = sorted(feature_cols[feature_idx], key=lambda x: -x[0])[:top_k]

            if not candidates:
                # Feature never fired in scan — fail
                results.append({
                    "layer": layer_idx, "feature": feature_idx,
                    "passed": False, "n_tested": 0,
                })
                continue

            passed = True
            n_tested = 0

            for (mag, scan_i) in candidates:
                try:
                    meta = ocrvqa_idx[scan_i]
                    sample = ocrvqa[meta["row_idx"]]
                    image = sample["image"].convert("RGB")

                    # Test all generic prompts, take best activation (matches original)
                    best_generic_max = 0.0
                    for raw_prompt in GENERIC_PROMPTS:
                        prompt = f"answer en {raw_prompt}"
                        input_ids, attn_mask, pixel_values = process_vlm_inputs(
                            image, prompt, processor, model_raw, device=device
                        )

                        with torch.no_grad():
                            with nns_model.trace(
                                input_ids=input_ids, attention_mask=attn_mask,
                                pixel_values=pixel_values, use_cache=False,
                            ) as tr:
                                layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()

                        act = layer_out.detach().squeeze(0).float()
                        with torch.no_grad():
                            codes = sae.encode(act.to(device)).detach().cpu()

                        # Max over ALL tokens (matches original max_all check)
                        max_all = float(codes[:, feature_idx].max().item())
                        best_generic_max = max(best_generic_max, max_all)

                    n_tested += 1

                    # If ANY top sample fails generic test → feature is lexical artifact
                    if best_generic_max <= activation_threshold:
                        passed = False
                        break

                except Exception:
                    continue

            results.append({
                "layer": layer_idx,
                "feature": feature_idx,
                "passed": passed,
                "n_tested": n_tested,
            })

            status = "PASS" if passed else "FAIL"
            print(f"[Lexical GPU{gpu_id}] L{layer_idx} F{feature_idx}: {status} "
                  f"(tested {n_tested}/{len(candidates)} top samples)", flush=True)

        del sae
        torch.cuda.empty_cache()

    out_path = out_dir / f"lexical_results_w{gpu_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    passed_count = sum(1 for r in results if r["passed"])
    print(f"[Lexical GPU{gpu_id}] {passed_count}/{len(results)} passed", flush=True)


def step7_lexical(n_gpus=N_GPUS):
    """Run lexical filtering on spatial candidate features (step 6 output).

    Per the paper: test spatial candidates with generic prompts, then step 8
    intersects adapted ∩ spatial ∩ lexical-passed.
    """
    print("\n" + "=" * 60)
    print("[Step 7] Lexical Artifact Filtering")
    print("=" * 60)

    # Load OCR-VQA candidates from step 6 (per-token results)
    csv_path = ANALYSIS_DIR / "ocrvqa_pertoken" / "ocrvqa_features_pertoken.csv"
    if not csv_path.exists():
        print("  No OCR-VQA features found. Run step 6 first.")
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    candidates = [(int(r["layer"]), int(r["feature"])) for _, r in df.iterrows()]
    print(f"  {len(candidates)} OCR-VQA candidates to test")

    if not candidates:
        return

    (ANALYSIS_DIR / "lexical").mkdir(parents=True, exist_ok=True)

    features_per_worker = math.ceil(len(candidates) / n_gpus)
    assignments = []
    for w in range(n_gpus):
        start = w * features_per_worker
        end = min(start + features_per_worker, len(candidates))
        worker_features = candidates[start:end]
        if worker_features:
            assignments.append((w, worker_features))

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for gpu_id, feats in assignments:
        p = mp.Process(target=_lexical_worker, args=(gpu_id, feats))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


# ======================== Step 8: Intersection ========================

def step8_intersection():
    """Compute adapted ∩ spatial ∩ lexical-filtered feature sets."""
    print("\n" + "=" * 60)
    print("[Step 8] Feature Intersection")
    print("=" * 60)

    import pandas as pd

    out_dir = ANALYSIS_DIR / "final_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load adapted
    adapted_path = ANALYSIS_DIR / "adapted" / "adapted_features_results.csv"
    adapted_by_layer = {}
    if adapted_path.exists():
        with open(adapted_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer"])
                indices = ast.literal_eval(row["adapted_indices"])
                adapted_by_layer[layer] = set(indices)
    print(f"  Adapted: {sum(len(v) for v in adapted_by_layer.values())} features")

    # Load OCR-VQA Fisher (per-token) candidates
    ocrvqa_path = ANALYSIS_DIR / "ocrvqa_pertoken" / "ocrvqa_features_pertoken.csv"
    ocrvqa_by_layer = {}
    if ocrvqa_path.exists():
        df = pd.read_csv(ocrvqa_path)
        for layer in df["layer"].unique():
            ocrvqa_by_layer[int(layer)] = set(df[df["layer"] == layer]["feature"].tolist())
    print(f"  OCR-VQA Fisher: {sum(len(v) for v in ocrvqa_by_layer.values())} features")

    # Load lexical results
    lexical_dir = ANALYSIS_DIR / "lexical"
    lexical_passed = {}
    if lexical_dir.exists():
        for f_path in sorted(lexical_dir.glob("lexical_results_w*.json")):
            with open(f_path) as f:
                worker_results = json.load(f)
            for r in worker_results:
                if r["passed"]:
                    layer = r["layer"]
                    if layer not in lexical_passed:
                        lexical_passed[layer] = set()
                    lexical_passed[layer].add(r["feature"])
    print(f"  Lexical passed: {sum(len(v) for v in lexical_passed.values())} features")

    # Intersection
    all_layers = sorted(set(adapted_by_layer.keys()) | set(ocrvqa_by_layer.keys()))
    final_features = []

    for layer in all_layers:
        adapted = adapted_by_layer.get(layer, set())
        ocr_set = ocrvqa_by_layer.get(layer, set())
        common = adapted & ocr_set

        if lexical_passed:
            lex = lexical_passed.get(layer, set())
            common = common & lex

        for fi in sorted(common):
            final_features.append({"layer": layer, "feature": fi})

        if common:
            print(f"  L{layer}: adapted={len(adapted)}, ocrvqa={len(ocr_set)}, "
                  f"lexical={len(lexical_passed.get(layer, set()))}, final={len(common)}")

    if final_features:
        df_final = pd.DataFrame(final_features)
        df_final.to_csv(out_dir / "final_ocrvqa_features.csv", index=False)

    summary = {
        "total_final": len(final_features),
        "total_adapted": sum(len(v) for v in adapted_by_layer.values()),
        "total_ocrvqa": sum(len(v) for v in ocrvqa_by_layer.values()),
        "total_lexical_passed": sum(len(v) for v in lexical_passed.values()),
    }
    with open(out_dir / "intersection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Final: {len(final_features)} features")
    return summary


# ======================== Main ========================

def main():
    global N_FIRING_SAMPLES, CHECKPOINT_DIR, ANALYSIS_DIR

    parser = argparse.ArgumentParser(description="Local analysis pipeline")
    parser.add_argument("--step", type=int, nargs="+", default=None,
                        help="Specific steps to run (default: all 1-8)")
    parser.add_argument("--gpus", type=int, default=N_GPUS)
    parser.add_argument("--n-firing-samples", type=int, default=N_FIRING_SAMPLES,
                        help="Number of samples for step 5 firing (default: 50000)")
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR),
                        help="SAE checkpoint directory (default: final checkpoints)")
    parser.add_argument("--analysis-dir", type=str, default=str(ANALYSIS_DIR),
                        help="Analysis output directory (default: analysis/)")
    args = parser.parse_args()

    N_FIRING_SAMPLES = args.n_firing_samples
    # Set env vars BEFORE updating local globals so spawned workers re-import with overrides
    os.environ["VLMSCOPE_CKPT_DIR"] = args.checkpoint_dir
    os.environ["VLMSCOPE_ANALYSIS_DIR"] = args.analysis_dir
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    ANALYSIS_DIR = Path(args.analysis_dir)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Config] CHECKPOINT_DIR={CHECKPOINT_DIR}")
    print(f"[Config] ANALYSIS_DIR={ANALYSIS_DIR}")
    print(f"[Config] N_FIRING_SAMPLES={N_FIRING_SAMPLES}")
    steps = args.step if args.step else [1, 2, 3, 4, 5, 6, 7, 8]

    t0 = time.time()

    if 1 in steps:
        step1_fvu_table()
    if 2 in steps:
        step2_cosine()
    if 3 in steps:
        step3_energy(n_gpus=args.gpus)
    if 4 in steps:
        step4_adapted()
    if 5 in steps:
        step5_firing(n_gpus=args.gpus)
    if 6 in steps:
        step6_spatial()

    # Step 7 reads spatial candidates from step 6 directly, then step 8 intersects all three
    if 7 in steps:
        step7_lexical(n_gpus=args.gpus)
    if 8 in steps:
        step8_intersection()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"[DONE] Analysis complete in {elapsed / 3600:.1f}h")
    print(f"  Results: {ANALYSIS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
