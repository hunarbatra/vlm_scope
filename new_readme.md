# VLM Scope: Sparse Autoencoder Analysis of Vision-Language Models

Pipeline for training and analyzing Sparse Autoencoders (SAEs) on Vision-Language Models to identify adapted, spatial, and visually-grounded features. Supports two VLM backends:

- **PaliGemma2-3B** — Gemma 2 backbone, Gemma Scope base SAEs
- **LLaVA-MORE** (original) — Llama 3.1 backbone, Llama Scope base SAEs

---

## Repository Structure

```
vlm_scope/
  finetune/
    paligemma2/              # PaliGemma2-3B pipeline
      utils.py               # Model loading, SAE init (TopK + JumpReLU)
      modal_train.py         # Modal: cache activations + train TopK SAEs (8 GPU)
      modal_train_jumprelu.py # Modal: train JumpReLU SAEs (8 GPU)
      modal_analysis.py      # Modal: full analysis pipeline Steps 1-8 (8 GPU)
      modal_val_fvu.py       # Modal: validation FVU with Full/Image/Text breakdown
      modal_upload_hf.py     # Modal: upload checkpoints to HuggingFace
      cache_activations.py   # Standalone: activation caching (local/any cluster)
      train_sae.py           # Standalone: SAE training (local/any cluster)
      orchestrate.py         # Standalone: local orchestration helper
    vqa/                     # LLaVA-MORE VQA fine-tuning (original)
    instruct/                # Instruction tuning experiments
    experiments/             # Experimental scripts
```

---

## Setup

### Environment

```bash
pip install torch>=2.1 transformers>=4.44 sae-lens>=4.0 nnsight>=0.3 \
    h5py tqdm huggingface-hub Pillow numpy scipy statsmodels pandas accelerate datasets
```

### Configuration

Copy `.env.template` to `.env` and fill in your tokens:
```bash
cp .env.template .env
# Edit .env with your HuggingFace token
```

### Modal Setup (optional, for cloud GPU training)

```bash
pip install modal
modal setup  # or: modal token set
export MODAL_PROFILE=your-profile  # if using named profiles
```

---

## PaliGemma2-3B Pipeline

### Model Details

| Parameter | Value |
|-----------|-------|
| Model | `google/paligemma2-3b-pt-224` |
| Backbone | Gemma 2 2B |
| Decoder layers | 26 |
| Hidden dim (d_in) | 2304 |
| Image tokens | 256 (224x224 input) |
| NNsight hook | `model.language_model.layers[i]` |
| Base SAE | Gemma Scope 2B (`google/gemma-scope-2b-pt-res`, width 16k) |
| SAE width (d_sae) | 16,384 |

### SAE Architectures

Two SAE architectures are supported:

**TopK SAE** (from sae-lens)
- Activation: select top-k features by magnitude
- k=50 (matching Llama Scope's architecture)
- Note: Gemma Scope natively uses JumpReLU, so TopK init from Gemma Scope has architecture mismatch

**JumpReLU SAE** (custom, matching Gemma Scope) — **recommended**
- Activation: `x * (x > threshold)` with learnable per-feature threshold
- Uses Straight-Through Estimator (STE) gradients via rectangle function
- Sparsity: targets L0 ~ 50 via penalty `coeff * ((L0/target) - 1)^2`
- Properly loads Gemma Scope weights **including threshold parameter**
- No architecture mismatch — recommended for Gemma Scope initialization

### Step 1: Cache Activations

Activations must be cached before training. This runs the VLM forward pass through NNsight hooks and saves per-layer residual stream activations to H5 files.

#### Option A: Modal (cloud GPUs)

Activation caching is included as Phase 1 of `modal_train.py`. It automatically runs before training.

```bash
cd finetune/paligemma2
MODAL_PROFILE=your-profile modal run modal_train.py
```

Phase 1 caches 55,000 samples (50k train + 5k val) across all 26 layers into H5 files.

#### Option B: Standalone (local or any cluster)

For running on a local GPU or non-Modal cluster:

```bash
cd finetune/paligemma2

# Cache activations for all layers
python cache_activations.py \
  --model_name google/paligemma2-3b-pt-224 \
  --output_dir /path/to/activations \
  --n_samples 55000 \
  --chunk_size 1000 \
  --from_layer 0 \
  --to_layer 26 \
  --device cuda
```

This produces 55 H5 files in the output directory (`chunk_0_1000.h5`, `chunk_1000_2000.h5`, ...).

#### Option C: Orchestrator (local multi-step)

The orchestrator runs caching + training as a single pipeline:

```bash
cd finetune/paligemma2

python orchestrate.py \
  --output_dir /path/to/results \
  --n_samples 55000 \
  --device cuda
```

### Step 2: Train SAEs

#### Option A: Modal (8-GPU parallel)

**TopK SAE Training:**
```bash
cd finetune/paligemma2
MODAL_PROFILE=your-profile modal run modal_train.py
```

**JumpReLU SAE Training** (recommended for Gemma Scope init):
```bash
cd finetune/paligemma2
MODAL_PROFILE=your-profile modal run modal_train_jumprelu.py
```

Both scripts distribute 26 layers across 8 GPUs via `modal.starmap`.

#### Option B: Standalone (local or any cluster)

```bash
cd finetune/paligemma2

# Train TopK SAE for a specific layer
python train_sae.py \
  --activations_dir /path/to/activations \
  --output_dir /path/to/results \
  --layer 0 \
  --method pretrained \
  --n_training_samples 50000 \
  --chunk_size 1000 \
  --device cuda

# Train all layers sequentially
for layer in $(seq 0 25); do
  python train_sae.py \
    --activations_dir /path/to/activations \
    --output_dir /path/to/results \
    --layer $layer \
    --method pretrained \
    --device cuda
done
```

For JumpReLU training without Modal, the training loop in `modal_train_jumprelu.py:train_worker()` can be adapted — the core logic uses `initialize_jumprelu_sae()` from `utils.py` and operates on cached H5 activations.

#### Training Configuration

| Parameter | TopK | JumpReLU |
|-----------|------|----------|
| k / target L0 | k=50 | target_l0=50 |
| Learning rate | 2e-4 / sqrt(d_sae/16384) | 7e-5 |
| Optimizer | Adam | Adam (betas=0.0, 0.999) |
| Bandwidth | — | 0.001 |
| Sparsity coeff | — | 1.0 |
| Batch size | 8 | 8 |
| Dataset | 50k VQAv2 train + 5k val | 50k VQAv2 train + 5k val |
| Intermediate ckpts | 25%, 50%, 75% | 25%, 50%, 75% |

### Step 3: Compute Validation FVU

Computes FVU broken down by token type (Full / Image / Text) on held-out validation set, matching the paper's Table 1 format.

Configure `SAE_TYPE` at the top of the script: `"topk"` or `"jumprelu"`.

#### Modal

```bash
cd finetune/paligemma2
MODAL_PROFILE=your-profile modal run modal_val_fvu.py
```

Uses 8 GPUs, each computing FVU for ~3-4 layers.

**Output:** `analysis/val_fvu_jumprelu/val_fvu_table.csv`, `val_fvu_summary.json`

### Step 4: Run Analysis Pipeline

Computes adapted features, spatial features, lexical filtering, and intersection — all on Modal using cached activations.

Configure `SAE_TYPE` at the top of `modal_analysis.py`: `"topk"` or `"jumprelu"`.

```bash
cd finetune/paligemma2
MODAL_PROFILE=your-profile modal run modal_analysis.py
```

**Pipeline steps (automated):**

| Step | Name | GPUs | Description |
|------|------|------|-------------|
| 1 | FVU Table | 0 | Extract final FVU from training logs |
| 2 | Cosine Similarity | 8 | Base vs finetuned W_dec per feature |
| 3 | Visual Energy Ev | 8 | Per-feature image vs text energy ratio |
| 4 | Adapted Features | 0 | Select Ev > epsilon AND low cosine |
| 5 | Firing Frequencies | 8 | Per-feature firing on VQA all + spatial subset |
| 6 | Spatial Features | 0 | Fisher exact test + odds ratio |
| 7 | Lexical Filtering | 8 | Generic prompt test to remove text artifacts |
| 8 | Intersection | 0 | Adapted ∩ Spatial ∩ Lexical-filtered |

**Output:** `analysis/{cosines,energy,adapted,firing,spatial,lexical,final_features}_jumprelu/`

### Step 5: Upload to HuggingFace

```bash
cd finetune/paligemma2
MODAL_PROFILE=your-profile modal run modal_upload_hf.py
```

Uploads SAE checkpoints, intermediate checkpoints, and training logs to HF repo `hunarbatra/vlm_scope_paligemma2_sae`.

### Regenerating H5 Activations

If you move to a new cluster or need to regenerate the cached activations (e.g., after losing the Modal volume):

**On Modal:**
Run `modal_train.py` — Phase 1 automatically caches all activations before training begins. If checkpoints already exist, Phase 2 will resume from them.

**On any cluster with GPU:**
```bash
cd finetune/paligemma2

# Full regeneration: 55k samples, all 26 layers
python cache_activations.py \
  --model_name google/paligemma2-3b-pt-224 \
  --output_dir /path/to/activations \
  --n_samples 55000 \
  --chunk_size 1000 \
  --from_layer 0 \
  --to_layer 26 \
  --device cuda
```

Requirements: ~40GB GPU memory (bfloat16 PaliGemma2-3B + NNsight hooks). Output: ~3.5TB of H5 files.

Once activations are cached, all analysis scripts (`modal_analysis.py` steps 2-8) and training scripts can operate on them without the VLM.

### Key Files

**`utils.py`** — Core utilities:
- `initialize_vlm_model()` — Load PaliGemma2-3B + processor
- `process_vlm_inputs(image, prompt, processor, model)` -> `(input_ids, attention_mask, pixel_values)`
- `get_image_token_positions(input_ids)` -> `(start, end)` of image token span
- `initialize_sae(layer_idx, ...)` — TopK SAE (sae-lens)
- `initialize_jumprelu_sae(layer_idx, ...)` — JumpReLU SAE (custom)
- `JumpReLUSAE` — JumpReLU autoencoder class with STE training support

### H5 Activation Format

Cached activations are stored in HDF5 files:
```
chunk_{start}_{end}.h5
  layer_{i}/
    sample_{j}          # shape: (seq_len, 2304), dtype: float32
      attrs:
        img_start: int  # start index of image tokens
        img_end: int    # end index of image tokens
```

---

## LLaVA-MORE Pipeline (Original)

### Model Details

| Parameter | Value |
|-----------|-------|
| Model | LLaVA-MORE (Llama 3.1 8B backbone) |
| Decoder layers | 32 |
| Hidden dim (d_in) | 4096 |
| Image tokens | 575 |
| NNsight hook | `model.layers[i]` |
| Base SAE | Llama Scope (`OpenMOSS-Team/Llama-Scope`, TopK, width 32k) |

### Key Differences from PaliGemma2

| Aspect | LLaVA-MORE | PaliGemma2 |
|--------|------------|------------|
| Model loading | `initialize_vlm_model("llava-more")` | `initialize_vlm_model("google/paligemma2-3b-pt-224")` |
| Input format | `(ids, mask, image_tensor, image_sizes)` | `(ids, mask, pixel_values)` |
| Hook path | `model.layers[i]` | `model.language_model.layers[i]` |
| Base SAE type | TopK (Llama Scope) | JumpReLU (Gemma Scope) |
| SAE loading | `SAE.from_pretrained("llama_scope_lxr_8x")` | `initialize_sae()` / `initialize_jumprelu_sae()` |
| Layers | 32 | 26 |

### Analysis Scripts (in `vision-language-scope/`)

**Adapted Features:**
```bash
# Compute Ev and Et per feature
python features/adapted/compute_feature_metrics.py \
  --sae_dir /path/to/checkpoints/text-only \
  --text_data /path/to/text_activations.h5 \
  --vlm_data /path/to/vlm_activations.h5

# Select adapted features
python features/adapted/select_and_plot_adapted_features.py \
  --metrics_dir results/metrics/ \
  --epsilon 0.01 --cosine_percentile 25.0
```

**Spatial Features:**
```bash
# Track firing frequencies
python features/spatial/track_firing_vqa.py \
  --sae-checkpoint-dir /path/to/checkpoints \
  --method text-only --from-layer 0 --to-layer 32

# Identify spatial features
python features/spatial/find_spatial_features.py \
  --vqa-json results/vqa_firing.json \
  --vsr-json results/vsr_firing.json \
  --odds-thr 3 --min-diff 0.005

# Lexical artifact filtering
python scripts/filter_common_features_generic.py
```

**Intersection:**
```bash
python features/spatial/find_adapted_spatial.py \
  --spatial-file results/spatial_features.csv \
  --adapted-file results/adapted_features.csv
```

**Ablation:**
```bash
python ablation/ablate_sae_feature_vqa.py \
  --pairs L13F16873 \
  --sae-checkpoint-dir /path/to/checkpoints/text-only \
  --apply-to-all-layers --max-samples 1000
```

---

## Paper Methodology Reference

The analysis follows the paper's methodology (Sections 4.2-4.3):

1. **SAE Adaptation (Section 4.2):**
   - Train SAEs on VLM activations from VQAv2
   - Compare pretrained (Scope init) vs random initialization
   - Measure FVU across layers

2. **Adapted Feature Identification (Section 4.2):**
   - Cosine similarity between base and finetuned decoder weights (low = more adapted)
   - Visual energy Ev = image activation energy / total energy (high = visually relevant)
   - Select: Ev > epsilon AND cosine in bottom percentile

3. **Spatial Feature Identification (Section 4.3):**
   - Compare firing frequencies: general VQA vs spatial-question subset
   - Fisher exact test + odds ratio for statistical significance
   - Lexical artifact filtering: replace questions with generic prompts, keep features that still fire on image tokens

4. **Final Feature Set:**
   - Intersection: adapted ∩ spatial ∩ lexical-filtered
   - Ablation experiments to measure causal impact on VQA accuracy
   - Steering experiments to amplify spatial reasoning

5. **Training Dynamics:**
   - Compare intermediate checkpoints (25/50/75/100%) for pretrained method
   - Track how features evolve during training

---

## HuggingFace Repos

- **PaliGemma2 SAEs**: `hunarbatra/vlm_scope_paligemma2_sae`
  - `pretrained/` — TopK SAE checkpoints (Gemma Scope init)
  - `random/` — TopK SAE checkpoints (random init)
  - `jumprelu/` — JumpReLU SAE checkpoints (Gemma Scope init, architecture-matched)
  - `intermediate/{25,50,75}pct/` — Training dynamics checkpoints
  - `logs/` — Training metrics CSVs

---

## Modal Volume Structure

Volume: `vlm-scope-data-v2`

```
/vol/results/paligemma2/
  run/                          # TopK SAE run
    activations/                # 55 H5 files, ~3.5 TB total
    checkpoints/                # 52 model + 52 optimizer .pt files
    checkpoint_{25,50,75}pct/   # Intermediate checkpoints
    logs/                       # 52 CSV training logs
  run_jumprelu/                 # JumpReLU SAE run
    checkpoints/
    checkpoint_{25,50,75}pct/
    logs/
  analysis/                     # Analysis outputs (suffixed _jumprelu for JumpReLU)
    val_fvu_jumprelu/           # Validation FVU tables
    cosines_jumprelu/           # Per-layer cosine similarity arrays
    energy_jumprelu/            # Per-layer Ev/Et arrays
    adapted_jumprelu/           # Adapted feature selections
    firing_jumprelu/            # Per-layer firing frequency JSONs
    spatial_jumprelu/           # Spatial feature CSVs
    lexical_jumprelu/           # Lexical filtering results
    final_features_jumprelu/    # Final intersection results
```
