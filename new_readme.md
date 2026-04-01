# VLM Scope: Sparse Autoencoder Analysis of Vision-Language Models

Pipeline for training and analyzing Sparse Autoencoders (SAEs) on Vision-Language Models to identify adapted, spatial, and visually-grounded features. Supports two VLM backends:

- **PaliGemma2-3B** (new) — Gemma 2 backbone, Gemma Scope base SAEs
- **LLaVA-MORE** (original) — Llama 3.1 backbone, Llama Scope base SAEs

---

## Repository Structure

```
vlm_scope_backup/
  vlm_scope/
    finetune/
      paligemma2/              # PaliGemma2-3B pipeline (Modal cloud)
        utils.py               # Model loading, SAE init (TopK + JumpReLU)
        modal_train.py         # Phase 1+2: cache activations + train TopK SAEs (8 GPU)
        modal_train_jumprelu.py # Train JumpReLU SAEs from Gemma Scope init (8 GPU)
        modal_analysis.py      # Full analysis pipeline Steps 1-8 (8 GPU)
        modal_val_fvu.py       # Validation FVU with Full/Image/Text breakdown (8 GPU)
        modal_upload_hf.py     # Upload checkpoints to HuggingFace
        cache_activations.py   # Standalone activation caching script
        train_sae.py           # Standalone SAE training script
        orchestrate.py         # Local orchestration helper
      vqa/                     # LLaVA-MORE VQA fine-tuning (original)
      instruct/                # Instruction tuning experiments
      experiments/             # Experimental scripts

vision-language-scope/         # Original LLaVA-MORE analysis codebase
  features/
    adapted/                   # Adapted feature metrics (Ev, cosine, selection)
    spatial/                   # Spatial feature identification (firing, Fisher test)
    hallucination/             # Hallucination feature analysis
    ocr/                       # OCR feature analysis
  ablation/                    # Feature ablation experiments
  experiments/                 # Attribution patching
  utils/                       # Shared utilities (SAE loading, data loading)
  finetune/vqa/                # LLaVA-MORE SAE training
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

**JumpReLU SAE** (custom, matching Gemma Scope)
- Activation: `x * (x > threshold)` with learnable per-feature threshold
- Uses Straight-Through Estimator (STE) gradients via rectangle function
- Sparsity: targets L0 ≈ 50 via penalty `coeff * ((L0/target) - 1)^2`
- Properly loads Gemma Scope weights **including threshold parameter**
- No architecture mismatch — recommended for Gemma Scope initialization

### Prerequisites

```bash
pip install modal
modal setup  # or: modal token set
```

Configure Modal profile if needed:
```bash
export MODAL_PROFILE=your-profile
```

### Step 1: Train SAEs

All training runs on Modal cloud GPUs. Activations are cached first (Phase 1), then SAEs are trained (Phase 2). Both phases use 8 GPUs in parallel via `starmap`.

**TopK SAE Training:**
```bash
cd vlm_scope/finetune/paligemma2
MODAL_PROFILE=hunar-oxford modal run modal_train.py
```

**JumpReLU SAE Training** (recommended for Gemma Scope init):
```bash
cd vlm_scope/finetune/paligemma2
MODAL_PROFILE=hunar-oxford modal run modal_train_jumprelu.py
```

Training configuration (both):
- Dataset: 50,000 VQAv2 training + 5,000 validation samples
- Chunk size: 1,000 samples
- All 26 layers
- Methods: `pretrained` (Gemma Scope init) + `random`
- Training batch size: 8
- Intermediate checkpoints at 25%, 50%, 75%

TopK-specific: k=50, LR = 2e-4 / sqrt(d_sae/16384)
JumpReLU-specific: target_l0=50, bandwidth=0.001, LR=7e-5, Adam betas=(0.0, 0.999)

**Output on Modal Volume** (`vlm-scope-data-v2`):
```
/vol/results/paligemma2/
  run/                          # TopK results
    activations/                # 55 H5 files (~3.5 TB)
    checkpoints/                # {method}_layer_{i}.pt
    checkpoint_{25,50,75}pct/   # Intermediate checkpoints
    logs/                       # metrics_{method}_layer_{i}.csv
  run_jumprelu/                 # JumpReLU results
    checkpoints/
    checkpoint_{25,50,75}pct/
    logs/
```

### Step 2: Compute Validation FVU

Computes FVU broken down by token type (Full / Image / Text) on held-out validation set, matching the paper's Table 1 format.

```bash
MODAL_PROFILE=hunar-oxford modal run modal_val_fvu.py
```

Uses 8 GPUs, each computing FVU for ~3-4 layers across both methods.

**Output:** `analysis/val_fvu/val_fvu_table.csv`, `val_fvu_summary.json`

### Step 3: Run Analysis Pipeline

Computes adapted features, spatial features, lexical filtering, and intersection — all on Modal using cached activations.

```bash
MODAL_PROFILE=hunar-oxford modal run modal_analysis.py
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

**Output:** `analysis/{cosines,energy,adapted,firing,spatial,lexical,final_features}/`

### Step 4: Upload to HuggingFace

```bash
MODAL_PROFILE=hunar-oxford modal run modal_upload_hf.py
```

Uploads SAE checkpoints, intermediate checkpoints, and training logs to HF repo `hunarbatra/vlm_scope_paligemma2_sae`.

### Key Files

**`utils.py`** — Core utilities:
- `initialize_vlm_model()` — Load PaliGemma2-3B + processor
- `process_vlm_inputs(image, prompt, processor, model)` → `(input_ids, attention_mask, pixel_values)`
- `get_image_token_positions(input_ids)` → `(start, end)` of image token span
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
  analysis/                     # Analysis outputs
    val_fvu/                    # Validation FVU tables
    cosines/                    # Per-layer cosine similarity arrays
    energy/                     # Per-layer Ev/Et arrays
    adapted/                    # Adapted feature selections
    firing/                     # Per-layer firing frequency JSONs
    spatial/                    # Spatial feature CSVs
    lexical/                    # Lexical filtering results
    final_features/             # Final intersection results
```
