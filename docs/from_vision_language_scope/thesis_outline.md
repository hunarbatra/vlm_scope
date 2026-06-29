### Title Page

- Thesis title: Cross-Modal Feature Specialization in Vision-Language Models via Sparse Autoencoders
- Author, affiliation, supervisors
- Date

### Abstract

- 1–2 paragraphs summarizing the goal: fine-tuning SAEs on VLM activations for VQA, discovering cross-modal and spatially specialized features, and validating them via attribution.
- Key methods: activation caching, SAE fine-tuning variants, geometry divergence, image/text activation analysis, reconstruction gaps, firing statistics on VQA/VSR, Fisher tests with FDR, direct logit attribution with attention heatmaps.
- Principal findings at a glance (bullet points): rotated features, spatially biased features, validation highlights.

### Acknowledgements (optional)

### Table of Contents


## 1. Introduction

- Problem motivation: understanding how VLM fine-tuning alters internal feature geometry and induces spatial reasoning capabilities.
- Research questions
  todo
- Contributions
  todo
- Summary of results and insights (high-level bullets).

## 2. Background and Related Work

### 2.1 Conceptual Background

- Vision–Language Models (VLMs): LLaVA family
  - A vision encoder (e.g., CLIP ViT) produces visual token embeddings that a projector maps into the LLM embedding space. Projected image tokens form a contiguous span [img_start, img_end) within the sequence and are processed jointly with text tokens by a shared transformer.

- Transformer residual stream and feature directions
  DLA and Attribution Patching?

- Sparse Autoencoders (SAEs) for interpretability
  - SAEs map residual activations to sparse codes and reconstruct them via a linear decoder; decoder columns are interpreted as candidate feature directions. Key notions: reconstruction quality (e.g., FVU) and sparsity of codes.

- Datasets (context only)
  - VQAv2: question–image pairs with short answers (we use the validation split).
  - VSR: caption-like statements about spatial relations; complements VQA for probing spatial behavior.

Operational details (masking rules, activation tracing, statistical testing, thresholds) are specified in Section 3 (Methods).

### 2.2 Related Work

- SAEs and feature geometry in LMs/VLMs: monosemanticity, superposition, and rotations/drift from fine-tuning.
- Cross-modal alignment in LLaVA-like VLMs: projector design and image–text token interplay.
- Spatial reasoning datasets and probing: capabilities vs dataset biases.
- Attribution in transformers: logit lens, direct logit attribution; attention visualization best practices and pitfalls.
- Multiple-hypothesis testing in large-scale feature selection: Fisher exact tests with FDR control.

## 3. Methods and System Design

### 3.1 Pipeline Overview

- Four-stage pipeline 
  - Stage 1: Fine-tuning SAEs on cached VLM activations.
  - Stage 2: Cross-modal suspect feature discovery.
  - Stage 3: Spatial feature detection across datasets.
  - Stage 4: Direct logit attribution, Attribution Patching and Ablation.

### 3.2 Data Processing and Activation Caching

- VLM input processing: image preprocessing, text tokenization, image token spans.
- Activation tracing per layer using NNsight; storing activations in HDF5 with per-sample variable length.
- Metadata in HDF5: `img_start`, `img_end` attributes; sequence length handling; `NUM_VIS_TOKENS`.
- Chunked caching for scalability; parameters: sample ranges, batch sizes, layer ranges.
- Output organization and file naming conventions.

### 3.3 SAE Training Regimes and Evaluation

- Methods: pretrained, random, text-only, image-only; masking strategies based on image token spans.
- Training configuration: batch sizes, optimizer, chunk-wise processing, resumption, logging (Weights & Biases).
- Metrics
  - Reconstruction loss; Fraction of Variance Unexplained (FVU); sparsity.
  - Validation cadence and per-layer logging.
- Checkpointing and organization by method; handling optimizer states.

### 3.4 Geometry Divergence Analysis

- Comparing encoder/decoder weights between LLM and VLM SAEs; feature-wise cosine similarities and L2 deltas.
- Layer-wise summaries: means, stds, counts below thresholds; histograms and comparisons across methods.
- Definition of “rotated features” via cosine thresholds.

### 3.5 Image/Text Activation Pattern Analysis

- Loading per-token activations with positions; building image/text masks from `img_start`, `img_end`.
- Selecting rotated features and measuring activation energy on image vs text tokens.
- Aggregations and visualizations: distributions, per-layer ratios, significance tests (if any).

### 3.6 Reconstruction Gap Analysis

- Evaluating normalized MSE for SAE reconstructions across domains and model types (LM vs VLM SAEs, text vs VLM activations).
- Defining gaps: cross-domain within-SAE and cross-SAE within-domain.
- Layer-wise trends and interpretation.

### 3.7 Feature Firing Statistics (VQA and VSR)

- Per-layer SAE inference on batched activations; masks for base/text-only/image-only.
- Counters and outputs
  - Feature firing counts, image-only vs text-only firing, activation sums, totals.
  - Sample-level top activations per feature for qualitative analysis.
  - JSON schema: `feature_firing_frequencies`, totals; auxiliary `.pt` basic metrics (e.g., image_firing_counts).
- Spatial subset for VQA via keywords/regex; VSR dataset filtering to true relations.

### 3.8 Spatial Feature Identification

- Statistical comparison (VQA vs VSR): frequency differences, odds ratios, Fisher’s exact test.
- Multiple hypothesis correction (FDR/BH), thresholds on p-value, odds ratio, minimum frequency difference.
- Output CSV of spatial feature candidates with per-layer indices and statistics.

### 3.9 Overlap with Suspect Features and Sample Enrichment

- Intersect spatial features with suspect features; robust parsing of indices.
- Enriching features with top-activating VQA samples, including question/answer/caption metadata.
- Outputs for downstream qualitative visualization.

### 3.10 Direct Logit Attribution and Attention Visualization

- Filtering VSR samples for target relation phrases; dataset handling and image retrieval fallbacks.
- Collecting per-layer residual vectors after attention (post W_O); computing cosine to SAE decoder direction for a target feature.
- Layer ranking by attribution strength; selecting exemplar samples.
- Generating attention heatmaps overlaid on images; expected artifacts (CSVs, PNGs).

### 3.11 Orchestration, Reproducibility, and Resource Management

- End-to-end orchestration: validation caching first, chunked training, checkpoint collation by method.
- Resume semantics; cleaning intermediate caches; directory structure for `results/`.
- Hardware utilization notes (CUDA, batch sizes, memory safety, empty cache calls).

## 4. Experiments

### 4.1 Experimental Setup

- Models: LLaVA-MORE configuration; SAE shapes per layer (brief).
- Datasets: VQAv2 validation range selections; VSR splits and filtering; spatial keyword list.
- Training/eval settings: sample counts, chunk sizes, batch sizes; layer ranges (e.g., 0–32); methods tested.
- Compute environment: GPUs, memory, runtime; seeds; logging.

### 4.2 Experiment Suite

- E1: SAE fine-tuning quality per method and layer
  - Metrics: reconstruction loss, FVU, sparsity; validation curves.
  - Comparison across methods and layers.
- E2: Geometry divergence
  - Per-layer encoder/decoder cosine distributions; low-cosine counts; deltas.
  - Multi-checkpoint comparison (pretrained, image-only, text-only, random).
- E3: Image vs Text activation behavior
  - Rotated features’ activation distributions on image vs text tokens; per-layer ratios.
- E4: Reconstruction gaps
  - NMSE comparisons across SAE types and data modalities; gap statistics.
- E5: Feature firing on VQA vs VSR
  - Aggregate firing rates; dead/dense feature counts; image-only vs text-only firing.
  - Spatial feature selection via Fisher + FDR; layer-wise counts and top features.
- E6: Overlap with suspect features
  - Intersection sizes and enrichment statistics; per-layer breakdown.
- E7: Direct logit attribution validation
  - Layer rankings for selected spatial features; exemplar attention heatmaps.
- E8: Qualitative analyses
  - Top-activating samples per feature (VQA and VSR); side-by-side comparisons.

## 5. Results

- R1: Fine-tuning metrics by method/layer (trends, notable layers).
- R2: Geometry divergence patterns (where rotations concentrate; encoder vs decoder).
- R3: Image-text activation preferences of rotated features.
- R4: Reconstruction gap behaviors across layers.
- R5: Spatial feature set with significance and effect sizes; overlap with suspect features.
- R6: Attribution validation: strongest layers, representative heatmaps, qualitative alignment with relation phrases.

## 6. Discussion

- Interpretation: what rotated features capture; emergence of spatial specialization.
- Cross-modal dynamics: interplay between text and image tokens; masking insights.
- Method comparisons: pretrained vs random vs text-only vs image-only.
- Robustness and failure cases: sensitivity to thresholds, image token span accuracy, dataset biases, network errors in image loading.
- Limitations: compute constraints, dataset scope, generalization to other VLMs.

## 7. Conclusion and Future Work

- Summary of key findings and takeaways.
- Future extensions
  - OCR and object recognition features; multi-feature joint attribution; interactive analysis.
  - Cross-dataset validation beyond VSR; architectural comparisons; ablation/steering studies.

## 8. Reproducibility and Implementation Notes

- Code structure overview with entry points
  - Activation caching: `finetune/vqa/cache_activations.py`
  - Training: `finetune/vqa/train_sae.py`, orchestrator `finetune/vqa/orchestrate.py`
  - Analyses: `experiments/*.py`
  - Firing tracking: `finetune/vqa/track_firing_vqa*.py`
  - Spatial stats: `features/spatial/find_spatial_features.py`
  - Validation: `finetune/vqa/direct_logit_attribution.py`
- How to run (brief command templates; environment and data prerequisites).
- Expected outputs and directory layout under `results/`.

## 9. Figures and Tables (Planned)

- Figures
  - F1: Pipeline overview diagram (stages and data flow).
  - F2: Per-layer histogram of decoder cosine similarities (LLM vs VLM) with mean lines.
  - F3: Multi-checkpoint comparison histograms per layer.
  - F4: Image vs text activation energy distributions for rotated features (selected layers).
  - F5: Reconstruction NMSE comparisons across SAE/data combinations by layer.
  - F6: Feature firing frequency distributions (VQA, VSR), dead/dense feature bars.
  - F7: Odds ratio vs adjusted p-value volcano-style plot for spatial features.
  - F8: Overlap visualization between suspect and spatial features (per-layer bars).
  - F9: Direct logit attribution layer ranking curves for selected features.
  - F10: Attention heatmaps over exemplar images for spatial relations.
  - F11: Qualitative grids of top-activating samples for selected features (VQA, VSR).
- Tables
  - T1: Training configurations and datasets per experiment.
  - T2: Summary statistics of cosine similarity and deltas per layer.
  - T3: Reconstruction gap metrics per layer.
  - T4: Spatial feature list with odds ratios, p_adj, freq diffs (top-N per layer).
  - T5: Overlap counts with suspect features by layer and method.

## 10. Appendices

- A. Extended plots for all layers and methods (cosines, gaps, ratios).
- B. Full spatial keyword list and regex rules used for VQA filtering.
- C. JSON/PT output schema details for firing trackers and analysis scripts.
- D. Additional attribution examples and failure cases.
- E. Environment details, exact command lines, and seed settings.

---

Implementation anchors (for cross-reference while writing):

- Activation caching: `finetune/vqa/cache_activations.py` (H5 with `img_start`/`img_end`, chunking, NNsight tracing)
- Orchestration: `finetune/vqa/orchestrate.py` (validation-first, chunk loop, W&B, checkpoint collation)
- SAE training/eval: `finetune/vqa/train_sae.py` (masking, FVU, sparsity, validation cadence)
- Geometry divergence: `experiments/compute_geometry_divergence.py` (encoder/decoder cosines; multi-checkpoint)
- Image/text ratio: `experiments/image_text_ratio.py` (token-level masks; rotated feature selection)
- Reconstruction gap: `experiments/reconstruction_gap.py` (NMSE across SAE/data combos)
- Suspect features: `experiments/select_suspect_features.py` (variance gap + cosine filter)
- VQA firing: `finetune/vqa/track_firing_vqa.py` (counts, magnitudes, sample maps)
- Spatial VQA subset: `finetune/vqa/track_firing_vqa_spatial.py` (regex keyword filtering)
- Spatial stats: `features/spatial/find_spatial_features.py` (Fisher, FDR, odds ratio, csv)
- Enrichment: `features/spatial/enrich_common_features_with_samples.py` (top-K samples, captions)
- DLA + heatmaps: `finetune/vqa/direct_logit_attribution.py` (residuals post-attn, attention overlays)

