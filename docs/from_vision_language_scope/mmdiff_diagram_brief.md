# MMDiff Pipeline — Design Brief for Diagram Generation

**Purpose:** standalone prompt you can paste into Claude Design (or any diagram tool). Every claim below is grounded in the code at `/home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2/` and `/data1/vlm_scope_sae_mix448_textonly/scripts/`.

---

## Paper title

**MMDiff: Multimodal Model Diffing for Feature Discovery and Control in Vision–Language Fine-Tuning**

## One-line abstract

MMDiff warm-starts a base-LLM SAE (Gemma-Scope / LLaMA-Scope) on multimodal-VLM LM-backbone activations, identifies *adapted* features that reoriented during visual-grounding fine-tuning, isolates a small subset of *concept-specific* features (spatial / safety / OCR) via Fisher contrast against VQA baseline + lexical filtering, and causally validates them with single-feature ablation and CAA-style steering.

---

## Full pipeline — 5 stages, stepwise

### Stage 0 — Multimodal-conditioned SAE training (one-time per VLM stack)

**Goal:** obtain an SAE basis that captures the *multimodal* semantic structure of the LM backbone.

**Inputs:**
- Multimodal-instruction-tuned VLM (PaliGemma2-3B-mix-448 → SigLIP + Gemma2-2B; or LLaVA-MORE → CLIP-ViT + LLaMA-3.1-8B).
- 50,000 VQAv2 training samples (image + question).
- Base-LLM SAE from public release (Gemma-Scope or LLaMA-Scope) — pretrained on text-only LLM activations.

**Process:**
1. **Warm-start each SAE from the base-LLM SAE.** Not random-init: `utils.initialize_jumprelu_sae()` calls `_load_gemma_scope_weights_jumprelu()` to load Gemma-Scope pretrained weights for every layer. This keeps the feature basis compatible with the text-only starting point.
2. **Fine-tune the SAE on multimodal LM-backbone activations.** The VLM processes `image + "answer en <question>"`; we collect residual-stream activations from *every transformer layer* of the LM backbone (26 layers for PaliGemma2, 32 for LLaMA-MORE).
3. **Text-only masking in the loss.** Image tokens are *masked out of the reconstruction + sparsity loss* (see `local_train_textonly.py:272`). The SAE reconstructs only text-token activations; image information flows into those text tokens via self-attention mixing. This is the "text-only SAE on multimodal activations" setup.
4. **JumpReLU for PaliGemma2 / TopK for LLaVA-MORE.** Width D=16,384. PaliGemma2: d_in=2304 (matches Gemma-2-2b). TopK with k=50 for LLaVA-MORE.
5. **Hyperparameters:** target_L0=50, bandwidth=0.001, LR=7e-5 warmup-then-cosine, batch=8, 8-GPU local training (~24 h per VLM stack).
6. **Output:** `N_LAYERS × D_SAE` adapted SAE basis. Checkpointed at `/data1/vlm_scope_sae_mix448_textonly/checkpoints/text-only_layer_{0..25}.pt`.

**Crucially, this SAE is now a fixed basis** reused for every downstream MMDiff contrast — no further training for any new target concept.

---

### Stage 1 — MMDiff pipeline (applied per target concept)

The core method. **Steps 1–8** identify a concept-specific feature set by contrasting firing distributions on a target vs baseline dataset, against the fixed multimodal SAE basis from Stage 0.

#### Step 1 — FVU table
- Read training-log FVU per layer; sanity-check the SAE converged.
- Filter out layers where SAE reconstruction is too poor.

#### Step 2 — Cosine drift (fine-tuned vs base SAE)
- For every (layer, feature), compute `cos(W_dec_base[feature], W_dec_multimodal[feature])`.
- **Low cosine = the feature rotated during multimodal fine-tuning = adapted to visual grounding.** This is the *diff* step in "Model Diffing."

#### Step 3 — Visual energy Ev
- For every (layer, feature), compute the ratio of `||activation||²` on **VLM multimodal inputs** vs on **text-only LLM (Gemma-2-2b)** inputs.
- **High Ev = feature fires preferentially in the multimodal context.**

#### Step 4 — Adapted feature set (the diff output)
- Select features satisfying `Ev > ε` AND `cos(W_dec_base, W_dec_mm) < τ_percentile`.
- These are the **features that both rotated geometrically AND gained visual preference** during fine-tuning — the candidate "adapted" population. Typical output: 50-100k features across all layers.
- This is the direct analog of Anthropic's stage-wise diffing; all subsequent filters operate on this pool.

#### Step 5 — Per-token firing on TARGET vs BASELINE
- For every adapted feature, count per-token firings across:
  - **TARGET distribution** — the concept-bearing dataset (VSR for spatial, VLSBench-UNSAFE for safety, OCR-Bench for OCR).
  - **BASELINE distribution** — 50,000 VQAv2 captions (generic multimodal task, no target concept).

#### Step 6 — Fisher exact contrast
- 2×2 table per (layer, feature): fires-on-target vs fires-on-baseline.
- Thresholds: `OR ≥ 3` AND `freq_diff ≥ 0.05`, FDR-corrected.
- Output: **candidate concept-detectors** — features that fire much more on target-concept content than on generic VQA.
- Typical output for multimodal safety: 2,590 detectors from ~86k adapted features.

#### Step 7 — Lexical-artifact filter
- **Problem:** a feature might fire on a concept only because the *prompt text* contains a keyword (e.g. "violent"), not because of actual visual understanding.
- **Test:** for each candidate, take top-5 most-activating samples. Replace the original prompt with 5 benign prompts (`"Describe this image."`, `"What do you see?"`, …). Keep the image.
- **Pass:** feature still fires (> 0.01 activation) on all top-5 samples with the benign prompt → it responds to the **visual concept**, not the prompt text.
- 8-GPU parallel. Typical pass rate: ~1,771 / 2,590 ≈ 68% for safety.

#### Step 8 — Intersection
- Final candidate set = **Adapted ∩ Fisher-selected ∩ Lexical-passed**.
- Typical output:
  - Spatial: ~N-spatial features across 26 layers.
  - Safety: **1,061 features** (from 1,061 unsafe detectors, 86k adapted pool, 2,590 Fisher set).
  - OCR: ~N-OCR features.

---

### Stage 2 — Causal validation (detector → driver)

Stage 1 yields **correlational detectors** — features selective for the target concept. Stage 2 tests whether *removing* a feature actually breaks the target behaviour.

**Intervention: single-feature 3-point projection ablation (all layers).**

```
feature_vec = SAE.W_dec[F] / ||SAE.W_dec[F]||   # unit direction

for each layer l in 0..N_LAYERS:
    for each text token t:
        attn_out[t]  -= (attn_out[t]  · feature_vec) * feature_vec
        mlp_out[t]   -= (mlp_out[t]   · feature_vec) * feature_vec
        layer_out[t] -= (layer_out[t] · feature_vec) * feature_vec
```

Ablates the feature direction from *every* transformer block (attention output + MLP output + residual stream).

**Twin-baseline eval — three contexts under the same intervention:**

| Metric | Measures | Passes if |
|---|---|---|
| **ΔTarget** — target-task accuracy | target-specific behaviour | \|Δ\| ≫ 0 (feature matters causally) |
| **ΔVQA** — yes/no VQA accuracy (1,000 samples) | capability preservation | ≈ 0 (ablation didn't break the model) |
| **ΔCtrl_ASR** — unsafe-generation rate on truly-benign prompts (MSSBench-safe, 100 samples) | specificity | ≈ 0 (ablation didn't create harmful generation) |

**Only features that pass all three become `DRIVERS`.** This is the "detector → driver" move; most SAE-interpretability work conflates detection and causation.

---

### Stage 3 — Applications (share Stage 0 SAE + Stage 2 drivers)

All applications reuse the same SAE basis and driver sets — no new training.

| Application | Mechanism | Representative result |
|---|---|---|
| **Interpret** | Auto-interp + manual inspection of top-activating samples per driver | Feature descriptions per domain (spatial relation, safety category, OCR glyph) |
| **Ablate** (safety) | 3-point projection (same as Stage 2 eval, but on full benchmark) | Single feature drops VLSBench ASR by up to **−28 pp** with ΔVQA ≈ 0 and ΔCtrl_ASR ≈ 0 |
| **Steer positively** — MMDiff-CAA | Apply a CAA-style activation-addition vector at MMDiff-identified (layer, feature); compare vs baseline CAA applied at the middle layer | MMDiff-CAA beats baseline CAA by up to **+15 pp** on VSR spatial relations |
| **Attribute** | Attribution patching from a driver back to attention heads | Sparse set of mid-layer heads drive spatial features |

---

### Stage 4 — Cross-cutting analysis (understanding fine-tuning)

MMDiff enables novel analyses the v1 paper could only gesture at:

- **Cross-stage spatial feature emergence.** Train SAEs at 25%, 50%, 75%, 100% of IT fine-tuning; apply MMDiff at each stage; track when spatial drivers emerge.
- **Base-vs-IT diffing.** Same contrast applied to two different checkpoints — reveals which features are IT-specific vs shared with base.
- **Cross-architecture diffing.** Validate that the *same pipeline* produces comparable driver sets on LLaVA-MORE + LLaMA and PaliGemma2 + Gemma.

---

## Diagram requirements

### Overall layout

Five stacked panels (top-to-bottom), roughly 16:9 each:

1. **Stage 0** — SAE training warm-start + fine-tune (horizontal flow: VLM → activations → warm-started SAE → trained SAE basis).
2. **Stage 1 — input distributions** — side-by-side TARGET vs BASELINE boxes (red vs gray) with per-domain subtitles.
3. **Stage 1 — Steps 1-4 (adapted identification)** — vertical flow: FVU → cosine → Ev → Adapted pool.
4. **Stage 1 — Steps 5-8 (concept isolation)** — Venn diagram with three overlapping ellipses: {Adapted, Fisher-selected, Lexical-passed}; the intersection is labeled "candidate feature set" with concrete counts (e.g., "1,061 unsafe features"). Flow arrow from Step 4 Adapted into the Venn.
5. **Stage 2 — causal validation** — horizontal: DETECTORS box → "Single-feature ablation (3-point projection)" block with three metric pills (ΔTarget / ΔVQA / ΔCtrl_ASR) → DRIVERS box.
6. **Stage 3 — applications** — four-column row of application cards (Interpret / Ablate / MMDiff-CAA Steer / Attribute), each with a one-line result.

### Color language

| Element | Color |
|---|---|
| SAE / Stage 0 | blue (#2e5fb5) — represents the trained basis |
| Target distribution | red (#c0392b) — the concept-bearing side of the contrast |
| Baseline distribution | gray (#555) — the generic side of the contrast |
| Adapted filter | orange (#cf6f30) — geometric rotation |
| Fisher filter | plum (#8b3a62) — statistical selectivity |
| Lexical filter | mustard (#b08a00) — visual-vs-textual disambiguation |
| Drivers / applications | green (#2e8b57) — causally validated, ready to use |

### Typography

- Title: 16 pt bold sans-serif.
- Stage headers: 12 pt bold colored (per stage color).
- Body: 9-10 pt sans-serif.
- Code/math: monospace (for decoder-projection formula + OR thresholds).
- Italic sublabels for dataset names and per-domain examples.

### Key text inserts / callouts

Include these *exact* labels and numbers in the figure:

- SAE panel: *"Warm-start from Gemma-Scope / LLaMA-Scope   →   fine-tune per layer on multimodal VLM residual-stream activations   (image tokens masked from loss)"*
- SAE panel caption: *"26 layers × 16,384 features (PaliGemma2)   ·   32 layers × 65,536 features (LLaVA-MORE)"*
- Venn intersection: **"Adapted ∩ Fisher-selected ∩ Lexical-passed"** with small "= 1,061 safety drivers" annotation.
- Fisher panel: *"OR ≥ 3, freq_diff ≥ 0.05, FDR-corrected"*
- Ablation projection block, three sub-boxes for attn/mlp/residual projection-subtraction.
- Applications row: numeric results per card:
  - Ablate: "up to −28 pp ΔASR on VLSBench (single feature, ΔVQA ≈ 0)"
  - MMDiff-CAA Steer: "+15 pp vs baseline CAA on VSR"
  - Interpret: "auto-interp on top-activating samples"
  - Attribute: "trace to sparse attention-head circuits"

### Visual metaphor suggestions

- Stage 0 — "filters ablue SAE through a multimodal lens": show a base-SAE sphere transforming into a denser/colored sphere with *image-tinted* features. Or simpler: a row of 26 small SAE-layer boxes being fed activations from the VLM backbone.
- Stage 1 input distributions — two side-by-side icons: a "target" picture (e.g. violent scene / spatial diagram / OCR image) vs a "baseline" picture (cat photo / VQAv2 style) to make the contrast concrete.
- Venn — the intersection should be the most prominent region, with the three outer regions dimmed. Use transparency 0.3 so overlaps are visible.
- Stage 2 — a big red "DETECTORS" shape getting sieved through the ablation evaluation; only the ones with the three check-marks (ΔTarget ≠ 0, ΔVQA ≈ 0, ΔCtrl ≈ 0) emerge green on the other side.

### Style

- Clean, publication-ready, white background, subtle grid lines only if they add structure.
- No 3D effects, no drop shadows.
- All boxes rounded (corner radius ~6-8 pt).
- Arrow style: solid, thin (~1.5 pt), arrowhead size ~18 pt.
- Dense but not cluttered — target one page of a NeurIPS / ICML paper.

---

## Per-stage data & code anchors (for reference)

```
Stage 0 (SAE training):
  /home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2/local_train_textonly.py
  /home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2/utils.py  (JumpReLUSAE, initialize_jumprelu_sae, warm-start from Gemma-Scope)

Stage 1 (Steps 1-8 pipeline):
  Steps 1-8 (spatial):           /home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2/local_analysis_textonly.py
  Step 5 (safety firing):        /data1/vlm_scope_sae_mix448_textonly/scripts/multimodal_safety/30_firing_vlsbench_unsafe.py
  Step 6 (safety Fisher):        /data1/vlm_scope_sae_mix448_textonly/scripts/multimodal_safety/31_fisher_vlsbench.py
  Step 7 (safety lexical):       /data1/vlm_scope_sae_mix448_textonly/scripts/multimodal_safety/32_lexical_filter_safety.py
  Step 8 (safety intersection):  /data1/vlm_scope_sae_mix448_textonly/scripts/multimodal_safety/33_intersect_unsafe_features.py

Stage 2 (ablation):
  Spatial:   /data1/vlm_scope_sae_mix448_textonly/scripts/ablation_per_relation_textonly.py  (3-point projection, lines 170-183)
  Safety:    /data1/vlm_scope_sae_mix448_textonly/scripts/multimodal_safety/35_ablate_safety_features.py

Stage 3 (steering / CAA):
  /data1/vlm_scope_sae_mix448_textonly/scripts/caa_recipe_G_spat_boost.py  (MMDiff-CAA: feature-boost applied at MMDiff-identified layer)
  /data1/vlm_scope_sae_mix448_textonly/scripts/caa_all_layers_inject.py   (baseline CAA: middle-layer injection)
```

---

## Short caption you can paste under the figure

> **Figure N: The MMDiff pipeline.** We train a multimodal SAE once per VLM stack, warm-started from a public base-LLM SAE (Gemma-Scope/LLaMA-Scope), fine-tuned on LM-backbone residual-stream activations collected while the VLM processes image+text inputs (image tokens masked from the loss). Against this fixed SAE basis, we apply per-concept contrasts: (1) identify **adapted** features that rotated and gained visual energy during multimodal fine-tuning; (2) isolate **concept-detectors** via Fisher contrast between target-concept firings (VSR spatial / VLSBench-unsafe / OCR) and VQAv2 baseline firings; (3) filter **visual-not-lexical** features via benign-prompt re-testing; (4) causally validate via single-feature ablation with twin VQA + MSSBench-safe baselines, converting detectors into **drivers**. The same drivers power interpretation, ablation, MMDiff-CAA steering, and attribution-patching applications across all three domains.
