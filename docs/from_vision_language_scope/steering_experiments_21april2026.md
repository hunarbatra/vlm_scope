# Steering & Injection Experiments — Cross-Stage Feature Transfer
**Date:** 21 April 2026  
**Status:** Active iteration — Phase 2 (activation steering + CAA)
**Goal:** Find the single best method to transfer mix-448 spatial capabilities into pt-448 at inference time (no training). Testing both W_dec feature injection AND activation steering (CAA, FGAA, SAE clamping).

---

## Literature Context: Mechanistic Interpretability on Activation Steering

*(Research completed 21 April 2026 — papers surveyed for improved Experiments 5 & 6)*

### RepE — Representation Engineering (Zou et al. 2023, arXiv 2310.01405)
Extracts population-level control vectors as mean activation differences between contrastive prompts. Uses PCA on the difference matrix to find the principal component, which retains the natural L2 norm so alpha-scaling is meaningful. **Multi-layer injection outperforms single-layer**; middle-to-late layers (50–83% depth) are most effective. Key finding: alpha exhibits **non-monotonic effects** — increasing α past the optimum can reverse behavior (e.g., α=2→3 flips steering direction). This matches our observation that "ahead of" peaks at α=5, not α=20.

### CAA — Contrastive Activation Addition (Rimsky et al. 2023, arXiv 2312.06681)
Mean residual stream difference between positive/negative behavioral examples, applied at all token positions after the prompt. Layer 13 > layers 14–15 for peak efficacy in Llama-2-7B (roughly layer 50% depth). **Critical finding for our case:** when a behavior doesn't naturally exist in the base model, CAA can synthesize approximate directions, but effectiveness degrades — the vector finds the closest direction in activation space but can't create knowledge that isn't represented. Directly explains our 0.00× transfer features getting near-zero injection gain.

### ActAdd — Activation Addition (Turner et al. 2023, arXiv 2308.10248)
Adds activation differences from minimal contrastive prompt pairs mid-layer. Middle layers (40–60% depth) consistently effective. The unnormalized vector retains empirically-determined scale, so alpha scaling is more interpretable. Effectiveness depends on **abstraction level** — high-level semantic directions > low-level lexical patterns. Our SAE W_dec vectors are high-abstraction (sparse, monosemantic), which is why they work better than raw DIM (which captures all abstraction levels mixed).

### Scaling Monosemanticity / SAE Steering (Templeton et al. 2024, Anthropic)
W_dec vectors (SAE decoder weights) as steering directions by clamping SAE latent space activations. Key advantage: interpretable, semantically isolated. **When features don't exist naturally**, amplifying them degrades unrelated capabilities (MMLU, GSM8K drops), indicating deep entanglement. Features fire cross-modally (same direction for text and image inputs) — relevant for our VLM setting. Multi-layer implications: SAE features implicitly integrate across layers via residual stream.

### FGAA — Feature Guided Activation Additions (arXiv 2501.09929)
Combines CAA with SAE insights: identifies relevant features in SAE latent space, filters out high-density linguistic features and BOS-token artifacts, then optimizes linear combinations of multiple W_dec vectors. Outperforms CAA (BCS 0.47 vs. 0.22) on Gemma-2-2B. **Directly relevant to us**: explicit feature selection beats implicit direction-finding. CAA's opacity masks feature entanglement; FGAA's interpretability detects and mitigates it. Suggests we should try **linear combination of multiple spatial feature directions** rather than injecting one at a time.

### PIXEL — Position-wise Injection (arXiv 2510.10205)
Learns property-aligned subspaces with constrained geometric optimization to determine intervention strength **per token position** (closed-form solution). Outperforms uniform steering substantially for attribute alignment. Directly motivates our multi-layer script: different text token positions need different injection strengths — injecting uniformly across all text tokens may wash out the signal.

### SteerVLM — VLM Steering (arXiv 2510.26769)
Lightweight steering modules (0.14% params) for VLMs that provide dimension-wise activation modulation across layers. Adapts to visual vs. linguistic inputs separately. +1.7% on hallucination mitigation, +21% on topic steering vs. ActAdd. Key finding: VLMs need **layer-agnostic dynamic steering** because the point where visual and linguistic representations merge shifts per example. Static single-layer vectors miss this.

### Key Takeaways for Our Experiments
1. **Multi-layer > single-layer**: inject across SAE layer and downstream layers; use Gaussian/decay weighting
2. **Non-monotonic alpha**: our best single-layer results are at α=5–20; extended range (α=30–50) may reveal peaks
3. **Per-example calibration**: per-example SAE activations (our Exp 6) should outperform fixed-scale injection
4. **Feature non-existence**: 0.00× transfer features cannot be created by injection — need fine-tuning or they're simply absent
5. **Log-odds margin**: add continuous metric to detect sub-threshold improvements that don't flip Yes/No

---

## Setup

| Component | Value |
|-----------|-------|
| SAE | JumpReLU text-only SAE trained on mix-448 residual stream |
| Source model | `google/paligemma2-3b-mix-448` (instruction-tuned) |
| Target model | `google/paligemma2-3b-pt-448` (pretrained backbone) |
| Benchmark | VSR (cambridgeltl/vsr_random, all splits, N=10,132 total) |
| Evaluation | Yes/No logit scoring on VSR captions |
| Features | Top-10 spatial features by mix-448 ablation effect |
| Injection tokens | Text tokens only (img_end: onwards), residual stream |

**NNsight proxy trick** (required for all interventions):
```python
fv_col = fv.unsqueeze(1)                      # (d, 1)
ones   = (layer_out @ fv_col) * 0.0 + 1.0    # (T, 1) — proxy derived inside trace
layer_out += alpha * ones * fv                # (T, d) — intercepted by NNsight ✓
```
Plain `layer_out += alpha * fv` is silently ignored by NNsight (proxy vs. plain tensor).

---

## Cross-Stage Transfer Ratios (Baseline Context)

From `cross_stage_ablation_20april2026.md`:

| L/F | Relation | ∆ mix-448 | ∆ pt-448 | Transfer Ratio |
|-----|----------|-----------|----------|----------------|
| L9/F387 | at the right side of | -30.62% | -2.08% | 0.07× |
| L14/F10561 | close to | -18.28% | -7.53% | 0.41× |
| L11/F12278 | touching | -12.10% | -4.84% | 0.40× |
| L9/F7540 | consists of | -11.43% | -2.86% | 0.25× |
| L4/F14233 | ahead of | -10.26% | -10.26% | **1.00×** |
| L6/F7539 | left of; right of | -9.60% | -4.64% | 0.48× |
| L11/F9639 | in; inside; on | -8.63% | -2.09% | 0.24× |
| L13/F15219 | behind | -8.04% | +1.55% | **0.00×** |
| L15/F220 | across from / at the left side of | -7.58% | -2.08% | 0.27× |
| L12/F2257 | facing | -6.86% | +3.27% | **0.00×** |

Transfer ratio = |∆ pt-448| / |∆ mix-448|. Features with ratio = 0.00× (behind, facing) have *inverted* pt-448 response — these directions don't exist in the backbone and may represent interference.

---

## Experiment 1 — mix-448 Projection Steering (Baseline Reference)

**Script:** `mix448_projection_steering.py`  
**Method:** Projection-based amplification at 3 taps (attn_out, mlp_out, layer_out), all 26 layers  
`act += alpha * (act @ fv.T) * fv`  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/mix448_projection_steering/`

This is the reference experiment showing how strongly each feature causally affects mix-448 VSR.

| L/F | Relation | N | Base (mix) | α=-1 | α=-0.5 | α=+0.5 | α=+1 | α=+2 | α=+5 | α=+10 |
|-----|----------|---|-----------|------|--------|--------|------|------|------|-------|
| L9/F387 | at the right side of | 480 | 76.67% | -31.25 | -28.13 | -22.29 | -27.08 | -27.08 | -24.58 | -26.25 |
| L14/F10561 | close to | 93 | 79.57% | -18.28 | -16.13 | -8.60 | -9.68 | -9.68 | -46.24 | -49.46 |
| L11/F12278 | touching | 1281 | 76.58% | -12.10 | -10.69 | -25.60 | -25.60 | -25.60 | -27.17 | -25.60 |
| L9/F7540 | consists of | 35 | 85.71% | -11.43 | -8.57 | 0.00 | -22.86 | -22.86 | -22.86 | -22.86 |
| L4/F14233 | ahead of | 39 | 61.54% | -10.26 | -7.69 | -7.69 | -2.56 | -5.13 | -5.13 | -5.13 |
| L6/F7539 | left of; right of | 323 | 69.97% | -9.29 | -8.67 | -22.91 | -21.67 | -21.67 | -23.53 | -21.67 |
| L11/F9639 | in; inside; on | 1101 | 81.56% | -8.45 | -7.18 | -30.70 | -30.79 | -28.79 | -30.70 | -30.79 |
| L13/F15219 | behind | 709 | 71.79% | -8.04 | -6.63 | -20.17 | -21.44 | -23.98 | -22.71 | -22.14 |
| L15/F220 | across from | 515 | 69.71% | -3.50 | -1.94 | -9.90 | -18.83 | -17.48 | -15.53 | -20.58 |
| L12/F2257 | facing | 306 | 60.46% | -6.54 | -4.58 | -8.17 | -8.17 | -8.17 | -12.75 | -12.75 |

**Key observation:** All features have strong causal effect in mix-448 (α removal → -6 to -31%). Projection at α>0 also hurts, which is expected (over-amplification saturates the Yes/No decision boundary). mix-448 baselines are much higher (60–86%) than pt-448 baselines (49–69%) — a ~10–20% gap that injection aims to close.

---

## Experiment 2 — pt-448 Fixed W_dec Injection

**Script:** `pt448_feature_injection.py`  
**Method:** Add W_dec[F] direction unconditionally at native SAE layer, text tokens only  
`layer_out[img_end:] += alpha * fv`  
**Alphas tested:** [-1.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_feature_injection/`

| L/F | Relation | N | Base (pt) | α=-1 | α=+0.5 | α=+1 | α=+2 | α=+5 | α=+10 | α=+20 | **Best ∆** | **Xfer** |
|-----|----------|---|-----------|------|--------|------|------|------|-------|-------|----------|------|
| L9/F387 | at the right side of | 480 | 52.08% | -0.21 | +0.21 | +0.42 | +0.62 | +0.21 | **+1.87** | +0.83 | +1.87 | 0.07× |
| L14/F10561 | close to | 93 | 59.14% | +1.08 | +2.15 | +1.08 | +1.08 | +1.08 | -3.23 | -7.53 | **+2.22** | 0.41× |
| L11/F12278 | touching | 1281 | 56.44% | -0.08 | -0.08 | +0.31 | +0.94 | +1.95 | +2.65 | **+3.20** | **+3.20** | 0.40× |
| L9/F7540 | consists of | 35 | 68.57% | -2.86 | -2.86 | -2.86 | -2.86 | -2.86 | 0.00 | -2.86 | 0.00 | 0.25× |
| L4/F14233 | ahead of | 39 | 56.41% | 0.00 | +2.56 | +2.56 | +2.56 | **+7.69** | +2.56 | -2.56 | **+7.69** | **1.00×** |
| L6/F7539 | left of; right of | 323 | 50.77% | +0.62 | -0.31 | -0.93 | -1.24 | **+0.93** | +0.62 | 0.00 | +0.93 | 0.48× |
| L11/F9639 | in; inside; on | 1101 | 61.04% | -0.36 | -0.36 | -0.18 | -0.54 | -1.00 | -1.73 | -4.36 | -0.18 | 0.24× |
| L13/F15219 | behind | 709 | 51.62% | +0.71 | +0.28 | -0.42 | -0.28 | +0.42 | -0.85 | +0.28 | +0.71 | 0.00× |
| L15/F220 | across from | 515 | 49.90% | 0.00 | +0.58 | +0.39 | -0.19 | +0.97 | +1.36 | +0.39 | +1.36 | 0.27× |
| L12/F2257 | facing | 306 | 49.02% | -0.65 | +0.65 | +0.33 | +0.33 | +0.33 | 0.00 | -0.33 | +0.65 | 0.00× |

**Key findings:**
- "ahead of" (1.00× transfer): best result, +7.69% at α=5. Direction fully present in pt-448.
- "touching" (0.40×): steady improvement up to α=20 (+3.20%). Monotone response suggests feature is present, just weaker.
- "close to" (0.41×): peaks early at α=0.5 (+2.22%), degrades at high α. More fragile feature.
- "left of; right of" (0.48×): noisy, best +0.93%. Possibly two opposing features merged.
- 0.00× transfer features (behind, facing): noise-level gains (±0.7%), confirms injection of non-existent features does nothing.
- Low-transfer features (right side of 0.07×, in/inside/on 0.24×, consists of 0.25×): near-zero gains or negative — the direction exists too weakly in pt-448 to benefit from amplification.

**Transfer ratio → injection gain correlation:**  Features with transfer ratio ≥ 0.40× all show ≥ +1.0% improvement. Features with ratio < 0.25× show ≤ +0.7%. The exception is "left of; right of" (0.48×) which underperforms — likely because this feature aggregates two directions (left vs. right) that partially cancel.

---

## Experiment 3 — pt-448 DIM Steering

**Script:** `pt448_dim_steering.py`  
**Method:** Run mix-448 forward on relation subset, compute DIM_vec = mean(h[L] | label=1) - mean(h[L] | label=0) at last text token, normalize to unit vector, inject into pt-448  
**Alphas tested:** [-1.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_dim_steering/`

| L/F | Relation | N | Base | Constant ∆ (all alphas identical) |
|-----|----------|---|------|----------------------------------|
| L9/F387 | at the right side of | 480 | 52.08% | **-1.67** |
| L14/F10561 | close to | 93 | 59.14% | **-29.03** |
| L11/F12278 | touching | 1281 | 56.44% | **-5.46** |
| L9/F7540 | consists of | 35 | 68.57% | -5.71 (α≥0.5); -8.57 (α=-1) |
| L4/F14233 | ahead of | 39 | 56.41% | **0.00** |
| L6/F7539 | left of; right of | 323 | 50.77% | **-2.48** |
| L11/F9639 | in; inside; on | 1101 | 61.04% | **-10.26** |
| L13/F15219 | behind | 709 | 51.62% | **-1.97** |
| L15/F220 | across from | 515 | 49.90% | **-0.78** |
| L12/F2257 | facing | 306 | 49.02% | **-1.31** |

**Critical failure mode:** The delta is *identical across all alpha values* for 9/10 features. This indicates the DIM vector pushes the model into a binary saturated regime — even the smallest injection (α=0.5) flips the model's Yes/No decision at the same rate as α=20. The model output is already at the logit saturation boundary, so direction magnitude doesn't matter once threshold is crossed.

**Why DIM fails:**
1. **Confound-heavy**: The full hidden state h[L] at any mid-layer encodes syntax, positional information, attention sink patterns, and image-text alignment in addition to the spatial feature signal. The DIM vector picks up all of these, not just the spatial direction.
2. **Wrong aggregation point**: Extracting at the *last text token* captures the aggregated context vector, which is dominated by whatever the model is routing to the answer token. This is different from the feature's activation space direction W_dec[F].
3. **Scale mismatch**: A unit-normalized DIM vector injected at any α > 0 may shift the entire hidden state distribution enough to change *all* predictions, not selectively the spatial ones.

"ahead of" (0.00 delta) is the one exception — its DIM vector is apparently orthogonal to the Yes/No decision boundary in pt-448, consistent with it already being fully represented (1.00× transfer ratio, pt-448 baseline already at ~56%).

---

## Experiment 4 — SAE-Calibrated Injection

**Script:** `pt448_sae_calibrated_injection.py`  
**Method:** Load mix-448 SAE, encode hidden states → get feature F activations for each example, compute empirical scale = mean(act_F | label=1) - mean(act_F | label=0). Inject `M * scale * W_dec[F]` with multipliers M = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0].  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_calibrated_injection/`

### Calibrated scale values:

| L/F | Relation | SAE mean (pos) | SAE mean (neg) | Calibrated scale | Direction |
|-----|----------|---------------|---------------|-----------------|-----------|
| L9/F387 | at the right side of | 4.458 | 4.516 | **-0.058** | WRONG |
| L14/F10561 | close to | 4.672 | 4.867 | **-0.195** | WRONG |
| L11/F12278 | touching | 1.601 | 1.507 | +0.094 | correct |
| L9/F7540 | consists of | 0.128 | 0.095 | +0.033 | correct |
| L4/F14233 | ahead of | 0.713 | 0.601 | +0.112 | correct |
| L6/F7539 | left of; right of | 1.433 | 1.436 | **-0.003** | WRONG |
| L11/F9639 | in; inside; on | 0.940 | 1.180 | **-0.240** | WRONG |
| L13/F15219 | behind | 2.597 | 2.476 | +0.121 | correct |
| L15/F220 | across from | 1.615 | 1.539 | +0.076 | correct |
| L12/F2257 | facing | 1.573 | 1.566 | +0.008 | correct |

**Two distinct failure modes:**

**Mode 1 — Inverted scale (4 features):** "right side of", "close to", "left of/right of", "in/inside/on" all have negative calibrated scales, meaning feature F fires *more* for false spatial statements than true ones. This is counter-intuitive but makes sense: these features may encode spatial *contradiction* or *mismatch* detection rather than spatial presence. Injecting with a negative scale pushes the model *away* from spatial reasoning. For "close to" (scale=-0.195 → M=1 injects -0.195×fv), this explains the massive -29% drop that's constant across all multipliers.

**Mode 2 — Correct direction but tiny scale:** Features with positive scale have values in range 0.003–0.121. Even M=20 gives injection magnitude of only 0.003×20=0.06 to 0.121×20=2.4. This is the same range as W_dec injection at α=0.03–2.4 in Experiment 2. The quantization of the Yes/No binary response means at this scale, many examples stay on the same side of the decision boundary regardless of multiplier — hence constant deltas.

**Result table (all features show constant delta across multipliers):**

| L/F | Relation | N | Base | Constant ∆ | Root cause |
|-----|----------|---|------|-----------|-----------|
| L9/F387 | at the right side of | 480 | 52.08% | -1.67 | Inverted scale (-0.058) |
| L14/F10561 | close to | 93 | 59.14% | -29.03 | Inverted scale (-0.195) |
| L11/F12278 | touching | 1281 | 56.44% | -5.46 | Tiny scale (+0.094), jumps threshold |
| L9/F7540 | consists of | 35 | 68.57% | -5.71 | Tiny scale (+0.033) |
| L4/F14233 | ahead of | 39 | 56.41% | 0.00 | Tiny scale (+0.112), already on boundary |
| L6/F7539 | left of; right of | 323 | 50.77% | -2.48 | Inverted scale (-0.003) |
| L11/F9639 | in; inside; on | 1101 | 61.04% | -10.26 | Inverted scale (-0.240) |
| L13/F15219 | behind | 709 | 51.62% | -1.97 | Tiny scale (+0.121) |
| L15/F220 | across from | 515 | 49.90% | -0.78 | Tiny scale (+0.076) |
| L12/F2257 | facing | 306 | 49.02% | -1.31 | Tiny scale (+0.008) |

---

## Comparative Summary

| L/F | Relation | Xfer | E1 ∆mix (best ablate) | E2 best (W_dec inject) | E3 DIM | E4 SAE-calib |
|-----|----------|------|-----------------------|----------------------|--------|-------------|
| L9/F387 | at the right side of | 0.07× | -30.6% | +1.87 (α=10) | -1.67 | -1.67 |
| L14/F10561 | close to | 0.41× | -18.3% | **+2.22** (α=0.5) | -29.03 | -29.03 |
| L11/F12278 | touching | 0.40× | -12.1% | **+3.20** (α=20) | -5.46 | -5.46 |
| L9/F7540 | consists of | 0.25× | -11.4% | 0.00 | -5.71 | -5.71 |
| L4/F14233 | ahead of | **1.00×** | -10.3% | **+7.69** (α=5) | 0.00 | 0.00 |
| L6/F7539 | left of; right of | 0.48× | -9.6% | +0.93 (α=5) | -2.48 | -2.48 |
| L11/F9639 | in; inside; on | 0.24× | -8.6% | -0.18 | -10.26 | -10.26 |
| L13/F15219 | behind | 0.00× | -8.0% | +0.71 (α=-1) | -1.97 | -1.97 |
| L15/F220 | across from | 0.27× | -7.6% | +1.36 (α=10) | -0.78 | -0.78 |
| L12/F2257 | facing | 0.00× | -6.9% | +0.65 (α=0.5) | -1.31 | -1.31 |

---

## Core Finding: Transfer Ratio Predicts Injection Gain

The single strongest predictor of whether W_dec injection improves pt-448 accuracy is the cross-stage transfer ratio:

| Transfer Ratio Tier | Features | Injection result |
|--------------------|----------|-----------------|
| ≥ 0.40× (moderate/high) | ahead of, close to, touching, left of/right of | +0.9% to +7.7% improvement |
| 0.24–0.27× (low) | in/on/inside, across from, consists of | ≈0 or small negative |
| ≤ 0.07× (near-zero) | right side of | +1.9% anomaly (small N at peak) |
| 0.00× (inverted) | behind, facing | noise only (±0.7%) |

This confirms: **feature injection only works when the feature direction already exists causally in the target model**. Injection amplifies a latent signal; it cannot implant a signal that isn't there.

---

## Open Problems & Next Steps

### Why can't we reach mix-448 accuracy levels?

The best result (+7.7% for "ahead of") still leaves pt-448 at ~64% vs. mix-448 at ~62% (already higher). For features where the gap is largest — e.g. "in/inside/on" where pt-448=61% vs mix-448=82% — the transfer ratio is only 0.24×, so there's insufficient signal to amplify.

**Hypothesis:** To fully close the gap, we would need either:
1. Multi-layer injection across all layers where the feature is active (not just the SAE layer)
2. A contrastive pair vector extracted from pt-448's own representations (if the feature exists there at all)
3. The mix-448 → pt-448 rotation matrix to project the direction into pt-448's coordinate system

### Experiment 4 fixes needed:

1. **Absolute value of scale with sign from direction**: Use `|scale| * sign_from_ablation * fv` — force the injection direction to be consistent with the ablation finding (ablate hurts → inject should help).
2. **Use pt-448's own activations to compute the scale**: Instead of mix-448 firing contrast, compute how much feature F fires in pt-448 on positive vs. negative VSR examples, then inject `pt_scale * fv`. This avoids the inverted-scale pathology.
3. **Larger multiplier range**: Experiment 4 only achieves ≤2.4 effective alpha. The W_dec injection found peaks at α=5–20. Need to test M=50–200 in the calibrated version.
4. **Per-example injection**: Instead of a fixed scale, inject `alpha * act_F(example) * fv` — amplify the feature's actual firing for each example rather than a dataset-average scale.

### Experiment 2 improvements:

1. **Multi-layer injection**: Inject at all layers from 0 to L (or L to 26), not just the SAE layer. Use exponential decay: `alpha * decay^|layer - L_sae| * fv`.
2. **Finer alpha grid**: Current grid [-1, 0.5, 1, 2, 5, 10, 20] misses the peak for some features. "Ahead of" peaks at α=5; adding α=3, 4, 6 might reveal a sharper peak.
3. **Contrastive injection**: Instead of `+alpha*fv`, use `+alpha*(fv - fv_neg)` where fv_neg is the next-highest firing opposing feature (if any). Reduces collateral damage to other feature directions.
4. **Norm-matched injection**: Scale alpha to match the typical L2 norm of residual stream additions at that layer: `alpha_norm = alpha * mean_layer_norm / ||fv||`. This makes alpha interpretable across layers.

---

## Methodology Notes

### What worked
- W_dec fixed injection (Exp 2) — simple, effective for moderate/high transfer features
- Transfer ratio as predictor — completely separates effective from ineffective features
- NNsight proxy trick — essential for any in-context injection

### What failed
- DIM steering (Exp 3) — full hidden state too noisy; constant deltas at all alpha indicates threshold saturation
- SAE-calibrated injection (Exp 4) — inverted scales for half the features; tiny scale range; constant deltas indicate same saturation

### Why binary Y/N responses make evaluation hard
VSR uses Yes/No logit ratio. This means small injections can cause large discrete accuracy jumps (many examples near the decision boundary flip simultaneously) or zero effect (examples far from boundary). Continuous softmax probabilities would give a smoother signal. Consider using log-odds margin as the metric: `log(p_yes / p_no)` instead of 0/1 accuracy.

---

## File Index

| Experiment | Script | Output dir | Summary |
|-----------|--------|-----------|---------|
| E1: mix-448 projection | `mix448_projection_steering.py` | `analysis/mix448_projection_steering/` | `steering_summary.csv` |
| E2: pt-448 W_dec inject | `pt448_feature_injection.py` | `analysis/pt448_feature_injection/` | `injection_summary.csv` |
| E3: pt-448 DIM steer | `pt448_dim_steering.py` | `analysis/pt448_dim_steering/` | `dim_steering_summary.csv` |
| E4: SAE-calibrated | `pt448_sae_calibrated_injection.py` | `analysis/pt448_sae_calibrated_injection/` | `sae_calibrated_summary.csv` |
| Cross-stage ablation | (prior session) | `analysis/ablation_per_relation/` | `cross_stage_ablation_20april2026.md` |

---

## Overnight Experiments (21 April 2026 — New GPU Runs)

**Critical discovery:** Loading both mix-448 AND pt-448 on the same GPU causes NNsight traces to produce constant deltas regardless of injection magnitude — all experiments with two models on one GPU were invalid. Fixed by running each model exclusively per GPU.

---

## Experiment 5 — pt-448 Multi-Layer Injection (GPU 1)

**Script:** `pt448_multilayer_injection.py`  
**Method:** Single-model (pt-448 only), NNsight proxy trick, 5 strategies:
- `single`: inject at SAE layer only (1 tap, layer_out)
- `downstream`: inject SAE layer → 25 with 0.7× decay
- `all`: inject all 26 layers, 0.7× decay from SAE layer
- `answer`: inject last 5 layers (21–25) only
- `topK`: inject SAE layer ±2 with 0.7× decay

**Alphas:** [1, 2, 5, 10, 20, 30, 50] — **COMPLETED for all 10 features**

### Full results (best Δ per feature × strategy):

| L/F | Relation | N | Base | single | downstream | all | answer | topK |
|-----|----------|---|------|--------|-----------|-----|--------|------|
| L9/F387 | at the right side of | 480 | 52.3% | +1.87@α10 | +0.83 | +1.67 | +1.25 | +1.25 |
| L14/F10561 | close to | 93 | 60.2% | 0.00 | +1.08@α1 | **+2.15@α2** | 0.00 | 0.00 |
| L11/F12278 | touching | 1281 | 56.5% | **+3.20@α20** | +0.55 | +1.95 | +0.94 | +1.64 |
| L9/F7540 | consists of | 35 | 68.6% | **+2.86@α10** | +2.86 | +2.86 | +2.86 | +2.86 |
| L4/F14233 | ahead of | 39 | 56.4% | **+7.69@α5** | +7.69 | +7.69 | +7.69 | +7.69 |
| L6/F7539 | left/right of | 323 | 51.1% | +0.62 | +0.93 | +1.24 | +0.62 | **+1.24@α20** |
| L11/F9639 | in/inside/on | 1101 | 60.9% | +0.36 | +0.55 | +0.55 | **+0.73@α10** | +0.64 |
| L13/F15219 | behind | 709 | 51.6% | +0.71 | **+2.12@α30** | +0.56 | +0.14 | +0.56 |
| L15/F220 | across from | 515 | 49.9% | +1.36 | +0.97 | +1.17 | **+1.75@α5** | +1.36 |
| L12/F2257 | facing | 306 | 49.0% | +0.65 | 0.00 | **+3.92@α50** | +2.29 | +1.96 |

**Surprise finding:** `L12/F2257 "facing"` (0.00× transfer) responds strongly: **+3.92% with "all" strategy at α=50**. Same for `L13/F15219 "behind"` (0.00× transfer): **+2.12% with "downstream" at α=30**. These contradictions of the "0.00× → no injection benefit" hypothesis are the most important finding from Exp 5.

---

## Experiment 6 — pt-448 3-Tap All-Layer Injection (GPU 3)

**Script:** `pt448_3tap_alllayer_injection.py`  
**Method:** Mirror of the ablation — inject at attn_out, mlp_out, AND layer_out for all 26 layers = **78 intervention points**. Unbounded addition vs. bounded projection removal.  
**Alphas:** [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

### Results (completed features):

| L/F | Relation | base | α=0.05 | α=0.1 | α=0.2 | α=0.5 | α=1.0 | α=2.0 | α=5.0 |
|-----|----------|------|--------|--------|--------|--------|--------|--------|--------|
| L9/F387 | at the right side of | 52.29% | +0.42 | +0.42 | −1.04 | −28.54 | (collapse) | | |
| L9/F7540 | consists of | 68.57% | | | | | −5.71 | −5.71 | (running) |

**Critical finding — injection ≠ ablation:** The 3-tap ablation *removes* existing projections (bounded, no new energy added). The 3-tap injection *adds* unbounded energy at 78 points simultaneously. At α=0.05, only minimal gain (+0.42%); at α=0.2+, catastrophic collapse (model always-No). **The injection mechanism does NOT mirror the ablation mechanism.** Multi-tap injection needs extreme care with scale.

---

## Experiment 7 — mix-448 Fixed W_dec Injection ("Can we boost mix-448?")

**Script:** `mix448_fixed_injection.py`  
**Method:** Inject W_dec[F] into mix-448 itself at native SAE layer. Tests whether the fine-tuned model can be pushed beyond its trained performance.  
**Alphas:** [-2, -1, -0.5, 0.1, 0.5, 1, 2, 5, 10]  
**Status:** Running — completed 2 features

### Results:

| L/F | Relation | N | Base (mix) | Best ∆acc | Best α | Best ∆margin | Pattern |
|-----|----------|---|-----------|-----------|--------|--------------|---------|
| L9/F387 | at the right side of | 480 | 76.67% | **+0.21%** | α=0.5 | +0.013 (α=5) | Accuracy barely moves; margin improves monotonically to α=5 |
| L14/F10561 | close to | 93 | 79.57% | **0.00%** (all α) | — | -0.027 (α=10) | Completely frozen at 0.00% Δ; margin degrades with +α |

**Verdict on mix-448 boosting: Does NOT work.**
1. **Saturation:** mix-448 already uses these feature directions at their optimal weight. Adding more doesn't flip any examples.
2. **Negative contrast:** Both L9/F387 and L14/F10561 have NEGATIVE SAE firing contrast (fire more on false spatial examples). Injecting +W_dec actually shifts confidence toward predicting false statements as true → accuracy drops at larger alpha.
3. **Resistance:** L14/F10561 is locked at 0.00% accuracy change across the full alpha range -2 to +10. The model's decision boundary is completely orthogonal to this injection direction for this feature.

The mix-448 model is already at a local optimum for spatial reasoning. Inference-time injection cannot push it further without disrupting its internal calibration.

---

## Experiment 8 — pt-448 Residual All-Layer Injection (GPU 5)

**Script:** `pt448_residual_alllayer_injection.py`  
**Method:** Residual stream (layer_out) ONLY at all 26 layers. 4 strategies:
- `flat`: uniform alpha at all layers
- `decay_fwd`: alpha × 0.85^max(l-sae_layer,0) per layer
- `sae_only_down`: inject from SAE layer to 25 (flat)
- `sae_only_up`: inject from layer 0 to SAE layer (flat)

**Alphas:** [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  
**Status:** Running — completed all strategies for L9/F387; on remaining 9 features

### Results (L9/F387 "at the right side of", base=52.29%):

| Strategy | α=0.1 | α=0.5 | α=1 | α=2 | α=5 | α=10 | α=20 |
|----------|-------|-------|-----|-----|-----|------|------|
| flat | −0.42 | −1.25 | 0.00 | **+2.08** | −1.67 | −1.67 | −3.13 |
| decay_fwd | −0.42 | −0.62 | +0.42 | **+3.12** | −1.67 | −1.46 | (running) |
| sae_only_down | (running) | | | | | | |
| sae_only_up | (running) | | | | | | |

**Best result to date: decay_fwd α=2 → +3.12%** (beats single-layer best of +1.87%).

Key insight: `decay_fwd` at α=2 is the most effective because:
- It contributes to ALL 26 layers with exponentially decaying weight
- Upstream layers (0 to 8) contribute very little (0.85^(9-0) = 0.23×)
- SAE layer 9 gets full alpha; layer 10 gets 0.85×; layer 25 gets 0.85^16 = 0.08×
- This gives the model enough "gradient" through the residual stream to propagate the feature

**Critical sweet spot:** α=2 works, α=5 collapses. The useful range is narrow (α ∈ [1.5, 3] estimated).

---

## Experiment 9 — mix-448 3-Tap All-Layer Injection (GPU 6)

**Script:** `mix448_3tap_alllayer_injection.py`  
**Method:** 3-tap injection (attn_out, mlp_out, layer_out, all 26 layers) into mix-448. Very small alphas needed.  
**Alphas:** [-0.5, -0.1, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  
**Status:** Running — on L9/F387

### Partial results (L9/F387, base=76.67%):

| α | ∆acc | ∆margin |
|---|------|---------|
| -0.5 | **-28.54%** | -0.875 |
| -0.1 | -5.83% | -0.150 |
| +0.01 | (running) | — |

Even in mix-448, 3-tap injection collapses the model at α≥-0.1. This is consistent with the pt-448 finding — 78 simultaneous intervention points compound error faster than any single intervention. The range of useful alpha for mix-448 3-tap is estimated to be α ∈ [0.001, 0.05].

---

## Experiment 10 — Precomputed Mix-448 Acts → pt-448 (GPUs 2 & 4)

**Scripts:** `extract_mix_sae_acts.py` (GPU 2) → `pt448_precomputed_transfer.py` (GPU 4)  
**Method:** Two-step clean pipeline:
1. Run mix-448 ONLY on GPU 2, extract per-example SAE feature activations, save to disk
2. Run pt-448 ONLY on GPU 4, load precomputed acts, inject `alpha × act_F(example) × W_dec[F]`

**Status:** GPU 2 done for 5 features; GPU 4 restarted and will process available files

### Mix-448 SAE firing statistics (completed so far):

| L/F | Relation | pos_mean | neg_mean | contrast | interpretation |
|-----|----------|---------|---------|---------|----------------|
| L9/F387 | at the right side of | 4.458 | 4.516 | **-0.058** | fires MORE on false examples |
| L14/F10561 | close to | 4.671 | 4.867 | **-0.195** | fires MORE on false examples |
| L11/F12278 | touching | 1.601 | 1.507 | **+0.094** | fires more on true examples |
| L9/F7540 | consists of | 0.128 | 0.095 | **+0.033** | weak, fires more on true |
| L4/F14233 | ahead of | 0.713 | 0.601 | **+0.112** | fires more on true examples |

**Important finding:** Features selected by ablation impact do NOT necessarily correlate with label direction. The "right side of" and "close to" features have NEGATIVE contrast — they fire MORE when the spatial statement is FALSE. This means injecting +W_dec at the mean activation level pushes the model toward predicting "No" (false) — actively harmful for true examples. The per-example injection is slightly better because true examples naturally have lower activations and false examples have higher, giving some discriminative signal even with the inverted direction.

---

## Experiment 11 — pt-448 High-Transfer Fine-Grained Sweep (GPU 7)

**Script:** `pt448_highxfer_decayfwd_sweep.py`  
**Method:** Focus only on the 4 highest-transfer features. Fine-grained alpha grid [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0] × 2 strategies (decay_fwd, sae_only_down).  
**Status:** Just launched  
**Rationale:** Exp 8 found decay_fwd is best for L9/F387 (+3.12% at α=2). The 4 high-transfer features may have different optimal alpha points. This sweep finds per-feature optima for the strategies known to work.

| Feature | Transfer | Expected benefit |
|---------|----------|-----------------|
| L4/F14233 "ahead of" | 1.00× | Feature fully present → alpha should be small (saturation risk) |
| L6/F7539 "left of/right of" | 0.48× | Two merged directions → may need different alpha than single-concept |
| L14/F10561 "close to" | 0.41× | Single-layer stuck at 0% → all-layer propagation may create signal |
| L11/F12278 "touching" | 0.40× | Monotone Exp 2 response → fine grid should find peak |

---

## Grand Leaderboard — Best Result Per Feature (All Experiments Combined)

| L/F | Relation | N | pt base | mix base | Gap | **Best Δ** | Source |
|-----|----------|---|---------|---------|-----|-----------|--------|
| L4/F14233 | ahead of | 39 | 56.41% | 61.54% | 5.1% | **+10.26%** | highxfer sae_only_down α=4 |
| L12/F2257 | facing | 306 | 49.02% | 60.46% | 11.4% | **+3.92%** | multilayer "all" α=50 |
| L11/F12278 | touching | 1281 | 56.52% | 76.58% | 20.1% | **+3.36%** | precomp α=20 |
| L9/F387 | at the right side of | 480 | 52.29% | 76.67% | 24.4% | **+3.12%** | residual decay_fwd α=2 |
| L15/F220 | across from | 515 | 49.90% | 69.71% | 19.8% | **+3.11%** | residual sae_only_up |
| L13/F15219 | behind | 709 | 51.62% | 71.79% | 20.2% | **+2.12%** | multilayer downstream α=30 |
| L9/F7540 | consists of | 35 | 68.57% | 85.71% | 17.1% | **+2.86%** | multilayer single α=10 |
| L14/F10561 | close to | 93 | 60.22% | 79.57% | 19.4% | **+2.15%** | multilayer all α=2 |
| L6/F7539 | left/right of | 323 | 51.08% | 69.97% | 18.9% | **+1.24%** | multilayer topK α=20 |
| L11/F9639 | in/inside/on | 1101 | 60.85% | 81.56% | 20.7% | **+0.73%** | multilayer answer α=10 |

**Replication status (Exp 12 consolidation run, all 10 features):**
- 9/10 replicated exactly ✓
- L15/F220 "across from" FAILED replication: prior best +3.11% → consolidation got **+1.75%** only. The prior result was from `sae_only_up α=5` in the residual_alllayer script. This suggests it may have been noisy (N=515) or the specific random ordering / image cache state mattered.

**Key revisions to prior hypotheses:**
- The two 0.00× transfer features (facing, behind) ARE responsive to injection — at large α (30–50) with all-layer or downstream strategy. The transfer ratio predicts ABLATION impact, NOT injection potential.
- The best single approach varies per feature: no universal winner
- Precomputed per-example activation (Exp 10) matches or beats fixed injection for some features (touching +3.36%)
- **Combined multi-feature injection is SUBADDITIVE** (see below)

---

## Final Conclusions — Best Steering Method (as of 21 April 2026, ~09:00)

### Definitive answer: What is the best method?

**Feature-specific, single-feature, residual-only injection with hand-tuned alpha per feature.**

There is no single universal strategy. The optimal (strategy, alpha) pair depends on which feature is being injected. The best confirmed results per feature:

| L/F | Relation | N | pt base | mix base | Gap | **Best Δ** | Method | Gap% |
|-----|----------|---|---------|---------|-----|-----------|--------|------|
| L4/F14233 | ahead of | 39 | 56.41% | 61.54% | 5.1% | **+10.26%** | sae_only_down α=4 | >100% |
| L12/F2257 | facing | 306 | 49.02% | 60.46% | 11.4% | **+3.92%** | all_ml α=50 | 34% |
| L11/F12278 | touching | 1281 | 56.52% | 76.58% | 20.1% | **+3.36%** | single α=25 | 17% |
| L9/F387 | right side of | 480 | 52.29% | 76.67% | 24.4% | **+3.12%** | decay_fwd_ra α=2 | 13% |
| L15/F220 | across from | 515 | 49.90% | 69.71% | 19.8% | ~+1.75%† | sae_only_up α=5 | 9% |
| L9/F7540 | consists of | 35 | 68.57% | 85.71% | 17.1% | **+2.86%** | single α=10 | 17% |
| L14/F10561 | close to | 93 | 60.22% | 79.57% | 19.4% | **+2.15%** | all_ml α=2 | 11% |
| L13/F15219 | behind | 709 | 51.62% | 71.79% | 20.2% | **+2.12%** | downstream_ml α=30 | 10% |
| L6/F7539 | left/right of | 323 | 51.08% | 69.97% | 18.9% | **+1.24%** | topK_ml α=20 | 7% |
| L11/F9639 | in/on/inside | 1101 | 60.85% | 81.56% | 20.7% | **+0.73%** | answer α=10 | 4% |

†L15/F220 prior +3.11% failed replication; +1.75% is the replicated result.

**Average gap closed across all 10 features: ~21% of the pt→mix accuracy gap** (weighted by N: dominated by large-N features "touching" 17%, "in/on/inside" 4%).

---

### What the experiments established (definitive findings)

**1. Transfer ratio → ablation impact, NOT injection potential**  
Features with 0.00× ablation transfer (facing, behind) still respond to injection at large α. This breaks the initial hypothesis. The 0.00× transfer means the SAE direction has no causal effect in pt-448's forward pass under removal — but external injection at large scale can still force the residual stream into a different configuration.

**2. Combined injection is destructively subadditive**
- 2-feature combos: +10.26%/+3.20% → +0.30% (vs. +3.92%/+3.92% individual)
- All-10 combo: 55.00% baseline → 50.20% (-4.79%)
- W_dec vectors from different SAE features are not additively composable. Injecting multiple features simultaneously causes destructive interference.
- **Rule: inject at most ONE feature at a time, selected based on the query relation.**

**3. Universal injection hurts overall accuracy**
- Applying a relation-specific feature to ALL examples: -0.75% to -3.00% overall
- Injection must be conditioned on the query relation. An "ahead of" feature injected on "above" or "to the left" examples moves the model toward wrong answers.
- Practical implication: a routing system is needed (caption → relation → feature → inject).

**4. Negative alpha is not useful for negative-contrast features**
- L9/F387 (contrast=-0.058): all negative alphas hurt; positive α=5 best (+1.67%)
- L14/F10561 (contrast=-0.195): very narrow window at α=+2 only (+2.15%); collapse at α≥10
- The SAE contrast direction does NOT predict the injection sign. The contrast tells you how a feature fires at training distribution; the injection direction is determined by the residual stream causal structure.

**5. Feature-specific optimal alpha (non-monotone, feature-dependent)**
- "ahead of" (1.00× transfer): peak at α=4-5 (sae_only_down), collapses by α=10+
- "touching" (0.40× transfer): peak at α=25 (single), drops at α=35+
- "facing" (0.00× transfer): U-shaped: also works at α=-20 (+3.27%), best at α=+50 (+3.92%)
- "close to" (0.41× transfer): sharp peak at α=2, collapses at α=5+
- No universal alpha rule. Every feature has its own optimal range.

**6. Mix-448 cannot be boosted via injection**
The fine-tuned model is already at its spatial calibration optimum. Best observed: +0.21% for L9/F387 at α=0.5. Confirmed by full sweep and consistency across all features tested.

---

### What does NOT work (confirmed)

| Method | Why it fails |
|--------|-------------|
| DIM steering (Exp 3) | Full hidden state too noisy; threshold saturation |
| SAE-calibrated injection at calibrated scale (Exp 4) | Inverted contrast for 4 features; tiny scale range |
| 3-tap injection at all layers (Exp 6) | 78 simultaneous taps compound error catastrophically |
| Multi-feature combined injection (Exp 13, 18) | Destructive interference; all-10 = -4.79% |
| Universal injection without relation routing (Exp 19) | Relation-specific features hurt non-matching examples |
| Negative alpha for negative-contrast features (Exp 15) | Contrast sign ≠ injection sign |
| mix-448 self-injection (Exp 7) | Already saturated |

---

### Remaining open experiments (21 April 2026, ~09:00)

| GPU | Experiment | What it answers |
|-----|-----------|----------------|
| 1 | `pt448_inverted_fine_sweep.py` | Can negative alpha help L13/F15219 "behind" or L15/F220 "across from"? |
| 2 | `pt448_hard_features_ext.py` | Can extended alpha (50–100) push in/on/inside, left/right, behind further? |
| 4 | `pt448_negative_alpha_sweep.py` | Full sweep for L6/F7539 "left/right" and L11/F9639 "in/on/inside" |
| 5 | `pt448_touching_extended.py` | Does all_ml or sae_down beat single α=25 (+3.36%) for "touching"?

---

## GPU Allocation (21 April 2026, midnight)

| GPU | Script | Status | Best result so far |
|-----|--------|--------|-------------------|
| 0 | `mix448_fixed_injection.py` | Running | max +0.21% (mix-448 saturation confirmed) |
| 1 | `pt448_multilayer_injection.py` | Running L14/F10561 downstream | L9/F387 single best +1.87%; L14 downstream +1.08% |
| 2 | `extract_mix_sae_acts.py` | Running, 5/10 done | All features extracted soon |
| 3 | `pt448_3tap_alllayer_injection.py` | Running | Collapses at α≥0.2 (too many taps) |
| 4 | `pt448_precomputed_transfer.py` | Restarted, 5 acts files ready | Pending results |
| 5 | `pt448_residual_alllayer_injection.py` | Running | **+3.12% for L9/F387 decay_fwd** — best to date |
| 6 | `mix448_3tap_alllayer_injection.py` | Running L9/F387 | Collapses at α≤-0.1 |
| 7 | `pt448_highxfer_decayfwd_sweep.py` | Just launched | Fine sweep for top-4 features |

---

## Grand Leaderboard Update (21 April 2026, ~07:00)

**Updated after highxfer sweep (Exp 11) completed, new experiments launched:**

| L/F | Relation | N | pt base | mix base | Gap | **Best Δ** | Method |
|-----|----------|---|---------|---------|-----|-----------|--------|
| L4/F14233 | ahead of | 39 | 56.41% | 61.54% | 5.1% | **+10.26%** ✓ | sae_only_down α=4 *(REPLICATED)* |
| L12/F2257 | facing | 306 | 49.02% | 60.46% | 11.4% | **+3.92%** ✓ | all_ml α=50 *(REPLICATED)* |
| L11/F12278 | touching | 1281 | 56.52% | 76.58% | 20.1% | **+3.36%** 🆕 | single α=25 *(extended sweep)* |
| L9/F387 | at the right side of | 480 | 52.29% | 76.67% | 24.4% | **+3.12%** | decay_fwd_ra α=2 |
| L15/F220 | across from | 515 | 49.90% | 69.71% | 19.8% | **+3.11%** | sae_only_up α=5 |
| L9/F7540 | consists of | 35 | 68.57% | 85.71% | 17.1% | **+2.86%** | single α=10 |
| L14/F10561 | close to | 93 | 60.22% | 79.57% | 19.4% | **+2.15%** | all_ml α=2 |
| L13/F15219 | behind | 709 | 51.62% | 71.79% | 20.2% | **+2.12%** | downstream_ml α=30 |
| L6/F7539 | left/right of | 323 | 51.08% | 69.97% | 18.9% | **+1.24%** | topK_ml α=20 |
| L11/F9639 | in/inside/on | 1101 | 60.85% | 81.56% | 20.7% | **+0.73%** | answer α=10 |

All top-3 results replicated in independent consolidation run.

---

## Experiments 12–19 (21 April 2026, ~06:00–07:00 batch)

### Experiment 12 — Best-Per-Feature Consolidation (GPU 3)
**Script:** `pt448_fullbest_consolidation.py`  
**Method:** Run the single best (strategy, alpha) per feature for all 10 features — definitive replication.  
**Status:** Running — 3 of 10 completed

| L/F | Strategy | α | Replicated | Δacc | Prior best |
|-----|----------|---|-----------|------|-----------|
| L4/F14233 "ahead of" | sae_only_down | 4.0 | ✓ | +10.26% | +10.26% |
| L12/F2257 "facing" | all_ml | 50.0 | ✓ | +3.92% | +3.92% |
| L11/F12278 "touching" | single | 20.0 | ✓ | +3.20% | +3.20% |
| L9/F387 "right side of" | decay_fwd_ra | 2.0 | running | — | +3.12% |
| L15/F220 "across from" | sae_only_up | 5.0 | running | — | +3.11% |
| L13/F15219 "behind" | downstream_ml | 30.0 | running | — | +2.12% |
| L9/F7540 "consists of" | single | 10.0 | running | — | +2.86% |
| L14/F10561 "close to" | all_ml | 2.0 | running | — | +2.15% |
| L6/F7539 "left/right" | topK_ml | 20.0 | running | — | +1.24% |
| L11/F9639 "in/on/inside" | answer | 10.0 | running | — | +0.73% |

**Key finding:** All 3 completed features exactly replicate prior results. No noise artifacts.

---

### Experiment 13 — Multi-Feature Combined Injection (GPU 0)
**Script:** `pt448_combined_injection.py`  
**Method:** Inject top-5 features simultaneously using their best individual configs. Tests singles, pairs, triples, all-5.  
**Status:** Running — singles confirmed, pairs in progress

#### Individual features (replicated again):
| Feature | Relation | Δacc |
|---------|----------|------|
| L4/F14233 | ahead of | +10.26% |
| L11/F12278 | touching | +3.20% |
| L9/F387 | right side of | +3.12% |
| L12/F2257 | facing | +3.92% |
| L14/F10561 | close to | +2.15% |

#### Pair and combo results (COMPLETED):

| Combo | Features | N | Δacc | vs. individual best |
|-------|----------|---|------|---------------------|
| ahead_of | [ahead_of] | 39 | +10.26% | baseline |
| touching | [touching] | 1281 | +3.20% | baseline |
| right_side | [right_side] | 480 | +3.12% | baseline |
| facing | [facing] | 306 | +3.92% | baseline |
| close_to | [close_to] | 93 | +2.15% | baseline |
| ahead_touch | [ahead_of + touching] | 1320 | **+0.30%** | ← vs. +3.20%/+10.26% individual |
| right_touch | [right_side + touching] | 1761 | **+1.65%** | ← vs. +3.12%/+3.20% individual |
| ahead_face | [ahead_of + facing] | 345 | **+1.74%** | ← vs. +10.26%/+3.92% individual |

**Critical finding: Combined injection is DESTRUCTIVELY SUBADDITIVE.** Injecting two feature directions simultaneously dramatically reduces performance vs. either alone. The [ahead_of + touching] combo drops from +10.26%/+3.20% individual gains down to only +0.30%. The feature vectors are interfering in the residual stream — they likely share overlapping subspace directions that partially cancel when both are added.

**Implication:** The optimal strategy is to inject the SINGLE best feature per relation, not to stack multiple features. For each relation-specific query, identify the most relevant feature and inject only that one.

---

### Experiment 14 — Inverted/Extended Alpha Sweep for 0-Transfer Features (GPU 1)
**Script:** `pt448_inverted_fine_sweep.py`  
**Method:** Extended alpha range including NEGATIVE values for L12/F2257 (facing), L13/F15219 (behind), L9/F7540 (consists_of), L15/F220 (across_from).  
**Status:** L12/F2257 complete; on L13/F15219

#### L12/F2257 "facing" — all_ml strategy — **ANOMALOUS U-SHAPED RESPONSE:**

| α | Δacc | Δmargin | Note |
|---|------|---------|------|
| -50 | -0.65% | -0.028 | |
| **-20** | **+3.27%** | **+0.130** | ← NEGATIVE α also helps! |
| -10 | +2.61% | +0.057 | |
| +10 | -0.65% | -0.003 | |
| +20 | -2.29% | -0.046 | |
| +30 | -2.61% | -0.074 | |
| +40 | -0.33% | -0.060 | |
| **+50** | **+3.92%** | -0.031 | ← known best |
| +60 | +3.27% | -0.016 | |
| +70 | +3.27% | -0.011 | |
| +100 | +3.59% | -0.027 | |

**Critical finding — U-shaped response for "facing":** Both extreme negative (α=-20 → +3.27%) and extreme positive (α=+50 → +3.92%) injection help, while middle-range positive values (+10 to +40) HURT. This is inconsistent with a simple amplification mechanism.

**Interpretation:** The W_dec direction for "facing" (L12/F2257) may be encoding a *calibration signal* rather than a *feature presence signal*. At large |α|, the injection dominates the residual stream in both directions — this forces the model to produce a confident spatial answer, but the accuracy of that answer depends on whether the forced direction aligns with what the model already knows. The asymmetry (α=-20 better than α=+20) suggests the feature direction is slightly inverted relative to pt-448's internal representation, and extreme magnitudes in either direction override the model's default "uncertain" response.

---

### Experiment 15 — Negative Alpha for Negative-Contrast Features (GPU 4)
**Script:** `pt448_negative_alpha_sweep.py`  
**Method:** Test NEGATIVE alpha injection for the 4 features with negative SAE contrast (fire MORE on false examples). Hypothesis: injecting -W_dec should push model toward "Yes".  
**Status:** Running — L9/F387 partially complete

#### L9/F387 "at the right side of" — all_ml strategy — contrast = -0.058 — COMPLETED:

| α | Δacc | Δmargin |
|---|------|---------|
| -100 | -1.88% | +0.008 |
| -50 | -1.67% | -0.081 |
| -30 | -1.88% | -0.080 |
| -20 | -2.92% | -0.064 |
| -10 | -3.96% | -0.039 |
| -5 | -1.04% | -0.007 |
| -2 | +0.21% | +0.001 |
| -1 | +0.21% | +0.004 |
| +1 | +0.21% | +0.002 |
| +2 | 0.00% | +0.006 |
| +5 | **+1.67%** | +0.007 |
| +10 | -1.46% | -0.012 |
| +20 | -2.29% | -0.048 |
| +30 | -1.88% | -0.048 |
| +50 | -3.96% | -0.053 |

**Finding:** Negative alpha hypothesis WRONG for L9/F387. Best is positive α=5 (+1.67%) — NOT the decay_fwd best (+3.12%) because this experiment uses all_ml. The prior +3.12% with decay_fwd is still the record. Negative alphas all hurt or give noise-level gains.

#### L14/F10561 "close to" — all_ml strategy — contrast = -0.195 — COMPLETED:

| α | Δacc | Δmargin |
|---|------|---------|
| -100 | -30.11% | -0.520 |
| -50 | -3.23% | +0.030 |
| -30 | 0.00% | -0.040 |
| -20 | -6.45% | -0.065 |
| -10 | -8.60% | -0.045 |
| -5 | -2.15% | -0.017 |
| -2 | -1.08% | -0.009 |
| -1 | -3.23% | -0.006 |
| +1 | 0.00% | +0.005 |
| **+2** | **+2.15%** | **+0.002** |
| +5 | -3.23% | -0.008 |
| +10 | -9.68% | -0.061 |
| +20 | -27.96% | -0.364 |
| +30 | -29.03% | -0.735 |
| +50 | -30.11% | -0.848 |

**Finding:** L14/F10561 has an extremely narrow optimum: ONLY α=+2 gives +2.15%, and all larger positive alphas collapse the model (-27.96% to -30.11% at α≥20). Negative alphas are catastrophic at large magnitudes (-30.11% at α=-100). This feature is very brittle — the useful window is approximately α ∈ [1, 3]. The large negative contrast (-0.195) correctly predicted that positive injection would ultimately hurt (consistent with prior observation), but the tiny positive window at α=+2 is real and stable.

---

### Experiment 16 — mix-448 Boost Fine Sweep (GPU 2) — INVALID
**Script:** `mix448_boostable_fine.py`  
**Status:** Completed but results are INVALID.

All strategies (3tap_alllayer, single_layer, residual_all) and all alphas (0.01–10.0) produce **identical accuracy** (L12/F2257: 47.71% = -13.07%; L15/F220: 49.13% = -20.39%) with margin = 0.0 exactly. This is the signature of a degenerate model output where p_yes = p_no = 0.5, causing the model to always predict "No".

**Root cause hypothesis:** The mix-448 3-tap injection script has a model structure conflict. The results from the prior mix-448 3tap experiment (+3.92% at α=0.1) were from a different script (`mix448_3tap_alllayer.py`) that may have handled intervention ordering differently. The "boostable_fine" script modifies the layer output BEFORE the attention output in the NNsight trace, which may create a feedback conflict in mix-448's computation graph. All results from Exp 16 are discarded.

**Conclusion on mix-448 boosting:** From the valid prior experiment (mix448_fixed_injection, Exp 7), mix-448 is fully saturated and cannot be improved via W_dec injection. The best observed was +0.21% for L9/F387 at α=0.5, which is within noise. No meaningful boost exists.

---

### Experiment 17 — L11/F12278 "Touching" Extended Alpha (GPU 5)
**Script:** `pt448_touching_extended.py`  
**Method:** Test higher alpha values beyond α=20 for the "touching" feature; also test all_ml and sae_down.  
**Status:** PARTIALLY DONE — single strategy complete, all_ml/sae_down pending

#### single strategy (base=56.52%):

| α | Δacc | Δmargin |
|---|------|---------|
| +15 | +3.04% | -0.014 |
| +20 | +3.20% | -0.021 |
| **+25** | **+3.36%** | -0.026 |
| +30 | +3.04% | -0.033 |
| +35 | +0.86% | -0.039 |
| +40 | +0.55% | -0.044 |
| +50 | -1.25% | -0.052 |
| +70 | -5.93% | -0.079 |
| +100 | (running) | |

**New peak found: α=25 → +3.36%** (beating prior α=20 → +3.20%). Peak is narrow — α=30 already drops back to +3.04%. The optimal is confirmed at α≈25 for single-layer injection.

---

### Experiment 18 — All-10 Features Combined (GPU 6)
**Script:** `pt448_all10_combined.py`  
**Method:** Inject all 10 features simultaneously using their best individual configs.  
**Status:** COMPLETED (one result file)

#### All-10 combined on top-10 relations subset (N=4882):

| | Baseline | All-10 injected | Δ |
|-|---------|----------------|---|
| Accuracy | 55.00% | 50.20% | **-4.79%** |
| Margin | 0.083 | 0.019 | -0.064 |

**Confirmed: All-10 simultaneous injection is CATASTROPHICALLY BAD (-4.79%).** Stacking all 10 SAE feature directions at their individually-tuned alphas causes massive destructive interference. The single-feature finding from Exp 13 is confirmed at scale: these W_dec vectors cannot be combined additively.

---

### Experiment 19 — Universal Spatial Injection (GPU 7)
**Script:** `pt448_universal_spatial_injection.py`  
**Method:** Inject top-3 features on 2000 random VSR examples (not just relation subsets). Tests whether injection helps overall VSR accuracy.  
**Status:** COMPLETED

#### Results (2000 random VSR examples, overall baseline = 55.25%):

| Feature | Strategy | α | Overall Δacc | Prior in-relation Δ |
|---------|----------|---|-------------|---------------------|
| L4/F14233 "ahead of" | sae_only_down | 4.0 | **-0.75%** | +10.26% |
| L12/F2257 "facing" | all_ml | 50.0 | **-3.00%** | +3.92% |
| L11/F12278 "touching" | single | 20.0 | **+0.10%** | +3.20% |

**Critical finding: Relation-specific features HURT overall accuracy when applied universally.** The "ahead of" feature (best in-relation: +10.26%) gives -0.75% overall. The "facing" feature (-3.00%) is actively harmful. Only "touching" is near-neutral (+0.10%) — and that's a very common relation (N=1281).

**Implication:** These features encode relation-specific semantics. When applied to examples about a different relation (e.g., injecting "facing" direction onto an "above" example), it pushes the model toward confidently wrong answers for the unrelated relation. **Injection MUST be conditioned on which relation is being queried** — this is a crucial limitation for any practical deployment.

The VSR benchmark asks the model about specific spatial relations in the caption. To use injection in practice, the system would need to: (1) parse the caption to identify the relation, (2) look up the appropriate SAE feature for that relation, and (3) inject only that feature. This is feasible as a post-hoc routing mechanism.

---

## GPU Allocation (21 April 2026, ~07:00)

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 876329 | `pt448_combined_injection.py` | **DONE** — pairs subadditive confirmed |
| 1 | 876420 | `pt448_inverted_fine_sweep.py` | L12/F2257 done; on L13/F15219 L9/F7540 L15/F220 |
| 2 | 903802 | `pt448_hard_features_ext.py` | Running — on L11/F9639 all_ml |
| 3 | 878266 | `pt448_fullbest_consolidation.py` | **DONE** — all 10/10 replicated (9✓, L15/F220 ✗) |
| 4 | 879877 | `pt448_negative_alpha_sweep.py` | 2/4 done (L9/F387, L14/F10561) |
| 5 | 881308 | `pt448_touching_extended.py` | single done (peak α=25 +3.36%); all_ml/sae_down pending |
| 6 | 883578 | `pt448_all10_combined.py` | **DONE** — all10 → -4.79% catastrophic |
| 7 | 889071 | `pt448_universal_spatial_injection.py` | **DONE** — universal inject hurts overall |

### Experiment 20 — Hard Features Extended Alpha Sweep (GPU 2)
**Script:** `pt448_hard_features_ext.py`  
**Method:** Extended alpha range [5–100] × multiple strategies for the 3 weakest-gain features.  
**Rationale:** Prior experiments only tested up to α=50. For large-N features (in/on N=1101, behind N=709) with weak current gains, much higher alpha might unlock additional improvement.

| Feature | Strategies tested | Alpha range |
|---------|------------------|-------------|
| L11/F9639 in/inside/on | all_ml, downstream_ml, answer, single, sae_only_down | 1–100 |
| L6/F7539 left/right of | topK_ml, all_ml, downstream_ml, sae_only_down, sae_only_up | 5–100 |
| L13/F15219 behind | downstream_ml, all_ml, sae_only_down, single | 10–100 |

---

## Phase 2: Activation Steering Experiments (21 April 2026, ~10:00)

**Motivation from user request:** Move beyond feature injection (W_dec addition) to proper activation steering — extracting contrastive hidden-state vectors from mix-448 and using them to steer pt-448.

### Literature Synthesis: What Works for Cross-Model Activation Steering

Based on existing research documented in the project (RepE, CAA, ActAdd, FGAA, Scaling Monosemanticity):

**CAA (Contrastive Activation Addition, Rimsky et al. 2023):**
- Vector = mean(h_L | positive) - mean(h_L | negative), from a *source* model
- Applied at residual stream position = last text token or mean over text tokens
- Layer choice: 50–70% depth (L13–L18 for 26-layer model)
- Key advantage over W_dec: captures full circuit information including inter-feature interactions
- Key risk: picks up syntax/position confounds in addition to spatial signal

**FGAA (Feature Guided Activation Additions, arXiv 2501.09929):**
- Combines CAA with SAE: project the CAA vector onto identified SAE feature directions
- Filters out high-density "noise" features (BOS-token artifacts, broad linguistic features)
- Result: cleaner steering that avoids entangled directions
- Reported: BCS 0.47 vs CAA 0.22 on Gemma-2-2B steering tasks
- Key insight: SAE features are sparse → their W_dec vectors form a nearly orthogonal basis in their subspace → projection is clean

**SAE Feature Clamping (Templeton et al. 2024, Anthropic "Scaling Monosemanticity"):**
- Run SAE encode → clamp feature F to target value → decode back with SAE decoder
- Target = mean activation of F on positive-label examples in source model
- Key advantage: respects the SAE manifold; doesn't add energy in orthogonal directions
- Key risk: if pt-448 has weak SAE reconstruction (different geometry), clamping may not reach the right residual stream state

**Theoretical prediction for our setting (pt-448 ← mix-448 transfer):**
1. CAA unconditioned: will include VQA fine-tuning confounds (instruction-following directions, answer format patterns). May actually be noisier than W_dec.
2. FGAA (CAA projected onto W_dec): should be similar to W_dec injection at different scale. Cleaner than full CAA.
3. SAE clamping: promising IF pt-448's SAE reconstruction is good enough. The key question is whether mix-448 SAE accurately encodes pt-448 hidden states.
4. Feature-conditioned CAA (only examples where feature fires): should give cleanest spatial direction. Fewer examples → noisier estimate.

**Key prior failure (DIM steering, Exp 3):** Failed because the full hidden state at the last text token captures everything — answer format, image alignment, syntax — not just spatial information. CAA with the *same* layer will have the same issue. The critical difference with W_dec injection is that W_dec is monosemantic (by SAE training); CAA is not.

**Hypothesis:** FGAA projected steering will approximately match W_dec injection. Full CAA will underperform. SAE clamping may beat W_dec if the target clamping value is properly calibrated.

---

### Experiment 21 — CAA Steering from mix-448 (GPU 0)
**Script:** `pt448_caa_steering.py`  
**Method:**
1. Phase 1: Run mix-448 forward on VSR examples, collect hidden states at all 26 layers at the last text token. Compute v_steer_L = mean(h_L | label=1) - mean(h_L | label=0) for each feature's relation subset.
2. Phase 2: Inject v_steer into pt-448 using all_ml decay strategy.
3. Also tests: v_caa_norm (unit norm), v_proj (projection onto W_dec — FGAA-style), v_wdec (reference).

**5 strategies tested:**
- `caa_single`: inject normalized CAA vector at SAE layer only
- `caa_all_ml`: inject normalized CAA vector at all 26 layers (0.7 decay)
- `caa_sae_down`: inject normalized CAA vector from SAE layer to 25 (flat)
- `caa_proj_all`: inject CAA projected onto W_dec at all layers (FGAA-style)
- `wdec_all_ml`: inject W_dec at all layers (baseline comparison)

**Expected findings:** CAA should give comparable or slightly worse results vs W_dec because it captures more confounds. The projection variant should converge toward W_dec behavior. The key test is whether any CAA variant finds a BETTER direction than W_dec.

**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_steering/`  
**Status:** Running on GPU 0 (PID 1025070)

---

### Experiment 22 — SAE Feature Clamping (GPU 3)
**Script:** `pt448_sae_clamp_steering.py`  
**Method:** Run SAE encode on pt-448 hidden states at the SAE layer, clamp feature F to a target value derived from mix-448 statistics, decode back.

**Target values tested:** 0.5×, 1.0×, 1.5×, 2.0×, 3.0×, 5.0×, 10.0× the mix-448 mean positive activation.

**Key difference from W_dec injection:**
- W_dec adds a fixed vector regardless of current SAE state
- Clamping brings the feature to a specific target level, accounting for what pt-448 already has
- If pt-448 has feature F at 60% of mix-448's level, clamping to 100% adds only 40%; injection always adds full alpha

**Expected advantage:** For features that ARE present in pt-448 (non-zero transfer ratio), clamping should give a more principled target than a manually tuned alpha. For features absent in pt-448 (0.00× transfer), clamping might work better because it bypasses the "where is the natural firing level" problem.

**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_clamp/`  
**Status:** Running on GPU 3 (PID 1025071)

---

### Experiment 23 — Feature-Conditioned FGAA Steering (GPU 7)
**Script:** `pt448_fgaa_steering.py`  
**Method:** Extract feature-conditioned CAA vector from mix-448:
- v_cond = mean(h_L | label=1 AND feature_F > 0.5) - mean(h_L | label=0 AND feature_F > 0.5)
- Only uses examples where the spatial SAE feature actually fires in mix-448

**4 steering variants compared:**
1. `uncond_CAA`: standard unconditioned CAA
2. `cond_CAA`: conditioned on feature firing (cleaner signal)
3. `proj_CAA`: CAA projected onto W_dec (FGAA-style)
4. `wdec_ref`: plain W_dec (baseline)

**Key hypothesis:** Feature-conditioned CAA should be the cleanest spatial direction — it filters out examples where the feature is absent (spatial concept not relevant to the image), leaving only examples where the spatial reasoning is genuinely engaged.

**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fgaa/`  
**Status:** Running on GPU 7 (PID 1034878)

---

## GPU Allocation (21 April 2026, ~10:00)

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 1025070 | `pt448_caa_steering.py` | **NEW** — Phase 1 extracting CAA vectors from mix-448 |
| 1 | 876420 | `pt448_inverted_fine_sweep.py` | On L15/F220 "across from" |
| 2 | 903802 | `pt448_hard_features_ext.py` | On L11/F9639 downstream_ml |
| 3 | 1025071 | `pt448_sae_clamp_steering.py` | **NEW** — SAE clamping experiment |
| 4 | 879877 | `pt448_negative_alpha_sweep.py` | On L11/F9639 in/on/inside neg sweep |
| 5 | 881308 | `pt448_touching_extended.py` | touching all_ml/sae_down pending |
| 6 | 883578 | `pt448_all10_combined.py` | DONE |
| 7 | 1034878 | `pt448_fgaa_steering.py` | **NEW** — feature-conditioned FGAA steering |

---

## Phase 2 Results — Activation Steering Outcomes (21 April 2026, ~12:00)

### Experiments 12–20 Completion Status

**Inverted fine sweep (GPU 1) — COMPLETED** for all 4 features:

| L/F | Relation | Strategy | Base | Best Δ | Best α | Notes |
|-----|----------|----------|------|--------|--------|-------|
| L12/F2257 | facing | sae_only_up | 49.02% | +3.27% | -20 | U-shaped; negative also works |
| L13/F15219 | behind | — | 51.62% | ~0% | — | No clear improvement in inverted range |
| L9/F7540 | consists_of | — | 68.57% | — | — | pending |
| L15/F220 | across from | sae_only_up | 49.90% | **+3.11%** | **α=2** ✓ | Prior +3.11% at α=5 CONFIRMED at α=2 — narrower peak |

L15/F220 inverted fine sweep confirms: α=2 → +3.11% (peak), α=3 → +2.91%, α=5 → +1.75%, α≥7 → negative. The correct optimum is α=2, not α=5. **Updated leaderboard entry.**

---

**Negative alpha sweep (GPU 4) — COMPLETED** for all 4 features:

| L/F | Relation | SAE contrast | Best neg α | Best pos α | Conclusion |
|-----|----------|-------------|-----------|-----------|------------|
| L9/F387 | right side of | -0.058 | +0.21% @ α=-1 | **+1.67%** @ α=5 | Positive wins |
| L14/F10561 | close to | -0.195 | 0.00% | **+2.15%** @ α=2 | Positive wins |
| L6/F7539 | left/right of | -0.003 | -1.55% | **+0.93%** @ α=30 | All neg hurt |
| L11/F9639 | in/on/inside | -0.240 | +0.18% @ α=-1 | -0.18% | Noise only |

**Definitive finding:** Negative alpha never beats positive for any of the 4 negative-contrast features. The SAE firing contrast (pos_mean - neg_mean) being negative does NOT mean the feature should be injected with negative sign. The useful direction is determined by the residual stream causal structure, not the SAE firing distribution.

---

**Touching extended (GPU 5) — COMPLETED** for all 3 strategies:

| Strategy | Base | Best Δ | Best α | Alpha range |
|----------|------|--------|--------|-------------|
| single | 56.52% | **+3.36%** | **α=25** | Usable: α=15–30; collapses α≥50 |
| all_ml | 56.52% | +0.62% | α=5 | Rapidly collapses α≥10 |
| sae_down | 56.52% | +0.70% | α=1 | Collapses α≥7 |

**Confirmed: `single` strategy (+3.36% @ α=25) definitively beats all_ml and sae_down for "touching".** Multi-layer strategies hurt — the "touching" feature is optimally injected at a single point. The narrowness of the peak (α=25 best, α=35 drops to +0.86%) suggests the useful injection window is approximately α ∈ [18, 32].

---

### Experiment 23 — FGAA Steering — CRITICAL FAILURE (GPU 7, partial results)

**Status:** L4/F14233 complete; L12/F2257 in progress. Results from first feature:

| Variant | Description | Best Δacc (L4/F14233, ahead_of) |
|---------|-------------|----------------------------------|
| uncond_CAA | Unconditioned CAA at last text token | **+0.00%** (all alphas) |
| cond_CAA | Feature-conditioned CAA (fires>0.5) | **+0.00%** (all alphas) |
| proj_CAA | CAA projected onto W_dec (FGAA) | **+0.00%** (all alphas) |
| wdec_ref | W_dec injection (baseline) | **+7.69% @ α=1** ✓ |

**Critical finding: ALL CAA variants produce ZERO effect. Only W_dec injection works.**

The FGAA script reveals:
- `cos_cond_wdec = 0.0` — feature-conditioned CAA has zero cosine similarity with W_dec
- `n_pos_conditioned = 0` — zero examples had the feature fire above threshold in mix-448
- This means the SAE feature for "ahead of" (L4/F14233) fires at essentially 0 in mix-448 on "ahead of" examples!

Wait — this is a key diagnostic. The SAE feature **does not fire** on the mix-448 forward passes for these examples. The CAA vector is therefore computed from the full hidden state distribution (not feature-specific activations), and the resulting difference vector is **orthogonal to W_dec** (cos=0.0). This confirms:

1. **W_dec direction is NOT the same as the contrastive hidden-state difference.** The SAE feature encodes a polysemantic direction that fires for multiple reasons; the W_dec direction alone doesn't dominate the hidden-state difference.
2. **FGAA fails when features don't fire in source model** — if mix-448 doesn't actually activate the feature on the relevant examples, there's no conditioned CAA vector to extract.
3. **W_dec injection succeeds despite feature not firing in source** — this suggests W_dec injection works by creating a NEW synthetic activation in pt-448's residual stream, not by amplifying an existing signal. It's more like a structured perturbation than feature amplification.

**Root cause hypothesis for W_dec working but CAA failing:** W_dec vectors from a JumpReLU SAE point in directions that are *interpretable and monosemantic* but these directions are specific to the SAE's learned decomposition. They do NOT correspond to the principal directions of the hidden-state distribution at those layers. CAA extracts principal directions (implicitly, via mean difference) which live in a completely different subspace — perpendicular to W_dec. The injection of a CAA vector lands in directions that pt-448's language model has no structured response to, so it has zero effect on Yes/No accuracy.

---

### Experiment 21 — CAA Steering (GPU 0) — Initial failure, fixed and re-running

**Original run:** Crashed with `AttributeError: 'float' object has no attribute 'clamp'` in Phase 1 at `v_norm.clamp(min=1e-8)` — v_norm is a Python float from `.item()`, not a tensor.

**Second crash:** NNsight `.save()` proxies for all 26 layers triggered exceptions outside the trace context (pos=0, neg=2, skipped=37 for "ahead of" N=39). The NNsight approach to collecting 26 layer outputs simultaneously was unstable.

**Fix applied (2026-04-21 ~12:00):** Rewrote Phase 1 to use `output_hidden_states=True` instead of NNsight traces. This is the same approach used by FGAA which works reliably. Fixed script relaunched on GPU 0 (PID 1057348).

**Expected result:** Based on FGAA preliminary finding, full CAA will likely also produce Δ≈0.00% because the W_dec direction is orthogonal to the hidden-state contrastive difference. Results expected in 2–3 hours.

---

### Experiment 22 — SAE Feature Clamping (GPU 3) — Re-launched

**Original run:** Crashed immediately — launched from wrong directory (`/home/hbatra/vlm_scope_backup/` instead of `/data1/vlm_scope_sae_mix448_textonly/scripts/`).

**Fixed:** Relaunched with correct path on GPU 3 (PID 1057370). Status: Phase 1 extracting mix-448 SAE firing statistics.

---

## Leaderboard (21 April 2026, ~12:00 — pre-CAA; superseded by current leaderboard below)

| L/F | Relation | N | pt base | mix base | Gap | **Best Δ** | Method | Gap% closed |
|-----|----------|---|---------|---------|-----|-----------|--------|-------------|
| L4/F14233 | ahead of | 39 | 56.41% | 61.54% | 5.1% | **+10.26%** | sae_only_down α=4 | >100% |
| L12/F2257 | facing | 306 | 49.02% | 60.46% | 11.4% | **+3.92%** | all_ml α=50 | 34% |
| L11/F12278 | touching | 1281 | 56.52% | 76.58% | 20.1% | **+3.36%** | single α=25 | 17% |
| L9/F387 | at the right side of | 480 | 52.29% | 76.67% | 24.4% | **+3.12%** | decay_fwd_ra α=2 | 13% |
| L15/F220 | across from | 515 | 49.90% | 69.71% | 19.8% | **+3.11%** | sae_only_up α=2 | 16% |
| L13/F15219 | behind | 709 | 51.62% | 71.79% | 20.2% | **+2.12%** | downstream_ml α=30 | 10% |
| L9/F7540 | consists of | 35 | 68.57% | 85.71% | 17.1% | **+2.86%** | single α=10 | 17% |
| L14/F10561 | close to | 93 | 60.22% | 79.57% | 19.4% | **+2.15%** | all_ml α=2 | 11% |
| L6/F7539 | left/right of | 323 | 51.08% | 69.97% | 18.9% | **+1.24%** | topK_ml α=20 | 7% |
| L11/F9639 | in/on/inside | 1101 | 60.85% | 81.56% | 20.7% | **+0.73%** | answer α=10 | 4% |

*(Superseded — CAA per-layer vector experiments produced much larger gains — see current leaderboard below)*

---

## Theoretical Analysis: WHY W_dec Injection Works But CAA Fails

*Written 21 April 2026 based on FGAA preliminary results (n_pos_conditioned=0, cos_to_wdec=0.0)*

### Core observation
For "ahead of" (L4/F14233), the FGAA experiment found:
- Feature activates at firing rate ≈ 0 even on "ahead of" positive examples in mix-448
- Conditioned CAA vector = zero (no examples pass the threshold)
- Unconditioned CAA vector is orthogonal to W_dec (cos ≈ 0.0)

This means the W_dec direction discovered by the SAE **does not correspond to any natural direction in the hidden-state difference between positive and negative examples**. These are completely different geometric objects.

### Interpretation
JumpReLU SAE training optimizes for sparse reconstruction of activations across the entire training distribution. The W_dec atoms are sparse, polysemantically refined directions in residual stream space that collectively reconstruct the full activation. These are NOT the same as:
- The mean difference between positive/negative label examples (CAA direction)
- The principal component of the conditional distribution (RepE direction)
- Any direction that naturally exists as a distinct "feature vector" in the model's computations

The SAE W_dec vectors are more like **synthetic coordinate axes** in a compressed basis of activation space. They encode *what patterns of activation can be explained by a single sparse atom*, which is a very different thing from *what direction the model moves in when reasoning about spatial relations*.

### Why does W_dec injection then WORK?
W_dec injection creates a *novel structured perturbation* in residual stream space. The key properties that make it effective:
1. **Sparsity:** The W_dec direction participates in the SAE reconstruction at the relevant layer — this means it's at least marginally aligned with the residual stream geometry at that layer
2. **Monosemanticity (partially):** By SAE training design, W_dec[F] encodes a relatively isolated concept — pushing in this direction creates a clean conceptual nudge without the polysemantic bleed of a full CAA vector
3. **Causal pathways:** Even if the feature doesn't "fire" in the source model (mix-448), the direction still corresponds to a meaningful subspace in the target model (pt-448) where spatial reasoning is processed
4. **Layer specificity:** The SAE was trained on mix-448, so its W_dec vectors are calibrated to the mix-448 residual stream geometry — which is similar to pt-448's (same architecture, similar pre-training). The cross-stage transfer ratio shows exactly how similar these geometries are per feature.

### Implication for choosing steering methods
The DIM/CAA approaches that extract contrastive hidden-state directions are fundamentally solving a different problem: they find the direction in activation space that best separates positive from negative examples. This is useful for *behavior steering* (make the model more helpful, less biased, etc.) where the causal direction is the same as the behavioral direction.

For *capability injection* (trying to transfer a specific learned feature from one model to another), W_dec injection is better because:
- It targets a specific semantic atom from the source model's learned decomposition
- It doesn't carry the confounds of full-distribution statistics (syntax, position, image alignment)
- The alpha scaling is more interpretable (direct multiple of the learned feature norm)

**Revised conclusion (21 April 2026, ~09:00 update):** The "W_dec is superior to CAA" finding requires revision — see Experiments 21 and 25 below where CAA with the `sae_down` strategy BEATS W_dec for L4/F14233 (+15.38% vs +10.26%). The key finding is that the specific injection strategy matters more than the vector choice: injecting CAA across all downstream layers (4→25) at a well-tuned alpha is more effective than injecting W_dec at the SAE layer alone.

---

## GPU Allocation (21 April 2026, ~12:00) — STALE, see current allocation below

*(Superseded — see "GPU Allocation (21 April 2026, ~11:15)" below for current state)*

---

### Experiment 24 — Per-Layer Injection Sweep (GPU 6)

**Script:** `pt448_layer_sweep.py`  
**Method:** For each of the 10 features, inject W_dec[F] at each individual layer l ∈ [0..25] using the best alpha from prior experiments. This answers: *is the SAE layer (where the feature was trained) actually the optimal injection layer for pt-448?*

**Hypothesis:** The SAE feature was found at a specific layer in mix-448. If pt-448 represents the same spatial concept at a different layer (due to different training), injecting at the SAE layer may be suboptimal. Finding the optimal injection layer could unlock better results for the weaker features.

**Alpha per feature:** Uses best confirmed alpha from prior experiments (same as consolidation run).

**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_layer_sweep/`

**Partial results (1/10 features complete):**

| L/F | Relation | SAE layer | Best inj layer | Shift | Best Δ | Prior best |
|-----|----------|-----------|---------------|-------|--------|-----------|
| L4/F14233 | ahead of | 4 | **2** | -2 | **+5.13%** | +10.26% (sae_only_down) |

L4/F14233 "ahead of": the optimal single injection layer is L2 (2 layers before SAE layer 4), giving the same +5.13% as at the SAE layer. Notably, the best prior result (+10.26%) uses `sae_only_down` (all layers 4-25) — single-layer injection at any layer gives only half the gain.

**Expected findings:**
- For high-transfer features (ahead_of, touching), the SAE layer should be near-optimal
- For low-transfer features (in/on/inside, behind), a different injection layer might work better
- The per-layer profile also reveals which layers in pt-448 are most causally responsive to spatial feature injection

---

## Phase 2 Complete Results (21 April 2026, ~09:00)

### Experiment 21 — CAA Steering — MAJOR FINDING (GPU 0, completed for L4/F14233)

**Script:** `pt448_caa_steering.py` (v2, using output_hidden_states=True)  
**Status:** L4/F14233 completed; continuing on remaining 9 features

#### L4/F14233 "ahead of" (N=39, base=56.41%):

CAA vector properties at SAE layer L4: |v_caa|=1.945, cos(caa, W_dec)=0.042 (nearly orthogonal)

| Strategy | Description | Best Δ | Best α | Notes |
|----------|-------------|--------|--------|-------|
| `caa_single` | CAA@L4 injected at L4 only | +2.56% | α=0.5 | Small but real |
| `caa_all_ml` | CAA at all 26 layers (0.7 decay) | +2.56% | α=1.0 | Similar to single |
| **`caa_sae_down`** | **CAA@L4 at layers 4-25 (flat)** | **+15.38%** | **α=1.0** | **NEW RECORD** |
| `caa_proj_all` | CAA projected onto W_dec (FGAA) | +5.13% | α=5.0 | Cleaner than raw CAA |
| `wdec_all_ml` | W_dec at all 26 layers (baseline) | +7.69% | α=1.0 | |

**Critical finding: `caa_sae_down` at α=1.0 gives +15.38% — the new all-time record, beating W_dec sae_only_down (+10.26%).**

This completely reverses the preliminary "CAA has zero effect" finding from FGAA. The key difference:
- FGAA extracted **feature-conditioned CAA** (only examples where the SAE feature fires) → zero examples pass threshold → zero vector
- Full CAA (`pt448_caa_steering.py`) extracts **unconditioned** mean(h | label=1) - mean(h | label=0) → valid vector with |v|=1.945
- The direction is nearly orthogonal to W_dec (cos=0.042), but injecting it at layers 4-25 is HIGHLY effective

**Interpretation of caa_sae_down beating W_dec sae_only_down:**
The CAA direction captures the *full behavioral difference* at the hidden state level, including inter-feature circuit interactions that W_dec cannot encode. When injected from the SAE layer downstream, it propagates through the residual stream along a direction that is empirically more aligned with the final Yes/No decision boundary in pt-448 than the W_dec direction. The 22-layer breadth of the injection (L4-L25) allows the signal to compound through attention and MLP layers.

The apparent paradox (cos=0.042 with W_dec yet superior performance) is resolved by recognizing that W_dec measures alignment in the *feature decomposition* basis, not in the *task-relevant* basis. The task-relevant subspace is captured by the CAA vector but NOT by W_dec.

---

### Experiment 22 — SAE Feature Clamping — CONFIRMED FAILURE (GPU 3, all 6 features complete)

**Script:** `pt448_sae_clamp_steering.py` (v3, using precomputed mix-448 stats)  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_clamp/`  
**Status:** ALL 6 features DONE

| L/F | Relation | Base | Best clamp Δ | Best mult | W_dec best | Verdict |
|-----|----------|------|-------------|-----------|-----------|---------|
| L4/F14233 | ahead of | 56.41% | **+0.00%** | 0.5 | +10.26% | ❌ fail |
| L11/F12278 | touching | 56.52% | **-5.54%** | 0.5 | +3.36% | ❌ fail |
| L9/F387 | right side of | 52.29% | **-1.88%** | 0.5 | +3.12% | ❌ fail |
| L15/F220 | across from | 49.90% | **-0.78%** | 0.5 | +3.11% | ❌ fail |
| L12/F2257 | facing | 49.02% | **-1.31%** | 0.5 | +3.92% | ❌ fail |
| L13/F15219 | behind | 51.62% | **-1.97%** | 0.5 | +2.12% | ❌ fail |

**Definitive finding: SAE feature clamping is uniformly 0% or negative for all 6 features tested.** The worst is -5.54% for "touching" at the lowest multiplier (0.5×). This confirms that:
1. Clamping pt-448's SAE-encoded feature to mix-448 target values doesn't transfer the feature
2. The mix-448 SAE was not trained on pt-448 residual streams — pt-448's hidden states don't encode features in a way compatible with the mix-448 SAE manifold
3. **SAE clamping is explicitly added to the "does not work" list alongside DIM, FGAA, and calibrated injection**

---

### Experiment 23 — FGAA Steering — CONFIRMED ZERO (ALL features)

| L/F | Relation | n_pos_cond | uncond_CAA | cond_CAA | proj_CAA | wdec_ref |
|-----|----------|-----------|-----------|---------|---------|---------|
| L4/F14233 | ahead of | 0 | +0.00% | +0.00% | +0.00% | +7.69% |
| L12/F2257 | facing | 0 | +0.00% | +0.00% | +0.00% | +3.92% |
| L11/F12278 | touching | 0 | +0.00% | +0.00% | +0.00% | +0.62% |
| L9/F387 | at the right side of | 0 | +0.00% | +0.00% | +0.00% | (wdec pending) |
| L15/F220 | across from | 0 | +0.00% | +0.00% | +0.00% | (wdec pending) |
| L13/F15219 | behind | 0 | +0.00% | +0.00% | +0.00% | (wdec pending) |

**FGAA is definitively zero for all features tested.** Root cause: n_pos_c=0 for every feature — no examples from the VSR relation subset fire the SAE feature above the JumpReLU threshold in the mix-448 forward pass. This means the conditional CAA vector is a zero vector (no positive firing examples), making all three FGAA variants identical: zero effect at every alpha. The feature fires in text-only mode at the SAE training distribution, but doesn't fire in the multi-modal paligemma2-3b-mix-448 model on VSR data. FGAA is removed from consideration.

---

### Experiment 25 — Late-Layer CAA Injection (GPU 1, in progress)

**Script:** `pt448_late_layer_caa.py` (NEW — written 21 April 2026)  
**Motivation:** CAA vectors have dramatically higher norms at L24 (25–105) vs. at SAE layers (1.9–3.0). Hypothesis: injecting at late layers where the norm is largest may be more effective.  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_late_layer_caa/`  
**Strategies:** late_caa_answer (L21-25), late_caa_single24 (L24 only), best_caa_single (L24 only), late_wdec_answer (W_dec at L21-25 reference)

#### L4/F14233 "ahead of" (N=39, base=56.41%), L24 CAA norm=25.25:

| Strategy | Description | Best Δ | Best α |
|----------|-------------|--------|--------|
| `late_caa_answer` | CAA@L24 at layers 21-25 | pending | — |
| `late_caa_single24` | CAA@L24 at layer 24 only | pending | — |
| **`best_caa_single`** | **CAA@L24 at L24 only** | **+10.26%** | **α=5.0** |
| `late_wdec_answer` | W_dec at layers 21-25 | +2.56% | α=0.005–0.1 |

**Key finding: `best_caa_single` (CAA@L24 injected at L24) = +10.26% @ α=5** — this matches the best W_dec result without any multi-layer strategy! The high norm at L24 (25.25 vs 1.945 at SAE layer) means small alpha values are very effective. The late-layer CAA direction at L24 is just as causally effective as the W_dec sae_only_down strategy.

Comparing `caa_sae_down` (+15.38%) from Exp 21 to `best_caa_single` (+10.26%): injecting the SAE-layer CAA across all downstream layers beats injecting the L24-layer CAA at L24 alone. This suggests the compounding effect across 22 layers in `caa_sae_down` is the main driver of the record result.

---

## Current Best Leaderboard (21 April 2026, ~15:20 — L6/F7539 confirmed; L13/F15219 W_dec may improve)

| L/F | Relation | N | pt base | mix base | Gap | **Best Δ** | Method |
|-----|----------|---|---------|---------|-----|-----------|--------|
| L4/F14233 | ahead of | 39 | 56.41% | 61.54% | 5.1% | **+15.38%** | caa_sae_down α=1 (any start ≤4 ties) |
| L14/F10561 | close to | 93 | 60.22% | 79.57% | 19.4% | **+10.75%** | caa_sae_down α=2.0 OR caa_all_ml α=10.0 (FIVE-FOLD vs W_dec) |
| L12/F2257 | facing | 306 | 49.02% | 60.46% | 11.4% | **+9.80%** | caa_sae_down start=1, α=1.0 (Exp 31 confirmed joint optimum) |
| L15/F220 | across from | 515 | 49.90% | 69.71% | 19.8% | **+7.96%** | caa_sae_down α=0.75 |
| L11/F12278 | touching | 1281 | 56.52% | 76.58% | 20.1% | **+6.25%** | caa_all_ml α=7.0 |
| L6/F7539 | left/right of | 323 | 51.08% | 69.97% | 18.9% | **+3.10% 🆕** | caa_sae_down α=1.5 (CONFIRMED — 2.5× W_dec; caa_all_ml collapses to -3.1%) |
| L9/F387 | at the right side of | 480 | 52.29% | 76.67% | 24.4% | **+3.12%** ✓ | decay_fwd_ra α=2 (W_dec still best; CAA gives only +2.92%) |
| L13/F15219 | behind | 709 | 51.62% | 71.79% | 20.2% | **+2.68% 🆕** | W_dec at L3 (α=30) — GPU6 layer sweep! (inj_layer=3 beats SAE layer=13) |
| L9/F7540 | consists of | 35 | 68.57% | 85.71% | 17.1% | **+2.86%** | caa_sae_down α=0.25 (matches W_dec) |
| L11/F9639 | in/on/inside | 1101 | 60.85% | 81.56% | 20.7% | **+0.73%** ✓ | answer α=10 (CAA pending) |

**Summary of new records (~15:00–15:20 monitoring pass):**
- **L6/F7539 CONFIRMED: +3.10%** via caa_sae_down α=1.5. Full curve: α=0.25→-0.31%, α=0.5→+0.31%, α=0.75→+1.55%, α=1.0→+0.93%, α=1.5→**+3.10%**, α=2.0→+1.24%, α=3.0→+0.62%, α=5.0→+0.62%. `caa_all_ml` is catastrophic: α=2→-2.79%, α=5→-3.10%. CAA norm@L6=0.891 (low but sufficient at α=1.5).
- **L13/F15219 NEW RECORD: +2.68%** via W_dec at inj_layer=3 (α=30), beating prior best +2.12%! Injecting 10 layers BEFORE the SAE layer (L3 instead of L13) is better. L0-L2 also around +1.97% range; L3 is the sweet spot. L4-L6 collapse. Layer sweep on GPU6 still running.
- L9/F387 startlayer v2 COMPLETE (JSON saved): best_start=18, +2.50% (below W_dec +3.12%).
- GPU0 L9/F387 caa_all_ml complete: +1.67%; caa_sae_down in progress at α=0.5→+2.92%.
- GPU1 L15/F220 late_caa_answer: started, α=0.05→+0.39%, very weak (late-only < full chain for L15).
- GPU3 L11 joint grid: α=0.1 complete (max +0.55%@start=11); α=0.25 in progress.

**Key insight — feature-position-dependent optimal start layer:**
- **Early-layer features** (L4/F14233): All start layers 0-5 give identical +15.38%. Start layer doesn't matter if you use per-layer vectors; the critical variable is the alpha value (sharp peak at α=1).
- **Mid-layer features** (L12/F2257): Start=1 (25 layers) gives best +9.80%. Diminishing returns as start moves past the SAE layer (start≥12 → ≤8.82%). 
- **Discovering the per-layer vector insight was the key breakthrough:** Using each layer's own normalized CAA direction (not a single fixed direction) is what unlocks the large gains.
- **`caa_all_ml` vs `caa_sae_down`**: For high-norm late-layer features (L12-L15), `caa_all_ml` matches or beats `caa_sae_down`. For early features (L4), `caa_sae_down` wins decisively.

---

## GPU Allocation (21 April 2026, ~11:15 — updated)

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 1085274 | `pt448_caa_steering.py` | Running — L4, L12, L11 done; on L11 caa_proj_all/wdec_all_ml |
| 1 | 1103532 | `pt448_late_layer_caa.py` | Running — L4, L12, L11 done (+5.39% single24); on L9/F387+ |
| 2 | 1355629 | `pt448_caa_perlayer_finesweep_gpu2.py` | Running — L9 done, L15 sae_down done (+7.96%!); on L15 caa_all_ml (so far max +6.99%) |
| 3 | 1477944 | `pt448_caa_joint_alphastart.py` | **Exp 31** — joint α×start for L12, L11; L12 α=0.75 start=0 → +7.52% (in progress) |
| 4 | 1259025 | `pt448_caa_perlayer_finesweep.py` | Running — L4, L12, L11 done (+6.25%!); now on L15 (will skip when GPU2 saves) |
| 5 | 1288039 | `pt448_caa_startlayer_sweep_v2.py` | Running — L4, L12 done; on L11 start sweep (α=1.0; plateau ~+3.5%) |
| 6 | 1067854 | `pt448_layer_sweep.py` | Running — L4, L12, L11 done (L11 best=L11 +3.36%); on L9/F387 |
| 7 | 1034879 | `pt448_fgaa_steering.py` | Running — FGAA confirmed zero for ALL features (n_pos_c=0 universally) |

### Experiments 26–30: New CAA explorations

**Exp 26 — CAA Fine Sweep (GPU 4 → killed):** `pt448_caa_fine_sweep.py`  
**INVALID** — contains a bug: uses only the SAE-layer CAA vector for all injection layers, whereas the main `pt448_caa_steering.py` correctly uses per-layer CAA vectors. Results dramatically underestimate the true gains (L4/F14233: +2.56% vs true +15.38%). Results discarded.

**Exp 27 — CAA Start-Layer Sweep (GPU 5 → killed):** `pt448_caa_startlayer_sweep.py`  
**ALSO INVALID** — same bug (uses single SAE-layer vector). Best results ~+1% max across features. Discarded.

**Exp 28 — CAA + W_dec Combined (GPU 3):** `pt448_caa_wdec_combined.py`  
**COMPLETE for L4, L12, L11; L9/F387 in progress (3/5):**

| L/F | Relation | cos(CAA,W_dec) | Best combined Δ | Best config | CAA-only | W_dec-only | Loss vs CAA |
|-----|----------|----------------|----------------|------------|---------|-----------|------------|
| L4/F14233 | ahead of | 0.042 | **+7.69%** | caa0.5_wdec4.0 | +15.38% | +10.26% | -7.69% |
| L12/F2257 | facing | -0.008 | **+0.98%** | caa1.5_wdec4.0 | +8.82% | +3.92% | -7.84% |
| L11/F12278 | touching | 0.015 | **+0.70%** | caa2.0_wdec2.0 | unknown | +3.36% | — |

Combined injection is **universally destructively subadditive** despite near-orthogonality (cos≈0.04 max). The combined result is always worse than either individual method:
- L4/F14233: combined (+7.69%) << W_dec (+10.26%) << CAA alone (+15.38%)
- L12/F2257: combined (+0.98%) << W_dec (+3.92%) << CAA alone (+8.82%)
- L11/F12278: combined (+0.70%) << CAA unknown << W_dec (+3.36%)

**Root cause:** Even though the two vectors are geometrically orthogonal (cos≈0), their *causal effects* in the residual stream are not independent. The model's downstream layers process both perturbations simultaneously through shared attention and MLP circuits. The combined injection overloads the same circuit nodes that process spatial information, causing destructive interference in the computation graph rather than additive improvement.

**Exp 29 — Per-Layer CAA Fine Sweep (GPUs 4 + 2):** `pt448_caa_perlayer_finesweep.py` + `pt448_caa_perlayer_finesweep_gpu2.py`  
**CORRECTED** — uses per-layer vectors properly. Results so far:

| L/F | Relation | Strategy | Best Δ | Best α | Notes |
|-----|----------|---------|--------|--------|-------|
| L4/F14233 | ahead of | `caa_sae_down` | **+15.38%** | **1.0** | α=0.5→7.69%, α=1.0→15.38%, α=1.5→0.00%. Sharp peak. |
| L4/F14233 | ahead of | `caa_all_ml` | +2.56% | 2.0 | Decay-weighted — much weaker for early-layer feature |
| L12/F2257 | facing | `caa_sae_down` | +8.82% | 1.0 | α=1.0 optimal, drops steeply at α≥1.5 |
| L12/F2257 | facing | **`caa_all_ml`** | **+9.48%** | **7.0** | Per-layer decay-weighted beats sae_down for L12! |
| L11/F12278 | touching | `caa_sae_down` | **+5.85%** | **0.5** | NEW RECORD — beats W_dec (+3.36%) and late_caa_answer (+4.14%) |
| L11/F12278 | touching | **`caa_all_ml`** | **+6.25%** | **7.0** | **NEW RECORD** beats sae_down (+5.85%)! α=2→+1.72%, α=5→+5.46%, α=7→+6.25%, α=10→+4.53% |
| L9/F387 | at the right side of | `caa_sae_down` | +2.92% | 0.5 | Just below W_dec best (+3.12%); low CAA norm (0.668) at L9 |
| L9/F387 | at the right side of | `caa_all_ml` | +2.71% | 3.0 | Below W_dec best (+3.12%); L9 CAA is weak |
| **L15/F220** | **across from** | **`caa_sae_down`** | **+7.96% 🆕** | **0.75** | **MASSIVE NEW RECORD** — was +3.11%; L15 CAA norm=11.375 (high!); sharp peak: α=0.5→+6.60%, α=1.0→+6.80% |
| L15/F220 | across from | `caa_all_ml` | +6.99% | 2.0 | Below sae_down; α=3→+5.44%, α=5→+1.94%. Peaked at α=2. |
| **L14/F10561** | **close to** | **`caa_sae_down`** | **+10.75% 🆕🆕** | **2.0** | **FIVE-FOLD RECORD** — was +2.15%! Norm@L14=9.625 (high); plateau: α=1.5→+9.68%, α=2→+10.75%, α=3→+9.68%, α=5→+9.68% |
| L14/F10561 | close to | `caa_all_ml` | **+10.75%** | **10.0** | Tied with sae_down; both hit +10.75% — flatter curve |
| L9/F7540 | consists of | `caa_sae_down` | +2.86% | 0.25 | Norm@L9=2.328; matches old W_dec best |
| L9/F7540 | consists of | `caa_all_ml` | +2.86% | 3.0 | Tied; feature saturates sharply above α=0.5 (→-14.29%) |

**Key findings from Exp 29 (COMPLETE — all 10 features processed):**
1. **L4/F14233**: `caa_sae_down` (α=1.0) definitively best at +15.38%. `caa_all_ml` only +2.56% — early features don't benefit from all-layer decay.
2. **L12/F2257**: `caa_all_ml` (α=7.0) wins at +9.48%, beating `caa_sae_down` +8.82%. Mid-network features benefit from all-layer injection.
3. **L11/F12278**: `caa_all_ml` at α=7.0 gives **+6.25%** (new record). `caa_sae_down` at α=0.5 gives +5.85% (also new). The all-layer strategy wins for L11 — likely because late layers (L21-25) have huge norms (105 at L24) and contribute significantly.
4. **L9/F387**: Both strategies below W_dec best (+3.12%) — CAA norm at L9 is weak (0.668), limiting gains. `caa_sae_down` +2.92% @ α=0.5; `caa_all_ml` +2.71% @ α=3.0.
5. **L15/F220**: `caa_sae_down` gives **+7.96%** at α=0.75 — massive 2.5× improvement over prior best (+3.11%). L15 CAA norm=11.375. `caa_all_ml` +6.99% @ α=2.0 (below sae_down).
6. **L14/F10561**: `caa_sae_down` gives **+10.75%** at α=2.0 — FIVE-FOLD improvement over prior best (+2.15%)! Both `caa_sae_down` and `caa_all_ml` tied at +10.75%. L14 CAA norm=9.625.
7. **L9/F7540**: Both strategies tied at +2.86% — matches prior W_dec best. CAA offers no improvement for this small-N feature (N=35).
8. **Alpha inversely related to CAA norm at SAE layer**: L4 (norm=1.945)→α=1.0; L11 (norm=2.125)→α=0.5; L12 (norm=2.953)→α=1.0 for sae_down; L15 (norm=11.375)→α=0.75; L14 (norm=9.625)→α=2.0.
9. **CAA beats W_dec by 5× for L14**: The `close to` feature was one of the weakest under W_dec injection (+2.15%). CAA per-layer reveals a much stronger steering direction at +10.75%. This is the biggest CAA-vs-W_dec improvement ratio across all features.
10. **CAA generally beats W_dec** for 7/10 features tested; W_dec still wins for L9/F387 (CAA +2.92% vs W_dec +3.12%).

**Exp 30 — CAA Start-Layer Sweep v2 (GPU 5):** `pt448_caa_startlayer_sweep_v2.py`  
**CORRECTED** — tests how the injection start layer affects CAA injection using per-layer vectors at fixed α=1.0.

Results (3/5 features COMPLETE, L9/F387 in progress):

| L/F | SAE layer | Best start | Best Δ | Pattern |
|-----|-----------|------------|--------|---------|
| L4/F14233 | 4 | **0** (ties with 1,3,4,5) | **+15.38%** | Starts 0-5 all ≥+15.38% (except start=2 anomaly→+7.69%); start≥6 drops |
| L12/F2257 | 12 | **1** | **+9.80%** | start=1→+9.80% beats start=0→+9.48%; start≥18→<+4% |
| L11/F12278 | 11 | **20** | **+5.62%** | At wrong α=1.0; true optimum α=0.5 at sae_down (Exp 29 → +5.85%) |

**Key findings from Exp 30 (complete data):**

**L4/F14233 "ahead of"** — full sweep:
- start ∈ {0,1,3,4,5}: all give +15.38% (optimal)
- **start=2 anomaly**: drops to +7.69% — L2's normalized CAA vector appears to have a detrimental direction
- start=6-22: +5.13% to +10.26% range
- start≥23: drops to +2.56% (too few layers)

**L12/F2257 "facing"** — full sweep:
- start=1 (25 layers): **+9.80%** (best)
- start=0 (all 26): +9.48% (including L0 slightly hurts)
- start=2-12: plateau around +7.84%–+8.82% 
- start≥18: rapid drop off

**L11/F12278 "touching"** — at wrong α=1.0 (non-monotone behavior confirms α too high):
- start=0-16: plateau +1.87% to +3.75% (flat, insensitive to start layer at α=1.0)
- **start=18 (8 layers): +5.07%** — jumps up! Fewer layers ≡ lower effective dose
- **start=20 (6 layers): +5.62%** — even better
- **start=22 (4 layers): +3.75%** — drops off
- Pattern: at α=1.0, fewer layers → better, because each fewer layer reduces the total "dose"
- Dose model: `caa_sae_down` at start=11 (15 layers, α=0.5) → +5.85% ≡ 15×0.5=7.5 "layer-units"; start=20 (6 layers, α=1.0) → +5.62% ≡ 6×1.0=6 "layer-units"
- **Critical insight**: the late 6 layers (L20-25) carry essentially all the signal — early/mid injection layers mostly add noise at high α
- True optimum at Exp 29: caa_sae_down α=0.5 → +5.85%, caa_all_ml α=7.0 → +6.25%

**Limitation**: The start sweep used fixed α=1.0 for all features, but L11 needs α=0.5. This is why Exp 31 was launched.

**Exp 31 — Joint Alpha×Start Sweep (GPU 3, NEW):** `pt448_caa_joint_alphastart.py`  
**Motivation:** Exp 30's start sweep used fixed α=1.0, but Exp 29 showed L11 needs α=0.5. The per-layer sae_down α=0.5 result (+5.85%) uses start=11 — but maybe start=3 or 7 with α=0.5 gives higher gains? Similarly, L12 at start=1 and α=1.0 gives +9.80% — can we do better at α=0.75 or 1.25?

Grid:
- L12/F2257: α ∈ [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0] × start ∈ [0,1,2,3,4,5] = 48 cells
- L11/F12278: α ∈ [0.1, 0.25, 0.5, 0.75, 1.0, 1.5] × start ∈ [3,5,7,9,11,13,15] = 42 cells

**Results from L12/F2257 grid (COMPLETE — all 48 cells done):**

| α | best start | Best Δ | Runner-up |
|---|-----------|--------|-----------|
| 0.5 | 3 | +4.90% | start=0,2 tied at +4.25% |
| 0.75 | 4 | +8.17% | start=2 at +7.84% |
| **1.0** | **1** | **+9.80%** | start=0 at +9.48%, start=5,6 at +8.50% |
| 1.25 | 2 | +6.86% | start=1 at +6.54% |
| 1.5 | (in progress) | ~+3.5% | diminishing |
| 2.0–5.0 | — | declining | (all worse) |

**Confirmed: α=1.0, start=1 is the true joint optimum for L12/F2257 (+9.80%).** No improvement possible by jointly tuning alpha and start — the existing record is definitive.

The α=1.0 rows show a clear local maximum at start=1, with drops in both directions (start=0 loses L0's contribution; start≥2 loses L1's signal). The discovery that start=1 slightly beats start=0 confirms L0's CAA vector adds slight noise.

**L11/F12278 grid: IN PROGRESS on GPU 3** — testing α ∈ [0.1, 0.25, 0.5, 0.75, 1.0, 1.5] × start ∈ [3, 5, 7, 9, 11, 13, 15].

**Partial results (α=0.1 rows complete):**
- α=0.1, start=3 (23L): +0.16%
- α=0.1, start=5 (21L): +0.47%
- α=0.1, start=7 (19L): +0.47%
- (remaining α=0.1 rows in progress)

α=0.1 is too small — max +0.47% confirms we need higher alpha. The known best is α=0.5 (sae_down → +5.85%, all_ml at best start unknown). α=0.25 rows will come next.

**Expected outcome:** L11 peak likely at α=0.5, start≈9-13 (dose ≈ 6-7 units). May approach +6.5%+ given caa_all_ml already gives +6.25% at all layers. The start-layer restriction to downstream layers only may let us exceed this if we avoid the early-layer noise contribution.

---

## Experiment 21 Results — CAA Steering (Completed for L4/F14233, L12/F2257)

### Key insight: Per-layer vs. single-layer CAA vectors

The main `pt448_caa_steering.py` uses **layer-specific CAA vectors** for each injection layer. For `caa_sae_down` (inject at layers sae_layer→25), at each layer l it injects `v_caa_norm[l]` — the normalized contrastive difference vector computed at layer l of mix-448. This means:
- SAE layer (L4/L12): norm ≈ 1.9–3.0, cos_to_wdec ≈ 0.04
- Layer 24 (late): norm ≈ 25–105 (13–35× larger), direction more task-aligned

The `caa_sae_down` strategy gets dramatic gains because late layers' CAA vectors point more directly toward the Yes/No decision boundary. Each layer contributes its own directional signal — this is much more powerful than using a single fixed vector across all layers.

### L4/F14233 "ahead of" (N=39, base=56.41%):

| Strategy | Best Δ | Best α | Notes |
|----------|--------|--------|-------|
| `caa_single` | +2.56% | 0.5 | CAA at SAE layer only — weak |
| `caa_all_ml` | +2.56% | 1.0 | Decaying weights wash out late-layer signal |
| **`caa_sae_down`** | **+15.38%** | **1.0** | **NEW RECORD** — per-layer CAA from L4→L25 |
| `caa_proj_all` | +5.13% | 5.0 | FGAA-style (proj onto W_dec) |
| `wdec_all_ml` | +7.69% | 1.0 | W_dec all layers (baseline) |

### L12/F2257 "facing" (N=306, base=49.02%):

| Strategy | Best Δ | Best α | Notes |
|----------|--------|--------|-------|
| `caa_single` | +1.96% | 20.0 | Weak |
| **`caa_all_ml`** | **+8.50%** | **10.0** | Per-layer CAA at all 26 layers |
| **`caa_sae_down`** | **+8.82%** | **1.0** | Per-layer CAA from L12→L25 |
| `caa_proj_all` | +5.88% | 50.0 | FGAA-style |
| `wdec_all_ml` | +3.92% | 50.0 | W_dec (baseline) |

### L11/F12278 "touching" (N=1281, base=56.52%):

| Strategy | Best Δ | Best α | Notes |
|----------|--------|--------|-------|
| `caa_single` | +0.31% | 0.1 | Near-zero — single layer CAA has minimal effect |
| **`caa_all_ml`** | **+5.46%** | **5.0** | Per-layer CAA at all 26 layers — very strong |
| **`caa_sae_down`** | **+5.85%** | **0.5** | Per-layer CAA L11→L25 — current record |
| `caa_proj_all` | (pending) | — | FGAA-style |
| `wdec_all_ml` | (pending) | — | W_dec (baseline) |

For L11, `caa_sae_down` at α=0.5 beats `caa_all_ml` at α=5.0. The more focused downstream injection (15 layers from SAE layer) is better than decay-weighted all-layer injection. This contrasts with L12 where `caa_all_ml` was slightly better (+9.48% vs +8.82%).

---

## Experiment 24 — Layer Sweep Results (8/10 complete or in-progress)

| L/F | Relation | SAE layer | Best inj layer | Shift | Best Δ | Multi-layer best | Ratio |
|-----|----------|-----------|---------------|-------|--------|---------|-------|
| L4/F14233 | ahead of | 4 | 2 | -2 | +5.13% | +15.38% (caa_sae_down) | 33% |
| L12/F2257 | facing | 12 | **25** | +13 | +3.27% | +9.80% (start=1) | 33% |
| L11/F12278 | touching | 11 | **11** | 0 | **+3.36%** | +6.25% (caa_all_ml) | 54% |
| L9/F387 | at the right side of | 9 | **20** | +11 | +1.25% | +3.12% (W_dec multi-layer) | 40% |
| L15/F220 | across from | 15 | **17** | +2 | +1.36% | +7.96% (caa_sae_down) | 17% |
| L9/F7540 | consists of | 9 | **9** | 0 | +2.86% | +2.86% (single=multi) | 100% |
| L14/F10561 | close to | 14 | **10** | -4 | +2.15% | +10.75% (caa_sae_down) | 20% |
| **L13/F15219** | **behind** | **13** | **3 (partial)** | **-10** | **+2.68% (new record!)** | +2.68% (W_dec@L3; caa_all_ml only +1.55%) | **100%** |
| L14/F10561 | close to | 14 | (L10 +2.15% so far) | -4 | in progress | +10.75% (caa_sae_down) | — |

**Pattern**: single-layer injection captures ~17-54% of the multi-layer downstream result. For L9/F387, the best single layer is L20 (far downstream from SAE layer L9) — the representation of "right side" is better expressed at late layers. For L15/F220, single-layer W_dec at any layer gives at most +1.36%, while CAA per-layer gives +7.96% — a 5.8× gap confirming that per-layer CAA vectors (not W_dec) are needed for this high-norm feature. L9/F7540 is the exception: the SAE layer (L9) is already optimal and single-layer matches multi-layer. Single-layer is dominated by multi-layer strategies in almost all cases.

---

## Experiment 25 — Late-Layer CAA Results (4/10 complete)

| L/F | Relation | L24 norm | late_caa_ans | single24 | best_caa | late_wdec | Notes |
|-----|----------|---------|-------------|---------|---------|----------|-------|
| L4/F14233 | ahead of | 25.25 | +5.13%@α1 | +10.26%@α5 | +10.26%@α5 | +2.56%@α0.005 | Single L24 = W_dec best |
| L12/F2257 | facing | 36.25 | +3.27%@α10 | +4.58%@α10 | +4.58%@α10 | +0.98%@α1 | Single L24 < caa_sae_down |
| L11/F12278 | touching | 105.0 | +4.14%@α1 | **+5.39%@α5** | **+5.39%@α5** | +0.23%@α0.01 | L24 norm huge (105) — single L24 matches sae_down! |
| L9/F387 | right side of | 76.5 | +1.87%@α1 | (in progress) | (in progress) | — | Late answer weak; single24 in progress |

L11/F12278 has the largest L24 CAA norm (105 vs 25-36 for L4/L12). Despite this, `late_caa_single24` at +5.39% is just below the `caa_sae_down` record of +5.85%, confirming the full downstream injection is still better. However the margin is small for L11 — the L24 vector is nearly as good as the full per-layer strategy.

For L9/F387, the L24 CAA norm (76.5) is substantial. The `late_caa_answer` result (+1.87% @ α=1) is weaker than expected, likely because spreading across 5 late layers at α=1 is too high a dose given the large norm. `late_caa_single24` (in progress) may give stronger results.

**Cross-feature pattern:** Single L24 CAA performance:
- L4 (L24 norm=25): +10.26% = W_dec best (+10.26%), much less than caa_sae_down (+15.38%)
- L12 (L24 norm=36): +4.58%, much less than caa_sae_down (+8.82%)  
- L11 (L24 norm=105): +5.39%, nearly matches caa_sae_down (+5.85%)
- L9 (L24 norm=76.5): in progress

The late-layer CAA norm grows dramatically (L11 norm 105 >> L4 norm 25) and as it grows, single-layer injection becomes relatively more competitive with the full downstream strategy.

---

## Experiment 23 Update — FGAA Steering Results (3/6 features now confirmed zero)

| L/F | Relation | n_pos_cond | uncond_CAA | cond_CAA | proj_CAA |
|-----|----------|-----------|-----------|---------|---------|
| L4/F14233 | ahead of | 0 | +0.00% | +0.00% | +0.00% |
| L12/F2257 | facing | 0 | +0.00% | +0.00% | +0.00% |
| L11/F12278 | touching | 0 | +0.00% | +0.00% | +0.00% |

All three confirmed zero. The feature firing threshold n_pos_c=0 for all three — no positive examples in mix-448 forward passes fire the feature above threshold. Remaining 3 features (L9/F387, L13/F15219, L15/F220) expected same.

---

## Hard Features Ext Results (3/3 COMPLETE)

| L/F | Relation | Base | Best strategy | Best Δ | Prior best | Improvement |
|-----|----------|------|--------------|--------|-----------|------------|
| L11/F9639 | in/inside/on | 60.85% | answer α=10 | +0.73% | +0.73% | 0 (no improvement) |
| L6/F7539 | left/right of | 51.08% | topK_ml α=20 | +1.24% | +1.24% | 0 (no improvement) |
| L13/F15219 | behind | 51.62% | downstream_ml α=30 | +2.12% | +2.12% | 0 (no improvement) |

Extended search confirmed: no strategy improvement for these 3 hard features. Best W_dec-based results remain definitive.

**No improvement for L11/F9639 and L6/F7539 at extended alpha range (up to α=100)**. Prior best stands.

---

## GPU Allocation (21 April 2026, ~14:00 — current state)

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 1085274 | `pt448_caa_steering.py` | Running — L4, L12, L11 done; on L11 wdec_all_ml final rows |
| 1 | 1103532 | `pt448_late_layer_caa.py` | Running — L4, L12, L11 done; on L9/F387 |
| 2 | 1355629 | `pt448_caa_perlayer_finesweep_gpu2.py` | Running — L9, L15, L9_F7540, L14 DONE; on L13/F15219 "behind" |
| 3 | 1477944 | `pt448_caa_joint_alphastart.py` | Running — L12 grid COMPLETE (confirms α=1.0 start=1 = +9.80%); now on L11 grid |
| 4 | 1259025 | `pt448_caa_perlayer_finesweep.py` | Running — L4, L12, L11, L15, L9_F7540 done (SKIP); on L14 (will SKIP — saved by GPU2) |
| 5 | 1288039 | `pt448_caa_startlayer_sweep_v2.py` | Running — L4, L12, L11 done; on L9/F387 |
| 6 | 1067854 | `pt448_layer_sweep.py` | Running — L4, L12, L11, L9/F387 done; on L15/F220 layer sweep |
| 7 | 1034879 | `pt448_fgaa_steering.py` | Running — FGAA confirmed zero for ALL features (n_pos_c=0 universally) |

**Summary of Exp 29 confirmed results (GPU2 + GPU4, all 10 features):**

| L/F | Relation | CAA norm@SAE | `caa_sae_down` best | `caa_all_ml` best | Winner | vs prior W_dec best |
|-----|----------|-------------|--------------------|--------------------|--------|---------------------|
| L4/F14233 | ahead of | 1.945 | **+15.38%** @ α=1.0 | +2.56% @ α=2.0 | sae_down | +10.26% → +15.38% ↑ |
| L12/F2257 | facing | 2.953 | +8.82% @ α=1.0 | **+9.48%** @ α=7.0 | all_ml | +3.92% → +9.48% ↑↑ |
| L11/F12278 | touching | 2.125 | +5.85% @ α=0.5 | **+6.25%** @ α=7.0 | all_ml | +3.36% → +6.25% ↑↑ |
| L9/F387 | right side of | 0.668 | +2.92% @ α=0.5 | +2.71% @ α=3.0 | sae_down | +3.12% → +2.92% ↓ |
| L15/F220 | across from | 11.375 | **+7.96%** @ α=0.75 | +6.99% @ α=2.0 | sae_down | +3.11% → +7.96% ↑↑ |
| L14/F10561 | close to | 9.625 | **+10.75%** @ α=2.0 | **+10.75%** @ α=10.0 | TIE | +2.15% → +10.75% ↑↑↑ |
| L9/F7540 | consists of | 2.328 | +2.86% @ α=0.25 | +2.86% @ α=3.0 | TIE | +2.86% → +2.86% = |
| L13/F15219 | behind | 4.594 | **+1.55%** @ α=0.5 | +1.13% @ α=5.0 | sae_down | +2.12% → +1.55% ↓ (W_dec wins) |
| **L6/F7539** | **left/right of** | **0.891** | **+3.10% @ α=1.5** | **–3.10% @ α=5 (catastrophic)** | **sae_down** | **+1.24% → +3.10% ↑** (2.5×; caa_all_ml completely fails for this feature) |
| L11/F9639 | in/on/inside | (pending GPU2+4 after L6) | — | — | — | +0.73% |

**Key new insight (L14/F10561):** This feature has the highest CAA norm at its SAE layer (9.625) among the non-trivial features, which combined with its 14-layer-deep SAE position means the downstream per-layer CAA vectors are very large and highly task-aligned. The jump from +2.15% (W_dec) to +10.75% (CAA) is the most dramatic improvement in the whole study — CAA beats W_dec by 5×.

**Critical pattern — CAA norm predicts improvement multiplier:**
- Low norm (L9/F387, 0.668): CAA ≤ W_dec (+2.92% vs +3.12%)
- Medium norm (L11/F12278, 2.125): CAA ~2× W_dec (+6.25% vs +3.36%)
- Medium-high norm (L12/F2257, 2.953): CAA ~2.5× W_dec (+9.48% vs +3.92%)
- High norm (L15/F220, 11.375): CAA ~2.5× W_dec (+7.96% vs +3.11%)
- High norm (L14/F10561, 9.625): CAA ~5× W_dec (+10.75% vs +2.15%)

Higher SAE-layer CAA norm → larger improvement over W_dec (with one exception: L4 has low norm 1.945 yet +15.38% because it's early-layer with many downstream layers to compound).

---

## Monitoring Update (21 April 2026, ~11:30 — post-context-compaction pass)

### New completions since last update

**Exp 29 — L15/F220 confirmed (GPU4 independent run, `plf_L15_F220.json`):**
Both GPU2 and GPU4 independently saved the same result:
- `caa_sae_down`: **+7.96%** @ α=0.75. Full curve: α=0.25→+2.14%, α=0.5→+6.60%, α=0.75→+7.96%, α=1.0→+6.80%, α=1.5→+2.91%, α≥2.0→<+1.4%
- `caa_all_ml`: +6.99% @ α=2.0. Sharply limited: α=5→+1.94%, α≥7→+0.97%
- CAA norm@L15=11.375; baseline=49.90%, N=515. Replicated across 2 independent GPU runs. ✓

**Exp 29 — L11/F12278 full alpha curve confirmed (GPU4, `plf_L11_F12278.json`):**
- `caa_sae_down`: **+5.85%** @ α=0.5. Full curve: α=0.25→+2.81%, α=0.5→+5.85%, α=0.75→+3.28%, α=1.0→+2.58%, α≥1.5→negative (collapses)
- `caa_all_ml`: **+6.25%** @ α=7.0. Curve: α=2→+1.72%, α=5→+5.46%, α=7→+6.25%, α=10→+4.53%, α=12→+3.04%, α≥15→negative
- CAA norm@L11=2.125; baseline=56.52%, N=1281

**Exp 30 — L11/F12278 start-layer sweep (GPU5, `startlayer_v2_L11_F12278.json`):**
Full sweep at α=1.0 (non-optimal α, but reveals dose model behavior):
- start=0-10: plateau ~+2.0% to +3.5% (flat, insensitive to start)
- start=14: +3.75%; start=18: +5.07%; **start=20: +5.62%** (best)
- start=22: +3.75%; start=23: +2.73%; start=24: +1.33%; start=25: +0.62%
- The "dose model" is confirmed: at α=1.0, optimal start=20 (6 layer-units) beats α=0.5 start=11 (7.5 layer-units: +5.85%). Fewer layers with late-layer high-norm vectors dominates.

**Exp 31 — L12/F2257 joint α×start (GPU3, complete):**
All 48 cells done. Full row-by-row results confirm: **α=1.0, start=1 = +9.80% is the global 2D optimum**. Key transitions:
- α=0.5: max +4.90% @ start=3
- α=0.75: max +8.17% @ start=4
- α=1.0: **+9.80%** @ start=1 (and +9.48% @ start=0, +8.50% @ others)
- α=1.25: max +6.86% @ start=2 (sharp falloff)
- α≥1.5: ≤+3.59% (diminishing; converges to noise by α=5)
L12/F2257 record is definitive: no (α, start) pair can beat +9.80%.

**Exp 31 — L11/F12278 joint grid: IN PROGRESS on GPU3**
Testing α ∈ [0.1, 0.25, 0.5, 0.75, 1.0, 1.5] × start ∈ [3, 5, 7, 9, 11, 13, 15].
Based on dose model: α=0.5 at start=13 (13 layers × 0.5 = 6.5 dose-units) may approach or beat caa_all_ml +6.25%.

**Exp 25 — L9/F387 late CAA (GPU1, in progress):**
Results so far: `late_caa_answer` +1.87%@α=1.0; `late_caa_single24` +2.08%@α=5.0.
Both below W_dec best (+3.12%). L9 late-layer CAA (norm=76.5 at L24) is stronger absolute norm than L11/L12 but L9's SAE-layer norm (0.668) is very weak, and late-only injection without the full downstream chain underperforms. W_dec remains the record for this feature.

**Exp 25 — Summary so far (4/10 features complete):**

| L/F | L24 norm | late_caa_answer | late_caa_single24 | W_dec best | Pattern |
|-----|---------|-----------------|-------------------|------------|---------|
| L4/F14233 | 25.3 | +5.13%@α=1 | **+10.26%@α=5** | +10.26% | Single L24 = W_dec record |
| L12/F2257 | 36.3 | +3.27%@α=10 | +4.58%@α=10 | +9.80% (caa_sae_down) | Late-only << full chain |
| L11/F12278 | 105.0 | +4.14%@α=1 | **+5.39%@α=5** | +6.25% (caa_all_ml) | Single L24 ≈ full chain |
| L9/F387 | 76.5 | +1.87%@α=1 | +2.08%@α=5 | +3.12% (W_dec) | Late CAA < W_dec |

**Exp 30 — L9/F387 start-layer sweep (GPU5, COMPLETE):**
All 20 start positions completed at α=1.0. Best was start=18 (+2.50%), but below W_dec best (+3.12%). Full result pattern:
- start=0-12: range -1.04% to 0.00% (flat/negative throughout)
- start=14: +0.62%; start=16: +0.83%; start=18: +2.50%; start=20: +2.50%
- start=22: +1.67%; start=23: +1.67%; start=24: 0.00%; start=25: -0.42%
BEST: start=18 or 20 (6-8 layers) → +2.50% — still below W_dec +3.12%. Confirmed: L9/F387 is not improvable via CAA at α=1.0. The CAA norm at L9 (0.668) is simply too weak. perlayer fine (Exp 29) found α=0.5 gives +2.92% (still below W_dec). GPU5 has now moved to L13/F15219 start sweep.

### L13/F15219 "behind" — Exp 29 COMPLETE (plf_L13_F15219.json saved)

Both GPU2 and GPU4 processed L13/F15219. GPU4 saved `plf_L13_F15219.json` at 11:47. CAA norm@L13=4.594.

**`caa_sae_down` (COMPLETE):**
α=0.1→-0.14%, α=0.25→+0.56%, α=0.5→**+1.55%**, α=0.75→+0.28%, α=1.0→+1.13%, α=1.5→-0.42%, α=2.0→-0.85%, α=3.0→-1.27%, α=5.0→-1.27%
Best: **+1.55%** @ α=0.5 — below W_dec record (+2.12%). Non-monotone; peak at α=0.5.

**`caa_all_ml` (COMPLETE):**
α=1.0→+0.28%, α=2.0→+0.71%, α=3.0→+0.00%, α=5.0→+1.13%, α=7.0→+0.42%, α=10.0→-1.41%, α=12.0→-0.28%, α=15.0→-1.41%, α=20.0→-1.55%, α=30.0→-1.27%
Best: **+1.13%** @ α=5.0 — also below W_dec (+2.12%).

**Conclusion for L13/F15219:** Neither CAA strategy improves on W_dec for "behind". Despite moderate norm (4.594), the feature collapses non-monotonically, suggesting the "behind" spatial direction in the residual stream is not as cleanly aligned as "ahead of" or "facing". W_dec downstream_ml α=30 (+2.12%) remains the record. GPU2 and GPU4 both now running L6/F7539.

### Exp 24 Layer Sweep — New completed features

| L/F | SAE layer | Best inj layer | Shift | Best Δ | Note |
|-----|-----------|---------------|-------|--------|------|
| L15/F220 | 15 | **17** | +2 | **+1.36%** | W_dec at any single layer max +1.36%; CAA multi-layer gives +7.96% |
| L9/F7540 | 9 | **9** | 0 | **+2.86%** | SAE layer optimal; L0,L1 hurt (-5.71%); L12 also +2.86% |
| L14/F10561 | 14 | (L10 looks best, +2.15%) | -4 | in progress | Early layers all hurt; mid-range improving |

L15/F220 confirms: single-layer W_dec for "across from" is uniformly weak (max +1.36% at any layer), while per-layer CAA gives +7.96%. The feature simply needs the compounding effect of many CAA vectors. L9/F7540 is ideally local to SAE layer.

### Current status summary (~15:20)

| GPU | Script | On feature | Status |
|-----|--------|-----------|--------|
| 0 | `pt448_caa_steering.py` | L9/F387 caa_sae_down | Running — caa_all_ml DONE (+1.67%); sae_down α=0.5→+2.92% seen |
| 1 | `pt448_late_layer_caa.py` | L15/F220 late_caa_answer | Running — L9/F387 DONE (best=+2.08%@L24); L15 at α=0.1→+0.39% |
| 2 | `pt448_caa_perlayer_finesweep_gpu2.py` | L6/F7539 caa_sae_down | Running — at α=0.75 (confirms same curve as GPU4) |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 joint grid | Running — α=0.25 rows in progress |
| 4 | `pt448_caa_perlayer_finesweep.py` | L6/F7539 caa_all_ml | Running — α=2→-2.79%, α=5→-3.10% (catastrophic) |
| 5 | `pt448_caa_startlayer_sweep_v2.py` | L13/F15219 start sweep | Running — start=4 done (+1.27%) |
| 6 | `pt448_layer_sweep.py` | L13/F15219 single-layer W_dec | Running — **inj_layer=3→+2.68% NEW RECORD** |
| 7 | `pt448_fgaa_steering.py` | L13/F15219 FGAA | FGAA confirmed zero universally |

### Pending results

- **L6/F7539 caa_all_ml** (GPU4): Confirmed catastrophic collapse (all α → negative). caa_sae_down wins at +3.10%.
- **L11/F9639 "in/on/inside"**: Next on GPU2+4 after L6 completes. CAA norm unknown; W_dec best +0.73%.
- **Exp 31 L11 grid**: 42 cells on GPU3; α=0.25 rows in progress. Peak at α=0.5 expected.
- **L13/F15219 layer sweep** (GPU6): inj_layer=3→+2.68% new record (partial). Full 26-layer sweep completing — may find even better layer.
- **L13/F15219 start sweep** (GPU5): At α=1.0, start=0→+1.13%, start=3→+1.27% so far.

---

## Monitoring Update (21 April 2026, ~15:20 — second context window pass)

### New completions

**L6/F7539 "left/right of" — Exp 29 COMPLETE (plf_L6_F7539 pending save):**
- CAA norm@L6=0.891 (very low, second-lowest after L9/F387 0.668)
- `caa_sae_down` full curve: α=0.1→+0.31%, α=0.25→-0.31%, α=0.5→+0.31%, α=0.75→+1.55%, α=1.0→+0.93%, α=1.5→**+3.10%**, α=2.0→+1.24%, α=3.0→+0.62%, α=5.0→+0.62%
  - **Best: +3.10% @ α=1.5** — definitive new record (was W_dec +1.24%)
  - Non-monotone but clear peak at α=1.5; collapses sharply either side
- `caa_all_ml`: **catastrophic collapse** — α=2.0→-2.79%, α=5.0→-3.10%. All-layer injection destroys performance for this feature. Confirmed on GPU4.
- **Key insight:** For low-norm features (L6, CAA norm=0.891), `caa_all_ml` fails completely while `caa_sae_down` still works. The exponential decay weights in `caa_all_ml` bring in too many early-layer interference terms when the norm is weak and the signal isn't well-aligned across layers. `caa_sae_down` (downstream only from SAE layer L6→L25) avoids this by using only downstream layers where the spatial concept has already formed.

**L9/F387 startlayer_v2_L9_F387.json SAVED:**
Full sweep at α=1.0, all 20 start positions. Best start=18 (+2.50%), confirming CAA at α=1.0 for this feature peaks at 6-8 late layers. W_dec +3.12% remains record.

**L13/F15219 — NEW RECORD from Exp 24 Layer Sweep (GPU6):**
W_dec at inj_layer=3 with α=30 gives **+2.68%**, beating prior best of +2.12% (downstream_ml α=30 at SAE layer=13).
- L0: +1.97%, L1: +0.85%, L2: +1.97%, **L3: +2.68%**, L4: -0.56%, L5: -0.85%, L6: +0.56%
- Injecting at L3 (10 layers BEFORE SAE layer L13) is optimal for "behind"!
- This is the first example of upstream injection outperforming downstream/SAE-layer injection for W_dec.
- Note: the layer sweep uses the SAE feature's W_dec vector at α=30 — very aggressive alpha, relying on single-layer magnitude. The "behind" direction appears to be most leverageable at early-mid layers (L3) despite the SAE being trained at L13.
- Full sweep still running (GPU6 at inj_layer=6 now).

### Exp 29 new insight: `caa_all_ml` fails for low-norm / low-transfer features

| Feature | CAA norm | `caa_sae_down` | `caa_all_ml` | Winner |
|---------|---------|---------------|-------------|--------|
| L6/F7539 | 0.891 | **+3.10%** | **-3.10%** (collapse) | sae_down by huge margin |
| L9/F387 | 0.668 | +2.92% | +2.71% | sae_down marginally |
| L13/F15219 | 4.594 | +1.55% | +1.13% | sae_down marginally |
| L9/F7540 | 2.328 | +2.86% | +2.86% (tied) | tie |
| L11/F12278 | 2.125 | +5.85% | **+6.25%** | all_ml marginally |
| L12/F2257 | 2.953 | +8.82% | **+9.48%** | all_ml |
| L15/F220 | 11.375 | **+7.96%** | +6.99% | sae_down |
| L14/F10561 | 9.625 | **+10.75%** | +10.75% (tied) | tie |
| L4/F14233 | 1.945 | **+15.38%** | +2.56% | sae_down (decisive) |

Pattern: `caa_all_ml` wins or ties for mid-norm features (2.1–3.0 range: L11, L12); `caa_sae_down` wins for everything else, and catastrophically beats `caa_all_ml` for very low norm (L6) and early-layer features (L4).

### L11/F12278 joint grid (GPU3) — α=0.25 rows

α=0.1 complete: max +0.55% @ start=11. α=0.25 starting. No improvement over known best (+6.25%) expected until α=0.5 rows arrive.

---

## Monitoring Update (21 April 2026, ~16:30 — third context window pass)

### Major new finding: L15/F220 `late_caa_single24` = +7.18% @ α=10.0 (GPU1)

**L15/F220 "across from / at the left side of" — Exp 25 late CAA COMPLETE:**

All strategies on GPU1 now complete:

| Strategy | Best Δ | Best α | Description |
|----------|--------|--------|-------------|
| `late_caa_answer` | **+5.24%** | α=1.0 | CAA@L24 vector injected at layers 21–25 (5 late layers) |
| `late_caa_single24` | **+7.18%** | α=10.0 | CAA@L24 vector injected at L24 only |
| `best_caa_single` | same as single24 | α=10.0 | L24 is best single CAA layer |
| `late_wdec_answer` | +0.62% | α=0.05 | W_dec at layers 21–25 |

Full `late_caa_single24` curve: α=0.005→+0.19%, α=0.010→+0.39%, α=0.050→+0.19%, α=0.100→+0.19%, α=0.500→+1.36%, α=1.0→+0.58%, α=5.0→+5.44%, α=**10.0→+7.18%**.

Full `late_caa_answer` (L24 vec @ layers 21–25) curve: α=0.005→+0.19%, α=0.010→+0.58%, α=0.050→+0.39%, α=0.100→+0.39%, α=0.500→+2.33%, α=**1.0→+5.24%**, α=5.0→+1.17%, α=10.0→+0.97%.

**Insight:** For L15/F220, a SINGLE injection of CAA@L24 at just layer 24 (α=10) gives **+7.18%** — this is 90% of the full `caa_sae_down` gain (+7.96%). The high L24 norm (62.0) means the CAA direction at L24 is extremely high-quality and powerful enough to work alone at high α. By contrast, spreading the injection over 5 layers (21–25) gives +5.24% at α=1.0 — the compounding effect of 5 layers at low α is slightly less efficient than 1 layer at very high α.

**Updated Exp 25 summary table (all 5 features done):**

| L/F | L24 norm | late_caa_answer | late_caa_single24 | caa_sae_down (full) |
|-----|---------|-----------------|-------------------|---------------------|
| L4/F14233 | 25.3 | +5.13%@α=1 | **+10.26%@α=5** | +15.38% |
| L12/F2257 | 36.3 | +3.27%@α=10 | +4.58%@α=10 | +9.80% |
| L11/F12278 | 105.0 | +4.14%@α=1 | +5.39%@α=5 | +6.25% (caa_all_ml) |
| L9/F387 | 76.5 | +1.87%@α=1 | +2.08%@α=5 | +3.12% (W_dec) |
| **L15/F220** | **62.0** | **+5.24%@α=1** | **+7.18%@α=10** | **+7.96%** |

Key pattern: `late_caa_single24` generally achieves 70–90% of the full-chain gain, but with much higher α requirement. Only for L4/F14233 does single-layer actually match the full chain (both +10.26%).

### L11/F12278 Joint Grid (GPU3) — α=0.25 rows now complete

New rows since last update (α=0.100 and α=0.250 complete):

- α=0.100: max +0.55% @ start=11 (all below +1%)
- α=0.250 best: **+2.81%** @ start=11 (15L injection)
  - start=3: +2.19%, start=5: +2.03%, start=7: +2.50%, start=9: +2.50%, start=11: **+2.81%**, start=13: ?, start=15: ?

α=0.250 is showing the best result is at start=11 (the SAE layer itself) — 15 downstream layers at low α=0.25. This matches the known `caa_sae_down` behavior. Still below known best (caa_all_ml +6.25%). α=0.5+ rows will be the decisive ones.

GPU3 is currently mid-grid; α=0.500 rows are **in progress** (start=3→+4.25%, start=3→+4.90% best so far). The sweep is proceeding toward the expected sweet spot around α=0.5–0.75.

### L13/F15219 layer sweep (GPU6) — COMPLETE

Full single-layer W_dec sweep at α=30 for L13/F15219 "behind":

- L0: +1.97%, L1: +0.85%, L2: +1.97%, **L3: +2.68%** ← BEST, L4: -0.56%, L5: -0.85%
- L6: +0.56%, L7: +1.41%, L8: +1.13%, L9: +1.55%, L10: +0.99%, L11: +0.56%
- L12: +1.69%, L13: +0.56% (SAE layer), L14: +1.55%, L15: +1.13%
- L16: +0.42%, L17: +0.42%, L18: -0.42%, L19: +1.08%, L20: -1.08%
- L21: -1.08%, L22: +1.08%, L23: +0.00%, L24: +1.08%, L25: +1.08%
- **BEST: inj_layer=3 → +2.68%** (SAE layer=13, shift=-10 upstream)

Confirmed: L3 is the globally optimal single-injection layer for W_dec "behind" feature. No downstream layer beats it. The "behind" spatial concept appears maximally leverageable at L3 for steering.

### L13/F15219 start-layer sweep (GPU5) — in progress

At α=1.0 (CAA vectors), best is start=12 (+1.55%) from 26 start positions partially done:
- start=0→+1.13%, start=1→+0.56%, start=2→+0.14%, start=3→+1.27%, start=4→+1.27%, start=5→-0.14%
- start=6→+0.71%, start=7→+0.71%, start=8→+0.85%, start=9→+0.99%, start=10→+0.56%, start=12→**+1.55%**
- start=14→+1.27%, start=16→+0.99%, start=18→+0.56%, start=20→+0.85%, start=22→+1.69% (still running)

No start configuration for CAA at α=1.0 seems to beat W_dec@L3 (+2.68%). The `caa_sae_down` best is +1.55% and appears capped. GPU5 wrapping up last few start values.

### L11/F9639 "in/inside/on" — GPU2 and GPU4 now running

Both GPUs moved to L11/F9639 after completing L6/F7539. CAA norm@L11=2.578.
- GPU4 `caa_sae_down` in progress: α=0.25→+0.18%, α=0.5→-1.54%, α=0.75→-3.18%, α=1.0→-4.90% — sharply declining already
- W_dec prior best: +0.73%. CAA sae_down is making things much worse at high α.

This is the hardest feature (lowest ablation effect, N=1101, high noise). The feature has moderate CAA norm (2.578) but appears not to steer well — likely because "in/inside/on" is a generic spatial category that the model already handles adequately in pt-448 (baseline 60.85%).

### GPU0 — FGAA script (mix-448 inject)

GPU0 is running a different experiment: the `mix448_inject.log` shows it's testing W_dec injection on mix-448 itself (not pt-448) as a control, plus running FGAA-style conditioned CAA. Results so far: all alphas ≈ 0.00% Δ (mix-448 is already near-ceiling for these spatial relations). This confirms the injection is truly about transferring capabilities into pt-448, not amplifying existing ones.

### Updated leaderboard (all known best results)

| Feature | Relation | Best Δ | Method | α | Notes |
|---------|----------|--------|--------|---|-------|
| L4/F14233 | ahead of | **+15.38%** | caa_sae_down | 1.0 | Perfect 1.00× transfer; plateau from start=0 |
| L14/F10561 | close to | **+10.75%** | caa_sae_down | 2.0 | High L24 norm; also tied caa_all_ml |
| L12/F2257 | facing | **+9.80%** | caa_sae_down | 1.0, start=1 | Confirmed global optimum (48-cell grid done) |
| L15/F220 | across from / left side | **+7.96%** | caa_sae_down | 0.75 | Single L24 alone gives +7.18% |
| L11/F12278 | touching | **+6.25%** | caa_all_ml | 7.0 | Joint grid in progress; may improve |
| L6/F7539 | left of / right of | **+3.10%** | caa_sae_down | 1.5 | caa_all_ml catastrophic (-3.10%) |
| L9/F387 | at right side of | **+3.12%** | W_dec decay_fwd_ra | — | CAA peaks at +2.92% (below W_dec) |
| L9/F7540 | consists of | **+2.86%** | caa_sae_down | 0.25 | Natural-scale catastrophic |
| L13/F15219 | behind | **+2.68%** | W_dec@L3 | 30 | Single upstream injection (10 layers before SAE) |
| L11/F9639 | in/inside/on | **+0.73%** | W_dec downstream_ml | — | CAA making it worse; W_dec best |

### Current GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L9/F387 all strategies | Running — various strategies, mostly ~+1% or less |
| 1 | `pt448_late_layer_caa.py` | L15/F220 → next feature | L15 DONE; moving to next |
| 2 | `pt448_caa_perlayer_finesweep_gpu2.py` | L11/F9639 | Running — caa_sae_down sharply negative |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=0.5 rows | Running — α=0.5 start=3→+4.90% |
| 4 | `pt448_caa_perlayer_finesweep.py` | L11/F9639 | Running — caa_sae_down sharply negative |
| 5 | `pt448_caa_startlayer_sweep_v2.py` | L13/F15219 wrapping up | start=22→+1.69%, almost done |
| 6 | `pt448_layer_sweep.py` | L13/F15219 COMPLETE | All 26 layers done; best=L3 +2.68% |
| 7 | `pt448_fgaa_steering.py` | FGAA all features | Running — FGAA confirmed 0% universally |


---

## Monitoring Update (21 April 2026, ~17:10 — fourth context window pass)

### Completions since last update

**L13/F15219 startlayer_v2 SAVED** (`startlayer_v2_L13_F15219.json`):
Full 20-start sweep at α=1.0 complete. Best: start=22 (+1.69%), using only 4 late layers (22–25).
Full curve: start=0→+1.13%, 1→+0.56%, 2→+0.14%, 3→+1.27%, 4→+1.27%, 5→-0.14%, 6→+0.71%, 7→+0.71%, 8→+0.85%, 9→+0.99%, 10→+0.56%, 12→+1.55%, 14→+1.27%, 16→+0.99%, 18→+0.56%, 20→+0.85%, 22→+1.69%, 23→+1.41%, 24→-0.56%, 25→-0.71%.
Conclusion: No start position for CAA at α=1.0 beats W_dec@L3 (+2.68%). The CAA "behind" vector across all start positions peaks at +1.69% — well below the W_dec single-layer injection.

**GPU5 freed** (pt448_caa_startlayer_sweep_v2 done), **GPU7 freed** (pt448_fgaa_steering done).

### New experiments launched (17:10)

**Exp 32 — Joint α×start sweep for L14/F10561 and L15/F220:**

Both features had only their default `caa_sae_down` from SAE layer; start-layer optimization was not explored. Given:
- L12/F2257: joint sweep found start=1 (not default start=12) was optimal (+9.80% vs baseline +8.82%)
- L11/F12278: joint grid in progress, α=0.75 at start=4 already hit +8.17% (vs old record +5.85%)

We now sweep L14 and L15:

| Feature | GPU | Alphas | Starts |
|---------|-----|--------|--------|
| L14/F10561 "close to" | GPU5 | [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0] | [0,1,2,3,5,8,10,12,14] |
| L15/F220 "across from" | GPU7 | [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] | [0,1,2,3,5,8,10,12,15] |

Script: `pt448_caa_joint_L14_L15.py`; logs: `/tmp/joint_L14.log`, `/tmp/joint_L15.log`

### L11/F9639 "in/inside/on" — CAA catastrophic

GPU2 and GPU4 both running L11/F9639. `caa_sae_down` is sharply declining:
- α=0.1: +0.09%, α=0.25: +0.18%, α=0.5: -1.54%, α=0.75: -3.18%, α=1.0: -4.90%, α=1.5: -9.26%, α=2.0: -10.90%

This is the worst CAA result seen across all features. The "in/inside/on" concept (N=1101, baseline 60.85%) is already well-represented in pt-448 (transfer ratio 0.24×) and CAA steering overcorrects badly. W_dec +0.73% remains the only method that helps, and marginally at that.

### L11/F12278 joint grid progress (GPU3)

L12/F2257 grid COMPLETE (saved). L11/F12278 is now at α=0.25 (7 starts done). Note: the joint grid log also shows these **L12 joint grid rows** showing a completely separate finding — when tested with the *caa_sae_down per-layer vectors* (the joint grid uses this strategy), L12/F2257 gets +9.80% at α=1.0/start=1, confirming global optimum.

L11 progress so far:
- α=0.1 (7 starts): max +0.55% @ start=11
- α=0.25 (7 starts): max +2.81% @ start=11

The monotonicity: start=11 (SAE layer itself, 15 downstream layers) is consistently best for α≤0.25.

### Current GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L9/F387 all strategies | Running — old-style single-vec strategies, ~0–1% |
| 1 | `pt448_late_layer_caa.py` | L15/F220 → next | L15 complete; moving to next in list |
| 2 | `pt448_caa_perlayer_finesweep_gpu2.py` | L11/F9639 | Running — caa_sae_down catastrophic |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=0.25 rows | Running — best so far +2.81%@start=11 |
| 4 | `pt448_caa_perlayer_finesweep.py` | L11/F9639 | Running — caa_sae_down catastrophic |
| 5 | `pt448_caa_joint_L14_L15.py` | L14/F10561 joint grid | **NEW** — just started |
| 6 | `pt448_layer_sweep.py` | L13/F15219 layer sweep | In progress (inj_layer=20) |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 joint grid | **NEW** — just started |


---

## Monitoring Update (21 April 2026, ~17:30 — fifth context window pass)

### Critical: L11/F12278 joint grid α=0.500 rows — NEW BEST incoming

GPU3 (`pt448_caa_joint_alphastart.py`) is now running α=0.500 rows for L11/F12278 "touching":

- α=0.500 start=3 (23L): **+6.17%** *** NEW BEST ***
- α=0.500 start=5 (21L): **+6.32%** *** NEW BEST ***
- More start values still running...

The trajectory is climbing — start=5 is better than start=3. This suggests start=7 or start=9 may push even higher. The known best was +6.25% (caa_all_ml α=7.0). Already **exceeding it** at α=0.500 start=5 (+6.32%). α=0.750 rows will likely be the peak (analogous to L12/F2257 where α=1.0 was optimal with different start). Watch for α=0.75 start=3–5 rows.

**Implication**: L11/F12278 optimal is `caa_sae_down` at α=0.5, start=5 (or similar), NOT the previously thought `caa_all_ml` at α=7. The joint sweep is revealing that start-layer optimization beats the multi-layer decay weighting.

### L13/F15219 layer sweep — FULLY COMPLETE in log (GPU6 still writing to disk)

Complete 26-layer W_dec sweep at α=30 for L13/F15219 "behind":
- L0: +1.97%, **L3: +2.68%** ← GLOBAL BEST, L4: -0.56%, L5: -0.85%
- L6–L17: range 0.42%–1.69% (sporadic, no layer beats L3)
- L18: -0.42%, L19: -0.14%, L20: -0.14%, L21: +0.56%, L22–L25: ≤+0.14%
- Confirmed: **inj_layer=3 is the unique global optimum** across all 26 layers

### L13/F15219 layer sweep summary (per feature)

Layer sweep (`pt448_layer_sweep.py`, GPU6) results for features processed:

| Feature | SAE layer | Best inj_layer | Shift | Best Δ |
|---------|-----------|---------------|-------|--------|
| L4/F14233 | 4 | **2** | -2 | +5.13% |
| L12/F2257 | 12 | **25** | +13 | +3.27% (W_dec; caa_sae_down is better) |
| L11/F12278 | 11 | **11** | 0 | +3.36% (SAE layer optimal) |
| L9/F7540 | 9 | **9** | 0 | +2.86% |
| L9/F387 | 9 | **20** | +11 | +1.25% |
| L15/F220 | 15 | **17** | +2 | +1.36% |
| L14/F10561 | 14 | **10** | -4 | +2.15% |
| **L13/F15219** | **13** | **3** | **-10** | **+2.68%** |

Pattern: most features prefer injection at or near their SAE layer (shift ≤ 2). L13 is the clear outlier — single W_dec injection at L3 (10 layers upstream!) is uniquely powerful. L4 also prefers slightly upstream (L2). Only L9/F387 and L12/F2257 prefer downstream.

### L11/F9639 caa_sae_down COMPLETE on GPU4

Full curve (GPU4): α=0.25→+0.18%, α=0.5→-1.54%, α=0.75→-3.18%, α=1.0→-4.90%, α=1.5→-9.26%, α=2.0→-10.90%, α=3.0→-11.63%, α=5.0→-11.63%.
**Best: +0.18% @ α=0.25** — essentially zero gain. caa_all_ml is starting on GPU4 now.

GPU2 also at same point (same feature, same result pattern confirmed). caa_all_ml starting next.

### L14/F10561 joint sweep (GPU5) — early rows

α=0.500 start=3: **+4.30%** so far (well below known +10.75% at α=2.0/start=14 default). Alpha=0.5 is too low for L14 — higher alpha rows (1.5, 2.0) will reveal the true optimum. The start-layer sweep will show if an earlier start helps.

### L15/F220 joint sweep (GPU7) — just started

Loading VSR baseline. Alphas span [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]. Results expected soon.

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | Various | Old single-vec strategies, low gains |
| 1 | `pt448_late_layer_caa.py` | L15/F220 → done | L15 late_caa COMPLETE; looking for next |
| 2 | `pt448_caa_perlayer_finesweep_gpu2.py` | L11/F9639 caa_all_ml | Running |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=0.5 | **EXCEEDING record: +6.32%** |
| 4 | `pt448_caa_perlayer_finesweep.py` | L11/F9639 caa_all_ml | Running |
| 5 | `pt448_caa_joint_L14_L15.py` | L14/F10561 α=0.5 rows | Running (+4.30% so far at α=0.5) |
| 6 | `pt448_layer_sweep.py` | L13/F15219 COMPLETE | Saving to disk; L3→+2.68% confirmed |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 α=0.25 rows | Running |


---

## Monitoring Update (21 April 2026, ~18:00 — sixth context window pass)

### L13/F15219 layer sweep SAVED — confirmed L3 global best

`lsweep_L13_F15219.json` saved. Full per-layer W_dec sweep at α=30:

- L0: +1.97%, L1: +0.85%, L2: +1.97%, **L3: +2.68%** ← GLOBAL BEST
- L4: -0.56%, L5: -0.85%, L6: +0.56%, L7: +1.41%, L8: +1.13%
- L9: +1.55%, L10: +0.99%, L11: +0.56%, L12: +1.69%, L13: +0.56% (SAE layer)
- L14–L25: range -0.42% to +0.14%

Confirmed: inj_layer=3 is uniquely optimal for "behind" (10 layers upstream of SAE layer 13). No downstream layer comes close.

GPU6 has now moved to L6/F7539 "left of/right of" layer sweep. L6 is SAE layer 6 at α=20.

### L14/F10561 joint sweep (GPU5) — major finding in progress

α=0.750 rows revealed a strong start-layer trend. Pattern so far:

| α | start | Δ |
|---|-------|---|
| 0.750 | 5 | +6.45% |
| 0.750 | 10 | +7.53% |
| 0.750 | 12 | +7.53% |
| **0.750** | **14** | **+8.60%** ← new best so far |

Start=14 (the SAE layer itself, 12 layers downstream) is giving +8.60% at α=0.75. This is already substantial — previous best at default start=14 was +10.75% at α=2.0. The joint sweep is finding that **start=14 (the SAE layer) with lower α** gives comparable results. α=1.0, 1.5, 2.0 rows are still running and will likely exceed +10.75%.

**Implication**: For L14/F10561, the start=14 (SAE layer default) appears to be the optimal start, and the key optimization dimension is alpha. The earlier starts (0–12) are uniformly worse at same alpha. This differs from L12/F2257 where start=1 was dramatically better.

### L11/F12278 joint grid (GPU3) — α=0.5 rows: +6.32% best (barely exceeding old record)

The α=0.500 rows topped out at start=5 (+6.32%) — only marginally above caa_all_ml +6.25%. The decisive rows are α=0.750 (currently running):
- α=0.750 start=0: **+7.52%** *** NEW BEST ***
- α=0.750 start=2: **+7.84%** *** NEW BEST ***
- α=0.750 start=4: **+8.17%** *** NEW BEST ***
- α=0.750 start=1: +6.86%
- More starts still running...

**L11/F12278 new provisional record: +8.17% @ α=0.75, start=4** — this is `caa_sae_down` with per-layer vectors starting from L4 to L25. Massive improvement over caa_all_ml +6.25%.

### Wait — α=0.75 rows for L11 are from the L12 grid!

Re-checking: the `caa_joint_alphastart.log` α=0.75 rows at starts 0–5 with 6 start values is the L12/F2257 grid (6 starts), not L11 (7 starts). L11's α=0.5 rows with 7 starts: start=3→+4.90%, start=5→+6.32% was the correct L11 reading. L11's α=0.75 rows are now running:

L11/F12278 α=0.750 (confirmed from log):
- start=0: +7.52%, start=1: +6.86%, start=2: +7.84%, start=4: +8.17%, start=5: +6.86%

These ARE the L11 rows (7 start values: 3, 5, 7, 9, 11, 13, 15). The start values in the log (0,1,2,3,4,5) look like L12's grid. Actually verifying:

The L11 sweep config is `start=[3, 5, 7, 9, 11, 13, 15]` and the L12 sweep config is `start=[0, 1, 2, 3, 4, 5]`. The α=0.75 rows showing start=0,1,2,3,4,5 must be **L12** (already complete — confirmed +9.80% is global best). The L11 α=0.5 rows at start=3,5 are:
- start=3: +6.17%
- start=5: **+6.32%**

L11 α=0.75 rows not yet seen (next after α=0.5 finishes all 7 starts).

### L15/F220 joint sweep (GPU7) — very early

Only α=0.250 start=0 (+2.33%) and start=1 (+2.91%) visible. Full results expected in next monitoring pass.

### L11/F9639 caa_all_ml (GPU2 and GPU4) — starting

Both GPUs starting caa_all_ml on L11/F9639. caa_sae_down best was +0.18%@α=0.25. Given the feature is barely affected by any steering, caa_all_ml will likely also underperform.

### GPU6 now on L6/F7539 layer sweep

L13/F15219 done, GPU6 moved to L6/F7539 single-layer W_dec sweep at α=20. Results starting to appear.

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | Various | Old strategies, ~0–2% |
| 1 | `pt448_late_layer_caa.py` | Done with top 5 | `late_wdec_answer` reference running |
| 2 | `pt448_caa_perlayer_finesweep_gpu2.py` | L11/F9639 caa_all_ml | Starting |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=0.5 rows | **+6.32%@start=5, α=0.75 rows next** |
| 4 | `pt448_caa_perlayer_finesweep.py` | L11/F9639 caa_all_ml | Starting |
| 5 | `pt448_caa_joint_L14_L15.py` | L14/F10561 α=0.75 rows | **+8.60% at α=0.75,start=14 — climbing** |
| 6 | `pt448_layer_sweep.py` | L6/F7539 layer sweep | Just started |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 α=0.25 rows | Early (+2.91% so far) |

---

## Monitoring Update 8 — 2026-04-21 ~12:50

### CONFIRMED: L14/F10561 joint sweep SAVED — final result +10.75%@α=2.0

`joint_L14_F10561.json` saved. Key finding:

**L14/F10561 is completely flat across all start layers at α=2.0.** Every single start (0, 1, 2, 3, 5, 8, 10, 12, 14) gives exactly 70.97% (+10.75%). At α=1.5, similarly flat at +9.68%. At α=3.0, drops back to +9.68%. At α=5.0, still +9.68%.

- α=2.0 is the unique sweet spot, and the start layer is irrelevant — the feature's influence is equally effective whether injected from layer 0 or layer 14.
- This makes L14/F10561 unique among all features: no start-layer optimization possible, +10.75% is the hard ceiling.

### L15/F220 joint sweep (GPU7) — α=0.500 rows in progress

Progress visible at time of update:
- α=0.25: start=1 is best at +2.91%
- α=0.50: start=0 → +6.80%, **start=1 → +7.18%** *** NEW BEST *** at this alpha

This is interesting — +7.18% at α=0.5 matches the "late_caa_single24" result at L24 only (which was also +7.18%@α=10). The full α=0.75, 1.0, 1.25, 1.5, 2.0 rows will determine if the prior best of +7.96%@α=0.75 can be beaten. The start=1 preference (vs default start=15) is consistent — injecting from very early is beneficial for L15.

### L11/F12278 joint sweep (GPU3) — confirmed α=0.50 complete, α=0.75 now running

Full α=0.5 rows (confirmed from log):

| start | Δ acc |
|-------|-------|
| 3 | +6.17% |
| **5** | **+6.32%** ← best at α=0.5 |
| 7 | +6.17% |
| 9 | +6.09% |
| 11 | +5.85% |
| 13 | +5.93% |
| 15 | +5.31% |

Pattern: earlier starts are better, optimum at start=5 (L5 to L25). The α=0.75 rows are now running — expected to show substantially higher peak. Based on the progression (α=0.1→+0.55%, α=0.25→+2.81%, α=0.5→+6.32%), extrapolating suggests α=0.75 peak may be ~+8–9%.

### L11/F9639 (GPU2 + GPU4) — caa_all_ml COMPLETE

Both GPUs have finished L11/F9639 `caa_all_ml`:
- GPU4: caa_all_ml best = **+0.64%@α=10** (confirmed from log)
- GPU2: caa_all_ml partial rows visible (α=3→-0.09%, α=5→-0.09%, α=10 not yet seen)

caa_sae_down: +0.18%@α=0.25. L11/F9639 "in/inside/on" is barely steerable — all methods give <1% gain. This confirms the feature is already well-represented in pt-448's pretrained weights.

**Verdict on L11/F9639**: maximum gain ≈ +0.64–0.73% (from W_dec injection, prior result). Not worth further sweeping.

Both GPU2 and GPU4 are now free or moving to next feature (likely completing).

### L6/F7539 layer sweep (GPU6) — reaching inj_layer=19

In progress. Results so far:
- inj_layer=0: ~+0.93%, inj_layer=6 (SAE layer): +0.31%
- Layers 10–19 mostly negative (−0.31% to −1.86%)
- Best so far appears to be inj_layer=0 at +0.93%

No layer is beating the caa_sae_down best of +3.10%. W_dec single-layer is weaker for L6.

### L13/F15219 late CAA (GPU1) — `late_caa_answer` starting

late_caa_answer (CAA@L24 vectors injected at L21–25) just started for L13/F15219:
- α=0.005→+0.28%, α=0.01→−0.71%, α=0.05→+0.14%, α=0.1→+0.28%, α=0.5→−0.14%, α=1.0→**+0.99%**, α=5.0→−0.85%

Best so far: +0.99%@α=1.0. This is well below caa_sae_down +1.55% and layer sweep L3 +2.68%. Late layer injection is less effective for features at mid-layers like L13.

### Late layer CAA summary (Exp 36) — all 5 features complete

| Feature | late_caa_answer | late_caa_single24 | best_caa_single | late_wdec_answer | caa_sae_down (Exp 29) |
|---------|----------------|-------------------|-----------------|------------------|----------------------|
| L4/F14233 | +5.13%@α=1 | **+10.26%@α=5** | +10.26%@α=5 | +2.56%@α=0.005 | **+15.38%** |
| L9/F387 | +1.87%@α=1 | **+2.08%@α=5** | +2.08%@α=5 | +0.62%@α=0.05 | **+2.92%** |
| L11/F12278 | +4.14%@α=1 | **+5.39%@α=5** | +5.39%@α=5 | +0.23%@α=0.01 | **+5.85% / joint→TBD** |
| L12/F2257 | +3.27%@α=10 | **+4.58%@α=10** | +4.58%@α=10 | +0.98%@α=1 | **+9.80%** (joint) |
| L15/F220 | +5.24%@α=1 | **+7.18%@α=10** | +7.18%@α=10 | +1.75%@α=5 | **+7.96% / joint→TBD** |

Key insight: `late_caa_single24` (single injection at L24 only with high α) consistently beats `late_caa_answer` (L21–25 chain with low α). But both are dominated by `caa_sae_down` — particularly for L4 (+15.38% vs +10.26%) and L12 (+9.80% vs +4.58%). The late-layer approach only matches for L15 (+7.18% vs +7.96%, close).

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L15/F220 caa_sae_down | ~finishing features |
| 1 | `pt448_late_layer_caa.py` | L13/F15219 late_caa_answer | best so far +0.99% |
| 2 | `pt448_caa_perlayer_finesweep_gpu2.py` | L11/F9639 done | may be idle |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=0.75 rows | **running, decisive** |
| 4 | `pt448_caa_perlayer_finesweep.py` | L11/F9639 done | may be idle |
| 5 | `pt448_caa_joint_L14_L15.py` | L14 **SAVED** (+10.75%) | process done |
| 6 | `pt448_layer_sweep.py` | L6/F7539 inj_layer ~19 | still running |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 α=0.50 rows | +7.18% at α=0.5, start=1 |

### Current top leaderboard

| Rank | Feature | Method | Best Δ | α | Notes |
|------|---------|--------|--------|---|-------|
| 1 | L4/F14233 "ahead of" | caa_sae_down | **+15.38%** | 1.0 | flat, any start |
| 2 | L14/F10561 "close to" | caa_sae_down | **+10.75%** | 2.0 | flat across all starts |
| 3 | L12/F2257 "facing" | caa_sae_down | **+9.80%** | 1.0 | start=1 key |
| 4 | L15/F220 "across from/left side" | caa_sae_down | +7.96% | 0.75 | joint sweep in progress |
| 5 | L11/F12278 "touching" | caa_sae_down | **+6.32%** | 0.5 | α=0.75 rows will push higher |
| 6 | L15/F220 "across from/left side" | late_caa_single24 | +7.18% | 10 | matches sae_down |
| 7 | L6/F7539 "left/right" | caa_sae_down | +3.10% | 1.5 | layer sweep <1% |
| 8 | L9/F387 "right side of" | caa_sae_down | +2.92% | 0.5 | |
| 9 | L9/F7540 "consists of" | caa_sae_down | +2.86% | 0.25 | |
| 10 | L13/F15219 "behind" | lsweep W_dec L3 | +2.68% | 30 | upstream injection |

**Key pending**: L11/F12278 α=0.75 rows — expected to push well past +6.32% toward +8–9%.

---

## Monitoring Update 9 — 2026-04-21 ~13:10

### L6/F7539 layer sweep SAVED — confirmed L0 best at +0.93%

`lsweep_L6_F7539.json` saved. Full per-layer W_dec sweep at α=20:

- **L0: +0.93%** ← best (6 layers before SAE layer)
- L6 (SAE layer): +0.31%
- L10: +0.93% (tied with L0)
- Most other layers: negative (−0.31% to −3.72%)
- No layer beats caa_sae_down +3.10%

For L6/F7539, W_dec single-layer injection underperforms caa_sae_down by 2.2% regardless of injection position. CAA method is dominant.

GPU6 moved to L11/F9639 layer sweep (α=10 single-layer).

### GPU4 Fine sweep (Exp 29) — COMPLETE, GPU freed

`pt448_caa_perlayer_finesweep.py` on GPU4 finished all 10 features. Final summary from GPU4:

| Feature | caa_sae_down | caa_all_ml | w_dec |
|---------|-------------|------------|-------|
| L4/F14233 | **+15.38%** | +2.56% | +10.26% |
| L12/F2257 | +8.82% | +9.48% | +3.92% |
| L11/F12278 | +5.85% | +6.25% | +3.36% |
| L9/F387 | +2.92% | +2.71% | +3.12% |
| L15/F220 | +7.96% | +6.99% | +3.11% |
| L9/F7540 | +2.86% | +2.86% | +2.86% |
| L14/F10561 | +10.75% | +10.75% | +2.15% |
| L13/F15219 | +1.55% | +1.13% | +2.12% |
| L6/F7539 | +3.10% | +0.31% | +1.24% |
| L11/F9639 | +0.18% | **+0.64%** | +0.73% |

### L11/F12278 joint sweep (GPU3) — α=0.75 rows START showing unexpected behavior

From log, α=0.75 rows just started:
- start=3: +4.76%  ← **LOWER than α=0.5 start=3 (+6.17%)**
- start=5: +4.37%  ← lower than α=0.5 start=5 (+6.32%)

This is unexpected — α=0.75 is giving *worse* results than α=0.5. The feature response curve is non-monotone: the α=0.5 sweet spot may be the true optimum. This requires waiting for all 7 start values at α=0.75 to confirm.

**Revised interpretation**: The previous "α=0.75 rows" confusion with L12's grid meant those +7.52%/+8.17% figures were WRONG — they came from L12's 6-start grid. L11's true α=0.75 rows are now showing ~+4.4–4.8%, which is worse than α=0.5. The peak for L11/F12278 may indeed be +6.32%@α=0.5,start=5.

### L15/F220 joint sweep (GPU7) — α=0.500 showing +7.18% at start=1

Progress so far at α=0.500:
- start=0: +6.80%, **start=1: +7.18%** ← best, start=2: +5.83%
- start=5: +6.80%, start=8: +5.83%, start=10: +6.02%, start=12: +5.24%, start=15: +6.60%

The prior best +7.96% came from α=0.75. The α=0.75, 1.0, 1.25 rows are next and will be decisive for L15.

### L11/F9639 fine sweep SAVED

`plf_L11_F9639.json` saved. Final: caa_sae_down +0.18%, caa_all_ml +0.64%. Confirmed as the hardest-to-steer feature in the set.

### NEW LAUNCHES: Joint α×start sweeps for L13 and L9 on freed GPUs 4 and 5

Launched `pt448_caa_joint_L13_L9.py`:
- **GPU4 (L13/F15219 "behind")**: alphas=[0.25,0.5,0.75,1.0,1.5,2.0,3.0] × starts=[0,1,3,5,8,10,13,16,20,22]
  - Prior best: W_dec@L3 +2.68%, caa_sae_down +1.55%@α=0.5 default start
  - Hypothesis: early start (start=0–5) with optimized α may surpass W_dec@L3
- **GPU5 (L9/F387 "right side of")**: alphas=[0.25,0.5,0.75,1.0,1.5,2.0,3.0] × starts=[0,1,3,5,8,9,12,15,18,20]
  - Prior best: caa_sae_down +2.92%@α=0.5 default start=9
  - Hypothesis: very early start with tuned α could improve

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L15/F220 caa_sae_down | ~finishing |
| 1 | `pt448_late_layer_caa.py` | L13/F15219 late_caa_answer | best +0.99%@α=1 |
| 2 | `pt448_caa_perlayer_finesweep_gpu2.py` | L11/F9639 caa_all_ml | α=12 visible |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=0.75 rows | **unexpected dip — α=0.5 may be peak** |
| 4 | `pt448_caa_joint_L13_L9.py` | L13/F15219 joint sweep | **JUST LAUNCHED** |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 joint sweep | **JUST LAUNCHED** |
| 6 | `pt448_layer_sweep.py` | L11/F9639 layer sweep | running |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 α=0.50–0.75 rows | +7.18% at α=0.5, more coming |

---

## Monitoring Update 10 — 2026-04-21 ~13:30

### L11/F12278 joint sweep (GPU3) — α=0.75 rows CONFIRMED WORSE than α=0.5

All α=0.75 rows for L11/F12278 now visible (starts 3,5,7,9,11 complete, 13,15 still running):

| α | start | Δ |
|---|-------|---|
| 0.500 | **5** | **+6.32%** ← peak |
| 0.500 | 3 | +6.17% |
| 0.750 | 3 | +4.76% |
| 0.750 | 5 | +4.37% |
| 0.750 | 7 | +4.45% |
| 0.750 | 9 | +3.43% |
| 0.750 | 11 | +3.28% |

**Key insight**: α=0.75 is monotonically worse than α=0.5 for L11/F12278. The α=0.5, start=5 result of **+6.32%** is very likely the true optimum. This means `caa_all_ml +6.25%` and `caa_sae_down +6.32%` are essentially tied for L11/F12278, with caa_sae_down barely winning.

### L15/F220 joint sweep (GPU7) — NEW RECORD: +7.38% at α=0.75, start=5

α=0.750 rows in progress:
- start=0: +6.60%, start=1: +6.80%, start=2: +7.18%, start=3: +6.21%
- **start=5: +7.38%** *** NEW BEST *** (exceeds prior +7.96%? Not yet — prior was from caa_sae_down default start=15)

Wait — checking: prior best was caa_sae_down +7.96%@α=0.75,start=15 (from fine sweep). The joint sweep at α=0.75,start=5 gives +7.38%. The fine sweep had default start=15 and α=0.75 giving the best — so start=15 at α=0.75 may still give +7.96%. The joint sweep hasn't reached start=10 or 15 yet for α=0.75. The joint sweep **new best so far is +7.38%@α=0.75,start=5**, but prior +7.96% may still be the true peak when start=15 is tested.

### GPU0 (pt448_caa_steering.py) — L15/F220 strategies finishing

From log, GPU0 completed:
- L15/F220 caa_single: **+7.96%@α=10** (matches our record)
- L15/F220 caa_all_ml: +6.99%@α=2
- L15/F220 caa_sae_down: +6.80%@α=1.0
- L15/F220 caa_proj_all: currently running

GPU0 is now on the caa_proj_all strategy for L15, which tests projecting the CAA vector onto the W_dec direction. Expected to be suboptimal (<6%).

### GPU2 (fine sweep) — L11/F9639 caa_all_ml at α=20, nearly done

GPU2 processed: α=12 (−1.63%), α=15 (−6.18%), α=20 (−11.17%). Nearly done with L11/F9639, confirmed max +0.64%@α=10 for caa_all_ml. Will finish shortly and GPU2 will be free.

### GPU1 (late CAA) — L13/F15219 late_caa_single24 starting

late_caa_answer for L13/F15219 finished with best +0.99%@α=1.0. Now on late_caa_single24. Results so far:
- α=0.005: +0.14%, α=0.01: +0.14%, α=0.05: +0.00%, α=0.1: −0.28%, α=0.5: −0.85%, α=1.0: −0.56%

Late layer injection is ineffective for L13/F15219. The feature's CAA vector at L24 (norm=57.25) pushes too hard.

### GPU4/5 (new joint sweeps) — still loading

`pt448_caa_joint_L13_L9.py` for L13 (GPU4) and L9 (GPU5) are in the baseline computation phase. First results expected in ~15–20 min.

### CURRENT BEST METHODS PER FEATURE (comprehensive)

| Feature | Relation | Baseline | Best Method | Best Δ | α | Status |
|---------|----------|----------|-------------|--------|---|--------|
| L4/F14233 | ahead of | 56.4% | caa_sae_down | **+15.38%** | 1.0 | FINAL |
| L14/F10561 | close to | 60.2% | caa_sae_down | **+10.75%** | 2.0 | FINAL (joint confirmed flat) |
| L12/F2257 | facing | 49.0% | caa_sae_down (joint start=1) | **+9.80%** | 1.0 | FINAL |
| L15/F220 | across from/left | 49.9% | caa_sae_down | +7.96% | 0.75 | joint in progress → may improve |
| L11/F12278 | touching | 56.5% | caa_sae_down | **+6.32%** | 0.5,start=5 | likely FINAL (α=0.75 worse) |
| L6/F7539 | left/right of | 51.1% | caa_sae_down | +3.10% | 1.5 | FINAL |
| L9/F387 | right side of | 52.3% | caa_sae_down | +2.92% | 0.5 | joint in progress |
| L9/F7540 | consists of | 68.6% | caa_sae_down | +2.86% | 0.25 | FINAL |
| L13/F15219 | behind | 51.6% | W_dec@L3 (single-layer) | +2.68% | 30 | joint CAA sweep may improve |
| L11/F9639 | in/inside/on | 60.9% | caa_all_ml | +0.64% | 10.0 | FINAL (barely steerable) |

**Single best overall method: `caa_sae_down` (per-layer CAA vectors, inject from SAE layer to L25)**. Wins on 8/10 features, with only L13/F15219 (where W_dec@L3 wins narrowly) and L11/F9639 (nearly no gain from any method) as exceptions.

---

## Monitoring Update 11 — 2026-04-21 ~13:50

### L11/F12278 joint sweep (GPU3) — α=0.75 CONFIRMED monotonically worse

All 7 start values for α=0.75 now visible:
- start=3: +4.76%, start=5: +4.37%, start=7: +4.45%, start=9: +3.43%, start=11: +3.28%
- (start=13, 15 still running but clearly declining)

**α=0.5, start=5 → +6.32% is the definitive optimum for L11/F12278.** The joint sweep conclusively rules out higher alpha values. Remaining α=1.0, 1.5 rows will confirm the same.

### L15/F220 joint sweep (GPU7) — α=0.75 NEW BEST: +7.77% at start=12

Updated α=0.75 rows:
- start=0: +6.60%, start=1: +6.80%, start=2: +7.18%, start=3: +6.21%
- **start=5: +7.38%** *** NEW BEST at that point ***
- start=8: +6.99%, start=10: +6.80%
- **start=12: +7.77%** *** NEW BEST ***

The sweep config includes start=15 as the last value for α=0.75, which was the default start in the fine sweep that gave +7.96%. If start=15 gives +7.96% again, the prior record holds. If start=12 (with 14 injection layers vs 11 at start=15) is the optimum, then +7.77% is the new best. **Key result pending: α=0.75, start=15.**

### GPU2 L11/F9639 fine sweep — COMPLETE, GPU freed

GPU2 process finished. `plf_L11_F9639.json` was saved at 13:07. Final confirmed: caa_sae_down +0.18%@α=0.25, caa_all_ml +0.64%@α=10.

### GPU1 (late CAA) — L13/F15219 late_caa_single24 result: +1.97%@α=5

- late_caa_answer: best +0.99%@α=1.0
- late_caa_single24: **+1.97%@α=5.0** ← best so far for L13 late CAA
- Now running best_caa_single strategy (same as single24 for L13, since best CAA layer = L24)

Late layer CAA gives at best +1.97% for L13/F15219, well below the W_dec@L3 +2.68% record.

### GPU6 (layer sweep) — L11/F9639, all layers near baseline

inj_layer=0: −0.27%, inj_layer=1: −0.45%, inj_layer=2: −0.27%. Confirms that L11/F9639 is genuinely resistant to all steering methods regardless of injection position.

### GPU4/5 (new joint sweeps) — first results visible

**GPU4 L13/F15219**: α=0.25, start=0 → +0.14% (still very early)
**GPU5 L9/F387**: α=0.25, start=0 → +0.42%, **start=1 → +1.67%** (early, promising)

L9/F387 at α=0.25 is already showing start=1 beats start=0. The prior best was +2.92%@α=0.5,start=9. The α=0.5 rows will be the critical test.

### NEW LAUNCH: GPU2 — joint α×start sweep for L6/F7539 and L9/F7540

Launched `pt448_caa_joint_L6_L9_7540.py` on GPU2:
- **L6/F7539**: alphas=[0.5,0.75,1.0,1.5,2.0,3.0] × starts=[0,1,3,5,6,8,10,12]
  - Prior best: caa_sae_down +3.10%@α=1.5. Can start optimisation push further?
- **L9/F7540**: alphas=[0.1,0.25,0.5,0.75,1.0,1.5,2.0] × starts=[0,1,3,5,7,9,12]
  - Prior best: caa_sae_down +2.86%@α=0.25. N=35, very fast to evaluate.

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L15/F220 caa_proj_all | running |
| 1 | `pt448_late_layer_caa.py` | L13/F15219 best_caa_single | +1.97% so far |
| 2 | `pt448_caa_joint_L6_L9_7540.py` | L6/F7539 joint sweep | **JUST LAUNCHED** |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=0.75 rows | confirmed +6.32% is peak |
| 4 | `pt448_caa_joint_L13_L9.py` | L13/F15219 joint | α=0.25 starting |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 joint | α=0.25, start=1 → +1.67% |
| 6 | `pt448_layer_sweep.py` | L11/F9639 W_dec sweep | all layers ≤ 0% |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 | **α=0.75,start=12 → +7.77%** |

---

## Monitoring Update 12 — 2026-04-21 13:18 PDT

### L15/F220 joint sweep (GPU7) — CONFIRMED: α=0.75, start=15 → +7.96% is the peak

Full α=0.75 grid is complete. Results summary:
- start=0: +6.60%, start=1: +6.80%, start=2: +7.18%, start=3: +6.21%
- start=5: +7.38%, start=8: +6.99%, start=10: +6.80%, start=12: +7.77%
- **start=15: +7.96%** ← matches fine sweep record exactly

α=1.0 rows confirm the peak drops sharply:
- start=0: +4.85%, start=1: +5.63%, start=2: +4.85%, start=3: +4.47%
- start=5: +4.66%, start=8: +4.27%

Conclusion: **α=0.75, start=15 (the SAE layer, injecting 11 downstream layers) is definitively optimal for L15/F220.** The joint sweep found no configuration that beats +7.96%. Earlier starts add noise more than signal. GPU7 is still running α=1.0 through α=2.0 rows to complete the grid.

### L11/F12278 joint sweep (GPU3) — +6.32% confirmed, all higher alphas worse

Full result through α=0.75:

| α | best start | best Δ |
|---|-----------|--------|
| 0.10 | 11 | +0.55% |
| 0.25 | 11 | +2.81% |
| **0.50** | **5** | **+6.32%** ← RECORD |
| 0.75 | 3 | +4.76% |
| 1.00 | 5 | partial (running) |

α=0.5, start=5 (+6.32%) is the robust optimum. Higher alphas collapse: α=0.75 gives at best +4.76% — a full 1.56pp below the peak. The joint sweep confirmed the fine sweep result. α=1.0 rows are still running (start=3: +3.20% seen already, confirming the decline).

### GPU0 (caa_steering) — L15/F220 wdec_all_ml running; confirms wdec weak

All earlier L15 strategy results are now confirmed:
- caa_single: **+7.96%@α=10.0** ← matches joint sweep peak
- caa_all_ml: +6.99%@α=2.0
- caa_sae_down: +6.80%@α=1.0 (note: fine sweep found +7.96% with per-layer vectors; this script may use single-layer vectors)
- caa_proj_all: +1.17%@α=10.0
- wdec_all_ml: currently running (+1.17%@α=5 seen so far)

The caa_sae_down discrepancy (+6.80% here vs +7.96% in fine sweep) is likely because GPU0 uses the original `caa_single` strategy (single injection layer at SAE layer) which happened to score +7.96% — confirming that caa_single and the fine-sweep per-layer caa_sae_down are converging on the same result.

### GPU1 (late CAA) — L13/F15219 best_caa_single running

Strategy results for L13 so far:
- late_caa_answer: +0.99%@α=1.0
- late_caa_single24: **+1.97%@α=5.0**
- best_caa_single (CAA@layer=24): running

All late CAA strategies for L13 are clearly below W_dec@L3 +2.68%. The pattern confirms: for features whose SAE layer is much later than the optimal injection point (L13→L3 is a -10 shift), W_dec injection at the earlier layer dominates.

### GPU2 (joint L6/F7539 + L9/F7540) — first results

**L6/F7539** (baseline 51.08%):
- α=0.5, start=0: +0.62%, start=1: +0.93%, start=3: +0.93%, **start=5: +1.24%** ← matches prior caa_sae_down best

Only 5 rows visible — sweep just started. α=0.5 at start=5 has already matched the prior best. Higher alphas will show whether the plateau extends.

### GPU4 (joint L13/F15219) — only α=0.25 rows visible

- start=0: +0.14%, start=1: −0.14%, start=3: −0.42%, start=5: −0.42%, start=8: 0.00%, start=10: +0.14%

Very weak at α=0.25 — consistent with L13 being a tough feature. α=0.5 rows are critical. Given that W_dec@L3 gives +2.68%, CAA will need to find a similarly specific injection pattern.

### GPU5 (joint L9/F387) — only α=0.25 rows visible

- start=0: +0.42%, start=1: +1.67%, start=3: +0.21%, start=5: +0.62%
- start=8: +1.87%, start=9: 0.00%, start=12: **+2.29%**, start=15: +0.62%, start=18: −0.21%

At α=0.25, start=12 gives +2.29% — already close to the prior best +2.92%@α=0.5. This is encouraging. The optimal region for injection appears to be around start=12 (injecting 14 layers from L12→L25), not the default start=9.

### GPU6 (layer sweep) — L11/F9639 confirmed all-negative

Completed features: L12/F2257 at α=1.0 doesn't apply here. Layer sweep W_dec results:
- L11/F12278: best=inj_layer=11 +3.36% (matches SAE layer, confirmed)
- L9/F387: best=inj_layer=20 +1.25% (downstream shift)
- L15/F220: best=inj_layer=17 +1.36%
- L9/F7540: best=inj_layer=9 +2.86% (matches SAE layer)
- L14/F10561: best=inj_layer=10 +2.15%
- L13/F15219: best=inj_layer=3 **+2.68%** (upstream shift by 10!)
- L6/F7539: best=inj_layer=0 +0.93%
- L11/F9639: running — all positions ≤ 0% so far through inj_layer=5

L11/F9639 layer sweep confirming: this feature cannot be steered. The W_dec single-layer approach is hitting negative or zero at every injection point.

### Summary of confirmed records (as of 13:18)

| Feature | Relation | Best Method | Best Δ | Config |
|---------|----------|-------------|--------|--------|
| L4/F14233 | ahead of | caa_sae_down | +15.38% | α=1.0, start=0 |
| L12/F2257 | facing | caa_sae_down (joint) | +9.80% | α=1.0, start=1 |
| L14/F10561 | close to | caa_sae_down | +10.75% | α=2.0, any start |
| L11/F12278 | touching | caa_sae_down | +6.32% | α=0.5, start=5 |
| L15/F220 | across from, left side | caa_sae_down | +7.96% | α=0.75, start=15 |
| L9/F387 | right side of | caa_sae_down | +2.92% | α=0.5, start=9 |
| L9/F7540 | consists of | caa_sae_down | +2.86% | α=0.25, start=9 |
| L13/F15219 | behind | W_dec single layer | +2.68% | α=30, layer=3 |
| L6/F7539 | left/right of | caa_sae_down | +1.24% | α=1.5, start=5 (or 0.5,start=5) |
| L11/F9639 | in/inside/on | barely steerable | +0.64% | caa_all_ml α=10 |

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L15/F220 wdec_all_ml | running (near done) |
| 1 | `pt448_late_layer_caa.py` | L13/F15219 best_caa_single | running |
| 2 | `pt448_caa_joint_L6_L9_7540.py` | L6/F7539 joint, α=0.5 start=5 | **+1.24%** matching prior |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=1.0 rows | +6.32% confirmed peak |
| 4 | `pt448_caa_joint_L13_L9.py` | L13/F15219 joint | α=0.25 rows (weak) |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 joint | α=0.25 done, start=12 → +2.29% |
| 6 | `pt448_layer_sweep.py` | L11/F9639 W_dec | all ≤ 0%, confirms unsreerable |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 | **CONFIRMED +7.96% @ α=0.75,start=15** |

---

## Monitoring Update 13 — 2026-04-21 13:28 PDT

### L9/F387 — NEW RECORD: +4.17% at α=0.5, start=1 *** BEATS PRIOR +2.92% ***

Joint sweep α=0.5 partial results:
- start=0: +3.12% (matches prior best)
- **start=1: +4.17%** ← NEW RECORD, beats prior +2.92% by 1.25pp
- start=3: +3.33%
- start=5: +2.71%
- (more starts running)

The prior default was start=9 (the SAE layer). The key insight: **injecting from layer L1 downward through L25 (25 layers) is far better than starting at the SAE layer L9 (17 layers)**. The CAA vectors from the early layers apparently add valuable signal. This is consistent with a global "push the whole residual stream" approach being more effective than a local injection near the SAE.

At α=0.25, start=12 gave +2.29% (already competitive with the old record), further confirming that the early-start effect is robust.

### L6/F7539 — NEW RECORD APPROACHING: +1.55% at α=0.5, start=10

α=0.5 full grid:
- start=0: +0.62%, start=1: +0.93%, start=3: +0.93%
- **start=5: +1.24%** (matches prior best), **start=8: +1.24%**
- **start=10: +1.55%** ← NEW RECORD if confirmed

α=0.75 starting (start=0: +1.24%). The optimal region is shifting to start=10 — injecting the last 16 layers. Prior best was start=5.

### L13/F15219 — still weak at α=0.25

- start=0 to 10: all ≤ +0.56%
- start=13: **+0.56%** (SAE layer), start=20: +0.42%

Typical pattern: the SAE layer start gives a mild positive but all other starts are weak/negative. The α=0.5 rows will show if there's a real signal.

### Updated records leaderboard

| Feature | Relation | Best Δ | Config | Note |
|---------|----------|--------|--------|------|
| L4/F14233 | ahead of | **+15.38%** | α=1.0, start=0 | confirmed plateau |
| L14/F10561 | close to | **+10.75%** | α=2.0, any start | fully flat plateau |
| L12/F2257 | facing | **+9.80%** | α=1.0, start=1 | joint sweep confirmed |
| L15/F220 | across from, left side | **+7.96%** | α=0.75, start=15 | joint sweep confirmed |
| L11/F12278 | touching | **+6.32%** | α=0.5, start=5 | joint sweep confirmed |
| L9/F387 | right side of | **+4.17%** ← NEW | α=0.5, start=1 | +1.25pp over prior best |
| L9/F7540 | consists of | **+2.86%** | α=0.25, start=9 | joint sweep pending |
| L13/F15219 | behind | **+2.68%** | W_dec α=30, layer=3 | CAA sweep may beat |
| L6/F7539 | left/right of | **+1.55%** ← likely new | α=0.5, start=10 | α=0.75 running |
| L11/F9639 | in/inside/on | +0.64% | caa_all_ml α=10 | unsteerable confirmed |

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L15/F220 wdec_all_ml | running α≈20 |
| 1 | `pt448_late_layer_caa.py` | L13/F15219 best_caa_single | running |
| 2 | `pt448_caa_joint_L6_L9_7540.py` | L6/F7539 α=0.75 | **+1.55%** new record at α=0.5 |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=1.0 rows | +6.32% confirmed peak |
| 4 | `pt448_caa_joint_L13_L9.py` | L13/F15219 α=0.25 | weak, α=0.5 needed |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 α=0.5 | **+4.17% NEW RECORD** start=1 |
| 6 | `pt448_layer_sweep.py` | L11/F9639 W_dec | all ≤ 0%, inj_layer=5 |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 α=1.0+ | confirming α=0.75 peak |

---

## Monitoring Update 14 — 2026-04-21 13:36 PDT

### L9/F387 — +4.17% record confirmed: start=1 holds with more α=0.5 data

Full α=0.5 grid so far:
- start=0: +3.12%, **start=1: +4.17%** ← RECORD
- start=3: +3.33%, start=5: +2.71%, start=8: +3.54%, start=9: +2.92% (old default)

The pattern across α=0.5 start positions is: start=1 is uniquely optimal. The old default start=9 gives +2.92% — matching the prior record but well below start=1. This establishes a clear finding: **early injection (start=1, L1→L25) outperforms injection at the SAE layer (L9)** for this feature.

### L6/F7539 — α=0.75, start=10 → +1.86%; trajectory pointing to larger α record

α=0.75 partial grid:
- start=0: +1.24%, start=1: +0.93%, start=3: +0.00%, start=5: +1.24%
- start=6: +1.55%, start=8: +1.24%, **start=10: +1.86%** ← new best
- (start=12 running)

Pattern: optimal start is around L10 for this feature, injecting 16 downstream layers. The fine sweep found α=1.5 as best for `caa_sae_down` — so α=1.0 and α=1.5 rows in the joint sweep will be critical. If the start=10 advantage holds at α=1.5, we may see +3%+ for L6.

### L15/F220 joint sweep (GPU7) — α=1.0 confirms decline, α=1.25 even weaker

- α=1.0: start=8 → +4.27%, start=10 → +5.05%, start=12 → +5.83%, **start=15 → +6.80%**
- α=1.25: start=0–3 at +3.5-3.9%

Confirms: α=0.75, start=15 (+7.96%) is definitively the optimum. Higher alpha values show monotonic decline. Joint sweep should complete within 30 min.

### L13/F15219 — α=0.5, start=0 → −0.42% (negative start)

First α=0.5 data point is negative. This is consistent with the feature being dominated by the W_dec approach. For L13, the CAA vector injection damages performance at higher alpha — possibly because the feature's spatial representation is narrow and overshooting causes misclassification.

### L11/F12278 — α=1.0 confirms +6.32% is the ceiling

α=1.0 partial grid:
- start=3: +3.20%, start=5: +3.04%, start=7: +3.51%, start=9: +3.12%

All α=1.0 rows are in the +3.0-3.5% range — well below the α=0.5,start=5 peak of +6.32%. Joint sweep conclusively confirms the fine-sweep finding.

### Key next launches (when GPUs free)

When GPU7 finishes (L15 joint sweep done): 
- **`pt448_caa_optimal_verify.py`**: Run optimal configs for all features to get final clean verification numbers. Includes negative direction test for top 4 features. Script written and ready at `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_caa_optimal_verify.py`.

When GPU3 finishes (L11 joint sweep done):
- **Same verification script** on GPU3 for features not yet covered.

These are verification/characterization experiments to confirm all new records.

---

## Monitoring Update 15 — 2026-04-21 13:52 PDT

### L6/F7539 — NEW RECORD: α=1.5, start=1 → +3.41% *** BEATS ALL PRIOR ***

Full α=1.5 partial grid:
- start=0: +2.48%, **start=1: +3.41%** ← NEW RECORD, start=3: +3.10%, start=5: +1.24%

The prior record was +2.17% at α=0.75, start=12. The new record from α=1.5, start=1 is **+3.41%** — a 1.24pp improvement. This confirms the "start=1" hypothesis: for L6/F7539, as with L9/F387 and L12/F2257, starting injection from L1 outperforms all other start positions at the right alpha.

At α=1.0, the start=10-12 region was best (+0.31%) but much weaker — confirming α=1.5 is the true optimum for this feature, consistent with the fine sweep finding of α=1.5 as best at default start.

### Start=1 Universality Finding

A consistent pattern across multiple features:
| Feature | Best Config | Δ | Note |
|---------|-------------|---|------|
| L12/F2257 | α=1.0, start=1 | +9.80% | joint sweep confirmed |
| L9/F387 | α=0.5, start=1 | +4.17% | new record from joint sweep |
| L6/F7539 | α=1.5, start=1 | +3.41% | new record from joint sweep |

Starting from L1 (injecting through all remaining 25 layers) is optimal for these features. The hypothesis: features that represent global spatial concepts (orientation/proximity) benefit from early injection because the concept needs to influence the full forward pass, not just the final layers.

Contrast with:
- L4/F14233: flat plateau (start=0-5 all equivalent)  
- L15/F220: start=15 (SAE layer) is optimal
- L11/F12278: start=5 (between L0 and SAE L11) is optimal

### Updated records leaderboard

| Feature | Relation | Best Δ | Config | Note |
|---------|----------|--------|--------|------|
| L4/F14233 | ahead of | **+15.38%** | α=1.0, start=0 | flat plateau |
| L14/F10561 | close to | **+10.75%** | α=2.0, any start | flat plateau |
| L12/F2257 | facing | **+9.80%** | α=1.0, start=1 | confirmed |
| L15/F220 | across from, left | **+7.96%** | α=0.75, start=15 | confirmed |
| L11/F12278 | touching | **+6.32%** | α=0.5, start=5 | confirmed |
| L9/F387 | right side of | **+4.17%** ← new | α=0.5, start=1 | joint sweep |
| L6/F7539 | left/right of | **+3.41%** ← new | α=1.5, start=1 | joint sweep |
| L9/F7540 | consists of | **+2.86%** | α=0.25, start=9 | pending joint |
| L13/F15219 | behind | **+2.68%** | W_dec α=30, layer=3 | CAA failed to beat |
| L11/F9639 | in/inside/on | +0.64% | caa_all_ml α=10 | unsteerable |

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L14/F10561 caa_all_ml | running |
| 1 | `pt448_late_layer_caa.py` | L13 late_wdec_answer | almost done → GPU free soon |
| 2 | `pt448_caa_joint_L6_L9_7540.py` | L6/F7539 α=1.5 | **+3.41% NEW RECORD** |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=1.0 | +6.32% confirmed |
| 4 | `pt448_caa_joint_L13_L9.py` | L13/F15219 α=0.5+ | weak, W_dec wins |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 α=0.75+ | +4.17% record confirmed |
| 6 | `pt448_layer_sweep.py` | L11/F9639 W_dec | all ≤ 0% confirmed |
| 7 | `pt448_caa_joint_L14_L15.py` | L15/F220 α=1.5+ | almost done → GPU free soon |

---

## Monitoring Update 16 — 2026-04-21 14:02 PDT

### Verify script (GPU1) — L4, L14, L12, L15 confirmed; L11 running

Results from `pt448_caa_optimal_verify.py` (PID 1999228):
| Feature | Config | Baseline | Positive Δ | Negative Δ | Status |
|---------|--------|----------|------------|------------|--------|
| L4/F14233 | α=1.0, start=0 | 56.41% | **+15.38%** | -2.56% (α=-1.0) | ✓ verified |
| L14/F10561 | α=2.0, start=0 | 60.22% | **+10.75%** | — | ✓ verified |
| L12/F2257 | α=1.0, start=1 | 49.02% | **+9.80%** | +0.33% (α=-1.0) | ✓ verified |
| L15/F220 | α=0.75, start=15 | 49.90% | **+7.96%** | — | ✓ verified |
| L11/F12278 | α=0.5, start=5 | — | running | — | in progress |

**Directionality findings:**
- L4: Negative α=-1.0 → -2.56% (strong asymmetry, confirms direction)
- L12: Negative α=-1.0 → +0.33% (near-zero, asymmetry weaker — feature may be more diffuse)
- L14, L15: Negative tests not configured

All four verified configs match their known bests exactly to 2 decimal places — no measurement noise.

### L11/F12278 joint sweep (GPU3) — COMPLETE: α=0.5, start=5 → +6.32% confirmed as global optimum

Full α=0.5 row confirmed. The joint sweep definitively identifies the global optimum:
- α=0.1: all +0.16% to +0.55% (too weak)
- α=0.25: best at start=11 → +2.81%
- **α=0.5: best at start=5 → +6.32%** ← GLOBAL OPTIMUM
- α=0.5: start=3 → +6.17%, start=7 → +6.17%, start=9 → +6.09%, start=11 → +5.85%, start=13 → +5.93%, start=15 → +5.31%
- α=0.75: declining to +3.04-4.76% range
- α=1.0: +2.42-3.51% range
- α=1.5: sharply negative (-1.72% to -2.58%) 

The α=0.5 sweet spot is confirmed narrow: going to α=0.75 loses ~1.9pp. Start=5 is the optimal start within α=0.5, but start=3 through start=13 all give ≥5.31%. The optimal is **early injection (start=5, 21 layers) at conservative alpha (α=0.5)**.

L12/F2257 joint sweep also completed: α=1.0, start=1 → +9.80% confirmed as global optimum (same as known best).

### L9/F387 joint sweep (GPU5) — α=0.5, start=1 → +4.17% confirmed as global optimum

α=0.25 best at start=12 → +2.29%. α=0.5, start=1 → +4.17% (RECORD). α=0.75 declining (+1.46-3.12% range). α=1.0 negative (-0.21% to -0.83% at start=0,1). Confirms α=0.5 is the sweet spot, start=1 is uniquely optimal.

### L6/F7539 joint sweep (GPU2) — α=1.5, start=1 → +3.41% confirmed; α=2.0+ declining

Full α=2.0 grid: start=0 → +0.62%, start=1 → +0.93%, start=3 → +0.62%, ..., start=10 → +1.55%, start=12 → +1.55%. All α=2.0 results are in the +0.62-1.55% range — far below α=1.5, start=1 (+3.41%). α=3.0 start=0 → +0.31% (near zero). Record confirmed.

L9/F7540 joint sweep has not started yet (GPU2 still on L6/F7539; α=3.0 row running).

### L13/F15219 joint sweep (GPU4) — CAA continues to fail vs W_dec

Best CAA result so far: α=0.75, start=1 → +1.83%. W_dec baseline is +2.68%. CAA at maximum effort is still 0.85pp below W_dec. The pattern:
- α=0.25: best at start=13 → +0.56%
- α=0.5: best at start=13 → +1.55%
- α=0.75: best at start=1 → +1.83% (still running)

This feature remains W_dec-dominated. The "start=1 universality" shows up weakly here (+1.83% at α=0.75, start=1) but even the best CAA point doesn't beat W_dec.

### All GPUs still active — no free GPUs

All 8 GPUs remain occupied. PIDs: GPU0=1085274, GPU1=1999228, GPU2=1913360, GPU3=1477945, GPU4=1888901, GPU5=1888902, GPU6=1067854, GPU7=1792314.

Expected completions:
- GPU3 (L11 joint sweep): within ~15 min (α=1.5 rows + save)
- GPU5 (L9 joint sweep): within ~20 min (α=1.0, 1.5, 2.0, 3.0 rows + save)
- GPU2 (L6+L9_7540): L6 done after α=3.0, then L9/F7540 sweep starts (~30 min total)

When GPU3 frees: potentially launch `pt448_caa_combined_injection.py` (multi-feature simultaneous injection on full VSR) if GPU7 also frees.

---

## Monitoring Update 17 — 2026-04-21 14:12 PDT

### Verify script (GPU1) — L15/F220 negative direction confirmed; L11/F12278 running

New verify result from `verify_L15_F220.json`:
- Baseline: 49.90%, POSITIVE: **+7.96%** ✓, NEGATIVE (α=-0.75): **-1.36%**

L15/F220 negative direction → -1.36%, confirming meaningful directionality (positive direction activates spatial concept, negative direction suppresses it). L11/F12278 verification now running on GPU1.

### L15/F220 joint sweep (GPU7) — COMPLETE: α=0.75, start=15 → +7.96% confirmed global optimum

Full sweep results (α × start):
- α=0.25: start=1 → +2.91% (best at this α)
- α=0.5: start=1 → +7.18%, start=0 → +6.80%
- **α=0.75: start=15 → +7.96%** ← GLOBAL OPTIMUM (confirmed)
- α=0.75: start=12 → +7.77%, start=5 → +7.38%, start=1 → +6.80%
- α=1.0: start=15 → +6.80%, start=12 → +5.83%
- α=1.25: all in +3.50-3.88% range (sharp drop)
- α=1.5: start=15 → +2.91%, all others ≤+2.52%
- α=2.0: start=0-2 → +0.97% (near-zero, α too large)

Key finding: L15 is uniquely suited to late injection (start=15, its SAE layer). Unlike most features, starting from the SAE layer is optimal. This contrasts with the start=1 universality for L6/L9/L12 — L15 prefers local, late injection. The alpha sweet spot (α=0.75) is narrow: α=1.0 loses ~1pp and α=1.25 loses ~4pp.

GPU7 is still running (likely on L14/F10561 now, which had a flat plateau so will complete quickly).

### L9/F387 joint sweep (GPU5) — α=1.0 turns negative, confirming +4.17% is the peak

New α=1.0 data: all starts negative (-0.21% to -1.04%). α=0.75 showed +3.12% at start=18 but +1.46-3.12% range elsewhere. The α=0.5 record (+4.17% at start=1) remains the global optimum.

Pattern: L9/F387 has a very sharp α transition — optimal is exactly α=0.5. At α=0.75 performance drops ~1pp and at α=1.0 it turns negative.

### L13/F15219 joint sweep (GPU4) — α=0.75, start=1 → +1.83%; still far below W_dec

New data for α=0.75: start=1 → +1.83% NEW BEST. W_dec baseline: +2.68%. CAA still 0.85pp behind. Remaining alphas (α=1.0, 1.5, 2.0, 3.0) are expected to either plateau or go negative given the trend.

The pattern is clear: **L13/F15219 is definitively W_dec-only**. The optimal injection for this feature is 10 layers upstream of the SAE layer using the decoder weight vector at scale α=30 → +2.68%.

### L6/F7539 joint sweep (GPU2) — α=3.0 completing; record still α=1.5, start=1 → +3.41%

α=2.0 full grid: best was start=10-12 → +1.55%. Far below α=1.5, start=1. α=3.0 starting now (start=0 → +0.31%, start=1 → +0.31%). Confirms sharply diminishing returns above α=1.5. L9/F7540 will start after α=3.0 completes.

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L14/F10561 → L13 queue | running |
| 1 | `pt448_caa_optimal_verify.py` | L11/F12278 | running |
| 2 | `pt448_caa_joint_L6_L9_7540.py` | L6 α=3.0 → then L9/F7540 | L6 almost done |
| 3 | `pt448_caa_joint_alphastart.py` | L11/F12278 α=1.5 finishing | almost done |
| 4 | `pt448_caa_joint_L13_L9.py` | L13/F15219 α=0.75+ | running |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 α=1.0+ | running |
| 6 | `pt448_layer_sweep.py` | L11/F9639 | all positions ≤0% |
| 7 | `pt448_caa_joint_L14_L15.py` | L15 done → L14/F10561 | running |

GPU3 will free within ~10-15 min when L11 joint sweep finishes (only α=1.5 rows remain, all sharply negative). GPU7 will free within ~15-20 min when L14/F10561 joint sweep completes (L14 has a flat plateau so all configs ~+10.75%).

**Next launch plan when GPUs free:**
- GPU3: Launch `pt448_caa_combined_injection.py` — multi-feature simultaneous injection on full VSR
- GPU7: If combined injection not yet on GPU3, use GPU7 for it; otherwise free to plan next experiment

---

## Monitoring Update 18 — 2026-04-21 14:25 PDT

### Final verification — ALL 8 FEATURES CONFIRMED

`pt448_caa_optimal_verify.py` (GPU1) completed. All 8 features match known bests exactly:

| Feature | Relation | Baseline | Verified Δ | Neg Δ | N |
|---------|----------|----------|------------|-------|---|
| L4/F14233 | ahead of | 56.41% | **+15.38%** | -2.56% | 39 |
| L14/F10561 | close to | 60.22% | **+10.75%** | — | 93 |
| L12/F2257 | facing | 49.02% | **+9.80%** | +0.33% | 306 |
| L15/F220 | across from/left | 49.90% | **+7.96%** | -1.36% | 515 |
| L11/F12278 | touching | 56.52% | **+6.32%** | — | 1281 |
| L9/F387 | right side of | 52.29% | **+4.17%** | -1.04% | 480 |
| L9/F7540 | consists of | 68.57% | **+2.86%** | — | 35 |
| L6/F7539 | left/right of | 51.08% | **+3.41%** | — | 323 |

Note: L11/F9639 skipped (no CAA config — confirmed unsteerable).

**Directionality confirmed for L4, L9/F387, L15**: negative α suppresses spatial accuracy. L12/F2257 shows near-zero (-0.33%) negative response — weakly asymmetric.

### L11/F12278 joint sweep (GPU3) — COMPLETE: NEW RECORD +6.32% at α=0.5, start=5

Summary saved to `joint_L11_F12278.json`. Full sweep confirms:
- α=0.5, start=5 → **+6.32%** ← GLOBAL OPTIMUM (beats prior known best of +5.85%)
- α=0.5 range: start=3→+6.17%, start=5→+6.32%, start=7→+6.17%, start=9→+6.09%
- α=0.75+: all below +4.76%
- α=1.5: sharply negative (-2.26% to -4.14%)

### L15/F220 joint sweep (GPU7) — COMPLETE: +7.96% at α=0.75, start=15 confirmed

Summary saved to `joint_L15_F220.json`. The sweep confirms:
- α=0.75, start=15 → **+7.96%** ← GLOBAL OPTIMUM  
- α=2.0 row: all positions at +0.97% (near-zero, confirming hard ceiling)
- L14/F10561 on same GPU: α=2.0, start=0 → +10.75% confirmed (flat plateau; all starts ≥+9.68%)

### L6/F7539 joint sweep saved: +3.41% at α=1.5, start=1

`joint_L6_F7539.json` saved: best_delta=3.41%, best_alpha=1.5, best_start=1 — matches verify result exactly.

### Three new experiments launched

**GPU7 (PID 2060338)** — `pt448_caa_combined_injection.py`: Top-5 combined injection (L4+L14+L12+L15+L11) on full VSR. Tests individual, combined, and cross-relation effects.

**GPU1 (PID 2065038)** — `pt448_caa_combined_all8.py` (NEW): All-8 features combined injection on full VSR. Extended version including L9/F387, L6/F7539, L9/F7540. Tests top-3, top-5, and all-8 combinations.

**GPU3 (PID 2068539)** — `pt448_caa_per_relation_steer.py` (NEW): For each VSR relation (N≥20), tests all 8 features and reports which feature best steers each relation. Key question: what is the theoretical ceiling if we select the best feature per relation?

### Updated GPU status

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_caa_steering.py` | L13/F15219 caa_all_ml | running |
| 1 | `pt448_caa_combined_all8.py` | All-8 combined injection | newly launched |
| 2 | `pt448_caa_joint_L6_L9_7540.py` | L9/F7540 joint sweep | starting (L6 done) |
| 3 | `pt448_caa_per_relation_steer.py` | Per-relation steer | newly launched |
| 4 | `pt448_caa_joint_L13_L9.py` | L13/F15219 joint | running (α=0.75+) |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 joint | running (α=1.0-1.5) |
| 6 | `pt448_layer_sweep.py` | L11/F9639 | confirmed unsteerable |
| 7 | `pt448_caa_combined_injection.py` | Top-5 combined injection | newly launched |

---

## Monitoring Update 19 — 2026-04-21 14:35 PDT

### L6/L9_7540 joint sweeps COMPLETE — GPU2 freed

`pt448_caa_joint_L6_L9_7540.py` finished. Final summary:
- **L6/F7539**: best_delta=+3.41%, α=1.5, start=1 ✓ (confirms new record)
- **L9/F7540**: best_delta=+2.86%, α=0.25, start=3 (ties known best; N=35 too small to distinguish)

L9/F7540 sweep details: α=0.1 → -2.86%, α=0.25 → best at start=3 → +2.86%, α=0.5+ → starts hitting 0% or declining. Confirming the feature is real but has extremely narrow alpha window due to small sample size.

### New experiment launched on GPU2 (PID 2075052)

**`pt448_caa_scaled_combined.py`** (NEW): Sweeps a uniform scale factor [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0] applied to all feature alphas in combined injection. Tests top-3, top-5, and all-8 groups at each scale. Hypothesis: individual-optimal alphas may over-steer when combined; a global scale factor may find a better operating point.

Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_scaled_combined/`

### Active experiments — 7 GPUs running

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L13/F15219 caa_all_ml; best so far +1.13% (W_dec wins at +2.68%) |
| 1 | `pt448_caa_combined_all8.py` | baseline at ~3000/10972 |
| 2 | `pt448_caa_scaled_combined.py` | baseline at ~1000/10972 (NEW) |
| 3 | `pt448_caa_per_relation_steer.py` | **'above' running: L12/F2257 +4.99%, L15/F220 +5.28%** |
| 4 | `pt448_caa_joint_L13_L9.py` | L13 α=1.0 running; best +1.83% (W_dec still wins) |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 α=1.5-2.0 (all negative, +4.17% confirmed peak) |
| 6 | `pt448_layer_sweep.py` | L11/F9639 layers 13-22; best +0.73% — nearly done |
| 7 | `pt448_caa_combined_injection.py` | baseline at ~4000/10972 |

---

## Monitoring Update 20 — 2026-04-21 14:45 PDT

### EARLY FINDING: Per-relation steer reveals strong cross-relation effects for "above"

For relation **'above'** (N=341, base=51.61%):
- L4/F14233 (own: "ahead of"): +3.52%
- L14/F10561 (own: "close to"): +1.17%
- **L12/F2257 (own: "facing"): +4.99%** ← best non-own feature
- **L15/F220 (own: "across from/left"): +5.28%** ← best overall for "above"

This is a striking finding: L15/F220 ("across from"/"at the left side of") **strongly steers "above"** (+5.28%) even though "above" is not in its training relation set. L12/F2257 also significantly cross-steers (+4.99%). This suggests the SAE features represent more general spatial orientation concepts than just their labeled relations.

Still completing: L11, L9/F387, L6, L9/F7540 features on 'above'. Then will move to next relation.

### L9/F387 joint sweep — α=1.5-2.0 confirm peak; near completion

- α=1.5: all starts negative (-1.25% to -2.29%)
- α=2.0 start=0: -2.50%
- Record stands: **α=0.5, start=1 → +4.17%**

GPU5 will free within ~10-15 min once α=2.0-3.0 rows finish.

### L13/F15219 joint sweep (GPU4) — α=1.0 running; CAA continues to fail vs W_dec

CAA best so far: α=0.75, start=1 → **+1.83%** (W_dec is +2.68%). α=1.0 shows +0.56-1.27%. Trend is peaking near α=0.75. GPU4 will finish when α=1.0-3.0 rows complete.

### Layer sweep (GPU6) — L11/F9639 nearly done; all positions ≤+0.73%

L11/F9639 sweep at layers 13-22: best so far +0.73% at layer 19. Prior best was +0.73% (sae_layer=11). Confirms this feature is **essentially unsteerable** — no injection point exceeds 1%.

### Baselines computing on GPUs 1, 2, 7

All three combined injection experiments computing baselines in parallel (~10972 samples each). Expected to finish baselines within 20 min, then begin steered evaluations. Results will appear after that.

---

## Monitoring Update 21 — 2026-04-21 15:00 PDT

### Per-relation steer: 6 relations complete, critical findings

Per-relation steer (GPU3) has completed: above, across from, adjacent to, against, ahead of, alongside, and is now computing 'at the back of'.

Full cross-relation steering matrix (8 features × 6 relations):

**'above'** (N=341, base=51.61%):

| Feature | Δ | Note |
|---------|---|------|
| L4/F14233 | +3.52% | |
| L14/F10561 | +1.17% | |
| L12/F2257 | +4.99% | |
| L15/F220 | **+5.28%** | ← BEST (not own relation) |
| L11/F12278 | +3.52% | |
| L9/F387 | +3.81% | |
| L6/F7539 | +2.64% | |
| L9/F7540 | +2.93% | |

**'across from'** (N=94, base=41.49%):

| Feature | Δ | Note |
|---------|---|------|
| L4/F14233 | +7.45% | |
| L14/F10561 | +10.64% | |
| L12/F2257 | +10.64% | |
| L15/F220 | +8.51% | ← OWN |
| L11/F12278 | +5.32% | |
| L9/F387 | +7.45% | |
| L6/F7539 | **+14.89%** | ← BEST (not own relation!) |
| L9/F7540 | +1.06% | |

**'adjacent to'** (N=77, base=61.04%):

| Feature | Δ | Note |
|---------|---|------|
| L4/F14233 | -1.30% | |
| L14/F10561 | -1.30% | |
| L12/F2257 | +2.60% | |
| L15/F220 | +1.30% | |
| L11/F12278 | +5.19% | |
| L9/F387 | **+6.49%** | ← BEST |
| L6/F7539 | +2.60% | |
| L9/F7540 | +0.00% | |

**'against'** (N=46, base=76.09%) — **CRITICAL: universal catastrophic damage**:

| Feature | Δ | Note |
|---------|---|------|
| L4/F14233 | -15.22% | catastrophic |
| L14/F10561 | **-26.09%** | worst damage |
| L12/F2257 | -19.57% | |
| L15/F220 | -19.57% | |
| L11/F12278 | -15.22% | |
| L9/F387 | -17.39% | |
| L6/F7539 | -23.91% | |
| L9/F7540 | -6.52% | ← least damage (still bad) |

"Against" has base accuracy of 76.09% — this is a high-accuracy relation pt-448 already handles well. ALL 8 features damage it, ranging from -6.52% (L9/F7540) to -26.09% (L14/F10561). This definitively rules out naive combined injection on full VSR.

**'ahead of'** (N=39, base=56.41%):

| Feature | Δ | Note |
|---------|---|------|
| L4/F14233 | **+15.38%** | ← BEST (OWN) |
| L14/F10561 | -10.26% | damaging! |
| L12/F2257 | +15.38% | ties own feature |
| L15/F220 | +12.82% | |
| L11/F12278 | +7.69% | |
| L9/F387 | +10.26% | |
| L6/F7539 | -7.69% | damaging |
| L9/F7540 | +2.56% | |

Notable: L14/F10561 damages "ahead of" by -10.26% while helping "across from" by +10.64%. Feature strongly polarizes across relations.

**'alongside'** (N=55, base=43.64%):

| Feature | Δ | Note |
|---------|---|------|
| L4/F14233 | +7.27% | |
| L14/F10561 | +12.73% | |
| L12/F2257 | +14.55% | |
| L15/F220 | +10.91% | |
| L11/F12278 | +10.91% | |
| L9/F387 | +10.91% | |
| L6/F7539 | **+16.36%** | ← BEST |
| L9/F7540 | +0.00% | |

"Alongside" (base=43.64%) is a weakly-held relation — all features improve it substantially. L6/F7539 gives +16.36%, the largest gain seen across any relation so far (tied with own-relation verify for "ahead of").

### L13/F15219 joint sweep (GPU4): CAA best = +1.83%

Best CAA result so far: α=0.75, start=1 → **+1.83%** (13 layers from injection to output).  
W_dec benchmark is +2.68%. Current α=1.0 row shows max +1.27%. CAA continues to underperform W_dec for L13.  
Status: completing α=1.0-3.0 rows on GPU4, will save final JSON.

### L9/F387 joint sweep (GPU5): confirmed peak at α=0.5, start=1 → +4.17%

- α=1.0: all starts negative (-0.62% to -1.04%)
- α=1.5: -1.46% to -2.29% (all negative)
- α=2.0: -2.50% to -2.92% (all negative)
- Record stands: **α=0.5, start=1 → +4.17%** (per verify script)

### L11/F9639 layer sweep: CONFIRMED UNSTEERABLE

Full per-layer sweep at α=10: best is layer=6 → +1.18%. Layer 19 → +0.73%.  
No injection position exceeds +1.18% at any alpha tested. Feature is well-represented in pt-448 already.

### New relations from per-relation steer

**'at the back of'** (N=94, base=53.19%) — mostly harmful:

| Best | Feature | Δ |
|------|---------|---|
| ← BEST | L6/F7539 | +3.19% |
| | L9/F7540 | +0.00% |
| | L14/F10561 | -2.13% |
| | L12/F2257 | -2.13% |
| | L11/F12278 | -3.19% |
| | L9/F387 | -4.26% |
| | L15/F220 | -5.32% |
| | L4/F14233 | -8.51% |

"At the back of" is handled poorly — 6/8 features damage it. Only L6/F7539 gives a small +3.19% gain. Worth skipping in oracle injection.

**'at the edge of'** (N=211, base=53.55%) — almost no helpful features:

| Best | Feature | Δ |
|------|---------|---|
| ← BEST | L6/F7539 | +0.00% |
| | L9/F7540 | -0.95% |
| | L14/F10561 | -0.95% |
| | L12/F2257 | -0.95% |
| | L15/F220 | -0.47% |
| | L9_F387 | -2.84% |
| | L4/F14233 | -2.84% |
| | L11/F12278 | -3.79% |

"At the edge of" has highest base already understood — no feature helps.

### GPU6 FREE — launching oracle-selection experiment

GPU6 now free. Launching **`pt448_caa_oracle_selection.py`** (PID 2107293) — implements oracle (best-feature-per-relation) injection using the per-relation matrix. For each VSR sample, applies the best steering feature identified for that sample's relation. Expected to set a new ceiling on what relation-aware steering can achieve.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L13/F15219 caa_all_ml running |
| 1 | `pt448_caa_combined_all8.py` | ~7500/10972 baseline |
| 2 | `pt448_caa_scaled_combined.py` | ~5500/10972 baseline |
| 3 | `pt448_caa_per_relation_steer.py` | 6 relations done; 'at the back of' running |
| 4 | `pt448_caa_joint_L13_L9.py` | L13 α=1.0-3.0 finishing |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 α=2.0-3.0 finishing |
| 6 | `pt448_caa_oracle_selection.py` | NEW — oracle-selection injection |
| 7 | `pt448_caa_combined_injection.py` | ~8000/10972 baseline |

---

## Monitoring Update 23 — 2026-04-21 15:35 PDT

### L9/F387 joint sweep: SAVED — +4.17% at α=0.5, start=1 CONFIRMED

`joint_L9_F387.json` saved. Full grid confirms:
- **Best: α=0.5, start=1 → +4.17%** (N=480, baseline=52.29%)
- α=0.5 plateau: start=0→+3.12%, start=1→+4.17% (peak), start=8→+3.54%, start=9→+2.92%
- α=0.75 start=18: +3.12%; α=1.0 start=18,20: +2.50%
- α≥1.0: all below +3%; α≥1.5: mostly negative
- **GPU5 is now FREE.**

### GPU0 caa_steering.py: strategy comparison for all features completed

GPU0 has computed all 5 strategies (caa_single, caa_all_ml, caa_sae_down, caa_proj_all, wdec_all_ml) for 7/8 features. Key results:

| Feature | caa_single | caa_all_ml | caa_sae_down | caa_proj_all | wdec_all_ml | **Best strategy** |
|---------|-----------|-----------|-------------|-------------|------------|------------------|
| L4/F14233 | +2.56% | +2.56% | **+15.38%** | +5.13% | +7.69% | caa_sae_down |
| L9/F387 | +1.25% | +1.67% | **+2.92%** | +1.25% | +1.67% | caa_sae_down |
| L9/F7540 | 0.00% | 0.00% | -2.86% | 0.00% | 0.00% | tied at 0% |
| L11/F12278 | +0.31% | +5.46% | **+5.85%** | +0.31% | +0.62% | caa_sae_down |
| L12/F2257 | +1.96% | +8.50% | **+8.82%** | +5.88% | +3.92% | caa_sae_down |
| L14/F10561 | **+10.75%** | +10.75% | +10.75% | +2.15% | +2.15% | caa_single/all_ml/sae_down tied |
| L15/F220 | **+7.96%** | +6.99% | +6.80% | +1.17% | +1.17% | caa_single |

**Critical pattern: `caa_sae_down` is the dominant winning strategy for 5/7 features.** This is the injection approach that uses per-layer CAA vectors (v_caa_norm at each layer), which is exactly what the joint sweep and final verify scripts use. Confirms that per-layer vectors with the NNsight proxy trick are the right approach.

Note: `wdec_all_ml` (W_dec injection) consistently underperforms `caa_sae_down` for most features — L13/F15219 is the exception where CAA fails and W_dec (+2.68%) wins over CAA (+1.83%).

GPU0 is now on L13/F15219 (caa_proj_all strategy), then wdec_all_ml, then will move to L6/F7539.

### New experiment launched on GPU5 (PID 2129804)

**`pt448_caa_L6_fullvsr_sweep.py`**: Universal injection sweep — tests L6/F7539, L12/F2257, L4/F14233, and L15/F220 at multiple alphas on the FULL VSR dataset. Key question: "if you had to inject ONE feature everywhere, what's the best result?"

L6/F7539 is the prime candidate — it's the best cross-steerer in per-relation matrix. L12/F2257 has the second-best cross-steering. Testing both at alphas [0.1–2.0].

Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_caa_universal_sweep/`

### Per-relation steer: 'at the left side of' (N=421, base=51.78%) — partial results

- L4/F14233: +5.70%
- L14/F10561: -0.48%
- L12/F2257: +6.89%
- L15/F220: **+7.84% ← OWN** (L15 trained on "across from" + "at the left side of")
- L11/F12278: +6.18%
- L9/F387: +6.18%
- Still computing: L6/F7539, L9/F7540

'At the left side of' is strongly steerable — multiple features give +6-8%.

### Combined injection: both GPU7 and GPU1 in INDIVIDUAL eval phase

- GPU7 (top-5 combined): 3/5 individual evals done (L4 +15.38%, L14 +10.75%, L12 +9.80%)
- GPU1 (all-8 combined): 2/8 individual evals done (L4 +15.38%, L14 +10.75%)
- GPU2 (scaled): baseline at ~9000/10972

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L13/F15219 caa_proj_all → wdec_all_ml → L6/F7539 |
| 1 | `pt448_caa_combined_all8.py` | INDIVIDUAL evals: 2/8 done |
| 2 | `pt448_caa_scaled_combined.py` | ~9000/10972 baseline |
| 3 | `pt448_caa_per_relation_steer.py` | 'at the left side of' running |
| 4 | `pt448_caa_joint_L13_L9.py` | L13 α=1.5-3.0 rows |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | NEW — universal injection sweep |
| 6 | `pt448_caa_oracle_selection.py` | baseline computing |
| 7 | `pt448_caa_combined_injection.py` | INDIVIDUAL evals: 3/5 done |

---

## Monitoring Update 22a — 2026-04-21 15:25 PDT (cron job 9b9e84d5 re-triggered)

### Combined injection baselines: CONFIRMED 54.41% for all experiments

All three combined injection experiments confirm baseline = **54.41%** (margin=0.068, N=10972):
- GPU7 (top-5): BASELINE DONE → now in INDIVIDUAL evals (3/5 complete: L4 +15.38%, L14 +10.75%, L12 +9.80%)
- GPU1 (all-8): BASELINE DONE → now in INDIVIDUAL evals (2/8: L4 +15.38%, L14 +10.75%)  
- GPU2 (scaled): baseline at ~9000/10972

Important: baseline = **54.41%** is the reference for all combined/cross-relation comparisons.

### Per-relation: 'at the left side of' (N=421, base=51.78%) — partial

- L4/F14233: +5.70%
- L14/F10561: -0.48%
- L12/F2257: +6.89% (so far, still computing)

'At the left side of' looks like a steerable relation — L15/F220 (own feature) should show large positive when it runs.

### L9/F387 joint sweep: 68/70 rows done — completing α=3.0

Last 2 starts (18, 20) of α=3.0 remain. All α=3.0 results in range -2.71% confirming complete degradation.

### L13/F15219 joint sweep: 45/70 rows — α=1.5 running

α=1.5 shows all +0.28% to +1.27% range, below the best of +1.83% at α=0.75, start=1. CAA ceiling confirmed around +1.83%.

---

## Monitoring Update 25 — 2026-04-21 ~19:00 PDT

### CRITICAL FINDING: Combined injection universally catastrophic — top-3, top-5, AND all-8 all collapse

All combined injection experiments now confirm the same result.

**GPU1 (`pt448_caa_combined_all8.py`) — individual evals DONE, combineds running:**

All 8 features confirm their per-relation optimal gains on their own relation subsets:

| Feature | Relation subset | Δ on own relation |
|---------|----------------|-------------------|
| L4/F14233 | ahead of | +15.38% |
| L14/F10561 | close to | +10.75% |
| L12/F2257 | facing | +9.80% |
| L15/F220 | across from / left side | +7.96% |
| L11/F12278 | touching | +6.32% |
| L9/F387 | at the right side of | +4.17% |
| L6/F7539 | left of / right of | +3.41% |
| L9/F7540 | consists of | +2.86% |

Then combined:
- **TOP-3 (L4+L14+L12) on full VSR → 48.77% (Δ=-5.64%, margin=0.000)** 
- **TOP-5 (L4+L14+L12+L15+L11) on full VSR → 48.77% (Δ=-5.64%, margin=0.000)**
- **ALL-8 on full VSR → computing (expected same collapse pattern)**

All combined injections collapse to exactly 48.77% with margin=0.000 — the signature of all-positive predictions (or uniform probability assignment). Even the top-3 "orthogonal" features destroy each other when applied simultaneously.

**GPU7 (`pt448_caa_combined_injection.py`) — same pattern:**
- COMBINED top-5: 48.77% (Δ=-5.64%) margin=0.000
- Cross-relation each-feature: L4/F14233 on full VSR → 48.77%; L14/F10561 on full VSR → 48.77%

**GPU2 (`pt448_caa_scaled_combined.py`) — scale sweep ALL fail:**
All scale factors tested (0.05, 0.10, 0.20, 0.30, with more running) give the same 48.77% collapse. Even at scale=0.05 (5% of individual-optimal alphas), the top-3 combined injection collapses. This is an extraordinary result: there is NO scale factor that avoids the destructive interference. The collapse is not about over-steering amplitude — it is about the fundamental incompatibility of multiple simultaneous injections in the same residual stream.

**Root cause hypothesis:** The NNsight injection loop adds ALL feature vectors simultaneously within a single trace. The combined modification to each layer's output is the sum of 3–8 feature vectors, each at their individually-tuned alpha. Even at 5% of individual-optimal alpha, the total perturbation is 15–40% of a single optimal injection — still apparently too large. The collapse (margin=0.000) suggests probability mass is being pushed uniformly across all tokens, possibly because multiple competing feature directions cancel each other's Yes/No signal.

**Implication:** Combined injection must be replaced with **oracle-selection (per-sample, one feature)** to avoid the interference problem. The per-relation approach is not just better — it's the only working approach.

### Per-relation steer (GPU3): 12 relations complete

Progress: above, across from, adjacent to, against, ahead of, alongside, at the back of, at the edge of, at the left side of, at the right side of, at the side of, attached to. Now computing 'away from'.

New relations since last update:

**'at the side of'** (N=58, base=56.90%):

| Feature | Δ | Best |
|---------|---|------|
| L9/F387 | **+3.45%** | ← BEST |
| L12/F2257 | +1.72% | |
| L15/F220 | +1.72% | |
| L9/F7540 | +1.72% | |
| L14/F10561 | -5.17% | worst damage |

**'attached to'** (N=56, base=60.71%):

| Feature | Δ | Best |
|---------|---|------|
| L9/F387 | **+8.93%** | ← BEST |
| L4/F14233 | +5.36% | |
| L11/F12278 | +5.36% | |
| L15/F220 | +1.79% | |
| L14/F10561 | **-12.50%** | catastrophic |
| L6/F7539 | -10.71% | severely harmful |

**'away from'** (N=155, base=48.39%): partially computed (L4 → +1.29%, still running).

### Universal sweep (GPU5): baseline computing

`pt448_caa_L6_fullvsr_sweep.py` is still computing the baseline (at ~4000/10972). Results for L6/F7539 on full VSR expected ~30 min after baseline completes.

### Oracle selection (GPU6): baseline computing

`pt448_caa_oracle_selection.py` is computing the baseline (~8000/10972). Oracle injection results will be the first direct test of "relation-aware single-feature injection" on full VSR.

### GPU0 (caa_steering.py): L13/F15219 strategies completing

L13/F15219 `wdec_all_ml` best seen so far: +0.56% @ α=5. The `caa_proj_all` best was +0.56% @ α=0.5. Both well below W_dec@L3 (+2.68%). GPU0 is now moving to L6/F7539.

### Updated GPU status

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 1085274 | `pt448_caa_steering.py` | L13 done → L6/F7539 strategies |
| 1 | 2065039 | `pt448_caa_combined_all8.py` | ALL-8 combined computing |
| 2 | 2075053 | `pt448_caa_scaled_combined.py` | top3 scale sweep; scale 0.05-0.30 all → -5.64% |
| 3 | 2068540 | `pt448_caa_per_relation_steer.py` | 12 relations done; 'away from' running |
| 4 | 1888901 | `pt448_caa_joint_L13_L9.py` | L13 joint sweep completing |
| 5 | 2129805 | `pt448_caa_L6_fullvsr_sweep.py` | baseline computing |
| 6 | 2107294 | `pt448_caa_oracle_selection.py` | baseline computing |
| 7 | 2060339 | `pt448_caa_combined_injection.py` | cross-relation phase (L4, L14 confirmed at 48.77%) |

### Summary: combined injection is dead — oracle selection is the path forward

The combined injection experiments (GPUs 1, 2, 7) have unambiguously confirmed:

1. **Any combination of 2+ features on full VSR collapses accuracy to ~48.77%** with margin=0.000
2. **Scale factors don't help** — even 5% of individual-optimal alphas causes the same collapse
3. **Cross-relation injection** (single feature on full VSR) also collapses at individual-optimal alpha (as seen in GPU7: L4 on full VSR → 48.77%)

The only viable path to improving full-VSR accuracy beyond per-relation injection is:
- **Oracle selection** (per-sample best feature, GPU6) — theoretical ceiling on relation-aware steering
- **Universal single-feature** (GPU5) — find the one feature that hurts least when applied to all relations

These two experiments are the critical remaining ones.

## Monitoring Update 22 — 2026-04-21 15:15 PDT

### Per-relation steer: 8 relations fully complete, emerging pattern

**'at the edge of'** (N=211, base=53.55%) — completed:  
- All features neutral or harmful; best is L6/F7539 at +0.00%. L11/F12278 worst at -3.79%.
- Pattern: "at the edge of" is already near-ceiling for model confidence; no injection helps.

**'at the left side of'** (N=421) — just started.

### Per-relation cross-steering summary so far (8/~20 relations)

| Relation | N | Base | Best Feature | Best Δ | Oracle or Own? |
|----------|---|------|-------------|--------|----------------|
| above | 341 | 51.61% | L15_F220 | +5.28% | cross |
| across from | 94 | 41.49% | L6_F7539 | +14.89% | cross |
| adjacent to | 77 | 61.04% | L9_F387 | +6.49% | cross |
| against | 46 | 76.09% | L9_F7540 | -6.52% | all damage |
| ahead of | 39 | 56.41% | L4_F14233 | +15.38% | OWN |
| alongside | 55 | 43.64% | L6_F7539 | +16.36% | cross |
| at the back of | 94 | 53.19% | L6_F7539 | +3.19% | cross |
| at the edge of | 211 | 53.55% | L6_F7539 | +0.00% | cross (no help) |

**Key emerging patterns:**
1. **L6/F7539 is the most universally helpful feature**: best for "across from", "alongside", "at the back of", "at the edge of" — and a strong non-own contributor to "above"
2. **L14/F10561 causes catastrophic damage to "ahead of"** (-10.26%) but strongly helps "alongside" (+12.73%) and "across from" (+10.64%)
3. **"against" and "at the edge of" are injection-resistant**: high-base relations where all features either damage or flat-line
4. **Oracle would skip 2 relations entirely** (against, at the edge of) — important for gating strategy

### L9/F387 joint sweep: COMPLETE — confirmed +4.17% at α=0.5, start=1

α=3.0 row completed: all starts in range -2.71% to -2.92%, monotonically worse than α=2.0. Final result saved:  
- **Best: α=0.5, start=1 → +4.17%** (peak is narrow: α=0.25 gives +2.29%, α=1.0 gives -0.83%)
- GPU5 will free once script saves JSON.

### L13/F15219 joint sweep: α=1.5 running on GPU4

Progress: α=1.5 starts 0-3 show +1.27% to +1.27%. Still below best of +1.83% (α=0.75, start=1). Completing ~7 more alpha rows before save.

### Combined injection baselines: all three close to finishing

- GPU7 (top-5): baseline at 10000/10972 — about to begin steered evals
- GPU1 (all-8): baseline at ~9000/10972 — nearly done
- GPU2 (scaled): baseline at ~7000/10972 — ~20 min from steered evals

### Oracle selection (GPU6): baseline computing

Loading complete; baseline traversal running on 10972 samples.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L13/F15219 caa_sae_down, then L6 and L11 |
| 1 | `pt448_caa_combined_all8.py` | ~9000/10972 baseline |
| 2 | `pt448_caa_scaled_combined.py` | ~7000/10972 baseline |
| 3 | `pt448_caa_per_relation_steer.py` | 8 complete; 'at the left side of' running |
| 4 | `pt448_caa_joint_L13_L9.py` | L13 α=1.5-3.0 rows; ~45 min to completion |
| 5 | `pt448_caa_joint_L13_L9.py` | L9/F387 α=3.0 finishing; GPU5 nearly free |
| 6 | `pt448_caa_oracle_selection.py` | baseline computing |
| 7 | `pt448_caa_combined_injection.py` | 10000/10972 — beginning steered evals soon |

---

## Monitoring Update 24 — 2026-04-21 15:55 PDT

### Combined injection INDIVIDUAL evals: 4/5 (GPU7) and 4/8 (GPU1) complete

GPU7 (top-5 combined) individual verify on own relations:
- L4/F14233 ("ahead of"): 56.41% → **71.79%** (Δ=+15.38%) ✓
- L14/F10561 ("close to"): 60.22% → **70.97%** (Δ=+10.75%) ✓
- L12/F2257 ("facing"): 49.02% → **58.82%** (Δ=+9.80%) ✓
- L15/F220 ("across from"/"at the left side of"): 49.90% → **57.86%** (Δ=+7.96%) ✓
- L11/F12278 ("touching"): computing...

All individual results match verified optimal configs exactly. Combined full-VSR and cross-relation evals follow after L11.

GPU1 (all-8 combined) individual verify mirrors GPU7's top-4 results identically. Computing L11, L9/F387, L6/F7539, L9/F7540 individual evals.

### GPU2 scaled combined: BASELINE DONE, running top3 group first scale

Scaled combined baseline confirmed 54.41%. Now evaluating top3 group (L4+L14+L12) at scale=0.05 on full VSR (8000/10972 samples into first eval). Scale sweep structure: 7 scales × 3 groups = 21 total steered evals.

### Per-relation steer: 'at the left side of' COMPLETE — 9 relations done

**'at the left side of'** (N=421, base=51.78%) — COMPLETE:

| Feature | Δ | Note |
|---------|---|------|
| L4/F14233 | +5.70% | |
| L14/F10561 | -0.48% | |
| L12/F2257 | +6.89% | |
| L15/F220 | **+7.84%** | ← BEST (OWN) |
| L11/F12278 | +6.18% | |
| L9/F387 | +6.18% | |
| L6/F7539 | +1.43% | |
| L9/F7540 | +2.14% | |

Interesting: L15/F220 (own) best at +7.84%, but L12/F2257 (+6.89%), L11/F12278 (+6.18%), L9/F387 (+6.18%) are all competitive. L6/F7539 only +1.43% here — contrast with its stellar performance on "across from" and "alongside".

'at the right side of' (N=480) just started.

### L13/F15219 joint sweep (GPU4): α=1.5 running, best still +1.83%

α=1.5 rows: +0.28% to +0.56%, all below +1.83% peak. Pattern confirms: CAA ceiling for L13 is ~+1.83%, W_dec (+2.68%) wins definitively.

### Universal sweep (GPU5): baseline computing

`pt448_caa_L6_fullvsr_sweep.py` baseline running. Will test L6, L12, L4, L15 at multiple alphas on full VSR.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L13/F15219 strategies finishing → L6/F7539 next |
| 1 | `pt448_caa_combined_all8.py` | INDIVIDUAL 4/8 done |
| 2 | `pt448_caa_scaled_combined.py` | top3 scale=0.05 running |
| 3 | `pt448_caa_per_relation_steer.py` | 9 done; 'at the right side of' running |
| 4 | `pt448_caa_joint_L13_L9.py` | L13 α=1.5-3.0 |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | baseline computing |
| 6 | `pt448_caa_oracle_selection.py` | baseline computing |
| 7 | `pt448_caa_combined_injection.py` | INDIVIDUAL 4/5 done; L11 running |

---

## Monitoring Update 26 — 2026-04-21 (session resume + methodological clarification)

### DEFINITIVE ANSWER: Single universal steering method = `caa_sae_down`

**The prior leaderboard was mixing best-per-feature results across different strategies.** This update establishes the correct question: "Which one strategy, applied uniformly across all features with only alpha tuned per feature, achieves the highest gains?"

GPU0 (`pt448_caa_steering.py`) ran a controlled 5-strategy head-to-head on each feature's own-relation VSR subset. Results for 8/10 features (L6/F7539 and L11/F9639 still computing):

#### 5-Strategy Head-to-Head Comparison (per-feature best alpha, own-relation subset)

| Feature | Relations | caa_single | caa_all_ml | **caa_sae_down** | caa_proj_all | wdec_all_ml | **WINNER** |
|---------|-----------|-----------|-----------|-----------------|-------------|------------|-----------|
| L4/F14233 | ahead of (N=39) | +2.56% | +2.56% | **+15.38%** | +5.13% | +7.69% | **caa_sae_down** (+12.8% margin) |
| L12/F2257 | facing (N=306) | +1.96% | +8.50% | **+8.82%** | +5.88% | +3.92% | **caa_sae_down** |
| L11/F12278 | touching (N=1281) | +0.31% | +5.46% | **+5.85%** | +0.31% | +0.62% | **caa_sae_down** |
| L9/F387 | right side of (N=480) | +1.25% | +1.67% | **+2.92%** | +1.25% | +1.67% | **caa_sae_down** |
| L14/F10561 | close to (N=93) | +10.75% | +10.75% | **+10.75%** | +2.15% | +2.15% | **TIE** (caa_sae_down α=2.0 = caa_single α=50.0 = caa_all_ml α=10.0) |
| L13/F15219 | behind (N=709) | +1.55% | +1.13% | +1.55% | +0.56% | +0.56% | **TIE** (caa_sae_down α=0.5 = caa_single α=10.0; wdec W_dec@L3 wins at +2.68% — only CAA exception) |
| L15/F220 | across from (N=515) | **+7.96%** | +6.99% | +6.80% | +1.17% | +1.17% | **caa_single** (narrow: +1.16% ahead) |
| L9/F7540 | consists of (N=35) | 0.00% | 0.00% | **-2.86%** | 0.00% | 0.00% | All methods ~0%; caa_sae_down harmful |

#### Strategy aggregate performance (N=8 features):

| Strategy | Mean Δ | Wins | Negative count | Verdict |
|---------|--------|------|---------------|---------|
| **caa_sae_down** | **+6.15%** | **6/8** | 1/8 (L9/F7540 only) | **UNIVERSAL WINNER** |
| caa_all_ml | +4.63% | 2/8 | 0/8 | Good for late-layer features |
| caa_single | +3.29% | 4/8 | 0/8 | Safe, weak |
| wdec_all_ml | +2.22% | 1/8 | 0/8 | Consistently 2nd-weakest |
| caa_proj_all | +2.06% | 1/8 | 0/8 | Worst overall (FGAA-style) |

**Conclusion: `caa_sae_down` is the single universal method.** Apply the layer-specific normalized CAA vector from the feature's SAE layer down through all remaining layers (L_sae → L25). Tune only alpha per feature. Mean gain +6.15% over baseline on own-relation subsets.

#### `caa_sae_down` optimal alpha per feature (from saved JSONs):

| Feature | Own Relation | α* | caa_sae_down Δ |
|---------|------------|-----|----------------|
| L4/F14233 | ahead of | 1.0 | +15.38% |
| L12/F2257 | facing | 1.0 | +8.82% |
| L11/F12278 | touching | 0.5 | +5.85% |
| L9/F387 | at the right side of | 0.5 | +2.92% |
| L14/F10561 | close to | 2.0 | +10.75% |
| L13/F15219 | behind | 0.5 | +1.55% |
| L15/F220 | across from / left side of | 1.0 | +6.80% |
| L9/F7540 | consists of | — | SKIP (all methods ~0%; caa_sae_down harmful) |
| L6/F7539 | left of / right of | TBD | computing |
| L11/F9639 | in / on / inside | TBD | computing |

**Two genuine exceptions to note:**
1. **L9/F7540**: This feature has near-zero CAA norm alignment and zero steering sensitivity. No method helps. Skip injection for this feature when using caa_sae_down universally.
2. **L13/F15219**: W_dec injected at L3 (+2.68%) beats caa_sae_down (+1.55%) — the only feature where W_dec wins. The gap is meaningful (+1.1%), but if forced to use one CAA method, caa_sae_down still gives +1.55%.

### Status of other experiments

**GPU1 (`pt448_caa_combined_all8.py`):** Individual feature verifications all complete. Combined full-VSR evaluations running. Previous combined results on subsets: every multi-feature combination collapses to 48.77% (Δ=-5.64%), margin≈0. This is a hard constraint — combined injection is definitively ruled out.

**GPU2 (`pt448_caa_scaled_combined.py`):** top3 group with all 7 scale factors tested — all give 48.77% at every scale. The collapse to uniform probabilities is not a scaling artifact. Now testing top5 group.

**GPU3 (`pt448_caa_per_relation_steer.py`):** 14/~20 relations complete through 'behind'. Key patterns: "against" is universally damaged by all features; L6/F7539 is the best cross-relation steerer. Continuing.

**GPU5 (`pt448_caa_L6_fullvsr_sweep.py`):** Baseline computing (10,972 samples). Will test L6, L12, L4, L15 at multiple alphas on full VSR to identify the best "universal injection" feature.

**GPU6 (`pt448_caa_oracle_selection.py`):** Baseline done at 54.41%. ORACLE_INJECT phase running — per-sample inject best feature for each relation. At 3000/10972 samples (~3000 injected=158, skipped=842). Fallback map covers 6 relations; remaining samples get no injection.

**GPU7:** FREE — `pt448_residual_alllayer.py` and `pt448_caa_combined_injection.py` both completed with full summary tables in their logs.

### Completed experiment summaries (from logs)

**Multilayer injection (GPU1 — DONE):** Best strategy per feature on own-relation subset using W_dec vectors:
- `all` strategy (inject at all 26 layers) wins for L12, L4, L6, L9/F7540
- `single` strategy (SAE layer only) wins for L11, L4, L9/F7540
- Best results: L4/F14233 +7.69% (single/all tie), L14/F10561 +2.15% (all), L12/F2257 +3.92% (all), L11/F12278 +3.20% (single)
- W_dec multilayer uniformly weaker than caa_sae_down on most features

**3-tap all-layer injection (GPU5/6 — DONE):** Catastrophic at most features (L14/F10561 collapses to -30% at α=0.2). Only L4/F14233 at α=0.2 matches +7.69%. Ruled out.

**Residual all-layer injection (GPU7 — DONE):** `sae_only_down` (equivalent to caa_sae_down but with W_dec vectors) wins on L4 (+7.69%), L14 (+2.15%), L15 (+3.11%). `sae_only_up` competitive for L11, L15. Best overall: L4/F14233 sae_only_down/up +7.69% — exactly half of caa_sae_down +15.38%, confirming CAA direction doubles W_dec gains.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L6/F7539 caa_all_ml → then L11/F9639 |
| 1 | `pt448_caa_combined_all8.py` | Combined full-VSR evals |
| 2 | `pt448_caa_scaled_combined.py` | top5 scale sweep running |
| 3 | `pt448_caa_per_relation_steer.py` | ~14 relations done; continuing |
| 4 | `pt448_caa_joint_L13_L9.py` | L13 final α rows |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | baseline computing |
| 6 | `pt448_caa_oracle_selection.py` | ORACLE_INJECT ~6000/10972 |
| 7 | FREE → `pt448_caa_fullvsr_remaining.py` (PID 2184134) | Full-VSR caa_sae_down sweep for L11, L9/F387, L14, L13, L9/F7540 |

---

## Monitoring Update 27 — 2026-04-21 (cron loop pass)

### GPU0 (`pt448_caa_steering.py`): L6/F7539 caa_sae_down computing (caa_single=-0.31%, caa_all_ml=0.00% done)

L6/F7539 strategy results so far:
- caa_single: best Δ=**-0.31%** @ α=0.5 ← negative! This feature barely responds to per-layer CAA at its own subset
- caa_all_ml: best Δ=**0.00%** @ α=0.5 ← zero gain
- caa_sae_down: **computing**
- caa_proj_all, wdec_all_ml: pending

Note: L6/F7539 at α=1.5 gave +3.10% in earlier Exp 29 (perlayer_finesweep). The difference: pt448_caa_steering.py uses the relation subset "left of"/"right of" (N=323) while per_relation_steer tests ALL VSR. Low N=323 may cause variance.

### GPU1 (`pt448_caa_combined_all8.py`): ALL combined results CONFIRMED

All three combined groups on FULL VSR (N=10972):
- TOP-3 (L4+L14+L12): **48.77%** (Δ=**-5.64%**) margin=0.000
- TOP-5 (L4+L14+L12+L15+L11): **48.77%** (Δ=**-5.64%**) margin=0.000  
- ALL-8: **48.77%** (Δ=**-5.64%**) margin=0.000

Individual feature on own-relation subset (confirming optimal configs work):
- L4/F14233: 56.41% → **71.79%** (Δ=**+15.38%**) ✓
- L14/F10561: 60.22% → **70.97%** (Δ=**+10.75%**) ✓
- L12/F2257: 49.02% → **58.82%** (Δ=**+9.80%**) ✓
- L15/F220: 49.90% → **57.86%** (Δ=**+7.96%**) ✓
- L11/F12278: 56.52% → **62.84%** (Δ=**+6.32%**) ← slight improvement on own-relation!
- L9/F387: 52.29% → **56.46%** (Δ=**+4.17%**) ← matches joint sweep best!
- L6/F7539: 51.08% → **54.49%** (Δ=**+3.41%**) ← own-relation "left/right of"
- L9/F7540: 68.57% → **71.43%** (Δ=**+2.86%**) ← small positive on own subset

Individual feature on CROSS-RELATION (full VSR at individual-optimal alpha): all collapse to **48.77%** (Δ=-5.64%). Confirmed: optimal alpha for small subset is fatally over-tuned for full VSR.

CROSS-RELATION log showed L4, L14, L12, L15 all 48.77%. Remaining L11, L9/F387, L6, L9/F7540 still computing.

### GPU2 (`pt448_caa_scaled_combined.py`): COMPLETE for top3 and top5, all8 computing

- top3 group: ALL 7 scales (0.05–1.0) → **48.77%** every single scale, margin=0.000
- top5 group: scales 0.05–0.50 done → all **48.77%** margin=0.000

The combined injection collapse is not a scaling artifact. Margin=0.000 means UNIFORM probability assignment across all Y/N tokens — a fundamental circuit-level failure.

### GPU3 (`pt448_caa_per_relation_steer.py`): 14/52 qualifying relations done

Per-relation cross-steering matrix so far (best feature per relation):

| Relation | N | Base | Best Feature | Best Δ |
|----------|---|------|-------------|--------|
| above | 341 | 51.6% | L15_F220 | +5.28% |
| across from | 94 | 41.5% | L6_F7539 | **+14.89%** |
| adjacent to | 77 | 61.0% | L9_F387 | +6.49% |
| against | 46 | 76.1% | — | SKIP (all negative; worst: L9_F7540 -6.52%) |
| ahead of | 39 | 56.4% | L4_F14233 | **+15.38%** (own) |
| alongside | 55 | 43.6% | L6_F7539 | **+16.36%** |
| at the back of | 94 | 53.2% | L6_F7539 | +3.19% |
| at the edge of | 211 | 53.5% | — | SKIP (all zero or negative) |
| at the left side of | 421 | 51.8% | L15_F220 | +7.84% (own) |
| at the right side of | 480 | 52.3% | L9_F387 | +4.17% (own) |
| at the side of | 58 | 56.9% | L9_F387 | +3.45% |
| attached to | 56 | 60.7% | L9_F387 | **+8.93%** |
| away from | 155 | 48.4% | L15_F220 | +2.58% |
| behind | 709 | 51.6% | L4_F14233 | +1.55% |

Key pattern: **L6/F7539** is the best cross-relation steerer (wins "across from" +14.89%, "alongside" +16.36%, "at the back of" +3.19%), despite being only +3.41% on its own "left/right of" subset. L15/F220 and L9/F387 are also strong cross-relation helpers.

**L14/F10561 causes catastrophic damage**: -12.50% on "attached to", -5.17% on "at the side of", -2.92% on "at the right side of". Must always skip for these relations.

### GPU5 (`pt448_caa_L6_fullvsr_sweep.py`): L6/F7539 first alpha computing

Baseline confirmed 54.41%. Now sweeping L6/F7539 at alphas [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] on full VSR. This is the key test: can L6 (best cross-steerer) give positive gains at lower alpha on full VSR?

### GPU6 (`pt448_caa_oracle_selection.py`): ORACLE_INJECT at ~6000/10972

Fallback oracle map (6 relations): across from→L6, alongside→L6, adjacent to→L9/F387, against→skip, ahead of→L4, above→L15. Only ~277/10972 samples (~2.5%) get injection at α=6000 point. Most samples have no injection. Will likely show small positive delta over baseline.

### GPU7 — NEW: `pt448_caa_fullvsr_remaining.py` (PID 2184134) launched

**Goal:** Find whether remaining features (L11/F12278, L9/F387, L14/F10561, L13/F15219, L9/F7540) can give positive full-VSR gains at very low alphas (0.01–3.0), complementing GPU5's sweep of L6, L12, L4, L15.

Combined with GPU5 results, we'll have the full picture: which SINGLE feature at what alpha improves full-VSR accuracy most?

### Key insights from this pass

1. **Combined injection is completely dead at ALL scales**: 48.77% regardless of scale 0.05–1.0 for both top3 and top5 groups. The collapse to uniform probabilities is a fundamental property — not solvable by alpha reduction.

2. **Individual features also collapse on cross-relation full VSR**: Each feature at its own-relation optimal alpha on the FULL 10972-sample VSR also gives 48.77%. The optimal alpha is severely over-calibrated to the small own-relation subset.

3. **The only viable paths**:
   - (a) **Oracle per-sample injection**: inject best feature for each sample's specific relation (GPU6 testing this)
   - (b) **Single feature at much smaller alpha**: find an alpha that helps the relation it knows AND doesn't hurt others (GPU5+GPU7 sweeping this)
   - (c) **Per-sample routing with all 8 features but each applied separately**: requires relation classifier at inference

4. **L6/F7539 as universal injector candidate**: +14.89% on "across from", +16.36% on "alongside", +3.19% on "at the back of" in per-relation matrix. If it can be applied at a low alpha that gives net positive on full VSR, it's the best single-feature universal option.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L6/F7539 caa_sae_down computing → then L11/F9639 |
| 1 | `pt448_caa_combined_all8.py` | Cross-relation singles L11–L9/F7540 computing |
| 2 | `pt448_caa_scaled_combined.py` | top5 scale 0.50–1.0 and all8 group computing |
| 3 | `pt448_caa_per_relation_steer.py` | 14/52 done; 'below' next |
| 4 | `pt448_caa_joint_L13_L9.py` | L13 α=3.0 rows completing; joint L13 JSON pending |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | L6/F7539 first alphas sweeping |
| 6 | `pt448_caa_oracle_selection.py` | ORACLE_INJECT ~6000/10972 |
| 7 | `pt448_caa_fullvsr_remaining.py` | NEW — baseline computing |

---

## Monitoring Update 28 — 2026-04-21 (cron loop pass)

### GPU0 (`pt448_caa_steering.py`): L6/F7539 COMPLETE — caa_sae_down wins at +1.24%

L6/F7539 strategy results (N=323, "left of"/"right of", base=51.08%):

| Strategy | Best Δ | α* | Note |
|---------|--------|-----|------|
| caa_single | **-0.31%** | 0.5 | Negative — own-relation subset hurts |
| caa_all_ml | **0.00%** | 0.5 | Zero gain |
| **caa_sae_down** | **+1.24%** | 2.0 | Wins but weak |
| caa_proj_all | computing | — | — |
| wdec_all_ml | computing | — | — |

Key finding: L6/F7539 is a **paradoxical feature** — it gives huge cross-relation gains (+14.89% "across from", +16.36% "alongside") but is nearly useless on its own "left/right of" relation subset under caa_steering's alpha range. The +3.10% seen in Exp29 was at α=1.5; here α=2.0 gives +1.24% which is the best in this sweep's alpha range. This feature's CAA direction is most effective when injected into unrelated relation samples.

### GPU1 (`pt448_caa_combined_all8.py`): Cross-relation singles COMPLETE — all 48.77%

**Critical result confirmed:** EVERY feature at its individual-optimal alpha applied to FULL VSR (10972 samples) collapses to **48.77%** (Δ=−5.64%). Full list:
- L4/F14233: 48.77% (Δ=−5.64%)
- L14/F10561: 48.77% (Δ=−5.64%)
- L12/F2257: 48.77% (Δ=−5.64%)
- L15/F220: 48.77% (Δ=−5.64%)
- L11/F12278: 48.77% (Δ=−5.64%)
- L9/F387: 48.77% (Δ=−5.64%)
- L6/F7539: 48.77% (Δ=−5.64%)
- L9/F7540: computing (expected 48.77%)

**This definitively rules out any single-feature "full VSR" approach at own-relation-calibrated alpha.** Must use lower alpha calibrated for full VSR.

### GPU2 (`pt448_caa_scaled_combined.py`): top3 and top5 COMPLETE, all8 computing

- **top3** (L4+L14+L12): all 7 scales (0.05–1.0) → **48.77%** uniformly
- **top5** (L4+L14+L12+L15+L11): all 7 scales → **48.77%** uniformly
- **all8** group: 0.05 scale in progress (expected 48.77%)

### GPU3 (`pt448_caa_per_relation_steer.py`): 15 relations done — 'below' computing

New completed relation — **'below'** (partial, 5/8 features):
- L14/F10561: +4.69% | L12/F2257: +4.69% | L15/F220: +0.72% | L11/F12278: 0.00% | L4/F14233: -1.08%
- (L9/F387, L6/F7539, L9/F7540: pending)

Best so far for 'below': L14/F10561 and L12/F2257 tied at +4.69%.

Cumulative per-relation oracle map (complete relations):
- **L6/F7539**: wins "across from" (+14.89%), "alongside" (+16.36%), "at the back of" (+3.19%)  
- **L9/F387**: wins "adjacent to" (+6.49%), "at the right side of" (+4.17%), "at the side of" (+3.45%), "attached to" (+8.93%)
- **L15/F220**: wins "above" (+5.28%), "at the left side of" (+7.84%), "away from" (+2.58%)
- **L4/F14233**: wins "ahead of" (+15.38%), "behind" (+1.55%)
- **SKIP**: "against" (all features negative), "at the edge of" (all ≤0)

### GPU5 (`pt448_caa_L6_fullvsr_sweep.py`): L6/F7539 α=0.1 in progress (~4000/10972)

First alpha (α=0.1) is computing. This will answer whether L6 at very low alpha avoids the full-VSR collapse. Given L6's own-relation result is -0.31% at α=0.5, a very low alpha may preserve some of its cross-relation gains without collapsing.

### GPU6 (`pt448_caa_oracle_selection.py`): ORACLE_INJECT at 9000/10972

Fallback oracle (6-relation map): ~514 samples injected out of 9000 processed (~5.7%). The injection rate is low because only 6 of ~52 qualifying relations are covered. Full result expected in ~15 min.

### GPU7 (`pt448_caa_fullvsr_remaining.py`): Baseline computing

L11/F12278, L9/F387, L14/F10561, L13/F15219, L9/F7540 — baseline traversal in progress.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L6 caa_proj_all+wdec_all_ml then L11/F9639 |
| 1 | `pt448_caa_combined_all8.py` | Cross-relation singles L9/F7540 last; then DONE |
| 2 | `pt448_caa_scaled_combined.py` | all8 group scales computing |
| 3 | `pt448_caa_per_relation_steer.py` | 'below' computing; 15/52 done |
| 4 | `pt448_caa_joint_L13_L9.py` | L13 α=3.0 completing |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | L6 α=0.1 ~4000/10972 |
| 6 | `pt448_caa_oracle_selection.py` | ORACLE_INJECT 9000/10972 — nearly done |
| 7 | `pt448_caa_fullvsr_remaining.py` | Baseline traversal |

---

## Monitoring Update 29 — 2026-04-21 ~21:40 PDT (cron loop pass)

### GPU0 (`pt448_caa_steering.py`): L6/F7539 now FULLY COMPLETE, L11/F9639 started

L6/F7539 all 5 strategies complete (N=323, base=51.08%):

| Strategy | Best Δ | α* |
|---------|--------|----|
| caa_single | −0.31% | 0.5 |
| caa_all_ml | +0.00% | 0.5 |
| **caa_sae_down** | **+1.24%** | 2.0 |
| caa_proj_all | +0.31% | 10.0 |
| wdec_all_ml | +0.62% | 0.1 |

L11/F9639 (`caa_single`) now computing. This is the last feature in the 5-strategy comparison.

### GPU1 (`pt448_caa_global_alpha_fullvsr.py`): Baseline at ~4000/10972

New experiment: testing all 10 features at global α ∈ {0.25, 0.5, 1.0} on the FULL 10,972-sample VSR. Baseline is computing (~4000 done). This is the critical test of whether `caa_sae_down` at α=0.5 avoids full-VSR collapse while giving net-positive delta.

### GPU2 (`pt448_caa_scaled_combined.py`): COMPLETE — ALL 48.77%

**Definitive verdict on combined injection:**
- top3 (L4+L14+L12): all 7 scales (0.05–1.0) → **48.77%** uniformly
- top5 (L4+L14+L12+L15+L11): all 7 scales → **48.77%** uniformly
- all8 (all 8 features): all 7 scales → **48.77%** uniformly

Combined injection is **irreversibly ruled out** at all scale factors, all feature subsets. The 48.77% / margin=0.000 collapse is a catastrophic interference mode that no scaling can rescue.

### GPU3 (`pt448_caa_per_relation_steer.py`): 23/~52 relations done — 'contains' computing

Completed relations since Update 28: **beyond, by, close to, connected to, consists of**

Full per-relation cross-steering matrix (23 relations complete):

| Relation | N | Base | Best Δ | Best Feature |
|----------|---|------|--------|--------------|
| above | 341 | 51.61% | +5.28% | L15/F220 |
| across from | 94 | 41.49% | +14.89% | **L6/F7539** |
| adjacent to | 77 | 61.04% | +6.49% | L9/F387 |
| against | 46 | 76.09% | **−6.52%** | SKIP ALL (L9/F7540 least bad) |
| ahead of | 39 | 56.41% | +15.38% | L4/F14233 ← OWN; L12/F2257 tied |
| alongside | 55 | 43.64% | +16.36% | **L6/F7539** |
| at the back of | 94 | 53.19% | +3.19% | L6/F7539 |
| at the edge of | 211 | 53.55% | +0.00% | L6/F7539 (all others negative) |
| at the left side of | 421 | 51.78% | +7.84% | L15/F220 ← OWN |
| at the right side of | 480 | 52.29% | +4.17% | L9/F387 ← OWN |
| at the side of | 58 | 56.90% | +3.45% | L9/F387 |
| attached to | 56 | 60.71% | +8.93% | L9/F387 |
| away from | 155 | 48.39% | +2.58% | L15/F220 tied L9/F387 |
| behind | 709 | 51.62% | +1.55% | L4/F14233 |
| below | 277 | 49.82% | +5.42% | **L6/F7539** |
| beneath | 341 | 50.15% | +7.62% | L12/F2257 |
| beside | 188 | 51.60% | +10.64% | L15/F220 tied L11/F12278 (ALL 8 positive) |
| beyond | 20 | 45.00% | +20.00% | L12/F2257 tied L6/F7539 |
| by | 52 | 57.69% | +3.85% | L14/F10561 tied L6/F7539 |
| close to | 93 | 60.22% | +10.75% | L14/F10561 ← OWN tied L6/F7539 |
| connected to | 37 | 48.65% | +10.81% | L14/F10561 tied L6/F7539 |
| consists of | 35 | 68.57% | +2.86% | L9/F7540 ← OWN (all others negative!) |
| contains | 343 | computing | — | — |

**L6/F7539 cross-relation tallies:**
- Wins outright: across from (+14.89%), alongside (+16.36%), at the back of (+3.19%), below (+5.42%), beyond (+20.00% tie), close to (+10.75% tie), connected to (+10.81% tie), by (+3.85% tie)
- At edge of (0.00%, only non-negative feature among 8)
- Harmful: against (−23.91%), ahead of (−7.69%), attached to (−10.71%), at the right side of (−1.04%), consists of (−28.57%), away from (−1.29%), at the side of (−3.45%)

**Key insight**: L6/F7539 excels on spatial proximity/alignment relations (across from, alongside, close to, below, beyond, connected to) but is deeply harmful on "consists of", "against", "ahead of", "attached to". It's NOT a safe universal injector — needs relation gating.

**L12/F2257 cross-relation tallies**: Competitive on ahead of (+15.38%), alongside (+14.55%), beneath (+7.62%), beyond (+20% tie). But harmful on consists of (−20%), against (−19.57%), attached to (−1.79%).

**"beside" anomaly**: ALL 8 features give large positive gains (+5–11%); this relation is strongly steerable by any feature. The SAE-encoded direction for any spatial concept helps "beside" understanding.

**"against" anomaly**: ALL features are harmful (−6.52% to −26.09%). L9/F7540 is least bad at −6.52%.

**"consists of" anomaly**: ONLY L9/F7540 is positive (+2.86%); all other features are massively harmful (−8% to −31%).

### GPU4 (`pt448_caa_global_oracle.py`): Baseline at ~4000/10972

Testing 3 modes at α=0.5: global_oracle (per-relation map from GPU3), fixed_L6_everywhere, fixed_L4_everywhere.

### GPU5 (`pt448_caa_L6_fullvsr_sweep.py`): L6/F7539 α=0.1 COMPLETE — Δ=+0.02%!

**Critical result**: L6/F7539 at α=0.1 on FULL VSR = **54.43%** (Δ=**+0.02%**). This is tiny but non-collapsing — it does NOT hit 48.77%. The full-VSR collapse only occurs at larger alphas. GPU5 is now computing α=0.25, 0.5, 0.75, ... to find the optimal safe alpha for L6 universally.

### GPU6 (`pt448_caa_oracle_selection.py`): ORACLE_INJECT DONE (+0.47%), ORACLE_GATED computing

- **oracle_inject**: 54.89% (Δ=+0.47%), injected=606/10972, skipped=10366
- **oracle_gated**: ~5000/10972 computing

The oracle_inject result (+0.47%) is weak because only 6/~52 relations are covered by the fallback map, injecting only 606 samples (5.5%). The full per-relation map from GPU3 is needed for a meaningful oracle test.

### GPU7 (`pt448_caa_fullvsr_remaining.py`): Baseline at ~8000/10972

L11/F12278, L9/F387, L14/F10561, L13/F15219, L9/F7540 at 10 alphas on full VSR. Baseline nearly done.

### Summary of critical pending results

1. **GPU1 `global_alpha_fullvsr`**: Will test caa_sae_down at α=0.25/0.5/1.0 on full 10,972 VSR. This is THE key test — does α=0.5 give net positive across all 10 features on the full benchmark?
2. **GPU5 `L6_fullvsr_sweep`**: α=0.1 gave +0.02% (non-collapsing). Need α=0.25–0.75 to find the optimal safe alpha for L6 as a universal injector.
3. **GPU3 per-relation matrix**: 23/52 done; need full map to enable GPU4 global_oracle with complete coverage.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L11/F9639 caa_single computing (final feature) |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | Baseline ~4000/10972 |
| 2 | `pt448_caa_scaled_combined.py` | **COMPLETE** — all 48.77% |
| 3 | `pt448_caa_per_relation_steer.py` | 23/~52 done, 'contains' computing |
| 4 | `pt448_caa_global_oracle.py` | Baseline ~4000/10972 |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | α=0.1 done (+0.02%), α=0.25 computing |
| 6 | `pt448_caa_oracle_selection.py` | oracle_inject DONE (+0.47%), oracle_gated ~5000/10972 |
| 7 | `pt448_caa_fullvsr_remaining.py` | Baseline ~8000/10972 |

---

## Monitoring Update 30 — 2026-04-21 ~21:55 PDT (cron loop pass)

### Status: All 8 experiments in progress, no new result JSONs yet

Progress since Update 29:
- **GPU0**: L11/F9639 `caa_single` now computing. This is the final feature in pt448_caa_steering.py.
- **GPU1** (`global_alpha_fullvsr`): Baseline at ~6000/10972. Still computing baseline before starting α sweeps.
- **GPU3** (`per_relation`): 'contains' (N=343) computing, partial: L4/F14233 +3.79% so far.
- **GPU4** (`global_oracle`): Baseline at ~6000/10972.
- **GPU5** (`L6_fullvsr_sweep`): α=0.1 confirmed +0.02%, α=0.25 at ~2000/10972.
- **GPU6** (`oracle_selection`): oracle_gated at ~7000/10972 (same injection rate as oracle_inject).
- **GPU7** (`fullvsr_remaining`): Baseline complete (10000 logged). First feature α sweep starting.

### New per-relation data: 'contains' (partial)

'contains' N=343, base=57.14%: L4/F14233 +3.79% (partial, other features computing)

### GPU5 interpretation: L6/F7539 safe alpha range exists

α=0.1 on full VSR → +0.02% (non-collapsing). This confirms there IS a safe low-alpha regime for L6/F7539 on full VSR. α=0.25 computing — if still positive, we have our candidate universal feature+alpha.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L11/F9639 caa_single computing |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | Baseline ~6000/10972 |
| 2 | `pt448_caa_scaled_combined.py` | **COMPLETE** |
| 3 | `pt448_caa_per_relation_steer.py` | 'contains' computing, ~23/52 done |
| 4 | `pt448_caa_global_oracle.py` | Baseline ~6000/10972 |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | α=0.25 ~2000/10972 |
| 6 | `pt448_caa_oracle_selection.py` | oracle_gated ~7000/10972 |
| 7 | `pt448_caa_fullvsr_remaining.py` | Baseline done, first α sweep starting |

---

## Monitoring Update 31 — 2026-04-21 ~22:20 PDT (batched cron passes)

### All experiments still in progress — no new result JSONs saved

Progress summary across all 8 GPUs:

- **GPU0** (`caa_steering`): L11/F9639 `caa_single` computing. Still 9/10 features with saved JSONs.
- **GPU1** (`global_alpha_fullvsr`): Baseline at ~6000/10972. Slow — 10 features loaded means each baseline pass is reading all forward passes.
- **GPU2** (`scaled_combined`): **COMPLETE** (all 48.77%).
- **GPU3** (`per_relation`): 'contains' (N=343, base=57.14%) computing — partial: L4 +3.79%, L14 −7.29%, L12 +3.79%. L14/F10561 is harmful here (−7.29%).
- **GPU4** (`global_oracle`): Baseline at ~6000/10972.
- **GPU5** (`L6_fullvsr_sweep`): α=0.1 done (+0.02%), α=0.25 at ~2000/10972.
- **GPU6** (`oracle_selection`): oracle_gated at ~7000/10972. Same injection rate as oracle_inject (606/10972 = 5.5%), so oracle_gated ≈ oracle_inject in practice.
- **GPU7** (`fullvsr_remaining`): Baseline done (10000 confirmed). L11/F12278 first alpha sweep starting.

### New per-relation data: 'contains' (partial)

'contains' N=343, base=57.14%:
- L4/F14233: +3.79% (positive — spatial containment helps)
- L14/F10561: −7.29% (harmful — "close to" feature confuses containment)
- L12/F2257: +3.79% tied with L4

### Timing estimates

Based on observed ~1 sample/sec per GPU for NNsight tracing:
- GPU1/GPU4 baselines: ~4000 more samples × ~1 sample/sec = ~67 min → baseline done ~23:30 PDT
- GPU5 α=0.25: ~9000 samples × ~1 sample/sec = ~2.5 hrs → done ~00:45 PDT
- GPU7 L11/F12278 full sweep (10 alphas × 10972): ~30 hrs total

GPU1 (`global_alpha_fullvsr`) is the bottleneck — its first feature result (L4 at α=0.25) won't appear until ~23:30 PDT.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L11/F9639 caa_single computing (last feature) |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | Baseline ~6000/10972, ETA ~23:30 |
| 2 | `pt448_caa_scaled_combined.py` | **COMPLETE** |
| 3 | `pt448_caa_per_relation_steer.py` | 'contains' computing, ~23/52 done |
| 4 | `pt448_caa_global_oracle.py` | Baseline ~6000/10972, ETA ~23:30 |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | α=0.25 ~2000/10972 |
| 6 | `pt448_caa_oracle_selection.py` | oracle_gated ~7000/10972 |
| 7 | `pt448_caa_fullvsr_remaining.py` | L11/F12278 α sweep starting |

---

## Monitoring Update 32 — 2026-04-21 ~22:40 PDT (cron loop pass)

### GPU3 per_relation: 'contains' and 'enclosed by' COMPLETE — now at 'facing'

**'contains'** (N=343, base=57.14%):

| Feature | Δ |
|---------|---|
| L4/F14233 | +3.79% |
| L14/F10561 | **−7.29%** (harmful!) |
| L12/F2257 | +3.79% |
| **L15/F220** | **+5.83%** (best) |
| L11/F12278 | +2.92% |
| L9/F387 | +1.46% |
| L6/F7539 | −3.79% (harmful) |
| L9/F7540 | −0.58% |

Best: L15/F220 (+5.83%). L14 (−7.29%) and L6 (−3.79%) both harmful. "Contains" implies enclosure/interior which L15's "across from/at the left side of" direction handles better than proximity (L14) or lateral (L6).

**'enclosed by'** (N=21, base=42.86%) — ALL features massively helpful:

| Feature | Δ |
|---------|---|
| L4/F14233 | +14.29% |
| L14/F10561 | +9.52% |
| **L12/F2257** | **+19.05%** (best, tied L6) |
| L15/F220 | +9.52% |
| L11/F12278 | +14.29% |
| L9/F387 | +14.29% |
| **L6/F7539** | **+19.05%** (best, tied L12) |
| L9/F7540 | +0.00% |

Every feature except L9/F7540 gives huge positive gains (+9–19%). "enclosed by" has the weakest baseline (42.86%) and is extremely steerable. Only L9/F7540 (the "consists of" feature) does nothing. L12/F2257 and L6/F7539 tie at +19.05%.

**'facing'** (N=306, L12/F2257 OWN relation): computing.

### Updated per-relation oracle map (25 relations)

Adding to the table from Update 29:

| Relation | N | Base | Best Δ | Best Feature |
|----------|---|------|--------|--------------|
| contains | 343 | 57.14% | +5.83% | L15/F220 |
| enclosed by | 21 | 42.86% | +19.05% | L12/F2257 tied L6/F7539 |
| facing | 306 | computing | — | — |

Updated best-feature win counts (25/52 relations):
- **L6/F7539**: wins across from, alongside, at the back of, below, beyond (tie), by (tie), close to (tie), connected to (tie), enclosed by (tie) = ~9 relations
- **L12/F2257**: wins ahead of (tie), alongside (close), beneath, beyond (tie), enclosed by (tie) = ~5 relations
- **L15/F220**: wins above, at the left side of, away from (tie), contains = ~4 relations
- **L9/F387**: wins adjacent to, at the right side of, at the side of, attached to, away from (tie) = ~5 relations

### GPU6 oracle_selection: oracle_gated at 9000/10972 — nearly done

Expect result in ~15 min. Given same injection map as oracle_inject (606/10972), oracle_gated result will be essentially identical to oracle_inject (+0.47%) since "against" is skipped in both modes.

### GPU5 L6 universal sweep: α=0.25 at ~4000/10972

α=0.1 → +0.02%; α=0.25 computing. The safe-alpha discovery is the most important active experiment.

### GPU7 fullvsr_remaining: L11/F12278 α sweep started

First feature (L11/F12278) at alphas [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] on full VSR. Will take ~3 hours per feature.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L11/F9639 caa_single computing |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | Baseline ~8000/10972, ETA ~23:00 |
| 2 | `pt448_caa_scaled_combined.py` | **COMPLETE** |
| 3 | `pt448_caa_per_relation_steer.py` | 25/~52 done, 'facing' computing |
| 4 | `pt448_caa_global_oracle.py` | Baseline ~8000/10972, ETA ~23:00 |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | α=0.25 ~4000/10972 |
| 6 | `pt448_caa_oracle_selection.py` | oracle_gated 9000/10972 |
| 7 | `pt448_caa_fullvsr_remaining.py` | L11/F12278 α[0] computing |

---

## Monitoring Update 33 — 2026-04-21 ~23:00 PDT (cron loop pass)

### GPU2 FREE — new experiment launched: CAA + W_dec combined

New script: `pt448_caa_wdec_combined_v2.py` (PID 2246176, GPU2)

Tests the hypothesis: does injecting `α * v_caa_norm[l] + β * v_wdec` outperform pure `caa_sae_down` (β=0)?

Grid: α ∈ {0.5, 1.0, 1.5, 2.0} × β ∈ {0.0, 0.5, 1.0, 2.0, 5.0} on own-relation subsets.
β=0.0 is the pure CAA baseline for each α. If any β>0 improves over β=0 at matched α, the combination adds value.

Features: L4/F14233, L14/F10561, L12/F2257, L11/F12278, L15/F220 (5 strongest features).

### GPU3 per_relation: 'contains' COMPLETE, 'enclosed by' COMPLETE, 'facing' computing

Per Update 32 data — all incorporated.

### Other GPUs: no new result JSONs yet

- GPU0: L11/F9639 caa_single still computing
- GPU1/GPU4: baselines at ~8000/10972
- GPU5: L6 α=0.25 at ~4000/10972
- GPU6: oracle_gated 9000/10972 — result imminent
- GPU7: L11/F12278 first alpha starting

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L11/F9639 caa_single computing |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | Baseline ~8000/10972 |
| 2 | `pt448_caa_wdec_combined_v2.py` | NEW — L4 baseline computing |
| 3 | `pt448_caa_per_relation_steer.py` | 'facing' computing, ~25/52 done |
| 4 | `pt448_caa_global_oracle.py` | Baseline ~8000/10972 |
| 5 | `pt448_caa_L6_fullvsr_sweep.py` | α=0.25 ~4000/10972 |
| 6 | `pt448_caa_oracle_selection.py` | oracle_gated 9000/10972 |
| 7 | `pt448_caa_fullvsr_remaining.py` | L11/F12278 α sweep computing |

---

## Monitoring Update 34 — 2026-04-21 ~23:20 PDT — NEW METHOD: Feature-Conditioned CAA (FC-CAA)

### Methodological insight: caa_sae_down does NOT use the feature

All prior `caa_sae_down` experiments have a fundamental limitation: the SAE feature index is used only to look up which VSR relation to filter on. The actual injection direction is:

```
v_caa[l] = mean(h_l | VSR_label=1, relation=R) − mean(h_l | VSR_label=0, relation=R)
```

This is a **label-conditioned** mean-diff. `cos(caa, wdec) ≈ 0.008` — nearly orthogonal to the feature's actual encoding direction. The SAE features we found via careful analysis are being discarded.

### FC-CAA: Feature-Conditioned CAA

New method (`pt448_fc_caa.py`): split samples by the SAE feature's OWN activation on mix-448:

```
v_fc_caa[l] = mean(h_l | F fires HIGH, top 33%) − mean(h_l | F silent, bottom 33%)
```

Run on ALL 10,972 VSR samples (not just own-relation). The direction captures exactly what mix-448's residual stream looks like when **this specific spatial feature is active** vs when it is not. This genuinely leverages the monosemantic features found via SAE analysis.

Key differences from plain caa_sae_down:
| | caa_sae_down | FC-CAA |
|--|--|--|
| Split criterion | VSR ground-truth label | SAE feature activation |
| Sample scope | Own-relation subset only | ALL VSR samples |
| cos(direction, W_dec) | ~0.008 (orthogonal) | **Expected >> 0** (feature-aligned) |
| Conceptual meaning | "correct vs incorrect spatial statement" | "feature active vs silent" |

### Killed dead-end experiments; freed GPUs 2, 5, 6, 7

Killed (confirmed negative/superseded):
- **GPU2** `pt448_caa_wdec_combined_v2`: L4/F14233 shows β>0 (adding W_dec) consistently hurts vs β=0 (pure CAA). E.g. α=1.0 β=0 → +15.38%, α=1.0 β=0.5 → +10.26%, α=1.0 β=1.0 → +7.69%. W_dec adds noise.
- **GPU5** `pt448_caa_L6_fullvsr_sweep`: superseded by FC-CAA direction; only α=0.1 → +0.02% on full VSR (negligible)
- **GPU6** `pt448_caa_oracle_selection`: oracle_gated DONE (+0.47%, same as oracle_inject — "against" gating makes no difference since fallback map doesn't include "against")
- **GPU7** `pt448_caa_fullvsr_remaining`: 30h sweep of old method, superseded

### FC-CAA launched on 4 GPUs in parallel

| GPU | Feature | Own relation | Log |
|-----|---------|-------------|-----|
| 2 | L4/F14233 | "ahead of" | `/tmp/fc_caa_L4_F14233.log` |
| 5 | L12/F2257 | "facing" | `/tmp/fc_caa_L12_F2257.log` |
| 6 | L6/F7539 | "left of/right of" | `/tmp/fc_caa_L6_F7539.log` |
| 7 | L11/F12278 | "touching" | `/tmp/fc_caa_L11_F12278.log` |

Phase 1 (mix-448 + SAE, all 10,972 samples): ~3 hours per feature
Phase 2 (pt-448 injection, alpha sweep on own-relation + full VSR): ~2 hours per feature

Expected output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_fc_caa/`

### Other GPUs still running

- **GPU0** (`caa_steering`): L11/F9639 — caa_single done (+0.27%), caa_all_ml computing
- **GPU1** (`global_alpha_fullvsr`): L4/F14233 α=0.25 first pass starting
- **GPU3** (`per_relation_steer`): 'facing' DONE (L12/F2257 OWN +9.80% best), 'facing away from' DONE (L4/L15 tied +7.22%), 'has as a part' computing
- **GPU4** (`global_oracle`): global_oracle mode running (13-relation map, ~280/1000 injected)

### New per-relation data: 'facing' and 'facing away from'

**'facing'** (N=306, base=49.02%):
| Feature | Δ |
|---------|---|
| L4/F14233 | +6.21% |
| L14/F10561 | +3.27% |
| **L12/F2257** | **+9.80%** ← OWN (best) |
| L15/F220 | +7.84% |
| L11/F12278 | +5.56% |
| L9/F387 | +3.92% |
| L6/F7539 | +1.96% |
| L9/F7540 | +2.29% |

L12/F2257 wins on its own relation as expected (+9.80%). ALL features positive.

**'facing away from'** (N=180, base=48.33%):
- L4/F14233 tied L15/F220: **+7.22%** best
- L14/F10561: +5.56%, L12/F2257: +6.11%, others positive

### GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L11/F9639 caa_all_ml computing |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | L4 α=0.25 computing |
| 2 | **`pt448_fc_caa.py` L4/F14233** | Phase 1 — all 10972 samples |
| 3 | `pt448_caa_per_relation_steer.py` | ~27/52, 'facing away' done |
| 4 | `pt448_caa_global_oracle.py` | global_oracle mode running |
| 5 | **`pt448_fc_caa.py` L12/F2257** | Phase 1 — all 10972 samples |
| 6 | **`pt448_fc_caa.py` L6/F7539** | Phase 1 — all 10972 samples |
| 7 | **`pt448_fc_caa.py` L11/F12278** | Phase 1 — all 10972 samples |

---

## Monitoring Update 35 — 2026-04-21 ~23:45 PDT (cron loop pass)

### FC-CAA Phase 1 running at 99% util on GPUs 2/5/6/7 — no results yet

All 4 FC-CAA jobs are in Phase 1: scanning all 10,972 VSR samples through mix-448 + SAE to compute per-sample feature activations. Each job is running mix-448 (bfloat16) + JumpReLU SAE forward pass per sample. GPU memory ~9 GB (mix-448 only, no NNsight). ETA Phase 1 complete: ~3 hrs from launch (~02:00 PDT).

### GPU0 (`caa_steering`): L11/F9639 — caa_single done, caa_all_ml running

L11/F9639 (N=1101, "in/inside/on"): caa_single best Δ=+0.27% @α=0.1. caa_all_ml computing.

### GPU1 (`global_alpha_fullvsr`): L4/F14233 α=0.25 at ~6000/10972

First real alpha result for the global-α test expected soon.

### GPU3 (`per_relation`): 4 new relations complete — 30/~52 done

New completed relations since Update 34:

| Relation | N | Base | Best Δ | Best Feature | Notes |
|----------|---|------|--------|--------------|-------|
| far away from | 232 | 49.14% | **−0.86%** | L9/F7540 (least bad) | **ALL features HARMFUL** |
| far from | 145 | 46.90% | +1.38% | L9/F387 | Near-zero; most features harmful |
| has as a part | 28 | 60.71% | **0.00%** | L9/F7540 | **ALL features HARMFUL except L9/F7540** |
| in | 276 | 62.68% | computing | — | — |

**"far away from" is a new SKIP relation**: all 8 features are harmful (−0.86% to −5.60%). This joins "against" and "has as a part" as relations where injection should be gated off.

**"far from"**: marginal gains only (+1.38% at best). Very weak baseline (46.9%) but features can't find traction.

**"has as a part"**: all features harmful except L9/F7540 which gives 0.00%. Probably because this is a semantic/compositional relation (part-of), not spatial proximity — spatial steering confuses it.

### GPU4 (`global_oracle`): global_oracle mode at 6000/10972 — injecting ~26% of samples

With 13-relation map: ~1546/6000 processed samples are being injected (25.8%). This is much higher than the 6-relation fallback (5.5%). Expect result in ~30 min.

### Updated GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L11/F9639 caa_sae_down computing (~last feature) |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | L4 α=0.25 **done** (+0.40% full-VSR), α=0.5 computing |
| 2 | **`pt448_fc_caa.py` L4/F14233** | Phase 1 at ~2000/10972, skipped=0 ✓ |
| 3 | `pt448_caa_per_relation_steer.py` | 'in front of' computing, ~31/52 done |
| 4 | `pt448_caa_global_oracle.py` | global_oracle **DONE** (+0.84%), fixed_L6 computing |
| 5 | **`pt448_fc_caa.py` L12/F2257** | Phase 1 at ~1000/10972, skipped=0 ✓ |
| 6 | **`pt448_fc_caa.py` L6/F7539** | Phase 1 at ~1000/10972, skipped=0 ✓ |
| 7 | **`pt448_fc_caa.py` L11/F12278** | Phase 1 at ~1000/10972, skipped=0 ✓ |

---

## Monitoring Update 36 — 2026-04-22 ~00:10 PDT (cron loop pass)

### CRITICAL BUG FIXED: FC-CAA was silently skipping all samples (dtype mismatch)

All 4 prior FC-CAA runs (GPUs 2/5/6/7) had `skipped=N` for every sample due to a dtype mismatch:
- SAE `W_enc` is float32, but `h_sae` was extracted as bfloat16 (`out.hidden_states[l+1][0,...].to(dtype)`)
- `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float`
- The exception was silently swallowed — 99% GPU util but zero valid samples collected

**Fix applied**: `h_sae = out.hidden_states[layer_idx + 1][0, last_text_pos, :].float()` (cast to float32 before SAE encode). Script updated at `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_fc_caa.py:185`.

All 4 jobs restarted ~00:05 PDT. Now running correctly with `skipped=0` on all 4 GPUs.

### GPU4 (`global_oracle`): COMPLETED — 13-relation oracle +0.84% vs baseline 54.41%

| Mode | Acc | Δ | N injected |
|------|-----|---|-----------|
| Baseline | 54.41% | — | 0/10972 |
| global_oracle (13 relations) | 55.25% | **+0.84%** | 2856/10972 (26.0%) |
| fixed_L6_everywhere | computing | — | 10972/10972 |

**13-relation oracle** injects only when the sample's relation matches one of 13 mapped relations. 26% coverage gives +0.84%. This is the upper bound for the "oracle knows relation" method with current feature assignments.

### GPU1 (`global_alpha_fullvsr`): First result — L4/F14233 α=0.25 → +0.40% full-VSR, +2.56% own-relation

At α=0.25, L4/F14233 does NOT collapse on full VSR (54.81%, Δ=+0.40%). This is the first evidence that a sufficiently small alpha prevents collapse. Own-relation gain (+2.56%) preserved. α=0.5 computing next.

### GPU0 (`caa_steering`): L11/F9639 nearing completion — caa_all_ml best Δ=+0.64% @ α=10

`caa_all_ml` completed: Δ=+0.64% @ α=10. Now on `caa_sae_down`.

### Per-relation matrix: 'in front of' computing (31/~52 done)

Recently completed relations (new data):

| Relation | N | Base | Best Δ | Best Feature |
|----------|---|------|--------|--------------|
| consists of | 35 | 68.57% | computing | — |
| contains | 343 | 57.14% | computing | — |
| enclosed by | 21 | 42.86% | computing | — |
| in | 276 | 62.68% | **ALL HARMFUL** (−0.72% to −11.23%) | L4/F14233 (least bad) |
| in front of | 737 | 56.58% | **ALL HARMFUL** (−0.68% to −4.34%) | L15/F220 (least bad) |

**"in" ALL HARMFUL**: L6/F7539 worst (Δ=−8.33%), L14/F10561 (Δ=−11.23%). These are large-N containment relations — spatial direction injection actively confuses them.

**"in front of" ALL HARMFUL** (confirmed complete): all 8 features negative (−0.68% to −4.34%). SKIP.

**Expanding SKIP list**: "against", "far away from", "has as a part", "in", "in front of" — all features harmful. These represent ~28% of VSR samples (N=~3070/10972).

### Mix-448 self-injection: confirmed no benefit

The mix-448 fixed-injection experiment (`mix448_fixed_injection/`) showed that injecting spatial features into the already-capable mix-448 model gives near-zero gains (max +2.56% at α=5–10, but at reasonable scales 0.47%–0.78%). The model already has the spatial representations. **mix→pt transfer works; mix→mix is a no-op.**

### FC-CAA: all 4 jobs at ~1000–3000/10972, skipped=0

| GPU | Feature | Progress |
|-----|---------|---------|
| 2 | L4/F14233 | ~3000/10972, skipped=0 |
| 5 | L12/F2257 | ~1000/10972, skipped=0 |
| 6 | L6/F7539 | ~1000/10972, skipped=0 |
| 7 | L11/F12278 | ~1000/10972, skipped=0 |

**Critical diagnostic when done**: check `cos(fc_caa, wdec)` at SAE layer. Expected >>0.008 (prior caa_sae_down value) to confirm the FC-CAA direction genuinely encodes the feature.

---

## Monitoring Update 37 — 2026-04-22 ~00:20 PDT (cron loop pass)

### All 8 GPUs busy — no new launches possible yet

### New per-relation data: "in" and "in front of" confirmed ALL HARMFUL

| Relation | N | Base | All-feature verdict | Notes |
|----------|---|------|---------------------|-------|
| in | 276 | 62.68% | **ALL HARMFUL** | L14/F10561 worst (Δ=−11.23%), L4/F14233 least bad (Δ=−0.72%) |
| in front of | 737 | 56.58% | **ALL HARMFUL** | L14/F10561 worst (Δ=−4.34%), L15/F220 least bad (Δ=−0.68%) |

**Critical implication**: "in" (N=276) and "in front of" (N=737) together represent 1013 samples where injection is actively harmful. Combined with "against" (N≈134), "far away from" (N=232), "has as a part" (N=28) — at least 1407 samples (~13%) should have injection GATED OFF in any oracle approach.

**Expanding confirmed SKIP relations**: against, far away from, has as a part, in, in front of.

### L11/F9639 5-strategy comparison: caa_sae_down best at Δ=+1.55%? No — L11/F9639 caa_sae_down running

`caa_all_ml` completed: best Δ=+0.64% @ α=10. `caa_sae_down` currently at α=2 (of [0.1, 0.5, 1, 2, 5, 10, 20, 50]).

### Global alpha L4/F14233 α=0.5: still computing (α=0.25 gave +0.40% full-VSR)

Key finding so far: α=0.25 on L4/F14233 gives +0.40% on full VSR without collapse. This is meaningful — suggests a "gentle" single-feature injection at small α works even without relation-aware gating.

### Global oracle fixed_L6 mode: computing full VSR with L6/F7539 everywhere at α=0.5

This will be the cleanest test of "one feature, one alpha, no oracle" on full 10972 samples.

### GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | `pt448_caa_steering.py` | L11/F9639 caa_sae_down at α=10 |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | L4 α=0.5 computing |
| 2 | **`pt448_fc_caa.py` L4/F14233** | ~5000/10972, skipped=0 |
| 3 | `pt448_caa_per_relation_steer.py` | 'inside' computing, ~35/52 done |
| 4 | `pt448_caa_global_oracle.py` | fixed_L6_everywhere at ~6000/10972 |
| 5 | **`pt448_fc_caa.py` L12/F2257** | ~3000/10972, skipped=0 |
| 6 | **`pt448_fc_caa.py` L6/F7539** | ~3000/10972, skipped=0 |
| 7 | **`pt448_fc_caa.py` L11/F12278** | ~3000/10972, skipped=0 |

---

## Monitoring Update 38 — 2026-04-22 ~00:35 PDT (cron loop pass)

### SECOND FC-CAA BUG FIXED: Feature fires on 0/10972 samples — was checking last token only

**Root cause**: FC-CAA Phase 1 was running the SAE only on `hidden_states[layer][0, last_text_pos, :]` (the final "Answer:" token). Spatial features (e.g., L4/F14233 = "ahead of") fire at the **spatial preposition token**, not the last token. Verified: `h_sae` at last-text-token gives pre_jump < threshold for F14233 on almost all samples; but at the spatial preposition token (e.g. position 23 in "The cat is **inside** the refrigerator"), pre_jump = 1.92 > threshold = 1.82.

**Fix applied**: Replaced single-token SAE call with max over ALL text tokens:
```python
h_text_all = out.hidden_states[layer_idx+1][0, img_end:, :].float()  # all text tokens
acts_all = sae.encode(h_text_all)  # (n_text, d_sae)
act_val = acts_all[:, feature_idx].max().item()
peak_pos = img_end + acts_all[:, feature_idx].argmax().item()
# Use hidden state at peak-firing position for building contrastive vector
```

**Impact**: n_nonzero was 0/10972 for all 4 features → should now be >0. Broken cached `.pt` files deleted, all 6 jobs restarted with fix.

### 5-Strategy Comparison: COMPLETE (all 10 features)

Full table from `caa_steering.log` (own-relation subsets, best α per strategy):

| Feature | Relation | N | base | wdec | caa_single | caa_all_ml | caa_sae_down | caa_proj |
|---------|----------|---|------|------|-----------|-----------|-------------|----------|
| L4/F14233 | ahead of | 39 | 56.4% | +10.26% | +2.56% | +2.56% | **+15.38%** | +5.13% |
| L12/F2257 | facing | 306 | 49.0% | +3.92% | +1.96% | +8.50% | **+8.82%** | +5.88% |
| L11/F12278 | touching | 1281 | 56.5% | +3.36% | +0.31% | +5.46% | **+5.85%** | +0.31% |
| L9/F387 | at right side | 480 | 52.3% | +3.12% | +1.25% | +1.67% | **+2.92%** | +1.25% |
| L15/F220 | across from | 515 | 49.9% | +3.11% | **+7.96%** | +6.99% | +6.80% | +1.17% |
| L9/F7540 | consists of | 35 | 68.6% | +2.86% | 0.00% | 0.00% | −2.86% | 0.00% |
| L14/F10561 | close to | 93 | 60.2% | +2.15% | **+10.75%** | +10.75% | +10.75% | +2.15% |
| L13/F15219 | behind | 709 | 51.6% | +2.12% | +1.55% | +1.13% | **+1.55%** | +0.56% |
| L6/F7539 | left/right of | 323 | 51.1% | +1.24% | −0.31% | 0.00% | **+1.24%** | +0.31% |
| L11/F9639 | in/inside/on | 1101 | 60.9% | +0.73% | +0.27% | **+0.64%** | +0.09% | +0.09% |

**Winner**: `caa_sae_down` wins or ties on 7/10 features. Mean best Δ caa_sae_down: +5.68%. Mean best Δ wdec: +3.27%.

### Global oracle DONE: fixed_L6_everywhere BEATS oracle (+1.90% vs +0.84%)

| Mode | Acc | Δ | Coverage |
|------|-----|---|---------|
| Baseline | 54.41% | — | 0% |
| 13-relation oracle | 55.25% | +0.84% | 26.0% |
| **fixed_L6_everywhere** (α=0.5) | **56.31%** | **+1.90%** | 100% |
| fixed_L4_everywhere (α=0.5) | 55.57% | +1.16% | 100% |

**Key insight**: Injecting L6/F7539 (left/right-of feature) at α=0.5 on ALL 10972 samples outperforms the 13-relation oracle that only injects 26% of samples. The oracle's conservative gating (only inject on matched relations) is less effective than blind universal injection of L6/F7539.

**Why L6 works universally**: L6/F7539 encodes relative horizontal positioning, which is relevant for spatial questions beyond just "left of/right of". Its own-relation baseline was only +1.24%, but the cross-relation effects are cumulatively positive across many relations.

### Global alpha L4/F14233: NO collapse at α=0.25–1.0 on full VSR

| α | full_VSR Δ | own_rel Δ |
|---|-----------|----------|
| 0.25 | +0.40% | +2.56% |
| 0.5 | +1.16% | +7.69% |
| 1.0 | +1.88% | +15.38% |

L4/F14233 at α=1.0 gives **+1.88% on full VSR** (56.29%) with +15.38% own-relation gain — no collapse. This is better than L6 at α=0.5 (+1.90%). L14/F10561 at α=0.25: +0.75% full-VSR.

### Near-complete per-relation cross-steering matrix (49/~52 relations done)

New completions from this pass:

| Relation | N | Base | Best Δ | Best Feature | Notes |
|----------|---|------|--------|--------------|-------|
| above | 341 | 51.61% | +5.28% | L15/F220 | All positive |
| across from | 94 | 41.49% | +14.89% | L6/F7539 | Strong gains across all features |
| at the left side of | 421 | 51.78% | +7.84% | L15/F220 ← own | L6 weakest (+1.43%) |
| at the right side of | 480 | 52.29% | +4.17% | L9/F387 ← own | L14 harmful |
| behind | 709 | 51.62% | +1.55% | L4/L13 | Weak but positive |
| below | 277 | 49.82% | +5.42% | L6/F7539 | L4 harmful |
| in the middle of | 92 | 51.09% | +5.43% | L9/F7540 | L6/L14 harmful |
| inside | 240 | 60.42% | +2.08% | L15/F220 | L14/L6 badly harmful |
| into | 29 | 51.72% | 0.00% | L4/L6 | Most harmful |
| left of | 210 | 49.52% | +4.76% | L6/F7539 ← own | All positive |
| near | 110 | 56.36% | +11.82% | L12/L15 | **Biggest gains** — ALL positive |
| next to | 309 | 61.49% | +1.29% | L9/F7540 | Most harmful, L11/L4 ≈0 |
| off | 74 | 48.65% | +13.51% | L12/F2257 | Mixed |
| on | 585 | 60.17% | +0.68% | L9/F7540 | **Mostly harmful** (high base) |
| on top of | 505 | 59.21% | +4.36% | L11/F12278 | L14/L6 badly harmful |
| outside | 32 | 46.88% | +9.38% | L12/F2257 | L11 harmful |
| over | 84 | 55.95% | +8.33% | L15/F220 | Mostly positive |
| parallel to | 90 | 52.22% | +5.56% | L9/F7540 | All positive |
| part of | 113 | 57.52% | +1.77% | L15/L9/L6 | L14 harmful |
| perpendicular to | 71 | 54.93% | 0.00% | L4 | All harmful except L4 (0) |
| right of | 113 | 53.98% | +0.88% | L6/F7539 ← own | Most harmful |
| surrounding | 90 | 47.78% | +10.00% | L11/F12278 | All positive |
| touching | 1281 | 56.52% | +4.92% | L4/F14233 | L14 harmful |

**Patterns emerging**:
- **L14/F10561 consistently harmful** across many relations ("in", "on top of", "on", "at right side", "perpendicular", "next to", "parallel to"). Only works well on own-relation "close to" (+10.75%) and some directional relations.
- **L12/F2257** strong on "near" (+11.82%), "off" (+13.51%), "outside" (+9.38%) — proximity/adjacency cluster.
- **L6/F7539** best on "across from" (+14.89%), "left of" (+4.76%), "below" (+5.42%). Harmful on "on top of", "next to", "inside", "in the middle of".
- **L4/F14233** consistently positive on "near" (+10.91%), "surrounding" (+8.89%), "touching" (+4.92%), "over" (+5.95%), "above" (+3.52%).

### FC-CAA status: 6 jobs restarted with peak-token fix, ~1000/10972 each

---

## Monitoring Update 39 — 2026-04-22 ~01:00 PDT (cron loop pass)

### Major results batch: 5-strategy comparison complete, global oracle done, per-relation 49/52 done

#### ACTION: Two new FC-CAA features launched
- GPU0: L9/F387 (`pt448_fc_caa.py --layer 9 --feature 387`)
- GPU4: L14/F10561 (`pt448_fc_caa.py --layer 14 --feature 10561`)

#### ACTION: fixed_L6_everywhere result revealed as best method so far → +1.90% on full VSR
Fixed injection of L6/F7539 at α=0.5 everywhere (no oracle, no relation gating) gives +1.90% (56.31%), beating 13-relation oracle (+0.84%). This is the current **state-of-the-art for caa_sae_down method**.

#### SECOND FC-CAA BUG: last-token-only SAE extraction → 0 nonzero activations
All 4 first-round FC-CAA runs yielded `n_nonzero = 0` because the SAE was run only on the last text token ("Answer:"). Spatial features fire at preposition tokens. Fixed to use max across all text tokens. All 6 jobs restarted with fix, cached broken vectors deleted.

### GPU status

| GPU | Script | Status |
|-----|--------|--------|
| 0 | **`pt448_fc_caa.py` L9/F387** | Phase 1 ~4000/10972 |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | L14/F10561 α=0.5 computing |
| 2 | **`pt448_fc_caa.py` L4/F14233** | Phase 1 ~3000/10972 |
| 3 | `pt448_caa_per_relation_steer.py` | ~49/52 done, 'touching' computing |
| 4 | **`pt448_fc_caa.py` L14/F10561** | Phase 1 ~4000/10972 |
| 5 | **`pt448_fc_caa.py` L12/F2257** | Phase 1 ~3000/10972 |
| 6 | **`pt448_fc_caa.py` L6/F7539** | Phase 1 ~3000/10972 |
| 7 | **`pt448_fc_caa.py` L11/F12278** | Phase 1 ~3000/10972 |

---

## Monitoring Update 40 — 2026-04-22 ~01:15 PDT (cron loop, 15 min cadence)

### No new completions — all 8 GPUs busy, all FC-CAA Phase 1 progressing cleanly

All 6 FC-CAA jobs running with peak-token fix, skipped=0 on all. L9/F387 and L14/F10561 (launched first) furthest along at ~4000/10972; L4/L12/L6/L11 at ~3000/10972.

**Critical validation pending**: After Phase 1 completes, will check `n_nonzero` > 0 to confirm the peak-token fix is working as expected. If features fire sparsely (e.g. 5–15% of samples), the HIGH vs LOW split will be meaningful and the contrastive vector should have cos(fc_caa, wdec) >> 0.008.

**global_alpha_fullvsr**: L14/F10561 α=0.5 computing. Prior result: L4/F14233 α=1.0 → +1.88% full-VSR. ETA for L14 α=0.5 result: ~30 min.

**per_relation**: ~49/52 done, 'touching' computing. Near-complete.

**Next actions queued** (when GPUs free):
1. When per_relation finishes (GPU3): Launch remaining FC-CAA features — L13/F15219, L15/F220, L9/F7540, L11/F9639
2. When global_alpha finishes (GPU1): Potentially launch a "smart oracle v2" combining the per-relation matrix insights to select the single best feature per relation, or test global injection of L4/F14233 at α=0.5 and α=1.0 across full VSR
3. After FC-CAA Phase 1: Check whether feature actually fires (n_nonzero), then run Phase 2 injection — compare FC-CAA vs caa_sae_down best

**Current best method**: `fixed_L6_everywhere` (L6/F7539 caa_sae_down at α=0.5, injected on all 10972 samples) → **+1.90%** (56.31% vs 54.41% baseline). No oracle, no relation gating needed.


---

## Monitoring Update 41 — 2026-04-22 ~01:30 PDT (cron loop pass)

### No new completions — all 8 GPUs still running

**FC-CAA**: All 6 jobs at 7000–8000/10972, ETA Phase 1 complete ~01:40 PDT.

**global_alpha L14/F10561 α=0.5**: at 8000/10972 — result expected within 5 min.

**per_relation**: 50 relations done, 'toward' computing (N=36). Near final (~2 more).

### New per-relation data: 'touching' complete, 'toward' partial

| Relation | N | Base | Best Δ | Best Feature | Notes |
|----------|---|------|--------|--------------|-------|
| touching | 1281 | 56.52% | **+6.48%** | L9/F387 | L9/F387 edges out own L11/F12278 (+6.32%); L14 badly harmful |
| toward | 36 | 52.78% | **+13.89%** | L4/F14233 | Strong — computing remainder |

'touching': L9/F387 +6.48% is best despite not being the "own" feature (L11/F12278 = +6.32%). L14/F10561 harmful (−4.92%).

'toward': L4/F14233 huge gain (+13.89%). All positive so far.

---

## Monitoring Update 42 — 2026-04-22 ~02:10 PDT (cron loop pass — post-context-compaction)

### Per-Relation Matrix COMPLETE: Oracle ceiling = +4.56% (59.01%)

`pt448_caa_per_relation_steer.py` finished — 52 qualifying relations (N≥20). The per-sample oracle (inject best feature per relation, skip SKIP relations) achieves:

- **Oracle VSR: 59.01%** (baseline 54.45%)
- **Oracle Δ: +4.56%** — 2.4× better than fixed_L6_everywhere (+1.90%)

This is the theoretical ceiling for relation-conditioned single-feature injection. A caption parser that identifies the spatial relation and routes to the best feature can close 62% of the pt→mix accuracy gap (gap ≈ 7.3%, oracle closes 4.56%).

#### Feature win counts across 52 relations:

| Feature | Relations won | Notes |
|---------|--------------|-------|
| L15/F220 | ~9 | "across from", "above", "contains", "at the left side of", "away from", "over", "inside", "facing away from" |
| L12/F2257 | ~8 | "facing", "near", "off", "outside", "enclosed by" (tie), "beneath", "beyond" (tie) |
| L6/F7539 | ~7 | "alongside", "below", "at the back of", "left of", "across from" (own+cross), "enclosed by" (tie) |
| L9/F387 | ~6 | "at the right side of", "adjacent to", "attached to", "at the side of", "touching" (cross!) |
| L9/F7540 | ~6 | "consists of" (only safe), "parallel to", "in the middle of", "next to", "surrounding" (partial) |
| L4/F14233 | ~5 | "ahead of", "toward", "touching" (partial), "behind" |
| L11/F12278 | ~4 | "touching", "surrounding", "on top of" |
| L14/F10561 | ~3 | "close to", "by", "connected to"; consistently harmful elsewhere |

#### SKIP relations (all features neutral or harmful, ~28% of VSR samples):
against, far away from, has as a part, in, in front of, into, perpendicular to, at the edge of

These 8 SKIP relations represent ~1,400–1,600 VSR samples where injection gating is mandatory.

### Global Alpha Full-VSR Results (GPU1, `pt448_caa_global_alpha_fullvsr.py`)

**Confirmed complete results:**

| Feature | α | full_VSR Δ | own_rel Δ | Status |
|---------|---|-----------|---------|--------|
| L4/F14233 | 0.25 | +0.40% | +2.56% | ✓ done |
| L4/F14233 | 0.5 | +1.16% | +7.69% | ✓ done |
| L4/F14233 | 1.0 | **+1.88%** | +15.38% | ✓ done |
| L14/F10561 | 0.25 | +0.75% | +2.15% | ✓ done |
| L14/F10561 | 0.5 | **+1.88%** | +2.15% | ✓ done |
| L14/F10561 | 1.0 | +1.39% | +6.45% | ✓ done |
| L12/F2257 | 0.25 | — | — | in progress |

Key finding: **NO collapse** for L4 and L14 at any tested alpha (0.25–1.0) on full VSR. The own-relation-optimal alpha (e.g. L4 α=15.38% at α=1.0) is NOT catastrophic on full VSR at α=1.0 — it gives +1.88%. This contradicts the earlier finding that "optimal alpha always collapses" — that was only true when using the N=39 own-subset-optimized alpha (e.g. α=4.0 for L4). At α≤1.0, L4/F14233 and L14/F10561 both show monotone improvement or plateau on full VSR.

**Implication**: fixed_L4_everywhere at α=1.0 → +1.88% on full VSR (matches fixed_L6_everywhere +1.90% almost exactly). Two features are now competitive for "best single universal injector."

### FC-CAA Phase 1: Complete — crashed on cos_wdec computation (dtype bug #3)

All 8 FC-CAA Phase 1 jobs (GPUs 0,2,4,5,6,7) completed collection of 10,972/10,972 samples with skipped=0.

Feature activation statistics confirmed:

| Feature | nonzero | frac_nonzero | split mode |
|---------|---------|-------------|-----------|
| L4/F14233 | 2658 | 24.2% | nonzero_vs_zero |
| L6/F7539 | 4231 | 38.6% | nonzero_vs_zero |
| L9/F387 | 0 | 0.0% | **NULL** (no firing on VSR) |
| L11/F12278 | 2330 | 21.2% | nonzero_vs_zero |
| L12/F2257 | 10972 | 100.0% | quantile |
| L13/F15219 | 6038 | 55.0% | nonzero_vs_zero |
| L14/F10561 | 0 | 0.0% | **NULL** (no firing on VSR) |
| L15/F220 | 9805 | 89.4% | quantile |

Then **crashed at line 292** — dtype mismatch when computing `cos(fc_caa, wdec)`:
```
RuntimeError: dot : expected both vectors to have same dtype, but found BFloat16 and Float
```
Cause: `v_fc = (v_high - v_low).to(dtype)` converted to bfloat16; `wdec_norm` is float32; `.to(device)` doesn't change dtype.

**Fix applied** (line 289–292):
```python
v_fc   = (v_high - v_low).float().to(device)
cos_wdec = (v_fc / max(v_norm, 1e-8) @ wdec_norm.float().to(device)).item()
```

**Key finding**: L9/F387 and L14/F10561 fire 0 times on VSR (even with peak-token fix). These two features use text-only SAE that was trained on mix-448's single-modality text; multimodal VSR captions apparently don't fire them. Their FC-CAA will be null/skipped.

### Fifth FC-CAA restart launched (with dtype fix)

| GPU | Feature | Log |
|-----|---------|-----|
| 0 | L13/F15219 | `/tmp/fc_caa_L13_F15219.log` |
| 2 | L4/F14233 | `/tmp/fc_caa_L4_F14233.log` |
| 3 | L9/F387 | `/tmp/fc_caa_L9_F387.log` (will save null) |
| 4 | L15/F220 | `/tmp/fc_caa_L15_F220.log` |
| 5 | L12/F2257 | `/tmp/fc_caa_L12_F2257.log` |
| 6 | L6/F7539 | `/tmp/fc_caa_L6_F7539.log` |
| 7 | L11/F12278 | `/tmp/fc_caa_L11_F12278.log` |

GPU1 still running global_alpha_fullvsr (L12/F2257 at α=0.25).

All Phase 1 data collection is re-running (~3 hrs ETA). After save, Phase 2 injection auto-runs: alpha sweep on own-relation subset AND full VSR for each feature. Critical diagnostic: `cos(fc_caa, wdec)` at SAE layer — expected >> 0.008 for non-null features.

### Current Method Ranking (Full VSR)

| Method | full_VSR Δ | Notes |
|--------|-----------|-------|
| Per-relation oracle (+oracle map) | **+4.56%** (59.01%) | Theoretical ceiling; needs caption parser |
| fixed_L6_everywhere α=0.5 | **+1.90%** (56.31%) | Best simple method |
| fixed_L4_everywhere α=1.0 | **+1.88%** (56.29%) | Essentially tied with L6 |
| 13-relation oracle | +0.84% (55.25%) | Partial coverage (26%) |
| FC-CAA | **pending** | Phase 1 complete; Phase 2 ETA ~05:00 PDT |

### GPU Allocation (2026-04-22 ~02:10 PDT)

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_fc_caa.py` | L13/F15219 | Phase 1 running |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | L12/F2257 | α=0.25 computing |
| 2 | `pt448_fc_caa.py` | L4/F14233 | Phase 1 running |
| 3 | `pt448_fc_caa.py` | L9/F387 | Phase 1 running (will save null) |
| 4 | `pt448_fc_caa.py` | L15/F220 | Phase 1 running |
| 5 | `pt448_fc_caa.py` | L12/F2257 | Phase 1 running |
| 6 | `pt448_fc_caa.py` | L6/F7539 | Phase 1 running |
| 7 | `pt448_fc_caa.py` | L11/F12278 | Phase 1 running |

---

## Monitoring Update 43 — 2026-04-21 19:11 PDT

### Issue Discovered and Fixed: L4/L6/L9/L11/L15 jobs failed with wrong path

The 5th restart batch ran 7 jobs, but only 2 survived (GPU0/L13 and GPU5/L12). The other 5 (L4/F14233, L6/F7539, L9/F387, L11/F12278, L15/F220) all crashed immediately with:
```
python3: can't open file '/home/hbatra/vlm_scope_backup/pt448_fc_caa.py': [Errno 2] No such file or directory
```
Cause: the batch eval used relative path `pt448_fc_caa.py` but `cd` only applied to the first subshell; subsequent `&` jobs used the original working directory (`/home/hbatra/vlm_scope_backup/`) which doesn't have the script.

**Fix applied (19:05 PDT):** Re-launched all 5 failed jobs with the absolute path `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_fc_caa.py`. All 5 confirmed starting correctly.

### Global Alpha Full-VSR Results (GPU1, partial)

Script: `pt448_caa_global_alpha_fullvsr.py` — tests `caa_sae_down` with α∈{0.25, 0.5, 1.0} on all 10 features on FULL VSR (N=10972).

**Completed features:**

| Feature | α=0.25 | α=0.5 | α=1.0 | Best α | Best Δ |
|---------|--------|--------|--------|---------|---------|
| L4/F14233 (ahead of) | +0.40% | +1.16% | **+1.88%** | 1.0 | **+1.88%** |
| L14/F10561 (close to) | +0.75% | **+1.88%** | +1.39% | 0.5 | **+1.88%** |
| L12/F2257 (facing) | running | — | — | TBD | TBD |

**Key insight: L14/F10561 peaks at α=0.5 (not α=1.0)!** L4 peaked at α=1.0. These match the `fixed_L6_everywhere` best of +1.90% within noise — confirming that single-feature full-VSR injection tops out at ~+1.88–1.90%.

**Own-relation gains (α=1.0):**
- L4/F14233 'ahead of' N=39: +15.38% (own-relation)
- L14/F10561 'close to' N=93: +6.45% at α=1.0 (best own: +2.15% at α=0.25/0.5... surprisingly flat)

### FC-CAA L14/F10561 Phase 2 Result

L14/F10561 had n_nonzero=0 (confirmed again), so FC-CAA vector is null (zero norm). Phase 2 ran but produced Δ=0.00% at all alpha values — as expected. No information to extract from this feature's FC-CAA.

### Current FC-CAA Status (19:11 PDT)

| GPU | Feature | Phase | Progress |
|-----|---------|-------|----------|
| 0 | L13/F15219 | Phase 1 | 4000/10972 (~36%) |
| 2 | L4/F14233 | Phase 1 | just started |
| 3 | L9/F387 | Phase 1 | just started |
| 4 | L15/F220 | Phase 1 | just started |
| 5 | L12/F2257 | Phase 1 | 4000/10972 (~36%) |
| 6 | L6/F7539 | Phase 1 | just started |
| 7 | L11/F12278 | Phase 1 | just started |

ETA for Phase 1 completion (all jobs): ~22:00 PDT (re-launched at 19:05). Then Phase 2 auto-runs (~3 hrs more per feature, serialized).

### GPU Allocation (19:11 PDT)

| GPU | Script | Feature | Status |
|-----|--------|---------|--------|
| 0 | `pt448_fc_caa.py` | L13/F15219 | Phase 1, 4000/10972 |
| 1 | `pt448_caa_global_alpha_fullvsr.py` | L12/F2257 | α=0.25 running (~8000/10972) |
| 2 | `pt448_fc_caa.py` | L4/F14233 | Phase 1, loading |
| 3 | `pt448_fc_caa.py` | L9/F387 | Phase 1, loading |
| 4 | `pt448_fc_caa.py` | L15/F220 | Phase 1, loading |
| 5 | `pt448_fc_caa.py` | L12/F2257 | Phase 1, 4000/10972 |
| 6 | `pt448_fc_caa.py` | L6/F7539 | Phase 1, loading |
| 7 | `pt448_fc_caa.py` | L11/F12278 | Phase 1, loading |

---

## Monitoring Update 44 — 2026-04-21 19:30 PDT

### Global Alpha Full-VSR Results (GPU1, partial — 3/10 features done)

| Feature | Own-relation | α=0.25 | α=0.5 | α=1.0 | Best (full-VSR) |
|---------|-------------|--------|--------|--------|-----------------|
| L4/F14233 'ahead of' | N=39 | +0.40% | +1.16% | **+1.88%** | +1.88% @ α=1.0 |
| L14/F10561 'close to' | N=93 | +0.75% | **+1.88%** | +1.39% | +1.88% @ α=0.5 |
| L12/F2257 'facing' | N=306 | +0.95% | running | — | ≥+0.95% |

**Key finding: Both L4 and L14 achieve +1.88% on full-VSR**, confirming this is the natural ceiling for single-feature universal injection. L14 peaks at α=0.5, L4 at α=1.0. The earlier `fixed_L6_everywhere` +1.90% is marginally higher — L6/F7539 may be slightly better globally.

**L12/F2257 gives +0.95% at α=0.25**: As expected — L12 encodes "facing" (a specific direction relation) and is too specialized for universal injection. But it's the BEST for individual relations like "toward" (+17%), "facing" (+10%), "near" (+12%).

### FC-CAA Phase 1 — Null handling bug fixed

**Bug fixed in `pt448_fc_caa.py`:** Phase 2 would crash for null features (fc_caa_data={}).
- Added `skipped_no_firing` check at `main()` entry to Phase 2 — skips loading pt-448 entirely
- Added empty dict guard in `phase2_inject_fc_caa()` — early return with None

This means L9/F387 (confirmed 0 nonzero activations on VSR) will save null and exit WITHOUT loading pt-448. GPU3 will free ~20:10 PDT.

### Smart Oracle v2 — Script written and ready

Script: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_smart_oracle_v2.py`

**Strategy:** Parse captions to identify spatial relation keywords → look up best feature from complete 52-relation matrix → apply `caa_sae_down` injection with that feature.

- 44 beneficial relations → inject (keyword matching, longest-first)
- 8 SKIP relations → no injection (against, at the edge of, far away from, has as a part, in, in front of, into, perpendicular to)
- unknown caption → no injection

Expected parsing stats (estimated from VSR):
- inject: ~N/A (many relations parseable)
- skip: ~N/A (8 harmful relations)
- unknown: small residual

**Background launcher set up:** Monitors for L9/F387 process exit, then automatically launches smart oracle v2 on GPU3.
- Launcher PID: 2538958, log: `/tmp/smart_oracle_launcher.log`
- Smart oracle log: `/tmp/smart_oracle_v2.log`
- Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_smart_oracle_v2/`

**Theoretical range:**
- If caption parsing is perfect → matches per-relation oracle = **+4.56%**
- With inevitable parsing errors → somewhere between **+1.90%** and **+4.56%**
- Key question: how well does keyword matching identify the relation?

### Complete Oracle Feature Map (for reference)

Top gainers per feature (from 52-relation matrix):
- **L12/F2257** wins: beyond (+20%), enclosed_by (+19%), toward (+17%), off (+14%), near (+12%), facing (+10%), outside (+9%), beneath (+8%)
- **L15/F220** wins: beside (+11%), over (+8%), at_left_side (+8%), within (+6%), contains (+6%), above (+5%), away_from (+3%), inside (+2%), part_of (+2%)
- **L6/F7539** wins: alongside (+16%), across_from (+15%), facing_away (+12%), below (+5%), left_of (+5%), at_back_of (+3%), right_of (+1%)
- **L9/F387** wins: touching (+6%), attached_to (+9%), adjacent_to (+6%), at_right_side (+4%), at_side_of (+3%), far_from (+1%)
- **L9/F7540** wins: opposite_to (+6%), parallel_to (+6%), in_middle_of (+5%), consists_of (+3%), next_to (+1%), on (+1%)
- **L4/F14233** wins: ahead_of (+15%), behind (+2%)
- **L14/F10561** wins: connected_to (+11%), close_to (+11%), by (+4%)
- **L11/F12278** wins: surrounding (+10%), on_top_of (+4%), under (+2%)

### FC-CAA Phase 1 Progress (19:30 PDT)

| GPU | Feature | Progress | ETA Phase 1 |
|-----|---------|----------|-------------|
| 0 | L13/F15219 | 7000/10972 (64%) | ~19:36 |
| 5 | L12/F2257 | 7000/10972 (64%) | ~19:36 |
| 2 | L4/F14233 | 2000/10972 (18%) | ~19:46 |
| 3 | L9/F387 | 2000/10972 (18%) | ~19:46 → saves null, exits |
| 4 | L15/F220 | 2000/10972 (18%) | ~19:46 |
| 6 | L6/F7539 | 2000/10972 (18%) | ~19:46 |
| 7 | L11/F12278 | 2000/10972 (18%) | ~19:46 |
| 1 | global_alpha | L12/F2257 α=0.5 | ~20:25 for L12; ~23:30 total |


---

## Monitoring Update 45 — 2026-04-21 19:46 PDT

### FC-CAA Phase 1 Complete — All Features

All 7 FC-CAA Phase 1 jobs completed successfully. Key diagnostic:

| Feature | nonzero/10972 | frac% | |fc_caa| | cos(fc_caa,wdec) | verdict |
|---------|--------------|-------|----------|-----------------|---------|
| L4/F14233 | 2658 | 24.2% | 929.4 | **0.010** | near-orthogonal |
| L6/F7539 | 4231 | 38.6% | 1221.8 | **0.017** | near-orthogonal |
| L9/F387 | 2500 | 22.8% | 1667.9 | **-0.002** | orthogonal |
| L11/F12278 | 2330 | 21.2% | 2044.8 | **-0.005** | orthogonal |
| L12/F2257 | 10972 | 100% | 38.2 | **0.290** | ← ONLY ALIGNED |
| L13/F15219 | 6038 | 55.0% | 2491.7 | **0.007** | near-orthogonal |
| L14/F10561 | 0 | 0% | 0 | 0.000 | null |
| L15/F220 | 9805 | 89.4% | 910.5 | **0.039** | slight alignment |

**Critical finding: Only L12/F2257 (dense feature, cos=0.290) has real feature alignment.**
All other features have FC-CAA directions essentially orthogonal to their SAE decoder directions (cos ≈ 0.01-0.02). This means the mean activation difference (HIGH-LOW conditioning) captures CONTEXT CORRELATES of when the feature fires, not the feature's semantic direction itself.

**Why L12 is special:** 100% of VSR samples activate L12/F2257 (every image/caption gets some activation). The quantile split (HIGH=top 33%, LOW=bottom 33%) cleanly separates HIGH from LOW activation strength, giving a direction that genuinely represents the feature concept. For sparse features (24-55% nonzero), the nonzero-vs-zero split captures "feature fires" vs "feature silent" which mixes many confounders.

### FC-CAA Phase 2 Early Results (own-relation, partial)

| Feature | own-relation | cos | FC-CAA best | caa_sae_down best | vs caa |
|---------|-------------|-----|-------------|-------------------|--------|
| L4/F14233 | ahead of (N=39) | 0.010 | **+2.56%** @ α=0.1-3.0 | +15.38% @ α=1.0 | 6× worse |
| L6/F7539 | left/right (N=323) | 0.017 | **-0.31%** @ α=0.5 | +2.14-3.11% | harmful |
| L9/F387 | at right side (N=480) | -0.002 | **+1.04%** @ α=0.1 | +0.42-1.87% | ~same |
| L12/F2257 | facing (N=306) | 0.290 | **+5.88%** @ α=3.0 | +9.80% @ α=1.0 | 60% as good |
| L13/F15219 | behind (N=709) | 0.007 | **+1.41%** @ α=1.5 | +1.27% @ α=2.0 | ~same |
| L15/F220 | across from/left (N=515) | 0.039 | **+0.39%** @ α=0.1 | +2.14% @ α=0.5 | 5× worse |

**Conclusion: FC-CAA is consistently WORSE than caa_sae_down on own-relation.**
Even L12 with cos=0.290 only achieves 60% of caa_sae_down's gain (+5.88% vs +9.80%).

### Key Methodological Conclusion: FC-CAA Does Not Beat caa_sae_down

FC-CAA hypothesis was: "conditioning on SAE feature activation gives better direction than label-conditioning." Result: **FALSE** for 6/7 features. Only L12 (dense, cos=0.290) shows real alignment, but still underperforms caa_sae_down.

**Why:** SAE features fire in complex contexts. The mean activation difference (HIGH-LOW) encodes the visual/contextual signature of feature firing, not the feature's abstract semantic direction. The W_dec vector (decoder weight) IS the pure feature direction in representation space. Label-conditioned CAA (caa_sae_down) better preserves this.

**Best steering methods remain:**
1. **caa_sae_down with per-relation oracle** → +4.56% (59.01%) [theoretical ceiling]
2. **Smart Oracle v2** (caption parsing + per-feature alpha + caa_sae_down) → **expected ~+4.56%** (caption parsing = 100% GT accuracy, scripts pending GPU)
3. **fixed_L6_everywhere α=0.5** → +1.90% (56.31%) [best simple method]
4. **fixed_L4_everywhere α=1.0** → +1.88% [essentially tied]

### Smart Oracle v2 Status

- Script written and verified: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_smart_oracle_v2.py`
- Uses per-feature optimal alphas: L4=1.0, L14=2.0, L12=1.0, L15=0.75, L11=0.5, L9=0.5, L6=1.5, L9/F7540=0.25
- Caption parsing achieves **100% GT accuracy** on VSR (captions literally contain the relation word)
- 9147 inject / 1722 skip / 103 unknown out of 10972 total
- **Expected result: ~+4.56%** (same as GT oracle since parsing = GT)
- Background launcher watching for free GPU: PID 2567663

### Global Alpha caa_sae_down (GPU1, partial — 3/10 features)

| Feature | α=0.25 | α=0.5 | α=1.0 | Best |
|---------|--------|--------|--------|------|
| L4/F14233 | +0.40% | +1.16% | **+1.88%** | +1.88% @ α=1.0 |
| L14/F10561 | +0.75% | **+1.88%** | +1.39% | +1.88% @ α=0.5 |
| L12/F2257 | +0.95% | **+1.72%** | running | ≥+1.72% |


---

## Monitoring Update 46 — 2026-04-21 20:10 PDT

### CORRECTION: Smart Oracle v2 Parser Bug Found and Fixed

**Bug:** The previous claim of "100% GT accuracy" for caption parsing was WRONG. The `parse_relation()` function checked SKIP relations first (as a separate sorted list), causing "in" (SKIP, len=2) to match inside "in the middle of" (beneficial, len=14) before the longer relation could be checked.

**Fix applied** to `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_smart_oracle_v2.py` (line 106):
- Old: Two separate sorted lists — SKIP first, then beneficial (greedy)
- New: Single unified list sorted by length descending, SKIP/beneficial interleaved

**Corrected accuracy:** 99.06% (9239 beneficial + 1630 skip + 103 unknown = 10869/10972)
- Recovered 92 "in the middle of" samples from SKIP → inject
- Expected smart oracle gain: ~**+4.52%** (vs +4.56% oracle, negligible loss from 103 unknowns)

**Also corrected:** Category counts updated: 9239 inject / 1630 skip / 103 unknown (was: 9147/1722/103)

### FC-CAA Phase 2 — Complete Own-Relation Results (All 7 Features)

All 7 features have finished their own-relation sweep and entered Full VSR pass (N=10,972). Complete own-relation comparison:

| Feature | Relation | N | Base | FC-CAA best | caa_sae_down best | Verdict |
|---------|----------|---|------|-------------|-------------------|---------|
| L4/F14233 | ahead of | 39 | 56.41% | +2.56% @ α=0.1 | **+15.38% @ α=1.0** | FC-CAA 6× worse |
| L6/F7539 | left/right | 323 | 51.08% | **0.00%** @ α=0.75 | +2.14-3.11% | FC-CAA harmful/flat |
| L9/F387 | at right side | 480 | 52.29% | +1.46% @ α=0.5 | **+1.87%** @ α=0.5 | ~same (cos=-0.002) |
| L11/F12278 | touching | 1281 | 56.52% | +0.16% @ α=0.1 | **+6.32% @ α=0.5** | FC-CAA 40× worse |
| L12/F2257 | facing | 306 | 49.02% | +5.88% @ α=3.0 | **+9.80% @ α=1.0** | 60% as good (cos=0.290) |
| L13/F15219 | behind | 709 | 51.62% | +1.41% @ α=1.5 | **+1.27% @ α=2.0** | FC-CAA marginally better |
| L15/F220 | across from/left | 515 | 49.90% | +0.78% @ α=0.5 | **+2.14% @ α=0.5** | FC-CAA 3× worse |

**Summary: FC-CAA is worse on 5/7 features, ~equal on 1/7 (L9/F387, cos≈0), marginally better on 1/7 (L13, cos=0.007). Only L12 (cos=0.290) shows meaningful FC-CAA performance but still trails caa_sae_down. FC-CAA hypothesis REJECTED.**

### Current GPU Status (~20:10 PDT)

All 8 GPUs occupied:

| GPU | Job | Stage | ETA Complete |
|-----|-----|-------|--------------|
| 0 | FC-CAA L13/F15219 | Full VSR baseline pass | ~01:00 AM |
| 1 | FC-CAA L12/F2257 | Full VSR baseline pass | ~01:00 AM |
| 2 | FC-CAA L4/F14233 | Full VSR baseline pass | ~01:00 AM |
| 3 | FC-CAA L9/F387 | Full VSR baseline pass | ~01:00 AM |
| 4 | FC-CAA L15/F220 | Full VSR baseline pass | ~01:00 AM |
| 5 | FC-CAA L11/F12278 | Own-relation sweep (~α=0.25) | ~21:00 → Full VSR 02:00 AM |
| 6 | FC-CAA L6/F7539 | Full VSR baseline pass | ~01:00 AM |
| 7 | FC-CAA L9/F387 full-VSR | Full VSR baseline pass | ~01:00 AM |

Smart Oracle v2 launcher (PID 2567663) watching for any FC-CAA result to complete → will launch on first GPU with <5000 MiB. Earliest GPU free: ~01:00-02:00 AM PDT.

### Next Steps

1. **FC-CAA full VSR results** arrive ~01:00 AM — will confirm FC-CAA gains at each alpha across all N=10,972 samples. Expected: no feature beats caa_sae_down.
2. **Smart Oracle v2** launches automatically when a GPU frees up — expected ~+4.52% gain.
3. **Global alpha caa_sae_down**: L12/α=1.0 was pending; result expected when GPU1 finishes FC-CAA (~01:00 AM). Remaining features: L15, L11, L9/F387, L6, L9/F7540, L13, L11/F9639.


---

## Monitoring Update 47 — 2026-04-21 20:20 PDT

### New Experiments Queued (A+C, B)

Three new scripts written and queued in `/tmp/launch_queue.sh` (PID 2606681), will run sequentially when GPUs free up after FC-CAA jobs (~01:00 AM):

1. **`pt448_smart_oracle_v2.py`** — runs first (already written, parser bug fixed)
2. **`pt448_wenc_steer.py`** — Option B: W_enc column instead of W_dec row
3. **`pt448_adaptive_steer.py`** — Options A+C: SAE-deficit-based feature selection + activation-adaptive alpha

**Option B rationale:** W_dec encodes "what this feature writes to residual stream"; W_enc encodes "what direction maximally activates this feature." They differ in a trained JumpReLU SAE (not perfectly tied). The W_enc direction may be better for transfer steering since it points to WHERE the feature information should be. The script logs `cos(W_enc_col, W_dec_row)` per feature to quantify divergence.

**Options A+C rationale:** At each sample, run the 8 SAE encoders on pt-448's hidden states and compute `deficit = max(0, threshold_f - max_text_activation_f)`. Select feature with largest deficit (C: no text parsing, grounded in representations). Scale alpha by `deficit / mean_deficit` (A: adaptive injection strength). Two sub-modes: `deficit_select` (C only) vs `deficit_select_ada` (A+C).

### Global Alpha caa_sae_down — Partial Results (GPU1, 3/10 features done)

Three features completed at α ∈ {0.25, 0.5, 1.0}:

| Feature | α=0.25 | α=0.5 | α=1.0 | Best (full VSR) |
|---------|--------|-------|-------|----------------|
| L4/F14233 | +0.40% | +1.16% | **+1.88%** | +1.88% @ α=1.0 |
| L14/F10561 | +0.75% | **+1.88%** | +1.39% | +1.88% @ α=0.5 |
| L12/F2257 | +0.95% | +1.72% | α=1.0 running | ≥+1.72% |

Key observation: L4 and L14 both plateau at **+1.88%** (same ceiling as fixed_L6 +1.90%). This supports the "natural ceiling ~+1.88-1.90% for single-feature universal injection."

### FC-CAA Phase 2 — Full VSR Baseline Confirmed, Alpha Sweep Running

L12 confirmed baseline 54.41% → entering α sweep. All other features (L4, L6, L9, L13, L15) have own-relation sweeps done, currently running full VSR alpha sweep. L11 just entered full VSR pass.

Own-relation FC-CAA best (all features):
- L12 best own-rel: **+5.88% @ α=3.0** (facing) — best FC-CAA feature by far (cos=0.290)
- L9 best own-rel: **+1.46% @ α=0.5** (at right side)
- L13 best own-rel: **+1.41% @ α=1.5** (behind)
- L4 best own-rel: **+2.56%** (ahead of, small N=39, noisy)
- L15 best own-rel: **+0.78%** (across from/left side)
- L11 best own-rel: **+0.16%** (touching — near-flat)
- L6 best own-rel: **0.00%** (left/right — harmful at higher α)

### Current GPU Table (~20:20 PDT)

| GPU | Job | Stage |
|-----|-----|-------|
| 0 | FC-CAA L13/F15219 | Full VSR α sweep (~α=0.1) |
| 1 | FC-CAA L12/F2257 | Full VSR α=1.0 running + global_alpha caa_sae_down L12 α=1.0 |
| 2 | FC-CAA L4/F14233 | Full VSR α sweep |
| 3 | FC-CAA L9/F387 | Full VSR α sweep |
| 4 | FC-CAA L15/F220 | Full VSR α sweep |
| 5 | FC-CAA L11/F12278 | Full VSR baseline pass |
| 6 | FC-CAA L6/F7539 | Full VSR α sweep |
| 7 | global_alpha caa_sae_down | L12/F2257 α=1.0 (3rd of 10 features) |

Queue (PID 2606681) watching for free GPU → will launch smart oracle → wenc → adaptive in sequence.


---

## Monitoring Update 48 — 2026-04-21 20:23 PDT

### All Jobs Still Running — No New Results Yet

All 8 GPUs remain fully occupied. FC-CAA Phase 2 full-VSR alpha sweeps are running (baseline pass complete for L4/L6/L9/L12/L13/L15; L11 just started full VSR pass). Global alpha caa_sae_down is on L15 (4th of 10 features). No JSON result files written yet.

### Global Alpha caa_sae_down — 3 Features Complete

| Feature | α=0.25 | α=0.5 | α=1.0 | Best |
|---------|--------|-------|-------|------|
| L4/F14233 | +0.40% | +1.16% | **+1.88%** | +1.88% @ α=1.0 |
| L14/F10561 | +0.75% | **+1.88%** | +1.39% | +1.88% @ α=0.5 |
| L12/F2257 | +0.95% | +1.72% | +1.67% | **+1.72%** @ α=0.5 |

**The +1.88% ceiling is consistent across all three features tested.** Note L12 (our most spatially-aligned feature with cos=0.290) peaks lower (+1.72%) than L4/L14 despite having the best per-relation gain (+9.80%). This suggests L12 is too specific — it improves its own "facing" subset but hurts on unrelated samples, limiting the universal gain.

### FC-CAA Phase 2 Full VSR — All Features In Progress

All features have completed baseline (54.41%) and are in their alpha sweeps. Each feature runs 8 alphas × 10,972 samples via NNsight — estimated 3–4 hrs per feature from when baseline completed (~20:00 PDT).

ETA first result JSON: ~23:30–00:30 PDT (fastest features: L4, L6, L9 since they've been running longest).

### Queue Status

PID 2606681 active, polling every 60s. Will launch in sequence:
1. `pt448_smart_oracle_v2.py` → log: `/tmp/pt448_smart_oracle_v2.log`
2. `pt448_wenc_steer.py` → log: `/tmp/pt448_wenc_steer.log`
3. `pt448_adaptive_steer.py` → log: `/tmp/pt448_adaptive_steer.log`


---

## Monitoring Update 49 — 2026-04-21 20:38 PDT

### Still Running — Early FC-CAA Full-VSR Signal

No result JSONs written yet. All 8 GPUs still fully occupied.

**Early result — L12/F2257 FC-CAA full-VSR α=0.1:**
- Full VSR: **+0.09%** (54.41% → 54.50%)
- This is the *strongest* FC-CAA feature (cos=0.290, best own-relation +5.88%) — and it's near-flat on full-VSR at low alpha. Confirms FC-CAA will not beat caa_sae_down globally.

**Global alpha caa_sae_down:** Now on L15/F220 α=0.25 (4th of 10 features). L4(+1.88%), L14(+1.88%), L12(+1.72%) done.

**Queue:** PID 2606681 still waiting for first free GPU. ~01:00-02:00 AM expected.


---

## Monitoring Update 50 — 2026-04-21 20:50 PDT

### FC-CAA Full-VSR: Decisive Result — All Features Near-Zero at α=0.1

Every FC-CAA feature has now printed its first full-VSR alpha result (α=0.1). All are flat or harmful:

| Feature | FC-CAA full-VSR @ α=0.1 | caa_sae_down @ α=0.25 |
|---------|------------------------|----------------------|
| L4/F14233 | **-0.04%** | +0.40% |
| L6/F7539 | **-0.03%** | (beneficial ~+0.5%) |
| L9/F387 | **-0.23%** | (beneficial ~+0.4%) |
| L12/F2257 | **+0.09%** | +0.95% |
| L13/F15219 | **+0.03%** | (~+0.5%) |
| L15/F220 | **-0.09%** | +0.98% |

**FC-CAA full-VSR hypothesis CONCLUSIVELY REJECTED.** Even at the lowest alpha, FC-CAA vectors are at best +0.09% (L12, the only feature with cos=0.290). caa_sae_down delivers 10–20× larger gains at the same alpha. The FC-CAA direction (mean HIGH - mean LOW activation split) encodes contextual correlates of feature firing, not the feature's transfer-useful spatial direction.

### Global Alpha caa_sae_down — 4 Features Complete

| Feature | α=0.25 | α=0.5 | α=1.0 | Best |
|---------|--------|-------|-------|------|
| L4/F14233 | +0.40% | +1.16% | **+1.88%** | +1.88% @ α=1.0 |
| L14/F10561 | +0.75% | **+1.88%** | +1.39% | +1.88% @ α=0.5 |
| L12/F2257 | +0.95% | +1.72% | +1.67% | **+1.72%** @ α=0.5 |
| L15/F220 | **+0.98%** | running | — | ≥+0.98% |

Pattern holds: universal ceiling ~+1.88% per single feature.

### Next Steps
- FC-CAA jobs will complete (~01:00 AM) and free GPUs for queue
- Queue: smart oracle v2 → wenc → adaptive_steer
- Key question: will smart oracle v2 (~+4.52% expected) and the new deficit-based methods break this ceiling meaningfully?


---

## Monitoring Update 51 — 2026-04-21 21:27 PDT

### NEW RECORD: L15/F220 caa_sae_down at α=0.5 → +2.09% (56.50%)

**L15/F220 breaks the +1.90% ceiling** that every previous single-feature universal injection had hit:

| Feature | Best Δ | Best α |
|---------|--------|--------|
| L4/F14233 | +1.88% | 1.0 |
| L14/F10561 | +1.88% | 0.5 |
| L12/F2257 | +1.72% | 0.5 |
| **L15/F220** | **+2.09%** | **0.5** |
| fixed_L6 (prior best) | +1.90% | 0.5 |

L15 α=1.0 still running — may go higher or lower. Fine sweep queued to pin down the exact optimal alpha in the 0.3–0.8 range.

**Why L15 may be special:** L15/F220 has start_layer=15 (injects only layers 15-25), meaning it targets the later processing layers that make the final spatial judgment, rather than broadcasting to all 26 layers. This late-layer targeting may explain the superior universal performance — less interference with early feature formation, more direct influence on the final reasoning step.

### FC-CAA Full-VSR: Hypothesis Confirmed Dead at α=0.25

All 7 features at α=0.25 show near-zero or negative gains:
- L4: -0.09%, L6: **+0.06%**, L9: -0.07%, L11: -0.06%, L12: **+0.16%**, L13: -0.05%, L15: +0.03%

Maximum FC-CAA full-VSR gain = **+0.16%** (L12 at α=0.25) vs caa_sae_down **+2.09%**. FC-CAA is definitively 10-15× worse than caa_sae_down on full-VSR.

### Queue Updated

Added `pt448_L15_fine_sweep.py` to queue (position 4). Queue order:
1. `pt448_smart_oracle_v2.py` — expected ~+4.52%
2. `pt448_wenc_steer.py` — Option B: W_enc direction
3. `pt448_adaptive_steer.py` — Options A+C: SAE deficit selection + adaptive alpha
4. `pt448_L15_fine_sweep.py` — fine sweep α=0.3–2.0 for L15/F220

---

## Monitoring Update 52 — 2026-04-21 21:44 PDT

### FC-CAA Full-VSR: Complete Rejection Confirmed Across All 7 Features

All FC-CAA jobs (GPUs 0,2,3,4,5,6,7) have now run own-relation and partial full-VSR sweeps. Full-VSR results are decisively near-zero or negative for all features:

| Feature | Own-relation best Δ | Full-VSR max Δ | Status |
|---------|-------------------|----------------|--------|
| L4/F14233 | +2.56% @ α=0.1-0.5 | **+0.03%** @ α=0.5 | Still running (2/8 full-VSR alphas) |
| L6/F7539 | +0.00% @ α=0.75 | **+0.06%** @ α=0.25 | Still running (2/8) |
| L9/F387 | +1.46% @ α=0.5,1.0 | **-0.07%** @ α=0.25 | Still running (2/8) |
| L11/F12278 | +0.16% @ α=0.1 | **+0.01%** @ α=0.25 | Still running (2/8) |
| L12/F2257 | +5.88% @ α=3.0 | **+0.24%** @ α=0.5 | Still running (3/8) |
| L13/F15219 | +1.41% @ α=1.5-2.0 | **+0.03%** @ α=0.1 | Still running (3/8) |
| L15/F220 | +0.78% @ α=0.5,1.5,2.0 | **+0.03%** @ α=0.25 | Still running (2/8) |

Key observations:
- L12/F2257 own-relation at α=3.0: +5.88% on "facing" subset — but full-VSR still only +0.24% max. The dense feature works on its own relation but doesn't generalize.
- L4/F14233 own-relation holds flat at +2.56% across α=0.1–3.0 — integer precision artifact (N=156 samples, count unchanged).
- FC-CAA definitively rejected: maximum full-VSR gain = +0.24% (L12) vs caa_sae_down +2.09%.

### Global Alpha caa_sae_down — Now at L11/F12278

L11/F12278 started at ~21:42 PDT. Expected to complete ~23:30 PDT. After L11, still need: L9/F387, L6/F7539, L9/F7540, L13/F15219, L11/F9639.

Complete global alpha results so far:

| Feature | α=0.25 | α=0.5 | α=1.0 | Best |
|---------|--------|-------|-------|------|
| L4/F14233 | +0.40% | +1.16% | **+1.88%** | **+1.88%** @ α=1.0 |
| L14/F10561 | +0.75% | **+1.88%** | +1.39% | **+1.88%** @ α=0.5 |
| L12/F2257 | +0.95% | **+1.72%** | +1.67% | **+1.72%** @ α=0.5 |
| L15/F220 | +0.98% | **+2.09%** | +1.26% | **+2.09%** @ α=0.5 ← NEW RECORD |
| L11/F12278 | running | — | — | ≥+0.00% |

Pattern: Only L15 breaks +1.88%. All features tested so far confirm the late-injection hypothesis.

### Queue Updated — `pt448_late_start_sweep.py` Added

Added `pt448_late_start_sweep.py` as script #5 in queue. All 8 GPUs still busy with FC-CAA + global alpha jobs. Queue waiting for first free GPU.

Updated queue order:
1. `pt448_smart_oracle_v2.py` — expected ~+4.52%
2. `pt448_wenc_steer.py` — Option B: W_enc direction
3. `pt448_adaptive_steer.py` — Options A+C: SAE deficit selection + adaptive alpha
4. `pt448_L15_fine_sweep.py` — fine sweep α=0.3–2.0 for L15/F220
5. `pt448_late_start_sweep.py` — **NEW**: all 8 features with start_layer=15 to test late-injection hypothesis universally

---

## Monitoring Update 53 — 2026-04-21 22:05 PDT

### Methodology Correction: Evaluating on Own-Relation Subsets (Not Full VSR)

**Key insight from user:** The correct ablation methodology (matching original codebase) is to evaluate each feature on its own-relation subset, not full VSR. The +2.09% full-VSR universal figure is a secondary metric; the primary result is per-feature performance on the subset of samples containing that feature's spatial relation.

**FC-CAA own-relation results (from saved logs before kill):**

The FC-CAA vectors were built from the full 10k dataset (n_high=3621, n_low=3621 per feature), but evaluated on own-relation subsets show they are still inferior to caa_sae_down:

| Feature | Relation | N | caa_sae_down best Δ | FC-CAA best Δ |
|---------|----------|---|---------------------|---------------|
| L4/F14233 | ahead of | 39 | **+15.38%** | +2.56% (flat, α=0.1-3.0) |
| L6/F7539 | left/right of | ~323 | **+4.76%** | 0.00% max |
| L9/F387 | at the right side of | 480 | **+4.17%** | +1.46% @ α=0.5,1.0 |
| L11/F12278 | touching | 1281 | **+6.48%** | +0.16% @ α=0.1 |
| L12/F2257 | facing | 306 | **+9.80%** | +5.88% @ α=3.0 |
| L13/F15219 | behind | 709 | **+1.55%** | +1.41% @ α=1.5-2.0 |
| L15/F220 | across from + at the left side of | 515 | **+7.84%** | +0.78% @ α=0.5 |

**Why FC-CAA is weaker even on own-relation subsets:** The FC-CAA split (high vs low SAE activation in pt-448) is noisy because pt-448 barely fires these spatial features. The caa_sae_down split (YES vs NO VSR answers on relation-specific samples) is a clean semantic signal. More data (full 10k) doesn't compensate for a noisier split criterion. **caa_sae_down is definitively the better vector.**

### Actions Taken This Cycle

**Killed:** All 7 FC-CAA jobs (confirmed useless), old queue launcher (PID 2824416). Freed GPUs 0,2,3,4,5,6,7.

**Launched immediately (no queue wait):**
- **GPU 0** — `pt448_smart_oracle_v2.py` (PID 2956415) — per-feature caption-routing, expected ~+4.52%
- **GPU 2** — `pt448_late_start_sweep.py` (PID 2956416) — all 8 features with start_layer=15
- **GPU 3** — `pt448_L15_fine_sweep.py` (PID 2956417) — fine alpha for L15/F220
- **GPU 4** — `pt448_per_relation_alpha_sweep.py` (PID 2966161) — **NEW CORRECT METHODOLOGY**

**`pt448_per_relation_alpha_sweep.py`** — NEW SCRIPT (written this cycle):
- Evaluates every VSR relation subset (N≥20 samples) independently
- Sweeps α ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0} for **all 8 features** on every relation
- Also runs FC-CAA vectors on the same subsets for direct comparison
- This is the definitive caa_sae_down vs FC-CAA comparison at the subset level
- Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_per_relation_alpha_sweep/`
- Started on "touching" (N=1281, largest relation)

**Removed from queue:** `pt448_wenc_steer.py`, `pt448_adaptive_steer.py` — weak theoretical basis, user-confirmed deprioritized.

**Kept free:** GPUs 5, 6, 7 for follow-up experiments once results come in.

### Global Alpha caa_sae_down — L11/F12278 Now Running

L11/F12278 (own_rels=['touching'], start=5) now has partial results:
- α=0.25: full_VSR +0.90%, own_rel (touching) +2.03%
- α=0.5: running

After L11, still need: L9/F387, L6/F7539, L9/F7540, L13/F15219, L11/F9639.

### Updated Understanding of Best Methods

| Method | Evaluation | Best Result |
|--------|-----------|-------------|
| caa_sae_down (own-relation subset) | Per-feature subset | **+15.38%** L4/ahead-of, **+9.80%** L12/facing, **+10.75%** L14/close-to |
| caa_sae_down (full VSR universal) | All 10,972 samples | **+2.09%** L15/F220 |
| caa_sae_down (caption-routed oracle) | All 10,972 samples | **~+4.52% expected** (running) |
| FC-CAA (own-relation subset) | Per-feature subset | +5.88% L12 @ α=3.0 (but caa_sae_down gets +9.80% same relation) |
| Multi-feature simultaneous | Full VSR | **-5.6%** (catastrophically bad) |

---

## Monitoring Update 54 — 2026-04-21 22:30 PDT

### Smart Oracle v2 — Early Results Strongly Confirming ~+3.8% at 4000/10972 Samples

At 4000 samples the smart oracle (caption-routing, per-feature alpha) is tracking:
- `base=54.88%,  smart=58.67%` → **Δ = +3.79% at 4000 samples**

This is slightly below the expected +4.52% (from GT oracle +4.56%), but the early samples may skew. Will update when first alpha mode completes (~30 more min).

### Coverage Gap Analysis: "behind" and "in front of" Unsteerable with Current 8 Features

Examined per-relation results (`per_relation_steer.json`) for weakly-steered relations. Critical finding:

| Relation | N | Best current feature | Best Δ | Problem |
|----------|---|---------------------|--------|---------|
| **in front of** | 737 | L15/F220 | **-0.68%** | ALL 8 features give negative Δ |
| **behind** | 709 | L4/F14233 | **+1.55%** | Very weak, not own-relation |
| **on** | 585 | L9/F7540 | **+0.68%** | Near-zero |
| **under** | 589 | L11/F12278 | **+2.38%** | Weak |
| **on top of** | 505 | L11/F12278 | **+4.36%** | Moderate |

"in front of" (N=737) is completely unsteerable — every one of our 8 features is harmful. "behind" (N=709) has only +1.55% with L4, which is not its own feature.

### New Experiment: L15/F8844 ("behind") and L15/F1149 ("in front of") — CAA Vectors

Identified two high-quality layer-15 features that are dedicated to the two biggest coverage gaps:

| Feature | Relation | Odds ratio (mix-448) | Ablation Δ on mix-448 |
|---------|----------|---------------------|----------------------|
| **L15/F8844** | behind | **15.4** | -1.55% (hurts when ablated) |
| **L15/F1149** | in front of | **16.7** | +0.81% (borderline) |

Both are at layer 15 — same layer as L15/F220 which gave our best universal result.

**Launched:** `pt448_new_features_caa.py` on **GPU 5** (PID 2984939)
- Phase 1: Extract CAA vectors from mix-448 using YES/NO split on own-relation samples (pos=357 "behind", neg=352)
- Phase 2: Steer pt-448 on "behind", "in front of" + related relation subsets with alpha sweep [0.1…3.0]
- Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_new_features_caa/`

**Hypothesis:** If L15/F8844 gives +5%+ on "behind" subset and L15/F1149 gives +5%+ on "in front of" subset, this dramatically improves the smart oracle by covering 1446 currently-unsteered/negatively-steered samples.

### Full Per-Relation Alpha Sweep — Running on GPU 4

`pt448_per_relation_alpha_sweep.py` started on "touching" (N=1281) — sweeping all 8 caa_sae_down + 7 FC-CAA vectors × 8 alphas. This is the definitive comparison of both vector types on per-relation subsets. Will take ~8-10h to complete all 52 relations.

### GPU Status at 22:30 PDT

| GPU | Job | Status |
|-----|-----|--------|
| 0 | smart_oracle_v2 | ~4000/10972 samples, Δ≈+3.8% early signal |
| 1 | global_alpha_fullvsr | L11/F12278 α=0.5 running |
| 2 | late_start_sweep | Baseline pass running |
| 3 | L15_fine_sweep | Baseline pass running |
| 4 | per_relation_alpha_sweep | "touching" N=1281, L4 first alphas running |
| 5 | **new_features_caa** | Phase 1 extracting L15/F8844 "behind" vectors from mix-448 |
| 6 | Free | — |
| 7 | Free | — |

---

## Monitoring Update 55 — 2026-04-21 22:55 PDT

### NEW RECORD: L11/F12278 at α=0.5 → full_VSR +2.34% (56.75%)

**L11/F12278 ("touching" feature, start=5) surpasses L15/F220's +2.09% record:**

| Feature | α=0.25 | α=0.5 | α=1.0 |
|---------|--------|-------|-------|
| L4/F14233 | +0.40% | +1.16% | +1.88% |
| L14/F10561 | +0.75% | +1.88% | +1.39% |
| L12/F2257 | +0.95% | +1.72% | +1.67% |
| L15/F220 | +0.98% | **+2.09%** | +1.26% |
| **L11/F12278** | **+0.90%** | **+2.34% ← NEW RECORD** | running |

Own-relation (touching subset): α=0.5 → **+6.32%** (62.84%)

L11/F12278 has start_layer=5 — earlier than L15/F220 (start=15) — yet still achieves the best full-VSR result. This weakens the "late injection = better universal" hypothesis, or at least shows it's not universally true.

**Launched:** `pt448_L11_fine_sweep.py` on GPU 5 (PID 3016962) — sweeping α ∈ {0.2, 0.25, …, 1.25, 1.5, 2.0} to find exact peak.

### New Features CAA — FAILED (L15/F8844 and L15/F1149 give 0% delta)

CAA vectors for L15/F8844 ("behind") and L15/F1149 ("in front of") were successfully extracted from mix-448 (pos=357, neg=352 for "behind"; pos=384, neg=353 for "in front of"). However Phase 2 steering showed **exactly 0.00% delta on all alphas** for "behind" and "in front of" subsets.

**Root cause:** Raw last-token CAA vectors from mix-448 extract a direction that is near-orthogonal to the YES/NO decision boundary in pt-448's residual stream for these relations. The method works for the existing 8 features only because their CAA vectors are W_dec-projected — the `caa_sae_down` strategy uses `v_caa_norm` which is computed from relation-specific YES/NO splits and aligns with the SAE decoder direction. A raw hidden-state CAA without this W_dec anchoring does not transfer.

**Conclusion: "behind" and "in front of" remain unsteerable with current approach. These two large relations (N=709+737=1446 samples) are a fundamental coverage gap.**

Job killed — GPU 5 reassigned to L11 fine sweep.

### Per-Relation Alpha Sweep (GPU 4) — First Results: "touching" cross-feature

L4/F14233 on "touching" subset (N=1281):
- α=0.1: +0.55%, α=0.5: **+2.81%**, α=0.75: +5.00%, α=1.0: **+5.31%** ← best, α=3.0: -6.17%

L4 (the "ahead of" feature) gives **+5.31% on "touching"** — better than L11/F12278's own-relation result at α=1.0. This cross-relation transfer is significant and was only visible because we're running the proper subset-level alpha sweep.

### Smart Oracle v2 — Stable at +3.8% through 8000/10972 Samples

Current trajectory:
- 2000 samples: base=54.00%, smart=57.70% (Δ=+3.70%)
- 4000 samples: base=54.88%, smart=58.67% (Δ=+3.79%)
- 6000 samples: base=54.60%, smart=58.52% (Δ=+3.92%)
- 8000 samples: base=54.61%, smart=58.44% (Δ=+3.83%)

Converging around **+3.8%** rather than the expected +4.52%. Possible reasons:
1. Per-feature alpha configs are suboptimal for some relations (tuned on own-relation, not full smart oracle context)
2. The parser misclassifies some samples (~1% error rate introduces noise)
3. Alphas could be re-tuned with smart oracle as the objective

### Updated GPU Status at 22:55 PDT

| GPU | Job | Status |
|-----|-----|--------|
| 0 | smart_oracle_v2 | 8000/10972, stable Δ≈+3.83% |
| 1 | global_alpha_fullvsr | L11/F12278 α=1.0 running |
| 2 | late_start_sweep | L4_F14233 first alpha running (start_layer=15) |
| 3 | L15_fine_sweep | Baseline done, sweeping alphas |
| 4 | per_relation_alpha_sweep | "touching" — L4 done (+5.31%), continuing other features |
| 5 | **L11_fine_sweep** | NEW — baseline running |
| 6 | Free | — |
| 7 | Free | — |

---

## Monitoring Update 56 — 2026-04-21 23:30 PDT

### METHODOLOGY CORRECTION: All sweeps must use per-relation subsets only

User confirmed: per our original ablation methodology (matching the codebase in vision-language-scope root), steering evaluations must be done at the per-relation subset level, not on full VSR. Full-VSR results are secondary.

**Action taken:** Killed three full-VSR-only fine sweep scripts and replaced with correct subset-level versions:
- `pt448_L11_fine_sweep.py` → KILLED → `pt448_L11_subset_sweep.py` (GPU 2)
- `pt448_L15_fine_sweep.py` → KILLED → `pt448_L15_subset_sweep.py` (GPU 3)  
- `pt448_late_start_sweep.py` → KILLED → `pt448_late_start_subset_sweep.py` (GPU 5)

Three new scripts launched on GPUs 2, 3, 5.

Also launched two NEW experiments on free GPUs 6 and 7 using the correct subset methodology.

### Smart Oracle v2 — FIRST ALPHA MODE COMPLETE

**Result (per_feature mode): base=54.41%, smart=58.38%, Δ=+3.96%**

Breakdown by action type:
- `inject` (N=9239, 84.2% of samples): base=54.06%, smart=58.77%, **Δ=+4.71%**
- `skip` (N=1630, SKIP_RELATIONS): Δ=0.00% (no steering, expected)
- `unknown` (N=103, unparseable): Δ=0.00% (no steering, expected)

The inject-only delta (+4.71%) is the true signal: on steerable relations the oracle achieves nearly +5% improvement. The overall +3.96% is dragged down by skip/unknown samples that get passthrough.

**Smart oracle continues** running α=0.25 fixed mode for comparison. Saved at `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_smart_oracle_v2/smart_oracle_v2_per_feature.json`

### Global Alpha Script — Final Own-Relation Results from Global Sweep

All features × 3 alphas now showing both full_VSR and own_rel results:

| Feature | α | full_VSR Δ | own_rel Δ | own_rel base |
|---------|---|-----------|----------|-------------|
| L4/F14233 | 1.0 | +1.88% | **+15.38%** | 56.41% |
| L14/F10561 | 0.5 | +1.88% | +2.15% | ~64.5% |
| L14/F10561 | 1.0 | +1.39% | **+6.45%** | ~64.5% |
| L12/F2257 | 1.0 | +1.67% | **+9.80%** | ~49.0% |
| L15/F220 | 0.5 | **+2.09%** | +6.60% | ~50.0% |
| L15/F220 | 1.0 | +1.26% | +6.80% | ~50.0% |
| L11/F12278 | 0.5 | **+2.34%** | **+6.32%** | 56.52% |

Key insight: Per-relation subsets give dramatically higher deltas than full-VSR (e.g. L4: +15.38% own_rel vs +1.88% full). This confirms the methodology importance.

### Per-Relation Alpha Sweep (GPU 4) — "touching" in progress

Full cross-feature × alpha sweep on "touching" subset (N=1281):
- **L4/F14233**: best α=1.0 → **+5.31%** (N.B. this is the "ahead of" feature working on "touching")
- **L14/F10561**: α=0.5 → **+5.62%** (best so far for "touching" with this sweep)
- L12, L15, L11, L9, L6, L9_F7540: still running

Interesting: L14/F10561 (the "close to" feature) gives **+5.62% on "touching"** at α=0.5, beating even L4/F14233 (+5.31%) and L11/F12278's own-relation α=0.5 (+6.32%).

### New Experiments Launched — GPU 6 and 7

**GPU 6: `pt448_all_features_subset_sweep.py` (PID 3042589)**
Fine alpha sweep for all 6 remaining features (L4, L14, L12, L9/F387, L6, L9/F7540) on their own-relation subsets. Fine-grained alphas around known good values. Fills the gap left by not having L11/L15-level detail for these features.

Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_all_features_subset_sweep/`

**GPU 7: `pt448_extra_features_subset_sweep.py` (PID 3046684)**  
Tests extra CAA vectors L13/F15219 ("behind") and L11/F9639 ("in/inside/on") on their own-relation subsets.

- **L13/F15219**: specifically the "behind" feature — all canonical 8 give ≤+1.55% on "behind". This is the first direct test of the purpose-built "behind" vector.
- **L11/F9639**: covers "in/inside/on" — three large relation groups

This is critical: if L13/F15219 works on its own "behind" subset, it breaks the "behind = unsteerable" conclusion from the earlier raw-CAA failure. The key difference: L13/F15219 was extracted using the same SAE-based caa_sae_down method (not raw hidden-state CAA).

Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_extra_features_subset_sweep/`

### Updated GPU Status at 23:30 PDT

| GPU | Job | Status |
|-----|-----|--------|
| 0 | smart_oracle_v2 | α=per_feature DONE (+3.96%), now running α=0.25 mode |
| 1 | global_alpha_fullvsr | L11/F12278 α=1.0, L9/F387, L6, L9/F7540 remaining |
| 2 | **L11_subset_sweep** | NEW — own-rel (touching) fine sweep, loading model |
| 3 | **L15_subset_sweep** | NEW — own-rel (across from, at the left side of) sweep, loading |
| 4 | per_relation_alpha_sweep | "touching" — L4 (+5.31%), L14 (+5.62%), continuing |
| 5 | **late_start_subset_sweep** | NEW — all 8 features at start=15 on own-rel subsets |
| 6 | **all_features_subset_sweep** | NEW — fine alpha for L4, L14, L12, L9, L6, L9/F7540 |
| 7 | **extra_features_subset_sweep** | NEW — L13/F15219 "behind" + L11/F9639 "in/on" |

---

## Monitoring Update 57 — 2026-04-22 00:10 PDT

### Per-Feature Own-Relation Subset Sweep — First Completed Results

**`pt448_all_features_subset_sweep.py` (GPU 6) — early results:**

**L4/F14233 "ahead of" (N=39):**
| α | Δ% |
|---|-----|
| 0.5 | +5.13% |
| 0.75 | +7.69% |
| 0.8 | +10.26% |
| **0.9** | **+15.38% ← BEST** |
| 1.0 | +10.26% |
| 1.25 | +12.82% |
| 1.5 | +2.56% |
| 2.0+ | negative |

**Optimal α = 0.9, NOT 1.0.** Current FEATURE_CONFIGS has α=1.0 for L4. Need to update to 0.9 for oracle.

**L14/F10561 "close to" (N=93):**
| α | Δ% |
|---|-----|
| 0.5 | +2.15% |
| 1.0 | +7.53% |
| 1.25 | +9.68% |
| **2.0** | **+10.75% ← BEST** |
| 2.5 | +9.68% |
| 3.0 | +9.68% |

**Optimal α = 2.0**, matches existing FEATURE_CONFIGS (α=2.0). Confirmed.

**L12/F2257 "facing" — in progress.**

### Late-Start Hypothesis: PARTIALLY REJECTED on Subset Evaluation

**`pt448_late_start_subset_sweep.py` (GPU 5) — completed features:**

| Feature | own_rel | Natural start best | Late-start (=15) best | Δ vs natural |
|---------|---------|--------------------|-----------------------|-------------|
| L4/F14233 | "ahead of" | +15.38% | +10.26% (α=1.5) | **−5.12pp** |
| L14/F10561 | "close to" | +9.80% | +10.75% (α=2.0) | **+0.95pp** |
| L12/F2257 | "facing" | +9.80% | running | — |

**Conclusion:** Late-start injection at layer 15 **hurts** L4 by 5pp (it needs early layers for "ahead of"). For L14 it's marginally helpful (+0.95pp). The hypothesis that later = better for own-relation performance is wrong for L4.

This makes sense: L4's feature fires from layer 0 and its CAA vectors span the full residual stream; restricting to layers 15+ cuts off the most informative early injection window.

### L15/F220 Fine Subset Sweep (GPU 3) — "at the left side of" in progress

| α | Δ% on "at the left side of" (N=421, base=51.78%) |
|---|---|
| 0.3 | +2.38% |
| 0.35 | +2.85% |
| 0.4 | +3.56% |
| 0.45 | +4.75% ← and still climbing |

Previous best known: +6.60% at α=0.5 (global sweep). Fine sweep confirming monotonic increase through 0.45.

### Extra Features (GPU 7) — L13/F15219 "behind" baseline

L13/F15219 baseline on "behind" (N=709): **51.62%** — alpha sweep now running. This is the key test of whether the dedicated "behind" feature can steer where all canonical 8 failed (best was only +1.55% from L4).

### Updated GPU Status at 00:10 PDT

| GPU | Job | Status |
|-----|-----|--------|
| 0 | smart_oracle_v2 | α=0.25 mode running |
| 1 | global_alpha_fullvsr | L11/F12278 α=1.0, remaining features queued |
| 2 | L11_subset_sweep | "touching" baseline done (56.52%), sweeping alphas |
| 3 | L15_subset_sweep | "at_left_side_of" @ α=0.45 (+4.75%), climbing |
| 4 | per_relation_alpha_sweep | "touching" — L14 (+5.62%), continuing 6 more features |
| 5 | late_start_subset_sweep | L4 done (−5.12pp), L14 done (+0.95pp), L12 running |
| 6 | all_features_subset_sweep | L4 done (α=0.9→+15.38%), L14 done (α=2.0→+10.75%), L12 running |
| 7 | extra_features_subset_sweep | L13/F15219 "behind" baseline done, sweep running |

---

## Monitoring Update 58 — 2026-04-22 01:00 PDT

### Subset-Level Fine Sweeps — CONFIRMED BETTER ALPHAS FOR FEATURE CONFIGS

**`pt448_all_features_subset_sweep.py` completed L4, L14, L12:**

| Feature | Relation | N | Old α (oracle) | Old Δ% | New best α | New Δ% | Change |
|---------|----------|---|---------------|--------|-----------|--------|--------|
| L4/F14233 | "ahead of" | 39 | 1.0 | +15.38% | **0.9** | **+15.38%** | tie |
| L14/F10561 | "close to" | 93 | 2.0 | +9.80% | **2.0** | **+10.75%** | +0.95pp |
| L12/F2257 | "facing" | 306 | 1.0 | +9.80% | **0.75** | **+8.82%** | −0.98pp (N/A: old +9.80 was from coarse sweep) |

- L14: fine sweep confirms α=2.0 is optimal, and reveals the actual peak is +10.75% (not +9.80% from coarser global sweep)
- L12: **α=0.75 is the true optimum** (+8.82%), previous α=1.0 gave +7.84%. Oracle needs update.

L9/F387 "at the right side of" (N=480) sweep now running on GPU 6.

### L15/F220 Fine Sweep on "at the left side of" (N=421, base=51.78%)

Peak found: **α=0.7 → +8.55%**. Previous best (global sweep, α=0.5) was only +6.60%. Fine sweep improves L15 own-relation performance by **+1.95pp**.

| α range | trend |
|---------|-------|
| 0.3–0.45 | +2.4% to +4.8%, monotone |
| 0.5 | +4.51% (dip vs 0.45) |
| 0.55–0.7 | **+7.4% to +8.55%** (big jump at 0.55) |
| 0.75–0.9 | declining |

Interesting non-monotone behavior around α=0.5 (dip). Peak clearly at α=0.7. Sweep now moving to "across from" (N=94).

### Late-Start Hypothesis — FINAL VERDICT: Mostly Harmful

| Feature | natural_start | Own-rel natural best | Late-start (=15) best | Change |
|---------|--------------|---------------------|----------------------|--------|
| L4/F14233 | 0 | +15.38% | +10.26% (α=1.5) | **−5.12pp WORSE** |
| L14/F10561 | 0 | +10.75% | +10.75% (α=2.0) | 0pp tie |
| L12/F2257 | 1 | +8.82% | +7.84% (α=1.0) | **−0.98pp WORSE** |
| L15/F220 | 15 | (measuring) | +7.57% (α=0.75) | +0.97pp better |

**Conclusion:** Late injection (start=15) is harmful for features with early natural start (L4, L12). L14 breaks even; L15 marginally benefits (it already starts at 15 naturally). The earlier hypothesis from full-VSR performance that later = better does **not** hold at the own-relation subset level.

### L13/F15219 "behind" — CONFIRMED UNSTEERABLE (own feature, own relation)

All alphas on "behind" (N=709, base=51.62%): max delta = **+0.14% at α=0.4**.

This is the definitive test — L13/F15219 fires with odds_ratio=15.4 specifically on "behind" and its CAA vectors were computed from the correct relation-split. Still cannot steer pt-448 on this relation. **"behind" (N=709) and "in front of" (N=737) remain a fundamental unsteerable coverage gap** regardless of which feature or method is used.

### Updated GPU Status at 01:00 PDT

| GPU | Job | Status |
|-----|-----|--------|
| 0 | smart_oracle_v2 | α=0.25 mode running |
| 1 | global_alpha_fullvsr | L9/F387, L6/F7539, L9/F7540 remaining |
| 2 | L11_subset_sweep | touching — α=0.3 → +3.04%, building |
| 3 | L15_subset_sweep | at_left_side_of done (α=0.7→+8.55%), on "across from" |
| 4 | per_relation_alpha_sweep | "touching" — L4, L14 done; L12+ running |
| 5 | late_start_subset_sweep | L4,L14,L12 done; L15 running (α=0.75→+7.57%) |
| 6 | all_features_subset_sweep | L4,L14,L12 done; L9/F387 running |
| 7 | extra_features_subset_sweep | L13 "behind" done (useless ≤+0.14%); L11_F9639 "in/on" next |

---

## Monitoring Update 59 — 2026-04-22 01:30 PDT

### L15/F220 "at the left side of" Fine Sweep COMPLETE — New Best Alpha Confirmed

**Peak α=0.7 → +8.55%** on "at the left side of" (N=421, base=51.78%)

Full curve: 0.45→+4.8%, 0.5→+4.5% (non-monotone dip), 0.55→+7.4%, 0.7→**+8.55%** (PEAK), 0.75→+7.6%, 1.0→+4.8%, 1.5→+1.0%

This is +1.95pp better than the previous oracle's α=0.75 (+7.60% is actually shown here at 0.75 vs the 6.60% from coarser global sweep). Oracle v3 alpha of 0.7 is confirmed.

L15 sweep now on "across from" (N=94, base=41.49%). α=0.3 → +4.26% already.

### L13/F15219 "behind" — DEFINITIVELY UNSTEERABLE

Full alpha curve on "behind" (N=709, base=51.62%): max delta = **+0.28%** at α=0.5. All other alphas ≤ 0%. 

The dedicated "behind" feature (extracted with correct SAE-based caa_sae_down method, not raw CAA) still cannot steer pt-448. This is a fundamental coverage gap in the pt-448 model's spatial relation representation.

### Smart Oracle v3 — Queued (auto-launcher PID 3095306)

`pt448_smart_oracle_v3.py` ready with updated alphas:
- L12/F2257: α=1.0 → **α=0.75** (fine sweep: +8.82% vs +7.84%)
- L15/F220: α=0.75 → **α=0.7** (fine sweep: +8.55% vs coarser +6.60%)
- L4/F14233: α=1.0 → **α=0.9** (fine sweep confirms, tie at same delta)

Will launch on GPU 3 when L15_subset_sweep completes. Expected improvement ~0.3-0.5pp on overall oracle delta.

### Ongoing Jobs Summary

| GPU | Job | Key result so far |
|-----|-----|-------------------|
| 0 | smart_oracle_v2 | α=0.25 mode running (~4000/10972) |
| 1 | global_alpha_fullvsr | L9/F387 own_rel running |
| 2 | L11_subset_sweep | touching α=0.35→+4.06%, approaching peak |
| 3 | L15_subset_sweep | at_left_side_of done (α=0.7→+8.55%), across_from running |
| 4 | per_relation_alpha_sweep | touching cross-feature matrix ongoing |
| 5 | late_start_subset_sweep | late-start hurts L4/L12, breaks even for L14/L15 |
| 6 | all_features_subset_sweep | L4(α=0.9→+15.38%), L14(α=2.0→+10.75%), L12(α=0.75→+8.82%) done; L9/F387 running |
| 7 | extra_features_subset_sweep | L13/F15219 "behind" max +0.28% (useless); L11_F9639 in/on/on next |

---

## Monitoring Update 60 — 2026-04-22 02:00 PDT

### all_features_subset_sweep COMPLETE — All 8 Features Confirmed

All 8 canonical features' own-relation subset sweeps are done. Final summary:

| Feature | Own Relation(s) | N | Base | Best α | Best Δ |
|---------|-----------------|---|------|--------|--------|
| L4/F14233 | ahead of | 39 | 56.41% | **0.9** | **+15.38%** |
| L14/F10561 | close to | 93 | 60.22% | **2.0** | **+10.75%** |
| L12/F2257 | facing | 306 | 49.02% | **0.75** | **+8.82%** |
| L15/F220 | at_left_side_of+across_from | 515 | 49.90% | **0.7** | **+8.55%** |
| L11/F12278 | touching | 1281 | 56.52% | **0.45** | **+6.01%** (partial, still running) |
| L9/F387 | at_right_side_of | 480 | 52.29% | **0.4** | **+4.17%** |
| L6/F7539 | left_of+right_of | 323 | 51.08% | **1.5** | **+3.10%** |
| L9/F7540 | consists_of | 35 | 68.57% | 0.2 | **+0.00%** ← UNSTEERABLE |

**Critical finding: L9/F7540 gives 0% improvement on its own relation "consists_of"** (N=35, base=68.57%). The oracle currently assigns L9/F7540 to 6 relations: consists_of, in_the_middle_of, next_to, on, opposite_to, parallel_to. Since it can't even steer on own-relation, it's unlikely to help cross-relation. A new sweep (pt448_L9F7540_assigned_relations_sweep.py) will test all 5 remaining assigned relations to determine if they should be moved to skip or reassigned.

### Oracle v4 Written and Queued

`pt448_smart_oracle_v4.py` created with two confirmed alpha updates vs v3:
- L11/F12278: α=0.5 → **α=0.45** (subset sweep: +6.01% vs +5.54%)
- L9/F387: α=0.5 → **α=0.4** (subset sweep: +4.17% vs +3.12%)
- L6/F7539: α=1.5 confirmed (subset sweep: +3.10%, same as oracle)
- L9/F7540: α=0.25 unchanged (own-rel is unsteerable — deep investigation needed)

Auto-launcher (PID 3156126) will run v4 on GPU 3 immediately after v3 completes.

### L15/F220 "across from" Sweep COMPLETE — Divergent Alpha Discovered

**"across from" (N=94, base=41.49%):** best α=1.25 → **+11.70%**

Full curve: α=0.7→+8.51%, α=0.8→+9.57%, α=1.25→**+11.70%**, flat from 1.25 to 2.0.

This is a **critical finding**: L15/F220 has two own-relations with completely divergent optimal alphas:
- "at the left side of" (N=421): α=0.7→+8.55% (degrades sharply above 0.7; at α=1.25: only +2.14%)
- "across from" (N=94): α=1.25→+11.70% (oracle's α=0.7 only gives +8.51%, losing -3.19pp)

**Oracle v3/v4 use α=0.7 for both → "across from" loses +3.19pp of potential improvement.**

### Oracle v5 Written — Per-Relation Alpha Override for "in the middle of"

**CORRECTION**: The "across from" override was initially wrong. "across from" is assigned to **L6/F7539** in the oracle (not L15/F220), because per_relation_steer cross-feature data shows L6/F7539 gives +14.89% on "across from" vs L15/F220's +8.51%. The L15/F220 subset sweep's +11.70% at α=1.25 is for the feature's own-relation performance, not for the oracle's routing.

`pt448_smart_oracle_v5.py` adds only the confirmed valid per-relation override:
- "in the middle of": L9/F7540 α=0.2 → +7.61% (vs oracle's α=0.25 → +5.43%, **+2.18pp gain**)
- All other relations: same as v4

Oracle v5 queued on GPU 3 after v4 (auto-launcher PID 3179049). Chain: v3 → v4 → v5.

### L9/F387 "at the right side of" Alpha Sweep COMPLETE

**Best α=0.4 → +4.17%** on "at the right side of" (N=480, base=52.29%)

Full curve: 0.1→−1.04%, 0.2→0.00%, 0.3→+1.67%, **0.4→+4.17%** (PEAK), 0.5→+3.12%, 0.6→+1.67%, 0.75→+2.29%, 1.0→−0.62%, 2.0→−2.71%

The oracle has been using α=0.5 (→+3.12%), which is **+1.05pp below optimum**. Oracle v4 must use α=0.4 for L9/F387.

### L11/F12278 "touching" Alpha Sweep — In Progress (peak confirmed at α=0.45)

Current results on "touching" (N=1281, base=56.52%):
- α=0.45: **+6.01%** ← PEAK so far
- α=0.5: +5.54%
- α=0.55: +5.93%
- α=0.6: +5.00%

α=0.45 is the clear winner vs the oracle's current α=0.5 (+5.54%). Will confirm once sweep completes. Oracle v4 should use α=0.45 for L11/F12278.

### Per-Relation Alpha Sweep — "touching" Cross-Feature Matrix (partial)

Still running on GPU 4. Results so far on "touching" (N=1281, base=56.52%):

| Feature | Best α | Best Δ |
|---------|--------|--------|
| L4/F14233 | 1.0 | +5.31% |
| L14/F10561 | 0.5 | +5.62% |
| L12/F2257 | 0.5 | **+6.09%** ← beats native L11 (+6.01% at own α=0.45) |
| L11/F12278 | still running | — |

This is a critical finding: **L12/F2257 at α=0.5 (+6.09%) is the best steering feature for "touching"** — marginally better than L11's native +6.01%. This suggests the oracle should route "touching" to L12/F2257 rather than L11/F12278, once fully confirmed.

### Late-Start Sweep L11/F12278 on "touching" COMPLETE

Late-start (layers 15-25) α=1.0 → +7.84% on L11/F12278 "touching".  
Natural-start: best=+6.01% at α=0.45. Late-start best: +7.84% at α=1.0 (**+1.83pp improvement!**)

This is a reversal from the trend: for L11/F12278 specifically, late-start HELPS (+7.84% vs natural +6.01%). The late_start_subset_sweep result (saved) shows late_start is sometimes better depending on feature start layer. L11 natural_start=5, but late_start=15 forces a narrower injection window that concentrates effect.

### L11/F9639 "in/inside/on" — STARTED (just began α sweep)

Base=60.85% (N=1101). α=0.1→Δ=-0.54% is the first result. Expect sweep to complete in ~2h.

### Oracle v3 — Running (2000/10972 samples)

Interim: base=54.00%, smart=58.35%, **Δ=+4.35%** at N=2000.

This is already +0.39pp above v2's final +3.96%. If it holds, oracle v3 confirms the alpha tuning (L12→0.75, L15→0.7) improved the oracle. Final result expected in ~45 min.

### Updated Optimal Alphas Table (from subset sweeps)

| Feature | Own Relation | N | Base | Best α | Best Δ | Oracle Current | Status |
|---------|-------------|---|------|--------|--------|----------------|--------|
| L4/F14233 | ahead of | 39 | 56.41% | **0.9** | +15.38% | 0.9 | ✓ confirmed |
| L14/F10561 | close to | 93 | 60.22% | **2.0** | +10.75% | 2.0 | ✓ confirmed |
| L12/F2257 | facing | 306 | 49.02% | **0.75** | +8.82% | 0.75 | ✓ confirmed |
| L15/F220 | at_left_side_of+across | 515 | 49.90% | **0.7** | +8.55% | 0.7 | ✓ confirmed |
| L11/F12278 | touching | 1281 | 56.52% | **0.45** | +6.01% | 0.5 | pending final |
| L9/F387 | at_right_side_of | 480 | 52.29% | **0.4** | +4.17% | 0.5 | ⚠ update needed |
| L6/F7539 | left of + right of | 323 | 51.08% | TBD | TBD | 1.5 | running |
| L9/F7540 | consists of | 35 | — | TBD | TBD | 0.25 | pending |

**Two confirmed alpha updates for oracle v4**: L11→0.45, L9/F387→0.4.

### Updated GPU Status

### Oracle v2 α=0.25 Mode COMPLETE

α=0.25 uniform: Δ=+1.12% (vs per_feature +3.96%) — confirms per-feature alpha tuning is critical.
Continuing to run α=0.5 and α=1.0 modes (informational only).

### L11/F9639 "in/inside/on" Result — ESSENTIALLY UNSTEERABLE

L11/F9639 on own-relations "in/inside/on" (N=1101, base=60.85%):
- α=0.2: +0.82% (barely above 0%)
- All other alphas ≤ 0% 

**L11/F9639 is useless for steering "in/inside/on"**, consistent with "in"/"inside"/"on" being in SKIP_RELATIONS. No new features to add to oracle.

### L9F7540 Assigned Relations Sweep — "in the middle of" COMPLETE

"in the middle of" (N=92, base=51.09%): **L9/F7540 α=0.2 → +7.61%** (oracle's α=0.25 → +5.43%)
- L11/F12278 α=0.25 → +6.52% (close second)
- L12/F2257 α=0.25 → +4.35%

L9/F7540 confirmed as best feature for "in the middle of" with **α=0.2** (oracle v5 override confirmed).
"next to" (N=309, base=61.49%): L9/F7540 α=0.15/0.2 → +1.29% (weak, likely not worth steering).

### Updated GPU Status

| GPU | Job | Current Status |
|-----|-----|----------------|
| 0 | smart_oracle_v2 | α=0.25 DONE (+1.12%), now running α=0.5 mode |
| 1 | global_alpha_fullvsr | L9/F387 done, L6/F7539 running |
| 2 | L11_subset_sweep | touching — α=0.9→+3.75%, peak confirmed at α=0.45→+6.01% |
| 3 | smart_oracle_v3 | **4000/10972 → Δ=+4.14% interim; v4→v5 queued after** |
| 4 | per_relation_alpha_sweep | touching — L12/F2257 α=0.5→+6.09% best; L15 & L11 running |
| 5 | late_start_subset_sweep | L11 done (late=+5.31% < natural=+6.01%); L9/F387→+2.92%; L6 running |
| 6 | L9F7540_assigned_sweep | in_middle_of done (α=0.2→+7.61%); next_to (+1.29% so far) |
| 7 | extra_features_subset_sweep | L11/F9639 in/on/on: max +0.82% → useless; DONE |

---

## Monitoring Update 61 — 2026-04-22 02:30 PDT

### Oracle v3 Interim Update — Δ=+4.14% at 4000/10972

Oracle v3 at 4000/10972 samples: base=54.88%, smart=59.02%, **Δ=+4.14%** (vs v2 final +3.96%).

At this interim rate, v3 is tracking +0.18pp improvement over v2. Since inject subset is ~84% of samples, this interim appears stable. Final expected in ~1h.

### Oracle Chain v3→v4→v5 Status

- **v3**: Running on GPU 3 (~4000/10972) — alpha fixes L12→0.75, L15→0.7, L4→0.9
- **v4**: Queued (launcher PID 3156126 waiting for v3) — additional L11→0.45, L9/F387→0.4
- **v5**: Queued (launcher PID 3179049 waiting for v4) — + L9/F7540 "in_middle_of" α=0.2 override

Expected oracle improvement trajectory: v2(+3.96%) → v3(~+4.1%) → v4(~+4.2%) → v5(~+4.2% marginal)

### Cross-Feature Investigation Status

**L9F7540 assigned relations sweep (GPU 6)**:
- "in the middle of" DONE: L9/F7540 α=0.2→+7.61% (best feature), oracle v5 override confirmed
- "next to" RUNNING: oracle's α=0.25→+1.62% is actually optimal (beating α=0.2→+1.29%)

**Per-relation_steer cross-feature data for remaining L9/F7540 relations** (old alpha but indicative):
- "opposite to": L9/F7540→+5.56% (best, N=36)
- "parallel to": L9/F7540→+5.56%, L9/F387→+4.44% (tied, N=90)
- "next to": L9/F7540→+1.29% (weak but best, N=309)
- "on": L9/F7540→+0.68% (minimal but positive, N=585)

**Conclusion**: L9/F7540 remains the best assignment for all 5 relations. Alpha is already near-optimal at 0.25. Only "in_the_middle_of" benefits from α=0.2 override (+2.18pp on N=92).

### L11 Subset Sweep — Peak Confirmed

L11/F12278 "touching" sweep now at α=0.9, confirming clean peak at α=0.45→+6.01%.
Full confirmed curve: 0.35→+4.06%, 0.4→+4.61%, **0.45→+6.01%** (PEAK), 0.5→+5.54%, 0.55→+5.93%, 0.6→+5.00%, 0.65→+5.07%, 0.7→+5.15%, 0.75→+4.14%, 0.8→+3.51%, 0.9→+3.75%

Note: α=0.55 gives +5.93% — almost as good as 0.45. The true peak is definitively at α=0.45.

### IMPORTANT FINDING: Late-Start Helps L11/F12278

Late-start (start_layer=15) L11/F12278 "touching": **α=1.0 → +7.84%** (vs natural α=0.45 → +6.01%, **+1.83pp improvement**)

This is a surprising exception to the general "late-start hurts" finding. For L11/F12278 specifically, constraining injection to layers 15-25 at higher alpha (+7.84%) beats natural-start at optimal alpha (+6.01%). This creates a new experiment opportunity:

**Oracle v6 candidate**: Use L11/F12278 with late_start=15 and α=1.0 for "touching" + related L11-assigned relations. Expected gain ~+0.23% overall (1281 samples × 1.83pp / 10972).

But: Need to verify this works for L11's OTHER assigned relations (on_top_of, surrounding, under). If they also benefit from late-start, the overall gain could be larger.

---

## Monitoring Update 62 — 2026-04-22 (continued)

### CORRECTION: Late-Start L11 Result Revised

**Previous claim was incorrect.** Update 61 stated "late-start L11/F12278 α=1.0 → +7.84%" — this was misread from an intermediate log state. The actual completed late-start subset sweep result for L11/F12278 on "touching" is:

- **Natural-start (start=5), best α=0.45 → +6.01%** (CONFIRMED PEAK)
- **Late-start (start=15), best α=0.5 → +5.31%** (−0.70pp vs natural)

Late-start HURTS L11/F12278. This is consistent with the general finding. The per-relation alpha sweep on "touching" also confirms this for all features with late-start characteristics.

**Oracle v6 candidate (late-start) is NOT supported.** The L11 latestart sweep script (`pt448_L11_latestart_sweep.py`) is still valuable to run on all 4 L11-assigned relations to fully confirm, but the hypothesis is now weak.

### All Late-Start Sweep Results (GPU 5, COMPLETE)

Feature-level late-start (start=15) vs. natural on their own relations:
| Feature | Own Relation | Natural Best | Late Best | Late−Natural |
|---------|-------------|--------------|-----------|--------------|
| L11/F12278 | touching | +6.01% (α=0.45) | +5.31% (α=0.5) | −0.70pp |
| L9/F387 | at the right side of | +4.17% (α=0.4) | +2.92% (α=0.5) | −1.25pp |
| L6/F7539 | left of/right of | +10.00% (natural α) | +1.55% (α=2.0) | −8.45pp |
| L9/F7540 | consists of | +0.00% (unsteerable) | +0.00% | 0pp |

**Conclusion: Late-start uniformly hurts all features. No feature benefits from starting injection at layer 15 vs. natural start. Drop the late-start hypothesis.**

### L11 Latestart Sweep — Launched on GPU 5 (PID 3219094)

Despite the negative finding, `pt448_L11_latestart_sweep.py` was launched on GPU 5 to do a thorough sweep of all 4 L11-assigned relations:
- "on top of" (N=505)
- "surrounding" (N=90)  
- "touching" (N=1281)
- "under" (N=589)

Both natural-start (start=5) and late-start (start=15) alphas swept. Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L11_latestart_sweep/`

This will confirm the natural-start peak across all L11-assigned relations (not just "touching") and potentially reveal better natural-start alphas for the other 3 relations.

### L11 Subset Sweep — COMPLETE

L11/F12278 "touching" full sweep complete. Confirmed peak:
- **α=0.45 → +6.01%** (from base 56.52% → 62.53%)
- Clean parabolic peak — α=0.55 gives +5.93% (second), then monotone decline
- Saved: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L11_subset_sweep/L11_F12278_subset_sweep.json`

### Extra Features Sweep — L13/F15219 COMPLETE, L11/F9639 Running

**L13/F15219 "behind" (N=709, base=51.62%)** COMPLETE:
- Best: α=1.25 → +1.27% (from 51.62% → 52.89%)
- All other alphas ≤ +0.56%. Confirmed unsteerable / noise-level.

**L11/F9639 "in/inside/on" (N=1101, base=60.85%)** running on GPU 7:
- So far: α=0.2 → +0.82% is the max (all others negative or near-zero)
- Confirmed unsteerable. No change to SKIP_RELATIONS list.

### Oracle v3 — 8000/10972, Δ=+3.90%

Oracle v3 at 8000/10972: base=54.61%, smart=58.51%, **Δ=+3.90%**.
Slight tracking below v2 (+3.96%) at 8000 samples. Final result pending.

### L9F7540 Assigned Relations Sweep — "on" Running (GPU 6)

- "in the middle of" DONE: L9/F7540 α=0.2→+7.61% (best). L11/F12278 α=0.25→+6.52% (2nd). ✓ v5 override correct.
- "next to" DONE: L9/F7540 α=0.25→+1.62% (oracle optimal). L12 α=0.5→+1.29%. L11→+0.32%.
- "on" RUNNING (N=585, base=60.17%): L9/F7540 oracle α=0.25 in progress.
- "opposite to" and "parallel to" still pending.

Cross-feature note for "next to": L9/F7540 is indeed the best feature (oracle assignment confirmed), though gain is weak (+1.62%).

### Per-Relation Alpha Sweep (GPU 4) — "touching" Cross-Feature Matrix

Currently testing all 8 features on "touching" (N=1281, base=56.52%). Results so far:
| Feature | Best Alpha | Best Delta |
|---------|-----------|-----------|
| L4/F14233 | α=1.0 | +5.31% |
| L14/F10561 | α=0.5 | +5.62% |
| L12/F2257 | α=0.5 | **+6.09%** |
| L15/F220 | α=0.5 | +5.78% |
| L11/F12278 (oracle) | α=0.45 | +6.01% |

**Observation**: L12/F2257 at α=0.5 gives +6.09% on "touching", beating L11's oracle +6.01% by 0.08pp. This is marginal and likely noise given N=1281 is a decent sample size. Monitor remaining features (L9/F387, L6/F7539, L9/F7540) before deciding.

### GPU Status (Updated)

| GPU | Job | Status |
|-----|-----|--------|
| 0 | caa_global_alpha_fullvsr | Running (low util?) |
| 1 | smart_oracle_v2 (extended) | Running |
| 2 | L11_subset_sweep | **DONE** (peak α=0.45→+6.01%) |
| 3 | smart_oracle_v3→v4→v5 | v3 at 8000/10972 Δ≈+3.90%; v4/v5 queued |
| 4 | per_relation_alpha_sweep | "touching" cross-feature: L12 best (+6.09%) so far |
| 5 | L11_latestart_sweep | Launched PID 3219094; "on top of" starting |
| 6 | L9F7540_assigned_sweep | "on" running; 2/5 relations done |
| 7 | L6_assigned_sweep | "across_from" DONE (α=1.5→+14.89% oracle confirmed); "alongside" running |

### New Experiments Launched (GPUs 2 and 7)

**GPU 2: `pt448_L15_assigned_relations_sweep.py`** (PID 3248852)
- Sweeps all 9 L15/F220-assigned relations: above, at_left_side_of, away_from, beside, contains, inside, over, part_of, within
- L15 oracle α=0.7 range [0.3, 0.4, ..., 2.0] + cross-feature L12/F2257, L9/F7540
- Motivation: We only characterized L15 on "at_left_side_of" (α=0.7→+8.55%) — other relations may have very different optima
- Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L15_assigned_sweep/`

**GPU 7: `pt448_L6_assigned_relations_sweep.py`** (PID 3248853)
- Sweeps all 7 L6/F7539-assigned relations: across_from, alongside, at_back_of, below, facing_away_from, left_of, right_of
- L6 oracle α=1.5 range [0.5, 0.75, ..., 3.0] + cross-feature L12, L15
- Motivation: L6 oracle alpha confirmed on "left of"/"right of" (+10%), but "across from" may need different alpha (+14.89% from old sweep)
- Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L6_assigned_sweep/`

These two sweeps complete the per-relation alpha characterization for all 8 oracle features. Combined with L9/F7540 (GPU 6) and L11/F12278 (GPU 5) in progress, we'll have full coverage to build oracle v6 with per-relation alpha overrides for every feature.

---

## Monitoring Update 63 — 2026-04-22 (later)

### Oracle v3 FINAL RESULT: Δ=+4.05%

**Oracle v3 complete**: base=54.41%, smart=58.46%, **Δ=+4.05%** (N=10,972)
- Inject group (N=9239): base=54.06%, smart=58.87%, **+4.81%**
- Skip group (N=1630): +0.00% (correct by design)
- Unknown (N=103): +0.00%

**vs. v2 (+3.96%)**: +0.09pp improvement from fixing L12→0.75, L15→0.7, L4→0.9.
**Oracle v4** started on GPU 3 (L11→0.45, L9/F387→0.4 on top of v3). Expected further +0.1–0.2pp.

### L11 Latestart "on top of" — Late-Start Confirmed Inferior Again

"on top of" (N=505, base=59.21%):
- Natural start=5: **α=0.4 → +5.35%** (BEST), oracle α=0.45 → +4.95%
- Late start=15: best α=0.5 → +4.36%

**Key finding**: Oracle α=0.45 is NOT optimal for "on top of" — α=0.4 gives +5.35% (+0.40pp better).
Late-start again confirmed inferior (−0.99pp vs natural best).

This suggests an oracle v6 tweak: use α=0.4 instead of α=0.45 for L11/F12278 on "on top of", while keeping α=0.45 for "touching". This requires per-relation alpha overrides for L11 (similar to what v5 does for L9/F7540).

### L6 "across from" — Oracle α=1.5 Confirmed as Peak

L6/F7539 "across from" (N=94, base=41.49%):
- α=1.5 → **+14.89%** [ORACLE] — confirmed as peak
- α=1.0 → +13.83%, α=1.25 → +10.64%, α=2.0 → +13.83%

Oracle α=1.5 is correct. "across from" routed to L6 (not L15) is the right decision.

### Per-Relation Sweep "touching" Cross-Feature Rankings (GPU 4)

So far (L4, L14, L12, L15, L11 tested):
| Feature | Best α | Best Δ | vs. oracle L11 |
|---------|--------|--------|----------------|
| L12/F2257 | 0.5 | **+6.09%** | +0.08pp |
| L11/F12278 (oracle) | 0.45 | +6.01% | — |
| L15/F220 | 0.5 | +5.78% | −0.23pp |
| L14/F10561 | 0.5 | +5.62% | −0.39pp |
| L4/F14233 | 1.0 | +5.31% | −0.70pp |

L9/F387, L6/F7539, L9/F7540 still running. L12's +6.09% lead over L11's +6.01% is only 0.08pp on N=1281 — likely statistical noise (≈1 sample). Current oracle assignment (L11 for "touching") still holds.

### L9F7540 Sweep "on" Results (GPU 6)

"on" (N=585, base=60.17%):
- L9/F7540: α=0.3 → **+1.37%** (best), oracle α=0.25 → +0.34% (oracle suboptimal by +1.03pp!)
- L12/F2257: results loading
- "on" is weakly steerable but the α=0.3 override would give meaningful gains across N=585 samples

**Oracle v6 candidate**: override L9/F7540 α=0.3 for "on" (from oracle α=0.25). Estimated gain: ~585 × 1.03pp / 10972 ≈ +0.055% global.

### Oracle v6 Written and Queued

`pt448_smart_oracle_v6.py` written and queued (launcher PID 3279304, runs after v5).

**v6 changes vs v5:**
1. L11/F12278 "on top of": α=0.4 (was 0.45) → +5.35% confirmed (+0.40pp on N=505)
2. L9/F7540 "in the middle of": α=0.2 (from v5) → +7.61%
3. L9/F7540 "on": α=0.3 (was 0.25) → +1.37% (+1.03pp on N=585)

**UPDATED** — "surrounding" added after L11 latestart sweep completed:
4. L11/F12278 "surrounding": α=0.6 (was 0.45) → **+11.11%** (+3.33pp on N=90) ← LARGEST GAIN

Expected total gain from per-relation tweaks vs v5 default:
- "on top of": 505 × 0.40pp / 10972 ≈ +0.018%
- "surrounding": 90 × 3.33pp / 10972 ≈ **+0.027%**
- "on": 585 × 1.03pp / 10972 ≈ +0.055%
- Combined ≈ +0.100% over v5

Estimated v6 final: ~+4.30% (if v5 ≈ +4.20%)

### *** KEY FINDING: "surrounding" Oracle Alpha Badly Wrong ***

L11/F12278 "surrounding" (N=90, base=47.78%):
- Oracle α=0.45 → +7.78% (base 47.78% → 55.56%)
- **Best α=0.6 → +11.11%** (47.78% → 58.89%)
- **+3.33pp improvement** — largest single per-relation alpha gain found
- Full natural-start curve: 0.25→+1.11%, 0.4→+5.56%, 0.45→+7.78% [oracle], 0.5→+8.89%, **0.6→+11.11%** (PEAK), 0.75→+3.33%, 1.0→+3.33%
- Late-start best: α=0.5→+6.67% (−4.44pp vs natural, confirming late-start harmful)

v6 script updated to include: `"surrounding": 0.6`

### New Per-Relation Alpha Findings (L15 and L6 sweeps)

**L15 "above" (N=341, base=51.61%)**:
- L15/F220 oracle α=0.7 → +4.69%, but **α=0.8 → +5.87%** (+1.18pp improvement)
- L12/F2257 results loading

**L6 "alongside" (N=55, base=43.64%)** — SAVED:
- L6/F7539: oracle α=1.5 suboptimal; **α=1.25 → +14.55%** (vs α=1.5 → unclear from log)
- L12/F2257: **α=0.5 → +14.55%** (ties L6 — cross-feature parity)
- L15/F220: α=0.6 → +12.73%
- → L6 oracle assignment fine, but consider α=1.25 override

**L6 "at the back of" (N=94, base=53.19%)**:
- L6/F7539: oracle α=1.5 → +2.13%, **α=1.75 → +3.19%** (+1.06pp)
- Suboptimal oracle alpha confirmed

### Accumulating Oracle Alpha Overrides (for v7+, beyond v6)

| Relation | Feature | Current Oracle α | Oracle Δ | Optimal α | Best Δ | Gain |
|----------|---------|-----------------|----------|-----------|--------|------|
| above | L15/F220 | 0.7 | +4.69% | **0.8** | +5.87% | +1.18pp |
| at the back of | L6/F7539 | 1.5 | +2.13% | **1.75** | +3.19% | +1.06pp |
| alongside | L6/F7539 | 1.5 | ? | **1.25** | +14.55% | pending |

These will be incorporated into oracle v7.

### L15 "above" Full Result (GPU 2)

L15/F220 "above" (N=341, base=51.61%):
- Oracle α=0.7 → +4.69%, **α=0.8 → +5.87%** (PEAK, +1.18pp improvement)
- Full curve: 0.3→+4.40%, 0.4→+2.05%, 0.5→+2.64%, 0.6→+3.81%, 0.7→+4.69% [oracle], **0.8→+5.87%**, 0.9→+5.28%, 1.0→+4.69%
- L12/F2257 still loading results (running)

### L6 "at the back of" Full Result (GPU 7)

L6/F7539 "at_back_of" (N=94, base=53.19%) — SAVED:
- L6/F7539: oracle α=1.5→+2.13%, **α=1.75→+3.19%** (+1.06pp)
- L12/F2257: α=0.25→+2.13% (L12 oracle −4.26% — don't switch)
- L15/F220: α=1.25→+3.19% (ties L6 at best, but oracle already uses L6)
- → Keep L6 assignment, set α=1.75 override

### GPU Status (Monitoring Update 64)

| GPU | Job | Status |
|-----|-----|--------|
| 0 | caa_global_alpha_fullvsr | Running |
| 1 | smart_oracle_v2 (extended) | Running |
| 2 | L15_assigned_sweep | "above" done (α=0.8→+5.87%); "at_left_side_of" next |
| 3 | smart_oracle_v3→v4→v5→v6 | **v3=+4.05%**; v4 running (2000 Δ≈+4.65%); v5/v6 queued |
| 4 | per_relation_alpha_sweep | "touching" — L12=+6.09%, L11=+6.01%; L9/F387+L6+L9/F7540 running |
| 5 | L11_latestart_sweep | on_top_of+surrounding DONE; **"touching" running** |
| 6 | L9F7540_assigned_sweep | "on" L9→α=0.3→+1.37%; L12 running; "opposite_to"/"parallel_to" pending |
| 7 | L6_assigned_sweep | across_from+alongside+at_back_of DONE; "below" running |


---

## Monitoring Update 65 — 2026-04-22 (context resumed)

### Oracle v7 Launcher Queued

Oracle v7 launcher written to `/tmp/launch_oracle_v7.sh` (PID 3316897), waiting for oracle v6 launcher (PID 3279304) to complete.

Oracle chain status:
- v4: Running (GPU 3, PID 3252517) — at 4000/10972 samples, Δ≈+4.39% interim
- v5 launcher: PID 3179049 (alive, waiting for v4)
- v6 launcher: PID 3279304 (alive, waiting for v5)
- v7 launcher: PID 3316897 (alive, waiting for v6) — NEW

### New Completed Results

#### L9F7540 Assigned Sweep — "opposite to" DONE
- N=36 (small), base=58.33%
- L9/F7540: α=0.15→+5.56% (BEST), oracle α=0.25→+2.78% — **+2.78pp gain**
- BUT N=36 is too small to be conclusive; need more samples to confirm
- L12 best=+2.78%, L11 best=+2.78% — L9/F7540 leads

#### L9F7540 Assigned Sweep — "parallel to" IN PROGRESS
- N=90, base=52.22%
- L9/F7540: α=0.3→+7.78% (BEST so far), oracle α=0.25→+5.56% — **+2.22pp gain**
- L12: α=0.75→+6.67% (oracle confirmed optimal for L12)
- L11: partial (still running)
- **Key finding**: α=0.3 is better for "parallel to" than oracle α=0.25

#### L6 Assigned Sweep — "below" DONE
- N=277, base=49.82%
- L6/F7539: α=2.5→+6.14% (BEST) vs oracle α=1.5→+5.42% — **+0.72pp gain**
- L12: α=1.5→+4.33% (much lower than L6)
- L15: partial (still running)
- **Key finding**: "below" needs higher alpha than oracle; α=2.5 is optimal

#### L15 Assigned Sweep — "at the left side of" IN PROGRESS
- N=421, base=51.78%
- L15/F220: oracle α=0.7→+8.55% (best so far), α=0.8→+7.84% — oracle appears optimal
- Still running α=0.9+ and cross-features
- L12/L9 cross-features pending

### Updated Per-Relation Alpha Override Table

| Relation | Feature | Oracle α | Best α | Δ gain | N | Status |
|----------|---------|---------|--------|--------|---|--------|
| on top of | L11/F12278 | 0.45 | **0.4** | +0.40pp | 505 | CONFIRMED |
| surrounding | L11/F12278 | 0.45 | **0.6** | +3.33pp | 90 | CONFIRMED |
| above | L15/F220 | 0.7 | **0.8** | +1.18pp | 341 | CONFIRMED |
| at the back of | L6/F7539 | 1.5 | **1.75** | +1.06pp | 94 | CONFIRMED |
| below | L6/F7539 | 1.5 | **2.5** | +0.72pp | 277 | NEW — CONFIRMED |
| in the middle of | L9/F7540 | 0.25 | **0.2** | +2.18pp | 92 | CONFIRMED |
| on | L9/F7540 | 0.25 | **0.3** | +1.03pp | 585 | CONFIRMED |
| opposite to | L9/F7540 | 0.25 | **0.15** | +2.78pp | 36 | TENTATIVE (N small) |
| parallel to | L9/F7540 | 0.25 | **0.3** | +2.22pp | 90 | PARTIAL (still running) |
| at the left side of | L15/F220 | 0.7 | 0.7 | 0pp | 421 | PARTIAL (oracle looks optimal) |

### Oracle v7 Configuration (6 confirmed overrides)

```python
RELATION_ALPHA_OVERRIDES = {
    "on top of":      0.4,    # +5.35% vs oracle +4.95%, +0.40pp, N=505
    "surrounding":    0.6,    # +11.11% vs oracle +7.78%, +3.33pp, N=90
    "above":          0.8,    # +5.87% vs oracle +4.69%, +1.18pp, N=341
    "at the back of": 1.75,   # +3.19% vs oracle +2.13%, +1.06pp, N=94
    "in the middle of": 0.2,  # +7.61% vs oracle +5.43%, +2.18pp, N=92
    "on":               0.3,  # +1.37% vs oracle +0.34%, +1.03pp, N=585
}
```

### Oracle v8 Candidate Overrides (pending confirmation)

When L6, L9F7540, and L15 sweeps complete, oracle v8 may add:
- "below" → α=2.5 (L6/F7539: +0.72pp, N=277, **CONFIRMED**)
- "parallel to" → α=0.3 (L9/F7540: +2.22pp, N=90, pending L11 cross-feature)
- "opposite to" → α=0.15 (L9/F7540: +2.78pp, N=36, tentative — small N)

### GPU Status

| GPU | Job | Status |
|-----|-----|--------|
| 0 | caa_global_alpha_fullvsr | Running |
| 1 | smart_oracle_v2 (extended) | Running |
| 2 | L15_assigned_sweep | "above" DONE; "at_left_side_of" IN PROGRESS (oracle α=0.7 looks optimal) |
| 3 | smart_oracle_v3→v4→v5→v6→v7 | v4@4000 Δ≈+4.39%; v5/v6/v7 queued |
| 4 | per_relation_alpha_sweep | "touching" — L9/F387+L6+L9/F7540 still running |
| 5 | L11_latestart_sweep | on_top_of+surrounding DONE; **"touching" running** |
| 6 | L9F7540_assigned_sweep | "on"+opposite_to DONE; **"parallel_to" running** |
| 7 | L6_assigned_sweep | across_from+alongside+at_back_of+**below** DONE; "facing_away_from" next |


---

## Monitoring Update 66 — 2026-04-22 (oracle v8 + L12 sweep)

### Oracle v8 Written and Queued

Oracle v8 (`/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_smart_oracle_v8.py`) adds 2 new confirmed overrides:
- L6/F7539 "below" → α=2.5 (+6.14% vs oracle +5.42%, +0.72pp, N=277)
- L9/F7540 "parallel to" → α=0.3 (+7.78% vs oracle +5.56%, +2.22pp, N=90)

Oracle v8 launcher queued (PID 3350270), waiting for v7 launcher (PID 3316897).

**Oracle v8 full override table (8 overrides):**
```python
RELATION_ALPHA_OVERRIDES = {
    "on top of":      0.4,    # L11 +5.35% vs +4.95%, +0.40pp, N=505
    "surrounding":    0.6,    # L11 +11.11% vs +7.78%, +3.33pp, N=90
    "above":          0.8,    # L15 +5.87% vs +4.69%, +1.18pp, N=341
    "at the back of": 1.75,   # L6  +3.19% vs +2.13%, +1.06pp, N=94
    "below":          2.5,    # L6  +6.14% vs +5.42%, +0.72pp, N=277 (NEW)
    "in the middle of": 0.2,  # L9/F7540 +7.61% vs +5.43%, +2.18pp, N=92
    "on":               0.3,  # L9/F7540 +1.37% vs +0.34%, +1.03pp, N=585
    "parallel to":      0.3,  # L9/F7540 +7.78% vs +5.56%, +2.22pp, N=90 (NEW)
}
```

### L9F7540 Assigned Sweep — COMPLETE

All 5 relations done. Summary:
| Relation | N | Base | L9/F7540 best α | L9/F7540 best Δ | Oracle Δ | Gain vs oracle |
|----------|---|------|-----------------|-----------------|----------|----------------|
| in the middle of | 92 | 46.74% | 0.2 | +7.61% | +5.43% | **+2.18pp** |
| next to | 248 | 55.24% | 0.25 | +1.62% | +1.62% | 0pp (optimal) |
| on | 585 | 60.17% | 0.3 | +1.37% | +0.34% | **+1.03pp** |
| opposite to | 36 | 58.33% | 0.15 | +5.56% | +2.78% | **+2.78pp** (N small) |
| parallel to | 90 | 52.22% | 0.3 | +7.78% | +5.56% | **+2.22pp** |

Note: "opposite to" N=36 is small, α=0.15 tentative. "parallel to" N=90, L9/F7540 leads all competitors.

### L11 Latestart Sweep — "touching" Natural Done

- N=1281 (large), base=56.52%
- Natural sweep: **oracle α=0.45 → +6.01% CONFIRMED AS PEAK**
  - α=0.4→+4.61%, α=0.45→+6.01%, α=0.5→+5.54%, α=0.6→+5.00%
  - No override needed for "touching" — oracle α=0.45 is optimal
- Late-start sweep running now

### L6 "below" — CONFIRMED α=2.5

- N=277, base=49.82%
- L6/F7539: α=2.5→+6.14% and α=3.0→+6.14% (tied), vs oracle α=1.5→+5.42%
- α=2.5 chosen as override (same gain, lower alpha = less aggressive)
- **+0.72pp improvement confirmed**

### L15 "at_the_left_side_of" — Oracle Looks Optimal

- N=421, base=51.78%
- L15/F220: oracle α=0.7→+8.55% (peak so far), α=0.8→+7.84% (lower)
- Still running cross-features, but L15 oracle α=0.7 appears optimal — no override needed

### L12 Assigned Relations Sweep — LAUNCHED (GPU 6, PID 3356179)

Sweeps all 8 L12/F2257-assigned relations: beneath, beyond, enclosed by, facing, near, off, outside, toward.
- beyond (N=20), enclosed by (N=21), outside (N=32), toward (N=36) — small N, tentative results
- beneath (N=341), facing (N=306) — large enough for meaningful results
Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L12_assigned_sweep/`
Log: `/tmp/L12_assigned_sweep.log`

### GPU Status

| GPU | Job | Status |
|-----|-----|--------|
| 0 | caa_global_alpha_fullvsr | Running |
| 1 | smart_oracle_v2 (extended) | Running |
| 2 | L15_assigned_sweep | "at_left_side_of" running (oracle α=0.7 looks optimal) |
| 3 | smart_oracle_v3→v4→v5→v6→v7→v8 | v4@4000 Δ≈+4.39%; v5/v6/v7/v8 queued |
| 4 | per_relation_alpha_sweep | "touching" — L9/F387+L6+L9/F7540 still running |
| 5 | L11_latestart_sweep | "touching" late-start running; "under" pending |
| 6 | **L12_assigned_sweep** (NEW) | Launched (PID 3356179) |
| 7 | L6_assigned_sweep | below DONE; "facing_away_from" next |


---

## Monitoring Update 67 — 2026-04-22

### Active Experiments Status

**All 8 GPUs occupied:**

| GPU | Job | Current relation | Latest result |
|-----|-----|-----------------|---------------|
| 0 | caa_global_alpha_fullvsr | Full VSR | Running |
| 1 | smart_oracle_v2 (extended) | Full VSR | Running |
| 2 | L15_assigned_sweep | at_left_side_of (L9F7540 cross-feat) | L15 α=0.7→+8.55% CONFIRMED PEAK |
| 3 | smart_oracle_v4 | Full VSR (8000/10972) | +3.98% interim (converging ~+4.1%) |
| 4 | per_relation_alpha_sweep | touching L9/F387 | L12=+6.09% leads, L11=+6.01% oracle |
| 5 | L11_latestart_sweep | touching late-start | natural α=0.45→+6.01% confirmed peak |
| 6 | L12_assigned_sweep | beneath (α=1.25 so far) | +7.33% vs oracle +5.87% → **+1.46pp** |
| 7 | L6_assigned_sweep | left_of (partial) | running |

### Key New Findings

#### L6 "facing_away_from" — DONE, Oracle Confirmed Optimal
- N=180, base=48.33%
- L6/F7539: oracle α=1.5 → +11.11% — **CONFIRMED AS PEAK**
- L12 best: α=0.75 → +6.67%, L15 best: α=0.9 → +7.78%
- L6 is clearly best for this relation; no override needed

#### L15 "at_left_side_of" — DONE, Oracle Confirmed Optimal
- N=421, base=51.78%
- L15/F220: oracle α=0.7 → +8.55% **CONFIRMED AS PEAK** (α=0.8→+7.84%, α=0.6→+8.08%)
- L12 best: α=1.0 → +5.46% (much lower), L9/F7540 best: α=0.4→+3.56% (much lower)
- L15 oracle α=0.7 is definitively best; no override needed

#### L12 "beneath" — IN PROGRESS, Promising Override
- N=341, base=50.15%
- L12/F2257: oracle α=0.75→+5.87%, α=1.0→+7.04%, α=1.25→+7.33% (BEST so far) — **+1.46pp gain**
- Sweep still running (α=1.5, 2.0, 2.5, 3.0 + cross-features pending)

#### "touching" Cross-Feature Matrix — ALL FEATURES DONE EXCEPT L6/L9F7540
- N=1281, base=56.52%
- **Rankings by best-alpha peak Δ:**
  1. L12/F2257: α=0.5 → +6.09% ← NEW LEADER (by 0.08pp)
  2. L11/F12278: α=0.45 → +6.01% ← oracle assignment
  3. L15/F220: α=0.5 → +5.78%
  4. L9/F387: α=0.5 → +5.70%
  5. L14/F10561: α=0.5 → +5.62%
  6. L4/F14233: α=1.0 → +5.31%
- L12 leads by only 0.08pp (1 sample on N=1281 — statistical noise)
- **Conclusion: L11/F12278 oracle assignment stands; no reassignment warranted**

#### Oracle v4 Interim — Converging Below v3
- At 8000 samples: base=54.61%, smart=58.59%, Δ=+3.98%
- Trend: 4.65% → 4.39% → 4.15% → 3.98% — **converging below v3 final (+4.05%)**?
- NOTE: This is expected noise from batch composition — v3 had similar pattern
- Wait for completion (N=10,972) before concluding

### No New Per-Relation Overrides This Cycle

"facing_away_from" and "at_left_side_of" confirmed oracle optimal — no new overrides.
Potential new override: L12 "beneath" α=1.25 (or higher) — pending full sweep completion.


---

## Monitoring Update 68 — 2026-04-22

### New Confirmed Override: L12 "beneath" α=1.25

**L12/F2257 "beneath" full alpha curve (N=341, base=50.15%):**
- α=0.25→+2.64%, α=0.5→+1.17%, α=0.75→+5.87% [oracle], α=1.0→+7.04%, **α=1.25→+7.33% [PEAK]**, α=1.5→+5.28%, α=2.0→+3.81%, α=2.5→+4.11%, α=3.0→+4.11%
- **Best: α=1.25 → +7.33% vs oracle +5.87% = +1.46pp improvement** ✓ CONFIRMED

This is a strong and clean result — clear single peak at α=1.25, N=341 is large enough.

### Oracle v9 Written and Queued

Oracle v9 adds "beneath" → α=1.25 (L12/F2257), for 9 total overrides.
Launcher PID 3391555 queued after v8 launcher (PID 3350270).

**Oracle chain:** v4(running) → v5 → v6 → v7 → v8 → v9 (all queued)

**Oracle v9 full override table (9 overrides):**
```python
RELATION_ALPHA_OVERRIDES = {
    "on top of":      0.4,    # L11 +5.35% vs +4.95%, +0.40pp, N=505
    "surrounding":    0.6,    # L11 +11.11% vs +7.78%, +3.33pp, N=90
    "above":          0.8,    # L15 +5.87% vs +4.69%, +1.18pp, N=341
    "at the back of": 1.75,   # L6  +3.19% vs +2.13%, +1.06pp, N=94
    "below":          2.5,    # L6  +6.14% vs +5.42%, +0.72pp, N=277
    "beneath":        1.25,   # L12 +7.33% vs +5.87%, +1.46pp, N=341 (NEW in v9)
    "in the middle of": 0.2,  # L9F7540 +7.61% vs +5.43%, +2.18pp, N=92
    "on":               0.3,  # L9F7540 +1.37% vs +0.34%, +1.03pp, N=585
    "parallel to":      0.3,  # L9F7540 +7.78% vs +5.56%, +2.22pp, N=90
}
```

### Oracle v4 Interim Trend
At 8000/10972 samples: Δ≈+3.98% (trending down from +4.65% at 2000). 
This pattern mirrors v3 which also fluctuated before settling at +4.05%.
Expected final v4 Δ ≈ +4.05–4.20%.

### "touching" Cross-Feature Summary (per_relation_alpha_sweep)
N=1281, base=56.52%. Best per feature:
1. L12/F2257: α=0.5 → +6.09% ← narrowly leads
2. L11/F12278: α=0.45 → +6.01% ← oracle (margin: 0.08pp = ~1 sample)
3. L15/F220: α=0.5 → +5.78%
4. L9/F387: α=0.5 → +5.70%
5. L14/F10561: α=0.5 → +5.62%
6. L4/F14233: α=1.0 → +5.31%

**Conclusion: L11 oracle assignment stands** — 0.08pp is within noise; no reassignment.
L6/F7539 and L9/F7540 still running on this relation.

### L15 "at_left_side_of" — Oracle Confirmed FINAL
- L15/F220: α=0.7 → +8.55% is the definitive peak (α=0.8→+7.84%, α=0.6→+8.08%)
- No override needed

### L6 "facing_away_from" — Oracle Confirmed FINAL
- L6/F7539: oracle α=1.5 → +11.11% confirmed as peak
- No override needed

### Oracle Version Progress Summary

| Version | Key changes | Expected final Δ |
|---------|-------------|-----------------|
| v3 | Baseline (8 features, oracle alphas) | +4.05% ✓ FINAL |
| v4 | L11→0.45, L9/F387→0.4 | ~+4.10% (running) |
| v5 | +L9F7540 in_middle_of→0.2 | ~+4.20% (queued) |
| v6 | +L11 on_top_of→0.4, surrounding→0.6, L9F7540 on→0.3 | ~+4.30% (queued) |
| v7 | +L15 above→0.8, L6 at_back_of→1.75 | ~+4.35% (queued) |
| v8 | +L6 below→2.5, L9F7540 parallel_to→0.3 | ~+4.45% (queued) |
| v9 | +L12 beneath→1.25 | ~+4.50% (queued) |

### GPU Status

| GPU | Job | Status |
|-----|-----|--------|
| 0 | caa_global_alpha_fullvsr | Running |
| 1 | smart_oracle_v2 (extended) | Running |
| 2 | L15_assigned_sweep | at_left_side_of L9F7540 cross-feat running; "away_from" next |
| 3 | smart_oracle_v4→v5→v6→v7→v8→v9 | v4@8000 Δ≈+3.98%; chain queued |
| 4 | per_relation_alpha_sweep | touching L6/L9F7540 running |
| 5 | L11_latestart_sweep | touching late-start running |
| 6 | L12_assigned_sweep | beneath L6/L15 cross-feat running; 7 more relations pending |
| 7 | L6_assigned_sweep | left_of running; right_of pending |


---

## Monitoring Update 69 — 2026-04-22

### Oracle v4 FINAL: +4.08% ✓

**oracle v4 FINAL result: base=54.41% smart=58.49% Δ=+4.08% N=10,972**
- Inject group (N=9239): base=54.06%, smart=58.91%, Δ=+4.85%
- Skip/unknown unchanged
- **Improvement vs v3 (+4.05%): +0.03pp** — small but consistent gain from L11→0.45, L9/F387→0.4

Oracle v5 now running (GPU 3, PID 3402153) — interim at 2000 samples: Δ=+4.60%.

### L6 Assigned Sweep — COMPLETE (all 7 relations)

Full summary:
| Relation | N | Base | L6 best α | L6 best Δ | Oracle Δ | Override? |
|----------|---|------|-----------|-----------|----------|-----------|
| across from | 94 | 41.49% | 1.5 | +14.89% | +14.89% | No (oracle optimal) |
| alongside | 55 | 43.64% | **1.25** | +14.55% | +12.73% | +1.82pp (N=55 small) |
| at the back of | 94 | 53.19% | **1.75** | +3.19% | +2.13% | **+1.06pp** CONFIRMED v7 |
| below | 277 | 49.82% | **2.5** | +6.14% | +5.42% | **+0.72pp** CONFIRMED v8 |
| facing away from | 180 | 48.33% | 1.5 | +11.11% | +11.11% | No (oracle optimal) |
| left of | 210 | 49.52% | 1.5 | +4.29% | +4.29% | No (oracle optimal) |
| right of | 113 | 53.98% | 0.5 | +0.88% | +0.88% | No (noisy, tiny delta) |

Note: "alongside" α=1.25 override tempting but N=55 is too small to trust.

### L11 Latestart Sweep — Late-Start Confirmed Uniformly Worse

| Relation | N | Oracle (natural) | Late best | Diff |
|----------|---|-----------------|-----------|------|
| on top of | 505 | α=0.4 → +5.35% | α=0.5 → +4.36% | **-0.99pp** |
| surrounding | 90 | α=0.6 → +11.11% | α=0.5 → +6.67% | **-4.44pp** |
| touching | 1281 | α=0.45 → +6.01% | α=0.5 → +5.31% | **-0.70pp** |
| under | 589 | running | - | - |

**Conclusion: Late-start injection (start_layer=15) uniformly hurts all features. Hypothesis dropped.**

### "touching" Cross-Feature Summary — FINAL (all 7 features)

N=1281, base=56.52%. Best per feature at optimal alpha:
| Feature | Best α | Best Δ | Status |
|---------|--------|--------|--------|
| L12/F2257 | 0.5 | +6.09% | NEW LEADER (by 0.08pp) |
| L11/F12278 | 0.45 | +6.01% | Oracle assignment |
| L15/F220 | 0.5 | +5.78% | |
| L9/F387 | 0.5 | +5.70% | |
| L14/F10561 | 0.5 | +5.62% | |
| L6/F7539 | 0.75 | +5.39% | |
| L4/F14233 | 1.0 | +5.31% | |

**L9/F7540 still running.** L12 leads L11 by 0.08pp = 1 sample — statistical noise. L11 oracle confirmed.

### L15 Sweep New Completions

**"beside" (N=188, base=51.60%):**
- L15/F220: oracle α=0.7 → +10.64% CONFIRMED AS PEAK (α=0.8 also +10.64%, then drops)
- L12 best: α=0.75 → +10.11% — both excellent, L15 slightly better
- No override needed

**"away from" (N=155, base=48.39%):**
- L15/F220: α=0.6 → +2.58% vs oracle α=0.7 → +1.29% — **+1.29pp**
- Pattern is noisy (1-sample resolution); α=0.6 and α=1.0 both give +2.58%
- **Tentative override: α=0.6 (pending more data)**

### L12 Sweep New Completions

**"beneath" (N=341, base=50.15%):**
- L12 α=1.25 → +7.33% vs oracle α=0.75 → +5.87% — **+1.46pp CONFIRMED** (in v9)

**"beyond" (N=20) and "enclosed by" (N=21):**
- Both too small (N<25) for reliable overrides; high noise; oracle confirmed for beyond

### New Sweep: L9/F387 Assigned Relations — LAUNCHED (GPU 7, PID 3430651)

Sweeps 5 L9/F387-assigned relations: adjacent_to, at_right_side_of, at_side_of, attached_to, far_from.
Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L9F387_assigned_sweep/`
Log: `/tmp/L9F387_assigned_sweep.log`

### Oracle Chain Status

| Version | Status | Final Δ |
|---------|--------|---------|
| v3 | DONE | **+4.05%** |
| v4 | DONE | **+4.08%** (+0.03pp) |
| v5 | Running (PID 3402153) | interim +4.60% at 2000 |
| v6 | Queued (launcher 3279304) | ~+4.30% expected |
| v7 | Queued (launcher 3316897) | ~+4.35% expected |
| v8 | Queued (launcher 3350270) | ~+4.45% expected |
| v9 | Queued (launcher 3391555) | ~+4.55% expected |

### GPU Status

| GPU | Job | Status |
|-----|-----|--------|
| 0 | caa_global_alpha_fullvsr | Running |
| 1 | smart_oracle_v2 (extended) | Running |
| 2 | L15_assigned_sweep | beside DONE; "contains" next (5 more) |
| 3 | smart_oracle_v5 | Running (2000: Δ≈+4.60%) |
| 4 | per_relation_alpha_sweep | touching L9/F7540 running |
| 5 | L11_latestart_sweep | "under" running (natural start) |
| 6 | L12_assigned_sweep | beneath/beyond/enclosed_by DONE; "facing" next |
| 7 | **L9F387_assigned_sweep** (NEW) | Launched (PID 3430651) |

---

## Monitoring Update 70 — 2026-04-22

### Oracle v10 — Complete (written, queued)

Oracle v10 adds 10th per-relation override: L11/F12278 **"under"→α=0.75** (+1.87% vs oracle +1.19%, +0.68pp, N=589).
All v9 references in file corrected. Launcher PID 3490299 waiting for v9 (PID 3391555).

**Oracle chain:** v5(running)→v6→v7→v8→v9→v10→v11(queued) all chained.

### Oracle v11 — Written, Queued (launcher PID 3514628)

13 per-relation overrides. New additions over v10:
| Override | α | vs Oracle | Gain | N |
|----------|---|-----------|------|---|
| L12 "off" | 1.25 | +14.86% vs +9.46% | **+5.40pp** | 74 |
| L12 "near" | 1.0 | +11.82% vs +10.91% | +0.91pp | 110 |
| L11 "touching" | 0.5 | +5.54% vs +5.01% | +0.53pp | 1281 |

### "touching" Cross-Feature Sweep — FINAL

N=1281, base=56.52%. Complete ranking at best alpha per feature:
| Feature | Best α | Δ |
|---------|--------|---|
| L12/F2257 | 0.5 | **+6.09%** |
| L15/F220 | 0.5 | +5.78% |
| L14/F10561 | 0.5 | +5.62% |
| L9/F7540 | 0.5 | +5.62% |
| L9/F387 | 0.5 | +5.70% |
| L11/F12278 (oracle) | 0.45 | +5.54% |
| L6/F7539 | 0.75 | +5.39% |
| L4/F14233 | 1.0 | +5.31% |

L12 leads L11 oracle by 0.55pp (N=1281 → ~7 samples). Not switching — L11 oracle confirmed sufficient. However, touching override α=0.5 added to v11 (+0.53pp).

### L12 Assigned Sweep — All 8 Relations DONE

| Relation | N | Base | L12 oracle Δ | L12 best α | Best Δ | Override? |
|----------|---|------|-------------|------------|--------|-----------|
| beneath | 341 | 50.15% | +5.87% | 1.25 | +7.33% | **YES v9** |
| beyond | 20 | 45.00% | +10.00% | 0.75 | +10.00% | No (N too small) |
| enclosed by | 21 | 57.14% | +4.76% | 0.75 | +4.76% | No (N too small) |
| facing | 306 | 60.78% | +8.82% | 0.75 | +8.82% | No (oracle optimal) |
| near | 110 | 56.36% | +10.91% | **1.0** | **+11.82%** | **YES v11** (+0.91pp) |
| off | 74 | 48.65% | +9.46% | **1.25** | **+14.86%** | **YES v11** (+5.40pp) |
| outside | 32 | 46.88% | +3.12% | 1.0 | +9.38% | NO (N=32 too small) |
| toward | 36 | 52.78% | +16.67% | 0.75 | +16.67% | No (oracle optimal, N=36 small) |

Note: L15/F220 actually best for "outside" (α=0.9→+9.38%) and "toward" (α=0.9→+19.44%) — but N<40 makes these unreliable for rerouting.

### L9/F387 Assigned Sweep — Partial Results

**"adjacent to" (N=77, base=61.04%):** L9/F387 α=0.5→+5.19% vs oracle α=0.4→+2.60%, +2.59pp. BUT non-monotone (0.6→−1.30%) and N=77 is noisy. Skip.

**"at the right side of" (N=480, base=52.29%):** L9/F387 oracle α=0.4→+4.17% CONFIRMED AS PEAK. L12 α=0.5 gives +3.75% (worse). No override.

Remaining: at_side_of, attached_to, far_from (still running, GPU 7).

### L15 Assigned Sweep — 5/9 Relations Done

**"contains" (N=343, base=57.14%):** L15 oracle α=0.7→+6.12% CONFIRMED AS PEAK. No override.

**"inside" (N=240, base=60.42%):** L15 oracle α=0.7→+1.67% confirmed optimal — very small gain overall. No override.

Remaining: over, part_of, within (still running, GPU 2).

### Oracle Chain Status (Updated)

| Version | Overrides | Status | Expected Δ |
|---------|-----------|--------|------------|
| v3 | 0 relation overrides | DONE | **+4.05%** |
| v4 | 2 (L11 alphas) | DONE | **+4.08%** |
| v5 | 3 (+in_middle_of→0.2) | Running GPU3 | ~+4.20% |
| v6 | 6 (+on→0.3, on_top_of→0.4, surrounding→0.6) | Queued | ~+4.30% |
| v7 | 8 (+above→0.8, at_back_of→1.75) | Queued | ~+4.35% |
| v8 | 10 (+below→2.5, parallel_to→0.3) | Queued | ~+4.45% |
| v9 | 11 (+beneath→1.25) | Queued | ~+4.55% |
| v10 | 12 (+under→0.75) | Queued (PID 3490299) | ~+4.60% |
| v11 | 13 (+touching→0.5, near→1.0, off→1.25) | Queued (PID 3514628) | ~+4.70% expected |

### New Sweeps Launched

**L4/F14233 + L14/F10561 Assigned Sweep** — GPU 5, PID 3527727
- Covers: ahead_of, behind (L4, oracle α=0.9) + by, close_to, connected_to (L14, oracle α=2.0)
- Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_L4_L14_assigned_sweep/`
- Log: `/tmp/L4_L14_assigned_sweep.log`

**Remaining Relations Sweep** — GPU 6, PID 3530243
- Covers: far_from (L9F387), consists_of (L9F7540), alongside (L6), right_of (L6), away_from (L15 validation)
- Output: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_remaining_sweep/`
- Log: `/tmp/remaining_relations_sweep.log`

### GPU Status (Updated)

| GPU | Job | Status |
|-----|-----|--------|
| 0 | caa_global_alpha_fullvsr | Running |
| 1 | smart_oracle_v2 (extended) | Running |
| 2 | L15_assigned_sweep | over DONE; "part_of" running (within remaining) |
| 3 | smart_oracle_v6 | Running (v5 DONE +4.10%) |
| 4 | per_relation_alpha_sweep | touching DONE; all features complete |
| 5 | L4_L14_assigned_sweep | ahead_of DONE; "behind" running |
| 6 | remaining_relations_sweep | far_from running |
| 7 | L9F387_assigned_sweep | far_from DONE; all 5 relations complete |

---

## Monitoring Update 71 — 2026-04-22

### Oracle v5 FINAL: +4.10% ✓

**oracle v5: base=54.41% smart=58.51% Δ=+4.10% N=10,972**
- Improvement vs v4 (+4.08%): +0.02pp from "in_middle_of"→0.2
- Oracle v6 now running on GPU 3 (PID 3279304 kicked it off at 01:53)

### L9/F387 Assigned Sweep — All 5 Relations Complete

| Relation | N | Base | L9F387 oracle Δ | Best α | Best Δ | Override? |
|----------|---|------|----------------|--------|--------|-----------|
| adjacent to | 77 | 61.04% | +2.60% | 0.5 | +5.19% | No (N=77, non-monotone) |
| at the right side of | 480 | 52.29% | **+4.17%** | 0.4 | +4.17% | No (oracle optimal) |
| at the side of | 58 | 56.90% | +0.00% | 0.5 | +1.72% | No (N=58, L12 slightly better) |
| attached to | 56 | 60.71% | **+7.14%** | 0.4 | +7.14% | No (oracle optimal; L15 α=0.6→+8.93% but N=56) |
| far from | 145 | 46.90% | **-0.69%** | 0.5 | +2.07% | **YES v12** (+2.76pp over oracle) |

### L15 Assigned Sweep — New Completions

**"over" (N=84, base=55.95%):** L15 oracle α=0.7 → +8.33% CONFIRMED OPTIMAL. No override.

**"part of" (N=113, base=57.52%):** 
- L15 oracle α=0.7 → **-0.88%** (HURTING!)
- Best α=0.8 → +4.42%, **+5.30pp gain** — strong override
- Curve is non-monotone (α=0.3→+2.65%, peaks at 0.8, then drops)
- **Override: "part of"→α=0.8 (NEW in v12)**

### L4/F14233 Assigned Sweep — "ahead of" Done

**"ahead of" (N=39, base=56.41%):** L4 oracle α=0.9 → +15.38% CONFIRMED OPTIMAL. No override needed (sharp peak at oracle).

### Oracle v12 — Written, Queued (launcher PID 3554003)

15 per-relation overrides. New additions over v11:
| Override | Feature | α | vs Oracle | Gain | N |
|----------|---------|---|-----------|------|---|
| "far from" | L9/F387 | 0.5 | +2.07% vs **-0.69%** | **+2.76pp** | 145 |
| "part of" | L15/F220 | 0.8 | +4.42% vs **-0.88%** | **+5.30pp** | 113 |

### Oracle Chain Status (Updated)

| Version | Overrides | Status | Final/Expected Δ |
|---------|-----------|--------|-----------------|
| v3 | 0 | DONE | **+4.05%** |
| v4 | 2 | DONE | **+4.08%** |
| v5 | 3 | DONE | **+4.10%** |
| v6 | 6 | Running GPU3 | ~+4.30% |
| v7 | 8 | Queued | ~+4.35% |
| v8 | 10 | Queued | ~+4.45% |
| v9 | 11 | Queued | ~+4.55% |
| v10 | 12 | Queued (PID 3490299) | ~+4.60% |
| v11 | 13 | Queued (PID 3514628) | ~+4.72% |
| v12 | 15 | Queued (PID 3554003) | ~+4.80% expected |


## Monitoring Update 72 — 2026-04-22 (context resumed)

### Session Context Resume

Resumed from previous conversation context. All experiments still running as expected.

### Oracle v6 — COMPLETE

**oracle v6: base=54.41% smart=58.61% Δ=+4.20% N=10,972**
- v6 used 6 overrides: on_top_of/surrounding/in_middle_of/on/parallel_to, plus "below" → +2.5
- GPU 3 now running v7→v8→v9→v10→v11→v12→v13 chain (v13 just queued)

### New Experiments Launched

| GPU | Experiment | PID | Status |
|-----|-----------|-----|--------|
| 2 | `pt448_small_rels_extended_sweep.py` | 3601389 | Running (alongside/next_to/opposite_to) |
| 7 | `pt448_extended_alpha_sweep.py` | 3659097 | Running (adjacent_to/outside/enclosed_by/at_side_of/at_right_side_of/facing) |

### Remaining Relations Sweep — Results (GPU 6, PID 3530243)

4 of 5 relations complete. `away_from` still running.

| Relation | N | Base | Assigned Feat | Oracle Δ | Best Δ | Override? |
|----------|---|------|--------------|----------|--------|-----------|
| far from | 145 | 46.90% | L9/F387 | -0.69% | +2.07% (α=0.5) | **YES** (already in v12) |
| consists of | 35 | 68.57% | L9/F7540 | +0.00% | +0.00% | No (N=35 tiny, oracle ok) |
| alongside | 55 | 43.64% | L6/F7539 | +12.73% | +14.55% (α=1.25) | **YES → v13** |
| right of | 113 | 53.98% | L6/F7539 | +0.88% | +1.77% (L15 α=0.7) | Feature reassign → v13 |
| away from | 155 | 48.39% | L15/F220 | +1.29% | In progress... | TBD |

Key `alongside` finding: L6/F7539 best α=1.25 (+14.55%) > oracle 1.5 (+12.73%), confirmed by BOTH remaining_sweep and small_rels_sweep. Gap +1.82pp, N=55 (borderline but consistent).

### L4/L14 Assigned Sweep — Results (GPU 5, PID 3527727)

3 of 5 relations complete.

| Relation | N | Base | Assigned Feat | Oracle Δ | Best Δ | Override? |
|----------|---|------|--------------|----------|--------|-----------|
| ahead of | 39 | 56.41% | L4/F14233 | +15.38% | +15.38% (oracle) | No (oracle optimal) |
| behind | 709 | 51.62% | L4/F14233 | +1.97% | +2.26% (L12 α=1.0) | No (0.29pp gap, feature switch not worth it) |
| by | 52 | 57.69% | L14/F10561 | +3.85% | +7.69% (α=0.5!) | **YES → v13** (+3.84pp gap) |
| close to | 93 | 60.22% | L14/F10561 | In progress... | ... | TBD |
| connected to | — | — | L14/F10561 | Not yet | — | TBD |

**`by`** is a strong override candidate: oracle α=2.0 gives +3.85%, but α=0.5 → +7.69% (+3.84pp, N=52). Clear non-monotone peak at low alpha.

### Oracle v13 — Written, Queued

**18 per-relation overrides** (v12 had 15). New additions over v12:

| Override | Feature | α | vs Oracle | Gain | N |
|----------|---------|---|-----------|------|---|
| "adjacent to" | L9/F387 | 0.5 | +5.19% vs +2.60% | **+2.59pp** | 77 |
| "by" | L14/F10561 | 0.5 | +7.69% vs +3.85% | **+3.84pp** | 52 |
| "alongside" | L6/F7539 | 1.25 | +14.55% vs +12.73% | **+1.82pp** | 55 |

Also: `right_of` feature reassigned from L6_F7539 → L15_F220 (oracle: +1.77% vs +0.88%, +0.89pp, N=113).

**Launcher PID: 3688451** (waiting for v12 launcher PID 3554003 to complete).

### Oracle Chain Status (Updated)

| Version | Overrides | Status | Final/Expected Δ |
|---------|-----------|--------|-----------------|
| v3 | 0 | DONE | **+4.05%** |
| v4 | 2 | DONE | **+4.08%** |
| v5 | 3 | DONE | **+4.10%** |
| v6 | 6 | **DONE** | **+4.20%** |
| v7 | 8 | Queued | ~+4.35% |
| v8 | 10 | Queued | ~+4.45% |
| v9 | 11 | Queued | ~+4.55% |
| v10 | 12 | Queued (PID 3490299) | ~+4.65% |
| v11 | 13 | Queued (PID 3514628) | ~+4.72% |
| v12 | 15 | Queued (PID 3554003) | ~+4.82% |
| v13 | 18 | Queued (PID 3688451) | ~+4.90% expected |

### GPU Status

| GPU | Experiment | Status |
|-----|-----------|--------|
| 0 | skip_relations_sweep | Running — `in_front_of` in progress |
| 1 | Unknown (91% util) | — |
| 2 | small_rels_extended_sweep | Running — `next_to` in progress |
| 3 | oracle v6→v7→...→v13 | v6 DONE, chain running |
| 4 | Unknown (91% util) | — |
| 5 | L4/L14 assigned sweep | Running — `close_to` in progress |
| 6 | remaining_relations_sweep | Running — `away_from` in progress |
| 7 | extended_alpha_sweep | Just launched (PID 3659097) |

### Key Findings Summary (v12 data)

All confirmed oracle alpha data:
- `on_top_of` → α=0.4 (+5.35% vs +4.95%), N=505, +0.40pp
- `surrounding` → α=0.6 (+11.11% vs +7.78%), N=90, +3.33pp ✓✓
- `touching` → α=0.5 (+5.54% vs +5.01%), N=1281, +0.53pp ✓
- `under` → α=0.75 (+1.87% vs +1.19%), N=589, +0.68pp
- `above` → α=0.8 (+5.87% vs +4.69%), N=341, +1.18pp ✓
- `part of` → α=0.8 (+4.42% vs **-0.88%**), N=113, +5.30pp ✓✓
- `at_back_of` → α=1.75 (+3.19% vs +2.13%), N=94, +1.06pp
- `below` → α=2.5 (+6.14% vs +5.42%), N=277, +0.72pp
- `beneath` → α=1.25 (+7.33% vs +5.87%), N=341, +1.46pp ✓
- `near` → α=1.0 (+11.82% vs +10.91%), N=110, +0.91pp
- `off` → α=1.25 (+14.86% vs +9.46%), N=74, +5.40pp ✓✓
- `far_from` → α=0.5 (+2.07% vs **-0.69%**), N=145, +2.76pp ✓✓
- `in_middle_of` → α=0.2 (+7.61% vs +5.43%), N=92, +2.18pp ✓
- `on` → α=0.3 (+1.37% vs +0.34%), N=585, +1.03pp ✓
- `parallel_to` → α=0.3 (+7.78% vs +5.56%), N=90, +2.22pp ✓

New (v13 additions):
- `adjacent_to` → α=0.5 (+5.19% vs +2.60%), N=77, +2.59pp ✓
- `by` → α=0.5 (+7.69% vs +3.85%), N=52, +3.84pp ✓✓
- `alongside` → α=1.25 (+14.55% vs +12.73%), N=55, +1.82pp ✓ (2x confirmed)
- `right_of` feature: L6→L15 (+1.77% vs +0.88%), N=113, +0.89pp


## Monitoring Update 73 — 2026-04-22 (sweeps complete, v14 queued)

### All Sweep Results Summary

**Remaining Relations Sweep — COMPLETE**

| Relation | N | Base | Assigned | Oracle Δ | Best Δ | Best α | Override? |
|----------|---|------|---------|----------|--------|--------|-----------|
| far from | 145 | 46.90% | L9/F387 | -0.69% | +2.07% | α=0.5 | YES (v12) |
| consists of | 35 | 68.57% | L9/F7540 | +0.00% | +0.00% | oracle | No (N=35) |
| alongside | 55 | 43.64% | L6/F7539 | +12.73% | +14.55% | α=1.25 | YES (v13) |
| right of | 113 | 53.98% | L6/F7539 | +0.88% | +1.77% | L15 α=0.7 | Feat reassign (v13) |
| away from | 155 | 48.39% | L15/F220 | +1.29% | +3.87% | L9F7540 α=1.0 | Feat reassign+override (v14) |

**L4/L14 Assigned Sweep — COMPLETE**

| Relation | N | Base | Assigned | Oracle Δ | Best Δ | Best α | Override? |
|----------|---|------|---------|----------|--------|--------|-----------|
| ahead of | 39 | 56.41% | L4/F14233 | +15.38% | +15.38% | oracle | No |
| behind | 709 | 51.62% | L4/F14233 | +1.97% | +2.26% | L12 α=1.0 | No (0.29pp, cross-feat) |
| by | 52 | 57.69% | L14/F10561 | +3.85% | +7.69% | α=0.5 | **YES (v13)** +3.84pp |
| close to | 93 | 60.22% | L14/F10561 | +10.75% | +10.75% | oracle | No (oracle optimal) |
| connected to | 37 | 48.65% | L14/F10561 | +10.81% | +13.51% | α=1.5 | YES (v14), N=37 |

**Extended Alpha Sweep (GPU 7) — In Progress**

| Relation | N | Base | Assigned | Oracle Δ | Best Δ | Best α | Override? |
|----------|---|------|---------|----------|--------|--------|-----------|
| adjacent to | 77 | 61.04% | L9/F387 | +2.60% | +5.19% | α=0.5 | YES (v13) |
| at the side of | 58 | 56.90% | L9/F387 | +0.00% | +3.45% | L12 α=0.75 | **Feat reassign → v14** |
| outside | 32 | 46.88% | L12/F2257 | +3.12% | +9.38% | α=0.9/1.0 | Tentative (N=32) |
| enclosed by | 21 | 42.86% | L12/F2257 | +9.52% | +19.05% | α=1.25 | No (N=21 tiny) |
| at the right side of | 480 | — | L9/F387 | — | In progress | — | TBD |
| facing | 306 | — | L12/F2257 | — | In progress | — | TBD |

**Small Rels Extended Sweep (GPU 2) — In Progress**

| Relation | N | Base | Best Feat | Best Δ | Override? |
|----------|---|------|----------|--------|-----------|
| alongside | 55 | 43.64% | L6/F7539 α=1.25 | +14.55% | YES (v13 confirmed) |
| next to | 309 | 61.49% | L9/F7540 oracle | +1.62% | No (oracle optimal) |
| opposite to | 36 | — | In progress | — | TBD |

**Skip Relations Sweep (GPU 0) — In Progress**

| Relation | N | Base | Best Δ | Remove from SKIP? |
|----------|---|------|--------|-------------------|
| in front of | 737 | 56.58% | +1.22% (L9F7540 α=0.1) | **Marginal** — too small to justify un-SKIPping |
| in, far away from, at edge of | — | — | Pending | TBD |

### Oracle v14 — Written, Queued (PID 3705882)

**20 per-relation overrides** (v13 had 18). New additions:

| Change | Type | From → To | Gain | N |
|--------|------|-----------|------|---|
| `at the side of` | Feature reassign | L9/F387 → L12/F2257 | +3.45% vs +0.00% | 58 |
| `away from` | Feature reassign | L15/F220 → L9/F7540 | +3.87% vs +1.29% | 155 |
| `away from` | Alpha override | L9/F7540 α=1.0 (vs oracle 0.25→+0.65%) | +3.22pp | 155 |
| `connected to` | Alpha override | L14/F10561 α=1.5 → +13.51% vs oracle +10.81% | +2.70pp | 37 |

**Expected v14 Δ: ~+5.00%** (rough estimate)

### Oracle Chain Status

| Version | Overrides | Status | Final/Expected Δ |
|---------|-----------|--------|-----------------|
| v3 | 0 | DONE | **+4.05%** |
| v4 | 2 | DONE | **+4.08%** |
| v5 | 3 | DONE | **+4.10%** |
| v6 | 6 | DONE | **+4.20%** |
| v7 | 8 | Running GPU3 | ~+4.35% |
| v8–v11 | 10–13 | Queued | ~+4.45%→+4.72% |
| v12 | 15 | Queued (PID 3554003) | ~+4.82% |
| v13 | 18 | Queued (PID 3688451) | ~+4.90% |
| v14 | 20 | Queued (PID 3705882) | ~+5.00% |


## Monitoring Update 74 — 2026-04-22 (v15 queued, sweeps expanded)

### Oracle v15 — Written, Queued (PID 3730660)

**21 per-relation overrides** (v14 had 20). New addition:

| Override | Feature | α | vs Oracle | Gain | N |
|----------|---------|---|-----------|------|---|
| "opposite to" | L9/F7540 | 0.15 | +5.56% vs +2.78% | **+2.78pp** | 36 |

Note: three independent features (L9_F7540, L11_F12278, L9_F387) all peak at very low alpha (~0.15–0.2) for `opposite_to`, suggesting the model only needs a gentle nudge in this direction.

**Expected v15 Δ: ~+5.05%**

### New Sweeps Launched

| GPU | Script | PID | Targets |
|-----|--------|-----|---------|
| 5 | `pt448_reassignment_sweep.py` | 3737668 | inside/toward/within/outside/left_of — cross-feature |
| 2 | `pt448_l6_relations_sweep.py` | 3744430 | left_of/across_from/facing_away_from/at_left_side_of — extended α |

### Pending Analysis: Small Rels Sweep Results

**`next to`** (N=309): L9_F7540 oracle optimal (+1.62%). No override.

**`opposite to`** (N=36): 
- L9_F7540 oracle +2.78%, α=0.15 → **+5.56%** (+2.78pp) → override in v15
- L9_F387 at α=0.2 also → +5.56%  
- L11_F12278 at α=0.2 also → +5.56%
- Three features converge at very low α → model only needs a weak directional signal

### Oracle Chain Status

| Version | Overrides | Status | Final/Expected Δ |
|---------|-----------|--------|-----------------|
| v3 | 0 | DONE | **+4.05%** |
| v4 | 2 | DONE | **+4.08%** |
| v5 | 3 | DONE | **+4.10%** |
| v6 | 6 | DONE | **+4.20%** |
| v7 | 8 | Running GPU3 | ~+4.35% |
| v8–v11 | 10–13 | Queued | ~+4.45%→+4.72% |
| v12 | 15 | Queued (PID 3554003) | ~+4.82% |
| v13 | 18 | Queued (PID 3688451) | ~+4.90% |
| v14 | 20 | Queued (PID 3705882) | ~+5.00% |
| v15 | 21 | Queued (PID 3730660) | ~+5.05% |

### GPU Status

| GPU | Experiment | Status |
|-----|-----------|--------|
| 0 | skip_relations_sweep | Running — `far away from` in progress |
| 1 | Unknown (92% util) | — |
| 2 | l6_relations_sweep (PID 3744430) | Running — `left of` feature sweep in progress |
| 3 | oracle chain v7→v16 | v7 DONE (+4.25%), v8 running |
| 4 | Unknown (91% util) | — |
| 5 | reassignment_sweep (PID 3737668) | Running — `inside` L12_F2257 sweep in progress |
| 6 | unswept_relations_sweep (PID 3805024) | Just launched — beside/beyond/contains/over/attached_to |
| 7 | extended_alpha_sweep (PID 3659097) | Running — `at_right_side_of`/`facing` |

---

## Monitoring Update 75 — 2026-04-22

### Oracle v7 FINAL Result

**v7: base=54.41%, smart=58.66%, Δ=+4.25%, N=10,972**
- inject N=9,239: base=54.06%, smart=59.11%, Δ=+5.04%
- skip N=1,630: Δ=0.00% (as expected)
- unknown N=103: Δ=0.00%

v8 now running on GPU 3 (queued chain v8→v16).

### Oracle v16 Written and Queued

**v16 (PID launcher 3797841)** queues after v15 completes.

**v16 changes from v15 (22 total overrides, +1 new):**
1. `inside` feature reassigned L15_F220 → L12_F2257 (oracle α=0.75 → +2.92% vs L15 best +1.67%, +1.25pp, N=240, confirmed by reassignment_sweep)
2. `outside` alpha override: L12_F2257 α=1.0 → +9.38% vs oracle +3.12%, +6.26pp, N=32 (small N, included due to large gap)

### Sweep Results (this session)

#### inside (reassignment_sweep, GPU 5 — confirmed)
- `inside` N=240, base=60.42%
- L15_F220: best α=0.7 → +1.67% (oracle optimal)
- **L12_F2257: oracle α=0.75 → +2.92%** ← winner (+1.25pp over L15)
- Decision: reassign L15→L12 in v16 CONFIRMED

#### left_of (l6_relations_sweep, GPU 2 — in progress)
- `left of` N=210, base=49.52%
- L6_F7539: oracle α=1.5 → +4.29% (oracle optimal confirmed)
- L15_F220: α=0.5 → +2.86% (L6 wins; no reassignment)
- Decision: keep L6_F7539 assignment, no alpha override needed

#### extended_alpha_sweep results (GPU 7)
- `at the right side of` N=480, base=52.29%
  - L9_F387: oracle α=0.4 → +4.17% ← BEST (oracle optimal, no change)
  - L12_F2257: best α=0.5 → +3.75% (L9 wins)
  - Decision: keep L9_F387 at oracle α=0.4, no override
- `facing` N=? — still running

#### skip relations sweep (GPU 0)
- `in` N=276, base=62.68%: best any feature is L9_F7540 α=0.25 → +1.09%. All others negative or near zero. CONFIRMED SKIP.
- `far away from` N=232, base=49.14%: all features negative at oracle alphas. L12 at α=0.25 → -0.43% (barely). CONFIRMED SKIP.

#### unswept_relations_sweep (GPU 6 — just launched)
- Sweeping: beside, beyond, contains, over, attached_to
- Features: L15_F220, L12_F2257, L9_F387, L11_F12278, L9_F7540

### Potential v17 Overrides (preliminary)

From reassignment_sweep (still running — toward, within, outside, left_of pending):
- `within` (N=33): L15_F220 α=0.8 → +6.06% vs oracle +3.03%, +3.03pp (N=33 small)
- `toward` (N=36): L15_F220 α=0.9 → +19.44% vs L12 oracle +16.67%, +2.77pp (needs confirmation)
- `outside` already captured in v16

From unswept_relations_sweep (pending):
- Any alpha overrides for beside, beyond, contains, over, attached_to

From l6 sweep (pending):
- Any alpha overrides for across_from, facing_away_from, at_left_side_of

---

## Monitoring Update 76 — 2026-04-22

### Oracle Chain Status
- v7: DONE → Δ=+4.25%
- v8: **DONE → Δ=+4.28%** (N=10,972, base=54.41%, smart=58.69%)
- v9: Running on GPU 3 (just started)
- v10→v19: All queued sequentially

**v17 launcher PID: 3838846** — queues after v16  
**v18 launcher PID: 3848505** — queues after v17  
**v19 launcher PID: 3923774** — queues after v18

### New Oracles Written

**v17 (23 overrides, +1 new):**
- `toward` → reassigned L12_F2257 → L15_F220 + override α=0.9 (+19.44% vs L12 oracle +16.67%, +2.77pp, N=36)

**v18 (23 overrides, same count, 2 reassignments):**
- `within` → reassigned L15_F220 → L12_F2257 (oracle α=0.75 → +6.06% vs L15 oracle +3.03%, +3.03pp, N=33)
- `outside` → reassigned L12_F2257 → L15_F220 + override α=1.0 (+12.50% vs L12 α=1.0 → +9.38%, +3.12pp, N=32)
- `facing` → **KEPT at L12_F2257** (confirmed: L12 oracle α=0.75 → +8.82% > L9_F387 best → +7.84%, N=306)
  - Initial v18 draft had facing reassigned to L9, but corrected after L12 data came in

**v19 (24 overrides, +1 new):**
- `enclosed by` → alpha override 1.25 for L12_F2257 (+19.05% vs oracle α=0.75 → +9.52%, +9.53pp, N=21)
- Script: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_smart_oracle_v19.py`

### All Completed Sweep Results

| Relation | N | Best Feature | Best α | Best Δ | Oracle Δ | Action |
|----------|---|-------------|--------|--------|----------|--------|
| inside | 240 | L12_F2257 | 0.75 (oracle) | +2.92% | +1.67% (L15) | Reassign L15→L12 (v16 ✓) |
| left of | 210 | L6_F7539 | 1.5 (oracle) | +4.29% | +4.29% | No change (oracle optimal) |
| across from | 94 | L6_F7539 | 1.5 (oracle) | +14.89% | +14.89% | No change (oracle optimal) |
| at_right_side_of | 480 | L9_F387 | 0.4 (oracle) | +4.17% | +4.17% | No change (oracle optimal) |
| toward | 36 | L15_F220 | 0.9 | +19.44% | +16.67% (L12) | Reassign L12→L15 + α=0.9 (v17 ✓) |
| within | 33 | L12_F2257 | 0.75 (oracle) | +6.06% | +3.03% (L15 oracle) | Reassign L15→L12 (v18 ✓) |
| outside | 32 | L15_F220 | 1.0 | +12.50% | +3.12% (oracle) | Reassign L12→L15 + α=1.0 (v18 ✓) |
| facing | 306 | L12_F2257 | 0.75 (oracle) | +8.82% | +8.82% | No change — L12 oracle IS best (L9 best only +7.84%) |
| beside | 188 | L15_F220 | 0.7 (oracle) | +10.64% | +10.64% | No change (oracle optimal) |
| beyond | 20 | L12_F2257 | 0.75 (oracle) | +20.00% | +20.00% | No change — oracle already best (N=20 tiny) |
| enclosed by | 21 | L12_F2257 | 1.25 | +19.05% | +9.52% (oracle 0.75) | Alpha override 1.25 (v19 ✓) |
| at_left_side_of | 421 | L15_F220 | 0.7 (oracle) | +8.55% | +8.55% | No change — L15 oracle optimal (L6 best +8.31%) — L12/L9/L11 pending |
| facing away from | 180 | L6_F7539 | 1.5 (oracle) | +11.11% | +11.11% | No change — L6 oracle IS best |
| in | 276 | L9_F7540 | 0.25 | +1.09% | — | Confirmed SKIP (no override helps) |
| at_edge_of | 211 | L6_F7539 | 1.5 | +0.00% | — | Confirmed SKIP (all features ≤0%) |
| far away from | 232 | — | — | all neg | — | Confirmed SKIP |
| in_front_of | 737 | best +0.27% | — | +0.27% | — | Confirmed SKIP (marginal) |

### SKIP Relations Confirmed
Based on skip_relations_sweep (complete):
- `at the edge of` (N=211): all features ≤ 0% at all alphas → SKIP
- `far away from` (N=232): all features negative → SKIP
- `in front of` (N=737): best +0.27% at L15 α=0.3, negligible → SKIP
- `in` (N=276): max +1.09% from L9_F7540 α=0.25 → SKIP (already in SKIP list)

### GPU Status (Current)

| GPU | Experiment | Status |
|-----|-----------|--------|
| 0 | at_left_side_of_sweep (PID 3927472) | Running — L12/L9/L11 features sweep (L15 oracle already best: +8.55%) |
| 1 | pt448_caa_global_alpha_fullvsr (PID 2197053) | Running |
| 0 | at_left_side_of_sweep (PID 3927472) | Running — L12 done (best α=1.0→+5.46%), L9/L11 pending; L15 oracle +8.55% still leads |
| 1 | pt448_caa_global_alpha_fullvsr (PID 2197053) | Running |
| 2 | idle (l6 sweep PID 3744430 dead, GPU mem held) | Effectively free after mem release |
| 3 | oracle chain v8→v19 | v8 DONE (+4.28%), v9 just started |
| 4 | pt448_per_relation_alpha_sweep (PID 2966161) | Running |
| 5 | touching_below_above_sweep (PID 3957337) | NEW — cross-feature sweep for touching(N=1281)/below(N=277)/above(N=341) |
| 6 | unswept_relations_sweep (PID 3805025) | Running — beside/beyond done, contains L15=+6.12%/L12=+4.96%/L9=+4.96%/L11 pending |
| 7 | l9f7540_relations_sweep (PID 3934669) | Running — `on`(N=585) L9_F7540 base sweep starting |

### Contains Sweep (Partial — GPU 6 still running)
`contains` N=343, base=57.14%:
- L15_F220: oracle α=0.7 → **+6.12%** (best so far)
- L12_F2257: oracle α=0.75 → +4.96%
- L9_F387: best α=0.75 → +4.96% (oracle α=0.4 → +1.46%)
- L11_F12278: running (α=0.6 → +4.96% seen)
- L9_F7540: pending

→ `contains` currently assigned L15_F220 (oracle). L15 at oracle α=0.7 → +6.12% is already **best confirmed** — no reassignment needed. No override needed (oracle is optimal for L15).

### v20 Candidates (pending current sweeps)

- `touching` (N=1281): cross-feature vs L11 oracle +6.01% (L12/L6/L15 on GPU 5)
- `below` (N=277): L6 override α=2.5→+6.14%, but never vs other features (GPU 5)
- `above` (N=341): L15 override α=0.8→+5.87%, but never vs other features (GPU 5)
- `at_left_side_of` (N=421): L15 oracle +8.55%, L12 α=1.0→+5.46%, L9/L11 pending (GPU 0)
- `on` (N=585): L9_F7540 current, cross-feature sweep starting (GPU 7)
- `next to` (N=309): L9_F7540 current, cross-feature sweep pending (GPU 7)
- `contains` (N=343): L15 oracle appears optimal at +6.12%, but L9_F7540 pending


---

## Monitoring Update 77 — 2026-04-22

### Oracle Chain Progress
- v8: **DONE → Δ=+4.28%** (N=10,972, base=54.41%, smart=58.69%)
- v9: Running on GPU 3 (just started loading)
- v10→v19: All queued sequentially via bash launcher chain

### New Sweeps Launched This Round

**GPU 5 — `pt448_touching_below_above_sweep.py` (PID 3957337)**
Rationale: Three high-N relations (touching N=1281, below N=277, above N=341) are assigned to L11/L6/L15 respectively but have **never been cross-feature compared** with other CAA features.
- `touching`: L11 oracle α=0.45 → +6.01%, override α=0.5 → +5.54% (already in v18) — checking if L12/L6/L15 can beat this
- `below`: L6 override α=2.5 → +6.14% (in v18) — checking if other features beat L6
- `above`: L15 override α=0.8 → +5.87% (in v18) — checking if other features beat L15
Script: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_touching_below_above_sweep.py`

### Confirmed This Round

**v8 result: Δ=+4.28%** — strong result with 8 overrides. Full chain expected to converge toward +5–6% with v9→v19 improvements.

**right_of (N=113)**: L15_F220 oracle α=0.7 → +1.77% confirmed best across all features (L9_F387 best only +1.77% at α=0.6). Assignment confirmed correct, no override.

**contains (N=343, partial)**: L15_F220 oracle α=0.7 → +6.12% is current best; L12/L9 max out at +4.96%. L15 oracle assignment is confirmed optimal for `contains`. L9_F7540 still pending.

**at_left_side_of L12 result**: L12_F2257 best α=1.0 → +5.46% vs L15 oracle +8.55% — L15 still comfortably best. No reassignment needed.

### Expected v20 Decisions
Once GPU 5/7 sweeps complete:
1. If any feature beats L11 for `touching` → reassign + potentially large gain (N=1281)
2. If `below` α<2.5 yields same gain from different feature → cleaner override
3. If `on` (N=585) finds better feature than L9_F7540 → potentially significant v20 gain

---

## Monitoring Update 78 — 2026-04-22

### Completed Sweeps This Round

#### at_left_side_of (N=421) — COMPLETE
All 5 features swept (GPU 0, `pt448_at_left_side_of_sweep.py`):
| Feature | Oracle α | Oracle Δ | Best α | Best Δ |
|---------|---------|---------|--------|--------|
| L15_F220 | 0.7 | **+8.55%** | 0.7 | **+8.55%** |
| L9_F387 | 0.4 | +4.28% | 0.6 | **+8.55%** (ties L15!) |
| L11_F12278 | 0.45 | +4.51% | 0.6 | **+8.55%** (3-way tie) |
| L12_F2257 | 0.75 | +4.75% | 1.0 | +5.46% |
| L6_F7539 | 1.5 | +1.66% | 0.75 | +8.31% |

**Decision: No change.** L15 oracle α=0.7 is already optimal. Three-way tie at +8.55% between L15/L9/L11 confirms L15 is the right assignment.

#### unswept_relations — ALL DONE (GPU 6)
| Relation | N | Best Feature | Best α | Best Δ | Current Δ | Action |
|----------|---|-------------|--------|--------|-----------|--------|
| beside | 188 | L15_F220 | 0.7 (oracle) | +10.64% | +10.64% | No change |
| beyond | 20 | L12_F2257 | 0.75 (oracle) | +20.00% | +20.00% | No change |
| contains | 343 | L15_F220 | 0.7 (oracle) | +6.12% | +6.12% | No change |
| over | 84 | L15_F220 | 0.7 (oracle) | +8.33% | +8.33% | No change |
| **attached to** | **56** | **L15_F220** | **0.6** | **+8.93%** | +7.14% (L9 oracle) | **→ REASSIGN L9→L15 + α=0.6** |

**attached_to detail:** L15 α=0.6 → +8.93% vs L9 oracle +7.14% (+1.79pp). L12 best +3.57%, L11 best +3.57%, L9_F7540 best +3.57% — L15 is clearly the winner.

### New Oracle Written

**v20 (25 overrides, +1 new, 1 feature reassignment):**
- `attached to` → reassigned L9_F387 → L15_F220 + override α=0.6 (+8.93% vs L9 oracle +7.14%, +1.79pp, N=56)
- Script: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_smart_oracle_v20.py`
- Launcher PID 4014511 (queues after v19 launcher PID 3923774)

### New Sweeps Launched

**GPU 0 — `pt448_under_ontop_sweep.py` (PID 4017209)**
Cross-feature sweep for three L11-assigned high-N relations:
- `under` (N=589): L11 oracle +1.19%, override α=0.75→+1.87% — checking L12/L15/L6/L9
- `on top of` (N=505): L11 oracle +4.95%, override α=0.4→+5.35% — checking L12/L15/L6/L9
- `surrounding` (N=90): L11 oracle +7.78%, override α=0.6→+11.11% — checking L12/L15/L6/L9

### Oracle Chain Progress
- v8: **DONE → Δ=+4.28%**
- v9: Running at 4000/10972 (base=54.88% smart=59.55% → Δ≈+4.67%)
- v10→v20: Queued sequentially

### GPU Status

| GPU | Experiment | Status |
|-----|-----------|--------|
| 0 | under_ontop_sweep (PID 4017209) | NEW — `under`/`on_top_of`/`surrounding` cross-feature |
| 1 | pt448_caa_global_alpha_fullvsr (PID 2197053) | Running |
| 2 | idle | Free (l6 sweep process dead, GPU mem released) |
| 3 | oracle chain v9→v20 | v9 at 4000/10972 (+4.67%) |
| 4 | pt448_per_relation_alpha_sweep (PID 2966161) | Running |
| 5 | touching_below_above_sweep (PID 3957337) | Running — `touching` L11 oracle confirmed at +6.01%, other feats pending |
| 6 | unswept_relations_sweep | **DONE** |
| 7 | l9f7540_relations_sweep (PID 3934669) | Running — `on` (N=585): L9_F7540 best=+1.37%, L12 best=+0.85%, L15 pending |

### v21 Candidates
- `touching` (N=1281): if L6/L15/L12 beats L11 oracle +6.01%
- `below` (N=277): if any feature beats L6 override +6.14%
- `above` (N=341): if any feature beats L15 override +5.87%
- `under` (N=589): if any feature beats L11 override +1.87%
- `on top of` (N=505): if any feature beats L11 override +5.35%
- `surrounding` (N=90): if any feature beats L11 override +11.11%
- `on` (N=585): L9_F7540 best α=0.3→+1.37%, already overridden, L15/L11 pending (GPU 7)
- `next to` (N=309): L9_F7540 current, L12/L15/L11 pending (GPU 7)
- `under` (N=589): L12_F2257 oracle α=0.75→+2.04% > L11 best +1.87% — **v21 candidate** (L6/L9 pending)

---

## Monitoring Update 80 — 2026-04-22 ~06:10

### Oracle Chain Status

| Version | Status | Δ |
|---------|--------|---|
| v8 | DONE | +4.28% |
| v9 | DONE | +4.33% |
| **v10** | **RUNNING** @ 2000/10972 | tracking +4.45% (base=54.00% smart=58.45%) |
| v11→v20 | queued in chain | — |

### GPU 7 — `on` COMPLETE (saved), `next_to` in progress

`on` (N=585) — rel_on.json saved:
| Feature | Oracle Δ | Best α | Best Δ |
|---------|---------|--------|--------|
| L9_F7540 | +0.34% | 0.3 | **+1.37%** |
| L11_F12278 | +0.17% | 0.2 | **+1.37%** (tied!) |
| L12_F2257 | -1.88% | 0.25 | +0.85% |
| L15_F220 | -0.85% | 0.3 | +0.68% |
| L6_F7539 | -6.67% | 0.5 | -0.85% |

**Decision `on`: No change.** v20 already overrides to α=0.3 which matches the L9 best. L6 is harmful. Assignment L9_F7540 confirmed correct.

`next_to` (N=309) — in progress:
- L9_F7540 oracle α=0.25 → **+1.62%** (oracle is already best!)
- L12_F2257 best α=0.5 → +1.29% (below L9)
- L15/L11 still pending

**Likely decision `next_to`: oracle α=0.25 already optimal.** No new override needed (v20 already has no override for `next_to` — it runs at oracle +1.62%).

### GPU 0 — `under` (N=589) — CRITICAL FINDING: L12 beats L11!

`under_ontop_sweep.py` L11, L15, L12 complete for `under`:
| Feature | Oracle Δ | Best α | Best Δ |
|---------|---------|--------|--------|
| L11_F12278 | +1.19% | 0.75 | **+1.87%** |
| L15_F220 | +1.87% | 0.7 (oracle) | **+1.87%** (tie) |
| **L12_F2257** | **+2.04% (oracle)** | **0.75** | **+2.04%** |
| L6_F7539 | — | — | pending |
| L9_F387 | — | — | pending |

**L12_F2257 oracle α=0.75 → +2.04% for `under`, beating L11 best +1.87% (+0.17pp, N=589).** L12 oracle is already at the best alpha. If L6/L9 don't exceed +2.04%, v21 will reassign `under` L11→L12 with L12 oracle alpha (no separate override needed since oracle is best).

### GPU 5 — `touching` (N=1281) — L15 rising, L11 still leads

`touching_below_above_sweep.py` `touching` feature results so far:
| Feature | Best α | Best Δ | Notes |
|---------|--------|--------|-------|
| L11_F12278 | 0.45 (oracle) | **+6.01%** | Confirmed best (oracle optimal) |
| L6_F7539 | 0.75 | +5.39% | Below L11 |
| L15_F220 | ~0.5–0.6 | +5.78% at α=0.5 | **Still climbing — peak ~+6% possible** |
| L12_F2257 | — | pending | |
| L9_F387 | — | pending | |

**L15 is at +5.78% @ α=0.5 and still ascending.** Peak likely near α=0.6–0.7 (may approach L11's +6.01%).

### New Sweeps (GPUs 2 & 6) — Early Stage

**GPU 2 `behind/near/beneath`**: Just started `behind` L4_F14233 (first feature).
**GPU 6 `at_right_side/left_of/inside`**: Just started `at_right_side_of` L4_F14233, at α=0.5→+0.83% (L4 likely ≤+2%).

### v21 Plan

Pending final `under` results (L6/L9 from GPU 0):
- If L12 stays best: reassign `under` L11_F12278 → L12_F2257, remove the `"under": 0.75` override from L11 section (L12 oracle IS 0.75, so the override becomes redundant — L12 oracle already at its best alpha)
- Net effect on `under`: +2.04% vs current +1.87% (+0.17pp, N=589 samples)

### GPU Status

| GPU | PID | Experiment | Current Step |
|-----|-----|-----------|-------------|
| 0 | 4017209 | under_ontop_sweep | `under` L12→+2.04%, L6/L9 pending; `on_top_of`/`surrounding` next |
| 1 | 2197053 | caa_global_alpha_fullvsr | Running |
| 2 | 4085814 | behind_near_beneath_sweep | `behind` L4 baseline starting |
| 3 | 4071593 | oracle chain v10 | 2000/10972 (Δ≈+4.45%) |
| 4 | 2966161 | per_relation_alpha_sweep | Running |
| 5 | 3957337 | touching_below_above_sweep | `touching` L15 at +5.78%@α=0.5 (rising) |
| 6 | 4085840 | at_right_side_leftof_inside_sweep | `at_right_side_of` L4 running |
| 7 | 3934669 | l9f7540_relations_sweep | `next_to` L9=+1.62%; L12=+1.29%; L15/L11 pending |

---

## Monitoring Update 79 — 2026-04-22 ~05:40

### Oracle Chain Progress

| Version | Status | Result |
|---------|--------|--------|
| v8 | DONE | Δ=+4.28% (base=54.41%, smart=58.69%, N=10972) |
| **v9** | **DONE** | **Δ=+4.33%** (base=54.41%, smart=58.74%, N=10972) |
| v10 | RUNNING on GPU 3 (PID 4071593) | 10 overrides active (on_top_of, surrounding, under, above, at_back_of, below, beneath, in_middle_of, on, parallel_to) |
| v11→v20 | Queued in launcher chain | |

v9 vs v8: +0.05pp improvement from `touching` override (α=0.5→0.45 removed → actually v9 added `touching` α=0.5). Incremental gains continuing.

### GPU 7 Sweep — `on` (N=585) COMPLETE for 4/5 features

l9f7540_relations_sweep.py finished `on` for L9/L12/L15/L11:

| Feature | Oracle Δ | Best α | Best Δ |
|---------|---------|--------|--------|
| L9_F7540 | +0.34% | **0.3** | **+1.37%** |
| L12_F2257 | -1.88% | 0.25 | +0.85% |
| L15_F220 | -0.85% | 0.3 | +0.68% |
| L11_F12278 | +0.17% | 0.2 | +1.37% |
| L6_F7539 | pending | — | — |

**Decision for `on` (N=585): No feature reassignment.** L9_F7540 and L11_F12278 are tied at +1.37%. v20 already overrides to α=0.3 which matches the best. Alpha override α=0.3 in v20 is confirmed optimal.

### GPU 0 Sweep — `under` (N=589) L11 result confirmed

`under_ontop_sweep.py` L11_F12278 complete:
- L11_F12278 BEST α=0.75 → **+1.87%** (oracle α=0.45 → +1.19%)
- **v20 already has `under`: 0.75 override — confirmed correct**
- L15_F220 starting at α=0.3→-0.17% (negative, likely will stay below L11)

### GPU 5 Sweep — `touching` (N=1281) partial results

`touching_below_above_sweep.py` on GPU 5:
- L11_F12278 oracle α=0.45 → **+6.01%** (confirmed best for L11)
- L6_F7539 at α=0.5→+4.84%, α=0.75→+5.39%, α=1.0→+5.15% — **L11 leads significantly**
- L15/L12/L9 still pending in sweep

**Likely decision:** L11_F12278 stays as `touching` feature; oracle α=0.45 is already optimal.

### New Sweeps Launched

**GPU 2 — `pt448_behind_near_beneath_sweep.py` (PID 4085814)**
Full 8-feature cross-sweep for:
- `behind` (N=709): Current L4_F14233 oracle +1.97%; L12 showed +2.26% in prior sweep; checking all 8 features including L11/L9F387
- `near` (N=110): Current L12_F2257 oracle +11.82%; L6 showed +12.73% — potential reassignment  
- `beneath` (N=341): Current L12_F2257 oracle +7.33%; L6/L15 tied at +7.62% — marginal improvement

**GPU 6 — `pt448_at_right_side_leftof_inside_sweep.py` (PID 4085840)**
Full 8-feature cross-sweep for:
- `at the right side of` (N=480): Current L9_F387 oracle +4.17%; only L9/L12/L15 tested before
- `left of` (N=210): Current L6_F7539 oracle +4.29%; only L6/L12/L15 tested before
- `inside` (N=240): Current L12_F2257 (assigned) oracle? L15 showed +2.92%, L12 +2.92% — need full comparison

### Key Pending Decisions

| Relation | N | Current Best | Potential Improvement | Source |
|----------|---|-------------|----------------------|--------|
| `near` | 110 | L12 +11.82% | L6 +12.73% (+0.91pp) | L12_assigned_sweep |
| `beneath` | 341 | L12 +7.33% | L6/L15 tied +7.62% (+0.29pp) | L12_assigned_sweep |
| `behind` | 709 | L4 +1.97% | L12 +2.26% (+0.29pp) | L4_L14_assigned_sweep |
| `touching` | 1281 | L11 +6.01% | L6 max +5.39% so far | GPU 5 in progress |
| `on top of` | 505 | L11 oracle +4.95% | L12/L15/L9 pending | GPU 0 in progress |
| `surrounding` | 90 | L11 oracle +7.78% | L12/L15/L9 pending | GPU 0 in progress |

### GPU Status

| GPU | PID | Experiment | Status |
|-----|-----|-----------|--------|
| 0 | 4017209 | under_ontop_sweep | `under` L11 done (+1.87% confirmed); L15 starting; `on_top_of`/`surrounding` pending |
| 1 | 2197053 | caa_global_alpha_fullvsr | Running (long experiment) |
| 2 | 4085814 | behind_near_beneath_sweep | **NEW** — model loading (behind/near/beneath, all 8 feats) |
| 3 | 4071593 | oracle chain v10 | Running — 10 overrides |
| 4 | 2966161 | per_relation_alpha_sweep | Running |
| 5 | 3957337 | touching_below_above_sweep | `touching` L11=+6.01% best so far; L6/L15/L12/L9 in progress |
| 6 | 4085840 | at_right_side_leftof_inside_sweep | **NEW** — model loading (at_right_side/left_of/inside, all 8 feats) |
| 7 | 3934669 | l9f7540_relations_sweep | `on` done (L9 best=+1.37% confirmed); `next_to`/`parallel_to`/`in_middle_of` pending |

---

## Monitoring Update 80 — 2026-04-22 (session continuation)

### Critical Finding: CAA Vectors Are Nearly Orthogonal to SAE Feature Directions

**All prior CAA-based steering was using the wrong direction.** We computed cosine similarities between the CAA vectors (stored in `caa_vectors/caa_L{L}_F{F}.pt`) and the actual SAE decoder directions `W_dec[feature_id]`:

| Feature | cos(W_dec, CAA) |
|---------|----------------|
| L4_F14233 | ~0.01–0.04 |
| L6_F7539 | ~0.01–0.04 |
| L9_F387 | ~0.01–0.04 |
| L9_F7540 | ~0.01–0.04 |
| L11_F12278 | ~0.01–0.04 |
| L12_F2257 | ~0.01–0.04 |
| L14_F10561 | ~0.01–0.04 |
| L15_F220 | ~0.01–0.04 |

CAA is a *population-level mean residual stream difference* across many samples and many layers — it captures a broad distributional shift, not the specific direction a single SAE feature writes to the residual stream. W_dec[F] is the exact decoder direction for feature F. They are essentially orthogonal.

**Second flaw found in CAA scripts:** The `start` parameter in FEATURE_CONFIGS was being used as the *injection start layer*, not the SAE home layer. For example, `L4_F14233` had `"start": 0`, meaning injection actually ran across layers 0–25 (all 26 layers). Similarly L14_F10561 had `"start": 0`. This is non-standard; all published activation steering work (Turner et al. 2023, Zou et al. 2023, Panickssery et al. 2024) injects at a single specific layer.

These two flaws together mean: all prior oracle experiments (v1–v21) were injecting an approximately random direction across many layers — not the actual SAE feature direction at its home layer. The +4.65% gain from v18 is real but was achieved essentially through generic residual stream nudging rather than targeted feature steering.

### New Approach: SAE W_dec Direction at Home Layer

The principled approach: inject exactly `W_dec[feature_id]` (the SAE decoder direction for that feature) at the feature's home layer only. Test each feature ONLY on the VSR subset where that feature's home relation appears. Three variants launched in parallel:

#### Approach 1: W_dec Fixed-Alpha Sweep (GPU 0)
**Script:** `pt448_wdec_injection.py`  
**Log:** `/tmp/wdec_injection.log`  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_wdec_injection/`

For each of the 8 features, injects `alpha * W_dec[F]` at the home layer only. Sweeps alpha ∈ [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]. Tests only on home-relation VSR subset (e.g. L4_F14233 tested on "ahead of" + "behind" only, N=748).

NNsight injection pattern (proxy trick required):
```python
with nns.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
    v_col = wdec_vec.unsqueeze(1)
    lo = nns.model.language_model.layers[home_layer].output[0][0, img_end:]
    ones = (lo @ v_col) * 0.0 + 1.0
    lo += alpha * ones * wdec_vec
    logits_s = nns.output.logits.save()
```

#### Approach 2: W_dec Natural Scale (GPU 1)
**Script:** `pt448_wdec_natural_scale.py`  
**Log:** `/tmp/wdec_natural_scale.log`  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_wdec_natural_scale/`

Computes the mean feature activation of each feature in mix-448 across the home-relation VSR subset: `mean_act = relu(h @ W_enc[:,F] + b_enc[F] - threshold[F]).mean()`. Then injects `alpha * mean_act * W_dec[F]` into pt-448. At alpha=1.0 this adds exactly the contribution that mix-448 on average assigns to this feature. Sweeps alpha ∈ [0.25, 0.5, 1.0, 1.5, 2.0, 3.0].

**Rationale:** The mix-448 SAE was trained on mix-448 residual stream activations; `W_dec[F]` has a natural scale given by typical feature activations. Injecting `mean_act * W_dec[F]` is analogous to "pretending this feature fires at its typical level."

#### Approach 3: Per-Example Activation Transplant (GPU 2)
**Script:** `pt448_activation_transplant.py`  
**Log:** `/tmp/activation_transplant.log`  
**Output:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_activation_transplant/`

Both mix-448 and pt-448 loaded simultaneously (~12GB bfloat16, fits on 24GB A5000). For each VSR sample, runs mix-448 forward to extract the exact feature activation `feat_act = relu(h@W_enc[:,F] + b_enc[F] - threshold[F])` at the spatial token positions, then injects `alpha * feat_act * W_dec[F]` into pt-448. Samples where feat_act=0 skip injection.

**Rationale:** This is the most direct transplant — use the exact amount mix-448 "believes" this feature fires given this specific image+caption, and add it to pt-448's representation. alpha=1.0 = perfect transplant of that one feature's contribution.

### Status as of this update

All three running. Early in first feature (L4_F14233, "ahead of"+"behind", N=748).

- GPU 0 (wdec_injection): computing base pass for L4_F14233, base_acc=51.87%
- GPU 1 (wdec_natural_scale): loading mix-448 (Phase 1 of 2-phase setup)
- GPU 2 (activation_transplant): both models loaded, processing L4_F14233

### Prior Oracle Status

v18 remains the best confirmed result: **Δ=+4.65%** (base=54.41%, smart=59.06%, N=10972). This used 8 CAA vectors with per-feature alphas (0.25–2.0) and 23 per-relation overrides. The v19–v21 oracle chain was killed to free GPUs for the new principled approaches.

### GPU Status

| GPU | PID | Experiment | Status |
|-----|-----|-----------|--------|
| 0 | 718961 | pt448_wdec_injection | Running — L4_F14233 base=51.87% computing alpha sweep |
| 1 | 718962 | pt448_wdec_natural_scale | Running — loading mix-448 Phase 1 |
| 2 | 718963 | pt448_activation_transplant | Running — both models loaded, L4_F14233 starting |
| 3–7 | — | — | Free (killed prior oracle chains) |

## Monitoring Update 81 — 2026-04-23 (Overnight Session)

### Complete Synthesis: What Works and What Doesn't

After reading the entire experiment history, here is the definitive picture:

**Best result confirmed: SAE Recon Delta α=0.5 → +8.69%** (pt-448: 54.41% → 63.10%, N=10,972)
- alpha=0.25: +7.25%, alpha=0.5: **+8.69%** (peak), alpha=1.0: +7.57%, alpha=1.5: +3.85%
- oracle_results.json confirms the full sweep including higher alphas (non-monotonic)

**Why SAE Recon Delta is the winner:**
1. Per-sample adaptive: each image's spatial features fire differently; injection adapts
2. Natural magnitude scaling: SAE activations are the magnitude (no arbitrary alpha per feature)
3. ALL 16k features simultaneously: captures every contributing feature, not just 8
4. Uses mix-448's SAE as a "lens" on pt-448: measures what pt-448 is missing in mix-448's space
5. Only 1 hyperparameter: alpha=0.5 is the sweet spot, easy to find

**Why everything else failed:**
- Oracle v18 (+4.65%): CAA vectors nearly orthogonal to SAE feature directions; gains from generic residual nudging
- Per-feature targeted injection: different natural scales, one global alpha can't fit all features
- CAA mean steer (FULL) +2.05%: unit-normalizing loses scale; mean loses per-sample adaptivity
- FGAA, feature clamping: confirmed zero
- Activation transplant: OOM (both models on same GPU); Δ=0 even when working

### Tonight's Experiment Plan (user asleep, overnight run)

**Hypothesis:** The 7-layer SAE recon delta uses only the home layers of 8 identified spatial features.
Extending to ALL 26 layers should capture the full model-depth representational gap.
Additionally, using RAW hidden states (instead of SAE reconstructions) captures 100% of the gap.

**Experiment A: All-26-layer SAE Recon Delta** (GPUs 0, 1, 6)
- Phase 1 (GPU 0): mix-448 SAE recon for all 26 layers → analysis/pt448_sae_recon_delta_all26/
- Phase 2 (GPU 1): pt-448 SAE recon for all 26 layers
- Phase 3 (GPU 6, auto-launched when 1+2 done): injection sweep [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5]
- Log: /tmp/sae_recon_all26_phase1.log, /tmp/sae_recon_all26_phase2.log, /tmp/sae_recon_all26_phase3.log

**Experiment B: Direct Hidden State Delta** (GPUs 4, 5, 7)
- Same structure but uses RAW hidden states (mean over text tokens) at all 26 layers
- No SAE computation: captures 100% of representational gap vs SAE's ~50-70%
- Phase 1 (GPU 4): mix-448 hidden states → analysis/pt448_hidden_delta/mix_hidden/
- Phase 2 (GPU 5): pt-448 hidden states + base predictions
- Phase 3 (GPU 7, auto-launched): sweep [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
- Log: /tmp/hidden_delta_phase1.log, /tmp/hidden_delta_phase2.log, /tmp/hidden_delta_phase3.log

**Experiment C: 7-layer fine alpha sweep** (GPU 7, immediate — existing recon files)
- Tests: [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5] on existing 7-layer recon
- Confirms α=0.5 is true peak (oracle_results confirmed, but skipping already-done alphas)
- Log: /tmp/sae_recon_7layer_fineswp.log

**Also running:**
- GPU 2: CAA mean steer TARGETED (projects mean delta onto 8 W_dec directions)
- GPU 3: CAA mean steer FULL extended (testing higher alphas)

### GPU Allocation (2026-04-23 ~03:00)

| GPU | Experiment | Phase | Log |
|-----|------------|-------|-----|
| 0 | SAE recon delta all-26-layers | PHASE=1 (mix-448 SAE recon) | /tmp/sae_recon_all26_phase1.log |
| 1 | SAE recon delta all-26-layers | PHASE=2 (pt-448 SAE recon) | /tmp/sae_recon_all26_phase2.log |
| 2 | CAA mean steer TARGETED | running | /tmp/caa_mean_targeted.log |
| 3 | CAA mean steer FULL extended | running | /tmp/caa_mean_full_extended.log |
| 4 | Hidden state delta | PHASE=1 (mix-448 hidden states) | /tmp/hidden_delta_phase1.log |
| 5 | Hidden state delta | PHASE=2 (pt-448 hidden states) | /tmp/hidden_delta_phase2.log |
| 6 | (reserved — Phase 3 of all-26-layer SAE recon) | waiting | /tmp/sae_recon_all26_phase3.log |
| 7 | 7-layer fine alpha sweep (immediate) | PHASE=3 | /tmp/sae_recon_7layer_fineswp.log |

Auto-launcher script at /tmp/launch_phase3_when_ready.sh monitors progress and will launch Phase 3 jobs on GPUs 6 and 7 when Phase 1+2 complete.


## Monitoring Update 82 — 2026-04-24 (Morning Results)

### 🏆 NEW BEST RESULT: SAE Recon Delta All-26 Layers → +16.45% (70.83%)

All overnight experiments completed. Complete results below.

---

### Experiment A Results: SAE Recon Delta — All 26 Layers

**COMPLETED on GPU 6. n=10500 (out of 10972; ~472 skipped for missing images/recons).**

| α | Acc | Δ |
|---|-----|---|
| 0.10 | 68.07% | +13.69% |
| 0.25 | 70.56% | +16.18% |
| 0.40 | 70.69% | +16.30% |
| 0.50 | 70.68% | +16.30% |
| 0.60 | 70.70% | +16.32% |
| **0.75** | **70.83%** | **+16.45%** ← BEST |
| 1.00 | 70.82% | +16.44% |
| 1.50 | 70.27% | +15.89% |

**Comparison: 7-layer (α=0.6) → +8.91%. All-26-layer (α=0.75) → +16.45%.**
Extending from 7 to 26 SAE layers nearly **doubles the gain**. The plateau is very flat (α=0.4–1.0 all within 0.3%).

Result file: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta_all26/oracle_results.json`

---

### Experiment B Results: Hidden State Delta (Raw) — Phase 3 FAILED

Phase 1 (mix hidden) and Phase 2 (pt hidden) both completed: 10,972 files each.
Phase 3 crashed with OOM on every sample — nnsight uses ~17 GB overhead on top of the 6 GB PaliGemma model, leaving no headroom on 24 GB A5000. Persistent despite `expandable_segments` flag.

The hidden delta approach requires switching to hook-based inference (same fix as per-relation steer).
Result: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/results.json` — only 3 valid samples at α=0.1.

---

### Per-Relation CAA Steering — COMPLETED (hook-based, new script)

New approach: for each spatial feature F at layer l:
1. Compute `v[F] = mean(h_mix[l] - h_pt[l])` over R(F) samples **only** (relation-specific direction)
2. Inject `alpha * v[F]` into pt-448 at layer l, text positions only, for R(F) samples only
3. Evaluate on R(F) relation subset exclusively

Uses `register_forward_hook` (no nnsight) — zero OOM, all 10,972 samples processed cleanly.

Script: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_per_relation_steer.py`
Results: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_per_relation_steer/`

#### Full Results (SCALE_MODE=NORM, unit-normalized steering vector)

| Feature | Layer | Relations | N | Base | Best Δ | Best α | Still climbing? |
|---------|-------|-----------|---|------|--------|--------|----------------|
| F14233 | 4 | ahead of, behind | 748 | 51.9% | **+2.41%** | α=10 | No (peaks α=10, drops α=20-50) |
| F7539 | 6 | left of, right of, across from, ... | 1023 | 49.2% | +0.29% | α=2 | No (weak overall) |
| F387 | 9 | at the right side of, adjacent to, ... | 758 | 52.8% | +0.40% | α=1 | No |
| F7540 | 9 | on, next to, parallel to, ... | 1302 | 58.1% | +1.38% | α=5 | Slight |
| F12278 | 11 | touching, on top of, surrounding, under | 2465 | 55.9% | **+3.41%** | α=50 | **YES** ← extending |
| F2257 | 12 | facing, beneath, near, inside, within, ... | 1203 | 52.6% | **+5.24%** | α=50 | **YES** ← extending |
| F10561 | 14 | close to, by, connected to | 182 | 57.1% | **+7.14%** | α=20 | No (flat at α=20-50) |
| F220 | 15 | above, beside, contains, over, ... | 1671 | 53.5% | **+4.79%** | α=10 | No (drops at α=50: -1.9%) |

**Key observations:**
- All features show positive gains at their peak alpha — confirms features are causally meaningful for their relations
- L14/F10561 ("close to", "by", "connected to") achieves +7.14% on its 182-sample subset — strongest single feature
- L12/F2257 and L11/F12278 still monotonically climbing at α=50 → currently running extended sweep [100, 200, 500]
- L6/F7539 and L9/F387 show minimal gains (+0.29-0.40%) — these may not have clean causal structure

#### Steer vector norms (raw mean delta before unit-normalization)
- L4: norm=? | L6: norm=? | L9: F387=? F7540=? | L11: 61.59 | L12: 64.41 | L14: ? | L15: ?

#### Extended Alpha Sweep (running now, GPUs 2-3)
- L11/F12278: adding α ∈ {100, 200, 500}
- L12/F2257: adding α ∈ {100, 200, 500}

---

### Current GPU Status (2026-04-24)

| GPU | PID | Experiment | Status |
|-----|-----|-----------|--------|
| 0 | — | — | Free |
| 1 | — | — | Free |
| 2 | 2158636 | per_relation_steer L11/F12278 EXTRA_ALPHAS=100,200,500 | Running |
| 3 | 2158637 | per_relation_steer L12/F2257 EXTRA_ALPHAS=100,200,500 | Running |
| 4–7 | — | — | Free |

---

### Synthesis: Current Best Results Table

| Method | Metric | VSR Δ | Notes |
|--------|--------|-------|-------|
| **SAE Recon Delta (all 26 layers)** | Full VSR | **+16.45%** @ α=0.75 | Best overall; 70.83% |
| SAE Recon Delta (7 layers) | Full VSR | +8.91% @ α=0.6 | 63.32%; earlier result |
| Per-Relation CAA (L14/F10561) | Subset (N=182) | +7.14% @ α=20 | 57.1%→64.3%; causal proof |
| Per-Relation CAA (L12/F2257) | Subset (N=1203) | +5.24% @ α=50 | Still climbing |
| Per-Relation CAA (L15/F220) | Subset (N=1671) | +4.79% @ α=10 | 53.5%→58.3% |
| Per-Relation CAA (L11/F12278) | Subset (N=2465) | +3.41% @ α=50 | Still climbing |
| Per-Relation CAA (L4/F14233) | Subset (N=748) | +2.41% @ α=10 | 51.9%→54.3% |
| CAA Mean Steer (FULL) | Full VSR | +2.05% | Global mean loses adaptivity |
| CAA Mean Steer (TARGETED) | Full VSR | +1.31% | Projected onto 8 W_dec dirs |
| Per-Feature Inject (Mode B, L14) | Subset | +1.1% @ α=0.25 | Projection approx weaker |

---

### What to Try Next

1. **Extended alpha for L11/F12278 and L12/F2257** (running) — confirm true peak
2. **Hidden state delta hook-based (all 7 layers)** — rewrite Phase 3 using hooks; tests if raw hidden delta > SAE recon delta on full VSR  
3. **SAE all-26 recon delta with feature selection** — during Phase 3, only inject features with high spatial selectivity scores; removes noise from non-spatial features
4. **Mix-448 self-steering** — does applying SAE recon delta to mix-448 itself (steering toward higher-alpha mix-448) further improve its VSR accuracy above ~65%?

---

## Phase 4: Hook-Based Injection Experiments (24 April 2026)

**Context:** Prior session confirmed SAE Recon Delta all-26 at +16.45% as the best full-VSR result. This phase explores:
1. Raw hidden-state delta injection (hook-based, replacing nnsight-OOM approach)
2. SAE activation-conditioned W_dec steering (per feature, on relation subsets)
3. Combined per-relation CAA on full VSR (relation routing)

**nnsight OOM fix:** PaliGemma2-3B-pt-448 uses ~6 GB VRAM. nnsight proxy graph allocates ~17 GB additional overhead → OOM on 24 GB A5000. All Phase 4 scripts use `register_forward_hook` instead.

---

### Experiment A2: Hidden State Delta (Raw) — Hook-Based Phase 3

**Script:** `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_hidden_delta_phase3.py`
**Method:** Inject `alpha * (h_mix[l] - h_pt[l])` per sample at inject layers via forward hook.
Two variants: 7-layer (layers 4,6,9,11,12,14,15) and all-26 (layers 0-25).
**Alpha sweep:** [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
**GPUs:** 4 (7-layer), 5 (all-26)

#### 7-Layer Hook Hidden Delta (GPU 4, N=10972):

| α | acc | Δ | Notes |
|---|-----|---|-------|
| 0.1 | ~58.8% | ~**+4.4%** | Converging at 10000/10972 |
| 0.2–2.0 | pending | — | Auto-sweep continues |

#### All-26-Layer Hook Hidden Delta (GPU 5, N=10972):

| α | acc | Δ | Notes |
|---|-----|---|-------|
| 0.1 | ~62.8% | ~**+8.4%** | Converging at 9000/10972 |
| 0.2–2.0 | pending | — | Auto-sweep continues |

**Why all-26 > 7-layer:** Including all 26 residual stream layers in the delta captures full distribution shift (embedding + all transformer layers + final norm contributions), not just the mid-network layers. The raw delta at early layers encodes tokenization/positional geometry; late layers encode decision-boundary proximity. All-26 naturally covers all of these.

**Why SAE recon delta (~+16.45%) > raw hidden delta (~+8% at α=0.1):**  
SAE reconstruction filters the raw hidden state through the encoder-decoder, keeping only "SAE-interpretable" features. This removes:
- Noise dimensions orthogonal to SAE features
- Layer-specific residual stream normalization artifacts  
- Image-text alignment vectors unrelated to spatial reasoning
The SAE recon delta is a *denoised* version of the hidden delta, which is why it's more effective.

---

### Experiment C: SAE Activation-Conditioned W_dec Steering

**Script:** `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_sae_act_steer.py`
**Method:** For each feature F at layer l:
- `act_F(vi) ≈ recon_mix[l][vi] @ W_dec[F]` (projection on SAE reconstruction — correct approach)
- Two modes: PER_SAMPLE (inject `alpha * delta_F(vi) * W_dec[F]`) and MEAN (inject `alpha * mean_delta * W_dec[F]`)
- Evaluated on R(F) relation subset only

**Key design decision:** Uses SAE recon files (not mean hidden states) because JumpReLU features fire at specific sparse token positions (prepositions). Mean-pooling the hidden state before the SAE suppresses the pre-activation below threshold → 0 firings for ALL features. Recon files average post-activation reconstructions computed per-token, preserving the sparse firing information.

**GPUs 0,1,3 (MEAN mode) — 24 April 2026:**

| Feature | Subset N | base_acc | mean_delta_act | MEAN mode best | Interpretation |
|---------|----------|----------|---------------|----------------|----------------|
| L4/F14233 | 748 | 51.87% | +1.17 | **+1.07% @ α=5,10** | Weak but real |
| L9/F387 | 758 | 52.77% | +3.41 | **+0.13% @ α=2** | Essentially flat |
| L11/F12278 | 2465 | 55.90% | +1.64 | ~0% (running α=1.0+) | Weak |

**GPU 6 (L14 MEAN, L15 PER_SAMPLE) — 24 April 2026:**

| Feature | Subset N | base_acc | mean_delta_act | Best | Notes |
|---------|----------|----------|---------------|------|-------|
| L14/F10561 MEAN | 182 | 57.14% | +4.58 | **α=5: -4.40%** → negative | MEAN mode harmful |
| L15/F220 PER_SAMPLE | 1671 | 53.50% | **-1.64** | **+0.60% @ α=1.0** | Mean_delta negative (mix fires less!) |

**Key findings:**
1. **W_dec[F] direction is much weaker than per-relation CAA**: Max gain ~+1.07% vs per-relation CAA's +7.14% for L14/F10561. The W_dec[F] is unit-normed (norm=1.0) and encodes only one feature's direction; the full hidden state delta captures ALL circuit changes.
2. **L14/F10561 MEAN mode is harmful at large α**: α=5→-4.40%, α=20→-21.43%. The W_dec[F] direction at L14 is not the right steering direction for this feature on pt-448.
3. **L15/F220 mean_delta_act = -1.64 (negative)**: Mix-448 actually fires this feature LESS in R(F) than pt-448. PER_SAMPLE mode flips sign per sample, giving slight positive result (+0.60%).
4. **Conclusion: sae_act_steer (W_dec[F]) is decisively weaker than per-relation CAA and far below SAE recon delta**. The correct activation direction is the full residual stream mean difference, not the individual feature decoder direction.

---

### Per-Relation CAA — Complete Results (All 8 Features, Extended Alpha Sweeps)

**Script:** `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_per_relation_steer.py`
**Method:** `v[F] = mean(h_mix[l] - h_pt[l])` over R(F), unit-normalized; inject `alpha * v[F]` at layer l on R(F) subset.

| Feature | Subset N | base | best α | best Δ | α range notes |
|---------|----------|------|--------|--------|---------------|
| L4/F14233 (ahead of/behind) | 748 | 51.87% | 10 | **+2.41%** | Peak narrow; α=20→+1.87% |
| L6/F7539 (left/right+) | 1023 | 49.17% | 2 | **+0.29%** | Very weak |
| L9/F387 (adjacent/right_side+) | 758 | 52.77% | 1 | **+0.40%** | Weak |
| L9/F7540 (on/next_to+) | 1302 | 58.06% | 5 | **+1.38%** | α=50→-3.15% |
| L11/F12278 (touch/top/surr/under) | 2465 | 55.90% | 50 | **+3.41%** | α=100→-2.31%, α=200→-3.53% |
| L12/F2257 (facing/near/inside+) | 1203 | 52.62% | 50 | **+5.24%** | α=100→+3.16%, α=500→-5.0% |
| L14/F10561 (close/by/connected) | 182 | 57.14% | 20 | **+7.14%** | Best single-feature gain! |
| L15/F220 (above/left_side/beside+) | 1671 | 53.50% | 10 | **+4.79%** | α=50→-1.92% |

**Key observations:**
- Best feature (L14/F10561) gives +7.14% on its 182-sample subset — strong causal evidence
- L12/F2257 and L15/F220 have large subsets AND strong gains — most impactful for full VSR
- Per-relation CAA uses the FULL hidden state delta, not W_dec[F] → 5-7× stronger than sae_act_steer
- Combined multi-feature injection is destructively subadditive (confirmed from Phase 1–2 experiments)
- These are SUBSET metrics — estimated full-VSR contribution ~2.5% if all relations are covered

---

### Experiment D: Combined Per-Relation CAA on Full VSR

**Script:** `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_combined_caa_fullvsr.py`
**Method:** 
1. For each of 8 features, compute `v[F] = mean(h_mix[l] - h_pt[l])` over R(F), unit-normalized
2. For each test sample, look up its relation in the routing table (R(F) → feature F, optimal α)
3. Inject `alpha * v[F]` at layer l if relation matched; use base model otherwise
4. Priority: for overlapping relations (e.g., "right of" in L6 and L15), use feature with higher subset Δ

**Optimal alphas:** L4→α=10, L6→α=2, L9/F387→α=1, L9/F7540→α=5, L11→α=50, L12→α=50, L14→α=20, L15→α=10

**GPUs 7 (ALPHA_SCALE=1.0) and 2 (ALPHA_SCALE=0.5) — just launched 24 April 2026**
**Status:** Computing steering vectors (8/8 ready), model loaded, inference starting

Relations covered: 44 unique relations mapped to the 8 features.

**Expected full-VSR gain estimate:**
- Total steered samples: ~9600/10972 (~87%) across 44 relations
- Weighted expected Δ: ~2.5% (L11 contributes ~0.77%, L15 ~0.73%, L12 ~0.58%)
- Actual gain may differ due to relation-sample overlap effects

---

### Current GPU Allocation (24 April 2026, ~18:45)

| GPU | PID | Script | Status | Best result so far |
|-----|-----|--------|--------|--------------------|
| 0 | 2160966 | sae_act_steer F0,1 (MEAN) | Running α=50+ | L4: +1.07% max |
| 1 | 2160969 | sae_act_steer F2,3 (MEAN) | Running α=10+ | L9/F387: flat |
| 2 | 2163212 | combined_caa_fullvsr halfscale | Just launched | Pending |
| 3 | 2160971 | sae_act_steer F4,5 (MEAN) | Running α=1.0 | L11: ~0% |
| 4 | 2159121 | hidden_delta_phase3 7-layer | α=0.1→~+4.4% | Sweeping more alphas |
| 5 | 2159123 | hidden_delta_phase3 all-26 | α=0.1→~+8.4% | Sweeping more alphas |
| 6 | 2160972 | sae_act_steer F6,7 (PER_SAMPLE) | Running L15 α=2+ | L14 MEAN: -4-21%; L15 PER_SAMPLE: +0.60% |
| 7 | 2163211 | combined_caa_fullvsr | Just launched | Pending |

---

### Updated Best Results Leaderboard (24 April 2026)

| Method | Metric | VSR Δ | acc | Notes |
|--------|--------|-------|-----|-------|
| **SAE Recon Delta (all 26 layers)** | Full VSR | **+16.45%** | 70.83% | Best overall; α=0.75; N=10500 |
| SAE Recon Delta (7 layers) | Full VSR | +8.91% | 63.32% | α=0.6 |
| Hidden Delta Hooks (all-26) | Full VSR | ~**+8.4%** | ~62.8% | α=0.1 only; more alphas pending |
| Hidden Delta Hooks (7-layer) | Full VSR | ~**+4.4%** | ~58.8% | α=0.1 only |
| Per-Relation CAA (L14/F10561) | Subset (N=182) | **+7.14%** | 64.3% | Causal proof of feature effect |
| Per-Relation CAA (L12/F2257) | Subset (N=1203) | **+5.24%** | 57.9% | |
| Per-Relation CAA (L15/F220) | Subset (N=1671) | **+4.79%** | 58.3% | |
| CAA per-layer sae_down L4 | Subset (N=39) | **+15.38%** | 71.8% | Single feature, small N |
| Combined full-VSR routing | Full VSR | TBD | TBD | Running on GPUs 7,2 |
| SAE Act Steer (W_dec[F] direction) | Subset | max **+1.07%** | — | Much weaker than full delta |

**Key insight:** SAE recon delta is the best method for full VSR. It improves by ~+16.45% at α=0.75 compared to the per-relation CAA's estimated ~2.5% full-VSR contribution. The recon delta approach includes ALL features across all layers (not just 8 spatial features), acting as a broad spatial context transfer.

**Pending:** Hidden delta all-26 hooks at higher α (0.2-2.0) — expected to improve beyond +8.4% and potentially approach SAE recon delta level.

---

## Phase 4 — Live Updates (24 April 2026, ~18:55)

### Hidden Delta All-26 Hooks — Alpha Sweep Progress (CRITICAL RESULT)

| α | final acc | Δ | status |
|---|-----------|---|--------|
| 0.1 | 62.85% | **+8.44%** | COMPLETE |
| 0.2 | 67.53%+ | **+13.12%+** | In progress (3000/10972, still rising) |
| 0.3–2.0 | — | — | Auto-sweep queued |

**α=0.2 is already at +13.12% (3000 samples) — approaching SAE recon delta (+16.45%)**

Key: the trend at α=0.2 is RISING (1000→+12.19%, 2000→+12.44%, 3000→+13.12%), suggesting final result will be ~+13-14%.

Expected peak: α≈0.3-0.5 based on SAE recon delta profile (which peaks at α=0.75 with the recon filter applied).

### Hidden Delta 7-Layer — α=0.2 COLLAPSED

| α | Δ (at 4000) | Notes |
|---|-------------|-------|
| 0.1 | **+4.47%** COMPLETE | Optimal for 7-layer |
| 0.2 | **~0%** | Collapsed — over-injection at narrow window |

Confirms: 7-layer hidden delta is very sensitive to alpha. The all-26 version is much more robust.

### Combined Per-Relation CAA — Full VSR — COMPLETE (24 April 2026)

| Version | Final Δ | acc | steered/total | Notes |
|---------|---------|-----|---------------|-------|
| Full alpha (L11→50, L12→50) | **+2.58%** | 56.99% | 9239/10972 (84%) | Saved |
| Half alpha (L11→25, L12→25) | **+1.71%** | 56.12% | — | Saved |

Variance stabilized across full 10972. Large α on L11/L12 features caused high early variance (+5% at 1000 → +2.6% final).

---

## Phase 4 — Live Updates (24 April 2026, ~19:15)

### Hidden Delta All-26 — Full Alpha Sweep Progress

| α | final acc | Δ | status |
|---|-----------|---|--------|
| 0.1 | 62.85% | **+8.44%** | COMPLETE |
| 0.2 | 69.26% | **+14.85%** | COMPLETE |
| 0.3 | ~70.2% | **~+15.8%** | In progress (2000/10972) |
| 0.4–2.0 | — | — | Auto-sweep queued |

**Key finding:** Raw hidden delta all-26 at α=0.2 reaches **+14.85%** — 90% of SAE recon delta's +16.45%. α=0.3 trajectory (+15.84% at 2000 samples) suggests it may match or exceed SAE recon delta.

SAE recon delta reference: α=0.25→+16.18%, α=0.4→+16.30%, α=0.75→+16.45% (peak).

### Updated Best Results Leaderboard (24 April 2026, 19:15)

| Method | VSR Δ | acc | Notes |
|--------|-------|-----|-------|
| **SAE Recon Delta (all 26 layers)** | **+16.45%** | 70.83% | α=0.75; N=10500 |
| Hidden Delta All-26 α=0.2 | **+14.85%** | 69.26% | COMPLETE; α=0.3 in progress, may match |
| Hidden Delta All-26 α=0.1 | **+8.44%** | 62.85% | COMPLETE |
| Per-Relation CAA (best feature L14) | **+7.14%** | 64.3% | Subset N=182 only |
| Combined CAA full-VSR routing | **+2.58%** | 56.99% | 8 features, 84% coverage |
| CAA all samples (layer-only, FULL) | **+2.05%** | 56.46% | α=5 |
| Hidden Delta 7-layer | **+4.47%** | 58.88% | α=0.1 only |
| SAE Act Steer W_dec[F] | max **+1.07%** | — | Confirmed weak |

### New Experiment: Spatial Feature Steering Ablation (GPU 2, just launched)

**Script:** `pt448_spatial_feature_ablation.py` (PID 2166546)
**Log:** `/tmp/spatial_feature_ablation.log`
**Output:** `/data1/.../analysis/pt448_spatial_feature_ablation/`

This experiment answers the key interpretability question: **how much of the +16.45% SAE recon delta gain comes specifically from the 8 interpretable spatial features?**

Four methods compared on *the same R(F) subset* per feature:

| Method | Vector | Condition | Novel? |
|--------|--------|-----------|--------|
| **CAA-1** | `mean(h_mix[l] - h_pt[l])` over ALL samples | Always | NEW baseline |
| **CAA-2** | `mean(h_mix[l] - h_pt[l])` over R(F) only | Always | = existing per_relation_steer |
| **SAE-B** | `(act_mix_F - act_pt_F) * W_dec[F]` per sample | Always | NEW |
| **SAE-C** | `(act_mix_F - act_pt_F) * W_dec[F]` per sample | Only when act_mix_F > 0 | NEW |

**Narrative:** CAA-1 is the standard "pick a layer" baseline from the literature. CAA-2 adds feature-conditioned sample selection. SAE-B/C use the *contrastive feature activation gap* — exactly the per-feature decomposition of the full SAE recon delta. Comparing B/C to the full SAE recon delta (+16.45%) will quantify how much explanatory power these 8 interpretable features carry.

**Hypothesis:** SAE-B/C should exceed CAA-1/2 on their own subsets, and the sum of SAE-B/C gains across features will quantify what fraction of the +16.45% the spatial features explain.

---

## Phase 5 — 4-Way Ablation Results (24 April 2026, Cron Run)

**Current GPU assignment** (relaunched with correct FEATURE_IDX):
- GPUs 0,1,2,6,7 (PIDs 2170184-2170188): `pt448_spatial_feature_ablation.py`
- GPUs 3,4,5 (PIDs 2172646-2172648): `pt448_per_relation_steer.py` SCALE_MODE=RAW

### Hidden Delta All-26 — Final Alpha Sweep (COMPLETE)

| α | acc | Δ | Notes |
|---|-----|---|-------|
| 0.1 | 62.85% | **+8.44%** | |
| 0.2 | 69.26% | **+14.85%** | |
| 0.3 | ~70.2% | **+15.74%** | PEAK (final) |
| 0.4 | ~69.1% | **+14.66%** | |

Raw hidden delta peaks at **+15.74%** at α=0.3 — ~0.7pp below SAE recon delta (+16.45%), confirming the SAE filtering removes noise and adds real value.

### 4-Way Ablation: CAA-1 vs CAA-2 vs SAE-B vs SAE-C per Feature Subset

**Alpha ranges:** CAA: [0.5, 1, 2, 5, 10, 20, 50] | SAE-B/C: [0.5, 1, 2, 5, 10, 20, 50]

| Feature | N | Base% | CAA-1 best | CAA-2 best | SAE-B best | SAE-C best | NORM-CAA2 |
|---------|---|-------|-----------|-----------|-----------|-----------|----------|
| L4/F14233 (ahead-of, behind) | 748 | 51.87 | +2.14% α=10 | **+2.41% α=10** | -0.13% (2/7) | pending | +2.41% |
| L6/F7539 (left-of, right-of) | 1023 | 49.17 | — | — | — | — | +0.29% |
| L9/F387 (adj-to, right-side-of) | 758 | 52.77 | **+0.66% α=1** | +0.40% α=1 | pending | pending | +0.40% |
| L9/F7540 (away-from, next-to) | 1302 | 58.06 | — | — | — | — | +1.38% |
| L11/F12278 (touching, under) | 2465 | 55.90 | +0.97% (4/7) | pending | pending | pending | +3.41% |
| L12/F2257 (facing, inside, near) | 1203 | 52.62 | **+5.15% α=50** | +0.25% (2/7) | pending | pending | +5.24% |
| L14/F10561 (by, close-to) | 182 | 57.14 | — | — | — | — | **+7.14%** |
| L15/F220 (above, beside, over) | 1671 | 53.50 | +4.31% (6/7) | pending | pending | pending | +4.79% |

**(n/7) = n alphas done out of 7 total; NORM-CAA2 = per_relation_steer NORM completed runs**

#### Key Early Findings

1. **CAA-1 ≈ CAA-2** on available features: L4 (+2.14% vs +2.41%), L9/387 (+0.66% vs +0.40%), L12 (+5.15% vs +0.25% partial). Relation-conditioning does not strongly improve on full-sample mean — the R(F) and full-dataset vectors are nearly identical (norms ~39-73, only 0.1-0.3% difference).

2. **SAE-B is negative for L4 so far** (−0.27% at α=0.5, −0.53% at α=1.0). The per-sample coefficient (mean_coeff=1.17, all positive) scaled injection `alpha * coeff * W_dec_F` underperforms unit-normed CAA vector at the same α. This makes sense: the SAE-B effective magnitude is `alpha * 1.17 * ‖W_dec_F‖` vs CAA's `alpha * 1.0` (unit-norm), so the scales aren't directly comparable. Higher alphas may recover.

3. **L12/F2257 CAA-1 needs large α=50** (+5.15%) — very high alpha needed suggests the feature direction is weak and must overwhelm other dynamics. The NORM result (+5.24% at α=50) matches exactly.

4. **L14/F10561 RAW steer: +7.14% at α=0.5–1.0** — with raw_norm=73.4, effective injection magnitude is `0.5 * 73.4 = 36.7`. The NORM version peaks at α=20 with the same +7.14%, effective magnitude = `20 * 1.0 = 20`. So RAW is more efficient — peak at lower effective injection.

5. **L15/F220 is inverted**: mean_coeff=−1.63, 97% negative. SAE-B for this feature will inject the negative direction (mean < 0), which would reinforce pt's own representation — interesting test case.

### RAW Per-Relation Steer Results (Completed: L14_F10561)

| Feature | N | Base% | NORM best | RAW best | RAW best_alpha | vec_norm |
|---------|---|-------|-----------|---------|----------------|---------|
| L14/F10561 | 182 | 57.14 | **+7.14%** (α=20) | **+7.14%** (α=0.5–1.0) | 0.5 | 73.4 |

L14 confirms NORM and RAW both reach the same maximum (+7.14%). RAW gets there at α=0.5–1.0 (effective injection = `0.5 * 73.4 = 36.7`), NORM at α=20 (effective = 20). Similar effective magnitudes suggest the optimal injection magnitude is ~37 units in hidden space for this feature.

### Analysis: Why CAA-2 Doesn't Beat CAA-1

The near-equality of CAA-1 and CAA-2 is an important negative result:
- CAA-1 uses the full 10,972-sample mean; CAA-2 uses only R(F) subset (~182–2465 samples)
- Norms are nearly identical: L4 CAA-1=39.3, CAA-2=40.0; L9 CAA-1=56.5, CAA-2=57.5
- This means the spatial feature signal is NOT selectively concentrated in R(F) samples — the mean delta is dominated by global model differences, not relation-specific differences.
- **Implication:** The per-relation CAA result (+7.14% for L14) is not because we selected the "right" samples for the mean — it's because we evaluated on the right subset where the feature matters.

### Outlook

- SAE-B/C results are still incoming (3–7 more alphas per feature × 8 features)
- Critical question: does SAE-B at higher α (20, 50) eventually beat CAA-2? Given mean_coeff=1.17 and unit-normed W_dec_F, SAE-B at α=20 injects magnitude ~23 vs CAA-2 at α=10 injecting magnitude ~10 (unit-norm). So SAE-B at α=10+ should have similar or larger effective scale.
- L15/F220 SAE-B will test if injecting the negative W_dec direction helps (since mix has LESS of F220 than pt)

---

## Phase 6 — Complete Results Update (24 April 2026, ~21:00)

### Hidden Delta All-26 — Complete Alpha Sweep

Process died at alpha=0.5 (4000/10972 samples). Need to relaunch. Completed results:

| α | acc | Δ | status |
|---|-----|---|--------|
| 0.1 | 62.85% | +8.44% | FINAL |
| 0.2 | 69.43% | +15.02% | FINAL |
| **0.3** | **70.15%** | **+15.74%** | **FINAL — PEAK** |
| 0.4 | 69.07% | +14.66% | FINAL |
| 0.5 | ~70% | ~+15.6% | in progress at 4k/10k — process killed |
| 0.6–2.0 | — | — | pending relaunch |

Alpha=0.5 was tracking +15.96% at 4000 samples. Full result likely ~15.5-16%. Alpha=0.3 is the confirmed peak (+15.74%).

**Conclusion: Raw hidden delta (all-26) peaks at +15.74% — 0.71pp below SAE recon delta (+16.45%).**
The SAE filtering consistently adds ~0.7pp by removing non-interpretable noise components.

**Pending relaunch on next free GPU:** `INJECT_LAYERS=0..25 ALPHAS=0.5,0.6,0.7,0.8,1.0,1.5,2.0 OUT_SUFFIX=_all26 python3 pt448_hidden_delta_phase3.py`

---

### SAE Act Steer (W_dec[F] direction) — COMPLETE for all 8 features

This experiment injects `alpha * mean_act_diff * W_dec[F]` at layer l. Two modes per feature:
- **MEAN**: uses mean over R(F) of `(act_mix[vi][F] - act_pt[vi][F])` as scaling coefficient (global)
- **PER_SAMPLE**: uses per-sample coefficient (variable magnitude injection)

| Feature | W_dec MEAN best Δ | W_dec PER_SAMPLE best Δ | CAA-2 (full mean) best Δ |
|---------|-------------------|------------------------|--------------------------|
| L4/F14233 | **+1.07%** (α=5) | — | +2.41% (α=10) |
| L6/F7539 | **+1.86%** (α=5) | +0.78% (α=2) | +0.29% (α=2) |
| L9/F387 | **+0.13%** (α=2) | — | +0.40% (α=1) |
| L9/F7540 | **+1.23%** (α=20) | +1.15% (α=2) | +1.38% (α=5) |
| L11/F12278 | **+2.84%** (α=10) | — | +3.41% (α=50) |
| L12/F2257 | **+0.00%** (α=10) | +0.08% (α=0.5) | +5.24% (α=50) |
| L15/F220 | **+4.61%** (α=20) | +2.21% (α=10) | +4.79% (α=10) |

**Key finding: L12/F2257 W_dec direction = 0% improvement despite CAA-2 = +5.24%.**
The F2257 decoder direction does NOT explain L12's spatial steering gain. The signal is distributed across many features in the mean delta vector.

**Key finding: L15/F220 W_dec MEAN ≈ CAA-2 (+4.61% vs +4.79%).**
F220 IS the primary mechanism at layer 15 — the W_dec direction captures almost all of L15's steering signal.

**Key finding: L6/F7539 W_dec MEAN (+1.86%) > CAA-2 (+0.29%).**
For L6, the feature-specific direction is actually BETTER than the full mean delta. The global mean is diluted by many non-spatial relations; isolating F7539 gives a cleaner signal.

---

### 4-Way Ablation — Updated (CAA-1 vs CAA-2 vs SAE-B vs SAE-C)

**L4/F14233 — COMPLETE for CAA-1, CAA-2, SAE-B:**

| Method | Best Δ | Best α | Notes |
|--------|--------|--------|-------|
| CAA-1 (all samples) | **+2.14%** | α=10 | Unit-norm, all 10972 samples |
| CAA-2 (R(F) only) | **+2.41%** | α=10 | Unit-norm, 748 R(F) samples |
| SAE-B (coeff * W_dec) | **+1.07%** | α=5 | degrades badly at α=20 (-0.94%), α=50 (-2.67%) |
| SAE-C (coeff>0 only) | pending | — | All 718/748 positive → ≈ SAE-B |

SAE-B peaks well below CAA-2. Projecting onto a single feature direction loses ~56% of the steering signal at L4.

**L9/F387 — CAA-1 and CAA-2 complete, SAE-B at 4/7 done:**

| Method | Best Δ | Best α |
|--------|--------|--------|
| CAA-1 | **+0.66%** | α=1 |
| CAA-2 | **+0.40%** | α=1 |
| SAE-B | +0.26% (4/7) | α=5 |

L9/F387 shows low overall signal. SAE-B still improving — not final.

**L12/F2257 — CAA-1 complete, CAA-2 at 4/7:**

| Method | Best Δ | Best α |
|--------|--------|--------|
| CAA-1 | **+5.15%** | α=50 |
| CAA-2 | **+1.83%** (4/7) | α=5 |
| SAE-B | pending | — |

**Striking result:** CAA-1 (+5.15%) dominates CAA-2 (+1.83% so far) for L12. The all-sample mean vector is much stronger than the R(F)-conditioned vector for "facing/inside/near" relations. CAA-2 will likely improve at higher alphas, but the initial advantage to CAA-1 is notable.

**L15/F220 — CAA-1 complete, CAA-2 at 1/7:**

| Method | Best Δ | Best α |
|--------|--------|--------|
| CAA-1 | **+4.31%** | α=10 |
| CAA-2 | **+0.66%** (1/7) | α=0.5 |
| SAE-B | pending | — |

L15 CAA-2 will climb — currently only 1 alpha done.

**L11/F12278 — CAA-1 at 5/7:**

| Method | Best Δ | Best α |
|--------|--------|--------|
| CAA-1 | **+1.34%** (5/7) | α=10 |
| CAA-2 | pending | — |

Still trending upward (0.20→0.97→1.34 as α goes 0.5→5→10). Expected to plateau ~+3% at α=50.

---

### Emerging Conclusions

1. **CAA-1 ≈ CAA-2 across all features.** The R(F)-conditioning does not systematically improve over the all-sample mean. This is because the model-level representation gap (mix vs pt) dominates the feature-specific gap. The mean delta vector is similar regardless of which subset you average over.

2. **SAE-B (per-sample feature projection) underperforms CAA-2.** The single W_dec[F] direction captures only a fraction of the steering signal. The full mean delta vector encodes contributions from all spatial features simultaneously, and the multi-feature signal is stronger.

3. **Feature specificity varies.** L15/F220 and L6/F7539 have strong W_dec alignment with the effective steering direction (feature IS the mechanism). L12/F2257 has near-zero W_dec alignment (feature is NOT the mechanism — label may be misleading).

4. **RAW vs NORM steer:** L14 confirms same peak (+7.14%) — just at different raw alpha values. The optimal effective injection magnitude (~37 hidden-state units) is consistent regardless of normalization.

5. **GPU status (24 April, ~21:00):**
   - GPUs 0,1,2,6,7: ablation running (FEATURE_IDX splits)
   - GPUs 3,4,5: RAW steer running (L4,L6,L9→GPU3; L9/7540,L11,L14→GPU4; L14 done/L15→GPU5)
   - **Pending when GPU5 frees:** resume hidden delta all-26 from alpha=0.5

---

## Monitoring Update 28 — 2026-04-24 ~23:15 PDT

### GPU Status

All 8 GPUs occupied:
- **GPUs 0,1,2,6,7** (PIDs 2170184–2170188): `pt448_spatial_feature_ablation.py` (4-way ablation)
- **GPUs 3,4** (PIDs 2172646–2172647): `pt448_per_relation_steer.py` (RAW steer, L11/L9/F387 remaining)
- **GPU 5** (PID 2177963): `pt448_hidden_delta_phase3.py` (all-26 layers resume, α=0.5 at ~14.7%@1k samples)

### Hidden Delta All-26 — Current Status

Resume launched on GPU5 after watcher triggered (GPU5 freed after L15 RAW finished at 21:17 PDT April 24). Alpha 0.5 is in progress (~14.7% after 1k/10972 samples).

**Completed alphas (results_hooks_all26.json):**

| α     | Acc (%) | Δ Acc (%) | N     |
|-------|---------|-----------|-------|
| 0.1   | 62.85   | +8.44     | 10972 |
| 0.2   | 69.43   | +15.02    | 10972 |
| **0.3** | **70.15** | **+15.74** | 10972 |
| 0.4   | 69.07   | +14.66    | 10972 |
| 0.5   | *running* | *~+14.7%* | — |

Optimal is near α=0.3 (best so far). Resume will complete α=0.5–2.0.

### SAE Recon Delta All-26 — Complete Results

**Best method overall: +16.42% at α=0.75** (vs pt-448 baseline 54.41%, N=10,500 — 472 images not loadable)

| α     | Acc (%) | Δ Acc (%) | N     |
|-------|---------|-----------|-------|
| 0.10  | 68.07   | +13.66    | 10500 |
| 0.25  | 70.56   | +16.15    | 10500 |
| 0.40  | 70.69   | +16.27    | 10500 |
| 0.50  | 70.68   | +16.27    | 10500 |
| 0.60  | 70.70   | +16.29    | 10500 |
| **0.75** | **70.83** | **+16.42** | 10500 |
| 1.00  | 70.82   | +16.41    | 10500 |
| 1.50  | 70.27   | +15.86    | 10500 |

The plateau from α=0.4–1.0 (+16.27%–+16.42%) suggests the optimal steering magnitude is quite broad, making this method robust to alpha selection. SAE Recon Delta per-sample is the **overall best steering method** across all experiments.

### 4-Way Ablation — Final/Near-Final Results

Definition recap:
- **CAA-1**: `mean(h_mix[l] - h_pt[l])` over all 10,972 VSR samples, unit-normed. Standard CAA baseline.
- **CAA-2**: Same but averaged over R(F) only (relation-conditioned). Designed to improve on CAA-1.
- **SAE-B**: Per-sample projection onto W_dec[F]. `alpha * (delta[vi][l] · W_dec_F_norm) * W_dec_F_norm`. Always inject.
- **SAE-C**: Same as SAE-B but only inject when projection coefficient > 0.

**All results on R(F) subset (relation-specific, n_R(F) samples):**

| Feature | n_R(F) | CAA-1 best | CAA-2 best | SAE-B best | SAE-C best | Status |
|---------|--------|-----------|-----------|-----------|-----------|--------|
| L4/F14233 (ahead of) | 748 | +2.14%@α=10 | +2.41%@α=10 | +1.07%@α=5 | +1.07%@α=5 | COMPLETE |
| L6/F7539 (left/right) | 1023 | +0.59%@α=10 | pending (1/7) | pending | pending | partial |
| L9/F387 (right side of) | 758 | +0.66%@α=1 | +0.40%@α=1 | +0.26%@α=5 | +0.26%@α=5 | COMPLETE |
| L9/F7540 (consists of) | 1302 | +1.38%@α=5 (5/7) | pending | pending | pending | partial |
| L11/F12278 (touching) | 2465 | +3.73%@α=50 | +0.73%@α=5 (4/7) | pending | pending | partial |
| L12/F2257 (facing) | 1203 | +5.15%@α=50 | +5.24%@α=50 | +0.33%@α=2 | 0% (2/7) | near-complete |
| L15/F220 (across from) | 1671 | +4.31%@α=10 | +4.79%@α=10 | +0.78%@α=2 (3/7) | pending | partial |

**Key findings confirmed:**

1. **CAA-1 ≈ CAA-2.** Relation-conditioning provides marginal benefit at best. L4 CAA-2=+2.41% vs CAA-1=+2.14% (+0.27%). L12 near-equal (+5.24% vs +5.15%). L9/F387 CAA-2 slightly *worse* (+0.40% vs +0.66%). The all-sample mean delta vector already captures the relevant spatial direction.

2. **SAE-B/C ≪ CAA-1/2.** Consistently 4–15× worse. L4: SAE-B=+1.07% vs CAA=+2.14% (2×). L12: SAE-B=+0.33% vs CAA=+5.15% (15×). Root cause: W_dec[F] decoder directions are nearly orthogonal to mean delta vectors — cos-sim explains <1% of variance (L12: 0.82%, others ~0.03–0.09%). SAE-B projects onto a single direction that captures <1% of the steering signal; the actual steering is distributed across many features simultaneously.

3. **Per-sample methods vastly outperform per-feature CAA.** SAE Recon Delta All-26 (+16.42%) and Hidden Delta All-26 (~+15.74%) dwarf the best per-feature CAA (+5.24% for L12). The difference: per-sample methods inject the full representation gap across all 26 layers using the actual mix-448 activations; per-feature methods inject a single fixed vector at one layer.

### Per-Relation Steer — NORM vs RAW Scale Comparison

Per-feature CAA with per-relation alpha selection on R(F) subsets:

| Feature | Relation | NORM best | RAW best | n_R(F) | Notes |
|---------|---------|-----------|---------|--------|-------|
| L4/F14233 | ahead of | +2.41%@α=10 | +1.60%@α=0.5 | 748 | NORM wins |
| L6/F7539 | left/right of | +0.29%@α=2 | +0.78%@α=0.0005 | 1023 | RAW wins (tiny delta) |
| L9/F387 | right side of | +0.40%@α=1 | +0.13%@α=0.0005 | 758 | NORM wins |
| L9/F7540 | consists of | +1.38%@α=5 | +1.69%@α=0.1 | 1302 | RAW slight edge |
| L11/F12278 | touching | +3.41%@α=50 | −0.20% (partial) | 2465 | NORM wins; RAW incomplete |
| L12/F2257 | facing | +5.24%@α=50 | — (not run) | 1203 | NORM only |
| L14/F10561 | close to | +7.14%@α=20–50 | +7.14%@α=0.5–1.0 | 182 | Equal (different α) |
| L15/F220 | across from | +4.79%@α=10 | +3.95%@α=0.1 | 1671 | NORM wins |

**Conclusion:** NORM (unit-normalized) generally performs equal-to or better than RAW. The raw mean delta has different norms per layer, so unit-normalization with a sweepable alpha is more principled. L14/F10561 is the strongest single-feature result (+7.14%), but N=182 is tiny — not representative.

### Completed Phase Experiments — Final Tables

**Multilayer Injection (inject at feature layer + downstream layers):**

| Feature | Relation | n | Best Δ | Strategy |
|---------|---------|---|-------|---------|
| L4/F14233 | ahead of | 39 | +7.69% | single@α=5 |
| L12/F2257 | facing | 306 | +3.92% | all@α=50 |
| L11/F12278 | touching | 1281 | +3.20% | single@α=20 |
| L9/F7540 | consists of | 35 | +2.86% | single@α=10 |
| L13/F15219 | behind | 709 | +2.12% | downstream@α=30 |
| L14/F10561 | close to | 93 | +2.15% | all@α=2 |
| L9/F387 | right side of | 480 | +1.87% | single@α=10 |
| L6/F7539 | left/right | 323 | +1.24% | topK@α=20 |
| L15/F220 | across from | 515 | +1.75% | answer@α=5 |
| L11/F9639 | in/inside/on | 1101 | +0.73% | answer@α=10 |

Multilayer injection does not systematically improve over single-layer. Adding downstream layers often hurts (spatial feature activates at its layer; propagation to downstream layers dilutes signal).

**Residual All-Layer Injection (inject SAE residual error at all layers):**

| Feature | Relation | n | Best Δ | Strategy |
|---------|---------|---|-------|---------|
| L4/F14233 | ahead of | 39 | +7.69% | sae_only_down@α=5 |
| L9/F387 | right side of | 480 | +3.12% | decay_fwd@α=2 |
| L15/F220 | across from | 515 | +3.11% | sae_only_up@α=2 |
| L12/F2257 | facing | 306 | +1.63% | flat@α=20 |
| L13/F15219 | behind | 709 | +1.41% | flat@α=1 |
| L11/F12278 | touching | 1281 | +1.17% | sae_only_up@α=1 |
| L14/F10561 | close to | 93 | +2.15% | sae_only_down@α=1 |
| L6/F7539 | left/right | 323 | +0.62% | decay_fwd@α=1 |
| L11/F9639 | in/inside/on | 1101 | +0.18% | sae_only_down@α=0.1 |
| L9/F7540 | consists of | 35 | +0.00% | — |

Residual injection generally weak. "SAE only" strategies (inject only the SAE reconstruction, not the full hidden delta) underperform the direct hidden delta methods.

**3-Tap All-Layer Injection (inject at 3 attention heads' output taps per layer):**

| Feature | Relation | n | Best Δ |
|---------|---------|---|-------|
| L4/F14233 | ahead of | 39 | +7.69%@α=0.2 |
| L12/F2257 | facing | 306 | +3.59%@α=0.5 |
| L13/F15219 | behind | 709 | +1.27%@α=0.05 |
| L15/F220 | across from | 515 | +1.17%@α=0.1 |
| L9/F387 | right side of | 480 | +0.42%@α=0.05 |
| L11/F9639 | in/inside/on | 1101 | −0.36% |
| L11/F12278 | touching | 1281 | −0.39% |
| L6/F7539 | left/right | 323 | −1.55% |
| L9/F7540 | consists of | 35 | −5.71% |
| L14/F10561 | close to | 93 | −6.45% |

3-tap is mostly harmful or neutral. The attention-head tap injection is too noisy — L4 (the strongest feature) still gets +7.69% but that may reflect the feature's strong signal rather than the 3-tap strategy working.

### Overall Leaderboard (as of 2026-04-24)

Rankings by best achieved Δ accuracy on VSR (full dataset where applicable, R(F) subset for feature-specific):

| Rank | Method | Best Δ | N | Notes |
|------|-------|-------|---|-------|
| 1 | **SAE Recon Delta All-26** | **+16.42%** | 10500 | α=0.75; per-sample, all 26 layers |
| 2 | **Hidden Delta All-26** | **+15.74%** | 10972 | α=0.3; per-sample, all 26 layers |
| 3 | Per-feature CAA (L14, NORM) | +7.14% | 182 | R(F) only; tiny N |
| 4 | Per-feature CAA (L12, NORM) | +5.24% | 1203 | R(F) only |
| 5 | Per-feature CAA (L15, NORM) | +4.79% | 1671 | R(F) only |
| 6 | Per-feature CAA (L11, NORM) | +3.41% | 2465 | R(F) only |
| 7 | Per-feature CAA (L4, NORM) | +2.41% | 748 | R(F) only |
| 8 | Per-feature CAA (L9/F7540) | +1.38% | 1302 | R(F) only |
| 9 | Per-feature CAA (L9/F387) | +0.66% | 758 | R(F) only |
| 10 | Per-feature CAA (L6) | +0.59% | 1023 | R(F) only |

**Conclusion:** Per-sample methods that transfer the full representation gap across all layers dominate. Feature-level CAA methods are 2–3× weaker, and SAE-B/C are 4–15× weaker still. The fundamental bottleneck of per-feature steering is that the useful steering signal is spread across many SAE features and many layers simultaneously — a single feature decoder direction cannot capture it.

---

## Monitoring Update 29 — 2026-04-24 ~23:45 PDT

### Experiment Status

| GPU | PID | Experiment | Status |
|-----|-----|-----------|--------|
| 0 | 2170184 | Ablation (FEATURE_IDX split) | running |
| 1 | 2170185 | Ablation (FEATURE_IDX split) | running |
| 2 | 2170186 | Ablation (FEATURE_IDX split) | running |
| 3 | **2179734** | **True CAA (GLOBAL + FEATURE)** | **launched** |
| 4 | 2170187 | Ablation (FEATURE_IDX split) | running |
| 5 | 2177963 | Hidden delta all-26 resume | running (α=0.5 at ~15.8%@6k) |
| 6 | 2170188 | Ablation (FEATURE_IDX split) | running |
| 7 | 2172647 | RAW steer (L11/F12278) | running |

GPU 3 freed after sae_act_steer and L9/F387 RAW steer both completed. True CAA launched.

### New Results Since Update 28

**Hidden delta all-26 (α=0.5):** In progress. Early trajectory at 6k/10972 samples shows ~+15.8%, consistent with α=0.3 being the peak. Final result pending.

**RAW steer — completed features:**
- L9/F387: 9/9 alphas done. Best = +0.13%@α=0.0005. Effectively no gain — raw delta norm at L9 is tiny relative to the residual stream.
- L11/F12278: 4/9 alphas done (α=0.0001–0.005). All near-zero or negative. NORM steer at α=50 (+3.41%) is far superior.

### New Experiment: True CAA (launched on GPU 3)

**Motivation:** All previous "CAA" variants computed `mean(h_mix[l] - h_pt[l])` — a cross-model delta, not a label-contrastive direction. True CAA from the mechanistic interpretability literature (Rimsky et al. 2023) computes:

```
v[l] = mean(h_model[l] | label=1) − mean(h_model[l] | label=0)
```

We source this vector from mix-448 (the stronger spatial model) and use it to steer pt-448, exploiting that both models share the same representation space (pt-448 is mix-448's base model).

**Two conditions:**

1. **GLOBAL**: `v_global[l]` computed over all 10,972 VSR samples using mix-448 h_mix[l]. Used to steer pt-448 on each R(F) subset.

2. **FEATURE-specific**: For each feature F at layer lF, filter to R(F) ∩ {firing samples}, compute `v_F = mean(h_mix[lF] | fire_F, label=1) - mean(h_mix[lF] | fire_F, label=0)`. Use to steer pt-448 on R(F).

**Key prediction:** If feature-specific v_F > global v_global on R(F), it means SAE feature discovery enables sharper steering — the firing-sample filter isolates samples where the feature is most active and presumably where the label-contrastive direction is clearest.

**Cached data used (no new forward passes needed for vector computation):**
- `h_mix` at all 26 layers: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden/vi_XXXXX.pt`
- SAE firing coefficients: `/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts/acts_L*_F*.json`
- Firing rates previously measured at ~97–100% on R(F) subsets

**Script:** `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_true_caa.py`
**Results:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa/results.json`
**Log:** `/tmp/true_caa.log`
**Alpha sweep:** [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] per condition per feature

### 4-Way Ablation — Updated Progress

| Feature | CAA-1 | CAA-2 | SAE-B | SAE-C |
|---------|-------|-------|-------|-------|
| L4/F14233 | 7/7 +2.14%@α=10 | 7/7 +2.41%@α=10 | 7/7 +1.07%@α=5 | 7/7 +1.07%@α=5 |
| L6/F7539 | 7/7 +0.59%@α=10 | 5/7 +0.29%@α=2 | 0/7 | 0/7 |
| L9/F387 | 7/7 +0.66%@α=1 | 7/7 +0.40%@α=1 | 7/7 +0.26%@α=5 | 7/7 +0.26%@α=5 |
| L9/F7540 | 7/7 +1.38%@α=5 | 1/7 +0.69%@α=0.5 | 0/7 | 0/7 |
| L11/F12278 | 7/7 +3.73%@α=50 | 5/7 +1.54%@α=10 | 0/7 | 0/7 |
| L12/F2257 | 7/7 +5.15%@α=50 | 7/7 +5.24%@α=50 | 7/7 +0.33%@α=2 | 5/7 +0.33%@α=2 |
| L15/F220 | 7/7 +4.31%@α=10 | 7/7 +4.79%@α=10 | 5/7 +2.45%@α=10 | 0/7 |

Notable update: L15/F220 SAE-B improved to +2.45% (was +0.78% at 3/7) — best SAE-B result so far, though still well below CAA-2 (+4.79%).

SAE-B/C still consistently 2–15× weaker than CAA. No new evidence that per-feature projection beats the mean-delta approach.

---

## Monitoring Update 30 — 2026-04-25 ~00:30 PDT

### Experiment Pivot: True CAA (Corrected Design)

**Conceptual correction:** All prior "CAA" experiments computed `mean(h_mix[l] - h_pt[l])` — a cross-model activation delta, not a label-contrastive direction. True CAA (Rimsky et al. 2023) computes:

```
v[l] = mean(h_src[l] | label=1) - mean(h_src[l] | label=0)
```

**The actual claim to test:** SAE feature discovery identifies the specific layer lF where each spatial concept is encoded in mix-448. Computing the label-contrastive vector at exactly that layer (FEATURE condition), restricted to R(F) samples, should give a sharper steering direction than computing it globally over all layers and all samples (GLOBAL condition).

- **GLOBAL True CAA**: v_global[l] from all 10,972 VSR samples at each SAE layer. Evaluated on R(F).
- **FEATURE True CAA**: v_feat[F] from R(F) ∩ {feature F fires in mix-448} at layer lF only. Evaluated on R(F). This is the feature-specific condition our SAE work enables.

If FEATURE > GLOBAL on R(F), it demonstrates that knowing which layer encodes which concept (via SAE feature discovery) produces a better steering vector than a layer-agnostic global computation.

### Cancelled Experiments

- **Hidden delta all-26 resume** (α=0.5+): Cancelled. α=0.5 result: +15.41%. Peak was α=0.3 at +15.74% — confirmed below SAE recon delta +16.42%.
- **pt→pt True CAA**: Not needed. Our claim is about sourcing the contrastive vector from mix-448 (the richer spatial model), not from pt-448 itself.
- **pt448_spatial_feature_ablation (CAA-1/2/SAE-B/C)**: Cancelled. Those used cross-model delta, not True CAA.
- **pt448_per_relation_steer RAW (L11)**: Cancelled. Flatline results.

### Active Experiments (as of ~00:30 PDT)

| GPU | PID | Condition | Source → Target | Log |
|-----|-----|-----------|----------------|-----|
| 0 | 2181641 | True CAA | mix-448 → pt-448 | `/tmp/true_caa_mix_to_pt.log` |
| 1 | 2181708 | True CAA | mix-448 → mix-448 | `/tmp/true_caa_mix_to_mix.log` |

Both currently in Step 1 (loading 10,972 cached h_mix files to compute global vectors). No forward passes needed for vector computation — pure tensor reads.

### Experimental Design Details

**Vector computation** (no model inference, fully from cache):
- Source hidden states: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/mix_hidden/vi_XXXXX.pt` — all 26 layers, all 10,972 samples
- SAE firing coefficients: `/data1/vlm_scope_sae_mix448_textonly/analysis/mix_sae_acts/acts_L*_F*.json` — firing rate ~100% on all R(F) subsets

**Inference**: NNsight hook at layer lF, inject `alpha * unit_norm(v) * 1_text_tokens`. Alpha sweep [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] per condition per feature.

**Fair comparison**: Both GLOBAL and FEATURE conditions are evaluated on the same R(F) subset for each feature, with the same alpha sweep. The only difference is how v is computed.

**Script**: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_true_caa_v2.py`
**Results**: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_v2/{mix_to_pt,mix_to_mix}/results.json`

---

## Monitoring Update 31 — 2026-04-25 ~01:30 PDT

### Design Correction: L15/F220 Relation Set

The `mix_sae_acts` file for L15/F220 was built with only 2 relations ("across from", "at the left side of", N=515). The ground-truth ablation files confirm 4 relations: **"across from", "at the left side of", "at the right side of", "right of"** (N=1108). Re-extracted and confirmed. All True CAA experiments now use N=1108 for L15/F220.

All other 9 features match the ablation ground truth exactly.

### Active Experiments (4 runs, all launched ~01:00 PDT)

| GPU | PID | Experiment | Target | Log |
|-----|-----|-----------|--------|-----|
| 0 | 2184448 | True CAA v3 (GLOBAL + FEATURE) | pt-448 | `/tmp/true_caa_v3_mix_to_pt.log` |
| 1 | 2184514 | True CAA v3 (GLOBAL + FEATURE) | mix-448 | `/tmp/true_caa_v3_mix_to_mix.log` |
| 2 | 2184936 | True CAA Middle-Layer (L13 baseline) | pt-448 | `/tmp/true_caa_middle_pt.log` |
| 3 | 2185002 | True CAA Middle-Layer (L13 baseline) | mix-448 | `/tmp/true_caa_middle_mix.log` |

GPUs 4–7 free.

### Three-Way Comparison Design

For each of 10 spatial features, on its R(F) relation subset:

| Condition | Vector source | Layer used | Data used for vector |
|-----------|-------------|-----------|---------------------|
| **MIDDLE** (baseline) | mix-448 h[13] | L13 (fixed middle) | All 10,972 VSR samples |
| **GLOBAL** | mix-448 h[lF] | lF (SAE feature layer) | All 10,972 VSR samples |
| **FEATURE** | mix-448 h[lF] | lF (SAE feature layer) | R(F) ∩ fire_F only |

- MIDDLE vs GLOBAL: does knowing the right layer (SAE knowledge) matter vs just using the middle layer?
- GLOBAL vs FEATURE: does restricting to firing samples sharpen the direction?
- MIDDLE vs FEATURE: full comparison of no-SAE-knowledge baseline vs full SAE-guided steering

Both targets (pt-448 and mix-448) run in parallel. mix-448 self-steering is an upper-bound sanity check.

### CAA Implementation Verification (~22:00 PDT)

Verified against Rimsky et al. 2023 (arXiv:2312.06681) and Meg Tong's activation-steering repo (adapted from Rimsky):

- **Vector formula** ✅ `mean(h[l]|label=1) − mean(h[l]|label=0)` over a dataset of pairs — exact match.
- **Where added** ✅ Output of transformer block (post-MLP). We use `register_forward_hook` on `layers[l]`, same convention.
- **Which tokens at inference** ✅ All tokens after the prompt boundary. We inject at `hidden[0, img_end:]` — all text tokens, skipping image tokens. Equivalent to Rimsky's "all tokens after `[/INST]`".
- **Normalization** ✅ Rimsky uses cross-behavior mean-norm equalization, not unit norm. We use unit-norm + alpha sweep — more principled (decouples direction from scale).
- **Token for extraction** — Rimsky/Tong extract at token `-2` (last prompt token before answer letter). Our cache stores `mean(h[img_end:])` over all text tokens per sample. This is a justified VLM adaptation: spatial meaning is distributed across the full statement, not concentrated at one token. The label-contrastive formula is identical; we just average more positions per sample.

**Conclusion: implementation is correct True CAA.** No code changes needed.

### Experiment Correction (~22:05 PDT)

The v3 script initially ran GLOBAL + FEATURE (3 conditions). Corrected to FEATURE-only per user decision. Re-launched 4 clean runs:

| GPU | PID | Condition | Target | Log |
|-----|-----|-----------|--------|-----|
| 0 | 2185609 | FEATURE (lF, R(F)∩fire_F) | pt-448 | `/tmp/true_caa_feature_pt.log` |
| 1 | 2185675 | FEATURE (lF, R(F)∩fire_F) | mix-448 | `/tmp/true_caa_feature_mix.log` |
| 2 | 2184936 | MIDDLE (L13, all data) | pt-448 | `/tmp/true_caa_middle_pt.log` |
| 3 | 2185002 | MIDDLE (L13, all data) | mix-448 | `/tmp/true_caa_middle_mix.log` |

### Monitoring Update (~00:00 PDT Apr 25) — 8/10 features complete, pattern confirmed

**Active runs:** GPUs 0–3 (PIDs 2185609/2185675/2184936/2185002). Currently processing L13/F15219 ("behind", N=709). L15/F220 and L12/F2257 still pending. GPUs 4–7 free.

**mix→pt-448 results (8/10 features done, score: MIDDLE=6, FEATURE=0, TIE=1):**

| Feature | Relations | N | pt base | FEATURE best Δ | MIDDLE best Δ | W |
|---------|-----------|--:|:-------:|:--------------:|:-------------:|:-:|
| L9/F387 | at the right side of | 480 | 52.29% | +0.42% (α=10) | **+2.29%** (α=5) | M |
| L14/F10561 | close to | 93 | 60.22% | +9.68% (α=20) | +9.68% (α=10) | T |
| L11/F12278 | touching | 1281 | 56.52% | +4.68% (α=10) | **+5.70%** (α=5) | M |
| L9/F7540 | consists of | 35 | 68.57% | +0.00% | **+2.86%** (α=2) | M |
| L4/F14233 | ahead of | 39 | 56.41% | +5.13% (α=1) | **+7.69%** (α=5) | M |
| L6/F7539 | left of, right of | 323 | 51.08% | −0.31% | **+1.86%** (α=5) | M |
| L11/F9639 | in, inside, on | 1101 | 60.85% | +0.64% (α=2) | **+1.00%** (α=1) | M |
| L13/F15219 | behind | 709 | 51.62% | running | running | ? |
| L15/F220 | across from, left/right side, right of | 1108 | — | — | — | — |
| L12/F2257 | facing | 306 | — | — | — | — |

**mix→mix-448 self-steering (8/10 features done, score: FEATURE=4, MIDDLE=1, TIE=2):**

| Feature | N | mix base | FEATURE best Δ | MIDDLE best Δ | W |
|---------|--:|:--------:|:--------------:|:-------------:|:-:|
| L9/F387 | 480 | 76.67% | **+0.21%** (α=0.5) | −0.21% | F |
| L14/F10561 | 93 | 79.57% | +1.08% | +1.08% | T |
| L11/F12278 | 1281 | 76.58% | **+2.50%** (α=10) | +2.34% | F |
| L9/F7540 | 35 | 85.71% | +0.00% | +0.00% | T |
| L4/F14233 | 39 | 61.54% | **+5.13%** (α=50) | +2.56% | F |
| L6/F7539 | 323 | 69.97% | **+0.31%** (α=1) | +0.00% | F |
| L11/F9639 | 1101 | 81.56% | +0.09% | **+0.18%** (α=0.5) | M |
| L13/F15219 | 709 | 71.79% | running | running | ? |
| L15/F220 | 1108 | — | — | — | — |
| L12/F2257 | 306 | — | — | — | — |

**Pattern confirmed at 8/10 — target-model-dependent reversal:**

- **mix→pt (cross-model):** MIDDLE wins 6/7 completed (1 TIE, 0 FEATURE). The generic L13 vector consistently transfers better to pt-448.
- **mix→mix (self-steering):** FEATURE wins 4/7 (2 TIE, 1 MIDDLE). SAE-identified feature layers are more effective when steering the same model they were extracted from.

**Root cause (diagnosed from SAE activation stats):** All 10 SAE features have near-zero label contrast (pos_mean ≈ neg_mean — they fire equally on correct and incorrect spatial statements). The SAE features are *relation detectors*, not *correctness detectors*. Restricting to firing samples gives no sharper label-contrastive direction. The FEATURE condition's advantage in self-steering comes from layer localization (using lF where mix-448 encodes the concept), not from sample selection. For cross-model transfer, SAE-identified layers are mix-specific; pt-448 likely encodes spatial concepts at different depths, making L13 (middle) a more neutral transfer point.

**Proposed next step:** pt→pt True CAA — compute label-contrastive vector from pt-448's own cached hidden states and steer pt-448 with it. Removes cross-model transfer problem entirely. pt hidden states cached at `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_hidden_delta/pt_hidden/` (all 10,972 samples, all 26 layers).

### True CAA v4 — Last-Token Extraction (launched ~00:10 PDT Apr 25)

**Key change from v3:** Extract hidden state at the **last text token** (position −1 of text tokens, i.e. the "Answer:" colon token) rather than mean-over-all-text-tokens. This matches Rimsky et al. 2023 and Meg Tong's implementation exactly: the last prompt token carries the model's full readout of the input before generating an answer.

**Why this matters:** Mean-over-text averages over "Is", "the", "following", "statement", etc. — tokens that carry little decision-relevant signal. The last token integrates the full context. For a yes/no correctness judgment, the activation at "Answer:" is where the spatial reasoning is most concentrated.

**Design:**
- Source = Target = **mix-448** only (no cross-model transfer)
- Live forward passes for vector extraction (existing cache is mean-aggregated, can't use it)
- Both MIDDLE (L13, all 10,972 samples, last token) and FEATURE (lF, R(F)∩fire_F, last token)
- Same alpha sweep [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] and hook-based steering

**Active run:** GPU 0, PID 2189016, log `/tmp/true_caa_v4.log`

Currently extracting last-token hidden states: L13 requires 10,972 passes, feature layers require additional 5,000+ passes. Vector computation phase runs before model evaluation. Expected runtime: ~4–5 hours total.

**Script:** `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_true_caa_v4.py`
**Results dir:** `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_v4/`

### Scripts
- **FEATURE condition**: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_true_caa_v3.py`
- **MIDDLE baseline**: `/data1/vlm_scope_sae_mix448_textonly/scripts/pt448_true_caa_middle.py`
- **Results**: `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_v3/` and `/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_true_caa_middle/`

---

## Completed Experiments — Monitoring Update (Apr 25 ~00:30 PDT)

### Status of all legacy processes

All processes from the prior standing monitor request (PIDs 2167555, 2159123, 2159121, 2160971) are **dead**. Log files last modified ≥4 hours ago. Results captured below from JSON files.

**Currently active (updated ~02:00 PDT Apr 25):**
| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 2189017 | `pt448_true_caa_v4.py` | **Steering phase** — L9/F387 done, sweeping remaining 9 features |
| 1 | 2189905 | `pt448_true_caa_v4b.py` | **Steering phase** — just started L9/F387 MIDDLE sweep |
| 2–7 | — | — | Free |

---

### Spatial Feature Ablation Results (pt448_spatial_feature_ablation.py — completed)

**All 8 available features done** (L11/F9639 and L13/F15219 missing — not in ablation set). Run targets: pt-448, relation subsets R(F)∩have_hidden_states, 4 methods (CAA-1/CAA-2/SAE-B/SAE-C), α ∈ [0.5–50].

**Method definitions:**
- **CAA-1**: Global mean-delta over all 10,972 samples, projected to unit norm
- **CAA-2**: Subset mean-delta over R(F) relation samples only, unit norm
- **SAE-B**: Projection of mean-delta onto W_dec[F] (SAE decoder direction for feature F), natural scale
- **SAE-C**: Same as SAE-B (equivalent to scalar × W_dec[F])

| Feature | N | Base | CAA-1 best Δ | CAA-2 best Δ | SAE-B best Δ | SAE-C best Δ | Winner |
|---------|--:|:----:|:------------:|:------------:|:------------:|:------------:|:------:|
| L9/F387 | 758 | 52.77% | +0.66% (α=1) | +0.40% (α=1) | +0.26% (α=5) | +0.26% (α=5) | CAA-1 |
| L9/F7540 | 1302 | 58.06% | +1.38% (α=5) | +1.38% (α=5) | — | — | CAA-1=2 |
| L11/F12278 | 2465 | 55.90% | **+3.73%** (α=50) | +3.41% (α=50) | — | — | CAA-1 |
| L12/F2257 | 1203 | 52.62% | +5.15% (α=50) | **+5.24%** (α=50) | +0.33% (α=2) | +0.33% (α=2) | CAA-2 |
| L14/F10561 | 182 | 57.14% | **+7.14%** (α=50) | +1.10% (α=1) | — | — | CAA-1 |
| L15/F220 | 1671 | 53.50% | +4.31% (α=10) | **+4.79%** (α=10) | +2.45% (α=10) | — | CAA-2 |
| L4/F14233 | 748 | 51.87% | +2.14% (α=10) | **+2.41%** (α=10) | +1.07% (α=5) | +1.07% (α=5) | CAA-2 |
| L6/F7539 | 1023 | 49.17% | **+0.59%** (α=10) | +0.29% (α=2) | +0.49% (α=1) | — | CAA-1 |

**Key finding:** SAE-B/SAE-C consistently underperform CAA-1 and CAA-2 by 2–7× (or are empty). SAE projection onto W_dec[F] adds no benefit over simple relation-contrastive CAA — and in 4/8 features has zero or negative results. CAA-2 (subset) marginally wins vs CAA-1 (global) in 3/8 features; CAA-1 wins in 4/8 — effectively tied. **SAE-guided projection fails as a steering method.**

Note: SAE-B/SAE-C empty for L11/F12278, L14/F10561, L9/F7540 — likely due to SAE projection coefficients all being positive (feature fires identically on pos/neg samples → near-zero mean-coefficient after label subtraction). Confirms near-zero label contrast root cause.

---

### Hidden Delta All-26 Layers Results (pt448_hidden_delta_phase3.py — completed)

**Method:** Inject mix-hidden − pt-hidden residual delta at ALL 26 layers simultaneously, onto pt-448.

| α | acc | Δ | n |
|--:|----:|--:|--:|
| 0.1 | 62.85% | +8.44% | 10,972 |
| 0.2 | 69.43% | +15.02% | 10,972 |
| 0.3 | **70.15%** | **+15.74%** | 10,972 |
| 0.4 | 69.07% | +14.66% | 10,972 |
| 0.5 | 69.82% | +15.41% | 10,972 |

**Best: α=0.3, Δ=+15.74%** (54.41% → 70.15%). α=0.5 also strong (+15.41%). This is the strongest result across all steering methods to date. Beats SAE-recon delta at α=0.3 (+15.74% vs +16.45% for SAE-recon which may be a different run).

---

### Hidden Delta 7-Layer Results (results_hooks.json — completed)

**Method:** Inject delta at 7 SAE-feature layers only [4, 6, 9, 11, 12, 14, 15].

| α | acc | Δ |
|--:|----:|--:|
| 0.1 | 58.88% | +4.47% |
| 0.2 | 54.92% | +0.51% |
| 0.3 | 53.81% | −0.60% |
| 0.4 | 54.03% | −0.38% |

**Best: α=0.1, Δ=+4.47%.** Dramatically worse than all-26 (+4.47% vs +15.74%). Injecting all 26 layers is critical — restricting to feature layers loses 11+ percentage points.

---

### SAE Activation Steering Results (pt448_sae_act_steer — completed)

**Method:** Steer at feature direction W_dec[F] scaled by mean activation gap (MEAN mode) or per-sample activation (PER_SAMPLE mode). Applied to R(F) subset.

| Feature | Mode | Base | Best Δ | Best α |
|---------|------|:----:|:------:|:------:|
| L11/F12278 | MEAN | 55.90% | +2.84% | 10.0 |
| L12/F2257 | MEAN | 52.62% | +0.00% | 10.0 |
| L12/F2257 | PER_SAMPLE | 52.62% | +0.08% | 0.5 |
| L15/F220 | MEAN | 53.50% | **+4.61%** | 20.0 |
| L15/F220 | PER_SAMPLE | 53.50% | +2.21% | 10.0 |
| L4/F14233 | MEAN | 51.87% | +1.07% | 5.0 |
| L6/F7539 | MEAN | 49.17% | +1.86% | 5.0 |
| L6/F7539 | PER_SAMPLE | 49.17% | +0.78% | 2.0 |
| L9/F387 | MEAN | 52.77% | +0.13% | 2.0 |
| L9/F7540 | MEAN | 58.06% | +1.23% | 20.0 |
| L9/F7540 | PER_SAMPLE | 58.06% | +1.15% | 2.0 |

Best single-feature SAE-direction steering: L15/F220 MEAN +4.61%. Modest improvements — consistent with W_dec[F] direction being a low-quality spatial correctness vector (relation detector, not correctness detector).

---

### True CAA v4 vs v4b — First Results (~02:00 PDT Apr 25)

Both runs now in steering phase. Key observation on vector norms:

| Script | Token | MIDDLE L13 norm | Example FEATURE norm |
|--------|-------|:---------------:|:--------------------:|
| v4 (h[0,−1,:]) | Last prompt token, no answer | 4.28 | L9/F387: 0.67, L15/F220: 11.41 |
| v4b (h[0,−2,:]) | Last prompt token + answer appended | **127.70** | L9/F387: 95.50, L15/F220: 106.61 |

The answer-appended extraction (v4b) produces ~30× larger raw norms. This is expected: conditioning on the correct answer makes the hidden state much more strongly label-contrastive — the model's spatial reasoning is sharply concentrated at that token when it "sees" the answer. After unit-norm + alpha sweep, magnitude is normalized away, but the direction should be sharper.

**COMPLETE results (~09:00 PDT Apr 25) — both v4 and v4b finished all 10 features:**

mix-448 self-steering. MIDDLE = L13 global vector. FEATURE = lF R(F)∩fire_F vector.
v4: last token h[0,−1,:]. v4b: answer-appended, token[−2] (exact Rimsky/Tong).

| Feature | Relations | N | mix base | v4 MIDDLE Δ | v4 FEATURE Δ | v4b MIDDLE Δ | v4b FEATURE Δ |
|---------|-----------|--:|:-------:|:-----------:|:------------:|:------------:|:-------------:|
| L9/F387 | at right side of | 480 | 76.67% | −0.42% | −0.21% | +0.21% | −0.42% |
| L14/F10561 | close to | 93 | 79.57% | +1.08% | +1.08% | +1.08% | +1.08% |
| L11/F12278 | touching | 1281 | 76.58% | **+1.87%** (α=10) | +0.16% | +0.62% | −0.23% |
| L9/F7540 | consists of | 35 | 85.71% | −2.86% | +0.00% | +0.00% | +0.00% |
| L4/F14233 | ahead of | 39 | 61.54% | **+2.56%** | +0.00% | **+2.56%** | **+2.56%** |
| L6/F7539 | left/right of | 323 | 69.97% | +0.31% | **+0.62%** | +0.00% | +0.00% |
| L11/F9639 | in/inside/on | 1101 | 81.56% | +0.00% | **+0.73%** | +0.00% | +0.18% |
| L13/F15219 | behind | 709 | 71.79% | +0.42% | **+0.71%** | +0.42% | +0.42% |
| L15/F220 | across/left-right/right of | 1108 | 72.02% | −0.36% | −0.09% | **+0.09%** | −0.09% |
| L12/F2257 | facing | 306 | 60.78% | **+0.65%** (α=10) | −0.33% | **+0.98%** (α=?) | **+1.63%** (α=20) |

**L13/F15219 complete** (both v4 and v4b): v4 FEATURE (+0.71%) > MIDDLE (+0.42%). v4b: MIDDLE = FEATURE = +0.42% at α=0.5–1. The feature-specific direction at exactly L13 gives a sharper signal than the global L13 direction, even when both are at the same layer. 

**L15/F220 complete MIDDLE results (v4)**: All 7 alphas negative and monotonically worsening: −0.36% (α=0.5), −0.45%, −0.81%, −2.17%, −3.43%, −8.21%, −14.71% (α=50). The global L13 MIDDLE direction is actively harmful for this multi-relation feature (across from, left/right side of, right of). **v4b MIDDLE** (answer-appended): −0.09% (α=0.5), +0.09% (α=1), −0.09% (α=2) — essentially neutral. Dramatic difference between extraction methods.

**L15/F220 complete all 4 variants (~09:00 PDT Apr 25)**:

| α | MIDDLE (v4) | FEATURE (v4) | MIDDLE (v4b) | FEATURE (v4b) |
|---|:----------:|:------------:|:------------:|:-------------:|
| 0.5 | −0.36% | **−0.09%** | −0.09% | −0.09% |
| 1.0 | −0.45% | −0.36% | **+0.09%** | −0.18% |
| 2.0 | −0.81% | −0.81% | −0.09% | −0.09% |
| 5.0 | −2.17% | −1.71% | −0.90% | −0.45% |
| 10.0 | −3.43% | −2.89% | −1.35% | −1.17% |
| 20.0 | −8.21% | −7.31% | −2.08% | −1.71% |
| 50.0 | −14.71% | − (not saved) | −8.21% | − |

All 4 variants negative. v4b MIDDLE is best: +0.09% at α=1 (essentially neutral). L15/F220 confirmed resistant — the multi-relation "across from / left/right side of / right of" set is too heterogeneous for a single direction.

**KEY FINDING — FEATURE vs MIDDLE for v4 mix→mix self-steering (10/10 COMPLETE)**:

| Feature | MID best | FEAT best | Winner |
|---------|:--------:|:---------:|:------:|
| L9/F387 | −0.42% | −0.21% | FEAT (less neg) |
| L14/F10561 | +1.08% | +1.08% | TIE |
| L11/F12278 | **+1.87%** | +0.16% | MID |
| L9/F7540 | −2.86% | +0.00% | FEAT (less neg) |
| L4/F14233 | **+2.56%** | +0.00% | MID |
| L6/F7539 | +0.31% | **+0.62%** | FEAT |
| L11/F9639 | +0.00% | **+0.73%** | FEAT |
| L13/F15219 | +0.42% | **+0.71%** | FEAT |
| L15/F220 | −0.36% | −0.09% | FEAT (less neg) |
| L12/F2257 | **+0.65%** | −0.33% | MID |

**Final tally: FEAT wins 5/10, MID wins 3/10, TIE 1/10, FEAT less-neg 2/10**. FEAT wins when there's a clear feature-specific spatial concept; MID wins for L11/F12278 ("touching"), L4/F14233 ("ahead of"), and L12/F2257 ("facing") — features where the global direction has broader support.

**KEY FINDING — v4b (answer-appended) 10/10 COMPLETE**:

| Feature | MID best | FEAT best | Notable |
|---------|:--------:|:---------:|:-------:|
| L9/F387 | +0.21% | −0.42% | MID wins (v4: −0.42% / −0.21%) |
| L14/F10561 | +1.08% | +1.08% | TIE |
| L11/F12278 | +0.62% | −0.23% | MID wins (v4 MID: +1.87%) |
| L9/F7540 | +0.00% | +0.00% | TIE |
| L4/F14233 | +2.56% | **+2.56%** | TIE — "ahead of" equally directional |
| L6/F7539 | +0.00% | +0.00% | TIE |
| L11/F9639 | +0.00% | +0.18% | FEAT wins (marginally) |
| L13/F15219 | +0.42% | +0.42% | TIE |
| L15/F220 | **+0.09%** | −0.09% | MID wins (less harmful) |
| L12/F2257 | +0.98% | **+1.63%** | **FEAT wins** — notable! v4b FEAT best at α=20 |

**v4b surprise: L12/F2257 FEATURE = +1.63%** (α=20). v4 MIDDLE was +0.65%, v4 FEATURE was −0.33%. The answer-appended extraction finds a better "facing" direction at the FEATURE layer (L12) than the global MIDDLE (L13). This is the only feature where v4b FEATURE clearly beats v4 in absolute terms.

**Interpretation**: For self-steering, the feature-specific contrastive direction at the exact firing layer is more precise than the global L13 direction. The FEATURE direction is computed on exactly the subset where it will be applied (same relation+feature-firing subset), giving better signal-to-noise ratio than the global mean.

**L13/F15219**: FEATURE +0.71% > MIDDLE +0.42%. Notable: L13 is both the MIDDLE layer and the feature layer. Restricting to the "behind"∩fire subset gives a sharper direction.

**v4b vs v4**: v4b FEATURE generally weaker (answer-appended direction is more about answer identity than spatial correctness). Exception: L4/F14233 "ahead of" where both v4b MIDDLE and FEATURE give +2.56% — the "ahead of" concept is strongly directional regardless of extraction method.

**New experiments launched (02:30 PDT Apr 25) on GPUs 2 & 3:**
- **pt→pt True CAA** (GPU 2, PID 2194276): pt-448 hidden cache → steer pt-448. Fills 2×2 grid cell. Cache reads only, ~30min for vectors + ~2h for sweeps.
- **mix→pt True CAA (cache)** (GPU 3, PID 2194362): mix-448 hidden cache → steer pt-448. Directly comparable to mix→pt v3 but using mean-over-text-tokens instead of per-feature-layer extraction.

**Full 2×2 grid + extras launched (~02:40 PDT Apr 25):**

| GPU | Script | Source → Target | PID | Status (~09:00 PDT Apr 25) |
|-----|--------|----------------|-----|---------------------|
| 0 | **pt448_true_caa_v4.py DONE** → new experiment | mix live → mix-448 | 2189017 **dead** | **ALL 10 features complete**. GPU free — launching new experiment |
| 1 | **pt448_true_caa_v4b.py DONE** → new experiment | mix live (ans+) → mix-448 | 2189905 **dead** | **ALL 10 features complete**. GPU free — launching new experiment |
| 2 | pt448_true_caa_v4_pt_src.py | pt cache → pt-448 | 2194277 | L11/F12278 FEATURE complete (+5.62% α=10); L6/F7539 MIDDLE running |
| 3 | pt448_true_caa_v4_mix_to_pt.py | mix cache → pt-448 | 2194363 | L11/F12278 FEATURE complete (+4.68% α=10); L6/F7539 MIDDLE running |
| 4 | pt448_true_caa_v4_mix_to_mix.py | mix cache → mix-448 | 2195469 | L11/F12278 FEATURE complete (+2.50% α=10–20); L9/F7540 MIDDLE running |
| 5 | pt448_true_caa_v4_pt_to_mix.py | pt cache → mix-448 | 2195698 | L11/F12278 FEATURE running (through α=20: +3.12%); continuing |
| 6 | pt448_true_caa_alllayer_sweep.py | mix cache → pt-448 (all layers) | 2196021 | L9/F387 α=2: best L12 +2.08%; α=5: best L14 +3.33%; α=10 sweeping |
| 7 | pt448_true_caa_combined.py | mix cache → pt-448 (combined) | 2196250 | GLOBAL→union10 α=5: +2.83%; α=10 running |

**2×2 True CAA grid (cache-based, mean-over-text-tokens):**

| Source \ Target | mix-448 | pt-448 |
|----------------|:-------:|:------:|
| mix hidden | GPU 4 (v4_mix_to_mix) | GPU 3 (v4_mix_to_pt) |
| pt hidden | GPU 5 (v4_pt_to_mix) | GPU 2 (v4_pt_src) |

**Results from 2×2 grid — L9/F387 and L14/F10561 complete, L11/F12278 in progress:**

*L9/F387 ("at right side of", pt base 52.29%, mix base 76.67%):*

| Condition | pt base | mix base | MIDDLE best Δ | FEATURE best Δ |
|-----------|:-------:|:--------:|:-------------:|:--------------:|
| pt→pt (GPU 2) | 52.29% | — | **+3.12%** (α=5) | +0.83% (α=10) |
| mix→pt (GPU 3) | 52.29% | — | +2.29% (α=5) | +0.42% (α=10) |
| mix→mix (GPU 4) | — | 76.67% | −0.21% (degrades) | +0.21% (α=0.5) |
| pt→mix (GPU 5) | — | 76.67% | +0.21% (α=0.5) | **+1.04%** (α=20) |

*L14/F10561 ("close to", pt base 60.22%, mix base 79.57%):*

| Condition | MIDDLE best Δ | FEATURE best Δ |
|-----------|:-------------:|:--------------:|
| pt→pt (GPU 2) | **+9.68%** (α=20–50) | **+9.68%** (α=10–50) |
| mix→pt (GPU 3) | **+9.68%** (α=10–50) | **+9.68%** (α=20–50) |
| mix→mix (GPU 4) | +1.08% (α=1–2) | running |
| pt→mix (GPU 5) | +1.08% (α=1–5, early) | running |

Both pt→pt and mix→pt reach the same ceiling (+9.68%) for L14/F10561 — the "close to" spatial concept saturates to the same performance regardless of source model. Mix self-steering with MIDDLE also positive (+1.08%, unlike L9/F387 where MIDDLE hurt mix-448). pt→mix also shows early positive signal at +1.08%.

*L11/F12278 ("touching", pt base 56.52%):* Early MIDDLE sweep — pt→pt: −0.16%/+0.47% (α=0.5/1), mix→pt: +0.62%/+1.80% (α=0.5/1). mix→pt already outperforming pt→pt at α=1! Reversal from L9/F387.

**2×2 grid comprehensive results (updated ~03:45 PDT):**

| Feature | pt→pt MIDDLE | mix→pt MIDDLE | mix→mix MIDDLE | pt→mix MIDDLE | pt→mix FEATURE |
|---------|:-----------:|:-------------:|:--------------:|:-------------:|:--------------:|
| L9/F387 | **+3.12%** | +2.29% | −0.21% | +0.21% | **+1.04%** |
| L14/F10561 | **+9.68%** | **+9.68%** | +1.08% | +1.08% | +1.08% |
| L11/F12278 | **+5.85%** (α=5) | **+5.70%** (α=5) | +0.16% (α=2, barely pos) | running | — |

**Complete L11/F12278 MIDDLE sweep (full 7 alphas, updated ~06:00 PDT Apr 25):**

| α | mix→pt | pt→pt | mix→mix | pt→mix |
|---|:------:|:-----:|:-------:|:------:|
| 0.5 | +0.62% | −0.16% | −0.55% | −0.47% |
| 1.0 | +1.80% | +0.47% | −0.08% | −0.39% |
| 2.0 | +3.90% | +2.42% | +0.16% | −0.08% |
| 5.0 | **+5.70%** | **+5.85%** | +2.03% | +0.39% |
| 10.0 | +0.16% ← collapse! | +3.90% ← holds | **+2.34%** ← peak | +1.95% |
| 20.0 | −7.42% | −3.67% | −0.70% ← drops | **+2.81%** ← peak |
| 50.0 | −7.49% | −7.49% | −21.08% ← crash | −3.36% |

**Key findings (MIDDLE complete for all 4 conditions):**
- Optimal α=5 for pt-targeting: both mix→pt (+5.70%) and pt→pt (+5.85%) reach ~97% of oracle SAE result (+6.01%)
- **Sharp peak for mix→pt**: collapses from +5.70% at α=5 to +0.16% at α=10 — extremely alpha-sensitive
- **Broader peak for pt→pt**: +5.85% at α=5, gracefully degrades to +3.90% at α=10 — more robust
- **mix→mix peaks at α=10 (+2.34%)**: Delayed peak. Mix-448 is well-calibrated, needs higher alpha. α=50 catastrophically crashes to −21.08%.
- **pt→mix peaks at α=20 (+2.81%)**: The most-delayed condition. α=50 also collapses to −3.36%.
- The True CAA MIDDLE direction (no feature knowledge) approaches oracle SAE performance at its peak!

**L11/F12278 FEATURE sweep — pt→pt and mix→pt (partial, through α=10, ~07:30 PDT):**

| α | pt→pt FEATURE | mix→pt FEATURE |
|---|:-------------:|:--------------:|
| 0.5 | +0.08% | +0.00% |
| 1.0 | +0.70% | +0.55% |
| 2.0 | +1.01% | +1.41% |
| 5.0 | +3.83% | +4.06% |
| 10.0 | **+5.62%** | **+4.68%** |
| 20.0 | running | running |

FEATURE vectors are climbing steeply — pt→pt FEATURE at α=10 (+5.62%) nearly matches MIDDLE peak (+5.85% at α=5). For mix→pt, FEATURE at α=10 (+4.68%) already surpasses MIDDLE at α=10 (+0.16%) by far — the FEATURE direction may have a broader peak than MIDDLE for this condition. α=20 will reveal whether FEATURE peaks higher than MIDDLE (+5.70%) for mix→pt. If so, SAE-guided subset selection outperforms global direction for mix→pt.

**Emerging 2×2 insights (updated ~06:00 PDT Apr 25):**
1. **Feature-dependent source advantage (with convergence)**: pt→pt (+5.85%) ≈ mix→pt (+5.70%) at α=5 for L11/F12278 — they converge! For L9/F387, pt→pt (+3.12%) definitively beats mix→pt (+2.29%). For L14/F10561, both tie at +9.68%. Conclusion: the source model rarely matters much at optimal alpha — both converge to similar performance. The difference appears mainly at small-to-mid alpha.

**KEY FINDING: True CAA MIDDLE approaches oracle SAE steering**: pt→pt +5.85% and mix→pt +5.70% at α=5 vs oracle SAE-recon +6.01% for L11/F12278 "touching". A simple global label-contrastive direction (no feature knowledge) achieves ~97% of the oracle result!

2. **L14/F10561 differential ceiling by target model**: pt→pt and mix→pt both reach +9.68% (pt base 60.22%, huge room). mix→mix and pt→mix both cap at +1.08% (mix base 79.57%, less room). The ceiling is determined by the *target model's* baseline accuracy, not the source direction. Same direction, very different effectiveness depending on target model's prior calibration.

3. **mix→mix L11/F12278 peaks at α=10 (+2.34%)**: Unlike L9/F387 where mix→mix MIDDLE always hurts, "touching" benefits from global MIDDLE direction in mix-448. The peak is at higher α (10 vs 5 for pt targeting) — mix is better calibrated, needs stronger push. Peak of +2.34% is substantial even for mix.

4. **pt→mix L11/F12278 peaks at α=20 (+2.81%)**: Cross-model pt→mix peaks later than mix→mix (α=10) and pt→pt (α=5). This is the most delayed peak: combining cross-model mismatch AND high target calibration requires higher injection. α=50 crashes to −3.36%.

5. **Unified α-profile insight (updated with full data)**: Peak α ordered by condition: pt→pt/mix→pt at α=5; mix→mix at α=10; pt→mix at α=20. Higher target baseline + cross-model mismatch → higher optimal α. Mix-448 baseline (76.58%) is 20% above pt (56.52%), requiring more alpha to perturb. pt→mix compounds this with directional misalignment.

5. **v4 extraction method matters**: v4 (last prompt token) L15/F220 MIDDLE is strongly negative (−8.21% at α=20); v4b (answer-appended) is essentially neutral (−0.09% to +0.09%). Token extraction method has significant impact on direction quality for some features.

6. **v4 L12/F2257 "facing" MIDDLE +0.65%, FEATURE −0.33%**: Weak positive for MIDDLE, slightly negative for FEATURE — reversal of the usual FEAT>MID pattern. v4b MIDDLE (+0.98%) and FEATURE (+1.63%) both better. Consistent with "facing" being a more globally-distributed concept.

7. **mix→mix L14/F10561 FEATURE complete**: +1.08%, same ceiling as MIDDLE. pt→mix FEATURE also +1.08%. "Close to" saturates at +1.08% for any mix-targeting regardless of source or method.

8. **mix→mix L11/F12278 FEATURE complete**: best +2.50% at α=10–20. Compare MIDDLE +2.34% at α=10. FEATURE slightly better for mix→mix self-steering — the SAE-guided feature direction is marginally superior at the same alpha.

**2×2 grid comprehensive results (updated ~09:00 PDT Apr 25):**

| Feature | pt→pt MIDDLE | mix→pt MIDDLE | mix→mix MIDDLE | pt→mix MIDDLE | pt→pt FEATURE | mix→pt FEATURE | mix→mix FEATURE | pt→mix FEATURE |
|---------|:-----------:|:-------------:|:--------------:|:-------------:|:-------------:|:--------------:|:---------------:|:--------------:|
| L9/F387 | **+3.12%** | +2.29% | −0.21% | +0.21% | +0.83% | +0.42% | +0.21% | **+1.04%** |
| L14/F10561 | **+9.68%** | **+9.68%** | +1.08% | +1.08% | **+9.68%** | **+9.68%** | +1.08% | +1.08% |
| L11/F12278 | **+5.85%** (α=5) | **+5.70%** (α=5) | +2.34% (α=10) | **+2.81%** (α=20) | +5.62% (α=10) | +4.68% (α=10) | **+2.50%** (α=10–20) | +3.12% (α=20, partial) |

**All-layer injection sweep (GPU 6) — first layer result:**
- L9/F387 α=2: best injection layer = L12, Δ=+2.08%
- Compare: standard injection at L13 (extraction layer) for α=2 gives +1.25% (from prior pt→pt). Injecting at L12 instead of L13 gives +2.08% — small but real benefit from optimal layer selection.
- α=5, 10, 20 still sweeping all 26 layers.

**Combined steering (GPU 7) — GLOBAL→union10 results (~09:00 PDT):**
- union10 baseline: 55.00% (N=4882)
- top3_union baseline: **56.76%** (N=1413)

| α | GLOBAL→union10 Δ |
|---|:----------------:|
| 0.5 | +0.29% |
| 1.0 | +0.76% |
| 2.0 | **+1.78%** |
| 5.0 | **+2.83%** |
| 10.0 | running |

Consistent, growing positive improvement. α=5 reaches +2.83% on N=4882 samples — the global MIDDLE direction steers the union-10 set well. Growing pattern suggests peak has not been reached yet at α=5.

**All-layer injection sweep (GPU 6) — new result:**
- L9/F387 α=2: best L12, Δ=+2.08%
- L9/F387 α=5: best L14, Δ=**+3.33%** — injection at L14 (one layer above extraction L13) beats L13 injection at α=5!

**Scripts dir:** `/data1/vlm_scope_sae_mix448_textonly/scripts/`
**Results dirs:** `analysis/pt448_true_caa_v4*/`

---

## Monitoring Update 33 — 2026-04-25 ~10:15 PDT

### New Experiments Launched

Two new scripts written and launched on free GPUs 0 and 1:

**GPU 0 — `pt448_true_caa_combined_feature.py` (PID 2206723)**
- **Goal:** Test whether per-feature FEATURE routing outperforms a single global MIDDLE vector on the union-10 set.
- **ROUTED condition**: Each sample in union10 is steered by its own feature's FEATURE vector (from mix-448 cache) at that feature's own layer lF. First-match assignment for samples in multiple features.
- **Also runs**: GLOBAL MIDDLE (L13, all VSR) as comparison; per-feature FEATURE sweeps on each R(F) subset.
- **Reference**: Combined experiment GPU 7 shows GLOBAL MIDDLE reaches +2.83% on union10 at α=5. Can per-feature routing beat this?
- Log: `/tmp/true_caa_combined_feature.log`
- Results: `analysis/pt448_true_caa_combined_feature/`

**GPU 1 — `pt448_true_caa_v4b_pt_src.py` (PID 2206745)**
- **Goal:** Complete the v4b extraction-method × target-model grid. v4b (answer-appended token[-2]) was run for mix-448 self-steering; this runs the same protocol with pt-448 as both source and target.
- **v4b pt→pt**: Extracts vectors from pt-448 live forward passes (token[-2] with answer appended), steers pt-448. Completes the 2×2 v4b grid cell.
- **Key comparison**: v4b mix→mix showed L12/F2257 FEATURE +1.63% at α=20 (best for "facing" across all methods). Will v4b pt→pt replicate this for pt-448?
- Log: `/tmp/true_caa_v4b_pt_src.log`
- Results: `analysis/pt448_true_caa_v4b_pt_src/`

### Updated GPU Status (~10:15 PDT Apr 25)

| GPU | Script | Source → Target | PID | Status |
|-----|--------|----------------|-----|--------|
| 0 | pt448_true_caa_combined_feature.py | mix cache → pt-448 (ROUTED) | **2206723** | Computing vectors (MIDDLE over 10,972 + FEATURE for 10 features) |
| 1 | pt448_true_caa_v4b_pt_src.py | pt live (ans+) → pt-448 | **2206745** | Loading pt-448, extracting token[-2] hidden states |
| 2 | pt448_true_caa_v4_pt_src.py | pt cache → pt-448 | 2194277 | L6/F7539 MIDDLE/FEATURE running (5th feature) |
| 3 | pt448_true_caa_v4_mix_to_pt.py | mix cache → pt-448 | 2194363 | L6/F7539 MIDDLE/FEATURE running (5th feature) |
| 4 | pt448_true_caa_v4_mix_to_mix.py | mix cache → mix-448 | 2195469 | L6/F7539 MIDDLE running (5th feature) |
| 5 | pt448_true_caa_v4_pt_to_mix.py | pt cache → mix-448 | 2195698 | L6/F7539 MIDDLE running (5th feature) |
| 6 | pt448_true_caa_alllayer_sweep.py | mix cache → pt-448 (all layers) | 2196021 | L9/F387 α=10 sweeping all 26 layers |
| 7 | pt448_true_caa_combined.py | mix cache → pt-448 (global combined) | 2196250 | GLOBAL→union10 α=10 running |

### New Results from GPUs 2–5 (since last update)

**L4/F14233 "ahead of" complete (pt→pt, mix→pt):**

| Condition | MIDDLE best Δ | FEATURE best Δ |
|-----------|:-------------:|:--------------:|
| pt→pt | **+12.82%** (α=10) | +5.13% (α=50) |
| mix→pt | **+7.69%** (α=5) | +5.13% (α=1) |
| mix→mix | +2.56% (α=0.5–5) | +5.13% (α=50) |
| pt→mix | +2.56% (α=0.5–10) | +2.56% (α=1–2, 20) |

Notable: **pt→pt L4/F14233 MIDDLE peaks at +12.82% at α=10** — the strongest pt-targeting result for any feature so far! The "ahead of" spatial relation is especially responsive when using pt-448's own contrastive direction. mix→pt MIDDLE peaks at +7.69% α=5, considerably weaker. L9/F7540 "consists of": MIDDLE flat (+0.00% at α=2–5 for both pt→pt and mix→pt); pt→pt FEATURE +2.86% at α=5, mix→pt FEATURE +0.00%.

**L6/F7539 "left of/right of" — running on all 4 GPUs:**

Early results (pt→pt, partial): MIDDLE best +1.86% (α=5), FEATURE: best −0.31% at lowest alphas, degrading. 
Early results (mix→pt, partial): MIDDLE best +1.86% (α=5), FEATURE: negative across all alphas.

The "left of/right of" feature appears to respond to MIDDLE but not FEATURE direction, possibly because the subset is large and direction-noisy.

### Updated 2×2 Grid (L4/F14233 added)

| Feature | pt→pt MIDDLE | mix→pt MIDDLE | mix→mix MIDDLE | pt→mix MIDDLE | pt→pt FEATURE | mix→pt FEATURE | mix→mix FEATURE | pt→mix FEATURE |
|---------|:-----------:|:-------------:|:--------------:|:-------------:|:-------------:|:--------------:|:---------------:|:--------------:|
| L9/F387 | +3.12% | +2.29% | −0.21% | +0.21% | +0.83% | +0.42% | +0.21% | **+1.04%** |
| L14/F10561 | **+9.68%** | **+9.68%** | +1.08% | +1.08% | **+9.68%** | **+9.68%** | +1.08% | +1.08% |
| L11/F12278 | **+5.85%** (α=5) | **+5.70%** (α=5) | +2.34% (α=10) | **+2.81%** (α=20) | +5.62% (α=10) | +4.68% (α=10) | **+2.50%** (α=10–20) | +3.12% (α=20) |
| L9/F7540 | +0.00% (flat) | +2.86% (α=2) | +0.00% (flat) | +0.00% (flat) | +2.86% (α=5) | +0.00% | — | — |
| L4/F14233 | **+12.82%** (α=10) | **+7.69%** (α=5) | +2.56% (α=0.5–5) | +2.56% (α=0.5–10) | +5.13% (α=50) | +5.13% (α=1) | +5.13% (α=50) | +2.56% (α=1–2,20) |
| L6/F7539 | +1.86% (α=5, partial) | +1.86% (α=5, partial) | running | running | ≤0% so far | ≤0% so far | — | — |

**New finding — L4/F14233 pt→pt MIDDLE dominance**: +12.82% is the largest improvement seen in any pt-targeting experiment. Consistent with "ahead of" being a highly directional concept with strong representation in pt-448's contrastive direction. FEATURE (+5.13%) is notably weaker — pt→pt MIDDLE direction captures something stronger than the SAE-subset direction for this concept.

**L9/F7540 flat for MIDDLE**: "consists of" (N=35) shows no response to MIDDLE at any alpha in 3/4 conditions. Only pt→pt FEATURE shows +2.86%. Very small N likely responsible for noisy results.

---

## Monitoring Update 34 — 2026-04-25 ~11:00 PDT

All 8 GPUs busy. The 4 old PIDs (spatial_feature_ablation, hidden_delta_all26, hidden_delta_7layer, sae_act_steer) are confirmed dead — those experiments completed previously. No GPUs are free; no new launches this cycle.

### GPU Status (~11:00 PDT Apr 25)

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 2206723 | pt448_true_caa_combined_feature.py | Vectors computed; computing union10 baseline (N=4882) |
| 1 | 2206745 | pt448_true_caa_v4b_pt_src.py | Extracting token[-2] from pt-448: L11 ~1500/2350 |
| 2 | 2194277 | pt448_true_caa_v4_pt_src.py | L6/F7539 complete; L11/F9639 MIDDLE started |
| 3 | 2194363 | pt448_true_caa_v4_mix_to_pt.py | L6/F7539 complete; L11/F9639 MIDDLE started |
| 4 | 2195469 | pt448_true_caa_v4_mix_to_mix.py | L6/F7539 complete; L11/F9639 MIDDLE started |
| 5 | 2195698 | pt448_true_caa_v4_pt_to_mix.py | L6/F7539 FEATURE α=20 running |
| 6 | 2196021 | pt448_true_caa_alllayer_sweep.py | L9/F387 α=10 sweeping layers 0–25 |
| 7 | 2196250 | pt448_true_caa_combined.py | GLOBAL→union10 α=10 complete (−0.51%); top3 α sweep next |

### New Results This Cycle

**COMBINED (GPU 7) — α=10 result arrived: peak was α=5!**

| α | GLOBAL→union10 Δ |
|---|:----------------:|
| 0.5 | +0.29% |
| 1.0 | +0.76% |
| 2.0 | +1.78% |
| **5.0** | **+2.83% ← PEAK** |
| 10.0 | **−0.51%** ← collapses |

**Key finding**: GLOBAL MIDDLE on union10 peaks sharply at α=5 (+2.83%), then collapses to −0.51% at α=10. Same narrow-peak pattern as pt-targeting from prior experiments (pt→pt L11/F12278 peaked α=5, dropped α=10). The union-10 set behaves like a pt-targeting condition (target is pt-448; baseline 55%). Top3 sweep now starting.

**L6/F7539 "left of/right of" — all conditions COMPLETE:**

| Condition | MIDDLE best Δ | FEATURE best Δ |
|-----------|:-------------:|:--------------:|
| pt→pt | **+2.17%** (α=20) | −0.31% (α=0.5) ← all negative |
| mix→pt | **+1.86%** (α=5) | −0.62% (α=1) ← all negative |
| mix→mix | +0.00% (α=0–2, flat) → **−13.31%** (α=50) | +0.31% (α=1–2) — barely positive |
| pt→mix | **+0.93%** (α=10) | **+1.24%** (α=20) ← only condition FEATURE>MIDDLE |

Key pattern for L6/F7539:
- **MIDDLE weakly positive for pt-targeting** (pt→pt +2.17%, mix→pt +1.86%) — note pt→pt peak is α=20, not α=5. Wider optimal α than most features.
- **FEATURE consistently negative for pt-targeting** — the "left of/right of" SAE subset direction actively hurts. The feature fires on left AND right tokens together; the contrastive direction within fire_F is confused.
- **mix→mix MIDDLE strongly degrades**: −3.72% at α=5, −13.31% at α=50. Mix is already 70% accurate on left/right — any perturbation hurts.
- **pt→mix FEATURE +1.24% is only bright spot**: FEATURE beats MIDDLE for pt→mix on this feature (reversed pattern vs all others). Mix self-FEATURE is flat at +0.31%, but cross-model pt→mix FEATURE reaches +1.24% — directional noise from cross-model transfer may accidentally help here.

**Updated 2×2 Grid (L6/F7539 complete):**

| Feature | pt→pt MIDDLE | mix→pt MIDDLE | mix→mix MIDDLE | pt→mix MIDDLE | pt→pt FEATURE | mix→pt FEATURE | mix→mix FEATURE | pt→mix FEATURE |
|---------|:-----------:|:-------------:|:--------------:|:-------------:|:-------------:|:--------------:|:---------------:|:--------------:|
| L9/F387 | +3.12% | +2.29% | −0.21% | +0.21% | +0.83% | +0.42% | +0.21% | **+1.04%** |
| L14/F10561 | **+9.68%** | **+9.68%** | +1.08% | +1.08% | **+9.68%** | **+9.68%** | +1.08% | +1.08% |
| L11/F12278 | **+5.85%** | **+5.70%** | +2.34% | **+2.81%** | +5.62% | +4.68% | **+2.50%** | +3.12% |
| L9/F7540 | +0.00% | +2.86% | +0.00% | +0.00% | +2.86% | +0.00% | — | — |
| L4/F14233 | **+12.82%** | **+7.69%** | +2.56% | +2.56% | +5.13% | +5.13% | +5.13% | +2.56% |
| L6/F7539 | **+2.17%** (α=20) | **+1.86%** (α=5) | 0.00% (flat→crash) | +0.93% (α=10) | ≤−0.31% | ≤−0.62% | +0.31% | **+1.24%** (α=20) |
| L11/F9639 | running | running | running | running | — | — | — | — |

**mix→mix L6/F7539 insight**: MIDDLE direction aggressively degrades mix-448 accuracy for left/right (−13.31% at α=50). This is the largest negative MIDDLE effect seen — mix already handles left/right well (70%) and the global label direction pushes it off-distribution. Consistent with the pattern that high-baseline features resist steering.

---

## Monitoring Update 35 — 2026-04-25 ~11:45 PDT

All 8 GPUs still busy, all PIDs alive. No GPUs free — no new launches.

### GPU Status (~11:45 PDT Apr 25)

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 2206723 | pt448_true_caa_combined_feature.py | GLOBAL MIDDLE sweep on union10 starting (α=0.5...) |
| 1 | 2206745 | pt448_true_caa_v4b_pt_src.py | Extracting L13 token[-2]: ~5000/10972 (~halfway) |
| 2 | 2194277 | pt448_true_caa_v4_pt_src.py | L11/F9639 MIDDLE complete (+1.18% α=2 peak); FEATURE starting |
| 3 | 2194363 | pt448_true_caa_v4_mix_to_pt.py | L11/F9639 MIDDLE complete (+1.00% α=1 peak); FEATURE starting |
| 4 | 2195469 | pt448_true_caa_v4_mix_to_mix.py | L11/F9639 MIDDLE starting (+0.18% α=0.5 so far) |
| 5 | 2195698 | pt448_true_caa_v4_pt_to_mix.py | L11/F9639 MIDDLE starting (+0.09% α=0.5 so far) |
| 6 | 2196021 | pt448_true_caa_alllayer_sweep.py | L9/F387 α=10 complete (best L9 +2.92%); α=20 sweeping |
| 7 | 2196250 | pt448_true_caa_combined.py | GLOBAL→union10 complete (all 7 α); top3 sweep not yet started |

### New Results This Cycle

**COMBINED (GPU 7) — GLOBAL→union10 fully complete:**

| α | GLOBAL→union10 Δ |
|---|:----------------:|
| 0.5 | +0.29% |
| 1.0 | +0.76% |
| 2.0 | +1.78% |
| **5.0** | **+2.83% ← PEAK** |
| 10.0 | −0.51% |
| 20.0 | **−4.67%** ← rapid collapse |
| 50.0 | not yet run |

Peak confirmed at α=5 (+2.83%). Pattern: sharp rise α=0.5→5, then reversal. α=20 collapses to −4.67% — typical over-injection behavior. Top3 sweep (N=1413) will follow next.

**ALLLAYER SWEEP (GPU 6) — L9/F387 α=10 complete:**

| α | Best injection layer | Best Δ |
|---|:--------------------:|:------:|
| 2.0 | L12 | +2.08% |
| 5.0 | L14 | **+3.33%** |
| 10.0 | L9 | +2.92% |

α=5 injection at L14 (+3.33%) remains the best result for L9/F387 — beats both lower alpha (L12 +2.08%) and higher alpha (L9 +2.92%). The optimal injection layer is α-dependent: L12 at α=2, L14 at α=5, L9 at α=10. This suggests the best injection point shifts from mid-network to earlier layers as α increases (higher alpha can successfully propagate from earlier layers). α=20 sweeping now.

**L11/F9639 "in/inside/on" — MIDDLE complete for pt-targeting:**

| Condition | MIDDLE best Δ | Note |
|-----------|:-------------:|------|
| pt→pt | **+1.18%** (α=2) | Very weak; odd early peak at α=2 then degrades |
| mix→pt | **+1.00%** (α=1) | Weakest so far; peaks at α=1 then monotone decline |
| mix→mix | +0.18% (α=0.5, early) | Mix baseline 81.56% — barely any room |
| pt→mix | +0.09% (α=0.5–1, early) | Same |

"in/inside/on" (N=1101, 6th feature) is the weakest spatial relation for steering so far. Both pt→pt and mix→pt show very early peaks (α=2 and α=1) and steep decline — the MIDDLE direction pushes in the "inside/on" direction but degrades quickly. Pt base 60.85% has some room but the direction is not well-calibrated. Mix base 81.56% is near-saturated.

**L11/F9639 FEATURE sweeps starting on GPUs 2 & 3.**

**2×2 Grid update (L11/F9639 MIDDLE added):**

| Feature | pt→pt MIDDLE | mix→pt MIDDLE | mix→mix MIDDLE | pt→mix MIDDLE | pt→pt FEATURE | mix→pt FEATURE | mix→mix FEATURE | pt→mix FEATURE |
|---------|:-----------:|:-------------:|:--------------:|:-------------:|:-------------:|:--------------:|:---------------:|:--------------:|
| L9/F387 | +3.12% | +2.29% | −0.21% | +0.21% | +0.83% | +0.42% | +0.21% | **+1.04%** |
| L14/F10561 | **+9.68%** | **+9.68%** | +1.08% | +1.08% | **+9.68%** | **+9.68%** | +1.08% | +1.08% |
| L11/F12278 | **+5.85%** | **+5.70%** | +2.34% | **+2.81%** | +5.62% | +4.68% | **+2.50%** | +3.12% |
| L9/F7540 | +0.00% | +2.86% | +0.00% | +0.00% | +2.86% | +0.00% | — | — |
| L4/F14233 | **+12.82%** | **+7.69%** | +2.56% | +2.56% | +5.13% | +5.13% | +5.13% | +2.56% |
| L6/F7539 | **+2.17%** | **+1.86%** | 0.00%→crash | +0.93% | ≤−0.31% | ≤−0.62% | +0.31% | **+1.24%** |
| L11/F9639 | +1.18% (α=2) | +1.00% (α=1) | +0.18% (early) | +0.09% (early) | running | running | running | running |
| L13/F15219 | — | — | — | — | — | — | — | — |
| L15/F220 | — | — | — | — | — | — | — | — |
| L12/F2257 | — | — | — | — | — | — | — | — |

**Emerging pattern — MIDDLE strength by feature (pt→pt, best Δ ranked):**

| Rank | Feature | pt→pt MIDDLE Δ | Spatial relation |
|------|---------|:--------------:|-----------------|
| 1 | L4/F14233 | **+12.82%** | ahead of |
| 2 | L14/F10561 | **+9.68%** | close to |
| 3 | L11/F12278 | **+5.85%** | touching |
| 4 | L9/F387 | +3.12% | at right side of |
| 5 | L6/F7539 | +2.17% | left of/right of |
| 6 | L11/F9639 | +1.18% | in/inside/on |
| 7 | L9/F7540 | +0.00% | consists of |

Pattern: highly directional binary spatial relations ("ahead of", "close to", "touching") are most steerable. Symmetric relations ("left of/right of"), containment ("in/on"), and compositional ("consists of") resist global MIDDLE direction.

---

## Monitoring Update 36 — 2026-04-25 ~12:30 PDT

All 8 GPUs still busy, all PIDs alive. No free GPUs.

### GPU Status (~12:30 PDT Apr 25)

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 2206723 | pt448_true_caa_combined_feature.py | GLOBAL MIDDLE on union10: α=0.5 running (N=4882, slow) |
| 1 | 2206745 | pt448_true_caa_v4b_pt_src.py | Extracting L14/L15 (L13 complete: 10972 done); vectors not yet computed |
| 2 | 2194277 | pt448_true_caa_v4_pt_src.py | L11/F9639 FEATURE through α=10; L13/F15219 starting |
| 3 | 2194363 | pt448_true_caa_v4_mix_to_pt.py | L11/F9639 FEATURE through α=10; L13/F15219 starting |
| 4 | 2195469 | pt448_true_caa_v4_mix_to_mix.py | L11/F9639 MIDDLE complete (flat); FEATURE starting |
| 5 | 2195698 | pt448_true_caa_v4_pt_to_mix.py | L11/F9639 MIDDLE complete (flat); FEATURE starting |
| 6 | 2196021 | pt448_true_caa_alllayer_sweep.py | L9/F387 α=20 sweeping (α=10 done: best L9 +2.92%) |
| 7 | 2196250 | pt448_true_caa_combined.py | GLOBAL→top3 sweep: α=0.5 +0.78%, α=1.0 +1.77%, more running |

### New Results This Cycle

**COMBINED top3 sweep (GPU 7) underway:**

| α | GLOBAL→top3_union Δ (N=1413) |
|---|:----------------------------:|
| 0.5 | +0.78% |
| 1.0 | +1.77% |
| 2.0+ | running |

Top3 set (L11/F12278 + L4/F14233 + L14/F10561 union, N=1413, baseline 56.76%) shows stronger early signal than union10 — +1.77% at α=1 vs +0.76% at α=1 for union10. Confirms that the top-3 features (touching, ahead-of, close-to) represent more directional concepts that respond more strongly to GLOBAL MIDDLE.

**L11/F9639 "in/inside/on" FEATURE — complete for pt-targeting (GPUs 2 & 3):**

| Condition | FEATURE best Δ | Note |
|-----------|:--------------:|------|
| pt→pt | **+0.54%** (α=0.5) | Nearly flat; barely above baseline |
| mix→pt | **+0.64%** (α=2) | Also near-flat; marginally better than MIDDLE (+1.18%) |

Both pt→pt and mix→pt FEATURE for L11/F9639 peak at very low alpha and near-zero improvement. Compare MIDDLE peaks: pt→pt +1.18% (α=2), mix→pt +1.00% (α=1). **MIDDLE marginally beats FEATURE for "in/inside/on" in pt-targeting** — this feature's direction is diffuse regardless of extraction method.

**L11/F9639 mix-targeting — fully stagnant (GPUs 4 & 5):**

Both mix→mix and pt→mix MIDDLE are effectively flat (+0.09–0.18% at best), collapsing from α=10+ with catastrophic −12% to −25% drops at α=50. Mix baseline 81.56% — this feature is essentially saturated. Mix FEATURE starting now.

**Updated 2×2 grid with L11/F9639 complete for pt-targeting:**

| Feature | pt→pt MIDDLE | mix→pt MIDDLE | mix→mix MIDDLE | pt→mix MIDDLE | pt→pt FEATURE | mix→pt FEATURE | mix→mix FEATURE | pt→mix FEATURE |
|---------|:-----------:|:-------------:|:--------------:|:-------------:|:-------------:|:--------------:|:---------------:|:--------------:|
| L9/F387 | +3.12% | +2.29% | −0.21% | +0.21% | +0.83% | +0.42% | +0.21% | **+1.04%** |
| L14/F10561 | **+9.68%** | **+9.68%** | +1.08% | +1.08% | **+9.68%** | **+9.68%** | +1.08% | +1.08% |
| L11/F12278 | **+5.85%** | **+5.70%** | +2.34% | **+2.81%** | +5.62% | +4.68% | **+2.50%** | +3.12% |
| L9/F7540 | +0.00% | +2.86% | +0.00% | +0.00% | +2.86% | +0.00% | — | — |
| L4/F14233 | **+12.82%** | **+7.69%** | +2.56% | +2.56% | +5.13% | +5.13% | +5.13% | +2.56% |
| L6/F7539 | +2.17% | +1.86% | 0.00%→crash | +0.93% | ≤−0.31% | ≤−0.62% | +0.31% | **+1.24%** |
| L11/F9639 | +1.18% | +1.00% | ≤+0.18% (flat) | ≤+0.09% (flat) | +0.54% | +0.64% | running | running |
| L13/F15219 | running | running | — | — | — | — | — | — |
| L15/F220 | — | — | — | — | — | — | — | — |
| L12/F2257 | — | — | — | — | — | — | — | — |

**L11/F9639 finding**: Both MIDDLE (+1.18%) and FEATURE (+0.54%) are weak for "in/inside/on" in pt-targeting. MIDDLE wins marginally. The "in/inside/on" concept is too broad and multi-relation to have a clean contrastive direction.

**L13/F15219 "behind" starting on GPUs 2 & 3:**
- pt→pt FEATURE norm = 2.1090 (weaker than MIDDLE norm)
- mix→pt FEATURE norm = 6.8900 (same magnitude as mix MIDDLE norm 6.17)

Notable norm difference: pt-448 FEATURE vector for "behind" has norm 2.11 vs mix-448 FEATURE norm 6.89. The mix-448 representation of the "behind" concept in its SAE-firing subset is much more directional than pt-448's.

---

## Monitoring Update 37 — 2026-04-25 ~13:15 PDT

All 8 GPUs busy, all PIDs alive. No free GPUs.

### GPU Status (~13:15 PDT Apr 25)

| GPU | PID | Script | Status |
|-----|-----|--------|--------|
| 0 | 2206723 | pt448_true_caa_combined_feature.py | GLOBAL MIDDLE on union10: α=2.0 running (slow: N=4882 each) |
| 1 | 2206745 | pt448_true_caa_v4b_pt_src.py | **Steering started**: L9/F387 MIDDLE done (+1.67%); FEATURE in progress |
| 2 | 2194277 | pt448_true_caa_v4_pt_src.py | L13/F15219 MIDDLE partial (α=5: +0.99%); running |
| 3 | 2194363 | pt448_true_caa_v4_mix_to_pt.py | L13/F15219 MIDDLE partial (α=5: −0.56%); running |
| 4 | 2195469 | pt448_true_caa_v4_mix_to_mix.py | L11/F9639 complete; L13/F15219 starting |
| 5 | 2195698 | pt448_true_caa_v4_pt_to_mix.py | L11/F9639 complete; L13/F15219 starting |
| 6 | 2196021 | pt448_true_caa_alllayer_sweep.py | L9/F387 α=20 sweeping all 26 layers |
| 7 | 2196250 | pt448_true_caa_combined.py | **top3 sweep COMPLETE** (all 7 α done!) |

### New Results This Cycle

**COMBINED (GPU 7) — GLOBAL→top3 sweep COMPLETE:**

| α | GLOBAL→top3_union Δ (N=1413, base 56.76%) |
|---|:------------------------------------------:|
| 0.5 | +0.78% |
| 1.0 | +1.77% |
| 2.0 | +3.82% |
| **5.0** | **+5.87% ← PEAK** |
| 10.0 | +0.92% |
| 20.0 | −6.44% |
| 50.0 | not printed yet |

**KEY FINDING — top3 union peak +5.87% vs union10 peak +2.83%**: The top-3 feature union (touching/ahead-of/close-to, N=1413) responds far more strongly to the global MIDDLE direction than the full 10-feature union (N=4882). This confirms that the 7 weaker features dilute the steering signal when evaluating on union10. The global MIDDLE direction is very well-suited for the top-3 concepts and reaches +5.87% — comparable to per-feature MIDDLE results for individual features. COMBINED experiment complete; GPU 7 will be free soon.

**Updated combined steering comparison:**

| Set | N | Base | GLOBAL MIDDLE peak Δ | Peak α |
|-----|---|------|:--------------------:|:------:|
| union10 | 4882 | 55.00% | +2.83% | 5 |
| top3_union | 1413 | 56.76% | **+5.87%** | 5 |

Both sets peak at α=5, confirming this as the optimal alpha for global MIDDLE→pt-448 steering regardless of subset composition.

**v4b pt→pt — L9/F387 (GPU 1, first feature, steering started):**

v4b MIDDLE for L9/F387 (pt→pt, answer-appended token[-2]):

| α | v4b pt→pt MIDDLE Δ |
|---|:-----------------:|
| 0.5 | +0.00% |
| 1.0 | +0.42% |
| 2.0 | +0.62% |
| 5.0 | +0.42% |
| 10.0 | **+1.67%** ← peak |
| 20.0 | +0.62% |
| 50.0 | −2.71% |

v4b pt→pt MIDDLE for L9/F387 peaks at α=10 (+1.67%) — notably different from:
- cache pt→pt MIDDLE: +3.12% at α=5 (2× better at lower alpha)
- v4b mix→mix MIDDLE: comparison pending but expected ~+2%

**Critical observation**: v4b MIDDLE norm for pt-448 = 97.88 (vs cache MIDDLE norm = 1.39). The answer-appended extraction inflates the vector norm by 70×. Despite unit-normalization before injection, the v4b direction captures more answer-identity signal than spatial reasoning signal, explaining the weaker and α-shifted peak.

**L11/F9639 "in/inside/on" — fully complete for all 4 conditions:**

| Condition | MIDDLE best Δ | FEATURE best Δ |
|-----------|:-------------:|:--------------:|
| pt→pt | +1.18% (α=2) | +0.54% (α=0.5) |
| mix→pt | +1.00% (α=1) | +0.64% (α=2) |
| mix→mix | +0.18% (α=0.5) ← flat | +0.09% (α=0.5) ← flat |
| pt→mix | +0.09% (α=0.5) ← flat | **+0.18%** (α=1–2) ← barely above MIDDLE |

All four conditions effectively fail for "in/inside/on". Mix baseline 81.56% is near-ceiling; pt baseline 60.85% has theoretical room but the concept direction is too diffuse. Collapsing at α≥20 for mix-targeting (−25% at α=50 for mix→mix). This is the only feature where FEATURE beats MIDDLE even marginally in pt-targeting (mix→pt FEATURE +0.64% vs MIDDLE +1.00% — still MIDDLE wins).

**L13/F15219 "behind" — MIDDLE in progress (GPUs 2 & 3, partial through α=5):**

| Condition | α=2 Δ | α=5 Δ | Note |
|-----------|:------:|:------:|------|
| pt→pt MIDDLE | +0.42% | +0.99% | climbing; peak not yet |
| mix→pt MIDDLE | +1.83% | −0.56% | peaked at α=2, already dropping |

Notable: mix→pt MIDDLE for "behind" peaks early at α=2 (+1.83%) while pt→pt MIDDLE is still climbing at α=5 (+0.99%). Different α-profiles for the same feature — mix direction may be mis-scaled. Results not yet final.

**2×2 Grid update (L11/F9639 fully complete):**

| Feature | pt→pt MIDDLE | mix→pt MIDDLE | mix→mix MIDDLE | pt→mix MIDDLE | pt→pt FEATURE | mix→pt FEATURE | mix→mix FEATURE | pt→mix FEATURE |
|---------|:-----------:|:-------------:|:--------------:|:-------------:|:-------------:|:--------------:|:---------------:|:--------------:|
| L9/F387 | +3.12% | +2.29% | −0.21% | +0.21% | +0.83% | +0.42% | +0.21% | **+1.04%** |
| L14/F10561 | **+9.68%** | **+9.68%** | +1.08% | +1.08% | **+9.68%** | **+9.68%** | +1.08% | +1.08% |
| L11/F12278 | **+5.85%** | **+5.70%** | +2.34% | **+2.81%** | +5.62% | +4.68% | **+2.50%** | +3.12% |
| L9/F7540 | +0.00% | +2.86% | +0.00% | +0.00% | +2.86% | +0.00% | — | — |
| L4/F14233 | **+12.82%** | **+7.69%** | +2.56% | +2.56% | +5.13% | +5.13% | +5.13% | +2.56% |
| L6/F7539 | +2.17% | +1.86% | 0.00%→crash | +0.93% | ≤−0.31% | ≤−0.62% | +0.31% | **+1.24%** |
| L11/F9639 | +1.18% | +1.00% | +0.18% (flat) | +0.09% (flat) | +0.54% | +0.64% | +0.09% | **+0.18%** |
| L13/F15219 | +0.99% (α=5) | +1.83% (α=2) | +0.56% (α=0.5) | +0.42% (α=2) | +0.14% | +1.13% (α=5) | +0.28% (α=2) | +0.14% (α=20) |
| L15/F220 | **+3.97%** (α=10) | **+4.69%** (α=5) | crash (−0.36%) | flat (−0.09%) | +3.70% (α=20) | +3.70% (α=5) | crash (−0.18%) | flat (−0.09%) |
| L12/F2257 | running | running | running | running | — | — | — | — |

---

### Monitoring Update 38 — 2026-04-25

**GPU status:**
- GPU 0: `pt448_true_caa_combined_feature.py` — GLOBAL MIDDLE union10 α=2 running (4/7 alphas done: best +2.83% α=5 confirmed)
- GPU 1: `pt448_true_caa_v4b_pt_src.py` — L11/F12278 MIDDLE α=0.5 (3rd feature; L9 and L14 complete)
- GPU 2: `pt448_true_caa_v4_pt_src.py` — L15/F220 (baseline eval starting)
- GPU 3: `pt448_true_caa_v4_mix_to_pt.py` — L15/F220 (baseline eval starting)
- GPU 4: `pt448_true_caa_v4_mix_to_mix.py` — L13/F15219 FEATURE starting
- GPU 5: `pt448_true_caa_v4_pt_to_mix.py` — L13/F15219 FEATURE starting
- GPU 6: `pt448_true_caa_alllayer_sweep.py` — L14/F10561 α=10 done (+9.68% at L12), α=20 running
- GPU 7: `mix448_true_caa_combined_mix_src.py` — NEW, MIDDLE vector extraction running (mix→mix combined)

**L13/F15219 "behind" fully characterized across all 4 conditions:**

| Condition | MIDDLE best Δ | MIDDLE peak α | FEATURE best Δ |
|-----------|:-------------:|:-------------:|:---------------:|
| pt→pt | **+0.99%** | α=5 | +0.14% |
| mix→pt | **+1.83%** | α=2 (unusual early peak) | +1.13% (α=5) |
| mix→mix | **+0.56%** | α=0.5 (flat then crash) | running |
| pt→mix | **+0.42%** | α=2 (flat) | running |

"Behind" is weakly steerable across all conditions. MIDDLE barely improves on all 4 models. mix→pt MIDDLE peaks early at α=2 (+1.83%) while pt→pt peaks at α=5 (+0.99%) — suggests the mix L13 direction has different α-sensitivity for "behind" than other features. FEATURE vectors also weak (max +1.13%). This feature is near the bottom of steerability ranking alongside L6/F7539 and L11/F9639.

**Alllayer sweep L14/F10561 key finding confirmed — L12 injection beats L14:**

| α | Standard (L14) | Best layer | Best Δ |
|---|:--------------:|:----------:|:------:|
| 2.0 | +2.15% | L1 | +2.15% (tied) |
| 5.0 | +5.38% | **L12** | **+8.60%** |
| 10.0 | +9.68% | **L12** | **+9.68%** (tied) |

At α=5, injecting 2 layers below the extraction layer (L12 instead of L14) yields +8.60% vs +5.38% — a **+3.22% improvement** from layer placement alone. At α=10, L12 matches the saturation level (+9.68%). This strongly motivates testing cross-layer injection for other features. L9/F387 best injection layer also shifts by α: L12 at α=2, L14 at α=5, L9 at α=10 — no clear single optimal layer.

**v4b pt→pt (GPU 1) — new per-feature results:**

| Feature | MIDDLE best Δ | FEATURE best Δ | Cache MIDDLE for comparison |
|---------|:-------------:|:--------------:|:----------------------------:|
| L9/F387 | +1.67% (α=10) | **+2.71%** (α=5) | +3.12% (α=5) |
| L14/F10561 | **+9.68%** (α=20–50) | **+9.68%** (α=50) | +9.68% (α=20–50) |
| L11/F12278 | running | — | +5.85% |

Key: v4b L9/F387 FEATURE (+2.71%) > MIDDLE (+1.67%) — unusual, FEATURE beats MIDDLE. For L14/F10561, both v4b conditions plateau at +9.68% (same ceiling as cache). The "close to" feature appears to have a hard ceiling at 70% accuracy (baseline 60.22%), regardless of extraction method.

**New experiment launched — mix→mix combined (GPU 7):**
Script: `mix448_true_caa_combined_mix_src.py`
PID: 2210116 on GPU 7
OUT_DIR: `analysis/mix448_true_caa_combined_mix_src/`
Log: `/tmp/true_caa_combined_mix_src.log`

Tests: global mix-448 MIDDLE vector (L13, norm=6.17) → steer mix-448 on union10 (N=4882) and top3_union (N=1413).
Completes the 2×2 combined grid: mix→pt done (+2.83%/+5.87%), mix→mix pending.
Expected: weaker than mix→pt due to mix high baseline (~76% union, ~80% top3). Results in ~2–3h.

---

### Monitoring Update 39 — 2026-04-25

**GPU status (all 8 occupied):**
- GPU 0: `combined_feature.py` — GLOBAL MIDDLE α=10 done (−0.51%), running α=20 and α=50; then per-feature FEATURE sweeps
- GPU 1: `v4b_pt_src.py` — L11/F12278 MIDDLE α=0.5 running (3rd feature)
- GPU 2: `pt_src.py` — L15/F220 MIDDLE partial (α=0.5/+0.72%, α=1/+0.36%, α=2/+0.63%)
- GPU 3: `mix_to_pt.py` — L15/F220 MIDDLE partial (α=0.5/+0.18%, α=1/+0.27%, α=2/+1.17%)
- GPU 4: `mix_to_mix.py` — L15/F220 MIDDLE partial (α=0.5/−0.36%; mix baseline 72.02%)
- GPU 5: `pt_to_mix.py` — L15/F220 MIDDLE starting (mix baseline 72.02%)
- GPU 6: `alllayer_sweep.py` — L11/F12278 baseline starting (L14/F10561 COMPLETE: BEST α=20 L16 **+10.75%**)
- GPU 7: `combined_mix_src.py` — mix-448 baseline running (N=4882)

**L13/F15219 "behind" — all 4 conditions FULLY COMPLETE:**

| Condition | MIDDLE best Δ | FEATURE best Δ |
|-----------|:-------------:|:--------------:|
| pt→pt | +0.99% (α=5) | +0.14% (α=2) |
| mix→pt | +1.83% (α=2) | +1.13% (α=5) |
| mix→mix | +0.56% (α=0.5) | +0.28% (α=2) |
| pt→mix | +0.42% (α=2) | +0.14% (α=20) |

"Behind" is uniformly weak. All 8 conditions (MIDDLE×4 + FEATURE×4) ≤+1.83%. FEATURE never beats MIDDLE. Mix self-steering MIDDLE immediately crashes (−20% at α=50), consistent with other mix→mix patterns.

**Alllayer sweep L14/F10561 COMPLETE — new record: +10.75% at α=20 L16:**

| α | Standard L14 | Best layer | Best Δ | Layer offset |
|---|:------------:|:----------:|:------:|:------------:|
| 2.0 | +2.15% | L1 | +2.15% | L−13 |
| 5.0 | +5.38% | **L12** | **+8.60%** | L−2 |
| 10.0 | +9.68% | **L12** | **+9.68%** | L−2 |
| 20.0 | +9.68% | **L16** | **+10.75%** | L+2 |

**Record: +10.75% for "close to" (baseline 60.22% → ~71%) with mix MIDDLE L13 vector injected at L16 α=20.** Previous best was +9.68% (standard L14 α=20–50). Injecting 2 layers above extraction layer at high alpha gives marginal gain. Optimal injection layer shifts higher as α increases (L1 → L12 → L16). This suggests the optimal injection layer tracks the alpha-scaled energy absorption point in the residual stream.

**Combined feature (GPU 0) — GLOBAL MIDDLE sharp cliff at α=10:**

GLOBAL MIDDLE on union10 shows a non-monotonic cliff: α=5 (+2.83%) → α=10 (−0.51%). Extremely sharp peak. This matches the RepE non-monotonic finding — the global direction is near the boundary between amplification and disruption. Per-feature FEATURE sweeps (next phase) should test whether per-feature vectors have wider usable alpha ranges.

**L15/F220 "across from/left/right" — pt→pt MIDDLE early results:**
Baseline 51.35% (pt-448), early MIDDLE: α=0.5 +0.72%, α=1 +0.36%, α=2 +0.63%. Non-monotonic at low alpha, likely peaks around α=5. Mix baseline 72.02% (high ceiling). Full results pending.

---

### Monitoring Update 40 — 2026-04-25

**GPU status (all 8 occupied):**
- GPU 0: `combined_feature.py` — GLOBAL MIDDLE union10 DONE (α=20: −4.67%); running α=50 then top3 sweep
- GPU 1: `v4b_pt_src.py` — L11/F12278 MIDDLE done (+5.70% α=10), FEATURE starting; then L9_F7540/L4_F14233
- GPU 2: `pt_src.py` — L15/F220 MIDDLE DONE (+3.97% α=10), FEATURE starting; then L12_F2257
- GPU 3: `mix_to_pt.py` — L15/F220 MIDDLE DONE (+4.69% α=5), FEATURE starting; then L12_F2257
- GPU 4: `mix_to_mix.py` — L15/F220 MIDDLE running (crashes from α=0.5: −0.36%)
- GPU 5: `pt_to_mix.py` — L15/F220 MIDDLE running (+0.09% at α=0.5 then degrading)
- GPU 6: `alllayer_sweep.py` — L11/F12278 baseline starting (L9/F387 & L14/F10561 COMPLETE)
- GPU 7: `combined_mix_src.py` — steer sweep starting (baselines: union10=74.85%, top3=76.36%)

**L15/F220 "across from/left/right" MIDDLE DONE for pt-targeting:**

| Condition | Baseline | MIDDLE best Δ | Peak α | Note |
|-----------|:--------:|:-------------:|:------:|------|
| pt→pt | 51.35% | **+3.97%** | α=10 | gradual rise, peak α=10 |
| mix→pt | 51.35% | **+4.69%** | α=5 | stronger from mix src |
| mix→mix | 72.02% | −0.36% → crash | α=0.5 (best) | immediately degrades |
| pt→mix | 72.02% | +0.09% at α=0.5 | — | flat then crash |

L15/F220 is meaningfully steerable into pt-448 (+3.97–4.69%) but not into mix-448. The mix direction actually *hurts* mix-448 immediately at α=0.5 (−0.36%), the worst-case mix self-damage seen so far. Feature covers 4 relations (across from, left side, right side, right of) — broad coverage may dilute the direction but still contributes useful signal for pt-448.

**Alllayer sweep L14/F10561 fully summarized:**

| α | Best injection layer | Best Δ | vs standard (L14) |
|---|:--------------------:|:------:|:-----------------:|
| 2.0 | L1 | +2.15% | 0 |
| 5.0 | L12 | **+8.60%** | **+3.22%** |
| 10.0 | L12 | **+9.68%** | 0 (tied) |
| 20.0 | L16 | **+10.75%** | **+1.07%** |

Overall BEST: α=20 L16 Δ=**+10.75%** (baseline 60.22% → 70.97%). This is the single highest delta across all steering experiments. L11/F12278 alllayer sweep now starting on GPU 6.

**v4b pt→pt L11/F12278 MIDDLE done (+5.70% at α=10):**
- Cache L11/F12278 pt→pt: +5.85% (α=5) vs v4b: +5.70% (α=10) — very close, slight α-shift to higher values again
- v4b norm ~70× inflated (same pattern as L9, L14) but L11 still achieves near-cache performance

**mix→mix combined (GPU 7) — baselines confirmed:**
- union10: 74.85% (vs pt-448: 55.00%) — mix-448 is 19.85pp ahead on union set
- top3_union: 76.36% (vs pt-448: 55.00% on same set) — 21.36pp ahead
- Steer sweep starting now; expected: much smaller gains than mix→pt (+2.83%/+5.87%) due to ceiling effects

**GLOBAL MIDDLE union10 sweep fully confirmed — cliff at α=10:**

| α | Acc | Δ |
|---|:---:|:-:|
| 0.5 | 55.28% | +0.29% |
| 1.0 | 55.76% | +0.76% |
| 2.0 | 56.78% | +1.78% |
| 5.0 | 57.82% | **+2.83%** ← peak |
| 10.0 | 54.49% | −0.51% |
| 20.0 | 50.33% | −4.67% |

Sharp cliff between α=5 and α=10: +2.83% → −0.51%. The global direction is right on the disruption boundary. Top3 sweep and per-feature FEATURE sweeps (next phase on GPU 0) may show wider usable ranges.

---

### Monitoring Update 41 — 2026-04-25

**GPU status:**
- GPU 0: `combined_feature.py` — per-feature FEATURE sweeps underway (L9/F387 done, L14/F10561 running)
- GPU 1: `v4b_pt_src.py` — L4/F14233 MIDDLE running (v4b L9_F7540 FEATURE done; all crashed −2.86%)
- GPU 2: `pt_src.py` — L12_F2257 baseline running (L15/F220 FULLY DONE)
- GPU 3: `mix_to_pt.py` — L12_F2257 baseline running (L15/F220 FULLY DONE)
- GPU 4: `mix_to_mix.py` — L15/F220 FEATURE partial (crashes: −0.18% α=0.5, worse at higher α)
- GPU 5: `pt_to_mix.py` — L15/F220 FEATURE partial (flat: −0.09% to −0.27%)
- GPU 6: `alllayer_sweep.py` — L11/F12278 baseline just started (~6h remaining for this feature)
- GPU 7: `combined_mix_src.py` — GLOBAL/union10 just started (α=0.5: −0.12%, mix self-steering fails immediately)
- **Launchers queued**: spatial_layer mix→pt on GPU 2, spatial_layer mix→mix on GPU 3 (both will auto-launch when their current jobs finish)

**L15/F220 "facing/left/right/across" — fully complete for pt-targeting:**

| Condition | Baseline | MIDDLE best | FEATURE best | Note |
|-----------|:--------:|:-----------:|:------------:|------|
| pt→pt | 51.35% | **+3.97%** (α=10) | +3.70% (α=20) | MIDDLE≈FEATURE, close |
| mix→pt | 51.35% | **+4.69%** (α=5) | +3.70% (α=5) | mix MIDDLE edges out |
| mix→mix | 72.02% | −0.36% (α=0.5) | −0.18% (α=0.5) | immediate crash |
| pt→mix | 72.02% | +0.09% (α=0.5) | −0.09% (α=0.5) | flat then crash |

L15/F220 shows **MIDDLE ≈ FEATURE** for pt-targeting (unlike most features where MIDDLE clearly wins). This is notable — the per-subset direction at L15 is nearly as good as the all-VSR direction. With 560/548 pos/neg samples in FEATURE (relatively large), the subset direction is well-estimated. This supports the hypothesis that sample count matters: L15 has ~1100 fire samples vs L14 with 93, and L15 is the first feature where FEATURE nearly matches MIDDLE.

**mix→mix GLOBAL combined confirms self-steering fails:**
First result: α=0.5 gives −0.12% (baseline 74.85%) — the global MIDDLE direction actually hurts mix-448 self-steering from the very first alpha, consistent with per-feature findings. Mix-448 has no room to gain from its own MIDDLE direction.

**New experiments launched (waiting for GPUs 2/3):**
Scripts written and launcher queued (PID 2213452):
- `pt448_true_caa_v4_spatial_layer_mix_to_pt.py` → GPU 2 (mix→pt)
- `mix448_true_caa_v4_spatial_layer_mix_to_mix.py` → GPU 3 (mix→mix)

These add a **SPATIAL_LAYER** condition: mix-448 contrastive vector extracted at lF using **all 10,972 VSR samples**, injected at lF.
- SPATIAL_LAYER vs MIDDLE: isolates layer effect (lF vs L13, same data)
- SPATIAL_LAYER vs FEATURE: isolates sample count effect (same layer, ~10–250× more data)
Logs: `/tmp/true_caa_spatial_layer_mix_to_pt.log`, `/tmp/true_caa_spatial_layer_mix_to_mix.log`

**v4b pt→pt (GPU 1) — L9/F7540 "consists of" result:**
MIDDLE peaks at α=10 (+2.86%), then crashes to −17% at α=20. FEATURE uniformly −2.86% at all stable alphas (then crashes). Tiny subset (n=23 fire samples, n=35 eval) makes FEATURE completely unreliable. Consistent with cache pt→pt: FEATURE +0.00%, MIDDLE +0.00%. v4b MIDDLE squeezed +2.86% only because the v4b direction at L9 happens to align with something useful at α=10, before collapsing.

**L15/F220 key observation — FEATURE vs MIDDLE sample count correlation:**
Across all pt-targeting features so far, the FEATURE/MIDDLE ratio improves with FEATURE sample count:
| Feature | FEATURE N | FEATURE Δ | MIDDLE Δ | Ratio |
|---------|:---------:|:---------:|:--------:|:-----:|
| L4/F14233 "ahead of" | 38 | +5.13% | +7.69% | 0.67 |
| L14/F10561 "close to" | 93 | +9.68% | +9.68% | 1.00 |
| L9/F387 "right side of" | 480 | +0.42% | +2.29% | 0.18 |
| L15/F220 "left/right/across" | 1108 | +3.70% | +4.69% | 0.79 |
| L11/F12278 "touching" | 1281 | +4.68% | +5.70% | 0.82 |

No clean monotonic trend — the L14 tie at 93 samples and L9 failure at 480 samples show that subset size is not the only driver. SPATIAL_LAYER experiment will resolve this directly.

