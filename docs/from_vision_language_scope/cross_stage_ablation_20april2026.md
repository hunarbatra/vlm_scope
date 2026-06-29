# Cross-Stage Ablation of Spatial Features in PaliGemma 2 3B
**Date:** 20 April 2026

## Overview

We investigate whether spatial features detected in the instruction-tuned PaliGemma 2 3B (mix-448) model are already present and causally active in the pretrained backbone (pt-448), or whether they emerge primarily from fine-tuning.

**Models:**
- `google/paligemma2-3b-mix-448` — instruction-tuned (fine-tuned on spatial reasoning tasks)
- `google/paligemma2-3b-pt-448` — pretrained only (no instruction tuning)

**SAEs:** JumpReLU SAEs trained on text-token activations of the mix-448 model (`text-only_layer_*.pt`). The same W_dec feature directions are applied to both models.

**Evaluation dataset:** VSR (`cambridgeltl/vsr_random`, all splits: train + dev + test, ~10,972 samples)

**Ablation method:** 3-point projection across all 26 layers — projects the feature direction out of `attn_out`, `mlp_out`, and `layer_out` at every transformer layer for text tokens only. Implemented via NNsight traces.

---

## Methodology

### Setup
For each detected spatial feature (e.g. L9/F387) in mix-448, we run the **exact same ablation** on both models:

1. Load mix-448 SAE decoder direction: `W_dec[387]` from layer 9
2. Run VSR on the relation-filtered subset (e.g. all "at the right side of" samples)
3. At each forward pass, project out `W_dec[387]` from the residual stream across all 26 layers (3-point: attn_out, mlp_out, layer_out) for text tokens
4. Compare accuracy drop vs baseline

This is done identically for mix-448 and pt-448 — same feature vector, same ablation, different model.

### What this measures

> "How much does removing this specific mix-448 spatial direction hurt each model's VSR accuracy?"

The comparison of the two deltas directly shows how much of this spatial capability pre-existed in the backbone vs was created by fine-tuning. No pt-448 SAE is needed — we are intentionally ablating the **exact same L9/F387 direction** in both models, not searching for an "equivalent" pt-448 feature.

---

## Results

Each feature is evaluated on the VSR subset filtered to its associated spatial relation(s). Baseline accuracy is computed per relation subset (not global VSR accuracy).

| Layer | Feature | VSR Relation | ∆ pt-448 (pre-trained) | ∆ mix-448 (instruction-tuned) | Transfer Ratio |
|-------|---------|--------------|------------------------|-------------------------------|----------------|
| 9 | 387 | at the right side of | -2.08% | -30.62% | 0.07× |
| 14 | 10561 | close to | -7.53% | -18.28% | 0.41× |
| 11 | 12278 | touching | -4.84% | -12.10% | 0.40× |
| 9 | 7540 | consists of | -2.86% | -11.43% | 0.25× |
| 4 | 14233 | ahead of | -10.26% | -10.26% | 1.00× |
| 6 | 7539 | left of; right of | -4.64% | -9.60% | 0.48× |
| 11 | 9639 | in; inside; on | -2.09% | -8.63% | 0.24× |
| 13 | 15219 | behind | +1.55% | -8.04% | 0.00× |
| 15 | 220 | across from; at the left side of | -2.08% | -7.58% | 0.27× |
| 12 | 2257 | facing | +3.27% | -6.86% | 0.00× |

*Transfer ratio = ∆ pt-448 / ∆ mix-448. Higher = more of the VSR drop is already present in the pretrained backbone.*

---

## Key Findings

### 1. Most spatial features are fine-tuning-dominant
The majority of features show significantly larger drops in mix-448 than in pt-448. Mean transfer ratio across features with negative pt-448 delta: ~0.30×. Fine-tuning amplifies these spatial directions by roughly 3× in causal effect.

### 2. One feature fully transfers: L4/F14233 "ahead of"
Transfer ratio of 1.00× — the drop is identical in both models (-10.26%). This suggests the "ahead of" spatial direction is fully encoded in the pretrained Gemma-2-2B backbone before any VLM fine-tuning. Fine-tuning neither strengthens nor weakens its causal role.

### 3. Two features show no transfer: L13/F15219 "behind", L12/F2257 "facing"
Positive ∆ in pt-448 (+1.55%, +3.27%) with meaningful negative ∆ in mix-448 (-8.04%, -6.86%). These directions appear to be created by fine-tuning — ablating them in pt-448 slightly improves performance, suggesting they represent noise or interference in the pretrained model. Fine-tuning repurposes these directions into genuine spatial features.

### 4. Moderate transfer for proximity/contact relations
"close to" (0.41×), "touching" (0.40×), "left of; right of" (0.48×) show the highest transfer among non-trivial features. Proximity and lateral relations have partial backbone representation.

### 5. Near-zero transfer for egocentric/oriented relations
"at the right side of" (0.07×), "in; inside; on" (0.24×) transfer very weakly. These finer-grained spatial distinctions appear to be primarily instruction-tuning-driven.

---

## Interpretation

The results support a **mixed encoding hypothesis**:

- The Gemma-2-2B pretrained backbone encodes some spatial relations (particularly coarse ones like "ahead of", proximity relations) causally and robustly
- Fine-tuning dramatically amplifies the causal role of these directions (~3× on average) and introduces new spatial directions that have no pretrained analogue
- The same W_dec direction is applied to both models by design — we are not searching for pt-448's own spatial features, we are asking how much the mix-448 spatial direction matters in the backbone

---

## Limitations

- No per-relation N weighting in transfer ratio computation
- pt-448 baselines vary by relation subset (61–65% for these subsets, vs ~52–55% global)
- Fine-tuning rotates activation space directions, so the mix-448 W_dec direction may not align perfectly with however pt-448 internally encodes the same spatial concept — this means pt-448 drops may slightly underestimate the true pretrained spatial sensitivity for that relation
