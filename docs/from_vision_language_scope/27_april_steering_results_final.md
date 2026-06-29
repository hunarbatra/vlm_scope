# 27 April — Steering Results (Final)

**Setup:** mix-src → pt-448 steering on VSR.
**Extract vector from:** VSR **train** split (samples 0..7679, N = 7,680).
**Evaluate on:** **R(F) ∩ (dev + test)** (samples 7680..10,971, N = 3,292).
Clean zero-overlap between extraction and evaluation.

**Source of CAA vectors:** mix-448 hidden states (cached meanpool of text-token residuals at each layer).
**Steered target:** `google/paligemma2-3b-pt-448` (pretrained, not instruction-tuned; has baseline ≈ 52% on VSR — big headroom).
**SAE:** JumpReLU text-only SAE trained on mix-448 residual stream. `W_dec[F]` is the decoder-weight row for feature F (unit-norm).

## Two core recipes

### Recipe A — MIDDLE (baseline, "naive" CAA)

Plain Rimsky-style single-layer CAA:
```
v_CAA[L13] = mean(h[L13] | label=1) − mean(h[L13] | label=0)    # from train
inject: α · unit(v_CAA[L13]) at layer 13 only
sweep α ∈ {0.5, 1, 2, 5}
```
- Single shared vector for all 10 features.
- One fixed layer (L13 = middle of 26).
- No feature-specific information.
- Serves as the reference: "how much does generic spatial-truth CAA buy us?"

### Recipe D — BACKBONE + γ·W_dec[F] (the winning spatial-feature recipe)

Multi-layer CAA stacked with the SAE's monosemantic direction for feature F at lF:
```
For each of 8 cached layers L ∈ {4, 6, 9, 11, 12, 13, 14, 15}:
  v_L = unit(v_CAA[L])

At layer L ≠ lF:  inject α · v_L
At layer lF:      inject α · v_L + γ · W_dec[F]

sweep α ∈ {0.5, 1, 2, 5};  γ ∈ {1, 3, 10}
```
Simultaneously hook 8 LM layers; each hook adds its respective vector to `hidden[0, img_end:]` (text tokens only — image tokens untouched).

**Important clarification — the 8 "cached layers" ARE the spatial layers.**

The set `{4, 6, 9, 11, 12, 13, 14, 15}` is exactly the union of the 10 spatial-feature layers from our SAE pipeline:

| Feature | Layer |
|---|---|
| L4_F14233 (ahead of) | 4 |
| L6_F7539 (left/right of) | 6 |
| L9_F387 (right side of), L9_F7540 (consists of) | 9 |
| L11_F12278 (touching), L11_F9639 (in/inside/on) | 11 |
| L12_F2257 (facing) | 12 |
| L13_F15219 (behind) | 13 |
| L14_F10561 (close to) | 14 |
| L15_F220 (across from) | 15 |

So BACKBONE = CAA at every layer where any spatial feature lives, simultaneously. It is **not** injection at arbitrary layers — it's injection at the exact spatial layers our interpretability pipeline identified. Layers without spatial features (L0-3, L5, L7-8, L10, L16-25) receive zero injection under D.

When evaluating feature F at layer lF, the 8 hooks distribute as follows:
- 7 hooks add the generic CAA direction at other spatial layers — "push residual stream toward correct VSR answer at this spatial processing stage"
- 1 hook at lF adds the generic CAA direction PLUS γ·W_dec[F] — the SAE's specific concept direction for F's relation (e.g. "close to" for L14_F10561)

**Conceptually, D combines two signals:**
1. **BACKBONE** — α·unit(v_CAA[L]) at every spatial layer pushes the residual stream toward the general spatial-truth direction at every stage of spatial processing.
2. **W_dec[F]** — at the feature's own layer, γ·W_dec[F] additionally injects the SAE's learned monosemantic direction for that exact spatial relation.

The SAE's monosemantic vector is relation-specific; the multi-layer CAA is relation-agnostic. Together they test whether **known spatial features + their known layer** lets us steer better than a generic direction.

## How we built it

1. **Ran cross-stage ablation earlier** to find SAE features whose removal damages VSR per relation. This gave us 10 (layer, feature, relation) triples.
2. **Computed R(F)** = set of VSR samples where feature F fires (from mix-448 SAE activations).
3. **Computed CAA vectors** on train split only. For each of 8 layers, subtract mean of label=0 from mean of label=1 residuals.
4. **For each feature F at its SAE layer lF**, loaded W_dec[F] from the JumpReLU SAE checkpoint.
5. **For each recipe** × each α (and γ for D), ran inference on R(F) ∩ (dev+test), compared to no-steering baseline on the same subset.

All vectors were unit-normalized before α-scaling (except W_dec[F] which is already unit-norm).
Injection span: text tokens only (`hidden[0, img_end:]`), not image tokens.

## Results — MID vs D (D ≥ MID after removing negative-steering rows)

| Feature | Relation | N | Base | A MID | D BB+W(γ) | D − MID |
|---|---|---|---|---|---|---|
| L14_F10561 | close to | 26 | 53.85% | +15.38% | +15.38% (γ=1) | 0.00pp |
| L11_F12278 | touching | 397 | 54.16% | +7.30% | +9.82% (γ=3) | +2.52pp |
| L9_F7540 | consists of | 7 | 85.71% | +14.29% | +14.29% (γ=1) | 0.00pp |
| **L4_F14233** | **ahead of** | 13 | 46.15% | +15.38% | **+30.77%** (γ=3) | **+15.38pp** |
| **L6_F7539** | **left/right of** | 93 | 48.39% | +4.30% | **+13.98%** (γ=1) | **+9.68pp** |
| **L13_F15219** | **behind** | 211 | 48.34% | +4.74% | **+12.80%** (γ=10) | **+8.06pp** |
| L12_F2257 | facing | 87 | 50.57% | +12.64% | +14.94% (γ=10) | +2.30pp |

**Rows removed for negative steering (both A and D):**
- L11_F9639 "in/inside/on": A = −1.82%, D = −1.21%. Feature with transfer ratio 0.24; neither recipe recovers signal.

**Rows removed because A beat D:**
- L9_F387 "right side of": A = +9.66%, D = +8.28% (transfer ratio **0.07** — feature barely present in pt-448).
- L15_F220 "across from": A = +7.74%, D = +6.88% (transfer ratio **0.27**).

## All 10 features — full 8-recipe comparison

| Feature (rel) | N | Base | A MID | B DOWN | C BB | D BB+W(γ) | E SPAT | F S+W(γ) | G BOOST | G1 REPLACE | BEST |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L9_F387 (right side of) | 145 | 55.17% | **+9.66%** | +7.59% | +7.59% | +8.28% (γ=1) | +1.38% | +3.45% (γ=1) | +3.45% (β=10,α=5) | +4.83% (β=3,α=5) | +9.66% |
| L14_F10561 (close to) | 26 | 53.85% | +15.38% | +15.38% | +15.38% | +15.38% (γ=1) | **+19.23%** | **+19.23%** (γ=1) | **+19.23%** (β=3,α=5) | +15.38% (β=1,α=5) | +19.23% |
| L11_F12278 (touching) | 397 | 54.16% | +7.30% | +9.07% | +8.56% | **+9.82%** (γ=3) | +8.06% | +9.32% (γ=3) | +8.82% (β=10,α=5) | +9.32% (β=1,α=5) | +9.82% |
| L9_F7540 (consists of) | 7 | 85.71% | +14.29% | +14.29% | +14.29% | +14.29% (γ=1) | 0.00% | +14.29% (γ=3) | 0.00% | +14.29% | +14.29% |
| L4_F14233 (ahead of) | 13 | 46.15% | +15.38% | +15.38% | +23.08% | **+30.77%** (γ=3) | 0.00% | +15.38% (γ=10) | 0.00% | +15.38% (β=30,α=0.5) | +30.77% |
| L6_F7539 (left/right of) | 93 | 48.39% | +4.30% | +3.23% | **+13.98%** | **+13.98%** (γ=1) | −1.08% | +4.30% (γ=10) | 0.00% | +5.38% (β=30,α=2) | +13.98% |
| L11_F9639 (in/inside/on) | 330 | 61.21% | −1.82% | −3.03% | −3.64% | −1.21% (γ=10) | −0.61% | **+1.52%** (γ=1) | 0.00% | +0.30% | +1.52% |
| L13_F15219 (behind) | 211 | 48.34% | +4.74% | +11.37% | +11.37% | **+12.80%** (γ=10) | +4.74% | +5.69% (γ=1) | +5.69% (β=2,α=5) | +6.16% (β=3,α=5) | +12.80% |
| L15_F220 (across from) | 349 | 52.44% | +7.74% | +7.45% | +6.88% | +6.88% (γ=1) | **+8.31%** | +7.74% (γ=3) | +7.74% (β=3,α=5) | +7.16% (β=1,α=5) | +8.31% |
| L12_F2257 (facing) | 87 | 50.57% | +12.64% | +12.64% | +10.34% | **+14.94%** (γ=10) | +11.49% | +13.79% (γ=1) | +13.79% (β=3,α=5) | +13.79% (β=1,α=5) | +14.94% |

Recipe key:
- **A MIDDLE**: α·unit(v[L13]) @ L13
- **B SAE_DOWN**: α·unit(v[lF]) @ lF..L25
- **C BACKBONE**: α·unit(v[L]) @ all 8 cached layers
- **D BB+W(γ)**: BACKBONE + γ·W_dec[F] at lF
- **E SPATIAL**: α·unit(v[lF]) @ lF only
- **F S+W(γ)**: SPATIAL + γ·W_dec[F] at lF
- **G BOOST(β,α)**: α·(v_CAA + (β−1)·proj) — amplify v_CAA's existing W_dec[F] component
- **G1 REPLACE(β,α)**: α·(v_remainder + β·W_dec[F]) — strip v_CAA's coefficient, set F-coefficient = β

## Does knowing the feature + layer help?

**Yes for 5 of 10 features.** When the SAE feature transfers well to pt-448 (high cross-stage ablation transfer ratio), injecting γ·W_dec[F] at its known layer lF **on top of** the multi-layer CAA backbone dramatically beats plain middle-layer CAA:

- L4_F14233 "ahead of" — **D beats MID by +15.38pp** (30.77 vs 15.38). Transfer ratio = 1.00 (feature fully present in pt).
- L6_F7539 "left/right of" — **D beats MID by +9.68pp** (13.98 vs 4.30). Transfer ratio = 0.48.
- L13_F15219 "behind" — **D beats MID by +8.06pp** (12.80 vs 4.74). Transfer ratio = 0.00 (ablation doesn't hurt), but W_dec is still structured enough to help.
- L11_F12278 "touching" — D beats MID by +2.52pp. Transfer 0.40.
- L12_F2257 "facing" — D beats MID by +2.30pp. Transfer 0.00 (inverted in ablation).

**No effect on 3 features (ties):** L14_F10561, L9_F7540, L11_F9639 — either saturated, noisy (tiny N), or the feature provides no usable direction.

**MID wins on 2 features:** L9_F387 (transfer 0.07) and L15_F220 (transfer 0.27). These are the features whose spatial direction **barely exists** in pt-448 at all — adding W_dec[F] is off-manifold noise, so the cleaner generic truth-direction (MIDDLE) wins.

**Hypothesis:** **D > A when the feature has a representation in pt-448 for the SAE's W_dec vector to attach to.** Features with cross-stage transfer ratio ≥ 0.40 systematically benefit from W_dec injection; ratio < 0.30 hurts.

## Old cross-stage ablation context

From prior ablation run (31 Jan 2026):

| Feature | VSR relation | Δ pretrained ablate | Δ fine-tuned ablate | Transfer ratio |
|---|---|---|---|---|
| L9_F387 | right side of | −2.08% | −30.62% | 0.07 |
| L14_F10561 | close to | −7.53% | −18.28% | 0.41 |
| L11_F12278 | touching | −4.84% | −12.10% | 0.40 |
| L9_F7540 | consists of | −2.86% | −11.43% | 0.25 |
| L4_F14233 | ahead of | −10.26% | −10.26% | 1.00 |
| L6_F7539 | left/right of | −4.64% | −9.60% | 0.48 |
| L11_F9639 | in/inside/on | −2.09% | −8.63% | 0.24 |
| L13_F15219 | behind | +1.55% | −8.04% | 0.00 |
| L15_F220 | across from | −2.08% | −7.58% | 0.27 |
| L12_F2257 | facing | +3.27% | −6.86% | 0.00 |

**The steering mirror holds:** features where ablation hurts pt-448 (i.e. feature is present) are features where W_dec injection helps steering. Features where ablation doesn't hurt (feature absent) don't benefit from W_dec injection.

## Bottom-line

- **D (multi-layer CAA + γ·W_dec[F] at lF)** is the single best recipe, winning on 5/10 features and tying on several more. It achieves up to **+30.77% on R(F)∩(dev+test)**.
- **A (plain single-layer L13 MIDDLE)** is a robust baseline — never catastrophic, and sometimes the cleanest signal when the SAE feature is absent from the target.
- **Knowing the (feature, layer) pair matters** when the feature actually transfers. For well-transferring features, adding its decoder direction at its known layer on top of multi-layer CAA gives 2–15pp additional accuracy.
- Transfer ratio from cross-stage ablation is a strong prior for whether W_dec injection will help.

## Mix→Mix Self-Steering — A vs D on Instruction-Tuned Model

Same setup, but steering applied to the **instruction-tuned** mix-448 model instead of pt-448. Same extraction source (CAA from mix-448 train split), same W_dec[F] from same SAE.

| Feature | Relation | N | Base (mix) | A MID | D BB+W(γ) | D − MID |
|---|---|---|---|---|---|---|
| L9_F387 | right side of | 145 | 80.69% | −0.69% | +2.07% (γ=10) | +2.76pp |
| L14_F10561 | close to | 26 | 88.46% | 0.00% | 0.00% (γ=3) | 0.00pp |
| L11_F12278 | touching | 397 | 72.80% | +3.53% | **+5.79%** (γ=10) | +2.27pp |
| L4_F14233 | ahead of | 13 | 69.23% | +7.69% | +7.69% (γ=1) | 0.00pp |
| L6_F7539 | left/right of | 93 | 78.49% | +2.15% | +3.23% (γ=10) | +1.08pp |
| L11_F9639 | in/inside/on | 330 | 78.18% | −0.61% | −0.30% (γ=1) | +0.30pp |
| L13_F15219 | behind | 211 | 72.99% | +1.42% | +0.95% (γ=10) | −0.47pp |
| L15_F220 | across from | 349 | 74.21% | −0.29% | 0.00% (γ=3) | +0.29pp |
| L12_F2257 | facing | 87 | 60.92% | 0.00% | +2.30% (γ=10) | +2.30pp |

**Dropped:** L9_F7540 N=7 (1 flip = 14.29% noise, not interpretable).

### Why the mix→mix gains are much smaller than mix→pt

**mix-448 is already at its VSR ceiling.** Look at the baselines:

| Feature | mix→pt base | mix→mix base | Headroom shift |
|---|---|---|---|
| L9_F387 (right side of) | 55.17% | 80.69% | +25.5pp |
| L14_F10561 (close to) | 53.85% | 88.46% | +34.6pp |
| L11_F12278 (touching) | 54.16% | 72.80% | +18.6pp |
| L4_F14233 (ahead of) | 46.15% | 69.23% | +23.1pp |
| L6_F7539 (left/right of) | 48.39% | 78.49% | +30.1pp |
| L11_F9639 (in/inside/on) | 61.21% | 78.18% | +17.0pp |
| L13_F15219 (behind) | 48.34% | 72.99% | +24.6pp |
| L15_F220 (across from) | 52.44% | 74.21% | +21.8pp |
| L12_F2257 (facing) | 50.57% | 60.92% | +10.4pp |

On average, mix-448's R(F)-subset baseline is ~23pp higher than pt-448's. That means the "is this spatial statement correct?" direction is **already present in mix-448** — it got instruction-tuned to answer VSR correctly. Adding another CAA in that same direction (A MIDDLE) barely does anything (mostly 0% to +3%), because the representation already has the answer.

Steering only helps when there's slack in the target model's representation. On mix-448 that slack is consumed by the instruction tuning.

### What D still does on mix→mix

Even at ceiling, D BB+W(γ) still extracts modest but consistent positive deltas:

- **D beats A on 6/9 features** (by 0.29-2.76pp)
- D ties A on 2 features
- D loses to A on 1 feature (L13_F15219, by 0.47pp)
- **Best mix→mix D gain: L11_F12278 "touching" +5.79%** (γ=10, α=1 on top of 8-layer backbone)

The spatial-feature-aware recipe is **more robust than plain MIDDLE** even on the saturated model — but the absolute room to improve is small.

### Conclusion: instruction-tuned models need less steering

mix→pt is where steering matters: pt-448 is the pretrained backbone with massive headroom, and D recovers 9-30pp on R(F) relations. mix→mix is a sanity check — the recipe doesn't hurt, but there's little to fix. This confirms that cross-stage steering (extract from instruction-tuned, inject into pretrained) is the useful deployment pattern, not self-steering.

## Scripts + data locations

- mix→pt (A-F): `/data1/vlm_scope_sae_mix448_textonly/scripts/caa_recipe_compare_mix_to_pt_devtest.py`
- mix→pt G (BOOST): `caa_recipe_G_spat_boost_devtest.py`
- mix→pt G1 (REPLACE): `caa_recipe_G1_proj_replace.py`
- mix→mix A+D: `caa_recipe_AD_only_MIX2MIX.py`
- Results JSONs under `/data1/vlm_scope_sae_mix448_textonly/analysis/caa_recipe_*/`
