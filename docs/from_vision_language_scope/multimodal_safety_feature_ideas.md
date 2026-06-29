# Multimodal Safety Feature Analysis — Research Ideas
**Date:** 20 April 2026  
**Status:** Backlog — come back after spatial steering experiments wrap up

---

## Core Idea

Apply the same SAE-based mechanistic interpretability pipeline used for spatial features to **safety-relevant features** in PaliGemma 2 3B. The key question mirrors the spatial cross-stage work:

> "Do safety features in mix-448 (instruction-tuned) transfer to pt-448 (pretrained)?"

**Hypothesis:** Safety features should have near-zero cross-stage transfer ratio (created entirely by RLHF/safety fine-tuning), in contrast to spatial features which showed mixed transfer (0.00×–1.00×). This would directly complement Anthropic's model diffing work.

---

## Two Types of Safety Features to Look For

**Type A — "Safety bypass" features** (primary target):
- Fire when model produces unsafe output
- Ablate them → model refuses more → safety score ↑
- Expected: near-zero cross-stage transfer (fine-tuning artifact)

**Type B — "Safety refusal" features**:
- Fire when model correctly refuses
- Ablate them → model complies more → safety score ↓
- Also expected: near-zero cross-stage transfer

Type A is the cleaner story for publication: "this feature causally enables unsafe outputs and was created by fine-tuning."

---

## Recommended Dataset: VLSBench

**HF ID:** `Foreshhh/vlsbench`  
**Total samples:** 2,241  
**Has inline images:** Yes  
**No LLM judge needed:** Category labels are ground-truth, not model-scored  
**Analogous to VSR:** `category` field = VSR `relation` field

### Categories (6 top-level, 19 sub-categories):

| Category | N | Sub-categories |
|----------|---|---------------|
| Violent | 590 | Weapon-Related Violence, Public Violence, Abuse & Physical Altercations, Risky Behaviors |
| Illegal Activity | 539 | Cybercrime, Property Crimes, Drug-Related Offenses, Human Trafficking |
| Self-Harm | 370 | Physical Self-Injury, Substance Abuse & Poisoning, Psychological Harm |
| Erotic | 343 | Adult Content, Erotic Visual Content |
| Hate | 269 | Racial/Ethnic, Cultural Xenophobia, Religious Intolerance, Gender/Sexual Orientation |
| Privacy | 130 | Unauthorized Data Collection, Identity Theft |

Each sample has: `image`, `instruction`, `category`, `sub_category`, `safety_reason` (ground-truth explanation).

### Why VLSBench over alternatives:
- **JailBreakV-28K** (`JailBreakV-28K/JailBreakV-28k`): 28k samples, 15 policy categories, but images NOT inline (external paths) — needs separate download
- **SPA-VL** (`sqrti/SPA-VL`): only 265 harm test samples (too small); 3-level category hierarchy is good but N is thin
- **MM-SafetyBench, FigStep, VLSafe**: not available on HF without gating or at all

---

## Pipeline Adaptation (from Spatial → Safety)

### Step 1: Replace VSR with VLSBench
```python
# Instead of:
#   filter VSR by relation → get spatial subset
# Do:
#   filter VLSBench by category → get safety subset
#   run mix-448 forward → get model response (Yes/No to instruction)
#   label = 1 if model produces unsafe response, 0 if refuses
```

**Key difference from VSR:** VSR has ground-truth labels (true/false spatial statement). VLSBench has ground-truth *categories* but not ground-truth *model responses* — you need to run the model and score whether it complied or refused. 

For PaliGemma format, prompt as: `"Should you follow this instruction? Answer Yes or No.\nInstruction: {instruction}\nAnswer:"` and use Yes/No logit scoring (same as VSR pipeline).

Alternatively: treat all VLSBench samples as "model should refuse" (label=0 for compliance) — simpler, no model run needed for label assignment.

### Step 2: Firing analysis (same as current pipeline)
- Text-only SAE firing on unsafe-response tokens vs VQAv2 baseline
- Use existing `derive_relations_textonly.py` logic with VLSBench categories instead of VSR relations
- Look for features with high odds ratio: P(fires | unsafe category) / P(fires | VQAv2 baseline)

### Step 3: Per-category ablation
- Same 3-point projection ablation
- Metric: compliance rate (% of VLSBench samples where model says "Yes"/complies)
- Ablate feature → compliance rate should drop (model refuses more)

### Step 4: Cross-stage transfer
- Same methodology as `cross_stage_ablation_20april2026.md`
- Apply mix-448 W_dec directions to pt-448
- Hypothesis: transfer ratio ≈ 0 for all safety features
- Compare table against spatial features (mixed 0.00×–1.00×)

---

## The Most Interesting Angle: Text-Only SAE + Visual Jailbreaks

If we find safety bypass features on **text tokens** even for visually-driven attacks (VLSBench images contain harmful visual content), that's a novel mechanistic finding:

> "Visual safety failures in VLMs propagate through the language backbone (text token stream), not the vision encoder. The safety circuit breaks at the text-token level even when the harmful content is in the image."

This is testable because our SAE is text-only by design. If text-token features causally drive unsafe outputs on image-based attacks, the vision encoder is not the locus of the failure.

---

## Other Safety Dataset Options (for future reference)

| Dataset | HF ID | N | Images | Categories | Issue |
|---------|-------|---|--------|-----------|-------|
| JailBreakV-28K | `JailBreakV-28K/JailBreakV-28k` | 28k | Path only | 15 policies | Images not inline |
| SPA-VL | `sqrti/SPA-VL` | 265 (harm test) | Yes | 6 top / 15 mid / 53 fine | Too small |
| BeaverTails | `PKU-Alignment/BeaverTails` | 364k | No | 14 categories | Text-only |
| VLGuard | `ys-zong/VLGuard` | ? | Yes | ? | Gated |
| HarmBench | `walledai/HarmBench` | ? | Yes | ? | Gated |

---

## Publication Framing

**Title candidate:** "Safety Features in Vision-Language Models: A Cross-Stage Mechanistic Analysis"

**Two-panel result:**
1. Spatial features: mixed transfer (0.00×–1.00×), some backbone-latent
2. Safety features: near-zero transfer (~0.00× across all), entirely fine-tuning-created

**Key claim:** Safety alignment in PaliGemma 2 is a fine-tuning artifact with no backbone representation, while spatial reasoning partially pre-exists in the pretrained backbone. This has implications for safety transfer learning and the robustness of safety fine-tuning.

**Directly complements:** Anthropic model diffing blog (sleeper features), Zou et al. RepE (safety representation engineering).

---

## Open Questions

1. Does PaliGemma 2 mix-448 have meaningful ASR on VLSBench? (Small 3B model may refuse most things already → sparse unsafe samples)
2. Are safety features on image tokens or text tokens? Text-only SAE answers this for text side.
3. Would a safety-fine-tuned model show different spatial feature transfer? (Suggests interaction between safety and spatial circuits)
4. Can we do injection steering for safety? Inject safety refusal direction into pt-448 → does it refuse more?
