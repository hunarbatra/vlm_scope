# 5 May: Steps for Steering (VSR Recipe — Exact Reproduction Guide)

This document records the exact pipeline that produced +15–30% per-feature deltas on VSR
and how to replicate it for any new dataset.

---

## What Worked (VSR, confirmed 2026-05-05)

**Script:** `scripts/caa_recipe_compare_mix_to_pt_devtest.py`
**Results file:** `analysis/caa_recipe_compare_mix_to_pt_devtest/results.json`

### Confirmed deltas (R(F)∩test subsets):

| Feature | Relation | N | Base | Recipe A (MIDDLE) | Recipe D (BB+WDEC) |
|---------|----------|---|------|-------------------|---------------------|
| L4_F14233 | ahead of | 13 | 46.15% | +15.38% (α=5) | **+30.77%** (α=2, γ=3) |
| L14_F10561 | close to | 26 | 53.85% | +15.38% (α=2) | +15.38% (α=2, γ=1) |
| L9_F7540 | consists of | 7 | 85.71% | +14.29% (α=2) | +14.29% |
| L6_F7539 | left/right of | 93 | 48.39% | +4.30% (α=5) | **+13.98%** (α=5, γ=1) |
| L11_F12278 | touching | 397 | 54.16% | +7.30% (α=5) | +9.82% (α=0.5, γ=3) |
| L12_F2257 | facing | 87 | 50.57% | +12.64% | +14.94% |
| L13_F15219 | behind | 211 | 48.34% | +4.74% | +12.80% |

---

## Step-by-Step Pipeline

### Prerequisites
- mix-448 model: `google/paligemma2-3b-mix-448`
- pt-448 model: `google/paligemma2-3b-pt-448`
- SAE checkpoints: `checkpoints/text-only_layer_{L}.pt` (layers 0–25)
- HF_TOKEN set

### Step 1: Build per-sample hidden state cache (mix-448)

Run mix-448 with `model.generate(max_new_tokens=1, use_cache=False)` on ALL samples.
Hook all 26 layers during prefill (shape[1]>1 guard). Save per sample:

```
vi_{si:05d}.pt → {layer_int: tensor([2304], bfloat16), "correct": bool}
```

Extraction formula (from `pt448_hidden_delta.py` line 167):
```python
h_dict[l] = hiddens[l][0, img_end:, :].mean(0).to(torch.bfloat16).cpu()
```
where `img_end` is from `get_image_token_positions(iids)`.

**For VSR:** `analysis/pt448_hidden_delta/mix_hidden/` — 7680 train + 2292 test = 10972 total
**For DocVQA:** `analysis_docvqa/mix_hidden_cache/` — splits["train"]=4279 samples
**For OCR:** `analysis_ocr/mix_hidden_cache/` — 1000 samples (0..799 train, 800..999 test)

Scripts:
- `scripts/build_hidden_cache_docvqa.py`
- `scripts/build_hidden_cache_ocr.py`

> **IMPORTANT:** Use `max_new_tokens=64` (not 1) to get a real response for the correctness label.
> The "correct" flag needs the model to actually attempt the answer.

### Step 2: Extract SAE feature activations per sample

For each target feature (layer L, feature F), load SAE checkpoint and run encode:
```python
h_text = captured["h"][0, img_end:, :].float()
feat_acts = sae.encode(h_text)[:, feature]  # [n_text_tokens]
acts[si] = feat_acts.mean().item()
```
R(F) = {si : acts[si] > 0}

Output: `sae_acts/acts_L{L}_F{F}.json` with `{"acts": {str(si): float}, "train_end": N, ...}`

Scripts:
- `scripts/extract_sae_acts_docvqa.py`
- `scripts/extract_sae_acts_ocr.py`

### Step 3: Compute label-aware CAA vectors from cache

```python
def compute_meanpool_caa(train_indices, layer):
    pos = neg = None; pn = nn = 0
    for si in train_indices:
        d = torch.load(f"vi_{si:05d}.pt")
        v = d[layer].float()
        if d["correct"]:
            pos = v if pos is None else pos + v; pn += 1
        else:
            neg = v if neg is None else neg + v; nn += 1
    return pos/pn - neg/nn
```

Unit-normalize: `v_unit = v / v.norm().clamp(min=1e-8)`

Compute at all needed layers: BACKBONE layers + MIDDLE layer (13) + each feature's layer + downstream layers for Recipe B.

### Step 4: Build R(F)∩test subsets

```python
ak = {int(x) for x in ad["acts"].keys() if ad["acts"][x] > 0}
test_subset = [si for si in test_indices if si in ak]
```

**Minimum useful subset size: ~5 samples.** For VSR, typical subsets are n=7–397 (5–30% firing rate).

### Step 5: Evaluate 6 recipes on R(F)∩test with pt-448

All recipes use pt-448 for inference with injection hooks on text tokens only:
```python
h[0, img_end:] = h[0, img_end:] + sv.unsqueeze(0)
```

| Recipe | Injection |
|--------|-----------|
| A. MIDDLE | `α · unit(v_CAA[L13])` at L13 |
| B. CAA_SAE_DOWN | `α · unit(v_CAA[lF])` at lF..25 |
| C. BACKBONE | `α · unit(v_CAA[L])` at each L in backbone |
| D. BACKBONE+WDEC | C + `γ · W_dec[F]` at lF only |
| E. SPATIAL_LAYER | `α · unit(v_CAA[lF])` at lF only |
| F. SPATIAL+WDEC | E + `γ · W_dec[F]` at lF |

Alpha sweep: {0.5, 1.0, 2.0, 5.0}. Gamma sweep: {1.0, 3.0, 10.0}.

**VSR backbone layers:** {4, 6, 9, 11, 12, 13, 14, 15} = union of all 10 spatial feature layers.

Correctness for VSR: Yes/No logit comparison (not model.generate).

Scripts:
- `scripts/caa_recipe_compare_mix_to_pt_devtest.py` (VSR — reference)
- `scripts/caa_recipe_docvqa.py` (DocVQA port)
- `scripts/caa_recipe_ocr.py` (OCR port)

---

## Why VSR Works But OCR/DocVQA Is Hard

VSR has a structural advantage: binary spatial relations (Yes/No) with 50/50 label balance,
and spatial SAE features fire selectively (5–30% of test samples). This gives:
- Clean, strong CAA direction (correct vs incorrect is semantically clear)
- R(F) subsets of n=7–397 where the feature is actually active
- No train/test correctness distribution gap

OCR and DocVQA problems:
1. **Train/test gap:** mix-448 gets 68–70% correct on train but only 34–40% correct on test.
   CAA vectors built from train don't transfer.
2. **Feature selectivity:** Most OCR/DocVQA SAE features either fire on 80–100% of samples
   (making R(F) ≈ full dataset) or fire selectively but with OR≈1 (no correct/incorrect signal).
3. **No structural labels:** Unlike VSR binary relations, DocVQA/OCR correctness is harder
   to decompose into clean feature directions.

**Best results so far:**
- OCR-Bench: +1.6pp (L17F13602, Recipe D α=20 γ=10); +1.3pp (L20F10687); +1.2pp (L19F10089, middle α=20)
- DocVQA: +0.37pp (L19F14093 α=1 γ=3); +0.28pp (L19F10089 α=1 γ=10); +0.19pp (middle α=10)
- All gains are marginal (<<1pp meaningful); OCR/DocVQA steering is confirmed hard.

---

## MathVerse MMDiff Pipeline (2026-05-05)

**Goal:** Find math-domain SAE features by contrasting MathVerse vs VQA v2, then steer.

**Dataset:** `hunarbatra/MathVerse_Vision_MCQ` (testmini, 430 samples), MCQ 4-choice.

**Pipeline:**
1. Steps 1-4: Reuse adapted/energy/cosine from OCR pipeline (same SAE checkpoints).
2. Step 5: MathVerse firing (sample-level, 430 × 26 layers × 8 GPUs). **Done: 0 failures.**
3. Step 6: Fisher test (VQA 50K vs MathVerse 430). **Done: 98,265 math features (OR≥3.0, freq_diff≥0.05).**
4. Step 7: Lexical filtering **skipped** — 90%+ pass rate, ~28h cost for no filtering benefit. Used adapted∩math directly.
5. Step 8: **Done** — 19,056 features (adapted ∩ math). Ablation subset: top 240 by OR with 0.1<freq_math<1.0.
6. Ablation: **Done** — 240 features, 3-point projection, MathVerse MCQ + VQA capability control.
7. CAA: **Baseline done. MMDiff running.**

**Key insight:** Pre-filtering to adapted∩math before lexical reduces work 5x (98K → 19K). Lexical adds no useful signal for MathVerse (90%+ pass rate).

**MathVerse ablation (240 features, baseline math=21.75%, VQA=88.40%):**

Top CAA targets (ablating hurts math, ΔVqa ≥ −2pp):

| Feature | ΔMath | ΔVqa | OR |
|---------|-------|------|----|
| L10_F15857 | **-1.42%** | -0.10% | 9,929 |
| L17_F2334 | -0.95% | +1.60% | 15,876 |
| L22_F7530 | -0.95% | +1.20% | 3,439 |
| L11_F8154 | -0.71% | — | 6,042 |
| L13_F4171 | -0.71% | +0.60% | 11,621 |
| L9_F4576 | -0.71% | +1.10% | 20,830 |
| L9_F11613 | -0.71% | -0.10% | 2,663 |

Top suppress targets (ablating helps math):

| Feature | ΔMath | ΔVqa |
|---------|-------|------|
| L21_F64 | **+3.55%** | -3.20% |
| L11_F13125 | +3.31% | +0.20% |
| L15_F15696 | +3.07% | -4.20% |

Note: max ablation drop is only 1.42pp — MathVerse is near-chance for 3B mix-448 (21.75% baseline on 4-choice MCQ).

**MathVerse CAA results:**

**Baseline CAA (L13 MIDDLE, mix-448 → mix-448):**

| α | ΔMath | ΔVqa |
|---|-------|------|
| 0.5 | -0.47pp | +0.20pp |
| 1.0 | -0.24pp | +0.10pp |
| 2.0 | -0.47pp | +0.30pp |
| 3.0 | -0.47pp | +0.40pp |
| 5.0 | -0.47pp | +0.40pp |

All alphas **negative** — L13 CAA direction degrades math. Near-chance baseline (21.75%) makes correct/incorrect residuals a noisy signal.

**MMDiff CAA (top-5 features, in progress):** see `caa_mmdiff_mathverse/results.json` when complete.

**DocVQA ablation (complete, 45 features):**

| Feature | ∆DocVQA | ∆Ctrl | OR |
|---------|---------|-------|-----|
| L19_F10089 | **-21.50%** | +2.0% | 21.8 |
| L17_F13602 | **-11.18%** | n/a | 12.6 |
| L21_F9577 | -5.61% | n/a | 36.2 |
| L19_F14093 | -2.49% | n/a | 27.6 |
| L20_F10687 | -1.85% | n/a | 6.4 |

DocVQA baseline: **62.61%** (mix-448, substring match, 5349 samples).
Note: DocVQA steering only yields ~+0.2–0.4pp max — domain difficulty.

**Scripts:**
- `scripts/multimodal_mathverse/local_analysis_mathverse.py` — full 8-step pipeline
- `scripts/multimodal_mathverse/ablation_mathverse.py` — causal ablation
- `scripts/multimodal_mathverse/caa_mathverse.py` — baseline + MMDiff CAA
- `scripts/multimodal_mathverse/auto_launch_mathverse.sh` — auto-launcher

---

## Launch Commands (Exact)

```bash
# VSR (reference — already done)
CUDA_VISIBLE_DEVICES=7 python3 -B scripts/caa_recipe_compare_mix_to_pt_devtest.py

# DocVQA full pipeline (sequential on one GPU)
CUDA_VISIBLE_DEVICES=X python3 -u scripts/build_hidden_cache_docvqa.py
CUDA_VISIBLE_DEVICES=X python3 -u scripts/extract_sae_acts_docvqa.py
CUDA_VISIBLE_DEVICES=X python3 -u scripts/caa_recipe_docvqa.py

# OCR full pipeline
CUDA_VISIBLE_DEVICES=X python3 -u scripts/build_hidden_cache_ocr.py
CUDA_VISIBLE_DEVICES=X python3 -u scripts/extract_sae_acts_ocr.py
CUDA_VISIBLE_DEVICES=X python3 -u scripts/caa_recipe_ocr.py
```

Always use `--gpu 0` when `CUDA_VISIBLE_DEVICES` is set (device ordinal resets to 0).

---

## Key Files

| File | Purpose |
|------|---------|
| `analysis/pt448_hidden_delta/mix_hidden/vi_NNNNN.pt` | VSR per-sample hidden cache |
| `analysis_docvqa/mix_hidden_cache/vi_NNNNN.pt` | DocVQA per-sample hidden cache |
| `analysis_ocr/mix_hidden_cache/vi_NNNNN.pt` | OCR per-sample hidden cache |
| `analysis/mix_sae_acts/acts_L{L}_F{F}.json` | VSR SAE feature activations |
| `analysis_docvqa/sae_acts/acts_L{L}_F{F}.json` | DocVQA SAE feature activations |
| `analysis_ocr/sae_acts/acts_L{L}_F{F}.json` | OCR SAE feature activations |
| `analysis/caa_recipe_compare_mix_to_pt_devtest/results.json` | VSR recipe results |
| `analysis_docvqa/caa_recipe_results/results.json` | DocVQA recipe results |
| `analysis_ocr/caa_recipe_results/results.json` | OCR recipe results |
| `checkpoints/text-only_layer_{L}.pt` | SAE checkpoints (layers 0–25) |
| `analysis_docvqa/splits.json` | DocVQA train/test index splits |
