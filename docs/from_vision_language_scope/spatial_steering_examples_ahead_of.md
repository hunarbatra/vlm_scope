# Spatial Steering — qualitative wins for L4/F14233 (“ahead of, behind”)

**Recipe D** (BACKBONE multi-layer CAA + W_dec[F] feature amplification):
- BACKBONE: α = 1.0 · unit(v_meanpool[L]) at every L ∈ [4, 6, 9, 11, 12, 13, 14, 15]
- + γ = 7.0 · W_dec[L4, F=14233] at L4 only

**Headline (R(F)∩test, n=8)**: baseline 62.50% → steered 87.50% (Δ +25.00pp; 3 wins, 1 losses).

**Collage**: ![collage](imgs/spatial_steering_ahead_of/collage.png)

## Cherry-picked wins (steered = correct, baseline = incorrect)

### 1. “The zebra is ahead of the person.”  (relation: *ahead of*  ·  GT: **Yes**)
![win1](docs/imgs/spatial_steering_ahead_of/win_01_vi9104.jpg)

- **Baseline (pt-448)**: → **No** (58% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (57% conf)  ✓

### 2. “The bicycle is ahead of the car.”  (relation: *ahead of*  ·  GT: **Yes**)
![win2](docs/imgs/spatial_steering_ahead_of/win_02_vi9807.jpg)

- **Baseline (pt-448)**: → **No** (58% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (56% conf)  ✓

### 3. “The cow is ahead of the bus.”  (relation: *ahead of*  ·  GT: **Yes**)
![win3](docs/imgs/spatial_steering_ahead_of/win_03_vi10780.jpg)

- **Baseline (pt-448)**: → **No** (59% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (50% conf)  ✓
