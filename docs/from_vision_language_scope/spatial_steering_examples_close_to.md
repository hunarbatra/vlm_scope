# Spatial Steering — qualitative wins for L14/F10561 (“by, close to, connected to”)

**Recipe D** (BACKBONE multi-layer CAA + W_dec[F] feature amplification):
- BACKBONE: α = 1.0 · unit(v_meanpool[L]) at every L ∈ [4, 6, 9, 11, 12, 13, 14, 15]
- + γ = 7.0 · W_dec[L14, F=10561] at L14 only

**Headline (R(F)∩test, n=20)**: baseline 60.00% → steered 70.00% (Δ +10.00pp; 3 wins, 1 losses).

**Collage**: ![collage](imgs/spatial_steering_close_to/collage.png)

## Cherry-picked wins (steered = correct, baseline = incorrect)

### 1. “The bench is close to the laptop.”  (relation: *close to*  ·  GT: **Yes**)
![win1](docs/imgs/spatial_steering_close_to/win_01_vi9067.jpg)

- **Baseline (pt-448)**: → **No** (59% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (54% conf)  ✓

### 2. “The person is close to the cat.”  (relation: *close to*  ·  GT: **Yes**)
![win2](docs/imgs/spatial_steering_close_to/win_02_vi10771.jpg)

- **Baseline (pt-448)**: → **No** (54% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (55% conf)  ✓

### 3. “The sheep is close to the truck.”  (relation: *close to*  ·  GT: **Yes**)
![win3](docs/imgs/spatial_steering_close_to/win_03_vi10326.jpg)

- **Baseline (pt-448)**: → **No** (53% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (57% conf)  ✓
