# Spatial Steering — qualitative wins for L6/F7539 (“left of, right of, across from, alongside, below, facing away from, at the back of”)

**Recipe D** (BACKBONE multi-layer CAA + W_dec[F] feature amplification):
- BACKBONE: α = 1.0 · unit(v_meanpool[L]) at every L ∈ [4, 6, 9, 11, 12, 13, 14, 15]
- + γ = 7.0 · W_dec[L6, F=7539] at L6 only

**Headline (R(F)∩test, n=69)**: baseline 50.72% → steered 47.83% (Δ -2.90pp; 10 wins, 12 losses).

**Collage**: ![collage](imgs/spatial_steering_left_right/collage.png)

## Cherry-picked wins (steered = correct, baseline = incorrect)

### 1. “The laptop is left of the sandwich.”  (relation: *left of*  ·  GT: **Yes**)
![win1](docs/imgs/spatial_steering_left_right/win_01_vi9325.jpg)

- **Baseline (pt-448)**: → **No** (51% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (64% conf)  ✓

### 2. “The cat is right of the umbrella.”  (relation: *right of*  ·  GT: **Yes**)
![win2](docs/imgs/spatial_steering_left_right/win_02_vi9871.jpg)

- **Baseline (pt-448)**: → **No** (57% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (57% conf)  ✓

### 3. “The couch is left of the bed.”  (relation: *left of*  ·  GT: **Yes**)
![win3](docs/imgs/spatial_steering_left_right/win_03_vi9514.jpg)

- **Baseline (pt-448)**: → **No** (53% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (59% conf)  ✓

### 4. “The cat is left of the sink.”  (relation: *left of*  ·  GT: **Yes**)
![win4](docs/imgs/spatial_steering_left_right/win_04_vi10740.jpg)

- **Baseline (pt-448)**: → **No** (59% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (53% conf)  ✓

### 5. “The bowl is right of the hot dog.”  (relation: *right of*  ·  GT: **Yes**)
![win5](docs/imgs/spatial_steering_left_right/win_05_vi10894.jpg)

- **Baseline (pt-448)**: → **No** (60% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (51% conf)  ✓

### 6. “The hot dog is right of the cup.”  (relation: *right of*  ·  GT: **Yes**)
![win6](docs/imgs/spatial_steering_left_right/win_06_vi9689.jpg)

- **Baseline (pt-448)**: → **No** (55% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (55% conf)  ✓

### 7. “The cow is right of the clock.”  (relation: *right of*  ·  GT: **Yes**)
![win7](docs/imgs/spatial_steering_left_right/win_07_vi9509.jpg)

- **Baseline (pt-448)**: → **No** (59% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (50% conf)  ✓

### 8. “The cup is left of the donut.”  (relation: *left of*  ·  GT: **Yes**)
![win8](docs/imgs/spatial_steering_left_right/win_08_vi10084.jpg)

- **Baseline (pt-448)**: → **No** (53% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (55% conf)  ✓

### 9. “The chair is left of the teddy bear.”  (relation: *left of*  ·  GT: **Yes**)
![win9](docs/imgs/spatial_steering_left_right/win_09_vi8934.jpg)

- **Baseline (pt-448)**: → **No** (51% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (56% conf)  ✓

### 10. “The pizza is right of the bottle.”  (relation: *right of*  ·  GT: **Yes**)
![win10](docs/imgs/spatial_steering_left_right/win_10_vi8987.jpg)

- **Baseline (pt-448)**: → **No** (54% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (53% conf)  ✓
