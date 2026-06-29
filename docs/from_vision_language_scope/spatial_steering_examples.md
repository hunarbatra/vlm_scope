# Spatial Steering — qualitative wins for L12/F2257 (“facing”)

**Recipe D** (BACKBONE multi-layer CAA + W_dec[F] feature amplification):
- BACKBONE: α = 1.0 · unit(v_meanpool[L]) at every L ∈ [4, 6, 9, 11, 12, 13, 14, 15]
- + γ = 7.0 · W_dec[L12, F=2257] at L12 only

**Headline (R(F)∩test, n=64)**: baseline 50.00% → steered 65.62% (Δ +15.62pp; 15 wins, 5 losses).

**Collage**: ![collage](imgs/spatial_steering/collage.png)

## Cherry-picked wins (steered = correct, baseline = incorrect)

### 1. “The cat is facing the person.”  (relation: *facing*  ·  GT: **Yes**)
![win1](docs/imgs/spatial_steering/win_01_vi10569.jpg)

- **Baseline (pt-448)**: → **No** (55% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (57% conf)  ✓

### 2. “The person is facing the banana.”  (relation: *facing*  ·  GT: **Yes**)
![win2](docs/imgs/spatial_steering/win_02_vi9796.jpg)

- **Baseline (pt-448)**: → **No** (51% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (60% conf)  ✓

### 3. “The person is facing the bear.”  (relation: *facing*  ·  GT: **Yes**)
![win3](docs/imgs/spatial_steering/win_03_vi9134.jpg)

- **Baseline (pt-448)**: → **No** (57% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (54% conf)  ✓

### 4. “The cat is facing the laptop.”  (relation: *facing*  ·  GT: **Yes**)
![win4](docs/imgs/spatial_steering/win_04_vi9290.jpg)

- **Baseline (pt-448)**: → **No** (51% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (59% conf)  ✓

### 5. “The cat is facing the refrigerator.”  (relation: *facing*  ·  GT: **Yes**)
![win5](docs/imgs/spatial_steering/win_05_vi9516.jpg)

- **Baseline (pt-448)**: → **No** (60% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (50% conf)  ✓

### 6. “The zebra is facing the person.”  (relation: *facing*  ·  GT: **Yes**)
![win6](docs/imgs/spatial_steering/win_06_vi9152.jpg)

- **Baseline (pt-448)**: → **No** (51% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (59% conf)  ✓

### 7. “The cat is facing the tv.”  (relation: *facing*  ·  GT: **Yes**)
![win7](docs/imgs/spatial_steering/win_07_vi10599.jpg)

- **Baseline (pt-448)**: → **No** (59% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (50% conf)  ✓

### 8. “The dog is facing the boat.”  (relation: *facing*  ·  GT: **Yes**)
![win8](docs/imgs/spatial_steering/win_08_vi9133.jpg)

- **Baseline (pt-448)**: → **No** (54% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (54% conf)  ✓

### 9. “The cat is facing the laptop.”  (relation: *facing*  ·  GT: **Yes**)
![win9](docs/imgs/spatial_steering/win_09_vi10942.jpg)

- **Baseline (pt-448)**: → **No** (55% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (53% conf)  ✓

### 10. “The dog is facing the horse.”  (relation: *facing*  ·  GT: **Yes**)
![win10](docs/imgs/spatial_steering/win_10_vi9855.jpg)

- **Baseline (pt-448)**: → **No** (55% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (53% conf)  ✓

### 11. “The book is facing the cat.”  (relation: *facing*  ·  GT: **Yes**)
![win11](docs/imgs/spatial_steering/win_11_vi9371.jpg)

- **Baseline (pt-448)**: → **No** (54% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (52% conf)  ✓

### 12. “The person is facing the bed.”  (relation: *facing*  ·  GT: **Yes**)
![win12](docs/imgs/spatial_steering/win_12_vi10537.jpg)

- **Baseline (pt-448)**: → **No** (56% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (50% conf)  ✓

### 13. “The person is facing the book.”  (relation: *facing*  ·  GT: **Yes**)
![win13](docs/imgs/spatial_steering/win_13_vi9909.jpg)

- **Baseline (pt-448)**: → **No** (52% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (53% conf)  ✓

### 14. “The teddy bear is facing the person.”  (relation: *facing*  ·  GT: **Yes**)
![win14](docs/imgs/spatial_steering/win_14_vi10682.jpg)

- **Baseline (pt-448)**: → **No** (50% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (55% conf)  ✓

### 15. “The bird is facing the elephant.”  (relation: *facing*  ·  GT: **Yes**)
![win15](docs/imgs/spatial_steering/win_15_vi9979.jpg)

- **Baseline (pt-448)**: → **No** (50% conf)  ❌
- **Steered (Recipe D)**: → **Yes** (55% conf)  ✓
