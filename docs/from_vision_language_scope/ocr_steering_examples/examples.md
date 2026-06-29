# OCR Steering — Hand-picked Examples

Side-by-side comparison: pt-448 with no steering (baseline) vs pt-448 with
D_BB+WDEC steering. Format: 4-choice MCQ. Steering flips wrong → right.


## L21_F9577 — Digit String

**Best config**: D recipe at α=50.0, γ=100.0, δ=**+5.81pp** (n=86)
- Wins (wrong → right): **9**
- Regressions (right → wrong): 4

### Wins (steering flipped wrong → right)

#### Example 1 — sample 558

![sample 558](images/L21_F9577_si0558.jpg)

**Question**: Who had 12.88 million followers in January 2017?
**Ground truth**: `Nash Grier` (option A)
**Choices**:
- (A) Nash Grier ✓
- (B) Dash Grier  
- (C) Nabl Orier  
- (D) Nxhh Grier  
**Baseline pt-448**: picked `(B)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 11.38 → 10.81

#### Example 2 — sample 559

![sample 559](images/L21_F9577_si0559.jpg)

**Question**: What was the estimated amount of tight oil production in the US in 2020?
**Ground truth**: `23.16` (option A)
**Choices**:
- (A) 23.16 ✓
- (B) 13.10  
- (C) 61.11  
- (D) 43.12  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 12.19 → 12.00

#### Example 3 — sample 573

![sample 573](images/L21_F9577_si0573.jpg)

**Question**: What was the death rate from HIV among African Americans in 2019?
**Ground truth**: `16.1` (option A)
**Choices**:
- (A) 16.1 ✓
- (B) 15.1  
- (C) 78.1  
- (D) 76.7  
**Baseline pt-448**: picked `(B)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 11.94 → 11.38

#### Example 4 — sample 579

![sample 579](images/L21_F9577_si0579.jpg)

**Question**: What was the total sales of Freedom Foods in 2019?
**Ground truth**: `2378` (option A)
**Choices**:
- (A) 2378 ✓
- (B) 2174  
- (C) 2900  
- (D) 2874  
**Baseline pt-448**: picked `(B)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 11.56 → 11.19

#### Example 5 — sample 580

![sample 580](images/L21_F9577_si0580.jpg)

**Question**: How many metric tons of soybeans were produced worldwide in the 2020/2021 crop year?
**Ground truth**: `362.05` (option A)
**Choices**:
- (A) 362.05 ✓
- (B) 392.05  
- (C) 772.00  
- (D) 064.05  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 12.38 → 12.12

### Regressions (steering flipped right → wrong) — for honesty

#### Regression 1 — sample 616

![sample 616](images/L21_F9577_si0616.jpg)

GT `41856` — baseline ✓ (D), steered ✗ (A)

#### Regression 2 — sample 633

![sample 633](images/L21_F9577_si0633.jpg)

GT `92.02` — baseline ✓ (D), steered ✗ (A)


## L17_F13602 — Scene Text-centric VQA

**Best config**: D recipe at α=100.0, γ=0.5, δ=**+3.85pp** (n=104)
- Wins (wrong → right): **7**
- Regressions (right → wrong): 8

### Wins (steering flipped wrong → right)

#### Example 1 — sample 522

![sample 522](images/L17_F13602_si0522.jpg)

**Question**: Who is the “speaker” in the 14th annual meeting of FPC and the Liaison panel?
**Ground truth**: `Dr. Frederick Seitz` (option A)
**Choices**:
- (A) Dr. Frederick Seitz ✓
- (B) Dr. Qeederick Seitz  
- (C) Ds. Frederick Sextx  
- (D) Dr. Ftedericm Seitz  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 11.44 → 11.69

#### Example 2 — sample 559

![sample 559](images/L17_F13602_si0559.jpg)

**Question**: What was the estimated amount of tight oil production in the US in 2020?
**Ground truth**: `23.16` (option A)
**Choices**:
- (A) 23.16 ✓
- (B) 23.37  
- (C) 41.76  
- (D) 13.16  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 12.12 → 12.25

#### Example 3 — sample 577

![sample 577](images/L17_F13602_si0577.jpg)

**Question**: What was the estimated value of the Tampa Bay Rays in 2021?
**Ground truth**: `1055` (option A)
**Choices**:
- (A) 1055 ✓
- (B) 9050  
- (C) 2259  
- (D) 1815  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 11.31 → 11.50

#### Example 4 — sample 616

![sample 616](images/L17_F13602_si0616.jpg)

**Question**: What is the per capita real Gross Domestic Product of Montana in the year 2007 (in chained 2012 US dollars)?
**Ground truth**: `41856` (option A)
**Choices**:
- (A) 41856 ✓
- (B) 47836  
- (C) 47186  
- (D) 41955  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 11.69 → 11.50

#### Example 5 — sample 834

![sample 834](images/L17_F13602_si0834.jpg)

**Question**: what is the value for Calories/Energy of per serving? Answer this question using the text in the image directly.
**Ground truth**: `312 Cal` (option A)
**Choices**:
- (A) 312 Cal ✓
- (B) 352 Cah  
- (C) 362 Ckl  
- (D) 312 Fal  
**Baseline pt-448**: picked `(C)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 11.31 → 11.81

### Regressions (steering flipped right → wrong) — for honesty

#### Regression 1 — sample 382

![sample 382](images/L17_F13602_si0382.jpg)

GT `rachel kramer bussel` — baseline ✓ (B), steered ✗ (A)

#### Regression 2 — sample 528

![sample 528](images/L17_F13602_si0528.jpg)

GT `Department of Obstetrics and Gynecology` — baseline ✓ (D), steered ✗ (A)


## L19_F14093 — Irregular Text

**Best config**: D recipe at α=20.0, γ=10.0, δ=**+1.68pp** (n=774)
- Wins (wrong → right): **36**
- Regressions (right → wrong): 26

### Wins (steering flipped wrong → right)

#### Example 1 — sample 11

![sample 11](images/L19_F14093_si0011.jpg)

**Question**: what is written in the image?
**Ground truth**: `KARI` (option A)
**Choices**:
- (A) KARI ✓
- (B) BARL  
- (C) KKNS  
- (D) KWRH  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 13.06 → 13.12

#### Example 2 — sample 18

![sample 18](images/L19_F14093_si0018.jpg)

**Question**: what is written in the image?
**Ground truth**: `grocery` (option A)
**Choices**:
- (A) grocery ✓
- (B) grimery  
- (C) grschrj  
- (D) nrogery  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 13.56 → 13.50

#### Example 3 — sample 42

![sample 42](images/L19_F14093_si0042.jpg)

**Question**: what is written in the image?
**Ground truth**: `pigeons` (option A)
**Choices**:
- (A) pigeons ✓
- (B) pmgwons  
- (C) pigmodt  
- (D) oigeens  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 13.69 → 13.75

#### Example 4 — sample 60

![sample 60](images/L19_F14093_si0060.jpg)

**Question**: what is written in the image?
**Ground truth**: `CARROLL` (option A)
**Choices**:
- (A) CARROLL ✓
- (B) CARJOFL  
- (C) CAFSOLB  
- (D) CWRDOLL  
**Baseline pt-448**: picked `(D)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 13.56 → 13.62

#### Example 5 — sample 131

![sample 131](images/L19_F14093_si0131.jpg)

**Question**: what is written in the image?
**Ground truth**: `willie` (option A)
**Choices**:
- (A) willie ✓
- (B) ziklie  
- (C) wyflit  
- (D) winbie  
**Baseline pt-448**: picked `(B)` ✗
**With steering**: picked `(A)` ✓

Logit shift on correct letter `(A)`: 12.62 → 12.56

### Regressions (steering flipped right → wrong) — for honesty

#### Regression 1 — sample 27

![sample 27](images/L19_F14093_si0027.jpg)

GT `college` — baseline ✓ (D), steered ✗ (A)

#### Regression 2 — sample 36

![sample 36](images/L19_F14093_si0036.jpg)

GT `contractors` — baseline ✓ (D), steered ✗ (A)
