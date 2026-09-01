# Error Analysis

- Protocol: `bias_matched`
- Split: `val_matched` (904 images)
- Checkpoint: `runs/p8-gated-kl1/model.pt`
- Fixed clean threshold: `0.306789` at `0.88%` FPR
- Conditions: `16`; crops per image: `8`
- Contact sheet: [`error_contact_sheet.png`](error_contact_sheet.png)

## Measured failure modes

- The highest false-positive rate occurs under `resize` (43 errors, 4.76% of real condition-cases).
- The highest false-negative rate occurs under `gaussian_noise` (5 errors, 0.37% of synthetic condition-cases).
- Generator-level attribution is unavailable for this split. SID_Set does not carry generator metadata, so the report marks synthetic generator as `unknown` instead of inventing a grouping.
- Visual review of the ranked sheet shows false positives concentrated in already defocused or low-detail photographs (fabric, horizon, and macro crops) after blur, resizing, or JPEG. These conditions weaken the camera-detail evidence without adding synthetic semantics.
- False negatives concentrate in near-white minimalist generations and a few generated scenes after strong Gaussian noise. The same underlying images recur across several conditions, indicating concentrated image/degradation interactions rather than uniform failure across the class.
- These are associations under controlled degradations, not causal claims about image semantics.

### By transform type

| Group | Evaluated | FP | FN | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| center_crop | 904 | 10 | 0 | 0.0221 | 0.0000 |
| clean | 904 | 4 | 0 | 0.0088 | 0.0000 |
| color_jitter | 1,808 | 20 | 2 | 0.0221 | 0.0022 |
| gaussian_blur | 2,712 | 45 | 2 | 0.0332 | 0.0015 |
| gaussian_noise | 2,712 | 49 | 5 | 0.0361 | 0.0037 |
| jpeg | 3,616 | 43 | 3 | 0.0238 | 0.0017 |
| resize | 1,808 | 43 | 2 | 0.0476 | 0.0022 |

### By source

| Group | Evaluated | FP | FN | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| sid_set | 14,464 | 214 | 14 | 0.0296 | 0.0019 |

### By generator where known

| Group | Evaluated | FP | FN | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| not_applicable | 7,232 | 214 | 0 | 0.0296 | 0.0000 |
| unknown | 7,232 | 0 | 14 | 0.0000 | 0.0019 |

## Highest-confidence errors

| Rank | Error | Condition | Score | Confidence | Source | Generator | Path |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | FP | gaussian_blur_s2 | 0.8914 | 0.5846 | sid_set | not_applicable | `data/train/sid_set/real/5714374aa94e89a1.jpg` |
| 2 | FP | jpeg_q50 | 0.8766 | 0.5698 | sid_set | not_applicable | `data/train/sid_set/real/012a69e95e063457.jpg` |
| 3 | FP | gaussian_noise_s0.02 | 0.8632 | 0.5564 | sid_set | not_applicable | `data/train/sid_set/real/9bc840cf2fa7cfcf.jpg` |
| 4 | FP | resize_s0.25 | 0.8545 | 0.5478 | sid_set | not_applicable | `data/train/sid_set/real/5714374aa94e89a1.jpg` |
| 5 | FP | gaussian_blur_s2 | 0.8522 | 0.5454 | sid_set | not_applicable | `data/train/sid_set/real/012a69e95e063457.jpg` |
| 6 | FP | jpeg_q50 | 0.8311 | 0.5243 | sid_set | not_applicable | `data/train/sid_set/real/9bc840cf2fa7cfcf.jpg` |
| 7 | FP | gaussian_blur_s2 | 0.8241 | 0.5173 | sid_set | not_applicable | `data/train/sid_set/real/9b98ce75f42d8fbe.jpg` |
| 8 | FP | jpeg_q30 | 0.7821 | 0.4753 | sid_set | not_applicable | `data/train/sid_set/real/9bc840cf2fa7cfcf.jpg` |
| 9 | FP | resize_s0.25 | 0.7659 | 0.4592 | sid_set | not_applicable | `data/train/sid_set/real/012a69e95e063457.jpg` |
| 10 | FP | color_jitter_s0.2_lo | 0.7615 | 0.4547 | sid_set | not_applicable | `data/train/sid_set/real/9bc840cf2fa7cfcf.jpg` |
| 11 | FP | gaussian_blur_s2 | 0.7522 | 0.4454 | sid_set | not_applicable | `data/train/sid_set/real/7bd336dd538daa87.jpg` |
| 12 | FP | resize_s0.25 | 0.7485 | 0.4417 | sid_set | not_applicable | `data/train/sid_set/real/af0919f88ad46c0f.jpg` |
| 13 | FN | gaussian_noise_s0.1 | 0.0122 | 0.2946 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
| 14 | FN | gaussian_noise_s0.05 | 0.1025 | 0.2042 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
| 15 | FN | color_jitter_s0.2_hi | 0.1050 | 0.2017 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
| 16 | FN | jpeg_q50 | 0.1662 | 0.1406 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
| 17 | FN | gaussian_noise_s0.05 | 0.1882 | 0.1186 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_061964.png` |
| 18 | FN | color_jitter_s0.2_lo | 0.1896 | 0.1172 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
| 19 | FN | gaussian_noise_s0.1 | 0.1900 | 0.1168 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_061964.png` |
| 20 | FN | gaussian_noise_s0.1 | 0.1906 | 0.1162 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_033353.png` |
| 21 | FN | jpeg_q30 | 0.1944 | 0.1123 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
| 22 | FN | gaussian_blur_s2 | 0.2162 | 0.0906 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
| 23 | FN | resize_s0.5 | 0.2584 | 0.0484 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
| 24 | FN | gaussian_blur_s1 | 0.2604 | 0.0464 | sid_set | unknown | `data/train/sid_set/synthetic/full_synthetic_043405.png` |
