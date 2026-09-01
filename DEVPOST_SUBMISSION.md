# Devpost Submission

## Project Name

**CamTrace-6M**

CamTrace-6M has **6,002,830 learned detector parameters**. Its complete
inference graph contains **309,969,038 parameters** when the frozen CLIP
ViT-L/14 backbone is included, remaining well below the 2B-parameter limit.

## Elevator Pitch

CamTrace-6M exposes benchmark shortcuts, then fuses semantic and camera-pipeline evidence to detect AI images after compression, blur, cropping, and reposting.

_Character count: 159/200._

## About the Project

### The problem: detectors can learn the benchmark instead of the image

AI-generated imagery creates real risks for trust and safety teams, journalists,
marketplaces, and everyday users. Yet a detector that works only on pristine
files is not enough: images are recompressed by social platforms, resized into
thumbnails, cropped into profile pictures, blurred, filtered, and screenshotted
before anyone needs to verify them.

We discovered an even deeper problem before training our model. A content-blind
probe using only dimensions and file-header metadata achieved **1.0000 ROC AUC**
on 2,000 COCO photographs versus 2,000 DALL·E Advanced images. It could infer
the label perfectly without looking at pixels. After both classes passed
through the same 256×256 crop, JPEG encoder, quality, and chroma subsampling,
that score fell to **0.6527**.

That result reframed the challenge. We were not trying to build the best
detector of PNG versus JPEG. We wanted a detector that identifies evidence in
the image and keeps its decision stable after realistic redistribution.

### Our solution: semantic evidence meets camera-pipeline forensics

**CamTrace-6M** combines two complementary forensic views:

1. **Frozen CLIP attention probe.** OpenAI's CLIP ViT-L/14 QuickGELU vision
   tower provides patch-level content representations. A compact trainable
   attention pooler learns which patches matter for provenance.
2. **Spatial Rich Model branch.** Thirty fixed high-pass forensic kernels run
   over RGB to produce 90 residual maps. A six-block CNN learns acquisition,
   rendering, and resampling traces that ordinary semantic models may ignore.
3. **Degradation-aware fusion.** A 392-parameter MLP estimates blur,
   JPEG-style blockiness, and image scale, then assigns per-image softmax
   weights to the semantic and residual logits.

At inference, CamTrace-6M scores overlapping native-resolution crops under four
views—identity, JPEG 90, 0.5× resize, and 80% crop. A trimmed mean produces the
final calibrated probability, while variance across crops and transformations
becomes a **stability signal** for downstream triage.

### How we built it

We trained on **11,139 SID_Set images**, with 1,221 validation images and a
stricter, class-balanced 904-image validation protocol. Instead of resizing
whole images and destroying forensic evidence, we take deterministic 224×224
crops at native resolution. Both classes pass through a common JPEG encoder to
suppress container, quantisation-table, chroma-subsampling, and resolution
shortcuts. We then match the validation set by fixed-encoder bytes per pixel
and report AUC inside five bpp quantile bins.

For each clean crop \(I\), the trainer samples one or two realistic
transformations to construct \(T(I)\): JPEG compression, Gaussian blur,
downscale/upscale, Gaussian noise, colour jitter, or centre cropping. The
objective encodes robustness directly:

\[
\mathcal{L} = \operatorname{CE}(f(I), y)
+ \operatorname{CE}(f(T(I)), y)
+ \lambda D_{\mathrm{KL}}\!\left(f(I)\,\|\,f(T(I))\right),
\qquad \lambda=1.
\]

The first two terms teach clean and transformed classification. The KL term
penalises a prediction that changes only because an image was reposted or
lightly edited. In the controlled two-branch ablation, consistency training
raised mean transformed AUC from **0.999526 to 0.999812**.

The frozen CLIP tower's clean patch tokens are cached as resumable FP16 shards,
making repeated ablations practical. Every run records its resolved config,
seed, parameter counts, epoch history, precision mode, and wall-clock. Preflight
checks reject held-out-data leakage, missing files, incompatible caches, or an
invalid device before consuming a GPU allocation.

### Measured robustness

The primary result uses the bias-matched SID_Set validation protocol: **904
images, balanced 452/452, eight crops per image**. The operating threshold is
fixed on clean real scores and reused unchanged under every degradation.

| Condition | CamTrace-6M ROC AUC | Accuracy at fixed threshold |
|---|---:|---:|
| Clean | **0.999980** | **99.56%** |
| JPEG quality 30 | 0.999775 | 97.90% |
| Gaussian blur, \(\sigma=2.0\) | 0.999692 | 96.57% |
| Resize to 0.25× then upscale | 0.999667 | 96.68% |
| Gaussian noise, \(\sigma=0.10\) | 0.998492 | 98.23% |
| Colour jitter, worst variant | 0.999863 | 98.89% |
| Centre crop, 80% | 0.999976 | 98.89% |
| **Mean across all 15 transformed conditions** | **0.999760** | **98.35%** |

CamTrace-6M retains **99.8512% of clean AUC** in the hardest tested condition.
Temperature scaling reduced validation NLL from **0.024487 to 0.020112** and
selected a threshold of **0.427470**, achieving **100% TPR at 0.88% measured
FPR** on the balanced calibration set.

The robustness table and deployment calibration use different, intentionally
separated score paths. The table fixes an **uncalibrated clean-crop threshold
of 0.307028** for the gated model. Deployment uses overlapping crops, four TTA
views, and temperature scaling, producing the **calibrated threshold 0.427470**.
Each threshold is reused only with the protocol that produced it.

These results are supported by five trained ablations, raw and bias-matched
robustness grids, fixed-threshold accuracy, bpp-stratified metrics, calibration,
and ranked error analysis committed in the public repository.

### Error analysis and trade-offs

Across 16 clean/transformed conditions, CamTrace-6M produced 214
false-positive condition-cases from 7,232 real cases (**2.96%**) and 14
false-negative condition-cases from 7,232 synthetic cases (**0.19%**).

The ranked contact sheet revealed a useful pattern: false positives concentrate
in already defocused or low-detail real photographs after aggressive blur,
resizing, or JPEG compression. False negatives concentrate in near-white,
minimalist generations and a few generated scenes under strong noise. The same
hard images recur across transformations, suggesting concentrated tails rather
than uniform collapse. This is exactly where the stability sidecar can help a
review workflow prioritise uncertain cases instead of treating every score as
equally reliable.

### Technical execution and feasibility

- **Portable contract:** `predict.py --image_dir ... --device cpu` writes the
  required `[{"image_path": "...", "pred": 0.9731}]` JSON using real released
  weights.
- **Deployment evidence:** CPU inference was validated end-to-end; a detailed
  sidecar adds calibrated decisions, crop/view counts, per-TTA predictions,
  and stability without changing the required JSON schema.
- **Reproducibility:** deterministic splits and crops, leakage tests, cache
  fingerprints, resolved run configs, fixed operating points, and **195 passing
  tests** with 12 expected environment/data-dependent skips.
- **Efficient experimentation:** only 6,002,830 detector parameters are learned;
  the final adaptive gate adds just 392. The released checkpoint is 24.1 MB and
  excludes the separately downloaded frozen CLIP weights.
- **Open evidence:** code, weights, calibration, robustness tables, exact run
  metrics, error counts, and the ranked contact sheet are all public.

### Challenges we overcame

**Shortcut leakage.** SID_Set's real images are JPEG while its generated images
are PNG at a uniform resolution. We built encoder matching, bidirectional split
isolation checks, a content-blind confound probe, and bpp-stratified evaluation
instead of accepting a misleading perfect benchmark score.

**Preserving forensic evidence.** Conventional full-image resizing erases the
high-frequency traces the SRM branch needs. Native-resolution crops preserve
those traces while still supporting fixed-size batching.

**Numerical stability.** The unclamped SRM residuals are heavy-tailed. BF16,
BatchNorm, and gradient clipping kept training stable; older GPUs automatically
use FP32 rather than unsafe FP16. This resolved the NaN failure encountered in
the first full run.

**Compute-aware iteration.** Frozen CLIP is still expensive under live
transformations. A 24.1 GiB resumable token cache accelerates clean passes,
shared-backbone evaluation avoids redundant work across ablations, and Slurm
jobs resolve the repository and node-local temporary storage explicitly.

### What we learned

The most important lesson is methodological: detection quality is not only
whether a label is correct, but **what evidence produced that answer**. A
perfect score can reveal a broken benchmark. We also learned that semantic and
camera-residual branches fail differently, consistency can encode a deployment
requirement directly into the objective, and error analysis belongs in the
model-development loop rather than at the end.

### Impact and what comes next

CamTrace-6M is designed as a decision-support component for trust and safety
teams, journalists, marketplaces, and authenticity tools. Its calibrated
probability supports ranking; its stability score exposes sensitivity to
reposting transformations; and its fixed-FPR operating point makes the cost of
false accusations explicit.

Next, we would extend generator-held-out training with WildFake, test complete
social-media recompression chains, repeat the ablations across multiple seeds,
and distil the CLIP branch for lower edge latency. The architecture and evidence
pipeline are already structured to support those extensions.

## Technology Stack

| Layer | Implementation |
|---|---|
| Development | VS Code, Jupyter, Python 3.11, `uv`, Hatchling |
| Deep learning | PyTorch 2.8.0, torchvision 0.23.0 |
| Vision | OpenCLIP 3.3.0, OpenAI ViT-L/14 QuickGELU, timm 1.0.29 |
| Forensics | Fixed SRM kernel bank and custom PyTorch residual CNN |
| Data and metrics | NumPy 2.4.6, Pillow 12.3.0, scikit-learn 1.9.0, PyArrow |
| Configuration | OmegaConf 2.3.1 and versioned YAML snapshots |
| Compute | NUS SoC Slurm, NVIDIA H100 NVL, CUDA 12.8; TITAN V FP32 calibration |
| Quality | pytest, deterministic tests, leakage and checkpoint-contract tests |

## Datasets and Assets Used

- **SID_Set:** binary detector training and controlled validation.
- **COCO val2017:** real images for the held-out content-blind confound audit;
  never used to train the detector.
- **DALL·E Advanced:** synthetic images for the held-out content-blind
  confound audit; never used to train the detector.
- **OpenAI CLIP ViT-L/14 QuickGELU:** frozen pretrained vision backbone loaded
  through OpenCLIP.

SID_Set is used under CC BY 4.0 with attribution and documented transformations.
COCO and WildFake/DALL·E images are not redistributed. Complete license,
dataset, model, and modification notices are in
[`THIRD_PARTY_NOTICES.md`](https://github.com/mathenaangeles/AIGC-Detector/blob/main/THIRD_PARTY_NOTICES.md).

## Try It Out

- **Public repository and complete README:** https://github.com/mathenaangeles/AIGC-Detector
- **CamTrace-6M weights and calibration:** https://github.com/mathenaangeles/AIGC-Detector/releases/tag/v1.0.0-techjam
- **Executed confound demonstration:** https://github.com/mathenaangeles/AIGC-Detector/blob/main/notebooks/01_confound_demo.ipynb
- **Full robustness table:** https://github.com/mathenaangeles/AIGC-Detector/blob/main/reports/robustness_table.md
- **Ranked error analysis:** https://github.com/mathenaangeles/AIGC-Detector/blob/main/reports/error_analysis.md
- **Public three-minute demo:** [PASTE_PUBLIC_YOUTUBE_URL_BEFORE_SUBMISSION]

## Submission Artifact

Upload `aigc-detector-devpost.zip`:

- Size: **28,877,702 bytes (27.54 MiB)**
- SHA-256: `bc9a2c11bb5ed4d673eebd5f25ac631e8b1f229307426d1cdf7f29bae7d84297`
- Selected model: **CamTrace-6M** (`p8-gated-kl1`)
- Includes source, configuration, reports, executed notebook, checkpoint, and
  calibration; excludes datasets, caches, environments, and frozen CLIP weights.

## Problem Statement

**Robust Detection of AI-Generated Images Under Real-World Transformations**

## Public Code Repository

https://github.com/mathenaangeles/AIGC-Detector
