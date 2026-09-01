# Devpost Submission

## Project Name

**AIGC Detector**

## Description

AIGC Detector fuses frozen CLIP semantics with camera-pipeline residuals to detect AI-generated images after JPEG compression, blur, resizing, noise, colour changes, and cropping.

## About the project

### Inspiration

AI-generated image detectors often look impressive on a clean benchmark and fail as soon as an image is shared through a messaging app, resized for a website, or saved as a JPEG. We found an even more basic problem: a benchmark can accidentally reveal its labels through file format and encoder settings. A detector may appear to recognize generated imagery while actually learning shortcuts such as “PNG means synthetic” or a particular JPEG quantization table.

We tested that hypothesis before training the model. A content-blind logistic-regression probe, using only dimensions and file-header metadata, achieved **1.0000 ROC AUC** on 2,000 COCO photographs and 2,000 DALL·E Advanced images. After both classes were converted to the same 256×256 crop, JPEG quality, and chroma subsampling, its AUC fell to **0.6527**. On our SID_Set training data, the same probe fell from **1.0000** to **0.7249** after matching. Those results changed our goal: we did not want to build the best detector of dataset packaging. We wanted a detector whose score survives when those shortcuts are removed and when real-world transformations damage the pixels.

### What it does

AIGC Detector returns a calibrated probability that an image is AI-generated. It combines two complementary views:

1. A frozen OpenAI CLIP ViT-L/14 vision encoder provides broad semantic features that can generalize beyond generators seen during training. A small trainable attention probe reads its patch tokens.
2. A fixed 30-filter Spatial Rich Model (SRM) bank exposes high-frequency residuals associated with acquisition and rendering pipelines. A compact CNN learns spatial statistics from the resulting 90 residual maps.

A degradation-aware gate combines the branch predictions according to the evidence available in each image. For example, aggressive JPEG compression or blur can weaken residual evidence, so the model can rely more heavily on the semantic branch. The final CPU inference path uses overlapping native-resolution crops, transformation-time augmentation, temperature calibration, and crop-level aggregation to produce one probability per image.

### How we built it

We begin with native-resolution images and take deterministic 224×224 crops rather than resizing the whole frame. Both classes pass through a common JPEG encoder during training and validation to suppress container, subsampling, dimension, and quantization-table shortcuts. We also create a caliper-matched validation subset based on fixed-encoder bytes per pixel (bpp), which gives us a stricter model-selection set.

For every clean crop $I$, we sample one or two transformations to construct $T(I)$. Training covers JPEG compression, Gaussian blur, downscale/upscale, Gaussian noise, brightness/contrast/saturation changes, and centre cropping. The objective is

$$
\mathcal{L} = \operatorname{CE}(f(I), y)
+ \operatorname{CE}(f(T(I)), y)
+ \lambda\,D_{\mathrm{KL}}\!\left(f(I)\,\|\,f(T(I))\right),
$$

with $\lambda=1.0$ in our default configuration. The cross-entropy terms teach classification on clean and degraded images; the consistency term discourages the prediction from changing merely because the image was post-processed.

The CLIP tower is frozen, and clean patch tokens are cached as resumable FP16 shards. This makes repeated experiments practical. The deployed detector contains **6,002,830 learned parameters** across its probes, fusion, and gate, and **309,969,038 total parameters** including the frozen CLIP encoder—well below the challenge limit. In the final gate-training phase, only 392 parameters are optimized. We train with AdamW, automatic mixed precision, gradient clipping, deterministic seed 1337, and early stopping on bias-matched validation AUC.

Evaluation is deliberately shortcut-aware. Alongside ordinary ROC AUC, we report:

- a content-blind bpp-only baseline;
- AUC on a bpp-matched validation set;
- mean AUC computed within five bpp quantile bins;
- clean-to-degraded AUC change at every transformation strength;
- a fixed operating point selected at 1% false-positive rate; and
- branch ablations for CLIP, SRM, and fused models.

The matched set makes the direction-normalized bpp-only baseline nearly random: **0.549840 ROC AUC** across 904 images. In a separate SRM front-end diagnostic, disabling the conventional truncated-linear-unit clamp reduced bpp predictability from $R^2=0.977$ to $R^2=0.492$ while improving standalone matched AUC from **0.8080** to **0.8183**. This counterintuitive result taught us that a standard forensic design choice can amplify exactly the shortcut we are trying to avoid.

### Results

These values come from the committed run metrics, bias-matched robustness grid, and calibration report for `p8-gated-kl1`. The primary controlled evaluation contains 904 images, balanced at 452 real and 452 synthetic images, and applies one common JPEG pipeline before scoring.

| Evaluation | Result |
|---|---:|
| Validation images | 1,221 |
| Bias-matched validation images | 904 |
| Best gate epoch | 0 of 3 epochs run |
| Clean validation ROC AUC during gate training | 0.999329 |
| Bias-matched validation ROC AUC during gate training | 0.999011 |
| Five-bin bpp-stratified matched AUC | 0.998605 |
| Clean AUC in the eight-crop robustness grid | 0.999980 |
| Mean ROC AUC across all 15 degraded conditions | 0.999760 |
| Worst-case degraded ROC AUC | 0.998492 (`gaussian_noise_s0.1`) |
| Largest drop from clean AUC | 0.001488 |
| TPR at the calibrated operating point | 1.0000 |
| Calibrated threshold | 0.427470 at 0.88% observed FPR |
| Learned detector parameters | 6,002,830; 392 optimized in the gate phase |
| Total parameters including frozen CLIP | 309,969,038 |
| Gate-training time | 41m 57s on one H100 NVL |
| Two-branch initialization training time | 2h 33m 13s on one H100 NVL |

The main comparison we care about is not only whether **0.999011** matched validation AUC is high, but whether it clears the matched bpp-only baseline of **0.549840** and remains stable under the complete degradation grid. The final model retains **99.8512%** of its clean AUC in the worst tested condition. These are single-seed, same-source SID_Set validation results, not evidence of unseen-generator generalization; the repository states that limitation explicitly.

### Challenges we ran into

**Shortcut leakage.** SID_Set's real samples are JPEG while its generated samples are PNG at a uniform resolution. A naïve split makes perfect classification possible without meaningful image understanding. We addressed this with encoder matching, strict train/evaluation path isolation, bpp controls, and matched validation.

**Preserving forensic evidence.** The SRM branch depends on high-frequency structure, but common preprocessing can erase it before the model sees it. We therefore crop at native resolution and never resize the whole image as a preprocessing shortcut.

**Robustness versus sensitivity.** Residual evidence is useful on clean images but fragile under compression and blur. Consistency training and per-image gated fusion let semantic and forensic evidence complement one another instead of forcing either branch to work in every condition.

**Compute and reproducibility.** A frozen ViT-L/14 is still expensive to run twice for every clean/transformed pair. A compatible, resumable token cache accelerates the clean path, while run snapshots record configuration, seed, parameter counts, timing, and metrics. Preflight checks reject missing images, incomplete caches, incompatible crop settings, or held-out data leakage before an expensive job starts.

**Honest evaluation.** Overall AUC can still benefit from compressibility differences after encoder matching. We learned to treat the bpp-only score as a floor and the gap between overall and bpp-stratified AUC as a warning signal, rather than assuming that preprocessing had removed every confound.

### What we learned

The biggest lesson was methodological: detector evaluation must ask what evidence the model used, not just whether its label was correct. A perfect score can be a symptom of a broken benchmark. We also learned that semantic and residual branches fail differently, that transformation consistency can encode an important deployment requirement directly into the loss, and that reproducibility mechanisms—data isolation, deterministic crops, cache metadata, fixed operating points, and explicit ablations—are part of the model rather than administrative extras.

### What's next

We plan to broaden generator-held-out evaluation, add compound social-media degradation chains, validate calibration across different image domains, and extend the output with uncertainty and interpretable branch-reliability indicators. We also want to study camera-model and editing-pipeline shifts so that “real” remains representative of the images encountered outside a curated benchmark.

## Built with

Use these as DevPost tags (16 total):

- Python
- PyTorch
- OpenCLIP
- CLIP
- Vision Transformer
- Convolutional Neural Networks
- scikit-learn
- NumPy
- Pillow
- OmegaConf
- Hugging Face
- CUDA
- Slurm
- Jupyter
- pytest
- uv

## Try it out links

- **Public code and README:** https://github.com/mathenaangeles/AIGC-Detector
- **Executed confound demonstration:** https://github.com/mathenaangeles/AIGC-Detector/blob/main/notebooks/01_confound_demo.ipynb
- **Downloadable model and calibration:** https://github.com/mathenaangeles/AIGC-Detector/releases/tag/v1.0.0-techjam

## Upload a File

Upload **`aigc-detector-devpost.zip`** (maximum 35 MB), containing:

```text
aigc-detector-devpost/
├── README.md
├── predict.py
├── configs/default.yaml
├── src/provenance/
├── reports/
│   ├── robustness_table.md/json
│   ├── robustness_table_raw.md/json
│   ├── calibration.md
│   ├── error_analysis.md/json
│   ├── error_contact_sheet.png
│   └── run_metrics/
├── notebooks/01_confound_demo.ipynb
├── model.pt
└── calibration.json
```

Exclude datasets, the CLIP feature cache, raw run directories, environment files, and the frozen CLIP backbone weights. The checkpoint and matching calibration JSON are also published as versioned GitHub Release assets.

Final archive verification:

- Archive size: `23,739,719 bytes` (`22.64 MiB`)
- SHA-256: `9fd5624015f02a310cc9c3d38751698a32c748b9c8c3588d9ed480da9a6ea888`
- Selected checkpoint/run: `p8-gated-kl1`

## Which Problem Statement did your team choose?

**AI-Generated Image Detection**

## Link to your project's public code repository with README

https://github.com/mathenaangeles/AIGC-Detector

## Submission verification

- Metrics come from the selected full run and generated reports, not a smoke test.
- The grid contains one clean and 15 degraded conditions.
- Gated fusion, calibrated inference, and CPU execution are implemented and tested.
- The code repository is public.
- The downloadable archive is below the 35 MB limit and excludes data and caches.
