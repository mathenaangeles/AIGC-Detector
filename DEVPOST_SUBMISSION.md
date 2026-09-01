# Devpost Submission

## Project Name

**AIGC Detector**

## Official Model Name

**CamTrace-6M**

The `6M` denotes **6,002,830 learned detector parameters**. The full inference
graph contains **309,969,038 parameters** once the frozen CLIP ViT-L/14
backbone is counted.

## Description

**CamTrace-6M** powers AIGC Detector by fusing a frozen CLIP attention probe with camera-pipeline residual evidence. On a controlled 904-image validation split it achieved **0.999760 mean ROC AUC across 15 degraded conditions**, with a calibrated threshold of **0.427470** at **0.88% observed FPR**.

## About the project

### Inspiration

AI-generated image detectors often look impressive on a clean benchmark and fail as soon as an image is shared through a messaging app, resized for a website, or saved as a JPEG. We found an even more basic problem: a benchmark can accidentally reveal its labels through file format and encoder settings. A detector may appear to recognize generated imagery while actually learning shortcuts such as “PNG means synthetic” or a particular JPEG quantization table.

We tested that hypothesis before training the model. A content-blind logistic-regression probe, using only dimensions and file-header metadata, achieved **1.0000 ROC AUC** on 2,000 COCO photographs and 2,000 DALL·E Advanced images. After both classes were converted to the same 256×256 crop, JPEG quality, and chroma subsampling, its AUC fell to **0.6527**. On our SID_Set training data, the same probe fell from **1.0000** to **0.7249** after matching. Those results changed our goal: we did not want to build the best detector of dataset packaging. We wanted a detector whose score survives when those shortcuts are removed and when real-world transformations damage the pixels.

### What it does

CamTrace-6M returns a calibrated probability that an image is AI-generated. It combines two complementary views:

1. A frozen OpenAI CLIP ViT-L/14 vision encoder provides broad content representations. A small trainable attention probe reads its patch tokens.
2. A fixed 30-filter Spatial Rich Model (SRM) bank exposes high-frequency residuals associated with acquisition and rendering pipelines. A compact CNN learns spatial statistics from the resulting 90 residual maps.

A degradation-aware gate combines the branch predictions according to the evidence available in each image. For example, aggressive JPEG compression or blur can weaken residual evidence, so the model can rely more heavily on the semantic branch. The final CPU inference path uses overlapping native-resolution crops, transformation-time augmentation, temperature calibration, and crop-level aggregation to produce one probability per image.

### How we built it

We begin with native-resolution images and take deterministic 224×224 crops rather than resizing the whole frame. Both classes pass through a common JPEG encoder during training and validation to suppress container, subsampling, dimension, and quantization-table shortcuts. We also create a caliper-matched validation subset based on fixed-encoder bytes per pixel (bpp), which gives us a stricter model-selection set.

For every clean crop **I**, we sample one or two transformations to construct **T(I)**. Training covers JPEG compression, Gaussian blur, downscale/upscale, Gaussian noise, brightness/contrast/saturation changes, and centre cropping. The objective is:

**L = CE(f(I), y) + CE(f(T(I)), y) + λ D_KL(f(I) ‖ f(T(I)))**

with **λ = 1.0** in our default configuration. The cross-entropy terms teach classification on clean and degraded images; the consistency term discourages the prediction from changing merely because the image was post-processed.

The CLIP tower is frozen, and clean patch tokens are cached as resumable FP16 shards. The deployed detector contains **6,002,830 learned parameters** across the attention probe, SRM CNN, fusion weights, and gate, plus the frozen CLIP tower for **309,969,038 total parameters**. Only the gate's **392 parameters** were optimized in the final P8 phase. We train with AdamW, automatic mixed precision, gradient clipping, deterministic seed 1337, and early stopping on bias-matched validation AUC.

Evaluation is deliberately shortcut-aware. Alongside ordinary ROC AUC, we report:

- a content-blind bpp-only baseline;
- AUC on a bpp-matched validation set;
- mean AUC computed within five bpp quantile bins;
- clean-to-degraded AUC change at every transformation strength;
- a fixed operating point selected at 1% false-positive rate; and
- branch ablations for CLIP, SRM, and fused models.

The matched set makes the direction-normalized bpp-only baseline nearly random: **0.549840 ROC AUC** across 904 images, recomputed directly from the committed matched manifest. In a separate SRM front-end diagnostic, disabling the conventional truncated-linear-unit clamp reduced bpp predictability from **R² = 0.977** to **R² = 0.492** while improving standalone matched AUC from **0.8080** to **0.8183**. This counterintuitive result taught us that a standard forensic design choice can amplify exactly the shortcut we are trying to avoid.

### Results

These are the measured values for the released CamTrace-6M checkpoint (`p8-gated-kl1`). The primary controlled protocol contains 904 images (452 real and 452 synthetic), uses eight crops per image, and passes both classes through the same JPEG pipeline before scoring. Values marked “not measured” were not produced by the completed experiment and are stated explicitly instead of being inferred or fabricated.

| Evaluation | Result |
|---|---:|
| Validation images | 1,221 |
| Bias-matched validation images | 904 (452 real, 452 synthetic) |
| Best epoch | Gate epoch 0 of 3 run |
| Clean validation ROC AUC | 0.999329 during gate training; 0.999980 in the eight-crop grid |
| Bias-matched validation ROC AUC | 0.999011 during gate training |
| Five-bin bpp-stratified mean AUC | 0.998605 on matched validation |
| Held-out COCO vs. DALL·E Advanced detector AUC | Not measured; these images were used only for the content-blind audit (1.0000 raw, 0.6527 matched) |
| Mean ROC AUC across all 15 degraded conditions | 0.999760 |
| Worst-case degraded ROC AUC | 0.998492 (`gaussian_noise_s0.1`) |
| Largest drop from clean AUC | 0.001488 |
| TPR at the 1% target-FPR operating point | 1.0000 TPR at 0.88% observed FPR (4/452 false positives) |
| Calibrated threshold at 1% target FPR | 0.427470 |
| Expected calibration error | Not measured; temperature scaling reduced NLL from 0.024487 to 0.020112 |
| CPU latency per image | 8,948 ms, warmed, batch 1, eight crops × four TTA views, Apple M5 Pro CPU |
| Learned detector parameters | 6,002,830; only 392 optimized in the final gate phase |
| Total parameters | 309,969,038 including frozen CLIP |
| Training time | 2h 33m 13s two-branch initialization + 41m 57s gate training on one H100 NVL |

CPU latency was measured after one warm-up over one real and one synthetic
SID_Set image: `17.896` seconds total, or `8.948` seconds per image. The test
used batch size 1, eight crops, four TTA views, eight OpenMP threads, and an
18-core Apple M5 Pro MacBook Pro with 24 GB RAM on macOS 26.4. Model loading was
excluded; image decoding, bias matching, both branches, gating, calibration,
and aggregation were included.

The final model clears the matched bpp-only baseline of **0.549840** and retains **99.8512%** of its clean AUC in the worst tested condition. These are single-seed, same-source SID_Set validation results, not evidence of unseen-generator generalization. Full tables, fixed-threshold accuracy, raw-protocol diagnostics, and ranked errors are committed under [`reports/`](reports/).

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

## Technology stack

| Layer | Actual implementation |
|---|---|
| Language and environment | Python 3.11, `uv`, Hatchling |
| Deep learning | PyTorch 2.8.0, torchvision 0.23.0 |
| Vision backbone | OpenCLIP 3.3.0, OpenAI `ViT-L-14-quickgelu` weights, timm 1.0.29 |
| Forensic branch | Fixed 30-kernel SRM bank implemented in PyTorch plus a six-block residual CNN |
| Fusion and calibration | 392-parameter PyTorch degradation gate; scikit-learn 1.9.0 metrics; temperature scaling implemented in `src/provenance/calibrate.py` |
| Image and numerical processing | Pillow 12.3.0, NumPy 2.4.6 |
| Configuration | OmegaConf 2.3.1 and versioned YAML run snapshots |
| Training infrastructure | NUS SoC Slurm cluster; NVIDIA H100 NVL for primary training/evaluation; CUDA 12.8 PyTorch build; TITAN V FP32 calibration validation |
| Testing and analysis | pytest (195 passed, 12 environment/data-dependent skips), Jupyter, PyArrow, Hugging Face Hub |
| Deployment interface | `predict.py` on CPU, overlapping crops, four-view TTA, trimmed-mean aggregation, JSON output plus a stability sidecar |

### Devpost tags

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

The archive excludes datasets, the 24.1 GiB CLIP feature cache, raw run directories, environment files, and the frozen CLIP backbone weights. The checkpoint and its cryptographically bound calibration JSON are also available separately in the GitHub Release.

Final archive verification:

- Archive size: `23,746,093 bytes` (`22.65 MiB`)
- SHA-256: `a8a9c0863aeff567c803f91a78ac52b7f454b6a1ab23e0bd907b182b9ffd8643`
- Selected checkpoint/run: `p8-gated-kl1`

## Which Problem Statement did your team choose?

**AI-Generated Image Detection**

## Link to your project's public code repository with README

https://github.com/mathenaangeles/AIGC-Detector

## Submission verification

- Metrics come from the selected full run and generated reports, not a smoke test.
- The grid contains one clean and 15 degraded conditions.
- Gated fusion, calibrated inference, and CPU execution are implemented and tested.
- The code repository and model release are public.
- The downloadable archive is below the 35 MB limit and excludes data and caches.
