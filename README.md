# AIGC Detector

Camera-pipeline provenance detection for AI-generated images, designed to remain useful after JPEG compression, blur, resizing, noise, colour changes, and cropping.

AIGC Detector starts from a simple problem with many AIGC benchmarks: a detector can score well by learning how the datasets were saved instead of whether an image was generated. The project therefore combines three things:

1. a content-blind probe that measures encoder and file-format shortcuts;
2. a bias-matched evaluation protocol that removes those shortcuts; and
3. a detector built around camera-pipeline evidence, trained with transformation consistency.

> **Current status: The confound probe, data isolation, native-resolution crop pipeline, frozen CLIP branch, SRM branch, feature cache, and joint training loop are implemented and tested. Degradation-aware fusion, the complete robustness-grid runner, and trained-weight inference are TBD.

## Project Contribution

The held-out demonstration set pairs COCO val2017 photographs with DALL·E Advanced images. Their encoding pipelines are visibly different even without decoding pixels: resolution, chroma subsampling, JPEG quantisation tables, and container formats nearly identify the class by themselves.

The repository's content-blind logistic-regression probe measured:

| Protocol | Images | ROC AUC |
|---|---:|---:|
| Raw COCO vs. DALL·E Advanced | 4,000 | **1.0000** |
| Fixed JPEG encoder + 256×256 native-resolution crops | 4,000 | **0.6527** |

The raw benchmark can therefore be solved perfectly without looking at image content. After matching encoder, subsampling, and dimensions, only bytes-per-pixel varies; the remaining AUC mostly measures the different compressibility of the image content.

The full, executed exhibit is in [`notebooks/01_confound_demo.ipynb`](notebooks/01_confound_demo.ipynb).

## Architecture

```text
native-resolution image
        │
        ├── deterministic random crop I ──────────────────────────────┐
        │                                                             │
        └── same crop + sampled degradation T ──> T(I)                │
                                                                      │
             ┌──────────────────────────┬──────────────────────────┐   │
             │ Frozen CLIP ViT-L/14     │ Fixed 30-filter SRM bank │   │
             │ patch-token encoder      │ + residual CNN           │   │
             │          ↓               │            ↓             │   │
             │ trainable attention      │       branch logits      │   │
             │ probe → branch logits    │                          │   │
             └──────────────┬───────────┴─────────────┬────────────┘   │
                            └── learned softmax fusion ┘                │
                                         ↓                             │
                                  P(AI-generated)                       │
                                                                       │
 loss = CE(f(I), y) + CE(f(T(I)), y) + λ KL(f(I) || f(T(I))) <────────┘
```

### Frozen CLIP branch

The OpenAI CLIP ViT-L/14 QuickGELU vision tower stays in evaluation mode with gradients disabled. A trainable attention-pooling probe reads its patch tokens and produces two-class logits. Clean-crop tokens can be cached as sharded FP16 arrays, so repeated experiments train the heads without rerunning the frozen tower.

Freshly transformed views cannot use the clean cache and pass through the frozen tower live during consistency training.

### SRM residual branch

Thirty fixed Spatial Rich Model high-pass kernels run depthwise over RGB, producing 90 residual maps. A small fully convolutional CNN reads their spatial statistics and pools the final feature map using both mean and standard deviation.

The traditional truncated-linear-unit threshold is implemented but disabled by default. On the bpp-matched validation subset, disabling it reduced bpp predictability from R² `0.977` to `0.492` and improved the measured standalone AUC from `0.8080` to `0.8183`. Batch normalization and gradient clipping handle the resulting heavy-tailed residuals.

### Fusion and objective

Selected branches train together. P7 currently combines their logits with learned global softmax weights. P8 will replace this with a per-image degradation-aware gate conditioned on blur, JPEG, and image-size estimates.

One degradation specification is sampled per batch and replayed at the same strength for every crop. The consistency KL is optimized jointly with clean and transformed cross-entropy:

```text
L = CE(f(I), y) + CE(f(T(I)), y) + λ · KL(f(I) || f(T(I)))
```

`λ` is configured by `train.lambda_consistency` and defaults to `1.0`.

### Parameter accounting

| Component | Parameters | Trainable now? |
|---|---:|:---:|
| CLIP ViT-L/14 vision encoder | ~303.9M | No |
| Attention-probe head | 1,447,682 | Yes |
| SRM residual CNN | 4,554,754 | Yes |
| Two-branch fusion weights | 2 | Yes |
| **Current two-branch total** | **~309.9M** | **~6.0M** |

The model is comfortably below the competition's 2B-parameter limit. The optional spectral head and degradation-aware gate are not included in the current count because they are not yet active.

## Repository layout

```text
configs/default.yaml                   Main experiment configuration
src/provenance/data.py                 Manifests, split isolation, crops, bpp matching
src/provenance/transforms.py           Six train/evaluation degradations
src/provenance/shortcut.py             Content-blind confound probe
src/provenance/branches/clip_probe.py  Frozen CLIP, attention probe, token cache
src/provenance/branches/srm.py         Fixed SRM bank and residual CNN
src/provenance/train.py                P7 joint consistency trainer
src/provenance/evaluate.py             AUC, bpp baseline, and stratified metrics
src/provenance/fuse.py                 P8 placeholder
src/provenance/calibrate.py            P10 placeholder
scripts/                               Data, manifest, cache, and cluster helpers
scripts/preflight_train.py             Data, device, disk, and cache validation
notebooks/01_confound_demo.ipynb       Executed confound demonstration
predict.py                             CPU-safe submission interface; currently a stub
tests/                                 Unit, leakage, and end-to-end smoke tests
```

## Requirements

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- Enough local disk for the selected datasets and feature cache
- A CUDA GPU for practical CLIP feature caching and full training
- CPU-only execution for tests, SRM smoke runs, and the current `predict.py` interface

Install the locked environment:

```bash
git clone <your-repository-url>
cd "AIGC Detector"
uv sync --all-groups
```

The first CLIP run downloads the configured OpenAI ViT-L/14 weights. Dataset and model caches are deliberately excluded from Git.

## Data setup

The default configuration expects this layout:

```text
data/
├── train/
│   ├── sid_set/
│   │   ├── real/
│   │   ├── synthetic/
│   │   └── tampered/
│   └── wildfake/                    optional until extracted
└── eval/                            strictly held out
    ├── coco_val2017/
    └── dalle_advanced/
```

`data/eval/` is never used for model fitting. Manifest construction checks both directions of this invariant and fails if an eval path receives a train/val split or an external path claims to be eval.

### Minimal training data: SID_Set

Download the configured SID_Set shard subset and extract the original encoded bytes:

```bash
scripts/download_data.sh sid
uv run python scripts/extract_sid_set.py
```

Tampered examples are retained in the manifest but excluded from binary training by default. They passed through a real camera pipeline, so treating them as ordinary synthetic negatives would contradict the camera-pipeline objective.

### Held-out confound demonstration data

Download and extract COCO val2017:

```bash
scripts/download_data.sh coco
mkdir -p data/eval/coco_val2017
unzip -q data/raw/val2017.zip -d data/eval/coco_val2017
```

Fetch only the DALL·E Advanced members required by the demonstration instead of downloading the entire 25.6 GB archive:

```bash
uv run python scripts/extract_dalle_advanced.py
```

WildFake is optional at P7. Its configured generator-aware split becomes active once its `Real/` and `Other_based/` directory structure is extracted under `data/train/wildfake/`. SID_Set contains no generator metadata, so the code deliberately refuses to invent a generator holdout for it.

### Build manifests

```bash
uv run python scripts/build_manifest.py
```

This writes:

- `data/manifest.csv` with deterministic train, val, and eval assignments;
- `data/manifest_val_matched.csv`, a caliper-matched validation subset based on fixed-quality crop-level JPEG bpp.

Use `--no-val-matched` to skip the additional decode pass during a quick manifest rebuild.

## Run the confound probe

Command line:

```bash
uv run python -m provenance.shortcut
```

For a faster check:

```bash
uv run python -m provenance.shortcut --max_per_class 100
```

The command prints the raw and bias-matched ROC AUC rows and writes `shortcut.json` under a timestamped run directory. To reproduce the narrative analysis, open and run:

```bash
uv run jupyter lab notebooks/01_confound_demo.ipynb
```

The probe reads dimensions and JPEG headers on the raw path without decoding pixel content. Bias matching necessarily decodes, crops, and re-encodes images to construct the counterfactual set.

## Cache frozen CLIP features

Run the training preflight first:

```bash
uv run python scripts/preflight_train.py
```

It validates manifests, eval isolation, both binary classes, image paths, matched validation, compatible cache coverage, and available disk. Cluster jobs add `--require-cuda`; CLIP jobs also add `--require-cache`.

Start with a dry run to inspect image count and disk requirements:

```bash
uv run python scripts/cache_features.py --split train val --dry_run
```

Then smoke-test a small balanced subset:

```bash
uv run python scripts/cache_features.py --split train val --limit 32
```

Run the complete cache:

```bash
uv run python scripts/cache_features.py --split train val --device cuda
```

The cache is resumable and stored as sharded FP16 NumPy arrays. At the default four crops per image it requires approximately 2 MiB per image. Cache metadata includes the CLIP variant, crop size, seed, and bias-matching settings; incompatible caches are rejected.

## Train

Run a fast SRM-only CPU smoke test first:

```bash
uv run python -m provenance.train \
  --branches srm \
  --device cpu \
  --epochs 1 \
  --limit 32
```

Train the complete P7 model on GPU:

```bash
uv run python -m provenance.train \
  --branches clip,srm \
  --device cuda \
  --early_stop_metric auc_val_matched
```

Run branch ablations with either:

```bash
uv run python -m provenance.train --branches clip --device cuda
uv run python -m provenance.train --branches srm --device cuda
```

Useful overrides include:

```text
--epochs N
--batch_size N
--lr FLOAT
--lambda_consistency FLOAT
--patience N
--no_amp
--out PATH
```

Every run writes:

```text
runs/<timestamp>/
├── config.yaml       resolved experiment snapshot
├── metrics.json      epoch history, AUC controls, timing, seed, parameter counts
└── model.pt          best checkpoint selected by the requested metric
```

Early stopping defaults to AUC on `val_matched`. If that manifest is unavailable, the trainer records a fallback to ordinary validation AUC. Overall and bpp-stratified validation AUC are logged together so shortcut-driven gains remain visible.

## NUS SoC Compute Cluster

Enable **SoC Compute Cluster** in My SoC Services, connect to the SoC network or VPN, and log in through a designated Slurm login node:

```bash
ssh <soc-userid>@xlogin.comp.nus.edu.sg
```

Git does not transfer `data/`, `cache/`, or `runs/`; all three are intentionally ignored. On a fresh cluster clone, run the SID_Set download, extraction, and manifest commands from **Data setup** before submitting jobs. Build the 22–24 GiB CLIP token cache on the cluster rather than copying a partial local cache.

Do not train on the login node. Request a GPU allocation using the current options documented by SoC; a typical interactive request is:

```bash
salloc -p gpu-long --gres=gpu:1 --time=08:00:00 --cpus-per-task=8 --mem=64G
srun --pty bash
```

Inside the allocation, activate `tmux`, enter the repository, run `uv sync --all-groups`, cache CLIP features, and launch the full training command above. Use `gpu` for jobs up to three hours and `gpu-long` for longer runs. Cluster GPU types and resource syntax can change, so confirm them with `sinfo` and the current SoC GPU documentation before submission.

The repository also includes complete non-interactive jobs:

```bash
sbatch scripts/slurm_cache.sbatch
sbatch --export=ALL,BRANCH_MODE=both,RUN_TAG=p7-both-kl1 scripts/slurm_train.sbatch
```

Set `BRANCH_MODE` to `clip`, `srm`, or `both`. Other supported environment overrides are `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `LAMBDA_CONSISTENCY`, `EARLY_STOP_METRIC`, `LIMIT`, `RUN_TAG`, and `OUT_DIR`. `LIMIT` is intended only for balanced smoke runs.

## Evaluation controls

The metrics layer currently supports:

```bash
uv run python -m provenance.evaluate --split val
uv run python -m provenance.evaluate --split val_matched
```

It reports a content-shortcut baseline using out-of-fold logistic regression on bpp, plus AUC computed inside bpp quantile bins. A model should clear the bpp-only baseline and retain performance after stratification.

The full clean-versus-transform checkpoint grid described in the implementation plan is not yet connected; that is P9 work.

## Inference contract

The required interface already runs on bare CPU Python:

```bash
python predict.py --image_dir path/to/images --out predictions.json
```

It recursively emits:

```json
[
  {"image_path": "example.jpg", "pred": 0.5}
]
```

**Current limitation:** `predict.py` is still the P1 safety stub and returns `0.5` for every image. Loading trained weights, overlapping multi-crop inference, transformation-time augmentation, calibration, and stability reporting are scheduled for P10. The stub status is printed to stderr so it cannot be mistaken for model output.

## Tests

Run the complete suite:

```bash
uv run pytest -q
```

The suite covers transform determinism, native-resolution crops, manifest stability, eval leakage in both directions, CLIP cache compatibility, the fixed SRM bank, the confound probe, bpp matching, loss gradients, and an end-to-end SRM-only training run that verifies checkpoint and metrics output.

## Configuration

All experiment defaults live in [`configs/default.yaml`](configs/default.yaml). Important settings include:

| Key | Default | Purpose |
|---|---:|---|
| `seed` | `1337` | Dataset splits, crops, transforms, and training |
| `data.crop_size` | `224` | Native-resolution square crop |
| `data.crops_per_image` | `4` | Training crop multiplicity |
| `data.bias_match` | `true` | Common JPEG encoder before model input |
| `data.match_quality` | `90` | Bias-matching JPEG quality |
| `model.srm.tlu_threshold` | `0.0` | Disable shortcut-amplifying residual clamp |
| `train.lambda_consistency` | `1.0` | KL consistency weight |
| `train.amp` | `true` | CUDA automatic mixed precision |
| `eval.bpp_bins` | `5` | Quantile bins for shortcut-controlled AUC |

CLI overrides are written into each run's config snapshot.

## Design constraints and limitations

- Whole images are never resized before cropping. Resizing rewrites the high-frequency evidence used by the SRM branch.
- COCO val2017 and DALL·E Advanced are held-out demonstration/evaluation data and never training data.
- SID_Set cannot support unseen-generator validation because it has no generator field.
- The residual bpp signal after bias matching is controlled and reported, not assumed to disappear completely.
- CLIP token caching speeds the clean pass, but transformed views still require a live frozen-backbone pass.
- The current learned fusion weights are global rather than degradation-aware.
- The spectral branch exists only as a placeholder and is disabled.
- The robustness report, calibrated operating point, real-weight CPU predictor, and Kaggle reproduction notebook remain to be completed in later phases.

## Roadmap

- degradation-aware gated fusion
- heckpoint-driven clean/transformation evaluation grid and robustness report
- multi-crop CPU inference, TTA stability, and temperature calibration
- qualitative false-positive/false-negative analysis

## References

The SRM branch follows the residual families introduced by Fridrich and Kodovský, *Rich Models for Steganalysis of Digital Images* (IEEE TIFS, 2012), with CNN/TLU precedent from Ye, Ni, and Yi (IEEE TIFS, 2017) and forensic two-stream precedent from Zhou et al., *Learning Rich Features for Image Manipulation Detection* (CVPR, 2018). Implementation details and deliberate deviations are documented in `src/provenance/branches/srm.py`.
