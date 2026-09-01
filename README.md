# AIGC Detector

Camera-pipeline provenance detection for AI-generated images, designed to remain useful after JPEG compression, blur, resizing, noise, colour changes, and cropping.

**Official model: CamTrace-6M.** The name refers to its 6,002,830 learned
detector parameters; the complete inference graph has 309,969,038 parameters
when the frozen CLIP ViT-L/14 backbone is included.

AIGC Detector starts from a simple problem with many AIGC benchmarks: a detector can score well by learning how the datasets were saved instead of whether an image was generated. The project therefore combines three things:

1. a content-blind probe that measures encoder and file-format shortcuts;
2. a bias-matched evaluation protocol that removes those shortcuts; and
3. a detector built around camera-pipeline evidence, trained with transformation consistency.

> **Measured Status:** 195 tests pass (12 environment/data-dependent skips). Five trained ablations, raw and bias-matched robustness grids, calibrated CPU inference, and ranked error analysis are complete. Results below are read directly from [`reports/`](reports/).

## Project Contribution

The held-out demonstration set pairs COCO val2017 photographs with DALL·E Advanced images. Their encoding pipelines are visibly different even without decoding pixels: resolution, chroma subsampling, JPEG quantisation tables, and container formats nearly identify the class by themselves.

The repository's content-blind logistic-regression probe measured:

| Protocol | Images | ROC AUC |
|---|---:|---:|
| Raw COCO vs. DALL·E Advanced | 4,000 | **1.0000** |
| Fixed JPEG encoder + 256×256 native-resolution crops | 4,000 | **0.6527** |

The raw benchmark can therefore be solved perfectly without looking at image content. After matching encoder, subsampling, and dimensions, only bytes-per-pixel varies; the remaining AUC mostly measures the different compressibility of the image content.

The full, executed exhibit is in [`notebooks/01_confound_demo.ipynb`](notebooks/01_confound_demo.ipynb).

## Measured Results

All detector results below use the SID_Set validation data available during the
72-hour build. The primary protocol re-encodes crops through one JPEG pipeline
and evaluates a resolution-matched subset of 904 images (452 per class). This
controls the measured storage shortcut, but it is not an unseen-generator or
external-dataset test.

### Robustness under the bias-matched protocol

Each cell is image-level ROC AUC over eight native-resolution crops. `Jitter`
is the worse of the low/high colour-jitter variants; `Mean transformed` averages
all 15 non-clean conditions. Full-precision results and fixed-threshold accuracy
are in [`reports/robustness_table.md`](reports/robustness_table.md).

| Method | Clean | JPEG30 | Blur 2.0 | Resize 0.25 | Noise 0.10 | Jitter | Crop 0.80 | Mean transformed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP attention probe | **1.00000** | 0.99968 | 0.99966 | 0.99954 | 0.99792 | 0.99976 | 0.99991 | 0.99967 |
| SRM only | 0.99944 | 0.98839 | 0.94047 | 0.92855 | 0.90833 | 0.99845 | 0.99950 | 0.97894 |
| CLIP + SRM, λ=0 | 0.99998 | 0.99959 | 0.99935 | 0.99944 | 0.99709 | 0.99975 | 0.99991 | 0.99953 |
| CLIP + SRM, λ=1 | 0.99998 | **0.99979** | **0.99976** | **0.99972** | **0.99878** | **0.99992** | **0.99998** | **0.99981** |
| Degradation-aware gate, λ=1 | 0.99998 | 0.99977 | 0.99969 | 0.99967 | 0.99849 | 0.99986 | 0.99998 | 0.99976 |

The KL consistency objective is the useful ablation: against the same two
branches without KL, it raises mean transformed AUC from `0.99953` to `0.99981`
and mean fixed-operating-point accuracy from `97.286%` to `98.311%`. The gate is
a measured parity result, not a claimed breakthrough: it reaches `98.348%` mean
transformed accuracy but `0.99976` mean transformed AUC. The standalone CLIP
probe remains strongest on mean fixed-threshold accuracy (`98.510%`) on this
saturated, same-source split. External generator testing is therefore required
before choosing a production checkpoint solely from these differences.

The uncontrolled raw grid tells the same ranking story with slightly inflated
scores; its full output is in
[`reports/robustness_table_raw.md`](reports/robustness_table_raw.md).

### Training ablation and wall-clock

These are single-run measurements on one NVIDIA H100 NVL in the NUS SoC
cluster. Time covers the trainer recorded in `metrics.json`; queueing and the
one-time 18.4-minute, 24.1 GiB CLIP cache build are excluded.

| Run | Trainable parameters | Epochs run | Best epoch | Matched-val AUC | Training time |
|---|---:|---:|---:|---:|---:|
| CLIP, λ=1 | 1,447,683 | 8 | 4 | 0.99812 | 2h 29m 35s |
| SRM, λ=1 | 4,554,755 | 10 | 9 | 0.99579 | 59m 22s |
| CLIP + SRM, λ=0 | 6,002,438 | 10 | 6 | **0.99932** | 2h 33m 57s |
| CLIP + SRM, λ=1 | 6,002,438 | 10 | 9 | 0.99908 | 2h 33m 13s |
| Gate only, initialized from λ=1 | 392 | 3 | 0 | 0.99901 | 41m 57s |

KL slightly lowers the clean matched-validation selection score while improving
the transformation grid. That is precisely why clean validation alone is an
insufficient selection criterion for a robustness-focused detector. These
measurements do **not** support the original planning estimate of training on a
T4 in under an hour, so this README does not make that claim.

For the remaining measured passes, the bias-matched robustness grid took
`13m 42s` on one H100 NVL, the raw grid took `18m 06s` on one H100 NVL, and
error analysis took `12m 26s` on one H100 NVL. Full multicrop/TTA calibration
took `45m 10s` on one TITAN V in FP32.

### Calibration and failure analysis

Temperature scaling on the 904-image matched split learned `T=0.162786`, reduced
NLL from `0.024487` to `0.020112`, and selected threshold `0.427470`. The
achieved FPR is `0.88%` (four of 452 real images), the closest conservative
discrete operating point below 1%; see
[`reports/calibration.md`](reports/calibration.md).

Across the 16-condition error-analysis grid, the selected gated checkpoint has
214 false-positive condition-cases out of 7,232 real cases (`2.96%`) and 14
false-negative condition-cases out of 7,232 synthetic cases (`0.19%`). Resize
has the highest grouped FPR (`4.76%`); Gaussian noise has the highest grouped
FNR (`0.37%`). Visual inspection shows false positives clustering around
already defocused or low-detail photographs after blur, resampling, or JPEG,
while false negatives cluster around near-white minimalist generations and a
few synthetic scenes after strong noise. Several conditions repeatedly fail on
the same underlying images, so these are concentrated sensitivities rather
than uniform collapse.

[View the ranked 12 FP / 12 FN contact sheet](reports/error_contact_sheet.png)
or read the complete [`error_analysis.md`](reports/error_analysis.md).

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
                            └── global or gated fusion ┘                │
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

Selected branches train together. The P7 control combines their logits with learned global softmax weights. P8 enables a per-image gate conditioned only on three cheap, content-light measurements: Laplacian variance for sharpness, an 8×8 blockiness-derived JPEG-quality proxy, and source image size. A two-layer MLP maps those measurements to softmax branch weights. The gate can warm-start from a P7 checkpoint and train with both forensic branches frozen.

One degradation specification is sampled per batch and replayed at the same strength for every crop. The consistency KL is optimized jointly with clean and transformed cross-entropy:

```text
L = CE(f(I), y) + CE(f(T(I)), y) + λ · KL(f(I) || f(T(I)))
```

`λ` is configured by `train.lambda_consistency` and defaults to `1.0`.

### Parameter accounting

| Component | Parameters | Trainable now? |
|---|---:|:---:|
| CLIP ViT-L/14 vision encoder | 303,966,208 | No |
| Attention-probe head | 1,447,682 | Yes |
| SRM residual CNN | 4,554,754 | Yes |
| Two-branch fusion weights | 2 | Yes |
| Degradation-aware gate | 392 | Gate-training phase only |
| **Gated model total** | **309,969,038** | **392 in P8; 6,002,438 in P7** |

The model is comfortably below the competition's 2B-parameter limit. Enabling P8 adds only the 392-parameter gate shown above. The optional spectral head is not included because it remains inactive.

`CamTrace-6M` counts the learned attention probe, SRM CNN, global fusion, and
degradation gate. It does not imply that the frozen CLIP backbone has only six
million parameters.

## Repository layout

```text
configs/default.yaml                   Main experiment configuration
src/provenance/data.py                 Manifests, split isolation, crops, bpp matching
src/provenance/transforms.py           Six train/evaluation degradations
src/provenance/shortcut.py             Content-blind confound probe
src/provenance/branches/clip_probe.py  Frozen CLIP, attention probe, token cache
src/provenance/branches/srm.py         Fixed SRM bank and residual CNN
src/provenance/train.py                P7 joint consistency trainer
src/provenance/evaluate.py             Checkpoint robustness grid and controlled metrics
src/provenance/fuse.py                 Degradation estimates and per-image softmax gate
src/provenance/inference.py            Overlapping crops and four-view TTA inference
src/provenance/calibrate.py            Temperature scaling and fixed-FPR threshold
scripts/                               Data, manifest, cache, and cluster helpers
scripts/preflight_train.py             Data, device, disk, and cache validation
scripts/error_analysis.py              Ranked errors, grouped evidence, contact sheet
notebooks/01_confound_demo.ipynb       Executed confound demonstration
predict.py                             Real-weight CPU submission interface and sidecar
tests/                                 Unit, leakage, and end-to-end smoke tests
```

## Requirements

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- Enough local disk for the selected datasets and feature cache
- A CUDA GPU for practical CLIP feature caching and full training
- CPU-only execution for tests, SRM smoke runs, and the required `predict.py` path

Install the locked environment:

```bash
git clone https://github.com/mathenaangeles/AIGC-Detector.git
cd AIGC-Detector
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

Train the P8 gate from a completed two-branch P7 checkpoint while keeping both
forensic branches fixed:

```bash
uv run python -m provenance.train \
  --branches clip,srm \
  --device cuda \
  --gating \
  --init_checkpoint runs/p7-both-kl1/model.pt \
  --freeze_branches \
  --epochs 4 \
  --patience 2 \
  --out runs/p8-gated
```

This preserves the P7 detector and isolates the gain from conditional fusion.
Omit `--freeze_branches` only for a deliberate joint-fine-tuning experiment.

Every run writes:

```text
runs/<timestamp>/
├── config.yaml       resolved experiment snapshot
├── metrics.json      epoch history, AUC controls, timing, seed, parameter counts
└── model.pt          best checkpoint selected by the requested metric
```

Early stopping defaults to AUC on `val_matched`. If that manifest is unavailable, the trainer records a fallback to ordinary validation AUC. Overall and bpp-stratified validation AUC are logged together so shortcut-driven gains remain visible.

## Reproduce the measured pipeline

From a configured checkout with SID_Set extracted, the measured path is:

```bash
# 1. Deterministic manifests and bias-matched validation subset
uv run python scripts/build_manifest.py

# 2. Validate data isolation, files, and disk
uv run python scripts/preflight_train.py

# 3. Build the frozen CLIP crop cache
sbatch scripts/slurm_cache.sbatch

# 4. Train P7 ablations; vary BRANCH_MODE, LAMBDA_CONSISTENCY, and RUN_TAG
sbatch --export=ALL,BRANCH_MODE=both,LAMBDA_CONSISTENCY=1.0,RUN_TAG=p7-both-kl1 \
  scripts/slurm_train.sbatch

# 5. Train the gate from the robust two-branch checkpoint
GATING=1 FREEZE_BRANCHES=1 INIT_CHECKPOINT=runs/p7-both-kl1/model.pt \
RUN_TAG=p8-gated-kl1 sbatch --export=ALL scripts/slurm_train.sbatch

# 6. Raw and controlled robustness grids
PROTOCOL=bias_matched OUT=reports/robustness_table.md \
  sbatch scripts/slurm_evaluate.sbatch
PROTOCOL=raw OUT=reports/robustness_table_raw.md \
  sbatch scripts/slurm_evaluate.sbatch

# 7. Calibration and error analysis
sbatch scripts/slurm_calibrate.sbatch
sbatch scripts/slurm_error_analysis.sbatch
```

The exact run metrics committed under [`reports/run_metrics/`](reports/run_metrics/)
record seed, epoch history, parameter counts, precision, and wall-clock. Slurm
scripts resolve the submission directory explicitly and place temporary worker
files on node-local storage.

## NUS SoC Compute Cluster

Enable **SoC Compute Cluster** in My SoC Services, connect to the SoC network or VPN, and log in through a designated Slurm login node:

```bash
ssh -J <soc-userid>@stujump.comp.nus.edu.sg \
  <soc-userid>@xlogin.comp.nus.edu.sg
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

Set `BRANCH_MODE` to `clip`, `srm`, or `both`. Other supported environment overrides are `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `LAMBDA_CONSISTENCY`, `EARLY_STOP_METRIC`, `LIMIT`, `RUN_TAG`, and `OUT_DIR`. P8 additionally uses `GATING=1`, `INIT_CHECKPOINT`, and `FREEZE_BRANCHES=1`. `LIMIT` is intended only for balanced smoke runs.

## Evaluation controls

P9 evaluates any number of checkpoints on clean crops and every configured
transform severity. Name methods with `NAME=RUN`; the name becomes the table row:

```bash
uv run python -m provenance.evaluate \
  --runs \
    clip=runs/p7-clip-kl1 \
    srm=runs/p7-srm-kl1 \
    both_no_kl=runs/p7-both-kl0 \
    both_kl=runs/p7-both-kl1 \
    gated=runs/p8-gated-kl1 \
  --protocol bias_matched \
  --device cuda \
  --out reports/robustness_table.md
```

`raw` defaults to the ordinary `val` split. `bias_matched` defaults to
`val_matched` and passes every crop through the common JPEG encoder before the
requested degradation. `--split` can explicitly override those defaults. The
runner computes CLIP tokens once per batch and shares them across methods.

The Markdown report contains ROC AUC for clean plus all 15 transformed
conditions, mean transformed AUC, and accuracy with each method's threshold
fixed on its clean real scores at the configured 1% FPR. The same threshold is
reused unchanged for every degradation. A machine-readable JSON file is written
beside the report.

On Slurm, run the bias-controlled headline table and the raw diagnostic table:

```bash
PROTOCOL=bias_matched OUT=reports/robustness_table.md \
  sbatch scripts/slurm_evaluate.sbatch

PROTOCOL=raw OUT=reports/robustness_table_raw.md \
  sbatch scripts/slurm_evaluate.sbatch
```

The script defaults to all four P7 ablations and `runs/p8-gated-kl1`. Override
its space-separated `EVAL_RUNS` environment variable if a run has a different
directory name. Use `LIMIT=20` only for a balanced smoke test.

## Inference contract

First calibrate the selected gated checkpoint. The calibration pass uses the
same overlapping crop grid, four TTA views, trimmed-mean aggregation, and
bias-matched protocol as deployment:

```bash
uv run python -m provenance.calibrate \
  --checkpoint runs/p8-gated-kl1/model.pt \
  --protocol bias_matched \
  --device cuda \
  --out runs/p8-gated-kl1/calibration.json \
  --report reports/calibration.md
```

It fits one positive temperature by image-level NLL and reports the calibrated
threshold whose clean-real false-positive rate is at most 1%. The JSON artifact
is cryptographically bound to the checkpoint and records the protocol, crop
grid, aggregation, and TTA settings. Inference uses BF16 on supported CUDA
devices and FP32 on older GPUs; it never falls back to FP16 because unclamped
SRM residuals can overflow that format.

The required prediction interface runs with real weights on CPU:

Download the selected checkpoint and its cryptographically bound calibration
artifact from the
[`v1.0.0-techjam` release](https://github.com/mathenaangeles/AIGC-Detector/releases/tag/v1.0.0-techjam),
then run:

```bash
uv run python predict.py \
  --image_dir path/to/images \
  --checkpoint aigc-detector-p8-gated-kl1-model.pt \
  --calibration aigc-detector-p8-gated-kl1-calibration.json \
  --device cpu \
  --out predictions.json
```

It recursively emits the exact minimal submission contract:

```json
[
  {"image_path": "example.jpg", "pred": 0.5}
]
```

It also writes `predictions_detailed.json` beside the requested output. Each
detailed row includes the calibrated decision, number of crops and views,
per-TTA predictions, and `stability`: the population variance across all crop
and TTA probabilities. Lower stability variance means the prediction is less
sensitive to crop location and deployment degradation.

Inference uses a deterministic 50%-overlap grid capped at eight crops, then
applies `{identity, jpeg90, resize0.5, crop80}` to every crop. Probabilities are
temperature-scaled and aggregated with a trimmed mean. `--protocol
bias_matched` is the default because it matches training and the primary P9
evaluation; raw inference is available explicitly but cannot reuse a
bias-matched calibration artifact.

On the SoC cluster, submit calibration non-interactively:

```bash
sbatch scripts/slurm_calibrate.sbatch
```

## Error analysis

P11 re-scores the selected gated checkpoint over the complete 16-condition
robustness grid. It fixes the decision threshold on clean real scores at 1%
FPR, ranks the 12 most confident false positives and 12 false negatives, and
writes a labelled contact sheet plus measured group summaries:

```bash
uv run python scripts/error_analysis.py \
  --checkpoint runs/p8-gated-kl1/model.pt \
  --protocol bias_matched \
  --device cuda
```

Outputs are `reports/error_analysis.md`, `reports/error_analysis.json`, and
`reports/error_contact_sheet.png`. Counts and error rates are grouped by
transform family, source, and generator where the manifest genuinely carries
generator metadata. SID_Set has no generator field, so its generated images are
reported as `unknown` rather than assigned a fabricated generator.

On Slurm:

```bash
sbatch scripts/slurm_error_analysis.sbatch
```

Generated Markdown, JSON, and contact-sheet files under `reports/` are intended
to be committed. Raw datasets, CLIP caches, checkpoints, Slurm logs, smoke-test
predictions, and `runs/` remain ignored. The deployment calibration JSON stays
beside its ignored checkpoint and should be packaged with that checkpoint for
submission rather than committed as a benchmark result.

## Tests

Run the complete suite:

```bash
uv run pytest -q
```

The suite covers transform determinism, native-resolution crops, manifest stability, eval leakage in both directions, CLIP cache compatibility, the fixed SRM bank, the confound probe, bpp matching, loss gradients, training checkpoints, robustness reports, overlapping inference crops, TTA aggregation, calibration binding, and the minimal prediction contract.

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
- The reported detector grids are same-source SID_Set validation, not external or unseen-generator evidence. SID_Set has no generator field, so generator-level claims would be fabricated.
- The residual bpp signal after bias matching is controlled and reported, not assumed to disappear completely.
- CLIP token caching speeds the clean pass, but transformed views still require a live frozen-backbone pass.
- The gate is technically successful but does not materially beat the ungated KL model on AUC; its tiny fixed-threshold accuracy gain needs confirmation across seeds and external sources.
- The spectral branch exists only as a placeholder and is disabled.
- The matched split is used for model selection, calibration, and reporting because no third labelled split was available. Calibration numbers are therefore descriptive, not an unbiased final-test estimate.
- Results are single-seed (`1337`) measurements without confidence intervals.
- The lightweight checkpoint excludes the frozen OpenAI CLIP weights; first use downloads them through `open_clip`. CPU inference works and was smoke-tested, but ViT-L/14 is not a low-latency edge model.

## What I would do with more time

1. Train with WildFake and hold out complete generator directories, then evaluate once on untouched COCO/DALL·E data.
2. Create disjoint model-selection, calibration, and final-test splits and repeat every ablation across at least three seeds with bootstrap confidence intervals.
3. Add more real camera pipelines, social-media recompression chains, screenshots, text-heavy graphics, and minimalist generations—the error sheet shows these tails matter.
4. Distil or replace ViT-L/14 for lower CPU latency while retaining the SRM branch and stability signal.
5. Test whether a regularized or supervised gate improves materially; otherwise ship the simpler ungated KL ensemble or CLIP probe selected on external data.
6. Add a fully executed end-to-end reproduction notebook and publish external-domain checkpoints once they have passed generator-held-out evaluation.

## Team contribution

This is a solo implementation. The project author performed the benchmark
audit, data protocol design, model and training implementation, cluster runs,
robustness evaluation, calibration, CPU inference validation, and error
analysis. External libraries and pretrained weights are credited below.

## References

The SRM branch follows the residual families introduced by Fridrich and Kodovský, *Rich Models for Steganalysis of Digital Images* (IEEE TIFS, 2012), with CNN/TLU precedent from Ye, Ni, and Yi (IEEE TIFS, 2017) and forensic two-stream precedent from Zhou et al., *Learning Rich Features for Image Manipulation Detection* (CVPR, 2018). Implementation details and deliberate deviations are documented in `src/provenance/branches/srm.py`.
