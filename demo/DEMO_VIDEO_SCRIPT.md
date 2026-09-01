# CamTrace-6M three-minute demo script

Target length: **2:40–2:55**. Hard stop at 3:00.

The generated `camtrace-6m-demo.mp4` is a submission-safe narrated fallback.
For the strongest judging impression, record the live path below and use the
same narration. Do not add commercial music, logos, memes, movie clips, or
unattributed images.

## Before recording

1. Extract `aigc-detector-devpost.zip` into a clean folder.
2. Open that folder in VS Code.
3. Open `demo/evaluation_inputs/` in the Explorer. These two images are
   attributed SID_Set excerpts used under CC BY 4.0; keep the attribution in
   the video description.
4. Open a terminal in the extracted project folder.
5. Run `uv sync --all-groups` before recording so installation time is not in
   the three-minute video.
6. Make the terminal font at least 18 pt and hide notifications.
7. Keep these tabs open in order:
   - `README.md`
   - `demo/evaluation_inputs/`
   - terminal
   - `predictions.json`
   - `predictions_detailed.json`
   - `reports/robustness_table.md`
   - `reports/error_analysis.md`

## Recording clicks on macOS

1. Press **Shift–Command–5**.
2. Click **Record Selected Portion**.
3. Drag the frame around VS Code and the terminal only; exclude the menu bar if
   possible.
4. Click **Options** → choose your microphone → enable **Show Mouse Clicks**.
5. Click **Record**.
6. Follow the timed script below.
7. Stop from the menu-bar stop icon before 3:00.

## Timed narration and actions

### 0:00–0:15 — Hook

**Click:** Show the README title and measured-status paragraph.

**Say:**

> This is CamTrace-6M, a robust AI-image detector. The surprising part is not
> our accuracy. It is that a content-blind probe classified the raw benchmark
> perfectly without looking at pixels. CamTrace is designed to remove that
> shortcut and survive real-world reposting.

### 0:15–0:38 — Problem insight

**Click:** Scroll to the content-blind probe table.

**Say:**

> File dimensions, PNG versus JPEG, chroma subsampling, and quantisation tables
> gave a raw AUC of one point zero. After a common crop and JPEG encoder, the
> same metadata probe fell to zero point six five. So we bias-match the data
> and report bytes-per-pixel-controlled metrics instead of trusting a headline
> benchmark score.

### 0:38–1:00 — Architecture

**Click:** Scroll to the architecture diagram.

**Say:**

> CamTrace fuses a frozen CLIP attention probe with a Spatial Rich Model branch
> containing thirty fixed forensic filters and a compact residual CNN. During
> training, clean and transformed views share a consistency objective. A tiny
> degradation-aware gate combines semantic and camera-pipeline evidence per
> image. The full graph is three hundred ten million parameters, below the two
> billion limit, while only six million detector parameters are learned.

### 1:00–1:20 — Inputs and license

**Click:** Open `demo/evaluation_inputs/` and preview both images.

**Say:**

> Here is an attributed CC BY test pair from SID-Set: one camera photograph and
> one synthetic image. These are demonstration inputs, not independent test
> evidence. Dataset attribution and every modification are documented in the
> repository's third-party notices.

### 1:20–1:50 — Real inference

**Click:** Switch to the terminal and run:

```bash
uv run python predict.py \
  --image_dir demo/evaluation_inputs \
  --checkpoint model.pt \
  --calibration calibration.json \
  --device cpu \
  --out predictions.json
```

**Say while it runs:**

> This is the exact judging contract on CPU. It recursively accepts an image
> directory and writes one AI-generated probability per path. No API key or
> hosted service is required. The frozen CLIP backbone is downloaded separately
> on first use and is not hidden inside our lightweight checkpoint.

### 1:50–2:10 — Outputs and stability

**Click:** Open `predictions.json`, then `predictions_detailed.json`.

**Say:**

> The minimal JSON contains only image path and prediction. The real image
> scores near zero and the synthetic image scores one in this example. The
> optional sidecar exposes thirty-two crop-and-transformation views, the fixed
> calibrated decision, and prediction variance as a stability signal for human
> review.

### 2:10–2:35 — Robustness evidence

**Click:** Open `reports/robustness_table.md` and show the clean and transformed
columns.

**Say:**

> On the balanced, bias-matched nine-hundred-and-four-image protocol, the
> selected model reaches zero point nine nine nine nine eight clean AUC and
> zero point nine nine nine seven six mean AUC over all fifteen required
> transformations. More importantly, the table keeps one clean threshold fixed
> across JPEG, blur, resize, noise, colour jitter, and crop conditions.

### 2:35–2:52 — Honest failure analysis and impact

**Click:** Open `reports/error_analysis.md` and show the summary, not the contact
sheet.

**Say:**

> We also publish the failures. False positives cluster in low-detail real
> photographs after blur or resize; false negatives cluster in minimalist
> generations under noise. CamTrace is decision support, not an oracle: its
> probability and stability signal help trust-and-safety reviewers prioritise
> uncertain cases while making false-positive cost explicit.

### 2:52–2:58 — Close

**Click:** Return to the README title.

**Say:**

> The code, weights, calibration, robustness grid, tests, and error analysis are
> public and reproducible. CamTrace-6M detects the image—not the way the dataset
> happened to be saved.

## YouTube upload clicks

1. Open https://youtube.com and sign in.
2. Click **Create** (camera-plus icon) → **Upload video**.
3. Select `demo/camtrace-6m-demo.mp4` or your stronger live recording.
4. Title: **CamTrace-6M — Robust AI-Generated Image Detection | TikTok TechJam 2026**
5. Paste `demo/YOUTUBE_DESCRIPTION.md` into the description.
6. Choose **No, it’s not made for kids**.
7. Click **Next** through Video elements and Checks.
8. Under Visibility, choose **Public**.
9. Click **Publish** and copy the public `youtu.be` URL.
10. Open the DevPost submission editor. Paste the URL into the required demo
    video field and replace `[PASTE_PUBLIC_YOUTUBE_URL_BEFORE_SUBMISSION]` in
    the project story.
11. Open the public video in a private/incognito window and confirm it plays.
12. Submit the DevPost project—not only save the draft.

## Final video-description attribution

The demonstration pair is adapted from SID_Set under CC BY 4.0. The images are
cropped, JPEG-normalised, and evaluated under transformation-time augmentation.
Full attribution: https://github.com/mathenaangeles/AIGC-Detector/blob/main/THIRD_PARTY_NOTICES.md
