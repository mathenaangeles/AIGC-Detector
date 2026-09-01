#!/usr/bin/env python3
"""Build a narrated, submission-safe CamTrace-6M demo video.

Requires Pillow, macOS `say`, ffmpeg, and the measured repository artifacts.
The output is a polished fallback; a live screen recording following
demo/DEMO_VIDEO_SCRIPT.md remains the strongest judging video.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
BUILD = DEMO / "video_build"
OUT = DEMO / "camtrace-6m-demo.mp4"
W, H = 1920, 1080
BG = "#08121f"
PANEL = "#10243a"
WHITE = "#f4f7fb"
MUTED = "#a9bfd4"
CYAN = "#45d6d1"
GOLD = "#ffca63"
RED = "#ff7474"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


SLIDES = [
    {
        "title": "CamTrace-6M",
        "subtitle": "Robust AI-generated image detection under real-world transformations",
        "bullets": [
            "0.99976 mean AUC across 15 real-world transformations",
            "JPEG · blur · resize · noise · colour jitter · crop",
            "6.0M learned detector parameters · CPU-ready inference",
        ],
        "seconds": 18,
        "narration": "This is CamTrace six M: robust AI image detection after compression, blur, resizing, noise, colour changes, and cropping. It reaches zero point nine nine nine seven six mean transformed A U C. But the key insight came before training: a content blind probe classified the raw benchmark perfectly without looking at pixels.",
    },
    {
        "title": "First: audit the benchmark",
        "subtitle": "A perfect score can reveal a broken protocol",
        "bullets": [
            "Header-only probe: 1.0000 AUC on raw COCO vs DALL-E Advanced",
            "Same crop + JPEG encoder + subsampling: 0.6527 AUC",
            "Bias matching, bpp controls, and strict held-out isolation",
        ],
        "seconds": 23,
        "narration": "File dimensions, PNG versus JPEG, chroma subsampling, and quantisation tables gave a raw A U C of one point zero. After a common crop and JPEG encoder, the same metadata probe fell to zero point six five. So we bias match the data and report bytes per pixel controlled metrics instead of trusting a headline score.",
    },
    {
        "title": "Complementary evidence",
        "subtitle": "Semantic generalisation + camera-pipeline forensics",
        "bullets": [
            "Frozen CLIP ViT-L/14 + trainable patch attention probe",
            "30 fixed SRM filters → 90 residual maps → compact CNN",
            "Consistency: CE(clean) + CE(transformed) + KL agreement",
            "392-parameter degradation-aware softmax gate",
            "309,969,038 total parameters; 6,002,830 learned detector parameters",
        ],
        "seconds": 25,
        "narration": "CamTrace fuses a frozen CLIP attention probe with a Spatial Rich Model branch containing thirty fixed forensic filters and a compact residual C N N. Clean and transformed views share a consistency objective. A tiny degradation aware gate combines semantic and camera pipeline evidence per image. The full graph is three hundred ten million parameters, below the two billion limit, while only six million detector parameters are learned.",
    },
    {
        "title": "Real released-model inference",
        "subtitle": "Required directory → JSON contract, running on CPU",
        "bullets": [
            "predict.py  demo/evaluation_inputs  →  predictions.json",
            "model.pt + calibration.json + CPU",
            "No API key or hosted service required",
            "8 overlapping crops × 4 TTA views per image",
        ],
        "seconds": 25,
        "narration": "This is the exact judging contract on C P U. It recursively accepts an image directory and writes one AI generated probability per path. No A P I key or hosted service is required. The frozen CLIP backbone is downloaded separately on first use and is not hidden inside our lightweight checkpoint.",
        "images": ["demo/evaluation_inputs/sid_real_ccby.jpg", "demo/evaluation_inputs/sid_synthetic_ccby.png"],
        "labels": ["Camera image — CC BY", "Synthetic image — CC BY"],
    },
    {
        "title": "Actual calibrated output",
        "subtitle": "Minimal output + optional stability evidence",
        "bullets": [
            "sid_real_ccby.jpg       P(AI) = 0.000004",
            "sid_synthetic_ccby.png  P(AI) = 1.000000",
            "Deployment threshold = 0.427470",
            "Detailed sidecar: decision, 32 views, per-TTA scores, stability",
        ],
        "seconds": 22,
        "narration": "The minimal JSON contains only image path and prediction. The real image scores near zero and the synthetic image scores one in this example. The optional sidecar exposes thirty two crop and transformation views, the fixed calibrated decision, and prediction variance as a stability signal for human review.",
    },
    {
        "title": "Measured robustness",
        "subtitle": "Balanced, bias-matched SID_Set validation — 904 images",
        "bullets": [
            "Clean ROC AUC                         0.999980",
            "Mean over all 15 transformations      0.999760",
            "Worst: Gaussian noise σ=0.10          0.998492",
            "Mean fixed-threshold accuracy          98.35%",
            "One clean threshold reused across every degradation",
        ],
        "seconds": 25,
        "narration": "On the balanced bias matched nine hundred and four image protocol, the selected model reaches zero point nine nine nine nine eight clean A U C and zero point nine nine nine seven six mean A U C over all fifteen required transformations. More importantly, the table keeps one clean threshold fixed across JPEG, blur, resize, noise, colour jitter, and crop conditions.",
    },
    {
        "title": "We publish the failures",
        "subtitle": "Decision support, not an oracle",
        "bullets": [
            "False positives: low-detail real photos after blur / resize",
            "False negatives: minimalist generations under strong noise",
            "214 FP and 14 FN condition-cases across 16 conditions",
            "Probability + stability → prioritised human review",
            "Limitation: same-source, single-seed validation; external generators next",
        ],
        "seconds": 25,
        "narration": "We also publish the failures. False positives cluster in low detail real photographs after blur or resize. False negatives cluster in minimalist generations under noise. CamTrace is decision support, not an oracle. Its probability and stability signal help trust and safety reviewers prioritise uncertain cases while making false positive cost explicit.",
    },
    {
        "title": "Open and reproducible",
        "subtitle": "Code · weights · calibration · reports · 195 passing tests",
        "bullets": [
            "github.com/mathenaangeles/AIGC-Detector",
            "MIT software license + complete third-party attribution",
            "CamTrace-6M detects the image — not the file-format shortcut",
        ],
        "seconds": 15,
        "narration": "The code, weights, calibration, robustness grid, tests, and error analysis are public and reproducible. CamTrace six M detects the image, not the way the dataset happened to be saved.",
    },
]


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def wrapped(draw, xy, text, width, *, size=42, fill=WHITE, bold=False, spacing=14):
    chars = max(20, int(width / (size * 0.55)))
    lines = textwrap.wrap(text, width=chars)
    draw.multiline_text(xy, "\n".join(lines), font=font(size, bold), fill=fill, spacing=spacing)
    return len(lines) * (size + spacing)


def render_slide(index: int, spec: dict) -> Path:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 24, H), fill=CYAN)
    draw.text((90, 70), spec["title"], font=font(72, True), fill=WHITE)
    draw.text((94, 170), spec["subtitle"], font=font(35), fill=CYAN)
    draw.line((94, 235, 1820, 235), fill="#254661", width=3)

    image_paths = spec.get("images", [])
    y = 300
    if image_paths:
        labels = spec.get("labels", [""] * len(image_paths))
        box_w, box_h = 475, 420
        for j, rel in enumerate(image_paths):
            src = Image.open(ROOT / rel).convert("RGB")
            src.thumbnail((box_w, box_h - 55))
            x = 95 + j * 515
            panel = Image.new("RGB", (box_w, box_h), PANEL)
            px = (box_w - src.width) // 2
            panel.paste(src, (px, 10))
            pd = ImageDraw.Draw(panel)
            pd.text((18, box_h - 43), labels[j], font=font(24, True), fill=GOLD)
            canvas.paste(panel, (x, y))
        bullet_x, bullet_y, bullet_w = 1140, y, 650
    else:
        bullet_x, bullet_y, bullet_w = 115, y, 1640

    for bullet in spec["bullets"]:
        draw.ellipse((bullet_x, bullet_y + 14, bullet_x + 18, bullet_y + 32), fill=GOLD)
        bullet_size = 30 if image_paths else 38
        used = wrapped(
            draw, (bullet_x + 42, bullet_y), bullet, bullet_w - 42,
            size=bullet_size,
        )
        bullet_y += max(72, used + 18)

    draw.rectangle((0, H - 72, W, H), fill="#0b1c2d")
    draw.text((90, H - 52), "CamTrace-6M · TikTok TechJam 2026", font=font(25, True), fill=MUTED)
    draw.text((W - 660, H - 52), "Demo images: SID_Set, CC BY 4.0 · See THIRD_PARTY_NOTICES.md", font=font(20), fill=MUTED)
    path = BUILD / f"slide-{index:02d}.png"
    canvas.save(path)
    return path


def run(*args):
    subprocess.run(args, check=True)


def main():
    for cmd in ("ffmpeg", "ffprobe", "say"):
        if not shutil.which(cmd):
            raise SystemExit(f"required command not found: {cmd}")
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True)
    segments = []
    for i, spec in enumerate(SLIDES, 1):
        slide = render_slide(i, spec)
        audio = BUILD / f"audio-{i:02d}.aiff"
        segment = BUILD / f"segment-{i:02d}.mp4"
        run("say", "-v", "Samantha", "-r", "176", "-o", str(audio), spec["narration"])
        run(
            "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(slide),
            "-i", str(audio), "-filter_complex",
            "[1:a]aresample=48000,pan=stereo|c0=c0|c1=c0,volume=5dB,alimiter=limit=0.95,apad=pad_dur=1[a]",
            "-map", "0:v", "-map", "[a]", "-t", str(spec["seconds"]),
            "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-ac", "2", "-disposition:a:0", "default", str(segment),
        )
        segments.append(segment)
    concat = BUILD / "concat.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in segments))
    run(
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(OUT),
    )
    print(OUT)


if __name__ == "__main__":
    main()
