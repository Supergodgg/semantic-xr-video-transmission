# Semantic XR Video Transmission

This repository contains the code and paper sources for a network-oriented evaluation of a semantic XR video transmission prototype under frame-level update loss.

The prototype represents video frames as semantic reuse events, full VAE latent updates, or residual latent updates. It evaluates latent residual coding (LRC), latitude-adaptive quantization (LAQ) for equirectangular panoramic video, receiver-side SwinIR restoration, and frame-level update loss behaviour against traditional codec baselines.

## Repository Layout

```text
src/
  Ninth_drift.py              # two-level semantic video prototype
  Tenth_drift.py              # LRC/LAQ-enhanced panoramic prototype

scripts/
  generate_packet_loss_videos.py
  compute_packet_loss_quality.py
  compute_packet_loss_all_metrics.py

paper/
  MSWiM2026_paper.tex
  bibliography.bib
  Figures/

results/
  masks/                      # frame-level loss masks used in experiments
  summaries/                  # generated loss-video summaries
  ninth_tenth_*_comparison.json
```

Large videos, model checkpoints, generated PDFs, and local build artifacts are intentionally not included.

## Main Components

- **Semantic reuse:** CLIP similarity decides whether a frame can reuse the previous receiver reconstruction.
- **Latent updates:** non-reused frames are sent as VAE latent payloads.
- **Latent residual coding:** temporally similar Level 2 frames can transmit residual latents instead of full latent tensors.
- **Latitude-adaptive quantization:** panoramic ERP frames use lower precision near polar regions and higher precision near the equator.
- **Frame-level update loss:** controlled loss masks remove complete frame updates and use previous-state fallback.

## Requirements

The scripts were developed in a Python research environment with GPU support. Core dependencies include:

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- Pillow
- scikit-image
- ffmpeg and ffprobe on `PATH`
- optional: `zstandard`

The main prototype scripts expect local paths to pretrained VAE and SwinIR resources. Update `PROJECT_ROOT`, `SWINIR_ROOT`, `VAE_PATH`, and `SWINIR_PATH` in `src/Ninth_drift.py` and `src/Tenth_drift.py` before running them in a new environment.

## Example Commands

Generate frame-level loss videos:

```bash
python scripts/generate_packet_loss_videos.py \
  --dataset vr \
  --methods ours=Tenth_drift_vr_15_output.mp4 h264=vr_h264_200k.mp4 vp9=vr_vp9_200k.mp4 \
  --loss-rates 0 10 20 30 \
  --seed 42
```

Compute PSNR and SSIM for generated loss videos:

```bash
python scripts/compute_packet_loss_quality.py \
  --root /path/to/experiment/root
```

Compile the paper:

```bash
cd paper
pdflatex -interaction=nonstopmode MSWiM2026_paper.tex
bibtex MSWiM2026_paper
pdflatex -interaction=nonstopmode MSWiM2026_paper.tex
pdflatex -interaction=nonstopmode MSWiM2026_paper.tex
```

## Notes

The reported payload values in the paper are visual payloads only. They do not include audio, container metadata, or deployable protocol headers. The frame-level loss model is an application-layer abstraction: it removes complete frame updates rather than simulating raw packet loss.
