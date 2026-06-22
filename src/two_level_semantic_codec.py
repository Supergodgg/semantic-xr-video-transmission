"""
Ninth Drift - Two-Level Semantic Video Codec

Core idea: CLIP embedding cosine similarity drives adaptive skip.
  Level 1 (sim > THR_HIGH):  Skip frame, reuse previous        → 0 KB
  Level 2 (all other frames): VAE latent + quantize + SwinIR   → ~10-15 KB

Each L2 frame is independently encoded — no dependency on previous frames,
handles scene changes naturally without generative models.

Evaluation: CLIP Score, LPIPS, SSIM, PSNR — all measured per-frame.

Usage:
  python two_level_semantic_codec.py --calibrate                           # auto threshold
  python two_level_semantic_codec.py --input my.mp4 --frames 99999 --calibrate --l2_scale 1.0
"""

import os
import sys
import gc
import argparse
import json
from dataclasses import dataclass, field
from typing import List

# ─── Project paths (adjust to your environment) ─────────────────────
PROJECT_ROOT = "/gpfs/home/zlin/VideoX-Fun"
SWINIR_ROOT  = "/gpfs/home/zlin/topic/SwinIR"
for p in [PROJECT_ROOT, SWINIR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

# ─── Device ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Model Paths ────────────────────────────────────────────────────
VAE_PATH    = "/gpfs/home/zlin/VideoX-Fun/models/Diffusion_Transformer/Z-Image"
SWINIR_PATH = "/gpfs/home/zlin/topic/SwinIR/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"

# ─── Codec Parameters ───────────────────────────────────────────────
IMAGE_SIZE = 512

# Adaptive threshold (CLIP cosine similarity)
# Use --calibrate to auto-detect from your video.
THR_HIGH = 0.998  # above this → Level 1 (skip); below → Level 2

# Level 2 params
# scale=4.0 → ~29KB, scale=1.5 → ~12KB, scale=1.0 → ~8KB
L2_QUANT_SCALE    = 1.5


# ════════════════════════════════════════════════════════════════════
# Data classes for clean logging
# ════════════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    idx: int
    level: int
    tx_bytes: int
    clip_sim_prev: float       # CLIP sim with previous frame (decision metric)
    clip_score: float          # CLIP sim between original and reconstructed (eval)
    lpips: float
    ssim: float
    psnr: float


@dataclass
class VideoResult:
    frame_results: List[FrameResult] = field(default_factory=list)
    level_counts: dict = field(default_factory=lambda: {1: 0, 2: 0})

    def summary(self):
        n = len(self.frame_results)
        if n == 0:
            return "No frames processed."

        avg_bytes = np.mean([f.tx_bytes for f in self.frame_results])
        avg_clip  = np.mean([f.clip_score for f in self.frame_results])
        avg_lpips = np.mean([f.lpips for f in self.frame_results if f.lpips >= 0])
        avg_ssim  = np.mean([f.ssim for f in self.frame_results])
        avg_psnr  = np.mean([f.psnr for f in self.frame_results])

        lines = [
            f"\n{'='*70}",
            f"  Ninth Drift — Summary ({n} frames)",
            f"{'='*70}",
            f"  Level distribution:  L1(skip)={self.level_counts[1]}  "
            f"L2(VAE+SwinIR)={self.level_counts[2]}",
            f"  Avg TX size:   {avg_bytes/1024:.2f} KB/frame",
            f"  Avg CLIP Score:{avg_clip:.4f}  (semantic fidelity, higher=better)",
            f"  Avg LPIPS:     {avg_lpips:.4f}  (perceptual distance, lower=better)",
            f"  Avg SSIM:      {avg_ssim:.4f}  (structural similarity, higher=better)",
            f"  Avg PSNR:      {avg_psnr:.2f} dB  (pixel fidelity, reference only)",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# 1. Load all models
# ════════════════════════════════════════════════════════════════════

def load_models():
    """Load all required models. Returns a dict of models.
    L3 (generative) removed — only CLIP, VAE, SwinIR, LPIPS needed.
    """
    models = {}

    # ── CLIP (for decision + evaluation) ──
    print("1/4  Loading CLIP (ViT-B/32)...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True
    ).to(DEVICE).eval()
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    models["clip_model"] = clip_model
    models["clip_proc"]  = clip_proc

    # ── VAE (for Level 2 latent encoding) ──
    print("2/4  Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae").to(DEVICE).eval()
    models["vae"] = vae

    # ── SwinIR (for Level 2 post-processing) ──
    print("3/4  Loading SwinIR...")
    from network_swinir import SwinIR as SwinIRNet
    swinir = SwinIRNet(
        upscale=1, in_chans=3, img_size=128, window_size=8, img_range=1.0,
        depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
        mlp_ratio=2, upsampler='', resi_connection='1conv',
    )
    pretrained = torch.load(SWINIR_PATH, map_location='cpu')
    param_key = 'params_ema' if 'params_ema' in pretrained else 'params'
    swinir.load_state_dict(pretrained[param_key], strict=True)
    swinir = swinir.to(DEVICE).eval()
    models["swinir"] = swinir
    print(f"     SwinIR ready ({sum(p.numel() for p in swinir.parameters())/1e6:.1f}M params)")

    # ── LPIPS (for evaluation) ──
    print("4/4  Loading evaluation models...")
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(DEVICE).eval()
        models["lpips"] = lpips_model
        print("     LPIPS loaded")
    except ImportError:
        models["lpips"] = None
        print("     LPIPS not available (pip install lpips)")

    print("\nAll models loaded.\n")
    return models


# ════════════════════════════════════════════════════════════════════
# 2. CLIP utilities
# ════════════════════════════════════════════════════════════════════

def get_clip_embedding(img_np, clip_model, clip_proc):
    """Extract CLIP image embedding. Input: RGB uint8 (H,W,3) → normalized embedding."""
    pil = Image.fromarray(img_np)
    inputs = clip_proc(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # Use vision model + projection explicitly for compatibility
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        emb = clip_model.visual_projection(vision_out.pooler_output)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def clip_cosine_similarity(emb1, emb2):
    """Cosine similarity between two CLIP embeddings."""
    return (emb1 @ emb2.T).item()


# ════════════════════════════════════════════════════════════════════
# 3. Level 2: Sparse latent residual (from Seventh Drift)
# ════════════════════════════════════════════════════════════════════

def encode_to_latent(img_np, vae):
    """Encode RGB image to VAE latent."""
    t = torch.from_numpy(img_np).float().to(DEVICE) / 127.5 - 1.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(t).latent_dist.sample()
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
    return latent


def decode_from_latent(latent, vae):
    """Decode VAE latent to RGB image."""
    with torch.no_grad():
        t = vae.decode(latent / vae.config.scaling_factor + vae.config.shift_factor).sample
    t = (t / 2 + 0.5).clamp(0, 1)
    return (t.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)


def swinir_denoise(img_np, swinir_model):
    """SwinIR denoising. Input/Output: RGB uint8 (H,W,3)."""
    window_size = 8
    img_t = torch.from_numpy(img_np).float().to(DEVICE) / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    _, _, h, w = img_t.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.no_grad():
        output = swinir_model(img_t)
    output = output[:, :, :h, :w].clamp(0, 1)
    return (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════
# 4. Evaluation metrics
# ════════════════════════════════════════════════════════════════════

def calc_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def calc_ssim(img1, img2):
    return compare_ssim(img1, img2, channel_axis=2, data_range=255)


def calc_lpips(img1, img2, lpips_model):
    if lpips_model is None:
        return -1.0
    t1 = torch.from_numpy(img1).float().to(DEVICE) / 127.5 - 1.0
    t2 = torch.from_numpy(img2).float().to(DEVICE) / 127.5 - 1.0
    t1 = t1.permute(2, 0, 1).unsqueeze(0)
    t2 = t2.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        score = lpips_model(t1, t2)
    del t1, t2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score.item()


def calc_clip_score(img1_np, img2_np, clip_model, clip_proc):
    """CLIP Score between two images (cosine similarity of CLIP embeddings)."""
    emb1 = get_clip_embedding(img1_np, clip_model, clip_proc)
    emb2 = get_clip_embedding(img2_np, clip_model, clip_proc)
    score = clip_cosine_similarity(emb1, emb2)
    del emb1, emb2
    return score


# ════════════════════════════════════════════════════════════════════
# 6. Calibration: scan video to auto-detect thresholds
# ════════════════════════════════════════════════════════════════════

def calibrate_thresholds(input_video, num_frames, clip_model, clip_proc,
                         l1_target_pct=20, **kwargs):
    """
    Scan video frames, compute CLIP sim distribution, pick THR_HIGH so that
    ~l1_target_pct% of frames become L1 (skip). All other frames → L2.

    Returns thr_high.
    """
    print(f"\n--- Calibration: scanning frames ---")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Cannot open {input_video}")
        return 0.998

    sims = []
    prev_emb = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (num_frames and idx >= num_frames):
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        cur_emb = get_clip_embedding(frame_rgb, clip_model, clip_proc)
        if prev_emb is not None:
            sim = clip_cosine_similarity(prev_emb, cur_emb)
            sims.append(sim)
        prev_emb = cur_emb
        idx += 1
        if idx % 20 == 0:
            print(f"  Scanned {idx} frames...")
    cap.release()

    if len(sims) < 5:
        print("  Too few frames for calibration, using defaults.")
        return 0.998

    sims = np.array(sims)
    print(f"\n  CLIP sim stats over {len(sims)} frame pairs:")
    print(f"    min={sims.min():.6f}  max={sims.max():.6f}")
    print(f"    mean={sims.mean():.6f}  std={sims.std():.6f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{p:02d}={np.percentile(sims, p):.6f}")

    # THR_HIGH: top l1_target_pct% are L1 skip
    thr_high = float(np.percentile(sims, 100 - l1_target_pct))

    print(f"\n  Auto threshold (L1≈{l1_target_pct}%, rest → L2):")
    print(f"    THR_HIGH = {thr_high:.6f}")
    print(f"--- Calibration done ---\n")
    return thr_high


# ════════════════════════════════════════════════════════════════════
# 7. Main codec loop
# ════════════════════════════════════════════════════════════════════

def run_ninth_drift(input_video, output_video, num_frames, models):
    """Main processing loop."""

    clip_model = models["clip_model"]
    clip_proc  = models["clip_proc"]
    vae        = models["vae"]
    swinir     = models["swinir"]
    lpips_m    = models["lpips"]

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Cannot open {input_video}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 24.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (IMAGE_SIZE, IMAGE_SIZE))

    os.makedirs("output_compare", exist_ok=True)

    result = VideoResult()

    # State
    prev_clip_emb   = None
    prev_recon_img   = None   # RGB uint8

    print(f"{'='*70}")
    print(f"Ninth Drift — Two-Level Semantic Video Codec")
    print(f"  THR_HIGH={THR_HIGH} (L1 skip, everything else → L2)")
    print(f"  L1: reuse previous frame (0 KB)")
    print(f"  L2: VAE latent + quantize(scale={L2_QUANT_SCALE}) + SwinIR (~12KB)")
    print(f"  Eval: CLIP Score + LPIPS + SSIM + PSNR")
    print(f"{'='*70}\n")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (num_frames and frame_idx >= num_frames):
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))

        # ── Compute CLIP embedding ──
        cur_clip_emb = get_clip_embedding(frame_rgb, clip_model, clip_proc)

        # ── Decide transmission level ──
        # Two-level system: L1 (skip) vs L2 (full quantized latent)
        # L2 is independent per-frame, so it handles scene changes naturally.
        if prev_clip_emb is None:
            # First frame → L2
            level = 2
            clip_sim_prev = 0.0
        else:
            clip_sim_prev = clip_cosine_similarity(prev_clip_emb, cur_clip_emb)
            if clip_sim_prev > THR_HIGH:
                level = 1
            else:
                level = 2

        # ── Execute Tx/Rx pipeline based on level ──
        tx_bytes = 0

        if level == 1:
            # ── Level 1: Skip — reuse previous frame ──
            recon_img = prev_recon_img.copy()
            tx_bytes = 0

        else:
            # ── Level 2: Full quantized latent (independent per-frame) ──
            import zlib
            cur_latent = encode_to_latent(frame_rgb, vae)
            q = torch.round(cur_latent * L2_QUANT_SCALE).to(torch.int8)
            compressed = zlib.compress(q.cpu().numpy().tobytes(), level=9)
            q_recv = np.frombuffer(zlib.decompress(compressed), dtype=np.int8).copy()
            rx_latent = torch.from_numpy(q_recv).float().to(DEVICE).reshape(q.shape) / L2_QUANT_SCALE
            tx_bytes = len(compressed)

            recon_img = decode_from_latent(rx_latent, vae)
            recon_img = swinir_denoise(recon_img, swinir)

        # ── Evaluate (with memory management) ──
        clip_score = calc_clip_score(frame_rgb, recon_img, clip_model, clip_proc)
        lpips_val  = calc_lpips(frame_rgb, recon_img, lpips_m)
        ssim_val   = calc_ssim(frame_rgb, recon_img)
        psnr_val   = calc_psnr(frame_rgb, recon_img)

        fr = FrameResult(
            idx=frame_idx, level=level, tx_bytes=tx_bytes,
            clip_sim_prev=clip_sim_prev, clip_score=clip_score,
            lpips=lpips_val, ssim=ssim_val, psnr=psnr_val,
        )
        result.frame_results.append(fr)
        result.level_counts[level] += 1

        # ── Log ──
        lp_str = f"LPIPS={lpips_val:.4f}" if lpips_val >= 0 else ""
        print(f"F{frame_idx:04d} [L{level}] | "
              f"sim={clip_sim_prev:.4f} | "
              f"{tx_bytes/1024:.1f}KB | "
              f"CLIP={clip_score:.4f} | "
              f"SSIM={ssim_val:.4f} | "
              f"PSNR={psnr_val:.1f}dB | "
              f"{lp_str}")

        # ── Save comparison image (first 16 frames) ──
        if frame_idx < 16:
            compare = np.concatenate([
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR),
            ], axis=1)
            cv2.imwrite(f"output_compare/frame_{frame_idx:04d}_L{level}.jpg", compare)

        # ── Update state ──
        prev_clip_emb  = cur_clip_emb
        prev_recon_img = recon_img.copy()
        out.write(cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))
        frame_idx += 1

        # ── Per-frame GPU cleanup ──
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    cap.release()
    out.release()

    # ── Print summary ──
    print(result.summary())

    # ── Per-level breakdown ──
    for lv in [1, 2, 3]:
        lv_frames = [f for f in result.frame_results if f.level == lv]
        if not lv_frames:
            continue
        avg_clip  = np.mean([f.clip_score for f in lv_frames])
        avg_lpips = np.mean([f.lpips for f in lv_frames if f.lpips >= 0]) if any(f.lpips >= 0 for f in lv_frames) else -1
        avg_ssim  = np.mean([f.ssim for f in lv_frames])
        avg_psnr  = np.mean([f.psnr for f in lv_frames])
        avg_kb    = np.mean([f.tx_bytes for f in lv_frames]) / 1024
        print(f"  Level {lv} ({len(lv_frames)} frames): "
              f"CLIP={avg_clip:.4f}  LPIPS={avg_lpips:.4f}  "
              f"SSIM={avg_ssim:.4f}  PSNR={avg_psnr:.1f}dB  "
              f"avg={avg_kb:.2f}KB")

    # ── Save detailed results to JSON ──
    json_results = {
        "config": {
            "THR_HIGH": THR_HIGH,
            "L2_QUANT_SCALE": L2_QUANT_SCALE,
        },
        "summary": {
            "total_frames": len(result.frame_results),
            "level_counts": {str(k): v for k, v in result.level_counts.items()},
            "avg_kb_per_frame": float(np.mean([f.tx_bytes for f in result.frame_results]) / 1024),
            "avg_clip_score": float(np.mean([f.clip_score for f in result.frame_results])),
            "avg_lpips": float(np.mean([f.lpips for f in result.frame_results if f.lpips >= 0])),
            "avg_ssim": float(np.mean([f.ssim for f in result.frame_results])),
            "avg_psnr": float(np.mean([f.psnr for f in result.frame_results])),
        },
        "frames": [
            {
                "idx": f.idx, "level": f.level, "tx_bytes": f.tx_bytes,
                "clip_sim_prev": float(round(f.clip_sim_prev, 6)),
                "clip_score": float(round(f.clip_score, 6)),
                "lpips": float(round(f.lpips, 6)),
                "ssim": float(round(f.ssim, 6)),
                "psnr": float(round(f.psnr, 4)),
            }
            for f in result.frame_results
        ],
    }

    json_path = output_video.replace(".mp4", "_results.json")
    with open(json_path, "w") as fp:
        json.dump(json_results, fp, indent=2)
    print(f"\n  Detailed results saved to: {json_path}")
    print(f"  Output video: {output_video}")

    return result


# ════════════════════════════════════════════════════════════════════
# 8. Entry point
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ninth Drift — Two-Level Semantic Video Codec")
    parser.add_argument("--input",  default="test_video4.mp4", help="Input video path")
    parser.add_argument("--output", default="two_level_semantic_codec_output.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process (99999 for full video)")
    parser.add_argument("--thr_high", type=float, default=None, help="CLIP sim threshold for L1 skip")
    parser.add_argument("--calibrate", action="store_true",
                        help="Auto-detect threshold by scanning the video first")
    parser.add_argument("--l1_pct", type=int, default=20,
                        help="Target %% of frames for Level 1 skip (default: 20)")
    parser.add_argument("--l2_scale", type=float, default=None,
                        help="L2 quantization scale (1.5=~12KB default, 1.0=~8KB, 2.0=~18KB)")
    args = parser.parse_args()

    print("Ninth Drift — Two-Level Semantic Video Codec")
    print(f"  L1: skip (0KB)  |  L2: VAE+quantize+SwinIR (~10-15KB)\n")

    models = load_models()

    # Apply L2 scale if specified
    if args.l2_scale is not None:
        L2_QUANT_SCALE = args.l2_scale

    # Calibrate or use manual threshold
    if args.calibrate:
        THR_HIGH = calibrate_thresholds(
            args.input, args.frames, models["clip_model"], models["clip_proc"],
            l1_target_pct=args.l1_pct,
        )
    else:
        THR_HIGH = args.thr_high if args.thr_high is not None else THR_HIGH

    run_ninth_drift(args.input, args.output, args.frames, models)
