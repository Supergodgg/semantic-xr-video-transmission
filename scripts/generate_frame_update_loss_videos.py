#!/usr/bin/env python3
"""
Generate frame-level update-loss videos for robustness experiments.

The script uses the same loss mask for all compared methods at a given loss
rate. Lost frames are replaced by the most recent received frame
(previous-frame copy), which matches a simple low-delay concealment model.

Dependencies:
  - Python 3.8+
  - ffmpeg and ffprobe available on PATH

Example:
  python /gpfs/home/zlin/topic/generate_frame_update_loss_videos.py \
    --dataset regular \
    --methods ours=lrc_laq_semantic_codec_regular_s15_output.mp4 h264=v4_h264_200k.mp4 vp9=v4_vp9_200k.mp4 \
    --loss-rates 0 10 20 30 \
    --seed 42

By default, input videos are read from the folder containing this script. Output
is written to a robustness_results folder next to this script. This is intended
for HPC usage where the script and videos are placed in the same topic folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps_expr: str
    fps_float: float
    frame_count: int


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool not found on PATH: {name}")


def available_ffmpeg_encoders() -> set[str]:
    result = run_command(["ffmpeg", "-hide_banner", "-encoders"])
    if result.returncode != 0:
        raise RuntimeError(f"Could not list ffmpeg encoders:\n{result.stderr}")
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def choose_encoder(requested: str) -> str:
    if requested != "auto":
        return requested

    encoders = available_ffmpeg_encoders()
    for candidate in ("libx264", "libopenh264", "mpeg4"):
        if candidate in encoders:
            return candidate

    raise RuntimeError("No supported MP4 video encoder found. Tried libx264, libopenh264, and mpeg4.")


def parse_fps(expr: str) -> float:
    if "/" in expr:
        numerator, denominator = expr.split("/", 1)
        return float(numerator) / float(denominator)
    return float(expr)


def ffprobe_info(video_path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")

    stream = streams[0]
    fps_expr = stream.get("avg_frame_rate") or "30/1"
    fps_float = parse_fps(fps_expr)
    frame_count_text = stream.get("nb_frames")
    if frame_count_text and frame_count_text != "N/A":
        frame_count = int(frame_count_text)
    else:
        duration = float(stream.get("duration", 0.0))
        frame_count = int(round(duration * fps_float))

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_expr=fps_expr,
        fps_float=fps_float,
        frame_count=frame_count,
    )


def extract_frames(video_path: Path, frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        str(frame_dir / "frame_%06d.png"),
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed for {video_path}:\n{result.stderr}")


def encode_frames(
    frame_dir: Path,
    output_video: Path,
    fps_expr: str,
    encoder: str,
    crf: int,
    preset: str,
    bitrate: str,
    qscale: int,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        fps_expr,
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
    ]

    if encoder == "libx264":
        command.extend(["-crf", str(crf), "-preset", preset])
    elif encoder == "libopenh264":
        command.extend(["-b:v", bitrate])
    elif encoder == "mpeg4":
        command.extend(["-q:v", str(qscale)])
    else:
        command.extend(["-b:v", bitrate])

    command.append(str(output_video))
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed for {output_video}:\n{result.stderr}")


def make_loss_indices(frame_count: int, loss_rate_percent: int, seed: int) -> set[int]:
    if loss_rate_percent <= 0:
        return set()
    if loss_rate_percent >= 100:
        raise ValueError("loss rate must be below 100%")

    rng = random.Random(seed + loss_rate_percent * 1009)
    candidates = list(range(1, frame_count + 1))
    if frame_count > 1:
        candidates.remove(1)
    lost_count = round(frame_count * loss_rate_percent / 100.0)
    lost_count = min(lost_count, len(candidates))
    return set(rng.sample(candidates, lost_count))


def apply_previous_frame_copy(source_dir: Path, target_dir: Path, frame_count: int, lost: set[int]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    previous_received = source_dir / "frame_000001.png"
    for index in range(1, frame_count + 1):
        source = source_dir / f"frame_{index:06d}.png"
        target = target_dir / f"frame_{index:06d}.png"
        if not source.exists():
            raise RuntimeError(f"Expected extracted frame not found: {source}")

        if index in lost:
            shutil.copy2(previous_received, target)
        else:
            shutil.copy2(source, target)
            previous_received = source


def write_mask_csv(mask_path: Path, frame_count: int, lost: set[int]) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    with mask_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "lost"])
        for index in range(1, frame_count + 1):
            writer.writerow([index, int(index in lost)])


def parse_methods(items: list[str], input_dir: Path) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Method must use name=file format: {item}")
        name, filename = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty method name in: {item}")
        path = Path(filename)
        if not path.is_absolute():
            path = input_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Video not found for method {name}: {path}")
        methods[name] = path
    return methods


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir,
        help="Directory containing input videos. Default: folder containing this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "robustness_results",
        help="Directory for generated videos, masks, and summaries. Default: script folder/robustness_results.",
    )
    parser.add_argument("--dataset", default="regular", help="Label used in output filenames.")
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Compared videos in name=file format, e.g. ours=a.mp4 h264=b.mp4.",
    )
    parser.add_argument("--loss-rates", nargs="+", type=int, default=[0, 10, 20, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encoder",
        default="auto",
        help="Output video encoder. Default: auto, trying libx264, libopenh264, then mpeg4.",
    )
    parser.add_argument("--crf", type=int, default=18, help="CRF used when encoder is libx264.")
    parser.add_argument("--bitrate", default="5M", help="Bitrate used when encoder is libopenh264 or another bitrate encoder.")
    parser.add_argument("--qscale", type=int, default=2, help="Quality scale used when encoder is mpeg4.")
    parser.add_argument("--preset", default="medium")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep temporary extracted/concealed PNG frames under output-dir/frames.",
    )
    args = parser.parse_args()

    require_tool("ffmpeg")
    require_tool("ffprobe")
    encoder = choose_encoder(args.encoder)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    methods = parse_methods(args.methods, input_dir)
    print(f"Using ffmpeg encoder: {encoder}")

    infos = {name: ffprobe_info(path) for name, path in methods.items()}
    reference_name = next(iter(infos))
    reference = infos[reference_name]
    for name, info in infos.items():
        if info.frame_count != reference.frame_count:
            raise RuntimeError(
                f"Frame-count mismatch: {reference_name} has {reference.frame_count}, "
                f"but {name} has {info.frame_count}. Use aligned videos before running this script."
            )
        if (info.width, info.height) != (reference.width, reference.height):
            print(
                f"Warning: resolution differs for {name}: {info.width}x{info.height}; "
                f"reference is {reference.width}x{reference.height}.",
                file=sys.stderr,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    videos_dir = output_dir / "videos"
    summary_path = output_dir / f"{args.dataset}_frame_update_loss_summary.csv"

    frame_root_context = tempfile.TemporaryDirectory()
    if args.keep_frames:
        persistent_frame_root = output_dir / "frames"
        persistent_frame_root.mkdir(parents=True, exist_ok=True)
        frame_root = persistent_frame_root
        temp_context = None
    else:
        temp_context = frame_root_context
        frame_root = Path(temp_context.name)

    try:
        extracted_dirs: dict[str, Path] = {}
        for name, path in methods.items():
            extracted_dir = frame_root / "extracted" / name
            print(f"Extracting frames for {name}: {path}")
            extract_frames(path, extracted_dir)
            extracted_dirs[name] = extracted_dir

        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "dataset",
                    "method",
                    "source_video",
                    "loss_rate_percent",
                    "frame_count",
                    "lost_frames",
                    "output_video",
                    "mask_csv",
                    "width",
                    "height",
                    "fps",
                ]
            )

            for loss_rate in args.loss_rates:
                lost = make_loss_indices(reference.frame_count, loss_rate, args.seed)
                mask_path = masks_dir / f"{args.dataset}_loss_{loss_rate}.csv"
                write_mask_csv(mask_path, reference.frame_count, lost)

                for name, path in methods.items():
                    concealed_dir = frame_root / "concealed" / f"{args.dataset}_{name}_loss_{loss_rate}"
                    output_video = videos_dir / f"{args.dataset}_{name}_loss_{loss_rate}.mp4"
                    print(f"Generating {output_video}")
                    apply_previous_frame_copy(extracted_dirs[name], concealed_dir, reference.frame_count, lost)
                    encode_frames(
                        concealed_dir,
                        output_video,
                        reference.fps_expr,
                        encoder,
                        args.crf,
                        args.preset,
                        args.bitrate,
                        args.qscale,
                    )
                    writer.writerow(
                        [
                            args.dataset,
                            name,
                            str(path),
                            loss_rate,
                            reference.frame_count,
                            len(lost),
                            str(output_video),
                            str(mask_path),
                            reference.width,
                            reference.height,
                            reference.fps_expr,
                        ]
                    )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    print(f"Done. Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
