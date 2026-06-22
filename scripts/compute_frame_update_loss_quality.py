#!/usr/bin/env python3
"""
Compute PSNR and SSIM for frame-level update-loss robustness videos.

The script is designed for the HPC folder layout used in this project:
  /gpfs/home/zlin/topic/
    test_video4.mp4
    test_video_vr.mp4
    robustness_results/videos/*.mp4

It compares each loss video against the corresponding original video and writes
a CSV summary. Compared videos are scaled to the reference resolution inside
ffmpeg so that methods with different output sizes can still be evaluated.

Dependencies:
  - Python 3.8+
  - ffmpeg available on PATH

Example:
  python /gpfs/home/zlin/topic/compute_frame_update_loss_quality.py
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationItem:
    dataset: str
    method: str
    loss_rate: int
    reference: Path
    distorted: Path


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


def parse_metric(stderr: str, metric: str) -> float:
    if metric == "psnr":
        matches = re.findall(r"average:([0-9.]+|inf)", stderr)
    elif metric == "ssim":
        matches = re.findall(r"All:([0-9.]+)", stderr)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if not matches:
        raise RuntimeError(f"Could not parse {metric} from ffmpeg output:\n{stderr[-3000:]}")

    value = matches[-1]
    if value == "inf":
        return float("inf")
    return float(value)


def compute_ffmpeg_metric(reference: Path, distorted: Path, metric: str, stats_path: Path) -> float:
    if metric not in {"psnr", "ssim"}:
        raise ValueError(f"Unsupported metric: {metric}")

    filter_graph = (
        f"[1:v][0:v]scale2ref=flags=bicubic[dist][ref];"
        f"[ref]setsar=1[ref_sar];"
        f"[dist]setsar=1[dist_sar];"
        f"[ref_sar][dist_sar]{metric}=stats_file='{stats_path.as_posix()}'"
    )
    command = [
        "ffmpeg",
        "-v",
        "info",
        "-i",
        str(reference),
        "-i",
        str(distorted),
        "-lavfi",
        filter_graph,
        "-f",
        "null",
        "-",
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg {metric} failed for {distorted} against {reference}:\n{result.stderr}"
        )
    return parse_metric(result.stderr, metric)


def discover_items(root: Path, videos_dir: Path) -> list[EvaluationItem]:
    references = {
        "regular": root / "test_video4.mp4",
        "vr": root / "test_video_vr.mp4",
    }
    for dataset, path in references.items():
        if not path.exists():
            raise FileNotFoundError(f"Reference video for {dataset} not found: {path}")

    pattern = re.compile(r"^(regular|vr)_(ours|h264|vp9)_loss_(0|10|20|30)\.mp4$")
    items: list[EvaluationItem] = []
    for video in sorted(videos_dir.glob("*.mp4")):
        match = pattern.match(video.name)
        if not match:
            continue
        dataset, method, loss_rate = match.groups()
        items.append(
            EvaluationItem(
                dataset=dataset,
                method=method,
                loss_rate=int(loss_rate),
                reference=references[dataset],
                distorted=video,
            )
        )

    if not items:
        raise RuntimeError(f"No frame-update-loss videos found in {videos_dir}")
    return items


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir,
        help="Folder containing original videos and robustness_results. Default: script folder.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Folder containing generated frame-update-loss videos. Default: root/robustness_results/videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Default: root/robustness_results/frame_update_loss_quality_metrics.csv.",
    )
    args = parser.parse_args()

    require_tool("ffmpeg")

    root = args.root.resolve()
    videos_dir = (args.videos_dir or (root / "robustness_results" / "videos")).resolve()
    output = (args.output or (root / "robustness_results" / "frame_update_loss_quality_metrics.csv")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    items = discover_items(root, videos_dir)
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        for item in items:
            print(f"Evaluating {item.dataset} {item.method} loss={item.loss_rate}%")
            psnr_stats = temp_dir / f"{item.dataset}_{item.method}_{item.loss_rate}_psnr.log"
            ssim_stats = temp_dir / f"{item.dataset}_{item.method}_{item.loss_rate}_ssim.log"
            psnr = compute_ffmpeg_metric(item.reference, item.distorted, "psnr", psnr_stats)
            ssim = compute_ffmpeg_metric(item.reference, item.distorted, "ssim", ssim_stats)
            rows.append(
                {
                    "dataset": item.dataset,
                    "method": item.method,
                    "loss_rate_percent": item.loss_rate,
                    "psnr": psnr,
                    "ssim": ssim,
                    "reference_video": str(item.reference),
                    "distorted_video": str(item.distorted),
                }
            )

    rows.sort(key=lambda row: (str(row["dataset"]), int(row["loss_rate_percent"]), str(row["method"])))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "loss_rate_percent",
                "psnr",
                "ssim",
                "reference_video",
                "distorted_video",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Metrics written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
