#!/usr/bin/env python3
"""
Compute packet-loss robustness metrics: PSNR, SSIM, CLIP similarity, and YOLO F1.

This script is designed for the project HPC layout:
  /gpfs/home/zlin/topic/
    test_video4.mp4
    test_video_vr.mp4
    robustness_results/videos/*.mp4

Metrics:
  - PSNR and SSIM are computed by ffmpeg after scaling compared videos to the
    reference resolution.
  - CLIP similarity is computed frame-by-frame on sampled frames, then averaged.
  - YOLO F1 is computed on sampled regular-video frames by treating detections
    on the original video as pseudo-ground-truth. This is not a human-labelled
    detection benchmark; it measures whether object-level detections are
    preserved after packet loss.

Dependencies:
  - ffmpeg and ffprobe on PATH
  - Python packages for CLIP:
      option A: open_clip_torch, torch, pillow
      option B: transformers, torch, pillow
  - Python packages for YOLO:
      ultralytics, torch, pillow

Example:
  python /gpfs/home/zlin/topic/compute_packet_loss_all_metrics.py \
    --sample-stride 10 \
    --yolo-classes 2 3 5 7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int


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


def ffprobe_info(video_path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr}")

    stream = json.loads(result.stdout)["streams"][0]
    frame_count_text = stream.get("nb_frames")
    if frame_count_text and frame_count_text != "N/A":
        frame_count = int(frame_count_text)
    else:
        fps_text = stream.get("avg_frame_rate", "30/1")
        if "/" in fps_text:
            num, den = fps_text.split("/", 1)
            fps = float(num) / float(den)
        else:
            fps = float(fps_text)
        frame_count = int(round(float(stream.get("duration", 0.0)) * fps))

    return VideoInfo(width=int(stream["width"]), height=int(stream["height"]), frame_count=frame_count)


def parse_quality_metric(stderr: str, metric: str) -> float:
    if metric == "psnr":
        matches = re.findall(r"average:([0-9.]+|inf)", stderr)
    elif metric == "ssim":
        matches = re.findall(r"All:([0-9.]+)", stderr)
    else:
        raise ValueError(metric)

    if not matches:
        raise RuntimeError(f"Could not parse {metric} from ffmpeg output:\n{stderr[-3000:]}")

    value = matches[-1]
    if value == "inf":
        return float("inf")
    return float(value)


def compute_ffmpeg_quality(reference: Path, distorted: Path, metric: str, stats_path: Path) -> float:
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
        raise RuntimeError(f"ffmpeg {metric} failed for {distorted}:\n{result.stderr}")
    return parse_quality_metric(result.stderr, metric)


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
        raise RuntimeError(f"No packet-loss videos found in {videos_dir}")
    return items


def extract_sampled_frames(video: Path, output_dir: Path, width: int, height: int, stride: int, max_frames: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vf = f"select='not(mod(n\\,{stride}))',scale={width}:{height}:flags=bicubic"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-frames:v",
        str(max_frames),
        str(output_dir / "frame_%06d.png"),
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed for {video}:\n{result.stderr}")


def sorted_frames(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("frame_*.png"))


def load_clip_backend(backend: str, model_name: str, pretrained: str, device: str):
    if backend in {"auto", "open_clip"}:
        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            model = model.to(device).eval()
            return "open_clip", model, preprocess, torch
        except Exception as exc:
            if backend == "open_clip":
                raise RuntimeError(f"open_clip backend failed: {exc}") from exc

    if backend in {"auto", "transformers"}:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(model_name).to(device).eval()
            processor = CLIPProcessor.from_pretrained(model_name)
            return "transformers", model, processor, torch
        except Exception as exc:
            if backend == "transformers":
                raise RuntimeError(f"transformers CLIP backend failed: {exc}") from exc

    raise RuntimeError(
        "Could not load a CLIP backend. Install open_clip_torch or transformers with cached CLIP weights."
    )


def compute_clip_similarity(
    ref_frames: list[Path],
    dist_frames: list[Path],
    backend_name: str,
    model,
    preprocess,
    torch_module,
    device: str,
) -> float:
    from PIL import Image

    count = min(len(ref_frames), len(dist_frames))
    if count == 0:
        return float("nan")

    similarities: list[float] = []
    with torch_module.no_grad():
        for ref_path, dist_path in zip(ref_frames[:count], dist_frames[:count]):
            ref_image = Image.open(ref_path).convert("RGB")
            dist_image = Image.open(dist_path).convert("RGB")
            if backend_name == "open_clip":
                ref_tensor = preprocess(ref_image).unsqueeze(0).to(device)
                dist_tensor = preprocess(dist_image).unsqueeze(0).to(device)
                ref_feat = model.encode_image(ref_tensor)
                dist_feat = model.encode_image(dist_tensor)
            else:
                ref_inputs = preprocess(images=ref_image, return_tensors="pt")
                dist_inputs = preprocess(images=dist_image, return_tensors="pt")
                ref_inputs = {k: v.to(device) for k, v in ref_inputs.items()}
                dist_inputs = {k: v.to(device) for k, v in dist_inputs.items()}
                ref_feat = model.get_image_features(**ref_inputs)
                dist_feat = model.get_image_features(**dist_inputs)

            ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
            dist_feat = dist_feat / dist_feat.norm(dim=-1, keepdim=True)
            similarities.append(float((ref_feat * dist_feat).sum(dim=-1).item()))

    return sum(similarities) / len(similarities)


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def detections_from_result(result, allowed_classes: set[int], conf_threshold: float) -> list[tuple[int, float, list[float]]]:
    detections: list[tuple[int, float, list[float]]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in allowed_classes or conf < conf_threshold:
            continue
        xyxy = [float(x) for x in box.xyxy[0].tolist()]
        detections.append((cls_id, conf, xyxy))
    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def match_detections(
    ref_dets: list[tuple[int, float, list[float]]],
    dist_dets: list[tuple[int, float, list[float]]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    matched_ref: set[int] = set()
    tp = 0
    for cls_id, _conf, box in dist_dets:
        best_index = None
        best_iou = 0.0
        for index, (ref_cls, _ref_conf, ref_box) in enumerate(ref_dets):
            if index in matched_ref or ref_cls != cls_id:
                continue
            score = iou_xyxy(box, ref_box)
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matched_ref.add(best_index)
            tp += 1
    fp = len(dist_dets) - tp
    fn = len(ref_dets) - tp
    return tp, fp, fn


def compute_yolo_pseudo_f1(
    ref_frames: list[Path],
    dist_frames: list[Path],
    model_name: str,
    allowed_classes: set[int],
    conf_threshold: float,
    iou_threshold: float,
    device: str,
) -> tuple[float, float, float]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("YOLO metric requires the ultralytics package.") from exc

    model = YOLO(model_name)
    count = min(len(ref_frames), len(dist_frames))
    tp_total = fp_total = fn_total = 0

    for ref_path, dist_path in zip(ref_frames[:count], dist_frames[:count]):
        ref_result = model.predict(str(ref_path), verbose=False, conf=conf_threshold, device=device)[0]
        dist_result = model.predict(str(dist_path), verbose=False, conf=conf_threshold, device=device)[0]
        ref_dets = detections_from_result(ref_result, allowed_classes, conf_threshold)
        dist_dets = detections_from_result(dist_result, allowed_classes, conf_threshold)
        tp, fp, fn = match_detections(ref_dets, dist_dets, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=script_dir)
    parser.add_argument("--videos-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-backend", default="auto", choices=["auto", "open_clip", "transformers"])
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--transformers-clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-classes", nargs="+", type=int, default=[2, 3, 5, 7])
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.50)
    args = parser.parse_args()

    require_tool("ffmpeg")
    require_tool("ffprobe")

    root = args.root.resolve()
    videos_dir = (args.videos_dir or (root / "robustness_results" / "videos")).resolve()
    output = (args.output or (root / "robustness_results" / "packet_loss_all_metrics.csv")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    items = discover_items(root, videos_dir)
    reference_infos = {
        dataset: ffprobe_info(path)
        for dataset, path in {
            "regular": root / "test_video4.mp4",
            "vr": root / "test_video_vr.mp4",
        }.items()
    }

    clip_state = None
    if not args.skip_clip:
        clip_model_name = args.clip_model
        clip_pretrained = args.clip_pretrained
        if args.clip_backend == "transformers":
            clip_model_name = args.transformers_clip_model
        print("Loading CLIP model...")
        clip_state = load_clip_backend(args.clip_backend, clip_model_name, clip_pretrained, args.device)
        print(f"Loaded CLIP backend: {clip_state[0]}")

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        for item in items:
            print(f"Evaluating {item.dataset} {item.method} loss={item.loss_rate}%")
            info = reference_infos[item.dataset]
            psnr = compute_ffmpeg_quality(item.reference, item.distorted, "psnr", temp_dir / "psnr.log")
            ssim = compute_ffmpeg_quality(item.reference, item.distorted, "ssim", temp_dir / "ssim.log")

            ref_dir = temp_dir / f"{item.dataset}_{item.method}_{item.loss_rate}_ref"
            dist_dir = temp_dir / f"{item.dataset}_{item.method}_{item.loss_rate}_dist"
            extract_sampled_frames(item.reference, ref_dir, info.width, info.height, args.sample_stride, args.max_frames)
            extract_sampled_frames(item.distorted, dist_dir, info.width, info.height, args.sample_stride, args.max_frames)
            ref_frames = sorted_frames(ref_dir)
            dist_frames = sorted_frames(dist_dir)
            sampled_frames = min(len(ref_frames), len(dist_frames))

            clip_similarity = float("nan")
            if clip_state is not None:
                clip_similarity = compute_clip_similarity(
                    ref_frames,
                    dist_frames,
                    clip_state[0],
                    clip_state[1],
                    clip_state[2],
                    clip_state[3],
                    args.device,
                )

            yolo_precision = yolo_recall = yolo_f1 = float("nan")
            if not args.skip_yolo and item.dataset == "regular":
                yolo_precision, yolo_recall, yolo_f1 = compute_yolo_pseudo_f1(
                    ref_frames,
                    dist_frames,
                    args.yolo_model,
                    set(args.yolo_classes),
                    args.yolo_conf,
                    args.yolo_iou,
                    args.device,
                )

            rows.append(
                {
                    "dataset": item.dataset,
                    "method": item.method,
                    "loss_rate_percent": item.loss_rate,
                    "psnr": psnr,
                    "ssim": ssim,
                    "clip_similarity": clip_similarity,
                    "yolo_precision": yolo_precision,
                    "yolo_recall": yolo_recall,
                    "yolo_f1": yolo_f1,
                    "sampled_frames": sampled_frames,
                    "reference_video": str(item.reference),
                    "distorted_video": str(item.distorted),
                }
            )

    rows.sort(key=lambda row: (str(row["dataset"]), int(row["loss_rate_percent"]), str(row["method"])))
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "method",
            "loss_rate_percent",
            "psnr",
            "ssim",
            "clip_similarity",
            "yolo_precision",
            "yolo_recall",
            "yolo_f1",
            "sampled_frames",
            "reference_video",
            "distorted_video",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format_float(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    print(f"Done. Metrics written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
