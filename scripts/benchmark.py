#!/usr/bin/env python3
"""
Performance Benchmark — Measure inference speed, latency, FPS, and resource usage.

Usage:
    python scripts/benchmark.py --weights models/best.pt --video samples/test.mp4
    python scripts/benchmark.py --weights models/best.pt --video samples/test.mp4 --frames 200
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_model(
    weights: str,
    video_path: str,
    device: str = "cpu",
    imgsz: int = 640,
    num_frames: int = 200,
    conf: float = 0.35,
) -> dict:
    """Run inference benchmark on a video."""
    from ultralytics import YOLO

    model = YOLO(weights)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("Video: %s (%dx%d, %.1f FPS, %d frames)",
                 video_path, width, height, source_fps, total_available)

    # Warm up
    logger.info("Warming up model...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy, device=device, verbose=False, conf=conf)

    # Benchmark
    inference_times = []
    total_times = []
    frames_processed = 0

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    logger.info("Running benchmark (%d frames)...", num_frames)

    while frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        total_start = time.perf_counter()

        # Inference
        inf_start = time.perf_counter()
        results = model.predict(
            frame, device=device, imgsz=imgsz, conf=conf, verbose=False
        )
        inf_time = (time.perf_counter() - inf_start) * 1000  # ms

        total_time = (time.perf_counter() - total_start) * 1000  # ms

        inference_times.append(inf_time)
        total_times.append(total_time)
        frames_processed += 1

        if frames_processed % 50 == 0:
            avg_fps = 1000 / statistics.mean(total_times[-50:])
            logger.info("  Frame %d/%d — avg FPS: %.1f", frames_processed, num_frames, avg_fps)

    cap.release()

    mem_after = process.memory_info().rss / 1024 / 1024

    # Calculate statistics
    results_dict = {
        "model": weights,
        "device": device,
        "imgsz": imgsz,
        "video": video_path,
        "video_resolution": f"{width}x{height}",
        "source_fps": source_fps,
        "frames_processed": frames_processed,
        "inference_time_ms": {
            "mean": statistics.mean(inference_times),
            "median": statistics.median(inference_times),
            "p95": sorted(inference_times)[int(len(inference_times) * 0.95)],
            "p99": sorted(inference_times)[int(len(inference_times) * 0.99)],
            "min": min(inference_times),
            "max": max(inference_times),
            "stdev": statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
        },
        "total_time_ms": {
            "mean": statistics.mean(total_times),
            "median": statistics.median(total_times),
            "p95": sorted(total_times)[int(len(total_times) * 0.95)],
            "p99": sorted(total_times)[int(len(total_times) * 0.99)],
        },
        "fps": {
            "mean": 1000 / statistics.mean(total_times),
            "median": 1000 / statistics.median(total_times),
            "p5": 1000 / sorted(total_times)[int(len(total_times) * 0.95)],
        },
        "memory_mb": {
            "before": mem_before,
            "after": mem_after,
            "delta": mem_after - mem_before,
        },
        "cpu_percent": psutil.cpu_percent(interval=0.1),
    }

    # GPU memory if CUDA
    if device.startswith("cuda"):
        try:
            import torch
            results_dict["gpu_memory_mb"] = {
                "allocated": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
        except Exception:
            pass

    return results_dict


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Model weights path")
    parser.add_argument("--video", type=str, required=True,
                        help="Test video path")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (default: cpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--frames", type=int, default=200,
                        help="Number of frames to benchmark")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        logger.error("Weights not found: %s", args.weights)
        sys.exit(1)

    if not Path(args.video).exists():
        logger.error("Video not found: %s", args.video)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PERFORMANCE BENCHMARK")
    logger.info("=" * 60)

    results = benchmark_model(
        weights=args.weights,
        video_path=args.video,
        device=args.device,
        imgsz=args.imgsz,
        num_frames=args.frames,
        conf=args.conf,
    )

    # Print results
    logger.info("\n--- BENCHMARK RESULTS ---")
    logger.info("Inference Time (ms):")
    for k, v in results["inference_time_ms"].items():
        logger.info("  %s: %.2f", k, v)
    logger.info("FPS:")
    for k, v in results["fps"].items():
        logger.info("  %s: %.1f", k, v)
    logger.info("Memory: %.1f MB (delta: %.1f MB)",
                 results["memory_mb"]["after"], results["memory_mb"]["delta"])

    # Save results
    output_dir = Path("artifacts/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results["benchmark_date"] = datetime.now().isoformat()

    output_path = output_dir / f"benchmark_{args.device}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved: %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
