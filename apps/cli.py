#!/usr/bin/env python3
"""
CLI Demo — OpenCV-based real-time safety monitoring.

The authoritative low-latency demo for webcam, video file, and RTSP input.

Usage:
    # Webcam
    python -m apps.cli --source 0 --config configs/app.yaml

    # Video file
    python -m apps.cli --source samples/construction.mp4 --save-output

    # RTSP (optional)
    python -m apps.cli --source "rtsp://..."

Controls:
    q         - Quit
    p / space - Pause/Resume (file input only)
    r         - Reset tracker and rules
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.safety_monitor.config import load_config
from src.safety_monitor.pipeline import SafetyMonitorPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("safety_monitor.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision Safety Monitor — CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  q         Quit
  p/space   Pause/Resume (file input)
  r         Reset tracker and rules

Examples:
  python -m apps.cli --source 0
  python -m apps.cli --source video.mp4 --save-output
  python -m apps.cli --source "rtsp://..." --config configs/app.yaml
        """,
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: camera index (0), file path, or RTSP URL (default: 0)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/app.yaml",
        help="Path to app configuration YAML (default: configs/app.yaml)"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Override model weights path"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override confidence threshold"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Override input image size"
    )
    parser.add_argument(
        "--save-output", action="store_true",
        help="Save output video"
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Custom output video path"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV display window"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config overrides from CLI args
    overrides = {}
    if args.weights:
        overrides.setdefault("model", {})["weights"] = args.weights
    if args.device:
        overrides.setdefault("model", {})["device"] = args.device
    if args.conf:
        overrides.setdefault("model", {})["conf_threshold"] = args.conf
    if args.imgsz:
        overrides.setdefault("model", {})["imgsz"] = args.imgsz

    # Load configuration
    config = load_config(args.config, overrides)

    logger.info("=" * 60)
    logger.info("VISION SAFETY MONITOR — CLI Demo")
    logger.info("=" * 60)
    logger.info("  Source: %s", args.source)
    logger.info("  Model: %s", config.model.weights)
    logger.info("  Device: %s", config.model.device)
    logger.info("  Confidence: %.2f", config.model.conf_threshold)
    logger.info("  Target classes: %s", config.classes.names)
    logger.info("=" * 60)

    # Create and initialize pipeline
    pipeline = SafetyMonitorPipeline(config)

    try:
        pipeline.initialize()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "To get started, copy the pre-trained weights:\n"
            "  cp dataset/results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt models/best.pt"
        )
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to initialize pipeline: %s", e)
        sys.exit(1)

    # Run
    summary = pipeline.run_video(
        source=args.source,
        show_display=not args.no_display,
        save_output=args.save_output,
        output_path=args.output_path,
    )

    # Print summary
    logger.info("Summary: %s", summary)

    return summary


if __name__ == "__main__":
    main()
