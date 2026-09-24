#!/usr/bin/env python3
"""
Model Export Script — Export YOLO model to ONNX format.

Usage:
    python scripts/export_model.py --weights models/best.pt --format onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Model weights path")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "torchscript", "engine"],
                        help="Export format (default: onnx)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--half", action="store_true", help="FP16 export")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch size")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        logger.error("Weights not found: %s", args.weights)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("MODEL EXPORT")
    logger.info("  Weights: %s", args.weights)
    logger.info("  Format: %s", args.format)
    logger.info("  Image size: %d", args.imgsz)
    logger.info("  FP16: %s", args.half)
    logger.info("=" * 60)

    from ultralytics import YOLO

    model = YOLO(args.weights)

    try:
        export_path = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            simplify=args.simplify,
        )
        logger.info("Model exported to: %s", export_path)

        # Copy to models/
        import shutil
        dest = Path("models") / Path(export_path).name
        shutil.copy2(export_path, dest)
        logger.info("Copied to: %s", dest)

    except Exception as e:
        logger.error("Export failed: %s", e)
        logger.info(
            "Note: Some export formats require additional dependencies.\n"
            "For ONNX: pip install onnx onnxruntime\n"
            "For TensorRT: requires NVIDIA GPU + TensorRT SDK"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
