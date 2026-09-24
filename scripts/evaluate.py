#!/usr/bin/env python3
"""
Evaluation Script — Evaluate trained YOLO model on validation/test splits.

Usage:
    python scripts/evaluate.py --weights models/best.pt --data data/processed/data.yaml
    python scripts/evaluate.py --weights models/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Model weights path")
    parser.add_argument("--data", type=str, default="data/processed/data.yaml",
                        help="Data YAML path")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val",
                        help="Evaluation split (default: val)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        logger.error("Weights not found: %s", args.weights)
        sys.exit(1)

    if not Path(args.data).exists():
        logger.error("Data YAML not found: %s", args.data)
        sys.exit(1)

    # Auto-detect device
    device = args.device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("  Weights: %s", args.weights)
    logger.info("  Data: %s", args.data)
    logger.info("  Split: %s", args.split)
    logger.info("  Device: %s", device)
    logger.info("=" * 60)

    from ultralytics import YOLO

    model = YOLO(args.weights)

    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=device,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        plots=True,
        verbose=True,
    )

    # Extract metrics
    metrics = {}
    try:
        metrics = {
            "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else None,
            "mAP50_95": float(results.box.map) if hasattr(results.box, "map") else None,
            "precision": float(results.box.mp) if hasattr(results.box, "mp") else None,
            "recall": float(results.box.mr) if hasattr(results.box, "mr") else None,
        }

        # Per-class metrics
        if hasattr(results.box, "ap50") and results.box.ap50 is not None:
            class_names = results.names if hasattr(results, "names") else {}
            per_class = {}
            for i, ap50 in enumerate(results.box.ap50):
                name = class_names.get(i, f"class_{i}")
                per_class[name] = {
                    "ap50": float(ap50),
                    "ap50_95": float(results.box.ap[i]) if hasattr(results.box, "ap") else None,
                }
                if hasattr(results.box, "p") and len(results.box.p) > i:
                    per_class[name]["precision"] = float(results.box.p[i])
                if hasattr(results.box, "r") and len(results.box.r) > i:
                    per_class[name]["recall"] = float(results.box.r[i])
            metrics["per_class"] = per_class

    except Exception as e:
        logger.warning("Could not extract all metrics: %s", e)

    logger.info("Results:")
    for k, v in metrics.items():
        if k != "per_class":
            logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    if "per_class" in metrics:
        logger.info("  Per-class:")
        for name, vals in metrics["per_class"].items():
            logger.info("    %s: AP50=%.4f", name, vals.get("ap50", 0))

    # Save evaluation report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    report = {
        "evaluation_date": datetime.now().isoformat(),
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "device": device,
        "metrics": metrics,
        "results_dir": str(results.save_dir) if hasattr(results, "save_dir") else None,
    }

    report_path = reports_dir / f"evaluation_{args.split}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Evaluation report: %s", report_path)
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
