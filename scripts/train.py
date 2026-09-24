#!/usr/bin/env python3
"""
Training Script — Fine-tune YOLO on the prepared safety dataset.

Supports smoke training (quick pipeline validation) and full training.

Usage:
    # Smoke test (2 epochs)
    python scripts/train.py --mode smoke

    # Full training
    python scripts/train.py --mode full

    # Custom settings
    python scripts/train.py --mode full --epochs 50 --batch 8 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_environment_info() -> dict:
    """Collect environment information for reproducibility."""
    import torch

    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    try:
        import ultralytics
        info["ultralytics_version"] = ultralytics.__version__
    except (ImportError, AttributeError):
        pass

    return info


def main():
    parser = argparse.ArgumentParser(description="Train YOLO safety detection model")
    parser.add_argument(
        "--mode", type=str, choices=["smoke", "full"], default="smoke",
        help="Training mode: 'smoke' (2 epochs) or 'full' (default: smoke)"
    )
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Training config YAML")
    parser.add_argument("--data", type=str, default=None,
                        help="Override data.yaml path")
    parser.add_argument("--model", type=str, default=None,
                        help="Override base model weights")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, mps")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()

    # Load training config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning("Config not found: %s — using defaults", config_path)
        cfg = {}

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    repro_cfg = cfg.get("reproducibility", {})
    output_cfg = cfg.get("output", {})

    # Select mode-specific parameters
    mode = args.mode
    if mode == "smoke":
        mode_params = train_cfg.get("smoke", {"epochs": 2, "batch": 8, "patience": 2})
    else:
        mode_params = train_cfg.get("full", {"epochs": 100, "batch": 16, "patience": 10})

    # Apply overrides
    data_yaml = args.data or data_cfg.get("yaml_path", "data/processed/data.yaml")
    base_model = args.model or model_cfg.get("base_weights", "yolov8n.pt")
    epochs = args.epochs or mode_params.get("epochs", 100)
    batch = args.batch or mode_params.get("batch", 16)
    imgsz = args.imgsz or model_cfg.get("imgsz", 640)
    device = args.device or cfg.get("device")
    name = args.name or f"{output_cfg.get('name', 'safety_train')}_{mode}"

    # Verify data.yaml exists
    if not Path(data_yaml).exists():
        logger.error(
            "Data YAML not found: %s\n"
            "Run dataset preparation first:\n"
            "  python scripts/prepare_dataset.py",
            data_yaml,
        )
        sys.exit(1)

    # Collect environment info
    env_info = get_environment_info()
    logger.info("Environment: %s", json.dumps(env_info, indent=2))

    # Auto-detect device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info("Using device: %s", device)

    logger.info("=" * 60)
    logger.info("TRAINING — Mode: %s", mode.upper())
    logger.info("  Data: %s", data_yaml)
    logger.info("  Base model: %s", base_model)
    logger.info("  Epochs: %d", epochs)
    logger.info("  Batch: %d", batch)
    logger.info("  Image size: %d", imgsz)
    logger.info("  Device: %s", device)
    logger.info("=" * 60)

    # Import and train
    from ultralytics import YOLO

    model = YOLO(base_model)

    # Set seed
    seed = repro_cfg.get("seed", 42)

    # Build training kwargs
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "patience": mode_params.get("patience", 10),
        "seed": seed,
        "deterministic": repro_cfg.get("deterministic", True),
        "project": output_cfg.get("project", "runs"),
        "name": name,
        "exist_ok": output_cfg.get("exist_ok", True),
        "verbose": True,
        "plots": True,
        "save": True,
        "val": True,
    }

    # Add training hyperparameters
    for key in ["optimizer", "lr0", "lrf", "momentum", "weight_decay",
                "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
                "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v",
                "degrees", "translate", "scale", "fliplr", "mosaic",
                "mixup", "close_mosaic"]:
        if key in train_cfg:
            train_kwargs[key] = train_cfg[key]

    logger.info("Starting training with: %s", {k: v for k, v in train_kwargs.items() if k != "data"})

    results = model.train(**train_kwargs)

    # Save training record
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    record = {
        "training_date": datetime.now().isoformat(),
        "mode": mode,
        "config": {
            "data": data_yaml,
            "base_model": base_model,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "seed": seed,
        },
        "environment": env_info,
        "results_dir": str(results.save_dir) if hasattr(results, "save_dir") else None,
    }

    record_path = reports_dir / f"training_record_{mode}.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info("Training record saved: %s", record_path)

    # Copy best weights to models/
    if hasattr(results, "save_dir"):
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            dest = Path("models") / "best.pt"
            dest.parent.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(best_pt, dest)
            logger.info("Best weights copied to %s", dest)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
