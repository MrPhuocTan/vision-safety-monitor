#!/usr/bin/env python3
"""
Dataset Preparation — Validate, filter, remap, and prepare dataset for training.

Reads from the inspected source dataset (default: dataset/css-data/),
filters to MVP target classes, remaps class IDs, and writes a clean
YOLO dataset to data/processed/.

Usage:
    python scripts/prepare_dataset.py [--source dataset/css-data] [--output data/processed]

Outputs:
    data/processed/train/images/, data/processed/train/labels/
    data/processed/valid/images/, data/processed/valid/labels/
    data/processed/test/images/,  data/processed/test/labels/
    data/processed/data.yaml
    reports/dataset_summary.json
    reports/dataset_summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Original class mapping (from the dataset's ppe_data.yaml)
ORIGINAL_CLASSES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}

# MVP target classes and their new IDs
MVP_CLASS_REMAP = {
    0: 0,   # Hardhat → 0
    2: 1,   # NO-Hardhat → 1
    5: 2,   # Person → 2
}

MVP_CLASS_NAMES = {
    0: "Hardhat",
    1: "NO-Hardhat",
    2: "Person",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def validate_yolo_line(line: str) -> tuple[bool, int | None, list[float] | None]:
    """
    Validate a single YOLO annotation line.

    Returns:
        (is_valid, class_id, [cx, cy, w, h]) or (False, None, None)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return False, None, None

    try:
        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:5]]

        # Check normalized bounds (allow slight over due to augmentation artifacts)
        for c in coords:
            if c < -0.01 or c > 1.01:
                return False, None, None

        # Clamp to [0, 1]
        coords = [max(0.0, min(1.0, c)) for c in coords]

        # Check w, h > 0
        if coords[2] <= 0 or coords[3] <= 0:
            return False, None, None

        return True, class_id, coords

    except (ValueError, IndexError):
        return False, None, None


def process_label_file(
    label_path: Path,
    class_remap: dict[int, int],
) -> tuple[list[str], Counter, int, int]:
    """
    Process a single label file: filter classes, remap IDs, validate.

    Returns:
        (new_lines, class_counts, total_original, skipped)
    """
    new_lines = []
    class_counts = Counter()
    total_original = 0
    skipped = 0

    try:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                total_original += 1
                valid, class_id, coords = validate_yolo_line(line)

                if not valid:
                    skipped += 1
                    continue

                if class_id not in class_remap:
                    # Not a target class — skip but don't count as error
                    continue

                new_id = class_remap[class_id]
                new_line = f"{new_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
                new_lines.append(new_line)
                class_counts[new_id] += 1

    except Exception as e:
        logger.warning("Error processing %s: %s", label_path, e)

    return new_lines, class_counts, total_original, skipped


def prepare_split(
    source_dir: Path,
    output_dir: Path,
    split_name: str,
    class_remap: dict[int, int],
) -> dict:
    """Prepare a single split: copy images, filter/remap labels."""
    src_images = source_dir / split_name / "images"
    src_labels = source_dir / split_name / "labels"
    dst_images = output_dir / split_name / "images"
    dst_labels = output_dir / split_name / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    stats = {
        "split": split_name,
        "images_processed": 0,
        "images_with_targets": 0,
        "images_negative": 0,  # No target class objects (kept as negatives)
        "total_original_objects": 0,
        "total_filtered_objects": 0,
        "skipped_invalid": 0,
        "class_counts": Counter(),
    }

    if not src_images.is_dir():
        logger.warning("Source images dir not found: %s", src_images)
        return stats

    image_files = sorted([
        f for f in src_images.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    for img_file in image_files:
        label_file = src_labels / (img_file.stem + ".txt")

        # Process label
        new_lines = []
        if label_file.exists():
            new_lines, counts, total_orig, skipped = process_label_file(
                label_file, class_remap
            )
            stats["total_original_objects"] += total_orig
            stats["total_filtered_objects"] += len(new_lines)
            stats["skipped_invalid"] += skipped
            stats["class_counts"] += counts

        # Copy image (symlink for speed, fallback to copy)
        dst_img = dst_images / img_file.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)

        # Write filtered label
        dst_lbl = dst_labels / (img_file.stem + ".txt")
        with open(dst_lbl, "w") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")

        stats["images_processed"] += 1
        if new_lines:
            stats["images_with_targets"] += 1
        else:
            stats["images_negative"] += 1

    stats["class_counts"] = dict(stats["class_counts"])
    return stats


def generate_data_yaml(output_dir: Path, class_names: dict[int, str]) -> Path:
    """Generate the data.yaml file for YOLO training."""
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": [class_names[i] for i in sorted(class_names.keys())],
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("Generated data.yaml at %s", yaml_path)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO training")
    parser.add_argument(
        "--source", type=str, default="dataset/css-data",
        help="Source dataset path (default: dataset/css-data)"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--reports-dir", type=str, default="reports",
        help="Reports output directory (default: reports)"
    )
    parser.add_argument(
        "--all-classes", action="store_true",
        help="Keep all 10 classes instead of filtering to MVP 3"
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.is_dir():
        logger.error("Source dataset not found: %s", source_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("DATASET PREPARATION")
    logger.info("  Source: %s", source_dir.resolve())
    logger.info("  Output: %s", output_dir.resolve())
    logger.info("=" * 60)

    # Select class mapping
    if args.all_classes:
        class_remap = {i: i for i in range(10)}
        class_names = ORIGINAL_CLASSES
        logger.info("Mode: ALL 10 classes")
    else:
        class_remap = MVP_CLASS_REMAP
        class_names = MVP_CLASS_NAMES
        logger.info("Mode: MVP 3 classes (Hardhat, NO-Hardhat, Person)")

    logger.info("Class remap: %s", class_remap)
    logger.info("Class names: %s", class_names)

    # Process each split
    all_stats = {}
    for split_name in ["train", "valid", "test"]:
        logger.info("Processing %s split...", split_name)
        stats = prepare_split(source_dir, output_dir, split_name, class_remap)
        all_stats[split_name] = stats
        logger.info("  Images: %d (with targets: %d, negative: %d)",
                     stats["images_processed"], stats["images_with_targets"],
                     stats["images_negative"])
        logger.info("  Objects: %d → %d filtered (%d invalid skipped)",
                     stats["total_original_objects"], stats["total_filtered_objects"],
                     stats["skipped_invalid"])
        logger.info("  Class counts: %s", stats["class_counts"])

    # Generate data.yaml
    yaml_path = generate_data_yaml(output_dir, class_names)

    # Build summary
    summary = {
        "preparation_date": __import__("datetime").datetime.now().isoformat(),
        "source_path": str(source_dir.resolve()),
        "output_path": str(output_dir.resolve()),
        "mode": "all_classes" if args.all_classes else "mvp_3_classes",
        "class_remap": {str(k): v for k, v in class_remap.items()},
        "class_names": {str(k): v for k, v in class_names.items()},
        "num_classes": len(class_names),
        "data_yaml_path": str(yaml_path),
        "splits": all_stats,
        "total_images": sum(s["images_processed"] for s in all_stats.values()),
        "total_objects": sum(s["total_filtered_objects"] for s in all_stats.values()),
        "original_dataset": {
            "source": "Construction Site Safety - v28 YOLOv5s (Roboflow)",
            "kaggle_slug": "snehilsanyal/construction-site-safety-image-dataset-roboflow",
            "license": "CC BY 4.0",
            "original_classes": 10,
        },
    }

    # Write JSON summary
    json_path = reports_dir / "dataset_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("JSON summary: %s", json_path)

    # Write Markdown summary
    md_path = reports_dir / "dataset_summary.md"
    with open(md_path, "w") as f:
        f.write("# Dataset Summary\n\n")
        f.write(f"**Prepared:** {summary['preparation_date']}\n\n")
        f.write(f"**Source:** `{summary['source_path']}`\n\n")
        f.write(f"**Mode:** {summary['mode']}\n\n")
        f.write(f"**Total Images:** {summary['total_images']}\n\n")
        f.write(f"**Total Objects:** {summary['total_objects']}\n\n")

        f.write("## Class Mapping\n\n")
        f.write("| New ID | Name | Original ID |\n|---|---|---|\n")
        for orig_id, new_id in sorted(class_remap.items(), key=lambda x: x[1]):
            f.write(f"| {new_id} | {ORIGINAL_CLASSES[orig_id]} | {orig_id} |\n")

        f.write("\n## Split Statistics\n\n")
        f.write("| Split | Images | With Targets | Negative | Objects |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for split_name, stats in all_stats.items():
            f.write(
                f"| {split_name} | {stats['images_processed']} | "
                f"{stats['images_with_targets']} | {stats['images_negative']} | "
                f"{stats['total_filtered_objects']} |\n"
            )

        f.write("\n## Object Counts by Class\n\n")
        all_class_ids = sorted(set().union(*[
            set(int(k) for k in s["class_counts"].keys())
            for s in all_stats.values()
            if s["class_counts"]
        ]))
        if all_class_ids:
            header = "| Split |"
            for cid in all_class_ids:
                header += f" {class_names.get(cid, str(cid))} |"
            header += " Total |\n"
            sep = "|---|" + "|".join("---:" for _ in all_class_ids) + "|---:|\n"
            f.write(header)
            f.write(sep)
            for split_name, stats in all_stats.items():
                row = f"| {split_name} |"
                total = 0
                for cid in all_class_ids:
                    val = stats["class_counts"].get(str(cid), stats["class_counts"].get(cid, 0))
                    total += val
                    row += f" {val} |"
                row += f" {total} |\n"
                f.write(row)

        f.write(f"\n## Data YAML\n\n`{yaml_path}`\n")

    logger.info("Markdown summary: %s", md_path)
    logger.info("=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
