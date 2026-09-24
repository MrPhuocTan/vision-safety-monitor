#!/usr/bin/env python3
"""
Dataset Inspector — Recursively inspects the dataset/ folder.

Generates a comprehensive inventory report before any transformation
or download occurs.

Usage:
    python scripts/inspect_dataset.py [--source dataset/]

Outputs:
    reports/local_dataset_inventory.md
    reports/local_dataset_inventory.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
LABEL_EXTENSIONS = {".txt"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".bz2", ".rar", ".7z"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".xml", ".csv"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".tflite"}


def scan_directory(root: Path) -> dict:
    """Recursively scan a directory and categorize all files."""
    inventory = {
        "images": [],
        "labels": [],
        "archives": [],
        "videos": [],
        "configs": [],
        "weights": [],
        "notebooks": [],
        "readmes": [],
        "other": [],
    }

    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        ext = path.suffix.lower()
        rel = str(path.relative_to(root))

        entry = {"path": rel, "size_bytes": path.stat().st_size}

        if ext in IMAGE_EXTENSIONS:
            inventory["images"].append(entry)
        elif ext in LABEL_EXTENSIONS:
            inventory["labels"].append(entry)
        elif ext in ARCHIVE_EXTENSIONS:
            inventory["archives"].append(entry)
        elif ext in VIDEO_EXTENSIONS:
            inventory["videos"].append(entry)
        elif ext in CONFIG_EXTENSIONS:
            inventory["configs"].append(entry)
        elif ext in WEIGHT_EXTENSIONS:
            inventory["weights"].append(entry)
        elif ext == ".ipynb":
            inventory["notebooks"].append(entry)
        elif path.name.lower().startswith("readme"):
            inventory["readmes"].append(entry)
        else:
            inventory["other"].append(entry)

    return inventory


def detect_annotation_format(label_path: Path) -> str:
    """Detect annotation format from actual label file content."""
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return "empty"

        # Check first non-empty line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                # YOLO format: class_id cx cy w h
                try:
                    int(parts[0])
                    floats = [float(p) for p in parts[1:]]
                    if all(0 <= v <= 1.0 + 1e-6 for v in floats):
                        return "yolo"
                except ValueError:
                    pass
            elif len(parts) >= 6:
                return "possibly_yolo_obb_or_other"

        return "unknown"
    except Exception as e:
        return f"error: {e}"


def find_data_yaml_files(root: Path) -> list[dict]:
    """Find and parse all data.yaml / *.yaml config files."""
    yaml_files = []
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, skipping YAML parsing")
        return yaml_files

    for path in root.rglob("*.yaml"):
        if path.name.startswith("."):
            continue
        try:
            with open(path, "r") as f:
                content = yaml.safe_load(f)
            if content and isinstance(content, dict):
                entry = {
                    "path": str(path.relative_to(root)),
                    "content": content,
                }
                # Check if this looks like a data config
                if any(k in content for k in ["names", "nc", "train", "val", "test"]):
                    entry["is_data_config"] = True
                    if "names" in content:
                        entry["class_mapping"] = content["names"]
                    if "nc" in content:
                        entry["num_classes"] = content["nc"]
                else:
                    entry["is_data_config"] = False
                yaml_files.append(entry)
        except Exception as e:
            yaml_files.append({
                "path": str(path.relative_to(root)),
                "error": str(e),
            })

    return yaml_files


def detect_splits(root: Path) -> dict:
    """Detect train/validation/test split structure."""
    splits = {}
    for split_name in ["train", "training", "valid", "validation", "val", "test", "testing"]:
        for candidate in root.rglob(split_name):
            if candidate.is_dir():
                images_dir = candidate / "images"
                labels_dir = candidate / "labels"
                split_info = {
                    "path": str(candidate.relative_to(root)),
                    "has_images_dir": images_dir.is_dir(),
                    "has_labels_dir": labels_dir.is_dir(),
                }
                if images_dir.is_dir():
                    imgs = [f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
                    split_info["image_count"] = len(imgs)
                if labels_dir.is_dir():
                    lbls = [f for f in labels_dir.iterdir() if f.suffix.lower() in LABEL_EXTENSIONS]
                    split_info["label_count"] = len(lbls)
                splits[split_name] = split_info

    return splits


def validate_labels(root: Path, splits: dict) -> dict:
    """Validate label files: pairing, coordinate validity, class IDs, empty labels."""
    validation = {
        "total_images": 0,
        "total_labels": 0,
        "paired": 0,
        "missing_labels": [],
        "orphan_labels": [],
        "empty_labels": [],
        "invalid_coordinates": [],
        "class_distribution": Counter(),
        "objects_per_split": {},
        "issues": [],
    }

    for split_name, split_info in splits.items():
        split_path = root / split_info["path"]
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        image_stems = {f.stem for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}
        label_stems = {f.stem for f in labels_dir.iterdir() if f.suffix.lower() in LABEL_EXTENSIONS}

        validation["total_images"] += len(image_stems)
        validation["total_labels"] += len(label_stems)

        paired = image_stems & label_stems
        validation["paired"] += len(paired)

        missing = image_stems - label_stems
        if missing:
            validation["missing_labels"].extend(
                [f"{split_name}/{s}" for s in sorted(list(missing))[:10]]
            )
            if len(missing) > 10:
                validation["issues"].append(
                    f"{split_name}: {len(missing)} images missing labels (showing first 10)"
                )

        orphans = label_stems - image_stems
        if orphans:
            validation["orphan_labels"].extend(
                [f"{split_name}/{s}" for s in sorted(list(orphans))[:10]]
            )

        # Validate label content
        split_class_counts = Counter()
        for label_file in sorted(labels_dir.iterdir()):
            if label_file.suffix.lower() not in LABEL_EXTENSIONS:
                continue

            try:
                with open(label_file, "r") as f:
                    lines = f.readlines()

                if not lines or all(l.strip() == "" for l in lines):
                    validation["empty_labels"].append(f"{split_name}/{label_file.name}")
                    continue

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        validation["invalid_coordinates"].append(
                            f"{split_name}/{label_file.name}:{line_num} - too few values"
                        )
                        continue

                    try:
                        class_id = int(parts[0])
                        coords = [float(p) for p in parts[1:5]]

                        # Validate normalized coordinates
                        for i, c in enumerate(coords):
                            if c < -0.01 or c > 1.01:
                                validation["invalid_coordinates"].append(
                                    f"{split_name}/{label_file.name}:{line_num} - "
                                    f"coord[{i}]={c} out of [0,1]"
                                )
                                break

                        validation["class_distribution"][class_id] += 1
                        split_class_counts[class_id] += 1

                    except ValueError:
                        validation["invalid_coordinates"].append(
                            f"{split_name}/{label_file.name}:{line_num} - parse error"
                        )
            except Exception as e:
                validation["issues"].append(f"Error reading {label_file.name}: {e}")

        validation["objects_per_split"][split_name] = dict(split_class_counts)

    # Convert Counter to regular dict for JSON
    validation["class_distribution"] = dict(validation["class_distribution"])

    # Truncate long lists for readability
    for key in ["missing_labels", "orphan_labels", "empty_labels", "invalid_coordinates"]:
        if len(validation[key]) > 20:
            total = len(validation[key])
            validation[key] = validation[key][:20]
            validation["issues"].append(f"{key}: showing 20 of {total} entries")

    return validation


def check_duplicates_across_splits(root: Path, splits: dict) -> list[str]:
    """Check for exact duplicate image filenames across splits."""
    split_images: dict[str, set[str]] = {}

    for split_name, split_info in splits.items():
        images_dir = root / split_info["path"] / "images"
        if images_dir.is_dir():
            split_images[split_name] = {
                f.name for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
            }

    duplicates = []
    split_names = list(split_images.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            shared = split_images[split_names[i]] & split_images[split_names[j]]
            if shared:
                duplicates.append(
                    f"{len(shared)} duplicate filenames between "
                    f"{split_names[i]} and {split_names[j]}"
                )

    return duplicates


def verify_sample_images(root: Path, splits: dict, num_samples: int = 3) -> list[dict]:
    """Verify a few sample image-annotation pairs are readable."""
    samples = []

    for split_name, split_info in splits.items():
        images_dir = root / split_info["path"] / "images"
        labels_dir = root / split_info["path"] / "labels"

        if not images_dir.is_dir():
            continue

        image_files = sorted([
            f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        for img_file in image_files[:num_samples]:
            sample = {
                "split": split_name,
                "image": img_file.name,
                "readable": False,
                "shape": None,
                "label_exists": False,
                "label_objects": 0,
            }

            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    sample["readable"] = True
                    sample["shape"] = list(img.shape)
            except Exception:
                pass

            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                sample["label_exists"] = True
                try:
                    with open(label_file) as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                    sample["label_objects"] = len(lines)
                except Exception:
                    pass

            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset folder recursively")
    parser.add_argument(
        "--source", type=str, default="dataset",
        help="Path to the dataset root folder (default: dataset/)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports",
        help="Output directory for reports (default: reports/)"
    )
    args = parser.parse_args()

    root = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        logger.error("Dataset path does not exist: %s", root)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("DATASET INSPECTION: %s", root.resolve())
    logger.info("=" * 60)

    # 1. Scan entire directory
    logger.info("Step 1: Scanning directory tree...")
    inventory = scan_directory(root)

    logger.info("  Images: %d", len(inventory["images"]))
    logger.info("  Labels: %d", len(inventory["labels"]))
    logger.info("  Archives: %d", len(inventory["archives"]))
    logger.info("  Videos: %d", len(inventory["videos"]))
    logger.info("  Configs: %d", len(inventory["configs"]))
    logger.info("  Weights: %d", len(inventory["weights"]))
    logger.info("  Notebooks: %d", len(inventory["notebooks"]))
    logger.info("  READMEs: %d", len(inventory["readmes"]))
    logger.info("  Other: %d", len(inventory["other"]))

    # 2. Find and parse YAML configs
    logger.info("Step 2: Finding data configuration files...")
    yaml_configs = find_data_yaml_files(root)
    data_configs = [y for y in yaml_configs if y.get("is_data_config")]
    logger.info("  Found %d data config(s)", len(data_configs))

    class_mapping = None
    for dc in data_configs:
        if "class_mapping" in dc:
            class_mapping = dc["class_mapping"]
            logger.info("  Class mapping from %s:", dc["path"])
            if isinstance(class_mapping, list):
                for i, name in enumerate(class_mapping):
                    logger.info("    %d: %s", i, name)
            elif isinstance(class_mapping, dict):
                for k, v in class_mapping.items():
                    logger.info("    %s: %s", k, v)

    # 3. Detect annotation format
    logger.info("Step 3: Detecting annotation format...")
    annotation_format = "unknown"
    for label_entry in inventory["labels"][:5]:
        fmt = detect_annotation_format(root / label_entry["path"])
        if fmt == "yolo":
            annotation_format = "yolo"
            break
        elif fmt != "empty":
            annotation_format = fmt
    logger.info("  Annotation format: %s", annotation_format)

    # 4. Detect splits
    logger.info("Step 4: Detecting train/val/test splits...")
    splits = detect_splits(root)
    for split_name, split_info in splits.items():
        logger.info("  %s: %s", split_name, split_info)

    # 5. Validate labels
    logger.info("Step 5: Validating labels...")
    validation = validate_labels(root, splits)
    logger.info("  Total images: %d", validation["total_images"])
    logger.info("  Total labels: %d", validation["total_labels"])
    logger.info("  Paired: %d", validation["paired"])
    logger.info("  Missing labels: %d", len(validation["missing_labels"]))
    logger.info("  Empty labels: %d", len(validation["empty_labels"]))
    logger.info("  Invalid coordinates: %d", len(validation["invalid_coordinates"]))
    logger.info("  Class distribution: %s", validation["class_distribution"])

    # 6. Check cross-split duplicates
    logger.info("Step 6: Checking cross-split duplicates...")
    duplicates = check_duplicates_across_splits(root, splits)
    if duplicates:
        for d in duplicates:
            logger.warning("  %s", d)
    else:
        logger.info("  No duplicate filenames across splits")

    # 7. Verify sample images
    logger.info("Step 7: Verifying sample image-annotation pairs...")
    samples = verify_sample_images(root, splits)
    for s in samples:
        logger.info("  [%s] %s — readable=%s, shape=%s, labels=%d",
                     s["split"], s["image"], s["readable"], s["shape"], s["label_objects"])

    # 8. Dataset decision
    logger.info("Step 8: Dataset source decision...")
    decision = "UNKNOWN"
    decision_reason = ""

    if (validation["total_images"] > 100
            and validation["paired"] >= validation["total_images"] * 0.95
            and annotation_format == "yolo"
            and len(splits) >= 2):
        decision = "LOCAL_VALID_COMPLETE"
        decision_reason = (
            f"Local dataset is valid and complete. "
            f"{validation['total_images']} images with {validation['paired']} paired labels "
            f"in YOLO format across {len(splits)} splits. No Kaggle download needed."
        )
    elif validation["total_images"] > 50 and annotation_format == "yolo":
        decision = "LOCAL_USABLE_INCOMPLETE"
        decision_reason = "Local dataset is usable but may be incomplete."
    elif validation["total_images"] < 10:
        decision = "LOCAL_INVALID"
        decision_reason = "Local dataset is too small or missing."
    else:
        decision = "NEEDS_REVIEW"
        decision_reason = "Manual review recommended."

    logger.info("  Decision: %s", decision)
    logger.info("  Reason: %s", decision_reason)

    # Build report
    report = {
        "inspection_date": __import__("datetime").datetime.now().isoformat(),
        "source_path": str(root.resolve()),
        "file_counts": {k: len(v) for k, v in inventory.items()},
        "annotation_format": annotation_format,
        "data_configs": [
            {k: v for k, v in dc.items() if k != "content"}
            for dc in data_configs
        ],
        "class_mapping": class_mapping,
        "splits": splits,
        "validation": validation,
        "cross_split_duplicates": duplicates,
        "sample_verification": samples,
        "decision": decision,
        "decision_reason": decision_reason,
        "videos": inventory["videos"],
        "weights": inventory["weights"],
        "archives": inventory["archives"],
    }

    # Write JSON report
    json_path = output_dir / "local_dataset_inventory.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("JSON report: %s", json_path)

    # Write Markdown report
    md_path = output_dir / "local_dataset_inventory.md"
    with open(md_path, "w") as f:
        f.write("# Local Dataset Inventory Report\n\n")
        f.write(f"**Inspection Date:** {report['inspection_date']}\n\n")
        f.write(f"**Source Path:** `{report['source_path']}`\n\n")

        f.write("## File Counts\n\n")
        f.write("| Category | Count |\n|---|---:|\n")
        for cat, count in report["file_counts"].items():
            f.write(f"| {cat} | {count} |\n")

        f.write(f"\n## Annotation Format\n\n`{annotation_format}`\n\n")

        if class_mapping:
            f.write("## Class Mapping\n\n")
            f.write("| ID | Name |\n|---|---|\n")
            if isinstance(class_mapping, list):
                for i, name in enumerate(class_mapping):
                    f.write(f"| {i} | {name} |\n")
            elif isinstance(class_mapping, dict):
                for k, v in sorted(class_mapping.items(), key=lambda x: str(x[0])):
                    f.write(f"| {k} | {v} |\n")

        f.write("\n## Splits\n\n")
        f.write("| Split | Images | Labels | Paired |\n|---|---:|---:|---:|\n")
        for split_name, split_info in splits.items():
            img_count = split_info.get("image_count", "?")
            lbl_count = split_info.get("label_count", "?")
            f.write(f"| {split_name} | {img_count} | {lbl_count} | ✓ |\n")

        f.write("\n## Object Counts by Split and Class\n\n")
        if validation["objects_per_split"]:
            all_classes = sorted(set().union(*[
                set(v.keys()) for v in validation["objects_per_split"].values()
            ]))
            header = "| Split | " + " | ".join(str(c) for c in all_classes) + " | Total |\n"
            sep = "|---|" + "|".join("---:" for _ in all_classes) + "|---:|\n"
            f.write(header)
            f.write(sep)
            for split_name, counts in validation["objects_per_split"].items():
                row = f"| {split_name} |"
                total = 0
                for c in all_classes:
                    val = counts.get(c, 0)
                    total += val
                    row += f" {val} |"
                row += f" {total} |\n"
                f.write(row)

        f.write("\n## Validation Issues\n\n")
        if validation["issues"]:
            for issue in validation["issues"]:
                f.write(f"- ⚠️ {issue}\n")
        else:
            f.write("✅ No issues found.\n")

        if duplicates:
            f.write("\n## Cross-Split Duplicates\n\n")
            for d in duplicates:
                f.write(f"- ⚠️ {d}\n")

        f.write(f"\n## Decision\n\n**{decision}**\n\n{decision_reason}\n")

        # Videos
        if inventory["videos"]:
            f.write("\n## Sample Videos\n\n")
            for v in inventory["videos"]:
                f.write(f"- `{v['path']}` ({v['size_bytes'] / 1024 / 1024:.1f} MB)\n")

        # Weights
        if inventory["weights"]:
            f.write("\n## Pre-trained Weights\n\n")
            for w in inventory["weights"]:
                f.write(f"- `{w['path']}` ({w['size_bytes'] / 1024 / 1024:.1f} MB)\n")

    logger.info("Markdown report: %s", md_path)
    logger.info("=" * 60)
    logger.info("INSPECTION COMPLETE")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    main()
