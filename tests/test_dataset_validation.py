"""Tests for dataset validation logic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def write_label(path: Path, lines: list[str]) -> None:
    """Helper to write a label file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class TestYoloLabelValidation:
    """Test YOLO label format validation."""

    def test_valid_yolo_line(self):
        from scripts.prepare_dataset import validate_yolo_line

        valid, cls_id, coords = validate_yolo_line("0 0.5 0.5 0.3 0.4")
        assert valid is True
        assert cls_id == 0
        assert len(coords) == 4
        assert all(0 <= c <= 1 for c in coords)

    def test_invalid_too_few_values(self):
        from scripts.prepare_dataset import validate_yolo_line

        valid, cls_id, coords = validate_yolo_line("0 0.5 0.5")
        assert valid is False

    def test_invalid_out_of_bounds(self):
        from scripts.prepare_dataset import validate_yolo_line

        valid, cls_id, coords = validate_yolo_line("0 1.5 0.5 0.3 0.4")
        assert valid is False

    def test_invalid_negative_dimensions(self):
        from scripts.prepare_dataset import validate_yolo_line

        valid, cls_id, coords = validate_yolo_line("0 0.5 0.5 0.0 0.4")
        assert valid is False

    def test_invalid_parse_error(self):
        from scripts.prepare_dataset import validate_yolo_line

        valid, cls_id, coords = validate_yolo_line("abc def ghi jkl mno")
        assert valid is False

    def test_empty_line(self):
        from scripts.prepare_dataset import validate_yolo_line

        valid, cls_id, coords = validate_yolo_line("")
        assert valid is False

    def test_edge_case_boundary_coords(self):
        from scripts.prepare_dataset import validate_yolo_line

        # Coords exactly at boundary should be valid
        valid, cls_id, coords = validate_yolo_line("5 1.0 1.0 0.01 0.01")
        assert valid is True

    def test_slight_overshoot_clamped(self):
        from scripts.prepare_dataset import validate_yolo_line

        # Slight over 1.0 is tolerated and clamped
        valid, cls_id, coords = validate_yolo_line("2 0.5 0.5 1.005 0.3")
        assert valid is True
        assert coords[2] == 1.0  # clamped


class TestClassRemapping:
    """Test class ID remapping logic."""

    def test_mvp_remap(self):
        from scripts.prepare_dataset import MVP_CLASS_REMAP

        assert MVP_CLASS_REMAP[0] == 0   # Hardhat → 0
        assert MVP_CLASS_REMAP[2] == 1   # NO-Hardhat → 1
        assert MVP_CLASS_REMAP[5] == 2   # Person → 2

    def test_non_target_class_filtered(self):
        from scripts.prepare_dataset import MVP_CLASS_REMAP

        # Class 1 (Mask) should not be in remap
        assert 1 not in MVP_CLASS_REMAP
        assert 3 not in MVP_CLASS_REMAP
        assert 6 not in MVP_CLASS_REMAP

    def test_process_label_file_filters_classes(self):
        from scripts.prepare_dataset import MVP_CLASS_REMAP, process_label_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write labels with mixed classes
            f.write("0 0.5 0.5 0.1 0.1\n")   # Hardhat → keep
            f.write("1 0.3 0.3 0.1 0.1\n")   # Mask → filter out
            f.write("2 0.6 0.6 0.2 0.2\n")   # NO-Hardhat → keep
            f.write("5 0.4 0.4 0.3 0.5\n")   # Person → keep
            f.write("9 0.7 0.7 0.2 0.3\n")   # Vehicle → filter out
            f.flush()

            new_lines, counts, total, skipped = process_label_file(
                Path(f.name), MVP_CLASS_REMAP
            )

        assert len(new_lines) == 3  # Hardhat, NO-Hardhat, Person
        assert counts[0] == 1  # Hardhat
        assert counts[1] == 1  # NO-Hardhat
        assert counts[2] == 1  # Person
        assert total == 5
        assert skipped == 0

        # Verify remapped IDs
        assert new_lines[0].startswith("0 ")   # Hardhat → 0
        assert new_lines[1].startswith("1 ")   # NO-Hardhat → 1
        assert new_lines[2].startswith("2 ")   # Person → 2
