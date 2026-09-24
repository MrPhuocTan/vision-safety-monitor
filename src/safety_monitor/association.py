"""
PPE-Person Association — Associates helmet detections with person tracks.

Uses geometric logic (IoU / containment) to link Hardhat/NO-Hardhat
detections to the correct person based on spatial proximity.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import AppConfig
from .schemas import Detection, TrackedDetection

logger = logging.getLogger(__name__)


def compute_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def is_helmet_above_person(
    helmet_bbox: list[float],
    person_bbox: list[float],
    vertical_ratio: float = 0.5,
) -> bool:
    """
    Check if a helmet detection is in the upper portion of a person bbox.

    The helmet center should be in the top `vertical_ratio` of the person height.
    """
    helmet_center_y = (helmet_bbox[1] + helmet_bbox[3]) / 2
    person_top = person_bbox[1]
    person_height = person_bbox[3] - person_bbox[1]

    if person_height <= 0:
        return False

    upper_boundary = person_top + person_height * vertical_ratio
    return helmet_center_y <= upper_boundary


def compute_containment(inner: list[float], outer: list[float]) -> float:
    """
    Compute what fraction of `inner` bbox area is contained within `outer`.

    Returns value in [0, 1].
    """
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])

    if inner_area <= 0:
        return 0.0

    return inter_area / inner_area


def associate_ppe_to_persons(
    tracked_detections: list[TrackedDetection],
    config: AppConfig,
) -> dict[int, str]:
    """
    Associate helmet/no-helmet detections with person tracks.

    For each person, find the best matching helmet detection based on:
    1. Helmet bbox should overlap with person bbox (IoU or containment).
    2. Helmet should be in the upper portion of the person.
    3. Best match wins.

    Args:
        tracked_detections: All tracked detections (persons + helmets).
        config: Application configuration.

    Returns:
        Dict mapping person track_id → helmet status ("hardhat", "no_hardhat", "unknown").
    """
    person_cls = config.classes.person_class_id
    hardhat_cls = config.classes.hardhat_class_id
    no_hardhat_cls = config.classes.no_hardhat_class_id
    iou_thresh = config.association.iou_threshold
    vertical_ratio = config.association.vertical_ratio

    # Separate persons and helmet detections
    persons = [t for t in tracked_detections if t.detection.class_id == person_cls]
    hardhats = [t for t in tracked_detections if t.detection.class_id == hardhat_cls]
    no_hardhats = [t for t in tracked_detections if t.detection.class_id == no_hardhat_cls]

    helmet_status: dict[int, str] = {}

    for person in persons:
        if person.track_id < 0:
            continue

        best_score = 0.0
        best_status = "unknown"

        # Check hardhat detections
        for hat in hardhats:
            containment = compute_containment(hat.detection.bbox, person.detection.bbox)
            iou = compute_iou(hat.detection.bbox, person.detection.bbox)
            score = max(containment, iou)

            if score >= iou_thresh:
                # Verify spatial relationship
                if is_helmet_above_person(hat.detection.bbox, person.detection.bbox, vertical_ratio):
                    if score > best_score:
                        best_score = score
                        best_status = "hardhat"

        # Check no-hardhat detections
        for no_hat in no_hardhats:
            containment = compute_containment(no_hat.detection.bbox, person.detection.bbox)
            iou = compute_iou(no_hat.detection.bbox, person.detection.bbox)
            score = max(containment, iou)

            if score >= iou_thresh:
                if is_helmet_above_person(no_hat.detection.bbox, person.detection.bbox, vertical_ratio):
                    # NO-Hardhat takes precedence only if it has a higher score
                    if score > best_score:
                        best_score = score
                        best_status = "no_hardhat"

        helmet_status[person.track_id] = best_status

    return helmet_status
