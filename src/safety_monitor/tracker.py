"""
Object Tracker — ByteTrack integration for stable person tracking.

Uses Ultralytics built-in tracker (ByteTrack or BoT-SORT) to maintain
stable track IDs across frames for detected persons.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import AppConfig
from .schemas import Detection, TrackedDetection

logger = logging.getLogger(__name__)


class Tracker:
    """
    Object tracker using Ultralytics built-in ByteTrack/BoT-SORT.

    Maintains stable IDs for people across frames.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None

    def set_model(self, model) -> None:
        """Set the YOLO model reference for tracking."""
        self._model = model

    def track(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[TrackedDetection]:
        """
        Run tracking on detections using the YOLO built-in tracker.

        This uses the Ultralytics model.track() which integrates ByteTrack.

        Args:
            frame: BGR image.
            detections: List of Detection objects (used for fallback).

        Returns:
            List of TrackedDetection with assigned track IDs.
        """
        if self._model is None:
            # Fallback: return detections without track IDs
            logger.warning("Model not set for tracker, using detection-only mode")
            return [
                TrackedDetection(detection=d, track_id=-1)
                for d in detections
            ]

        tracker_type = self.config.tracker.type
        persist = True

        try:
            results = self._model.track(
                frame,
                device=self.config.model.device,
                imgsz=self.config.model.imgsz,
                conf=self.config.model.conf_threshold,
                iou=self.config.model.iou_threshold,
                max_det=self.config.model.max_det,
                classes=self.config.classes.target_ids,
                tracker=f"{tracker_type}.yaml",
                persist=persist,
                verbose=False,
                agnostic_nms=True,
            )
        except Exception as e:
            logger.warning("Tracker error: %s — falling back to detection-only", e)
            return [
                TrackedDetection(detection=d, track_id=-1)
                for d in detections
            ]

        tracked = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    class_id = int(boxes.cls[i].cpu().item())
                    confidence = float(boxes.conf[i].cpu().item())

                    # Get track ID
                    track_id = -1
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().item())

                    class_name = self.config.classes.names.get(
                        class_id,
                        result.names.get(class_id, f"class_{class_id}")
                    )

                    det = Detection(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    )
                    tracked.append(TrackedDetection(
                        detection=det,
                        track_id=track_id,
                    ))

        return tracked

    def reset(self) -> None:
        """Reset tracker state (clear all tracks)."""
        if self._model is not None:
            try:
                self._model.predictor = None
            except Exception:
                pass
        logger.info("Tracker state reset")
