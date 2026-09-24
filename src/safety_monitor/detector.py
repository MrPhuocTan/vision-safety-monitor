"""
YOLO Object Detector — Load and run the trained YOLO model.

Provides a backend-neutral result schema and supports filtering
to target classes at inference time.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import AppConfig
from .schemas import Detection

logger = logging.getLogger(__name__)


class Detector:
    """
    YOLO-based object detector.

    Loads the model once and provides detections filtered to target classes.
    Supports PyTorch (.pt) and ONNX (.onnx) backends.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None
        self._device = config.model.device
        self._class_names = config.classes.names
        self._target_ids = config.classes.target_ids

    def load(self) -> None:
        """Load the YOLO model."""
        from ultralytics import YOLO

        weights_path = self.config.model.weights
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}. "
                f"Copy the trained model to this path or update configs/app.yaml."
            )

        logger.info("Loading YOLO model from %s (device=%s)",
                     weights_path, self._device)

        self._model = YOLO(weights_path)

        # Warm up
        logger.info("Model loaded. Running warm-up inference...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            device=self._device,
            verbose=False,
            conf=0.5,
        )
        logger.info("Detector ready.")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> tuple[list[Detection], float]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image (numpy array).
            conf_threshold: Override confidence threshold.
            iou_threshold: Override NMS IoU threshold.

        Returns:
            (detections, inference_time_ms)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        conf = conf_threshold or self.config.model.conf_threshold
        iou = iou_threshold or self.config.model.iou_threshold

        start = time.perf_counter()

        results = self._model.predict(
            frame,
            device=self._device,
            imgsz=self.config.model.imgsz,
            conf=conf,
            iou=iou,
            max_det=self.config.model.max_det,
            classes=self._target_ids,  # Filter at inference
            verbose=False,
            agnostic_nms=True,
        )

        inference_time = (time.perf_counter() - start) * 1000  # ms

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    class_id = int(boxes.cls[i].cpu().item())
                    confidence = float(boxes.conf[i].cpu().item())

                    # Get class name
                    class_name = self._class_names.get(
                        class_id,
                        result.names.get(class_id, f"class_{class_id}")
                    )

                    detections.append(Detection(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    ))

        return detections, inference_time

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_info(self) -> dict:
        """Get model metadata."""
        if self._model is None:
            return {}
        return {
            "weights": self.config.model.weights,
            "device": self._device,
            "imgsz": self.config.model.imgsz,
            "target_classes": self._target_ids,
        }
