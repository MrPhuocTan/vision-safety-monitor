"""
Frame Renderer — Draw overlays, bounding boxes, and alerts on video frames.

Handles all visualization: bounding boxes, labels, confidence, track IDs,
restricted-zone polygon overlay, FPS counter, violation count, and alert banners.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .config import AppConfig
from .schemas import (
    PersonState,
    TrackedDetection,
    ViolationEvent,
)

logger = logging.getLogger(__name__)


class Renderer:
    """
    Renders annotations and overlays on video frames.

    All rendering is done on a copy of the frame to avoid
    modifying the original.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._display = config.display

    def render(
        self,
        frame: np.ndarray,
        tracked_detections: list[TrackedDetection],
        person_states: dict[int, PersonState],
        violations: list[ViolationEvent],
        fps: float = 0.0,
        frame_index: int = 0,
    ) -> np.ndarray:
        """
        Render all overlays on a frame.

        Args:
            frame: BGR image (will be copied internally).
            tracked_detections: All tracked detections to draw.
            person_states: Current person states for status indicators.
            violations: Active violations in this frame.
            fps: Current processing FPS.
            frame_index: Current frame number.

        Returns:
            Annotated BGR frame (copy of input).
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Draw zone overlay
        if self._display.show_zone_overlay:
            self._draw_zones(output, w, h)

        # 2. Draw bounding boxes
        self._draw_detections(output, tracked_detections, person_states)

        # 3. Draw info overlay (FPS, violation count)
        self._draw_info_overlay(output, fps, frame_index, person_states)

        # 4. Draw alert banner for active violations
        if violations:
            self._draw_alert_banner(output, violations)

        return output

    def _draw_zones(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw restricted zone polygons as semi-transparent overlays."""
        for zone in self.config.zones:
            if len(zone.polygon) < 3:
                continue

            # Convert normalized coordinates to pixel coordinates
            points = np.array([
                [int(p[0] * width), int(p[1] * height)]
                for p in zone.polygon
            ], dtype=np.int32)

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], tuple(self._display.zone_color))
            cv2.addWeighted(
                overlay, self._display.zone_alpha,
                frame, 1 - self._display.zone_alpha,
                0, frame,
            )

            # Border
            cv2.polylines(frame, [points], True, tuple(self._display.zone_color), 2)

            # Zone label
            cx = int(np.mean([p[0] for p in points]))
            cy = int(np.mean([p[1] for p in points]))
            cv2.putText(
                frame, f"⚠ {zone.name}",
                (cx - 60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                tuple(self._display.zone_color), 2,
            )

    def _draw_detections(
        self,
        frame: np.ndarray,
        tracked: list[TrackedDetection],
        person_states: dict[int, PersonState],
    ) -> None:
        """Draw bounding boxes with labels, confidence, and track IDs."""
        person_cls = self.config.classes.person_class_id
        hardhat_cls = self.config.classes.hardhat_class_id
        no_hardhat_cls = self.config.classes.no_hardhat_class_id

        for td in tracked:
            det = td.detection
            x1, y1, x2, y2 = [int(c) for c in det.bbox]

            # Pick color based on class
            if det.class_id == person_cls:
                color = tuple(self._display.person_color)
                # Override color based on helmet status
                state = person_states.get(td.track_id)
                if state:
                    if state.helmet_status == "no_hardhat":
                        color = tuple(self._display.no_hardhat_color)
                    elif state.helmet_status == "hardhat":
                        color = tuple(self._display.hardhat_color)
            elif det.class_id == hardhat_cls:
                color = tuple(self._display.hardhat_color)
            elif det.class_id == no_hardhat_cls:
                color = tuple(self._display.no_hardhat_color)
            else:
                color = (200, 200, 200)

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._display.bbox_thickness)

            # Build label text
            parts = [det.class_name]
            if self._display.show_confidence:
                parts.append(f"{det.confidence:.2f}")
            if self._display.show_track_ids and td.track_id >= 0:
                parts.insert(0, f"#{td.track_id}")

            # Add helmet status for persons
            if det.class_id == person_cls:
                state = person_states.get(td.track_id)
                if state and state.helmet_status != "unknown":
                    status_icon = "✓" if state.helmet_status == "hardhat" else "✗"
                    parts.append(status_icon)

            label = " ".join(parts)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                self._display.font_scale, 1,
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                color, -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._display.font_scale,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    def _draw_info_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        frame_index: int,
        person_states: dict[int, PersonState],
    ) -> None:
        """Draw FPS counter, frame number, and violation count."""
        h, w = frame.shape[:2]
        y_offset = 30

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 260, 0), (w, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if self._display.show_fps:
            text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame, text, (w - 250, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            y_offset += 25

        # Frame counter
        cv2.putText(
            frame, f"Frame: {frame_index}",
            (w - 250, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )
        y_offset += 25

        if self._display.show_violation_count:
            # Count active violations
            violation_count = sum(
                1 for s in person_states.values()
                if s.helmet_status == "no_hardhat" or s.current_zone_id is not None
            )
            color = (0, 0, 255) if violation_count > 0 else (0, 255, 0)
            cv2.putText(
                frame, f"Violations: {violation_count}",
                (w - 250, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

    def _draw_alert_banner(
        self,
        frame: np.ndarray,
        violations: list[ViolationEvent],
    ) -> None:
        """Draw a warning banner at the top of the frame."""
        h, w = frame.shape[:2]
        banner_height = 50

        # Red banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height),
                       tuple(self._display.alert_banner_color), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Warning text
        text = f"⚠ ALERT: {len(violations)} violation(s) detected!"
        cv2.putText(
            frame, text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
