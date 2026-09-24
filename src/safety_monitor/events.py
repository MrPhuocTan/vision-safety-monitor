"""
Event Manager — Record violation events and save evidence images.

Manages JSONL event storage and evidence image capture.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import AppConfig
from .schemas import ViolationEvent

logger = logging.getLogger(__name__)


class EventManager:
    """
    Manages violation event recording and evidence storage.

    Events are appended to a JSONL file. Evidence images are saved
    with bounding box annotations.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._events_file = Path(config.output.events_file)
        self._evidence_dir = Path(config.output.evidence_dir)
        self._events: list[ViolationEvent] = []

        # Ensure directories exist
        self._events_file.parent.mkdir(parents=True, exist_ok=True)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)

    def record_event(
        self,
        event: ViolationEvent,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record a violation event and optionally save evidence.

        Args:
            event: The violation event to record.
            frame: Current video frame for evidence capture.
        """
        # Save evidence image
        if frame is not None and self.config.output.save_evidence:
            evidence_path = self._save_evidence(event, frame)
            event.evidence_path = str(evidence_path)

        # Store event
        self._events.append(event)

        # Append to JSONL file
        self._append_to_file(event)

        logger.info(
            "EVENT [%s] Track#%d Frame#%d — %s (evidence: %s)",
            event.event_type,
            event.track_id,
            event.frame_index,
            event.details,
            event.evidence_path,
        )

    def _save_evidence(
        self,
        event: ViolationEvent,
        frame: np.ndarray,
    ) -> Path:
        """Save an evidence image with bbox annotation."""
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = (
            f"{event.event_type}_track{event.track_id}_"
            f"frame{event.frame_index}_{timestamp}.jpg"
        )
        filepath = self._evidence_dir / filename

        # Draw bbox on evidence frame
        evidence = frame.copy()
        if event.bbox:
            x1, y1, x2, y2 = [int(c) for c in event.bbox]
            color = (0, 0, 255)  # Red
            cv2.rectangle(evidence, (x1, y1), (x2, y2), color, 3)
            label = f"{event.event_type} Track#{event.track_id}"
            cv2.putText(
                evidence, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        cv2.imwrite(str(filepath), evidence)
        return filepath

    def _append_to_file(self, event: ViolationEvent) -> None:
        """Append event to JSONL file."""
        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error("Failed to write event to %s: %s", self._events_file, e)

    def get_events(
        self,
        event_type: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> list[ViolationEvent]:
        """Get recorded events, optionally filtered."""
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if track_id is not None:
            events = [e for e in events if e.track_id == track_id]

        return events

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def events_by_type(self) -> dict[str, int]:
        """Count events by type."""
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts

    def load_events_from_file(self) -> list[dict]:
        """Load events from the JSONL file on disk."""
        events = []
        if self._events_file.exists():
            try:
                with open(self._events_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            events.append(json.loads(line))
            except Exception as e:
                logger.error("Failed to load events: %s", e)
        return events

    def clear(self) -> None:
        """Clear all in-memory events."""
        self._events.clear()
