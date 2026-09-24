"""
Data schemas for the Vision Safety Monitor.

Provides backend-neutral data structures used across all components
to decouple detector output from business logic and UI.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """Types of safety violation events."""
    NO_HARDHAT = "NO_HARDHAT"
    ZONE_INTRUSION = "ZONE_INTRUSION"
    NO_HARDHAT_IN_ZONE = "NO_HARDHAT_IN_ZONE"


@dataclass
class Detection:
    """A single object detection result."""
    bbox: list[float]           # [x1, y1, x2, y2] in pixels
    class_id: int               # Original model class ID
    class_name: str             # Human-readable class name
    confidence: float           # Detection confidence [0, 1]

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center point (foot position estimate)."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3],
        )

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class TrackedDetection:
    """A detection with an assigned track ID."""
    detection: Detection
    track_id: int


@dataclass
class PersonState:
    """Temporal state for a tracked person."""
    track_id: int
    # Helmet state
    no_hardhat_frames: int = 0
    has_hardhat_frames: int = 0
    # Zone state
    in_zone_frames: int = 0
    current_zone_id: Optional[str] = None
    # Cooldown tracking: event_type -> last alert timestamp
    last_alert_time: dict[str, float] = field(default_factory=dict)
    # Last seen
    last_seen_frame: int = 0
    last_seen_time: float = 0.0
    # Current status
    helmet_status: str = "unknown"  # "hardhat", "no_hardhat", "unknown"


@dataclass
class FrameResult:
    """Complete result for a single processed frame."""
    frame_index: int
    timestamp: float                                # Time since start (seconds)
    detections: list[Detection] = field(default_factory=list)
    tracked_persons: list[TrackedDetection] = field(default_factory=list)
    tracked_helmets: list[TrackedDetection] = field(default_factory=list)
    person_states: dict[int, PersonState] = field(default_factory=dict)
    violations: list[ViolationEvent] = field(default_factory=list)
    fps: float = 0.0
    inference_time_ms: float = 0.0


@dataclass
class ViolationEvent:
    """A safety violation event to be recorded."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_id: str = ""
    frame_index: int = 0
    track_id: int = 0
    event_type: str = ""           # EventType value
    confidence: float = 0.0
    zone_id: str = ""
    evidence_path: str = ""
    bbox: list[float] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "event_id": self.event_id,
            "timestamp_utc": self.timestamp_utc,
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "confidence": self.confidence,
            "zone_id": self.zone_id,
            "evidence_path": self.evidence_path,
            "bbox": self.bbox,
            "details": self.details,
        }
