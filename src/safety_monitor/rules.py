"""
Safety Rule Engine — Helmet compliance and restricted zone rules.

Pure business logic with no model dependencies. Manages temporal state,
persistence thresholds, cooldowns, and event deduplication.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from shapely.geometry import Point, Polygon

from .config import AppConfig, ZoneDefinition
from .schemas import (
    EventType,
    PersonState,
    TrackedDetection,
    ViolationEvent,
)

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Safety compliance rule engine.

    Evaluates:
    1. Helmet violations: Person detected as NO-Hardhat for N consecutive frames.
    2. Zone intrusions: Person enters restricted zone for N consecutive frames.

    Features:
    - Per-track temporal state management.
    - Configurable persistence thresholds (no single-frame alerts).
    - Per-track cooldown to prevent duplicate alert spam.
    - Track timeout for clearing stale tracks.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._person_states: dict[int, PersonState] = {}
        self._zone_polygons: dict[str, Polygon] = {}
        self._frame_width = 1
        self._frame_height = 1

        # Build zone polygons
        for zone in config.zones:
            self._build_zone_polygon(zone)

    def _build_zone_polygon(self, zone: ZoneDefinition) -> None:
        """Create a Shapely polygon from normalized zone coordinates."""
        if len(zone.polygon) < 3:
            logger.warning("Zone %s has fewer than 3 vertices, skipping", zone.id)
            return

        self._zone_polygons[zone.id] = Polygon(zone.polygon)
        logger.info("Zone '%s' (%s) registered with %d vertices",
                     zone.name, zone.id, len(zone.polygon))

    def set_frame_size(self, width: int, height: int) -> None:
        """Set frame dimensions for coordinate normalization."""
        self._frame_width = max(1, width)
        self._frame_height = max(1, height)

    def get_or_create_state(self, track_id: int) -> PersonState:
        """Get or create temporal state for a tracked person."""
        if track_id not in self._person_states:
            self._person_states[track_id] = PersonState(track_id=track_id)
        return self._person_states[track_id]

    def update_helmet_state(
        self,
        track_id: int,
        helmet_status: str,
        frame_index: int,
        current_time: float,
    ) -> None:
        """Update helmet compliance state for a person."""
        state = self.get_or_create_state(track_id)
        state.last_seen_frame = frame_index
        state.last_seen_time = current_time
        state.helmet_status = helmet_status

        if helmet_status == "no_hardhat":
            state.no_hardhat_frames += 1
            state.has_hardhat_frames = 0
        elif helmet_status == "hardhat":
            state.has_hardhat_frames += 1
            state.no_hardhat_frames = 0
        # "unknown" doesn't reset either counter

    def check_zone(
        self,
        track_id: int,
        person_detection: TrackedDetection,
        frame_index: int,
        current_time: float,
    ) -> Optional[str]:
        """
        Check if a person is inside any restricted zone.

        Uses the foot-point (bottom-center) of the person bbox
        for zone containment check.

        Returns:
            Zone ID if person is in a zone, None otherwise.
        """
        if not self._zone_polygons:
            return None

        state = self.get_or_create_state(track_id)
        foot = person_detection.detection.foot_point

        # Normalize foot point to [0, 1]
        norm_x = foot[0] / self._frame_width
        norm_y = foot[1] / self._frame_height
        point = Point(norm_x, norm_y)

        for zone_id, polygon in self._zone_polygons.items():
            if polygon.contains(point):
                state.in_zone_frames += 1
                state.current_zone_id = zone_id
                return zone_id

        # Not in any zone — reset
        state.in_zone_frames = 0
        state.current_zone_id = None
        return None

    def evaluate(
        self,
        tracked_persons: list[TrackedDetection],
        helmet_status: dict[int, str],
        frame_index: int,
        current_time: float,
        source_id: str = "",
    ) -> list[ViolationEvent]:
        """
        Evaluate all rules for the current frame.

        Args:
            tracked_persons: Person detections with track IDs.
            helmet_status: Map of track_id → helmet status.
            frame_index: Current frame index.
            current_time: Current timestamp (seconds).
            source_id: Video source identifier.

        Returns:
            List of new ViolationEvents (only for persistent, non-duplicate violations).
        """
        violations = []
        rules_cfg = self.config.rules

        # Clean up stale tracks
        self._cleanup_stale_tracks(current_time)

        for person in tracked_persons:
            track_id = person.track_id
            if track_id < 0:
                continue

            # Update helmet state
            h_status = helmet_status.get(track_id, "unknown")
            self.update_helmet_state(track_id, h_status, frame_index, current_time)

            state = self._person_states[track_id]

            # --- Rule 1: Helmet violation ---
            if state.no_hardhat_frames >= rules_cfg.helmet_persistence_frames:
                if self._can_alert(track_id, EventType.NO_HARDHAT.value, current_time):
                    event = ViolationEvent(
                        source_id=source_id,
                        frame_index=frame_index,
                        track_id=track_id,
                        event_type=EventType.NO_HARDHAT.value,
                        confidence=person.detection.confidence,
                        bbox=person.detection.bbox,
                        details=f"Person #{track_id} without hardhat for "
                                f"{state.no_hardhat_frames} frames",
                    )
                    violations.append(event)
                    state.last_alert_time[EventType.NO_HARDHAT.value] = current_time

            # --- Rule 2: Zone intrusion ---
            zone_id = self.check_zone(track_id, person, frame_index, current_time)
            if zone_id and state.in_zone_frames >= rules_cfg.zone_persistence_frames:
                if self._can_alert(track_id, EventType.ZONE_INTRUSION.value, current_time):
                    event = ViolationEvent(
                        source_id=source_id,
                        frame_index=frame_index,
                        track_id=track_id,
                        event_type=EventType.ZONE_INTRUSION.value,
                        confidence=person.detection.confidence,
                        zone_id=zone_id,
                        bbox=person.detection.bbox,
                        details=f"Person #{track_id} in restricted zone '{zone_id}' for "
                                f"{state.in_zone_frames} frames",
                    )
                    violations.append(event)
                    state.last_alert_time[EventType.ZONE_INTRUSION.value] = current_time

            # --- Rule 3: Combined (no hardhat in zone) ---
            if (zone_id
                    and state.no_hardhat_frames >= rules_cfg.helmet_persistence_frames
                    and state.in_zone_frames >= rules_cfg.zone_persistence_frames):
                if self._can_alert(track_id, EventType.NO_HARDHAT_IN_ZONE.value, current_time):
                    event = ViolationEvent(
                        source_id=source_id,
                        frame_index=frame_index,
                        track_id=track_id,
                        event_type=EventType.NO_HARDHAT_IN_ZONE.value,
                        confidence=person.detection.confidence,
                        zone_id=zone_id,
                        bbox=person.detection.bbox,
                        details=f"Person #{track_id} without hardhat in zone '{zone_id}'",
                    )
                    violations.append(event)
                    state.last_alert_time[EventType.NO_HARDHAT_IN_ZONE.value] = current_time

        return violations

    def _can_alert(self, track_id: int, event_type: str, current_time: float) -> bool:
        """Check if cooldown has elapsed for this track + event type."""
        state = self._person_states.get(track_id)
        if state is None:
            return True

        # If this event type has never been alerted, allow it
        if event_type not in state.last_alert_time:
            return True

        last_time = state.last_alert_time[event_type]
        elapsed = current_time - last_time
        return elapsed >= self.config.rules.alert_cooldown_seconds

    def _cleanup_stale_tracks(self, current_time: float) -> None:
        """Remove tracks that haven't been seen for a while."""
        timeout = self.config.rules.track_timeout_seconds
        stale = [
            tid for tid, state in self._person_states.items()
            if current_time - state.last_seen_time > timeout
        ]
        for tid in stale:
            del self._person_states[tid]

    @property
    def active_tracks(self) -> dict[int, PersonState]:
        """Get all active person states."""
        return dict(self._person_states)

    @property
    def active_violations_count(self) -> int:
        """Count currently active violations (persons in violation state)."""
        count = 0
        for state in self._person_states.values():
            if state.no_hardhat_frames >= self.config.rules.helmet_persistence_frames:
                count += 1
            if state.in_zone_frames >= self.config.rules.zone_persistence_frames:
                count += 1
        return count

    def reset(self) -> None:
        """Reset all rule engine state."""
        self._person_states.clear()
        logger.info("Rule engine state reset")
