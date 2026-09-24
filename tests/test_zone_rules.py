"""Tests for restricted zone rules (point-in-polygon)."""

from __future__ import annotations

import pytest

from src.safety_monitor.config import AppConfig, RulesConfig, ZoneDefinition
from src.safety_monitor.rules import RuleEngine
from src.safety_monitor.schemas import Detection, TrackedDetection


def make_config(zones: list[ZoneDefinition] | None = None) -> AppConfig:
    """Create a minimal config for testing."""
    config = AppConfig()
    if zones is not None:
        config.zones = zones
    config.rules = RulesConfig(
        helmet_persistence_frames=3,
        zone_persistence_frames=3,
        alert_cooldown_seconds=10.0,
        track_timeout_seconds=60.0,
    )
    return config


def make_person(
    track_id: int,
    x1: float, y1: float, x2: float, y2: float,
    confidence: float = 0.9,
) -> TrackedDetection:
    """Create a person detection."""
    return TrackedDetection(
        detection=Detection(
            bbox=[x1, y1, x2, y2],
            class_id=5,
            class_name="Person",
            confidence=confidence,
        ),
        track_id=track_id,
    )


class TestPointInPolygon:
    """Test zone containment checks."""

    def test_person_inside_zone(self):
        """Person foot-point inside the zone polygon."""
        zone = ZoneDefinition(
            id="zone-1",
            polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        )
        config = make_config(zones=[zone])
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1, 20, 10, 80, 90)  # foot at (50, 90) → normalized (0.5, 0.9)

        zone_id = engine.check_zone(1, person, frame_index=0, current_time=0.0)
        assert zone_id == "zone-1"

    def test_person_outside_zone(self):
        """Person foot-point outside the zone polygon."""
        zone = ZoneDefinition(
            id="zone-1",
            polygon=[[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
        )
        config = make_config(zones=[zone])
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        # Person at far left, foot at (15, 90) → normalized (0.15, 0.9)
        person = make_person(1, 5, 10, 25, 90)
        zone_id = engine.check_zone(1, person, frame_index=0, current_time=0.0)
        assert zone_id is None

    def test_person_on_boundary(self):
        """Person exactly on zone boundary."""
        zone = ZoneDefinition(
            id="zone-1",
            polygon=[[0.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]],
        )
        config = make_config(zones=[zone])
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        # Person foot at (25, 80) → normalized (0.25, 0.8) — inside
        person = make_person(1, 10, 20, 40, 80)
        zone_id = engine.check_zone(1, person, frame_index=0, current_time=0.0)
        assert zone_id == "zone-1"

    def test_no_zones_configured(self):
        """No zones configured returns None."""
        config = make_config(zones=[])
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1, 10, 10, 50, 90)
        zone_id = engine.check_zone(1, person, frame_index=0, current_time=0.0)
        assert zone_id is None

    def test_multiple_zones(self):
        """Person can be in one of multiple zones."""
        zones = [
            ZoneDefinition(id="zone-left", polygon=[[0.0, 0.0], [0.4, 0.0], [0.4, 1.0], [0.0, 1.0]]),
            ZoneDefinition(id="zone-right", polygon=[[0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]]),
        ]
        config = make_config(zones=zones)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        # Person on the right: foot at (80, 90) → (0.8, 0.9)
        person = make_person(1, 70, 10, 90, 90)
        zone_id = engine.check_zone(1, person, frame_index=0, current_time=0.0)
        assert zone_id == "zone-right"

    def test_zone_persistence_counter_increments(self):
        """Zone persistence counter increases each frame person is in zone."""
        zone = ZoneDefinition(
            id="zone-1",
            polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        )
        config = make_config(zones=[zone])
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1, 20, 10, 80, 90)

        for i in range(5):
            engine.check_zone(1, person, frame_index=i, current_time=i * 0.033)

        state = engine.get_or_create_state(1)
        assert state.in_zone_frames == 5

    def test_zone_counter_resets_when_leaving(self):
        """Zone counter resets when person leaves the zone."""
        zone = ZoneDefinition(
            id="zone-1",
            polygon=[[0.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]],
        )
        config = make_config(zones=[zone])
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        # Person inside zone
        person_inside = make_person(1, 10, 60, 40, 90)
        engine.check_zone(1, person_inside, frame_index=0, current_time=0.0)
        engine.check_zone(1, person_inside, frame_index=1, current_time=0.033)
        assert engine.get_or_create_state(1).in_zone_frames == 2

        # Person moves outside zone
        person_outside = make_person(1, 60, 10, 90, 40)
        engine.check_zone(1, person_outside, frame_index=2, current_time=0.066)
        assert engine.get_or_create_state(1).in_zone_frames == 0
