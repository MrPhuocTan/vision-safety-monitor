"""Tests for temporal rules — persistence thresholds and single-frame rejection."""

from __future__ import annotations

import pytest

from src.safety_monitor.config import AppConfig, RulesConfig, ZoneDefinition
from src.safety_monitor.rules import RuleEngine
from src.safety_monitor.schemas import Detection, EventType, TrackedDetection


def make_config(
    helmet_persistence: int = 5,
    zone_persistence: int = 5,
    cooldown: float = 30.0,
) -> AppConfig:
    config = AppConfig()
    config.rules = RulesConfig(
        helmet_persistence_frames=helmet_persistence,
        zone_persistence_frames=zone_persistence,
        alert_cooldown_seconds=cooldown,
        track_timeout_seconds=60.0,
    )
    config.zones = [
        ZoneDefinition(
            id="zone-1",
            polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        )
    ]
    return config


def make_person(track_id: int = 1) -> TrackedDetection:
    return TrackedDetection(
        detection=Detection(
            bbox=[20, 10, 80, 90],
            class_id=5,
            class_name="Person",
            confidence=0.9,
        ),
        track_id=track_id,
    )


class TestHelmetPersistence:
    """Test that single-frame violations don't trigger alerts."""

    def test_single_frame_no_alert(self):
        """A single no-hardhat frame should NOT trigger an alert."""
        config = make_config(helmet_persistence=5)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)
        helmet_status = {1: "no_hardhat"}

        violations = engine.evaluate(
            [person], helmet_status,
            frame_index=0, current_time=0.0,
        )

        assert len(violations) == 0

    def test_persistent_violation_triggers_alert(self):
        """After N persistent frames, an alert should trigger."""
        persistence = 3
        config = make_config(helmet_persistence=persistence)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)
        helmet_status = {1: "no_hardhat"}

        all_violations = []
        for i in range(persistence + 1):
            violations = engine.evaluate(
                [person], helmet_status,
                frame_index=i, current_time=i * 0.033,
            )
            all_violations.extend(violations)

        # Should have exactly 1 alert after persistence threshold is reached
        assert len(all_violations) == 1
        assert all_violations[0].event_type == EventType.NO_HARDHAT.value

    def test_hardhat_resets_counter(self):
        """Wearing a hardhat resets the no-hardhat counter."""
        config = make_config(helmet_persistence=5)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)

        # 3 frames of no-hardhat
        for i in range(3):
            engine.evaluate(
                [person], {1: "no_hardhat"},
                frame_index=i, current_time=i * 0.033,
            )

        state = engine.get_or_create_state(1)
        assert state.no_hardhat_frames == 3

        # Put on hardhat
        engine.evaluate(
            [person], {1: "hardhat"},
            frame_index=3, current_time=3 * 0.033,
        )

        state = engine.get_or_create_state(1)
        assert state.no_hardhat_frames == 0
        assert state.has_hardhat_frames == 1

    def test_unknown_status_no_reset(self):
        """Unknown status should not reset either counter."""
        config = make_config(helmet_persistence=10)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)

        # Build up no-hardhat count
        for i in range(3):
            engine.evaluate(
                [person], {1: "no_hardhat"},
                frame_index=i, current_time=i * 0.033,
            )

        state = engine.get_or_create_state(1)
        assert state.no_hardhat_frames == 3

        # Unknown frame
        engine.evaluate(
            [person], {1: "unknown"},
            frame_index=3, current_time=3 * 0.033,
        )

        # Counter should not reset
        state = engine.get_or_create_state(1)
        assert state.no_hardhat_frames == 3  # unchanged


class TestZonePersistence:
    """Test zone intrusion persistence thresholds."""

    def test_single_frame_zone_no_alert(self):
        """A single frame in zone should NOT trigger alert."""
        config = make_config(zone_persistence=5)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)
        helmet_status = {1: "hardhat"}

        violations = engine.evaluate(
            [person], helmet_status,
            frame_index=0, current_time=0.0,
        )

        assert len(violations) == 0

    def test_persistent_zone_triggers_alert(self):
        """After N frames in zone, an alert should trigger."""
        persistence = 3
        config = make_config(zone_persistence=persistence, helmet_persistence=100)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)
        helmet_status = {1: "hardhat"}

        all_violations = []
        for i in range(persistence + 1):
            violations = engine.evaluate(
                [person], helmet_status,
                frame_index=i, current_time=i * 0.033,
            )
            all_violations.extend(violations)

        zone_violations = [v for v in all_violations if v.event_type == EventType.ZONE_INTRUSION.value]
        assert len(zone_violations) == 1


class TestTrackTimeout:
    """Test stale track cleanup."""

    def test_stale_track_removed(self):
        """Tracks not seen for timeout period should be removed."""
        config = make_config()
        config.rules.track_timeout_seconds = 5.0
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        # Create a track
        person = make_person(1)
        engine.evaluate(
            [person], {1: "no_hardhat"},
            frame_index=0, current_time=0.0,
        )
        assert 1 in engine.active_tracks

        # Advance time past timeout with no detections
        engine.evaluate(
            [], {},
            frame_index=200, current_time=10.0,
        )

        assert 1 not in engine.active_tracks
