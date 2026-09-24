"""Tests for event deduplication — cooldown and per-track event limits."""

from __future__ import annotations

import pytest

from src.safety_monitor.config import AppConfig, RulesConfig, ZoneDefinition
from src.safety_monitor.rules import RuleEngine
from src.safety_monitor.schemas import Detection, EventType, TrackedDetection


def make_config(
    helmet_persistence: int = 2,
    cooldown: float = 5.0,
) -> AppConfig:
    config = AppConfig()
    config.rules = RulesConfig(
        helmet_persistence_frames=helmet_persistence,
        zone_persistence_frames=2,
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


class TestCooldown:
    """Test that cooldown prevents duplicate alert spam."""

    def test_cooldown_blocks_duplicate(self):
        """After an alert fires, same event shouldn't fire during cooldown."""
        config = make_config(helmet_persistence=2, cooldown=5.0)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)
        helmet_status = {1: "no_hardhat"}

        all_violations = []

        # Run enough frames to trigger first alert
        for i in range(10):
            violations = engine.evaluate(
                [person], helmet_status,
                frame_index=i, current_time=i * 0.1,  # 0.1s per frame
            )
            all_violations.extend(violations)

        # Should only have 1 alert (cooldown blocks duplicates)
        helmet_violations = [
            v for v in all_violations
            if v.event_type == EventType.NO_HARDHAT.value
        ]
        assert len(helmet_violations) == 1

    def test_alert_fires_after_cooldown(self):
        """After cooldown expires, a new alert should fire."""
        cooldown = 2.0
        config = make_config(helmet_persistence=2, cooldown=cooldown)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)
        helmet_status = {1: "no_hardhat"}

        all_violations = []

        # Phase 1: trigger first alert (frames 0-2, time 0-0.2)
        for i in range(5):
            violations = engine.evaluate(
                [person], helmet_status,
                frame_index=i, current_time=i * 0.1,
            )
            all_violations.extend(violations)

        first_count = len([
            v for v in all_violations if v.event_type == EventType.NO_HARDHAT.value
        ])
        assert first_count == 1

        # Phase 2: advance past cooldown (time > 2.0)
        for i in range(5, 10):
            violations = engine.evaluate(
                [person], helmet_status,
                frame_index=i, current_time=3.0 + i * 0.1,  # well past cooldown
            )
            all_violations.extend(violations)

        total_alerts = len([
            v for v in all_violations if v.event_type == EventType.NO_HARDHAT.value
        ])
        assert total_alerts == 2  # first + second after cooldown

    def test_different_tracks_independent_cooldown(self):
        """Different tracks should have independent cooldowns."""
        config = make_config(helmet_persistence=2, cooldown=10.0)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person1 = make_person(1)
        person2 = make_person(2)
        helmet_status = {1: "no_hardhat", 2: "no_hardhat"}

        all_violations = []
        for i in range(5):
            violations = engine.evaluate(
                [person1, person2], helmet_status,
                frame_index=i, current_time=i * 0.1,
            )
            all_violations.extend(violations)

        # Both tracks should trigger independently
        track1_alerts = [v for v in all_violations if v.track_id == 1]
        track2_alerts = [v for v in all_violations if v.track_id == 2]
        assert len(track1_alerts) >= 1
        assert len(track2_alerts) >= 1

    def test_leave_and_reenter_new_alert(self):
        """Track leaving and re-entering after cooldown should trigger new event."""
        cooldown = 1.0
        config = make_config(helmet_persistence=2, cooldown=cooldown)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(1)
        helmet_status_bad = {1: "no_hardhat"}
        helmet_status_good = {1: "hardhat"}

        all_violations = []

        # Phase 1: trigger alert
        for i in range(5):
            violations = engine.evaluate(
                [person], helmet_status_bad,
                frame_index=i, current_time=i * 0.1,
            )
            all_violations.extend(violations)

        # Phase 2: put on hardhat (reset counter)
        for i in range(5, 10):
            engine.evaluate(
                [person], helmet_status_good,
                frame_index=i, current_time=i * 0.1,
            )

        # Phase 3: take off hardhat again, past cooldown
        for i in range(10, 20):
            violations = engine.evaluate(
                [person], helmet_status_bad,
                frame_index=i, current_time=2.0 + i * 0.1,  # past cooldown
            )
            all_violations.extend(violations)

        helmet_alerts = [
            v for v in all_violations if v.event_type == EventType.NO_HARDHAT.value
        ]
        assert len(helmet_alerts) == 2  # first + re-violation


class TestEventConsistency:
    """Test that event records contain required fields."""

    def test_event_has_required_fields(self):
        """Violation events should have all required schema fields."""
        config = make_config(helmet_persistence=1, cooldown=0.0)
        engine = RuleEngine(config)
        engine.set_frame_size(100, 100)

        person = make_person(42)

        # Trigger an alert
        for i in range(3):
            violations = engine.evaluate(
                [person], {42: "no_hardhat"},
                frame_index=i, current_time=i * 0.033,
                source_id="test-camera",
            )
            if violations:
                event = violations[0]
                assert event.event_id  # UUID
                assert event.timestamp_utc  # ISO-8601
                assert event.source_id == "test-camera"
                assert event.frame_index >= 0
                assert event.track_id == 42
                assert event.event_type == EventType.NO_HARDHAT.value
                assert event.confidence > 0
                assert len(event.bbox) == 4
                assert event.details  # Non-empty description
                break
        else:
            pytest.fail("Expected a violation event within 3 frames")
