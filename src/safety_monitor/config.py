"""
Configuration loader for the Vision Safety Monitor.

Loads YAML config with CLI override support. All magic constants
are centralized here and in configs/app.yaml.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model inference configuration."""
    weights: str = "models/best.pt"
    device: str = "cpu"
    imgsz: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_det: int = 100


@dataclass
class ClassConfig:
    """Target class configuration."""
    target_ids: list[int] = field(default_factory=lambda: [0, 2, 5])
    names: dict[int, str] = field(default_factory=lambda: {0: "Hardhat", 2: "NO-Hardhat", 5: "Person"})
    person_class_id: int = 5
    hardhat_class_id: int = 0
    no_hardhat_class_id: int = 2


@dataclass
class TrackerConfig:
    """Object tracker configuration."""
    type: str = "bytetrack"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8


@dataclass
class AssociationConfig:
    """PPE-to-Person association configuration."""
    iou_threshold: float = 0.3
    vertical_ratio: float = 0.5


@dataclass
class RulesConfig:
    """Safety rule engine configuration."""
    helmet_persistence_frames: int = 10
    zone_persistence_frames: int = 15
    alert_cooldown_seconds: float = 30.0
    track_timeout_seconds: float = 5.0


@dataclass
class ZoneDefinition:
    """A single restricted zone."""
    id: str = "zone-1"
    name: str = "Restricted Area"
    polygon: list[list[float]] = field(
        default_factory=lambda: [[0.1, 0.6], [0.4, 0.6], [0.4, 0.95], [0.1, 0.95]]
    )
    color: list[int] = field(default_factory=lambda: [0, 0, 255])


@dataclass
class OutputConfig:
    """Output and storage configuration."""
    save_video: bool = True
    save_evidence: bool = True
    evidence_dir: str = "artifacts/evidence"
    output_dir: str = "artifacts/outputs"
    events_file: str = "artifacts/outputs/events.jsonl"
    video_codec: str = "mp4v"
    video_fps: float | None = None


@dataclass
class DisplayConfig:
    """Visualization / overlay configuration."""
    show_fps: bool = True
    show_violation_count: bool = True
    show_zone_overlay: bool = True
    show_track_ids: bool = True
    show_confidence: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.6
    person_color: list[int] = field(default_factory=lambda: [255, 200, 0])
    hardhat_color: list[int] = field(default_factory=lambda: [0, 255, 0])
    no_hardhat_color: list[int] = field(default_factory=lambda: [0, 0, 255])
    zone_color: list[int] = field(default_factory=lambda: [0, 0, 255])
    zone_alpha: float = 0.25
    alert_banner_color: list[int] = field(default_factory=lambda: [0, 0, 200])


@dataclass
class AppConfig:
    """Top-level application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    classes: ClassConfig = field(default_factory=ClassConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    zones: list[ZoneDefinition] = field(default_factory=lambda: [ZoneDefinition()])
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Convert a dict to a dataclass, handling nested dataclasses."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            logger.warning("Unknown config key: %s", key)
            continue

        ft = field_types[key]
        # Handle nested dataclasses
        if isinstance(value, dict) and hasattr(ft, '__dataclass_fields__'):
            kwargs[key] = _dict_to_dataclass(ft, value)
        elif isinstance(value, list) and key == "zones":
            kwargs[key] = [_dict_to_dataclass(ZoneDefinition, z) if isinstance(z, dict) else z for z in value]
        elif isinstance(value, dict) and key == "names":
            # Convert string keys to int for class name mapping
            kwargs[key] = {int(k): v for k, v in value.items()}
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """
    Load application configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. Defaults to configs/app.yaml.
        overrides: Dict of overrides to apply on top of YAML config.

    Returns:
        Populated AppConfig dataclass.
    """
    if config_path is None:
        config_path = Path("configs/app.yaml")
    else:
        config_path = Path(config_path)

    config_data: dict[str, Any] = {}

    if config_path.exists():
        logger.info("Loading config from %s", config_path)
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                config_data = loaded
    else:
        logger.warning("Config file not found: %s — using defaults", config_path)

    # Apply CLI overrides
    if overrides:
        config_data = _deep_update(config_data, overrides)

    # Build nested dataclass config
    config = AppConfig(
        model=_dict_to_dataclass(ModelConfig, config_data.get("model", {})),
        classes=_dict_to_dataclass(ClassConfig, config_data.get("classes", {})),
        tracker=_dict_to_dataclass(TrackerConfig, config_data.get("tracker", {})),
        association=_dict_to_dataclass(AssociationConfig, config_data.get("association", {})),
        rules=_dict_to_dataclass(RulesConfig, config_data.get("rules", {})),
        zones=[
            _dict_to_dataclass(ZoneDefinition, z) if isinstance(z, dict) else z
            for z in config_data.get("zones", [{"id": "zone-1"}])
        ],
        output=_dict_to_dataclass(OutputConfig, config_data.get("output", {})),
        display=_dict_to_dataclass(DisplayConfig, config_data.get("display", {})),
    )

    # Ensure output directories exist
    os.makedirs(config.output.evidence_dir, exist_ok=True)
    os.makedirs(config.output.output_dir, exist_ok=True)

    return config
