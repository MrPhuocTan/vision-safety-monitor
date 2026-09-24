"""
Safety Monitor Pipeline — Orchestrates the full processing pipeline.

Connects: Video Source → Detector → Tracker → Association → Rules → Renderer → Events

This is the main entry point for processing video frames through
the complete safety monitoring system.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .association import associate_ppe_to_persons
from .config import AppConfig, load_config
from .detector import Detector
from .events import EventManager
from .renderer import Renderer
from .rules import RuleEngine
from .schemas import FrameResult, TrackedDetection
from .tracker import Tracker
from .video_source import VideoSource

logger = logging.getLogger(__name__)


class SafetyMonitorPipeline:
    """
    Main processing pipeline for the Vision Safety Monitor.

    Processes video frames through detection, tracking, association,
    rule evaluation, rendering, and event recording.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = Detector(config)
        self.tracker = Tracker(config)
        self.rule_engine = RuleEngine(config)
        self.event_manager = EventManager(config)
        self.renderer = Renderer(config)

        # FPS tracking
        self._fps_window: deque[float] = deque(maxlen=30)
        self._frame_count = 0

        # Video writer
        self._video_writer: Optional[cv2.VideoWriter] = None

    def initialize(self) -> None:
        """Load models and prepare pipeline."""
        logger.info("Initializing Safety Monitor Pipeline...")
        self.detector.load()

        # Share model reference with tracker for integrated tracking
        self.tracker.set_model(self.detector._model)

        logger.info("Pipeline initialized and ready.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        source_id: str = "",
    ) -> tuple[np.ndarray, FrameResult]:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR image.
            frame_index: Frame number.
            timestamp: Seconds since stream start.
            source_id: Video source identifier.

        Returns:
            (annotated_frame, frame_result)
        """
        frame_start = time.perf_counter()

        # Set frame dimensions for rule engine
        h, w = frame.shape[:2]
        self.rule_engine.set_frame_size(w, h)

        # Step 1: Detect objects
        detections, inference_time = self.detector.detect(frame)

        # Step 2: Track objects (uses integrated YOLO tracking)
        tracked = self.tracker.track(frame, detections)

        # Step 3: Separate persons and helmet detections
        person_cls = self.config.classes.person_class_id
        tracked_persons = [t for t in tracked if t.detection.class_id == person_cls]
        tracked_helmets = [t for t in tracked if t.detection.class_id != person_cls]

        # Step 4: Associate helmets with persons
        helmet_status = associate_ppe_to_persons(tracked, self.config)

        # Step 5: Evaluate rules
        violations = self.rule_engine.evaluate(
            tracked_persons, helmet_status, frame_index, timestamp, source_id
        )

        # Step 6: Record events
        for event in violations:
            self.event_manager.record_event(event, frame)

        # Step 7: Compute FPS
        frame_time = time.perf_counter() - frame_start
        self._fps_window.append(frame_time)
        current_fps = len(self._fps_window) / sum(self._fps_window) if self._fps_window else 0

        # Step 8: Render overlays
        person_states = self.rule_engine.active_tracks
        annotated = self.renderer.render(
            frame, tracked, person_states, violations, current_fps, frame_index
        )

        # Build frame result
        result = FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            detections=[t.detection for t in tracked],
            tracked_persons=tracked_persons,
            tracked_helmets=tracked_helmets,
            person_states=person_states,
            violations=violations,
            fps=current_fps,
            inference_time_ms=inference_time,
        )

        self._frame_count += 1
        return annotated, result

    def run_video(
        self,
        source: str | int,
        show_display: bool = True,
        save_output: bool = False,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Process a complete video source.

        Args:
            source: Video source (camera index, file path, or RTSP URL).
            show_display: Whether to show the OpenCV window.
            save_output: Whether to save the output video.
            output_path: Custom output path (default: auto-generated).

        Returns:
            Summary statistics dict.
        """
        video = VideoSource(source)

        try:
            video.open()
        except Exception as e:
            logger.error("Failed to open video source: %s", e)
            return {"error": str(e)}

        source_id = video.source_id
        logger.info("Processing video: %s (FPS=%.1f, %dx%d)",
                     source_id, video.fps, video.width, video.height)

        # Setup video writer
        if save_output or self.config.output.save_video:
            if output_path is None:
                output_path = str(
                    Path(self.config.output.output_dir) / f"{source_id}_output.mp4"
                )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec)
            out_fps = self.config.output.video_fps or video.fps
            self._video_writer = cv2.VideoWriter(
                output_path, fourcc, out_fps, (video.width, video.height)
            )
            logger.info("Output video: %s", output_path)

        # Process frames
        total_frames = 0
        total_violations = 0
        paused = False

        try:
            for frame_data in video.frames():
                if paused and not video.is_live:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("p") or key == ord(" "):
                        paused = False
                    elif key == ord("q"):
                        break
                    continue

                annotated, result = self.process_frame(
                    frame_data.frame,
                    frame_data.frame_index,
                    frame_data.timestamp,
                    source_id,
                )

                total_frames += 1
                total_violations += len(result.violations)

                # Write output video
                if self._video_writer is not None:
                    self._video_writer.write(annotated)

                # Show display
                if show_display:
                    cv2.imshow("Safety Monitor", annotated)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        logger.info("User quit (q key)")
                        break
                    elif key == ord("p") or key == ord(" "):
                        paused = True
                        logger.info("Paused (press p/space to resume)")
                    elif key == ord("r"):
                        self.tracker.reset()
                        self.rule_engine.reset()
                        logger.info("Tracker and rules reset")

                # Progress logging
                if total_frames % 100 == 0:
                    logger.info(
                        "Frame %d | FPS: %.1f | Violations: %d | Events: %d",
                        total_frames, result.fps, total_violations,
                        self.event_manager.event_count,
                    )

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        finally:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            video.close()
            if show_display:
                cv2.destroyAllWindows()

        summary = {
            "source": str(source),
            "source_id": source_id,
            "total_frames": total_frames,
            "total_violations": total_violations,
            "total_events": self.event_manager.event_count,
            "events_by_type": self.event_manager.events_by_type,
            "output_video": output_path if save_output else None,
        }

        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("  Frames: %d", total_frames)
        logger.info("  Events: %d", self.event_manager.event_count)
        logger.info("  Events by type: %s", self.event_manager.events_by_type)
        logger.info("=" * 60)

        return summary

    def reset(self) -> None:
        """Reset all pipeline state."""
        self.tracker.reset()
        self.rule_engine.reset()
        self.event_manager.clear()
        self._fps_window.clear()
        self._frame_count = 0
