"""
Video Source Adapter — Unified interface for webcam, video file, and RTSP.

Normalizes different input sources and exposes frame, frame index,
source timestamp, FPS, width, and height. Handles end-of-file
and camera read failures cleanly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for a single video frame and its metadata."""
    frame: np.ndarray          # BGR image
    frame_index: int           # 0-based frame number
    timestamp: float           # Seconds since start of stream
    source_fps: float          # Source FPS
    width: int                 # Frame width in pixels
    height: int                # Frame height in pixels


class VideoSource:
    """
    Unified video source for webcam, local file, or RTSP stream.

    Args:
        source: Camera index (int or "0"), file path (str), or RTSP URL.
        loop: Whether to loop file playback.
    """

    def __init__(self, source: str | int, loop: bool = False):
        self._source_raw = source
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._start_time = 0.0
        self._is_file = False
        self._is_webcam = False
        self._is_rtsp = False

        # Determine source type
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self._source = int(source) if isinstance(source, str) else source
            self._is_webcam = True
        elif isinstance(source, str) and source.startswith("rtsp://"):
            self._source = source
            self._is_rtsp = True
        else:
            self._source = str(source)
            self._is_file = True
            if not Path(self._source).is_file():
                raise FileNotFoundError(f"Video file not found: {self._source}")

    def open(self) -> None:
        """Open the video source."""
        logger.info("Opening video source: %s (type=%s)",
                     self._source_raw, self.source_type)

        if self._is_rtsp:
            self._cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self._source_raw}")

        self._frame_index = 0
        self._start_time = time.monotonic()

        logger.info("  Resolution: %dx%d", self.width, self.height)
        logger.info("  FPS: %.1f", self.fps)
        if self._is_file:
            logger.info("  Total frames: %d", self.total_frames)

    def close(self) -> None:
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Video source closed: %s", self._source_raw)

    def read(self) -> Optional[FrameData]:
        """
        Read the next frame.

        Returns:
            FrameData if successful, None if end-of-stream or error.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()

        if not ret:
            if self._is_file and self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None

        elapsed = time.monotonic() - self._start_time
        frame_data = FrameData(
            frame=frame,
            frame_index=self._frame_index,
            timestamp=elapsed,
            source_fps=self.fps,
            width=frame.shape[1],
            height=frame.shape[0],
        )
        self._frame_index += 1
        return frame_data

    def frames(self) -> Iterator[FrameData]:
        """Iterator over all frames."""
        while True:
            frame_data = self.read()
            if frame_data is None:
                break
            yield frame_data

    @property
    def source_type(self) -> str:
        if self._is_webcam:
            return "webcam"
        elif self._is_rtsp:
            return "rtsp"
        else:
            return "file"

    @property
    def source_id(self) -> str:
        """Human-readable source identifier."""
        if self._is_webcam:
            return f"webcam-{self._source}"
        elif self._is_file:
            return Path(self._source).stem
        else:
            return str(self._source_raw)

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    @property
    def width(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def total_frames(self) -> int:
        if self._cap is None or not self._is_file:
            return -1
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def is_live(self) -> bool:
        return self._is_webcam or self._is_rtsp

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
