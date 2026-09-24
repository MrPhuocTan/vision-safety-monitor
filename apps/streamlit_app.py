#!/usr/bin/env python3
"""
Streamlit Dashboard — Safety Monitor Web Interface.

Provides video upload, configuration, result playback, event browsing,
and evidence viewer.

Usage:
    streamlit run apps/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.safety_monitor.config import load_config
from src.safety_monitor.pipeline import SafetyMonitorPipeline

# Page config
st.set_page_config(
    page_title="Vision Safety Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_events_from_file(events_path: str) -> list[dict]:
    """Load events from JSONL file."""
    events = []
    path = Path(events_path)
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events


def main():
    # --- Header ---
    st.title("🦺 Vision Safety Monitor")
    st.markdown("Real-Time Safety Helmet Compliance and Restricted-Zone Monitoring")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Config file
        config_path = st.text_input(
            "Config file", value="configs/app.yaml",
            help="Path to application config YAML"
        )

        # Model settings
        st.subheader("Model")
        weights = st.text_input("Weights path", value="models/best.pt")
        device = st.selectbox("Device", ["cpu", "cuda", "mps"], index=0)
        conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
        imgsz = st.selectbox("Image size", [320, 416, 640, 1280], index=2)

        # Rule settings
        st.subheader("Rules")
        helmet_persist = st.slider("Helmet violation frames", 1, 30, 10)
        cooldown = st.slider("Alert cooldown (s)", 1.0, 120.0, 30.0, 1.0)

    # --- Main Area: Tabs ---
    tab_upload, tab_events, tab_evidence, tab_results = st.tabs([
        "📹 Video Processing", "📋 Event History", "🖼️ Evidence", "📊 Results"
    ])

    # --- Tab: Video Processing ---
    with tab_upload:
        st.header("Video Upload & Processing")

        source_type = st.radio(
            "Select Video Source", 
            ["Sample Video", "Upload New Video", "Webcam (Live)"], 
            horizontal=True
        )

        video_path_to_process = None

        if source_type == "Sample Video":
            sample_videos = {
                "Hardhat Demo (hardhat.mp4)": "dataset/source_files/source_files/hardhat.mp4",
                "Japan PPE (JapanPPE.mp4)": "dataset/source_files/source_files/JapanPPE.mp4",
                "Indian Workers (indianworkers.mp4)": "dataset/source_files/source_files/indianworkers.mp4"
            }
            selected_sample = st.selectbox("Choose a sample video", list(sample_videos.keys()))
            video_path_to_process = sample_videos[selected_sample]

        elif source_type == "Webcam (Live)":
            video_path_to_process = 0
            st.info("Webcam processing will start in a separate window. Press 'q' in that window to stop.")

        else:
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=["mp4", "avi", "mov", "mkv"],
                help="Upload a construction site video for safety analysis"
            )
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    video_path_to_process = tmp.name

        col1, col2 = st.columns(2)
        with col1:
            save_output = st.checkbox("Save output video", value=True)
        with col2:
            show_progress = st.checkbox("Show progress", value=True)

        if video_path_to_process is not None:
            if st.button("🚀 Start Processing", type="primary"):
                # Build config overrides
                overrides = {
                    "model": {
                        "weights": weights,
                        "device": device,
                        "conf_threshold": conf_threshold,
                        "imgsz": imgsz,
                    },
                    "rules": {
                        "helmet_persistence_frames": helmet_persist,
                        "alert_cooldown_seconds": cooldown,
                    },
                    "zones": [],
                }

                try:
                    config = load_config(config_path, overrides)
                    pipeline = SafetyMonitorPipeline(config)

                    with st.spinner("Loading model..."):
                        pipeline.initialize()

                    # Process video
                    from src.safety_monitor.video_source import VideoSource

                    video = VideoSource(video_path_to_process)
                    video.open()

                    progress_bar = st.progress(0) if show_progress else None
                    status_text = st.empty()
                    result_placeholder = st.empty()

                    total_frames = video.total_frames
                    processed = 0
                    all_violations = []

                    for frame_data in video.frames():
                        annotated, result = pipeline.process_frame(
                            frame_data.frame,
                            frame_data.frame_index,
                            frame_data.timestamp,
                            video.source_id,
                        )

                        # Hiển thị trực tiếp video ngay trên trình duyệt
                        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        result_placeholder.image(img_rgb, channels="RGB", use_container_width=True)

                        for event in result.violations:
                            pipeline.event_manager.record_event(event, frame_data.frame)
                            all_violations.append(event)

                        processed += 1

                        if progress_bar and total_frames > 0:
                            progress_bar.progress(min(processed / total_frames, 1.0))

                        if processed % 30 == 0:
                            status_text.text(
                                f"Frame {processed}/{total_frames} | "
                                f"FPS: {result.fps:.1f} | "
                                f"Events: {pipeline.event_manager.event_count}"
                            )

                    video.close()

                    st.success(
                        f"✅ Processing complete! "
                        f"Processed {processed} frames, "
                        f"found {pipeline.event_manager.event_count} events."
                    )

                    # Show summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Frames", processed)
                    col2.metric("Events", pipeline.event_manager.event_count)
                    col3.metric("Avg FPS", f"{result.fps:.1f}")

                except FileNotFoundError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.exception(e)

        else:
            st.info(
                "👆 Please select a video source above to begin safety analysis."
            )

    # --- Tab: Event History ---
    with tab_events:
        st.header("Event History")

        events_path = st.text_input(
            "Events file", value="artifacts/outputs/events.jsonl"
        )

        if st.button("🔄 Refresh Events"):
            st.rerun()

        events = load_events_from_file(events_path)

        if events:
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                event_types = list(set(e.get("event_type", "") for e in events))
                filter_type = st.multiselect("Filter by type", event_types, default=event_types)
            with col2:
                track_ids = sorted(set(e.get("track_id", 0) for e in events))
                filter_track = st.multiselect("Filter by track", track_ids, default=track_ids)

            # Apply filters
            filtered = [
                e for e in events
                if e.get("event_type") in filter_type
                and e.get("track_id") in filter_track
            ]

            st.metric("Total Events", len(filtered))

            # Display as table
            if filtered:
                import pandas as pd
                df = pd.DataFrame(filtered)
                display_cols = [
                    c for c in ["timestamp_utc", "event_type", "track_id",
                                "frame_index", "confidence", "zone_id", "details"]
                    if c in df.columns
                ]
                st.dataframe(df[display_cols], use_container_width=True)

                # Export
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📥 Export CSV", csv_data, "events.csv", "text/csv"
                    )
                with col2:
                    json_data = json.dumps(filtered, indent=2)
                    st.download_button(
                        "📥 Export JSON", json_data, "events.json", "application/json"
                    )
        else:
            st.info("No events found. Process a video first.")

    # --- Tab: Evidence ---
    with tab_evidence:
        st.header("Evidence Images")

        evidence_dir = Path("artifacts/evidence")
        if evidence_dir.exists():
            images = sorted(evidence_dir.glob("*.jpg"), reverse=True)

            if images:
                st.text(f"Found {len(images)} evidence images")

                # Grid display
                cols = st.columns(3)
                for i, img_path in enumerate(images[:12]):
                    with cols[i % 3]:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, caption=img_path.name, use_container_width=True)
            else:
                st.info("No evidence images found. Process a video first.")
        else:
            st.info("Evidence directory does not exist yet.")

    # --- Tab: Results ---
    with tab_results:
        st.header("Evaluation & Benchmark Results")

        # Show evaluation results if available
        eval_files = list(Path("reports").glob("evaluation_*.json")) if Path("reports").exists() else []
        if eval_files:
            for eval_file in eval_files:
                with open(eval_file) as f:
                    eval_data = json.load(f)

                st.subheader(f"Evaluation: {eval_file.name}")
                metrics = eval_data.get("metrics", {})

                cols = st.columns(4)
                cols[0].metric("mAP@50", f"{metrics.get('mAP50', 0):.4f}")
                cols[1].metric("mAP@50-95", f"{metrics.get('mAP50_95', 0):.4f}")
                cols[2].metric("Precision", f"{metrics.get('precision', 0):.4f}")
                cols[3].metric("Recall", f"{metrics.get('recall', 0):.4f}")

                if "per_class" in metrics:
                    st.subheader("Per-Class Results")
                    import pandas as pd
                    pc_data = []
                    for name, vals in metrics["per_class"].items():
                        pc_data.append({
                            "Class": name,
                            "AP@50": vals.get("ap50", 0),
                            "AP@50-95": vals.get("ap50_95", 0),
                            "Precision": vals.get("precision", 0),
                            "Recall": vals.get("recall", 0),
                        })
                    st.dataframe(pd.DataFrame(pc_data), use_container_width=True)
        else:
            st.info("No evaluation results found. Run `python scripts/evaluate.py` first.")

        # Show benchmark results
        bench_files = list(Path("artifacts/benchmarks").glob("benchmark_*.json")) if Path("artifacts/benchmarks").exists() else []
        if bench_files:
            st.divider()
            for bench_file in bench_files:
                with open(bench_file) as f:
                    bench_data = json.load(f)

                st.subheader(f"Benchmark: {bench_file.name}")
                inf = bench_data.get("inference_time_ms", {})
                fps = bench_data.get("fps", {})

                cols = st.columns(4)
                cols[0].metric("Mean FPS", f"{fps.get('mean', 0):.1f}")
                cols[1].metric("Mean Latency", f"{inf.get('mean', 0):.1f} ms")
                cols[2].metric("P95 Latency", f"{inf.get('p95', 0):.1f} ms")
                cols[3].metric("P99 Latency", f"{inf.get('p99', 0):.1f} ms")


if __name__ == "__main__":
    main()
