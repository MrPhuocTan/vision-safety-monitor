#!/usr/bin/env python3
"""
Vision Safety Monitor — Web Dashboard.

Usage:
    python run.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.safety_monitor.config import load_config
from src.safety_monitor.pipeline import SafetyMonitorPipeline

# ─── Page Config ───
st.set_page_config(
    page_title="Safety Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Top bar */
    .top-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .top-bar h1 {
        color: #fff;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 700;
    }
    .top-bar .badge {
        background: #e94560;
        color: #fff;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-card .number {
        font-size: 2rem;
        font-weight: 800;
        color: #e94560;
    }
    .stat-card .label {
        color: #8888aa;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .stat-card.green .number { color: #10b981; }
    .stat-card.blue .number { color: #3b82f6; }

    /* Video container */
    .video-container {
        border: 2px solid #2a2a4a;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Sidebar cleanup */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


def load_events(path: str) -> list[dict]:
    """Load events from JSONL file."""
    events = []
    p = Path(path)
    if p.exists():
        for line in p.read_text().strip().split("\n"):
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def render_stat_card(label: str, value: str, variant: str = "") -> str:
    cls = f"stat-card {variant}" if variant else "stat-card"
    return f'<div class="{cls}"><div class="number">{value}</div><div class="label">{label}</div></div>'


def main():
    # ─── Top Bar ───
    st.markdown("""
    <div class="top-bar">
        <h1>🦺 Safety Monitor</h1>
        <span class="badge">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    # ─── Sidebar (Settings) ───
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        config_path = "configs/app.yaml"
        weights = st.text_input("Weights", value="models/best.pt")
        device = st.selectbox("Device", ["cpu", "mps", "cuda"], index=0)
        conf = st.slider("Sensitivity", 0.1, 0.9, 0.35, 0.05,
                          help="Lower = detect more (may include false positives)")
        helmet_frames = st.slider("Violation threshold (frames)", 1, 30, 10,
                                  help="Consecutive frames before flagging a violation")
        cooldown = st.slider("Alert cooldown (seconds)", 1.0, 120.0, 30.0, 1.0)

    # ─── Tabs ───
    tab_monitor, tab_history, tab_evidence = st.tabs([
        "📹 Monitor", "📋 Event Log", "📸 Evidence"
    ])

    # ═══════════════════════════════════════════
    # TAB 1: MONITOR
    # ═══════════════════════════════════════════
    with tab_monitor:

        # Source selection row
        col_source, col_action = st.columns([3, 1])

        with col_source:
            source_type = st.radio(
                "Video Source",
                ["Sample Video", "Upload File", "Webcam"],
                horizontal=True, label_visibility="collapsed"
            )

        video_path = None

        if source_type == "Sample Video":
            samples = {
                "🎬 Hardhat Demo": "dataset/source_files/source_files/hardhat.mp4",
                "🎬 Japan PPE": "dataset/source_files/source_files/JapanPPE.mp4",
                "🎬 Indian Workers": "dataset/source_files/source_files/indianworkers.mp4",
            }
            with col_action:
                pick = st.selectbox("Video", list(samples.keys()), label_visibility="collapsed")
            video_path = samples[pick]

        elif source_type == "Upload File":
            uploaded = st.file_uploader(
                "Drop a video file here",
                type=["mp4", "avi", "mov", "mkv"],
                label_visibility="collapsed",
            )
            if uploaded:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded.getvalue())
                    video_path = tmp.name

        elif source_type == "Webcam":
            video_path = 0

        # Start button
        if video_path is not None:
            if st.button("▶  Start Monitoring", type="primary", use_container_width=True):

                overrides = {
                    "model": {
                        "weights": weights,
                        "device": device,
                        "conf_threshold": conf,
                        "imgsz": 640,
                    },
                    "rules": {
                        "helmet_persistence_frames": helmet_frames,
                        "alert_cooldown_seconds": cooldown,
                    },
                    "zones": [],
                }

                try:
                    config = load_config(config_path, overrides)
                    pipeline = SafetyMonitorPipeline(config)

                    with st.spinner("Initializing system..."):
                        pipeline.initialize()

                    from src.safety_monitor.video_source import VideoSource
                    video = VideoSource(video_path)
                    video.open()

                    # Layout: video left, stats right
                    col_vid, col_stats = st.columns([3, 1])

                    with col_vid:
                        video_frame = st.empty()

                    with col_stats:
                        st_fps = st.empty()
                        st_persons = st.empty()
                        st_violations = st.empty()
                        st_events = st.empty()
                        progress = st.empty()

                    total = video.total_frames
                    processed = 0

                    for frame_data in video.frames():
                        annotated, result = pipeline.process_frame(
                            frame_data.frame,
                            frame_data.frame_index,
                            frame_data.timestamp,
                            video.source_id,
                        )

                        # Show video
                        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_frame.image(img_rgb, channels="RGB", use_container_width=True)

                        # Record events
                        for event in result.violations:
                            pipeline.event_manager.record_event(event, frame_data.frame)

                        processed += 1

                        # Update stats every 5 frames (perf optimization)
                        if processed % 5 == 0 or processed == 1:
                            n_persons = len(result.tracked_persons)
                            n_violations = sum(
                                1 for s in result.person_states.values()
                                if s.helmet_status == "no_hardhat"
                            )

                            st_fps.markdown(render_stat_card("FPS", f"{result.fps:.0f}", "green"), unsafe_allow_html=True)
                            st_persons.markdown(render_stat_card("Persons", str(n_persons), "blue"), unsafe_allow_html=True)
                            st_violations.markdown(render_stat_card("Violations", str(n_violations)), unsafe_allow_html=True)
                            st_events.markdown(render_stat_card("Total Events", str(pipeline.event_manager.event_count), "blue"), unsafe_allow_html=True)

                            if total > 0:
                                pct = min(processed / total, 1.0)
                                progress.progress(pct, text=f"{processed}/{total}")

                    video.close()

                    st.success(f"✅ Complete — {processed} frames processed, {pipeline.event_manager.event_count} violations recorded.")

                except FileNotFoundError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.exception(e)
        else:
            st.info("Select a video source above, then press **Start Monitoring**.")

    # ═══════════════════════════════════════════
    # TAB 2: EVENT LOG
    # ═══════════════════════════════════════════
    with tab_history:
        events = load_events("artifacts/outputs/events.jsonl")

        if events:
            # Filters
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                types = list(set(e.get("event_type", "") for e in events))
                sel_types = st.multiselect("Type", types, default=types)
            with col2:
                tracks = sorted(set(e.get("track_id", 0) for e in events))
                sel_tracks = st.multiselect("Person ID", tracks, default=tracks)
            with col3:
                if st.button("🔄 Refresh"):
                    st.rerun()

            filtered = [
                e for e in events
                if e.get("event_type") in sel_types and e.get("track_id") in sel_tracks
            ]

            # Summary row
            c1, c2, c3 = st.columns(3)
            c1.markdown(render_stat_card("Total Events", str(len(filtered))), unsafe_allow_html=True)

            no_hat = sum(1 for e in filtered if e.get("event_type") == "NO_HARDHAT")
            c2.markdown(render_stat_card("No Hardhat", str(no_hat)), unsafe_allow_html=True)

            unique_persons = len(set(e.get("track_id") for e in filtered))
            c3.markdown(render_stat_card("Unique Persons", str(unique_persons), "blue"), unsafe_allow_html=True)

            st.markdown("---")

            # Table
            if filtered:
                import pandas as pd
                df = pd.DataFrame(filtered)
                show_cols = [c for c in ["timestamp_utc", "event_type", "track_id", "frame_index", "confidence", "details"] if c in df.columns]
                st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

                # Export
                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button("📥 Export CSV", df.to_csv(index=False), "events.csv", "text/csv")
                with col_b:
                    st.download_button("📥 Export JSON", json.dumps(filtered, indent=2), "events.json", "application/json")
        else:
            st.info("No events recorded yet. Run a monitoring session first.")

    # ═══════════════════════════════════════════
    # TAB 3: EVIDENCE
    # ═══════════════════════════════════════════
    with tab_evidence:
        evidence_dir = Path("artifacts/evidence")

        if evidence_dir.exists():
            images = sorted(evidence_dir.glob("*.jpg"), reverse=True)

            if images:
                st.markdown(f"**{len(images)}** evidence captures")

                cols = st.columns(4)
                for i, img_path in enumerate(images[:16]):
                    with cols[i % 4]:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # Clean caption: extract type and person ID
                            name = img_path.stem
                            parts = name.split("_")
                            caption = f"{parts[0]}_{parts[1]} — ID {parts[2].replace('track', '#')}" if len(parts) >= 3 else name
                            st.image(img_rgb, caption=caption, use_container_width=True)
            else:
                st.info("No evidence captures yet.")
        else:
            st.info("No evidence directory found.")


if __name__ == "__main__":
    main()
