#!/usr/bin/env python3
"""
Vision Safety Monitor — One-command launcher.

Usage:
    python run.py
    python run.py --port 8501
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Vision Safety Monitor")
    parser.add_argument("--port", type=int, default=8501, help="Port to host the app (default: 8501)")
    args = parser.parse_args()

    app_path = Path(__file__).parent / "apps" / "streamlit_app.py"

    print(f"\n  🦺 Vision Safety Monitor")
    print(f"  ───────────────────────────────")
    print(f"  Starting on http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#e94560",
        "--theme.backgroundColor", "#0f0f23",
        "--theme.secondaryBackgroundColor", "#1a1a2e",
        "--theme.textColor", "#eaeaea",
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n  Shutting down...")


if __name__ == "__main__":
    main()
