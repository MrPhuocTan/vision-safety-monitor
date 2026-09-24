#!/usr/bin/env python3
"""
Vision Safety Monitor — One-command launcher.

Automatically uses the project's virtual environment if available.

Usage:
    python run.py
    python run.py --port 8501
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / "venv" / "bin" / "python"


def ensure_venv():
    """Re-launch this script using the venv Python if we're not already in it."""
    # Already inside venv
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        return

    # Venv exists → re-exec with venv python
    if VENV_PYTHON.exists():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

    # No venv at all → try system python (hope deps are installed globally)
    print("⚠  No virtual environment found. Trying system Python...")


def main():
    ensure_venv()

    parser = argparse.ArgumentParser(description="Vision Safety Monitor")
    parser.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")
    args = parser.parse_args()

    app_path = PROJECT_DIR / "apps" / "streamlit_app.py"

    print(f"\n  🦺 Safety Monitor")
    print(f"  ─────────────────────────")
    print(f"  http://localhost:{args.port}")
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
        subprocess.run(cmd, cwd=str(PROJECT_DIR))
    except KeyboardInterrupt:
        print("\n  Shutting down...")


if __name__ == "__main__":
    main()
