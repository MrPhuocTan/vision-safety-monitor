#!/usr/bin/env bash
# =============================================================================
# Download Kaggle Dataset (Fallback)
#
# This script downloads the Construction Site Safety dataset from Kaggle.
# Only use this if the local dataset/ folder is missing, incomplete, or invalid.
#
# Prerequisites:
#   1. Install: pip install kaggle
#   2. Authenticate via one of:
#      a) Set KAGGLE_API_TOKEN environment variable
#      b) Place kaggle.json in ~/.kaggle/kaggle.json
#
# Usage:
#   bash scripts/download_kaggle_dataset.sh
# =============================================================================

set -euo pipefail

KAGGLE_SLUG="snehilsanyal/construction-site-safety-image-dataset-roboflow"
RAW_DIR="data/raw"

echo "=============================================="
echo "Kaggle Dataset Download (Fallback)"
echo "=============================================="
echo "Slug: ${KAGGLE_SLUG}"
echo "Destination: ${RAW_DIR}"
echo ""

# Check kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: Kaggle CLI not found."
    echo ""
    echo "Install it with:"
    echo "  pip install kaggle"
    echo ""
    echo "Then configure authentication:"
    echo "  Option 1: export KAGGLE_API_TOKEN='{\"username\":\"...\",\"key\":\"...\"}'"
    echo "  Option 2: Place kaggle.json at ~/.kaggle/kaggle.json"
    echo ""
    echo "See: https://github.com/Kaggle/kaggle-api#api-credentials"
    exit 1
fi

# Check authentication
if ! kaggle datasets list --max-size 1 &> /dev/null; then
    echo "ERROR: Kaggle authentication failed."
    echo ""
    echo "Configure authentication:"
    echo "  Option 1: export KAGGLE_API_TOKEN='{\"username\":\"...\",\"key\":\"...\"}'"
    echo "  Option 2: Place kaggle.json at ~/.kaggle/kaggle.json"
    exit 1
fi

# Create directory
mkdir -p "${RAW_DIR}"

# Download
echo "Downloading dataset..."
kaggle datasets download \
    -d "${KAGGLE_SLUG}" \
    -p "${RAW_DIR}" \
    --unzip

echo ""
echo "Download complete. Files in ${RAW_DIR}:"
ls -la "${RAW_DIR}"
echo ""
echo "Next step: Run dataset preparation:"
echo "  python scripts/prepare_dataset.py --source ${RAW_DIR}"
echo "=============================================="
