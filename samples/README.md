# Sample Videos and Test Data

This folder contains sample videos and test data for the Vision Safety Monitor.

## Available Sample Videos

The dataset includes sample construction site videos in `dataset/source_files/source_files/`:

- `hardhat.mp4` — Workers on a construction site with hardhats
- `JapanPPE.mp4` — PPE compliance footage from Japan
- `indianworkers.mp4` — Indian construction workers

## Usage

```bash
# Process a sample video with CLI
python -m apps.cli --source dataset/source_files/source_files/hardhat.mp4 --save-output --no-display

# Copy a sample here for convenience
cp dataset/source_files/source_files/hardhat.mp4 samples/
```
