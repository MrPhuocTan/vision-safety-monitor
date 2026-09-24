# Local Dataset Inventory Report

**Inspection Date:** 2026-09-16T15:42:41.731632

**Source Path:** `/Users/mrtun/Documents/Work/Portfolio/Project/DTSS/dataset`

## File Counts

| Category | Count |
|---|---:|
| images | 2825 |
| labels | 2803 |
| archives | 0 |
| videos | 3 |
| configs | 3 |
| weights | 3 |
| notebooks | 1 |
| readmes | 0 |
| other | 3 |

## Annotation Format

`yolo`

## Class Mapping

| ID | Name |
|---|---|
| 0 | Hardhat |
| 1 | Mask |
| 2 | NO-Hardhat |
| 3 | NO-Mask |
| 4 | NO-Safety Vest |
| 5 | Person |
| 6 | Safety Cone |
| 7 | Safety Vest |
| 8 | machinery |
| 9 | vehicle |

## Splits

| Split | Images | Labels | Paired |
|---|---:|---:|---:|
| train | ? | ? | ✓ |
| valid | 114 | 114 | ✓ |
| test | 82 | 82 | ✓ |

## Object Counts by Split and Class

| Split | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | 79 | 21 | 69 | 74 | 106 | 166 | 44 | 41 | 55 | 42 | 697 |
| test | 110 | 28 | 41 | 79 | 90 | 174 | 92 | 61 | 44 | 41 | 760 |

## Validation Issues

✅ No issues found.

## Decision

**LOCAL_VALID_COMPLETE**

Local dataset is valid and complete. 196 images with 196 paired labels in YOLO format across 3 splits. No Kaggle download needed.

## Sample Videos

- `source_files/source_files/JapanPPE.mp4` (13.5 MB)
- `source_files/source_files/hardhat.mp4` (4.0 MB)
- `source_files/source_files/indianworkers.mp4` (7.3 MB)

## Pre-trained Weights

- `results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt` (6.0 MB)
- `results_yolov8n_100e/kaggle/working/runs/detect/train/weights/last.pt` (6.0 MB)
- `results_yolov8n_100e/kaggle/working/yolov8n.pt` (6.2 MB)
