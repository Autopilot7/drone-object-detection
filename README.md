# Drone-Based Object Search & Localization Challenge

Multi-approach implementation for spatio-temporal object detection in drone footage.

## Overview

This project implements three different approaches for detecting and localizing target objects in drone videos:
- **Approach A**: Deep Learning with SOTA models (DINOv2, YOLOv8, ByteTrack)
- **Approach B**: Traditional Computer Vision (SIFT, ORB, Kalman filtering)
- **Approach C**: Hybrid approach combining both methods

## Quick Start

### Installation

```powershell
# Install dependencies
pip install -r requirements.txt
```

### Run All Approaches

```powershell
# Run all three approaches and compare results
python run_all_approaches.py

# Run specific approach only
python run_all_approaches.py --approaches deep_learning

# Adjust processing speed (process every N frames)
python run_all_approaches.py --frame-skip 2
```

### Expected Output

Results are saved to `output/` directory:
- Predictions in JSON format (competition-ready)
- Comparison report with ST-IoU scores
- Summary table and detailed per-video metrics

## Project Structure

```
CV/
├── src/
│   ├── data_loader.py              # Dataset loading utilities
│   ├── config.py                   # Configuration and hyperparameters
│   ├── models/                     # Deep learning models
│   │   ├── reference_encoder.py   # DINOv2/CLIP encoder
│   │   ├── detector.py            # YOLO + similarity matching
│   │   ├── tracker.py             # ByteTrack
│   │   └── pipeline.py            # End-to-end DL pipeline
│   ├── traditional/                # Traditional CV methods
│   │   ├── feature_extractor.py   # SIFT/ORB/AKAZE
│   │   ├── detector.py            # Feature-based detection
│   │   ├── temporal_filter.py     # Kalman filtering
│   │   └── pipeline.py            # Traditional CV pipeline
│   ├── hybrid/                     # Hybrid approach
│   │   └── pipeline.py            # CV + DL fusion
│   ├── evaluation/                 # Evaluation metrics
│   │   ├── st_iou.py              # ST-IoU metric
│   │   ├── compare.py             # Multi-approach comparison
│   │   └── visualize.py           # Visualization tools
│   └── utils/                      # Helper utilities
├── notebooks/                      # Jupyter notebooks
├── observing/                      # Dataset (train/)
├── output/                         # Results (generated)
├── run_all_approaches.py           # Master execution script
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── USAGE.md                        # Detailed usage guide
└── IMPLEMENTATION_SUMMARY.md       # Implementation details
```

## Key Features

### Approach A: Deep Learning (SOTA)
- DINOv2 for reference image encoding
- YOLOv8 for object detection
- ByteTrack for temporal tracking
- Multi-scale detection support
- Expected ST-IoU: ~0.5-0.7

### Approach B: Traditional CV
- SIFT/ORB feature matching
- RANSAC geometric verification
- Kalman filter tracking
- Fast inference (~15-20 FPS)
- Expected ST-IoU: ~0.2-0.4

### Approach C: Hybrid
- CV candidate generation + DL verification
- Balanced speed/accuracy trade-off
- Reduces false positives
- Expected ST-IoU: ~0.4-0.6

## Dataset

- 14 training videos across 7 object categories
- Each video: 3 reference images + 3-5 min drone footage (25 fps)
- Ground truth: Frame-level bounding box annotations
- Categories: Backpack, Jacket, Laptop, Lifering, MobilePhone, Person, WaterBottle

## Evaluation Metric

**Spatio-Temporal IoU (ST-IoU)**: Jointly measures when and where objects are detected

```
ST-IoU = Σ(spatial_IoU_per_frame) / total_frames_in_union
```

## Documentation

- **USAGE.md**: Detailed usage instructions and examples
- **IMPLEMENTATION_SUMMARY.md**: Technical implementation details
- **notebooks/**: Interactive examples and experiments

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- OpenCV 4.8+
- See `requirements.txt` for complete list

## GPU Support

Recommended for deep learning approach:
- GPU: 8GB+ VRAM (RTX 3070 or better)
- CPU fallback supported (slower)

## Troubleshooting

### GPU not detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Out of memory
- Increase `--frame-skip` parameter
- Use smaller YOLO model (e.g., `yolov8m.pt`)

### Slow processing
- Use `--frame-skip 3` or higher for faster testing
- Traditional CV approach is fastest

## Results

Run the comparison to see detailed results:

```powershell
python run_all_approaches.py
```

Output includes:
- Mean ST-IoU per approach
- Per-video scores
- Per-category breakdown
- Visualization plots


============================================================
            Approach  Mean ST-IoU  Std ST-IoU  Min ST-IoU  Max ST-IoU  Median ST-IoU
Deep Learning (clip)     0.051775    0.086118         0.0    0.306028       0.006329

============================================================
BEST APPROACH: Deep Learning (clip)
Mean ST-IoU: 0.0518
============================================================