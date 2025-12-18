# Usage Guide - Drone Object Detection Challenge

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run All Approaches

To run all three approaches and compare results:

```powershell
python run_all_approaches.py
```

This will:
- Run Approach B (Traditional CV with SIFT/ORB)
- Run Approach A (Deep Learning with YOLO + DINOv2)
- Run Approach C (Hybrid approach)
- Compare all results and generate reports

### 3. Run Specific Approach

Run only specific approaches:

```powershell
# Traditional CV only
python run_all_approaches.py --approaches traditional

# Deep Learning only
python run_all_approaches.py --approaches deep_learning

# Hybrid only
python run_all_approaches.py --approaches hybrid

# Multiple approaches
python run_all_approaches.py --approaches traditional deep_learning
```

### 4. Adjust Frame Processing

Process every N frames (default is 2 for faster processing):

```powershell
# Process all frames (slower but more accurate)
python run_all_approaches.py --frame-skip 1

# Process every 5 frames (faster but less accurate)
python run_all_approaches.py --frame-skip 5
```

## Using Notebooks

### Data Exploration

```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```

Explore dataset statistics, visualize reference images, and analyze annotations.

### Approach-Specific Notebooks

1. **Traditional CV (Approach B)**:
   ```powershell
   jupyter notebook notebooks/03_approach_b_traditional.ipynb
   ```

2. **Deep Learning (Approach A)**:
   ```powershell
   jupyter notebook notebooks/02_approach_a_dl.ipynb
   ```

3. **Hybrid (Approach C)**:
   ```powershell
   jupyter notebook notebooks/04_approach_c_hybrid.ipynb
   ```

### Comparison and Results

```powershell
jupyter notebook notebooks/05_comparison_results.ipynb
```

## Understanding the Output

### Results Location

All results are saved to `output/` directory:

- `predictions_traditional.json` - Traditional CV predictions
- `predictions_deep_learning.json` - Deep Learning predictions
- `predictions_hybrid.json` - Hybrid predictions
- `comparison_report/` - Detailed comparison report
  - `summary.csv` - Summary table
  - `detailed_results.json` - Per-video scores

### Submission Format

Predictions are saved in the competition format:

```json
[
  {
    "video_id": "Backpack_0",
    "detections": [
      {
        "bboxes": [
          {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
          {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
        ]
      }
    ]
  }
]
```

## Customization

### Traditional CV Parameters

Edit `src/config.py`:

```python
# Feature matching
FEATURE_TYPES = ["SIFT", "ORB", "AKAZE"]
DEFAULT_FEATURE_TYPE = "SIFT"
MATCH_RATIO_THRESHOLD = 0.75
MIN_MATCH_COUNT = 10

# Tracking
TRACK_BUFFER = 30
```

### Deep Learning Parameters

```python
# Models
DINOV2_MODEL = "facebook/dinov2-large"
YOLO_MODEL = "yolov8x.pt"

# Detection
CONFIDENCE_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.7
```

### Hybrid Parameters

```python
# Hybrid approach uses lower thresholds for candidate generation
CV_CONFIDENCE = 0.2
DL_SIMILARITY = 0.7
```

## Troubleshooting

### GPU Issues

If GPU is not detected:

```python
import torch
print(torch.cuda.is_available())  # Should be True
```

If False, check:
1. CUDA installation
2. PyTorch CUDA version matches your CUDA version

### Memory Issues

If running out of memory:
1. Increase `frame_skip` parameter
2. Use smaller YOLO model (e.g., `yolov8m.pt` instead of `yolov8x.pt`)
3. Process fewer videos at a time

### DINOv2 Download Issues

If DINOv2 fails to download:

```python
# Alternative: Use CLIP instead
from src.models.reference_encoder import ReferenceEncoder
encoder = ReferenceEncoder(model_name="clip")
```

## Performance Tips

1. **Fast Testing**: Use `--frame-skip 5` for quick testing
2. **Production**: Use `--frame-skip 1` or `--frame-skip 2` for submission
3. **Multi-Scale**: Enable multi-scale detection for small objects (slower)
4. **Tracking**: Enable tracking for temporal consistency (recommended)

## Expected Performance

Based on the evaluation metric (ST-IoU):

- **Traditional CV**: ~0.2-0.4 ST-IoU (baseline)
- **Deep Learning**: ~0.5-0.7 ST-IoU (best performance)
- **Hybrid**: ~0.4-0.6 ST-IoU (balanced)

*Note: Actual performance depends on dataset and hyperparameters*

## Citation

If you use this code, please cite:

```
@misc{drone_object_detection_2025,
  title={Drone Object Detection Challenge - Multi-Approach Implementation},
  year={2025},
  author={Your Name}
}
```

