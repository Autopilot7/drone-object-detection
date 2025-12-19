# Debug Guide - Finding Why Detections Fail

## Problem
The model processes all videos but finds 0 detections for most videos (13/14 videos have 0 detections).

## Changes Made

### 1. Updated `src/models/detector.py`
Added debug logging to show:
- Number of raw YOLO detections (before similarity filtering)
- Similarity statistics (min, max, mean)
- Top 3 similarity scores
- Number of objects passing threshold

### 2. Updated `src/models/pipeline.py`
- Enabled debug logging for first 10 frames of each video
- Shows detailed detection info for early frames

### 3. Created `debug_video.py`
Quick debug script to test first N frames of a single video with full logging.

## How to Use

### Option 1: Run Full Pipeline with Debug (First 10 Frames)
```bash
python run_all_approaches.py --approaches deep_learning --frame-skip 2
```
- Will show debug info for first 10 frames of each video
- Then continue normal processing

### Option 2: Debug Single Video (Recommended)
```bash
# Debug first 20 frames of Backpack_0
python debug_video.py --video_id Backpack_0 --max_frames 20

# Debug first 50 frames of Lifering_1 (the one that worked)
python debug_video.py --video_id Lifering_1 --max_frames 50

# Try different videos
python debug_video.py --video_id Jacket_0 --max_frames 20
python debug_video.py --video_id Laptop_0 --max_frames 20
```

## What to Look For

### A. YOLO Detection Count
```
[DEBUG] YOLO detected 0 objects (conf > 0.3)
```
- **If 0**: YOLO is not detecting ANYTHING
  - → Problem: YOLO confidence threshold too high
  - → Problem: Objects are too small/far
  - → Solution: Lower confidence threshold to 0.1

- **If > 0**: YOLO is detecting objects, but similarity filtering rejects them

### B. Similarity Scores
```
[DEBUG] Similarities: min=0.123, max=0.287, mean=0.201
[DEBUG] Top 3 similarities: ['0.287', '0.245', '0.198']
[DEBUG] Threshold: 0.3
```
- **If max < threshold**: All detections rejected
  - → Problem: Reference images don't match target well
  - → Problem: DINOv2 embeddings not capturing similarity
  - → Solution: Lower threshold OR use different encoder

- **If max > threshold but still 0 detections**: Check NMS or other filtering

## Next Steps Based on Results

### Case 1: YOLO detects 0 objects
1. Lower YOLO confidence threshold in `src/config.py`:
   ```python
   CONFIDENCE_THRESHOLD = 0.1  # From 0.5 → 0.1
   ```

2. Try multi-scale detection:
   ```python
   pipeline = DeepLearningPipeline(use_multiscale=True)
   ```

### Case 2: YOLO detects objects but similarity too low
1. Lower similarity threshold in `src/config.py`:
   ```python
   SIMILARITY_THRESHOLD = 0.2  # From 0.3 → 0.2 or even 0.15
   ```

2. Try CLIP instead of DINOv2:
   ```python
   pipeline = DeepLearningPipeline(encoder_model="clip")
   ```

3. Check reference images quality:
   - Are they clear?
   - Do they show the target object from similar angles?
   - Are they the right object?

### Case 3: Lifering_1 works but others don't
- Compare debug output of Lifering_1 vs others
- What's different about Lifering_1?
  - Larger object?
  - Better reference images?
  - Different viewing angle?

## Commands to Run on Server

```bash
# 1. Pull latest changes
cd ~/drone-object-detection
git pull origin main

# 2. Debug a failing video
python debug_video.py --video_id Backpack_0 --max_frames 20

# 3. Debug the working video
python debug_video.py --video_id Lifering_1 --max_frames 20

# 4. Compare the outputs to understand the difference
```

## Expected Debug Output Example

```
[Frame 0]
  [DEBUG] YOLO detected 15 objects (conf > 0.3)
  [DEBUG] Similarities: min=0.123, max=0.287, mean=0.201
  [DEBUG] Top 3 similarities: ['0.287', '0.245', '0.198']
  [DEBUG] Threshold: 0.3
  [DEBUG] Objects passing threshold: 0/15
```

This tells us:
- ✓ YOLO is working (15 detections)
- ✗ Similarity scores are too low (max 0.287 < 0.3 threshold)
- → Solution: Lower threshold to 0.2

