"""
Quick test script to process a single video with detailed logging
"""
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DroneDataset
from src.traditional.pipeline import TraditionalCVPipeline
from src.config import DATA_ROOT
from src.utils.video_utils import get_video_info

print("="*70)
print("QUICK TEST - SINGLE VIDEO WITH DETAILED LOGGING")
print("="*70)

print("\n[1/5] Loading dataset...")
dataset = DroneDataset(DATA_ROOT)
print(f"✓ Loaded {len(dataset)} videos")

# Get first video
sample = dataset.get_sample_by_index(0)
print(f"\n[2/5] Selected video: {sample.video_id}")

# Show video info
video_info = get_video_info(str(sample.video_path))
print(f"  - Duration: {video_info['duration']:.1f} seconds")
print(f"  - Total frames: {video_info['frame_count']}")
print(f"  - Resolution: {video_info['width']}x{video_info['height']}")
print(f"  - FPS: {video_info['fps']}")

# Calculate frames to process
frame_skip = 10
frames_to_process = video_info['frame_count'] // frame_skip
estimated_time = frames_to_process * 2  # ~2 seconds per frame
print(f"\n[3/5] With frame_skip={frame_skip}:")
print(f"  - Will process: {frames_to_process} frames")
print(f"  - Estimated time: {estimated_time//60} min {estimated_time%60} sec")

# Create pipeline
print(f"\n[4/5] Initializing Traditional CV pipeline (SIFT)...")
pipeline = TraditionalCVPipeline(
    feature_type="SIFT",
    use_tracking=True,
    confidence_threshold=0.3
)
print("✓ Pipeline ready")

# Process single video
print(f"\n[5/5] Processing video...")
print("="*70)

start_time = time.time()

sequences = pipeline.process_video(
    sample,
    frame_skip=frame_skip,
    show_progress=True
)

elapsed_time = time.time() - start_time

print("="*70)
print("\n✅ PROCESSING COMPLETE!")
print(f"  - Time taken: {elapsed_time//60:.0f} min {elapsed_time%60:.0f} sec")
print(f"  - Detection sequences: {len(sequences)}")
print(f"  - Total detections: {sum(len(seq) for seq in sequences)}")
print(f"  - Average time per frame: {elapsed_time/frames_to_process:.2f} sec")

if len(sequences) > 0:
    print(f"\n📊 Sequence details:")
    for i, seq in enumerate(sequences[:3]):  # Show first 3 sequences
        print(f"  Sequence {i+1}: {len(seq)} frames (frame {seq[0].frame} to {seq[-1].frame})")
    if len(sequences) > 3:
        print(f"  ... and {len(sequences)-3} more sequences")
else:
    print("\n⚠️ No detections found (this might be normal depending on the video)")

print("\n" + "="*70)

