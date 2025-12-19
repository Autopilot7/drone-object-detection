"""
Quick debug script to test detection on first N frames with detailed logging
"""
import sys
import logging
import argparse
import cv2

from src.data_loader import DroneDataset
from src.models.pipeline import DeepLearningPipeline
from src.config import DATA_ROOT
from src.utils.video_utils import get_video_info, extract_frame_at_index

logging.basicConfig(level=logging.INFO, format='%(message)s')


def debug_video(video_id: str, max_frames: int = 20):
    """
    Debug a video by processing first N frames with detailed logging
    
    Args:
        video_id: Video ID to debug
        max_frames: Maximum number of frames to process
    """
    print("=" * 70)
    print(f"DEBUG VIDEO: {video_id}")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = DroneDataset(DATA_ROOT)
    
    if video_id not in dataset.get_video_ids():
        print(f"❌ Error: Video ID '{video_id}' not found!")
        print(f"Available videos: {dataset.get_video_ids()}")
        return
    
    # Get video sample
    sample = dataset.get_sample(video_id)
    if sample is None:
        print(f"❌ Error: Could not load sample for '{video_id}'")
        return
    
    video_path = sample.video_path
    reference_images = sample.reference_images
    video_info = get_video_info(str(video_path))
    
    print(f"\n[2/4] Video Info:")
    print(f"  - Video ID: {video_id}")
    print(f"  - Total frames: {video_info['total_frames']}")
    print(f"  - Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  - FPS: {video_info['fps']:.1f}")
    print(f"  - Reference images: {len(reference_images)}")
    
    # Initialize pipeline
    print(f"\n[3/4] Initializing Deep Learning pipeline...")
    pipeline = DeepLearningPipeline(
        similarity_threshold=0.3,  # Current threshold
        confidence_threshold=0.3,
        use_tracking=False  # Disable tracking for debugging
    )
    
    # Set reference images
    pipeline.detector.set_reference_images(reference_images)
    print("✓ Reference embeddings computed")
    
    # Process frames
    print(f"\n[4/4] Processing first {max_frames} frames with DEBUG logging:")
    print("=" * 70)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    processed = 0
    total_detections = 0
    
    while frame_idx < video_info['total_frames'] and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        print(f"\n  [Frame {frame_idx}]")
        
        # Detect with debug enabled
        bboxes, confidences = pipeline.detector.detect_in_frame(frame, debug=True)
        
        if bboxes:
            print(f"    ✓ Found {len(bboxes)} detection(s) after filtering")
            total_detections += len(bboxes)
        
        frame_idx += 1
        processed += 1
    
    cap.release()
    
    print("\n" + "=" * 70)
    print(f"✅ DEBUG COMPLETE")
    print(f"Processed: {processed} frames")
    print(f"Total detections: {total_detections}")
    print("=" * 70)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if total_detections == 0:
        print("  ⚠️  No detections found! Possible issues:")
        print("     1. YOLO is not detecting any objects (check raw detection count)")
        print("     2. Similarity scores are too low (check max similarity)")
        print("     3. Reference images don't match target object well")
        print("     → Try: Visualize reference images and check ground truth")
    else:
        print(f"  ✓ Found {total_detections} detections in first {processed} frames")
        print("     → This is promising! Run full video to see complete results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug video detection with detailed logging"
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default="Backpack_0",
        help="Video ID to debug (e.g., Backpack_0, Lifering_1)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=20,
        help="Maximum number of frames to process (default: 20)"
    )
    
    args = parser.parse_args()
    
    debug_video(args.video_id, args.max_frames)

