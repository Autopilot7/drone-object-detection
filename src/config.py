"""
Configuration file for the Drone Object Detection Challenge
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "observing" / "train"
SAMPLES_DIR = DATA_ROOT / "samples"
ANNOTATIONS_FILE = DATA_ROOT / "annotations" / "annotations.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Video settings
VIDEO_FPS = 25
FRAME_SKIP = 1  # Process every N frames (1 = process all frames)

# Deep Learning Models
DINOV2_MODEL = "facebook/dinov2-large"  # or "facebook/dinov2-giant"
CLIP_MODEL = "openai/clip-vit-large-patch14"
YOLO_MODEL = "yolov8x.pt"  # x for maximum accuracy, n/s/m/l for speed

# Feature matching (Traditional CV)
FEATURE_TYPES = ["SIFT", "ORB", "AKAZE"]
DEFAULT_FEATURE_TYPE = "SIFT"
MATCH_RATIO_THRESHOLD = 0.75  # Lowe's ratio test
MIN_MATCH_COUNT = 10  # Minimum matches for valid detection
RANSAC_THRESHOLD = 5.0

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.1  # For reference image matching

# Tracking settings
TRACK_BUFFER = 30  # ByteTrack buffer
TRACK_THRESH = 0.6  # Track confidence threshold
MATCH_THRESH = 0.8  # Matching threshold for tracking

# Evaluation
ST_IOU_FRAME_TOLERANCE = 0  # Tolerance in frame matching

# Training/Validation split
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42

# Computational settings
DEVICE = "cuda"  # Will auto-detect in code
BATCH_SIZE = 8
NUM_WORKERS = 4

# Create necessary directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

