"""
Video processing utilities for extracting and handling video frames
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple, List
from tqdm import tqdm


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata (fps, frame count, resolution)
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def extract_frames(
    video_path: str,
    frame_skip: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    show_progress: bool = True
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Extract frames from video as generator
    
    Args:
        video_path: Path to video file
        frame_skip: Extract every N frames (1 = all frames)
        start_frame: Starting frame number
        end_frame: Ending frame number (None = until end)
        show_progress: Show progress bar
        
    Yields:
        Tuple of (frame_number, frame_image)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_numbers = range(start_frame, min(end_frame, total_frames), frame_skip)
    
    iterator = tqdm(frame_numbers, desc="Extracting frames") if show_progress else frame_numbers
    
    for frame_idx in iterator:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_idx, frame_rgb
    
    cap.release()


def extract_frame_at_index(video_path: str, frame_idx: int) -> np.ndarray:
    """
    Extract a single frame at specific index
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        
    Returns:
        Frame as RGB numpy array
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_frames_to_dir(
    video_path: str,
    output_dir: str,
    frame_skip: int = 1,
    prefix: str = "frame"
) -> List[str]:
    """
    Extract and save frames to directory
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        frame_skip: Save every N frames
        prefix: Prefix for saved frame filenames
        
    Returns:
        List of saved frame paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_frames = []
    
    for frame_idx, frame in extract_frames(video_path, frame_skip):
        filename = f"{prefix}_{frame_idx:06d}.jpg"
        filepath = output_path / filename
        
        # Convert RGB back to BGR for saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), frame_bgr)
        saved_frames.append(str(filepath))
    
    return saved_frames


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 25.0,
    codec: str = 'mp4v'
) -> None:
    """
    Create video from list of frames
    
    Args:
        frames: List of frames (RGB numpy arrays)
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
    """
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in tqdm(frames, desc="Creating video"):
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def resize_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    keep_aspect: bool = True
) -> np.ndarray:
    """
    Resize frame with optional aspect ratio preservation
    
    Args:
        frame: Input frame
        target_size: Target (width, height)
        scale_factor: Scale factor (alternative to target_size)
        keep_aspect: Keep aspect ratio
        
    Returns:
        Resized frame
    """
    if scale_factor is not None:
        new_width = int(frame.shape[1] * scale_factor)
        new_height = int(frame.shape[0] * scale_factor)
        target_size = (new_width, new_height)
    
    if target_size is None:
        return frame
    
    if keep_aspect:
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        padded = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded
    else:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

