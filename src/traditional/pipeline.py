"""
Complete Traditional CV pipeline for object detection and tracking
"""
import numpy as np
from typing import List
from pathlib import Path
from tqdm import tqdm

from ..data_loader import BBox, VideoSample
from ..utils.video_utils import extract_frames
from .detector import TraditionalDetector
from .temporal_filter import SimpleTracker, group_detections_into_sequences


class TraditionalCVPipeline:
    """
    End-to-end traditional CV pipeline
    """
    
    def __init__(
        self,
        feature_type: str = "SIFT",
        use_tracking: bool = True,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize pipeline
        
        Args:
            feature_type: Feature extraction method (SIFT, ORB, AKAZE)
            use_tracking: Enable temporal tracking
            confidence_threshold: Detection confidence threshold
        """
        self.detector = TraditionalDetector(feature_type=feature_type)
        self.use_tracking = use_tracking
        self.confidence_threshold = confidence_threshold
        
        if use_tracking:
            self.tracker = SimpleTracker(
                max_age=30,
                min_hits=3,
                iou_threshold=0.3
            )
    
    def process_video(
        self,
        video_sample: VideoSample,
        frame_skip: int = 1,
        show_progress: bool = True
    ) -> List[List[BBox]]:
        """
        Process video and return detections
        
        Args:
            video_sample: Video sample with reference images
            frame_skip: Process every N frames
            show_progress: Show progress bar
            
        Returns:
            List of detection sequences
        """
        # Set reference images
        print(f"\n[{video_sample.video_id}] Loading reference images...")
        self.detector.set_reference_images(video_sample.reference_images)
        print(f"[{video_sample.video_id}] ✓ Reference images loaded")
        
        # Get video info
        from ..utils.video_utils import get_video_info
        video_info = get_video_info(str(video_sample.video_path))
        total_frames = video_info['frame_count']
        frames_to_process = total_frames // frame_skip
        print(f"[{video_sample.video_id}] Video: {total_frames} frames total, will process {frames_to_process} frames (skip={frame_skip})")
        
        # Reset tracker if using
        if self.use_tracking:
            self.tracker = SimpleTracker(
                max_age=30,
                min_hits=3,
                iou_threshold=0.3
            )
        
        all_detections = []
        detection_count = 0
        
        # Process video frames
        print(f"[{video_sample.video_id}] Starting frame processing...")
        frame_iterator = extract_frames(
            str(video_sample.video_path),
            frame_skip=frame_skip,
            show_progress=show_progress
        )
        
        processed_frames = 0
        for frame_idx, frame in frame_iterator:
            processed_frames += 1
            
            # Log progress every 100 frames
            if processed_frames % 100 == 0:
                print(f"[{video_sample.video_id}] Processed {processed_frames}/{frames_to_process} frames ({100*processed_frames//frames_to_process}%) - Detections: {detection_count}")
            
            # Detect in frame
            bboxes, confidences = self.detector.detect_in_frame(
                frame,
                self.confidence_threshold
            )
            
            if bboxes:
                detection_count += len(bboxes)
            
            if self.use_tracking and bboxes:
                # Update tracker
                tracks = self.tracker.update(bboxes, confidences)
                
                # Get track bboxes
                frame_bboxes = self.tracker.get_track_bboxes(frame_idx)
                all_detections.extend(frame_bboxes)
            elif bboxes:
                # Without tracking, just add detections
                for bbox in bboxes:
                    bbox_obj = BBox(
                        frame=frame_idx,
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3])
                    )
                    all_detections.append(bbox_obj)
        
        # Group detections into sequences
        print(f"[{video_sample.video_id}] ✓ Finished processing {processed_frames} frames")
        print(f"[{video_sample.video_id}] Total detections: {len(all_detections)}")
        
        sequences = group_detections_into_sequences(
            all_detections,
            max_frame_gap=10
        )
        
        print(f"[{video_sample.video_id}] Grouped into {len(sequences)} sequences")
        
        return sequences
    
    def process_dataset(
        self,
        samples: List[VideoSample],
        frame_skip: int = 1,
        show_progress: bool = True
    ) -> dict:
        """
        Process multiple videos
        
        Args:
            samples: List of video samples
            frame_skip: Process every N frames
            show_progress: Show progress bar
            
        Returns:
            Dictionary mapping video_id to detection sequences
        """
        results = {}
        
        iterator = tqdm(samples, desc="Processing videos") if show_progress else samples
        
        for sample in iterator:
            if show_progress:
                iterator.set_description(f"Processing {sample.video_id}")
            
            sequences = self.process_video(
                sample,
                frame_skip=frame_skip,
                show_progress=False
            )
            
            results[sample.video_id] = sequences
        
        return results

