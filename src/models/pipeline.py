"""
End-to-end deep learning pipeline for object detection and tracking
"""
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from ..data_loader import BBox, VideoSample
from ..utils.video_utils import extract_frames
from .detector import ReferenceMatchingDetector, MultiScaleDetector
from .tracker import ByteTracker
from .reference_encoder import ReferenceEncoder


class DeepLearningPipeline:
    """
    Complete deep learning pipeline with detection and tracking
    """
    
    def __init__(
        self,
        yolo_model: str = "yolov8x.pt",
        encoder_model: str = "dinov2",
        use_tracking: bool = True,
        use_multiscale: bool = False,
        similarity_threshold: float = 0.7,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize pipeline
        
        Args:
            yolo_model: YOLO model name
            encoder_model: Reference encoder model
            use_tracking: Enable ByteTrack tracking
            use_multiscale: Use multi-scale detection
            similarity_threshold: Similarity threshold
            confidence_threshold: Detection confidence threshold
            device: Device to use
        """
        self.use_tracking = use_tracking
        self.use_multiscale = use_multiscale
        
        # Initialize detector
        self.detector = ReferenceMatchingDetector(
            yolo_model=yolo_model,
            encoder_model=encoder_model,
            similarity_threshold=similarity_threshold,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # Multi-scale wrapper
        if use_multiscale:
            self.detector = MultiScaleDetector(
                base_detector=self.detector,
                scales=[0.75, 1.0, 1.25]
            )
        
        # Initialize tracker
        if use_tracking:
            self.tracker = ByteTracker(
                track_thresh=0.6,
                match_thresh=0.8,
                track_buffer=30
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
        print(f"\n[{video_sample.video_id}] Loading reference images and encoding...")
        self.detector.set_reference_images(video_sample.reference_images)
        print(f"[{video_sample.video_id}] ✓ Reference embeddings computed")
        
        # Get video info
        from ..utils.video_utils import get_video_info
        video_info = get_video_info(str(video_sample.video_path))
        total_frames = video_info['frame_count']
        frames_to_process = total_frames // frame_skip
        print(f"[{video_sample.video_id}] Video: {total_frames} frames total, will process {frames_to_process} frames (skip={frame_skip})")
        
        # Reset tracker if using
        if self.use_tracking:
            self.tracker.reset()
        
        # Track ID to BBox mapping
        track_bboxes = {}
        detection_count = 0
        
        # Process video frames
        print(f"[{video_sample.video_id}] Starting YOLO detection + similarity matching...")
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
            bboxes, confidences = self.detector.detect_in_frame(frame)
            
            if bboxes:
                detection_count += len(bboxes)
            
            if self.use_tracking and bboxes:
                # Update tracker
                tracks = self.tracker.update(
                    np.array(bboxes),
                    np.array(confidences)
                )
                
                # Group by track ID
                for track in tracks:
                    if track.track_id not in track_bboxes:
                        track_bboxes[track.track_id] = []
                    
                    bbox_obj = BBox(
                        frame=frame_idx,
                        x1=int(track.bbox[0]),
                        y1=int(track.bbox[1]),
                        x2=int(track.bbox[2]),
                        y2=int(track.bbox[3])
                    )
                    track_bboxes[track.track_id].append(bbox_obj)
            
            elif bboxes:
                # Without tracking, create sequence per frame
                # Group consecutive detections later
                if 0 not in track_bboxes:
                    track_bboxes[0] = []
                
                for bbox in bboxes:
                    bbox_obj = BBox(
                        frame=frame_idx,
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3])
                    )
                    track_bboxes[0].append(bbox_obj)
        
        # Convert to list of sequences
        print(f"[{video_sample.video_id}] ✓ Finished processing {processed_frames} frames")
        print(f"[{video_sample.video_id}] Total detections: {detection_count}")
        
        sequences = list(track_bboxes.values())
        
        # If not using tracking, group consecutive detections
        if not self.use_tracking and sequences:
            from ..traditional.temporal_filter import group_detections_into_sequences
            all_detections = []
            for seq in sequences:
                all_detections.extend(seq)
            sequences = group_detections_into_sequences(all_detections, max_frame_gap=10)
        
        print(f"[{video_sample.video_id}] Grouped into {len(sequences)} track(s)")
        
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


class OptimizedPipeline(DeepLearningPipeline):
    """
    Optimized pipeline with batch processing and caching
    """
    
    def __init__(self, *args, batch_size: int = 8, **kwargs):
        """
        Initialize optimized pipeline
        
        Args:
            batch_size: Batch size for processing
            *args, **kwargs: Arguments for base pipeline
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
    
    def process_video_batched(
        self,
        video_sample: VideoSample,
        frame_skip: int = 1,
        show_progress: bool = True
    ) -> List[List[BBox]]:
        """
        Process video with batched inference
        
        Args:
            video_sample: Video sample
            frame_skip: Frame skip
            show_progress: Show progress
            
        Returns:
            List of detection sequences
        """
        # Set reference images
        self.detector.set_reference_images(video_sample.reference_images)
        
        # Reset tracker
        if self.use_tracking:
            self.tracker.reset()
        
        # Extract all frames first
        frames_data = []
        frame_iterator = extract_frames(
            str(video_sample.video_path),
            frame_skip=frame_skip,
            show_progress=False
        )
        
        for frame_idx, frame in frame_iterator:
            frames_data.append((frame_idx, frame))
        
        # Process in batches
        track_bboxes = {}
        
        iterator = range(0, len(frames_data), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches")
        
        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, len(frames_data))
            batch = frames_data[batch_start:batch_end]
            
            for frame_idx, frame in batch:
                # Detect (TODO: can be batched for YOLO)
                bboxes, confidences = self.detector.detect_in_frame(frame)
                
                if self.use_tracking and bboxes:
                    tracks = self.tracker.update(
                        np.array(bboxes),
                        np.array(confidences)
                    )
                    
                    for track in tracks:
                        if track.track_id not in track_bboxes:
                            track_bboxes[track.track_id] = []
                        
                        bbox_obj = BBox(
                            frame=frame_idx,
                            x1=int(track.bbox[0]),
                            y1=int(track.bbox[1]),
                            x2=int(track.bbox[2]),
                            y2=int(track.bbox[3])
                        )
                        track_bboxes[track.track_id].append(bbox_obj)
        
        sequences = list(track_bboxes.values())
        return sequences

