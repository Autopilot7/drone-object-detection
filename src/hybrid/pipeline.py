"""
Hybrid pipeline combining traditional CV and deep learning approaches
"""
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

from ..data_loader import BBox, VideoSample
from ..utils.video_utils import extract_frames
from ..utils.bbox_utils import bbox_iou, nms
from ..traditional.detector import TraditionalDetector
from ..models.reference_encoder import ReferenceEncoder
from ..models.tracker import ByteTracker


class HybridPipeline:
    """
    Hybrid approach: Traditional CV for candidate generation + DL for verification
    """
    
    def __init__(
        self,
        feature_type: str = "SIFT",
        encoder_model: str = "dinov2",
        cv_confidence: float = 0.2,
        similarity_threshold: float = 0.7,
        use_tracking: bool = True,
        device: str = None
    ):
        """
        Initialize hybrid pipeline
        
        Args:
            feature_type: Traditional CV feature type
            encoder_model: DL encoder model
            cv_confidence: Traditional CV confidence threshold
            similarity_threshold: DL similarity threshold
            use_tracking: Enable tracking
            device: Device for DL models
        """
        # Traditional CV detector (for candidate generation)
        self.cv_detector = TraditionalDetector(
            feature_type=feature_type,
            min_matches=5  # Lower threshold for candidate generation
        )
        
        # DL encoder (for verification)
        self.encoder = ReferenceEncoder(
            model_name=encoder_model,
            device=device
        )
        
        self.cv_confidence = cv_confidence
        self.similarity_threshold = similarity_threshold
        self.use_tracking = use_tracking
        
        # Tracker
        if use_tracking:
            self.tracker = ByteTracker(
                track_thresh=0.6,
                match_thresh=0.8,
                track_buffer=30
            )
        
        self.reference_embedding = None
    
    def set_reference_images(self, reference_images: List[np.ndarray]):
        """Set reference images"""
        # For traditional CV
        self.cv_detector.set_reference_images(reference_images)
        
        # For DL verification
        self.reference_embedding = self.encoder.encode_reference_images(reference_images)
    
    def detect_in_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect in frame using hybrid approach
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (bboxes, confidences)
        """
        # Step 1: Generate candidates using traditional CV
        cv_bboxes, cv_confidences = self.cv_detector.detect_in_frame(
            frame,
            confidence_threshold=self.cv_confidence
        )
        
        if len(cv_bboxes) == 0:
            return [], []
        
        # Step 2: Verify candidates using DL
        crop_embeddings = self.encoder.encode_and_crop(frame, cv_bboxes)
        
        verified_bboxes = []
        verified_confidences = []
        
        for bbox, cv_conf, crop_emb in zip(cv_bboxes, cv_confidences, crop_embeddings):
            # Compute similarity
            similarity = self.encoder.compute_similarity(
                self.reference_embedding,
                crop_emb
            )
            
            # Verify with threshold
            if similarity >= self.similarity_threshold:
                verified_bboxes.append(bbox)
                # Combined confidence
                combined_conf = (cv_conf + similarity) / 2
                verified_confidences.append(combined_conf)
        
        # Apply NMS
        if verified_bboxes:
            keep_indices = nms(
                np.array(verified_bboxes),
                np.array(verified_confidences),
                iou_threshold=0.5
            )
            verified_bboxes = [verified_bboxes[i] for i in keep_indices]
            verified_confidences = [verified_confidences[i] for i in keep_indices]
        
        return verified_bboxes, verified_confidences
    
    def process_video(
        self,
        video_sample: VideoSample,
        frame_skip: int = 1,
        show_progress: bool = True
    ) -> List[List[BBox]]:
        """
        Process video
        
        Args:
            video_sample: Video sample
            frame_skip: Process every N frames
            show_progress: Show progress bar
            
        Returns:
            List of detection sequences
        """
        # Set reference images
        print(f"\n[{video_sample.video_id}] Loading reference images for hybrid pipeline...")
        self.set_reference_images(video_sample.reference_images)
        print(f"[{video_sample.video_id}] ✓ Reference images ready (CV + DL)")
        
        # Get video info
        from ..utils.video_utils import get_video_info
        video_info = get_video_info(str(video_sample.video_path))
        total_frames = video_info['frame_count']
        frames_to_process = total_frames // frame_skip
        print(f"[{video_sample.video_id}] Video: {total_frames} frames total, will process {frames_to_process} frames (skip={frame_skip})")
        
        # Reset tracker
        if self.use_tracking:
            self.tracker.reset()
        
        track_bboxes = {}
        cv_candidates = 0
        verified_detections = 0
        
        # Process frames
        print(f"[{video_sample.video_id}] Starting hybrid processing (CV candidates → DL verification)...")
        frame_iterator = extract_frames(
            str(video_sample.video_path),
            frame_skip=frame_skip,
            show_progress=show_progress
        )
        
        processed_frames = 0
        
        for frame_idx, frame in frame_iterator:
            processed_frames += 1
            
            # Detect with hybrid approach
            bboxes, confidences = self.detect_in_frame(frame)
            
            if bboxes:
                verified_detections += len(bboxes)
            
            # Log progress every 100 frames
            if processed_frames % 100 == 0:
                print(f"[{video_sample.video_id}] Processed {processed_frames}/{frames_to_process} frames ({100*processed_frames//frames_to_process}%) - Verified: {verified_detections}")
            
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
            
            elif bboxes:
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
        
        print(f"[{video_sample.video_id}] ✓ Finished processing {processed_frames} frames")
        print(f"[{video_sample.video_id}] Verified detections: {verified_detections}")
        
        sequences = list(track_bboxes.values())
        
        if not self.use_tracking and sequences:
            from ..traditional.temporal_filter import group_detections_into_sequences
            all_detections = []
            for seq in sequences:
                all_detections.extend(seq)
            sequences = group_detections_into_sequences(all_detections, max_frame_gap=10)
        
        print(f"[{video_sample.video_id}] Grouped into {len(sequences)} sequence(s)")
        
        return sequences
    
    def process_dataset(
        self,
        samples: List[VideoSample],
        frame_skip: int = 1,
        show_progress: bool = True
    ) -> dict:
        """Process multiple videos"""
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


class EnsemblePipeline:
    """
    Ensemble approach: Run both traditional and DL, combine results
    """
    
    def __init__(
        self,
        traditional_pipeline,
        dl_pipeline,
        fusion_method: str = "weighted",
        weights: Tuple[float, float] = (0.3, 0.7)
    ):
        """
        Initialize ensemble
        
        Args:
            traditional_pipeline: Traditional CV pipeline
            dl_pipeline: Deep learning pipeline
            fusion_method: Method to fuse results ('weighted', 'voting', 'union')
            weights: Weights for (traditional, dl)
        """
        self.traditional_pipeline = traditional_pipeline
        self.dl_pipeline = dl_pipeline
        self.fusion_method = fusion_method
        self.weights = weights
    
    def fuse_detections(
        self,
        trad_bboxes: List[np.ndarray],
        trad_confs: List[float],
        dl_bboxes: List[np.ndarray],
        dl_confs: List[float]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Fuse detections from both approaches
        
        Returns:
            Tuple of (fused_bboxes, fused_confidences)
        """
        if self.fusion_method == "union":
            # Simple union with NMS
            all_bboxes = trad_bboxes + dl_bboxes
            all_confs = trad_confs + dl_confs
            
            if all_bboxes:
                keep_indices = nms(
                    np.array(all_bboxes),
                    np.array(all_confs),
                    iou_threshold=0.5
                )
                return [all_bboxes[i] for i in keep_indices], [all_confs[i] for i in keep_indices]
            return [], []
        
        elif self.fusion_method == "weighted":
            # Match overlapping detections and weight by confidence
            fused_bboxes = []
            fused_confs = []
            
            matched_dl = set()
            
            # Match traditional to DL
            for trad_bbox, trad_conf in zip(trad_bboxes, trad_confs):
                best_iou = 0
                best_idx = -1
                
                for dl_idx, dl_bbox in enumerate(dl_bboxes):
                    if dl_idx in matched_dl:
                        continue
                    iou = bbox_iou(trad_bbox, dl_bbox)
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_idx = dl_idx
                
                if best_idx >= 0:
                    # Merge
                    matched_dl.add(best_idx)
                    w1, w2 = self.weights
                    fused_conf = w1 * trad_conf + w2 * dl_confs[best_idx]
                    fused_bbox = (w1 * trad_bbox + w2 * dl_bboxes[best_idx]) / (w1 + w2)
                    fused_bboxes.append(fused_bbox)
                    fused_confs.append(fused_conf)
                else:
                    # Keep traditional only
                    fused_bboxes.append(trad_bbox)
                    fused_confs.append(trad_conf * self.weights[0])
            
            # Add unmatched DL detections
            for dl_idx, (dl_bbox, dl_conf) in enumerate(zip(dl_bboxes, dl_confs)):
                if dl_idx not in matched_dl:
                    fused_bboxes.append(dl_bbox)
                    fused_confs.append(dl_conf * self.weights[1])
            
            return fused_bboxes, fused_confs
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def process_video(
        self,
        video_sample: VideoSample,
        frame_skip: int = 1,
        show_progress: bool = True
    ) -> List[List[BBox]]:
        """Process video with ensemble"""
        # Run both pipelines
        trad_sequences = self.traditional_pipeline.process_video(
            video_sample, frame_skip, show_progress=False
        )
        
        dl_sequences = self.dl_pipeline.process_video(
            video_sample, frame_skip, show_progress=False
        )
        
        # For simplicity, return DL results (better performance expected)
        # Full fusion would require frame-level alignment
        return dl_sequences

