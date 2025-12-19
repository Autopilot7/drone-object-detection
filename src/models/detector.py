"""
Object detector using YOLO with reference image matching
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
from PIL import Image

from .reference_encoder import ReferenceEncoder
from ..data_loader import BBox
from ..utils.bbox_utils import nms, clip_bbox
from ..config import SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD, YOLO_MODEL


class YOLODetector:
    """
    YOLO-based detector with reference image matching
    """
    
    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        confidence_threshold: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model name
            confidence_threshold: Detection confidence threshold
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model
        self.model = YOLO(model_name)
        self.model.to(self.device)
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in image
        
        Args:
            image: Input image (RGB)
            conf_threshold: Confidence threshold
            
        Returns:
            Tuple of (bboxes, confidences, class_ids)
        """
        conf_threshold = conf_threshold or self.confidence_threshold
        
        # Run detection
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract detections
        boxes = results[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        return bboxes, confidences, class_ids
    
    def detect_all_objects(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect all objects (for reference matching)
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (bboxes, confidences)
        """
        bboxes, confidences, _ = self.detect(image)
        return bboxes, confidences


class ReferenceMatchingDetector:
    """
    Detector that combines YOLO with reference image matching
    """
    
    def __init__(
        self,
        yolo_model: str = None,
        encoder_model: str = "dinov2",
        similarity_threshold: float = None,
        confidence_threshold: float = None,
        device: Optional[str] = None
    ):
        """
        Initialize detector
        
        Args:
            yolo_model: YOLO model name
            encoder_model: Reference encoder model
            similarity_threshold: Similarity threshold for reference matching
            confidence_threshold: YOLO confidence threshold
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
        yolo_model = yolo_model or YOLO_MODEL
        
        # Initialize components
        self.yolo_detector = YOLODetector(yolo_model, confidence_threshold, device)
        self.encoder = ReferenceEncoder(encoder_model, device)
        
        # Reference embedding (to be set)
        self.reference_embedding = None
    
    def set_reference_images(self, reference_images: List[np.ndarray]):
        """
        Set reference images and compute embedding
        
        Args:
            reference_images: List of reference images
        """
        self.reference_embedding = self.encoder.encode_reference_images(reference_images)
    
    def detect_in_frame(
        self,
        frame: np.ndarray,
        debug: bool = False
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect target object in frame
        
        Args:
            frame: Input frame (RGB)
            debug: Enable debug logging
            
        Returns:
            Tuple of (bboxes, confidences)
        """
        if self.reference_embedding is None:
            raise ValueError("Reference images not set. Call set_reference_images() first.")
        
        # Detect all objects with YOLO
        all_bboxes, all_confidences = self.yolo_detector.detect_all_objects(frame)
        
        # DEBUG: Log raw YOLO detections
        if debug:
            print(f"    [DEBUG] YOLO detected {len(all_bboxes)} objects (conf > {self.confidence_threshold})")
        
        if len(all_bboxes) == 0:
            return [], []
        
        # Extract and encode crops
        crop_embeddings = self.encoder.encode_and_crop(frame, all_bboxes)
        
        # Compute similarities
        similarities = []
        for crop_emb in crop_embeddings:
            sim = self.encoder.compute_similarity(self.reference_embedding, crop_emb)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # DEBUG: Log similarity statistics
        if debug and len(similarities) > 0:
            print(f"    [DEBUG] Similarities: min={similarities.min():.3f}, max={similarities.max():.3f}, mean={similarities.mean():.3f}")
            # Show top 3 similarities
            top_k = min(3, len(similarities))
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_sims = similarities[top_indices]
            print(f"    [DEBUG] Top {top_k} similarities: {[f'{s:.3f}' for s in top_sims]}")
            print(f"    [DEBUG] Threshold: {self.similarity_threshold}")
        
        # Filter by similarity threshold
        matched_indices = np.where(similarities >= self.similarity_threshold)[0]
        
        # DEBUG: Log filtering results
        if debug:
            print(f"    [DEBUG] Objects passing threshold: {len(matched_indices)}/{len(all_bboxes)}")
        
        if len(matched_indices) == 0:
            return [], []
        
        # Get matched detections
        matched_bboxes = [all_bboxes[i] for i in matched_indices]
        # Use YOLO confidence directly (similarity already filtered)
        # Multiplication makes confidence too low for tracking
        matched_confidences = [
            all_confidences[i]  # Use YOLO conf only
            for i in matched_indices
        ]
        
        # Apply NMS
        if len(matched_bboxes) > 0:
            keep_indices = nms(
                np.array(matched_bboxes),
                np.array(matched_confidences),
                iou_threshold=0.5
            )
            matched_bboxes = [matched_bboxes[i] for i in keep_indices]
            matched_confidences = [matched_confidences[i] for i in keep_indices]
        
        return matched_bboxes, matched_confidences
    
    def process_video(
        self,
        video_path: str,
        frame_skip: int = 1,
        show_progress: bool = True
    ) -> List[Tuple[int, List[np.ndarray], List[float]]]:
        """
        Process entire video
        
        Args:
            video_path: Path to video
            frame_skip: Process every N frames
            show_progress: Show progress bar
            
        Returns:
            List of (frame_idx, bboxes, confidences)
        """
        from ..utils.video_utils import extract_frames
        from tqdm import tqdm
        
        detections = []
        
        frame_iterator = extract_frames(
            video_path,
            frame_skip=frame_skip,
            show_progress=show_progress
        )
        
        for frame_idx, frame in frame_iterator:
            bboxes, confidences = self.detect_in_frame(frame)
            
            if bboxes:
                detections.append((frame_idx, bboxes, confidences))
        
        return detections


class MultiScaleDetector:
    """
    Multi-scale detector for small object detection
    """
    
    def __init__(
        self,
        base_detector: ReferenceMatchingDetector,
        scales: List[float] = [0.75, 1.0, 1.25]
    ):
        """
        Initialize multi-scale detector
        
        Args:
            base_detector: Base detector
            scales: Scales to test
        """
        self.base_detector = base_detector
        self.scales = scales
    
    def detect_in_frame(
        self,
        frame: np.ndarray,
        debug: bool = False
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect at multiple scales
        
        Args:
            frame: Input frame
            debug: Enable debug logging
            
        Returns:
            Tuple of (bboxes, confidences)
        """
        all_bboxes = []
        all_confidences = []
        
        original_h, original_w = frame.shape[:2]
        
        for scale in self.scales:
            # Resize frame
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            import cv2
            scaled_frame = cv2.resize(frame, (new_w, new_h))
            
            # Detect
            bboxes, confidences = self.base_detector.detect_in_frame(scaled_frame, debug=debug)
            
            # Scale bboxes back to original size
            for bbox, conf in zip(bboxes, confidences):
                scaled_bbox = bbox / scale
                scaled_bbox = clip_bbox(scaled_bbox, original_w, original_h)
                all_bboxes.append(scaled_bbox)
                all_confidences.append(conf)
        
        if not all_bboxes:
            return [], []
        
        # Apply NMS across scales
        keep_indices = nms(
            np.array(all_bboxes),
            np.array(all_confidences),
            iou_threshold=0.5
        )
        
        final_bboxes = [all_bboxes[i] for i in keep_indices]
        final_confidences = [all_confidences[i] for i in keep_indices]
        
        return final_bboxes, final_confidences

