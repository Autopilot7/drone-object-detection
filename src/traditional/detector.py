"""
Traditional CV-based object detection using feature matching
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..data_loader import BBox
from ..utils.bbox_utils import bbox_from_keypoints, nms, clip_bbox
from .feature_extractor import FeatureExtractor, FeatureMatcher


class TraditionalDetector:
    """
    Object detector using traditional feature matching
    """
    
    def __init__(
        self,
        feature_type: str = "SIFT",
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 5.0,
        min_matches: int = 10,
        bbox_margin: float = 20.0
    ):
        """
        Initialize detector
        
        Args:
            feature_type: Feature extraction method (SIFT, ORB, AKAZE)
            ratio_threshold: Lowe's ratio test threshold
            ransac_threshold: RANSAC threshold for geometric verification
            min_matches: Minimum number of matches for valid detection
            bbox_margin: Margin around matched keypoints for bbox
        """
        self.feature_extractor = FeatureExtractor(feature_type)
        self.matcher = FeatureMatcher(feature_type, ratio_threshold)
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.bbox_margin = bbox_margin
        
        # Reference features (to be set during initialization)
        self.ref_keypoints = []
        self.ref_descriptors = []
        self.ref_images = []
    
    def set_reference_images(self, reference_images: List[np.ndarray]):
        """
        Set reference images and extract their features
        
        Args:
            reference_images: List of reference images (RGB)
        """
        self.ref_images = reference_images
        self.ref_keypoints, self.ref_descriptors = \
            self.feature_extractor.extract_from_images(reference_images)
    
    def detect_in_frame(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.0,
        verbose: bool = False
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect object in a single frame
        
        Args:
            frame: Input frame (RGB)
            confidence_threshold: Minimum confidence (match count ratio)
            verbose: Print debug info
            
        Returns:
            Tuple of (bboxes, confidences)
        """
        if not self.ref_descriptors:
            return [], []
        
        # Extract features from frame
        if verbose:
            print(f"  Extracting features from frame...")
        frame_kp, frame_desc = self.feature_extractor.extract(frame)
        if verbose:
            print(f"  Found {len(frame_kp) if frame_kp else 0} keypoints")
        
        if frame_desc is None or len(frame_desc) == 0:
            return [], []
        
        # Match against all reference images
        all_matches = []
        all_ref_kp = []
        
        for ref_kp, ref_desc in zip(self.ref_keypoints, self.ref_descriptors):
            if ref_desc is None:
                continue
            
            matches = self.matcher.match(ref_desc, frame_desc)
            
            if len(matches) >= self.min_matches:
                # Geometric verification
                filtered_matches, H = self.matcher.filter_matches_geometric(
                    ref_kp, frame_kp, matches,
                    self.ransac_threshold, self.min_matches
                )
                
                if len(filtered_matches) >= self.min_matches:
                    all_matches.append(filtered_matches)
                    all_ref_kp.append(ref_kp)
        
        if not all_matches:
            return [], []
        
        # Create bounding boxes from matched keypoints
        bboxes = []
        confidences = []
        
        for matches, ref_kp in zip(all_matches, all_ref_kp):
            # Get matched keypoint coordinates in frame
            query_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches])
            
            # Create bounding box
            bbox = bbox_from_keypoints(query_pts, self.bbox_margin)
            bbox = clip_bbox(bbox, frame.shape[1], frame.shape[0])
            
            # Calculate confidence based on number of matches
            confidence = min(1.0, len(matches) / 50.0)  # Normalize by expected good match count
            
            if confidence >= confidence_threshold:
                bboxes.append(bbox)
                confidences.append(confidence)
        
        # Apply NMS to remove overlapping detections
        if len(bboxes) > 0:
            keep_indices = nms(np.array(bboxes), np.array(confidences), iou_threshold=0.5)
            bboxes = [bboxes[i] for i in keep_indices]
            confidences = [confidences[i] for i in keep_indices]
        
        return bboxes, confidences
    
    def detect_in_video(
        self,
        video_path: str,
        frame_skip: int = 1,
        confidence_threshold: float = 0.3,
        show_progress: bool = True
    ) -> List[Tuple[int, List[np.ndarray], List[float]]]:
        """
        Detect object throughout video
        
        Args:
            video_path: Path to video file
            frame_skip: Process every N frames
            confidence_threshold: Minimum confidence threshold
            show_progress: Show progress bar
            
        Returns:
            List of (frame_number, bboxes, confidences) tuples
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
            bboxes, confidences = self.detect_in_frame(frame, confidence_threshold)
            
            if bboxes:
                detections.append((frame_idx, bboxes, confidences))
        
        return detections
    
    def detections_to_bboxes(
        self,
        detections: List[Tuple[int, List[np.ndarray], List[float]]]
    ) -> List[BBox]:
        """
        Convert detections to BBox format
        
        Args:
            detections: List of (frame_idx, bboxes, confidences)
            
        Returns:
            List of BBox objects
        """
        bbox_list = []
        
        for frame_idx, bboxes, confidences in detections:
            for bbox in bboxes:
                bbox_obj = BBox(
                    frame=frame_idx,
                    x1=int(bbox[0]),
                    y1=int(bbox[1]),
                    x2=int(bbox[2]),
                    y2=int(bbox[3])
                )
                bbox_list.append(bbox_obj)
        
        return bbox_list


class TemplateMatchingDetector:
    """
    Simple template matching detector (baseline comparison)
    """
    
    def __init__(self, method: int = cv2.TM_CCOEFF_NORMED):
        """
        Initialize template matching detector
        
        Args:
            method: OpenCV template matching method
        """
        self.method = method
        self.templates = []
    
    def set_reference_images(self, reference_images: List[np.ndarray]):
        """Set reference images as templates"""
        self.templates = reference_images
    
    def detect_in_frame(
        self,
        frame: np.ndarray,
        scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
        threshold: float = 0.7
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect using multi-scale template matching
        
        Args:
            frame: Input frame (RGB)
            scales: List of scales to try
            threshold: Matching threshold
            
        Returns:
            Tuple of (bboxes, confidences)
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        all_bboxes = []
        all_scores = []
        
        for template in self.templates:
            template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
            h, w = template_gray.shape
            
            for scale in scales:
                # Resize template
                scaled_w = int(w * scale)
                scaled_h = int(h * scale)
                
                if scaled_w > frame.shape[1] or scaled_h > frame.shape[0]:
                    continue
                
                scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h))
                
                # Template matching
                result = cv2.matchTemplate(frame_gray, scaled_template, self.method)
                
                # Find peaks
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    bbox = np.array([pt[0], pt[1], pt[0] + scaled_w, pt[1] + scaled_h])
                    score = result[pt[1], pt[0]]
                    
                    all_bboxes.append(bbox)
                    all_scores.append(float(score))
        
        if not all_bboxes:
            return [], []
        
        # Apply NMS
        keep_indices = nms(np.array(all_bboxes), np.array(all_scores), iou_threshold=0.3)
        bboxes = [all_bboxes[i] for i in keep_indices]
        scores = [all_scores[i] for i in keep_indices]
        
        return bboxes, scores

