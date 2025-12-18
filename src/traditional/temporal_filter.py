"""
Temporal filtering and tracking using Kalman filter and optical flow
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from ..data_loader import BBox
from ..utils.bbox_utils import bbox_iou


@dataclass
class Track:
    """
    Object track with state estimation
    """
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: Optional[np.ndarray] = None  # Kalman filter state
    
    def update(self, bbox: np.ndarray, confidence: float):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
    
    def predict(self):
        """Predict next state"""
        self.age += 1
        self.time_since_update += 1


class KalmanBBoxTracker:
    """
    Bounding box tracker using Kalman filter
    """
    
    def __init__(self):
        """Initialize Kalman filter for bbox tracking"""
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)
        
        # Transition matrix
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = 1.0
        
        # Measurement matrix
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
    
    def init_track(self, bbox: np.ndarray):
        """
        Initialize track with first detection
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # Convert to [x, y, w, h]
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # Initialize state
        self.kf.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
    
    def predict(self) -> np.ndarray:
        """
        Predict next bbox position
        
        Returns:
            Predicted bbox [x1, y1, x2, y2]
        """
        prediction = self.kf.predict()
        
        # Convert back to [x1, y1, x2, y2]
        x, y, w, h = prediction[:4, 0]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        return np.array([x1, y1, x2, y2])
    
    def update(self, bbox: np.ndarray):
        """
        Update tracker with new measurement
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # Convert to [x, y, w, h]
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)


class SimpleTracker:
    """
    Simple multi-object tracker using IoU matching
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.next_id = 0
        self.frame_count = 0
    
    def update(
        self,
        detections: List[np.ndarray],
        confidences: List[float]
    ) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of bboxes [x1, y1, x2, y2]
            confidences: List of confidence scores
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matched_tracks, matched_detections = self._match_detections_to_tracks(
            detections, confidences
        )
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self.tracks[track_idx].update(detections[det_idx], confidences[det_idx])
        
        # Create new tracks for unmatched detections
        unmatched_detections = set(range(len(detections))) - set(matched_detections)
        for det_idx in unmatched_detections:
            track = Track(
                track_id=self.next_id,
                bbox=detections[det_idx],
                confidence=confidences[det_idx],
                hits=1
            )
            self.tracks.append(track)
            self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_age
        ]
        
        # Return confirmed tracks
        confirmed_tracks = [
            track for track in self.tracks
            if track.hits >= self.min_hits
        ]
        
        return confirmed_tracks
    
    def _match_detections_to_tracks(
        self,
        detections: List[np.ndarray],
        confidences: List[float]
    ) -> Tuple[List[int], List[int]]:
        """
        Match detections to existing tracks using IoU
        
        Returns:
            Tuple of (matched_track_indices, matched_detection_indices)
        """
        if not detections or not self.tracks:
            return [], []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, det_bbox in enumerate(detections):
                iou_matrix[i, j] = bbox_iou(track.bbox, det_bbox)
        
        # Greedy matching (can be improved with Hungarian algorithm)
        matched_tracks = []
        matched_detections = []
        
        while True:
            # Find maximum IoU
            if iou_matrix.size == 0:
                break
            
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, det_idx = max_idx
            
            matched_tracks.append(track_idx)
            matched_detections.append(det_idx)
            
            # Remove matched row and column
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0
        
        return matched_tracks, matched_detections
    
    def get_track_bboxes(self, frame_idx: int) -> List[BBox]:
        """
        Get confirmed track bboxes for a frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            List of BBox objects
        """
        confirmed_tracks = [
            track for track in self.tracks
            if track.hits >= self.min_hits and track.time_since_update == 0
        ]
        
        bboxes = []
        for track in confirmed_tracks:
            bbox = BBox(
                frame=frame_idx,
                x1=int(track.bbox[0]),
                y1=int(track.bbox[1]),
                x2=int(track.bbox[2]),
                y2=int(track.bbox[3])
            )
            bboxes.append(bbox)
        
        return bboxes


def group_detections_into_sequences(
    detections: List[BBox],
    max_frame_gap: int = 10
) -> List[List[BBox]]:
    """
    Group temporally close detections into sequences
    
    Args:
        detections: List of all detections
        max_frame_gap: Maximum frame gap to consider same sequence
        
    Returns:
        List of detection sequences
    """
    if not detections:
        return []
    
    # Sort by frame number
    detections = sorted(detections, key=lambda x: x.frame)
    
    sequences = []
    current_sequence = [detections[0]]
    
    for i in range(1, len(detections)):
        frame_gap = detections[i].frame - detections[i-1].frame
        
        if frame_gap <= max_frame_gap:
            current_sequence.append(detections[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [detections[i]]
    
    # Add last sequence
    if current_sequence:
        sequences.append(current_sequence)
    
    return sequences

