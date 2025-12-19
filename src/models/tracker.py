"""
Object tracking using ByteTrack algorithm
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from ..utils.bbox_utils import bbox_iou, bbox_to_xywh, bbox_to_xyxy


@dataclass
class TrackState:
    """Track state enumeration"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass
class STrack:
    """Single object track"""
    
    def __init__(self, bbox: np.ndarray, score: float):
        """
        Initialize track
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            score: Confidence score
        """
        self.bbox = bbox
        self.score = score
        self.track_id = 0
        self.frame_id = 0
        self.tracklet_len = 0
        self.state = TrackState.New
        
        # Kalman filter state (simplified)
        self.mean = None
        self.covariance = None
        
        # Initialize from bbox
        self._init_track()
    
    def _init_track(self):
        """Initialize track state"""
        xywh = bbox_to_xywh(self.bbox)
        self.mean = np.r_[xywh, np.zeros(4)]  # [x, y, w, h, vx, vy, vw, vh]
        self.covariance = np.eye(8) * 10
    
    def predict(self):
        """Predict next state"""
        # Simple constant velocity model
        self.mean[:4] += self.mean[4:]
        
        # Update bbox
        self.bbox = bbox_to_xyxy(self.mean[:4])
    
    def update(self, bbox: np.ndarray, score: float):
        """
        Update track with new detection
        
        Args:
            bbox: New bounding box
            score: Detection score
        """
        self.bbox = bbox
        self.score = score
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        
        # Update mean
        new_xywh = bbox_to_xywh(bbox)
        velocity = new_xywh - self.mean[:4]
        self.mean[:4] = new_xywh
        self.mean[4:] = velocity * 0.5  # Smooth velocity
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = TrackState.Lost
    
    def mark_removed(self):
        """Mark track as removed"""
        self.state = TrackState.Removed
    
    @staticmethod
    def multi_predict(tracks: List['STrack']):
        """Predict multiple tracks"""
        for track in tracks:
            track.predict()


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    
    Simplified implementation of ByteTrack algorithm
    """
    
    def __init__(
        self,
        track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 25
    ):
        """
        Initialize ByteTracker
        
        Args:
            track_thresh: Threshold for high-confidence detections
            match_thresh: Matching threshold
            track_buffer: Number of frames to keep lost tracks
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        
        self.tracked_tracks = []  # Active tracks
        self.lost_tracks = []  # Lost tracks
        self.removed_tracks = []  # Removed tracks
        
        self.frame_id = 0
        self.track_id_count = 0
    
    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray
    ) -> List[STrack]:
        """
        Update tracker with new detections
        
        Args:
            detections: Detection bboxes [N, 4] in format [x1, y1, x2, y2]
            scores: Detection scores [N]
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        
        # Create new tracks from detections
        if len(detections) > 0:
            detections = np.array(detections)
            scores = np.array(scores)
            
            # Separate high and low confidence detections
            high_indices = scores >= self.track_thresh
            low_indices = (scores >= 0.1) & (scores < self.track_thresh)
            
            detections_high = detections[high_indices]
            scores_high = scores[high_indices]
            
            detections_low = detections[low_indices]
            scores_low = scores[low_indices]
        else:
            detections_high = np.array([])
            scores_high = np.array([])
            detections_low = np.array([])
            scores_low = np.array([])
        
        # Predict existing tracks
        STrack.multi_predict(self.tracked_tracks)
        STrack.multi_predict(self.lost_tracks)
        
        # First association: match high-confidence detections with tracked tracks
        if len(detections_high) > 0:
            matches, unmatched_tracks, unmatched_dets = self._associate(
                self.tracked_tracks,
                detections_high,
                scores_high,
                self.match_thresh
            )
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                self.tracked_tracks[track_idx].update(
                    detections_high[det_idx],
                    scores_high[det_idx]
                )
            
            # Handle unmatched tracks - try matching with low confidence detections
            unmatched_tracked = [self.tracked_tracks[i] for i in unmatched_tracks]
        else:
            unmatched_tracked = self.tracked_tracks
            unmatched_dets = []
        
        # Second association: match remaining tracks with low confidence detections
        if len(detections_low) > 0 and len(unmatched_tracked) > 0:
            matches2, unmatched_tracks2, unmatched_dets_low = self._associate(
                unmatched_tracked,
                detections_low,
                scores_low,
                0.5
            )
            
            # Update matched tracks
            for track_idx, det_idx in matches2:
                unmatched_tracked[track_idx].update(
                    detections_low[det_idx],
                    scores_low[det_idx]
                )
            
            # Mark remaining as lost
            for i in unmatched_tracks2:
                unmatched_tracked[i].mark_lost()
        else:
            # Mark unmatched tracks as lost
            for track in unmatched_tracked:
                track.mark_lost()
        
        # Create new tracks from unmatched high-confidence detections
        if len(detections_high) > 0:
            for det_idx in unmatched_dets:
                track = STrack(detections_high[det_idx], scores_high[det_idx])
                track.track_id = self._next_id()
                track.frame_id = self.frame_id
                self.tracked_tracks.append(track)
        
        # Move lost tracks
        self.lost_tracks = [track for track in self.tracked_tracks if track.state == TrackState.Lost]
        self.tracked_tracks = [track for track in self.tracked_tracks if track.state == TrackState.Tracked]
        
        # Remove old lost tracks
        self.lost_tracks = [
            track for track in self.lost_tracks
            if self.frame_id - track.frame_id <= self.track_buffer
        ]
        
        # Return active tracks
        return self.tracked_tracks
    
    def _associate(
        self,
        tracks: List[STrack],
        detections: np.ndarray,
        scores: np.ndarray,
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU
        
        Returns:
            Tuple of (matches, unmatched_track_indices, unmatched_detection_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det_bbox in enumerate(detections):
                iou_matrix[i, j] = bbox_iou(track.bbox, det_bbox)
        
        # Greedy matching
        matches = []
        unmatched_tracks = []
        unmatched_dets = []
        
        matched_tracks = set()
        matched_dets = set()
        
        # Sort by IoU
        indices = np.argsort(-iou_matrix.ravel())
        
        for idx in indices:
            track_idx = idx // len(detections)
            det_idx = idx % len(detections)
            
            if track_idx in matched_tracks or det_idx in matched_dets:
                continue
            
            if iou_matrix[track_idx, det_idx] >= thresh:
                matches.append((track_idx, det_idx))
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
        
        # Unmatched
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _next_id(self) -> int:
        """Get next track ID"""
        self.track_id_count += 1
        return self.track_id_count
    
    def reset(self):
        """Reset tracker"""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.track_id_count = 0

