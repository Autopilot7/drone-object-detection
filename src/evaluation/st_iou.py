"""
Spatio-Temporal IoU (ST-IoU) evaluation metric implementation
"""
import numpy as np
from typing import Dict, List, Tuple
from ..data_loader import BBox
from ..utils.bbox_utils import bbox_iou


def compute_spatial_iou(bbox1: BBox, bbox2: BBox) -> float:
    """
    Compute spatial IoU between two bounding boxes
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        IoU value
    """
    return bbox_iou(bbox1.to_xyxy(), bbox2.to_xyxy())


def compute_st_iou_single_sequence(
    ground_truth: List[BBox],
    predictions: List[BBox],
    frame_tolerance: int = 0
) -> float:
    """
    Compute ST-IoU for a single detection sequence
    
    Args:
        ground_truth: List of ground truth bounding boxes
        predictions: List of predicted bounding boxes
        frame_tolerance: Tolerance in frame matching (0 = exact match)
        
    Returns:
        ST-IoU value between 0 and 1
    """
    if not ground_truth:
        return 0.0 if predictions else 1.0
    
    if not predictions:
        return 0.0
    
    # Create frame-to-bbox mappings
    gt_frames = {bbox.frame: bbox for bbox in ground_truth}
    pred_frames = {bbox.frame: bbox for bbox in predictions}
    
    # Get all frames (union)
    all_frames = set(gt_frames.keys()) | set(pred_frames.keys())
    
    if not all_frames:
        return 0.0
    
    # Calculate sum of spatial IoUs for intersection frames
    iou_sum = 0.0
    intersection_count = 0
    
    for frame in all_frames:
        gt_bbox = gt_frames.get(frame)
        pred_bbox = pred_frames.get(frame)
        
        # Check if both GT and prediction exist for this frame
        if gt_bbox and pred_bbox:
            spatial_iou = compute_spatial_iou(gt_bbox, pred_bbox)
            iou_sum += spatial_iou
            intersection_count += 1
        elif frame_tolerance > 0:
            # Check nearby frames within tolerance
            matched = False
            if gt_bbox:
                for offset in range(-frame_tolerance, frame_tolerance + 1):
                    nearby_frame = frame + offset
                    if nearby_frame in pred_frames:
                        spatial_iou = compute_spatial_iou(gt_bbox, pred_frames[nearby_frame])
                        iou_sum += spatial_iou
                        intersection_count += 1
                        matched = True
                        break
            elif pred_bbox:
                for offset in range(-frame_tolerance, frame_tolerance + 1):
                    nearby_frame = frame + offset
                    if nearby_frame in gt_frames:
                        spatial_iou = compute_spatial_iou(gt_frames[nearby_frame], pred_bbox)
                        iou_sum += spatial_iou
                        intersection_count += 1
                        matched = True
                        break
    
    # ST-IoU = sum(spatial_IoUs) / union_frame_count
    union_frame_count = len(all_frames)
    st_iou = iou_sum / union_frame_count if union_frame_count > 0 else 0.0
    
    return st_iou


def match_sequences(
    ground_truth_sequences: List[List[BBox]],
    predicted_sequences: List[List[BBox]]
) -> List[Tuple[int, int, float]]:
    """
    Match predicted sequences to ground truth sequences using Hungarian algorithm
    
    Args:
        ground_truth_sequences: List of GT sequences
        predicted_sequences: List of predicted sequences
        
    Returns:
        List of (gt_idx, pred_idx, st_iou) tuples
    """
    if not ground_truth_sequences or not predicted_sequences:
        return []
    
    n_gt = len(ground_truth_sequences)
    n_pred = len(predicted_sequences)
    
    # Compute ST-IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    
    for i, gt_seq in enumerate(ground_truth_sequences):
        for j, pred_seq in enumerate(predicted_sequences):
            iou_matrix[i, j] = compute_st_iou_single_sequence(gt_seq, pred_seq)
    
    # Simple greedy matching (can be replaced with Hungarian algorithm)
    matches = []
    used_gt = set()
    used_pred = set()
    
    # Sort by IoU value (descending)
    candidates = []
    for i in range(n_gt):
        for j in range(n_pred):
            candidates.append((i, j, iou_matrix[i, j]))
    
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    for gt_idx, pred_idx, iou_val in candidates:
        if gt_idx not in used_gt and pred_idx not in used_pred:
            matches.append((gt_idx, pred_idx, iou_val))
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
    
    return matches


def compute_st_iou_video(
    ground_truth_sequences: List[List[BBox]],
    predicted_sequences: List[List[BBox]],
    frame_tolerance: int = 0
) -> float:
    """
    Compute ST-IoU for a video with multiple detection sequences
    
    Args:
        ground_truth_sequences: List of ground truth sequences
        predicted_sequences: List of predicted sequences
        frame_tolerance: Tolerance in frame matching
        
    Returns:
        Mean ST-IoU across all sequences
    """
    if not ground_truth_sequences:
        return 0.0 if predicted_sequences else 1.0
    
    if not predicted_sequences:
        return 0.0
    
    # Match sequences
    matches = match_sequences(ground_truth_sequences, predicted_sequences)
    
    # Calculate mean ST-IoU
    if not matches:
        return 0.0
    
    total_iou = sum(iou for _, _, iou in matches)
    
    # Penalize unmatched sequences
    n_unmatched_gt = len(ground_truth_sequences) - len(matches)
    n_unmatched_pred = len(predicted_sequences) - len(matches)
    
    # Average over all GT sequences (unmatched contribute 0)
    mean_st_iou = total_iou / len(ground_truth_sequences)
    
    return mean_st_iou


def evaluate_dataset(
    ground_truth: Dict[str, List[List[BBox]]],
    predictions: Dict[str, List[List[BBox]]],
    frame_tolerance: int = 0
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate entire dataset
    
    Args:
        ground_truth: Dictionary mapping video_id to GT sequences
        predictions: Dictionary mapping video_id to predicted sequences
        frame_tolerance: Tolerance in frame matching
        
    Returns:
        Tuple of (mean_st_iou, per_video_st_iou)
    """
    per_video_scores = {}
    
    for video_id in ground_truth.keys():
        gt_sequences = ground_truth.get(video_id, [])
        pred_sequences = predictions.get(video_id, [])
        
        st_iou = compute_st_iou_video(gt_sequences, pred_sequences, frame_tolerance)
        per_video_scores[video_id] = st_iou
    
    # Calculate mean across all videos
    mean_st_iou = np.mean(list(per_video_scores.values())) if per_video_scores else 0.0
    
    return mean_st_iou, per_video_scores


def compute_temporal_metrics(
    ground_truth: List[BBox],
    predictions: List[BBox]
) -> Dict[str, float]:
    """
    Compute temporal precision, recall, and F1 score
    
    Args:
        ground_truth: List of ground truth bounding boxes
        predictions: List of predicted bounding boxes
        
    Returns:
        Dictionary with temporal metrics
    """
    if not ground_truth:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    gt_frames = set(bbox.frame for bbox in ground_truth)
    pred_frames = set(bbox.frame for bbox in predictions)
    
    if not pred_frames:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Calculate temporal overlap
    intersection = gt_frames & pred_frames
    
    precision = len(intersection) / len(pred_frames) if pred_frames else 0.0
    recall = len(intersection) / len(gt_frames) if gt_frames else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_st_iou(
    ground_truth: List[BBox],
    predictions: List[BBox],
    frame_tolerance: int = 0
) -> float:
    """
    Simple wrapper for computing ST-IoU from flat lists of BBox
    
    Args:
        ground_truth: Flat list of ground truth bounding boxes
        predictions: Flat list of predicted bounding boxes
        frame_tolerance: Tolerance in frame matching
        
    Returns:
        ST-IoU value
    """
    # Treat all boxes as single sequence
    gt_sequences = [ground_truth] if ground_truth else []
    pred_sequences = [predictions] if predictions else []
    
    return compute_st_iou_video(gt_sequences, pred_sequences, frame_tolerance)


def compute_spatial_metrics(
    ground_truth: List[BBox],
    predictions: List[BBox],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute spatial precision, recall at given IoU threshold
    
    Args:
        ground_truth: List of ground truth bounding boxes
        predictions: List of predicted bounding boxes
        iou_threshold: IoU threshold for considering a match
        
    Returns:
        Dictionary with spatial metrics
    """
    if not ground_truth:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Group by frame
    gt_by_frame = {}
    pred_by_frame = {}
    
    for bbox in ground_truth:
        if bbox.frame not in gt_by_frame:
            gt_by_frame[bbox.frame] = []
        gt_by_frame[bbox.frame].append(bbox)
    
    for bbox in predictions:
        if bbox.frame not in pred_by_frame:
            pred_by_frame[bbox.frame] = []
        pred_by_frame[bbox.frame].append(bbox)
    
    # Count matches
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    all_frames = set(gt_by_frame.keys()) | set(pred_by_frame.keys())
    
    for frame in all_frames:
        gt_bboxes = gt_by_frame.get(frame, [])
        pred_bboxes = pred_by_frame.get(frame, [])
        
        matched_gt = set()
        matched_pred = set()
        
        # Match predictions to GT
        for i, pred_bbox in enumerate(pred_bboxes):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt_bbox in enumerate(gt_bboxes):
                if j in matched_gt:
                    continue
                
                iou = compute_spatial_iou(gt_bbox, pred_bbox)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
        
        false_positives += len(pred_bboxes) - len(matched_pred)
        false_negatives += len(gt_bboxes) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

