"""
Bounding box utilities for IoU, NMS, and coordinate transformations
"""
import numpy as np
from typing import List, Tuple, Union


def bbox_iou(bbox1: Union[List, np.ndarray], bbox2: Union[List, np.ndarray]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)
    
    # Calculate intersection coordinates
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return float(iou)


def bbox_iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between two sets of bounding boxes (vectorized)
    
    Args:
        bboxes1: Array of shape (N, 4) with format [x1, y1, x2, y2]
        bboxes2: Array of shape (M, 4) with format [x1, y1, x2, y2]
        
    Returns:
        IoU matrix of shape (N, M)
    """
    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)
    
    if bboxes1.ndim == 1:
        bboxes1 = bboxes1.reshape(1, -1)
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2.reshape(1, -1)
    
    # Calculate areas
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    
    # Calculate intersection
    x1_inter = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0].T)
    y1_inter = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1].T)
    x2_inter = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2].T)
    y2_inter = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3].T)
    
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Calculate union
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    # Calculate IoU
    iou = inter_area / np.maximum(union_area, 1e-6)
    
    return iou


def nms(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """
    Non-Maximum Suppression
    
    Args:
        bboxes: Array of shape (N, 4) with format [x1, y1, x2, y2]
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(bboxes) == 0:
        return []
    
    bboxes = np.array(bboxes)
    scores = np.array(scores)
    
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def bbox_from_keypoints(keypoints: np.ndarray, margin: float = 10.0) -> np.ndarray:
    """
    Create bounding box from keypoints
    
    Args:
        keypoints: Array of shape (N, 2) with (x, y) coordinates
        margin: Margin to add around keypoints
        
    Returns:
        Bounding box [x1, y1, x2, y2]
    """
    if len(keypoints) == 0:
        return np.array([0, 0, 0, 0])
    
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    
    x1 = max(0, np.min(x_coords) - margin)
    y1 = max(0, np.min(y_coords) - margin)
    x2 = np.max(x_coords) + margin
    y2 = np.max(y_coords) + margin
    
    return np.array([x1, y1, x2, y2])


def bbox_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """
    Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]
    
    Args:
        bbox: [x1, y1, x2, y2]
        
    Returns:
        [x, y, w, h] where (x, y) is top-left corner
    """
    return np.array([
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
    ])


def bbox_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """
    Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
    
    Args:
        bbox: [x, y, w, h]
        
    Returns:
        [x1, y1, x2, y2]
    """
    return np.array([
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3]
    ])


def clip_bbox(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Clip bounding box to image boundaries
    
    Args:
        bbox: [x1, y1, x2, y2]
        img_width: Image width
        img_height: Image height
        
    Returns:
        Clipped bounding box
    """
    bbox = np.array(bbox).copy()
    bbox[0] = np.clip(bbox[0], 0, img_width)
    bbox[1] = np.clip(bbox[1], 0, img_height)
    bbox[2] = np.clip(bbox[2], 0, img_width)
    bbox[3] = np.clip(bbox[3], 0, img_height)
    return bbox


def bbox_area(bbox: np.ndarray) -> float:
    """
    Calculate bounding box area
    
    Args:
        bbox: [x1, y1, x2, y2]
        
    Returns:
        Area in pixels
    """
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def scale_bbox(bbox: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale bounding box around its center
    
    Args:
        bbox: [x1, y1, x2, y2]
        scale: Scale factor (>1 enlarges, <1 shrinks)
        
    Returns:
        Scaled bounding box
    """
    bbox = np.array(bbox)
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    new_width = width * scale
    new_height = height * scale
    
    return np.array([
        center_x - new_width / 2,
        center_y - new_height / 2,
        center_x + new_width / 2,
        center_y + new_height / 2
    ])

