"""
Visualization utilities for detections and evaluation results
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path

from ..data_loader import BBox
from ..utils.video_utils import extract_frame_at_index, create_video_from_frames


def draw_bbox(
    image: np.ndarray,
    bbox: BBox,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        image: Input image
        bbox: Bounding box
        color: Color (R, G, B)
        thickness: Line thickness
        label: Optional label text
        
    Returns:
        Image with bbox drawn
    """
    img = image.copy()
    
    # Draw rectangle
    cv2.rectangle(img, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(
            img,
            (bbox.x1, bbox.y1 - text_height - 5),
            (bbox.x1 + text_width, bbox.y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            label,
            (bbox.x1, bbox.y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return img


def visualize_detections(
    video_path: str,
    predictions: List[List[BBox]],
    ground_truth: Optional[List[List[BBox]]] = None,
    sample_frames: int = 6
) -> None:
    """
    Visualize predictions and ground truth on video frames
    
    Args:
        video_path: Path to video
        predictions: Predicted detection sequences
        ground_truth: Ground truth sequences (optional)
        sample_frames: Number of frames to visualize
    """
    # Collect all frames with annotations
    pred_frames = set()
    for sequence in predictions:
        for bbox in sequence:
            pred_frames.add(bbox.frame)
    
    gt_frames = set()
    if ground_truth:
        for sequence in ground_truth:
            for bbox in sequence:
                gt_frames.add(bbox.frame)
    
    all_frames = sorted(pred_frames | gt_frames)
    
    if not all_frames:
        print("No frames to visualize")
        return
    
    # Sample frames
    step = max(1, len(all_frames) // sample_frames)
    sampled_frames = all_frames[::step][:sample_frames]
    
    # Create frame-to-bbox mappings
    pred_frame_bboxes = {}
    for sequence in predictions:
        for bbox in sequence:
            if bbox.frame not in pred_frame_bboxes:
                pred_frame_bboxes[bbox.frame] = []
            pred_frame_bboxes[bbox.frame].append(bbox)
    
    gt_frame_bboxes = {}
    if ground_truth:
        for sequence in ground_truth:
            for bbox in sequence:
                if bbox.frame not in gt_frame_bboxes:
                    gt_frame_bboxes[bbox.frame] = []
                gt_frame_bboxes[bbox.frame].append(bbox)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, frame_num in enumerate(sampled_frames):
        if idx >= sample_frames:
            break
        
        # Extract frame
        frame = extract_frame_at_index(video_path, frame_num)
        
        # Draw ground truth (green)
        if frame_num in gt_frame_bboxes:
            for bbox in gt_frame_bboxes[frame_num]:
                frame = draw_bbox(frame, bbox, color=(0, 255, 0), label="GT")
        
        # Draw predictions (red)
        if frame_num in pred_frame_bboxes:
            for bbox in pred_frame_bboxes[frame_num]:
                frame = draw_bbox(frame, bbox, color=(255, 0, 0), label="Pred")
        
        axes[idx].imshow(frame)
        axes[idx].set_title(f"Frame {frame_num}")
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(sampled_frames), sample_frames):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_st_iou_results(
    per_video_scores: Dict[str, float],
    title: str = "ST-IoU Scores per Video"
) -> None:
    """
    Plot ST-IoU scores
    
    Args:
        per_video_scores: Dictionary mapping video_id to ST-IoU
        title: Plot title
    """
    video_ids = list(per_video_scores.keys())
    scores = list(per_video_scores.values())
    
    # Extract categories
    categories = ['_'.join(vid.split('_')[:-1]) for vid in video_ids]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color by category
    unique_categories = sorted(set(categories))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
    category_colors = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    
    bar_colors = [category_colors[cat] for cat in categories]
    
    bars = ax.bar(range(len(scores)), scores, color=bar_colors)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(video_ids, rotation=45, ha='right')
    ax.set_ylabel('ST-IoU Score')
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean line
    mean_score = np.mean(scores)
    ax.axhline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_comparison(
    results_dict: Dict[str, Dict[str, float]],
    title: str = "Approach Comparison"
) -> None:
    """
    Compare results from multiple approaches
    
    Args:
        results_dict: Dictionary mapping approach_name to per_video_scores
        title: Plot title
    """
    approaches = list(results_dict.keys())
    video_ids = list(list(results_dict.values())[0].keys())
    
    # Prepare data
    data = []
    for video_id in video_ids:
        for approach in approaches:
            score = results_dict[approach].get(video_id, 0.0)
            data.append({
                'Video': video_id,
                'Approach': approach,
                'ST-IoU': score
            })
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Plot grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Per-video comparison
    ax = axes[0]
    df_pivot = df.pivot(index='Video', columns='Approach', values='ST-IoU')
    df_pivot.plot(kind='bar', ax=ax)
    ax.set_title('ST-IoU per Video')
    ax.set_ylabel('ST-IoU Score')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Approach')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Mean scores
    ax = axes[1]
    mean_scores = df.groupby('Approach')['ST-IoU'].mean()
    bars = ax.bar(range(len(mean_scores)), mean_scores.values)
    ax.set_xticks(range(len(mean_scores)))
    ax.set_xticklabels(mean_scores.index)
    ax.set_ylabel('Mean ST-IoU')
    ax.set_title('Mean ST-IoU by Approach')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def create_annotated_video(
    video_path: str,
    predictions: List[List[BBox]],
    output_path: str,
    ground_truth: Optional[List[List[BBox]]] = None,
    fps: float = 25.0
) -> None:
    """
    Create video with annotations
    
    Args:
        video_path: Input video path
        predictions: Predicted sequences
        output_path: Output video path
        ground_truth: Ground truth sequences (optional)
        fps: Output video fps
    """
    from ..utils.video_utils import extract_frames, create_video_from_frames
    from tqdm import tqdm
    
    # Create frame-to-bbox mappings
    pred_frame_bboxes = {}
    for sequence in predictions:
        for bbox in sequence:
            if bbox.frame not in pred_frame_bboxes:
                pred_frame_bboxes[bbox.frame] = []
            pred_frame_bboxes[bbox.frame].append(bbox)
    
    gt_frame_bboxes = {}
    if ground_truth:
        for sequence in ground_truth:
            for bbox in sequence:
                if bbox.frame not in gt_frame_bboxes:
                    gt_frame_bboxes[bbox.frame] = []
                gt_frame_bboxes[bbox.frame].append(bbox)
    
    # Process frames
    annotated_frames = []
    
    for frame_idx, frame in tqdm(extract_frames(video_path, show_progress=False), desc="Annotating"):
        # Draw ground truth
        if frame_idx in gt_frame_bboxes:
            for bbox in gt_frame_bboxes[frame_idx]:
                frame = draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2, label="GT")
        
        # Draw predictions
        if frame_idx in pred_frame_bboxes:
            for bbox in pred_frame_bboxes[frame_idx]:
                frame = draw_bbox(frame, bbox, color=(255, 0, 0), thickness=2, label="Pred")
        
        annotated_frames.append(frame)
    
    # Create video
    if annotated_frames:
        create_video_from_frames(annotated_frames, output_path, fps)


def plot_confusion_matrix(
    ground_truth: List[List[BBox]],
    predictions: List[List[BBox]],
    iou_threshold: float = 0.5
) -> None:
    """
    Plot temporal confusion matrix (frame-level presence/absence)
    
    Args:
        ground_truth: Ground truth sequences
        predictions: Predicted sequences
        iou_threshold: IoU threshold for positive match
    """
    from ..evaluation.st_iou import compute_spatial_metrics
    
    # Flatten to single list
    gt_flat = []
    for seq in ground_truth:
        gt_flat.extend(seq)
    
    pred_flat = []
    for seq in predictions:
        pred_flat.extend(seq)
    
    # Get frame-level presence
    gt_frames = set(bbox.frame for bbox in gt_flat)
    pred_frames = set(bbox.frame for bbox in pred_flat)
    
    # Calculate confusion matrix elements
    tp = len(gt_frames & pred_frames)  # True positives
    fp = len(pred_frames - gt_frames)  # False positives
    fn = len(gt_frames - pred_frames)  # False negatives
    
    # Note: TN not applicable for detection task
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    matrix = np.array([[tp, fn], [fp, 0]])
    labels = [['TP\n{}'.format(tp), 'FN\n{}'.format(fn)],
              ['FP\n{}'.format(fp), 'TN\nN/A']]
    
    sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Predicted Positive', 'Predicted Negative'])
    
    ax.set_title('Temporal Detection Confusion Matrix')
    plt.tight_layout()
    plt.show()

