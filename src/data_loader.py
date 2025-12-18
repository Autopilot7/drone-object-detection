"""
Data loading utilities for drone object detection dataset
"""
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BBox:
    """Bounding box with frame information"""
    frame: int
    x1: int
    y1: int
    x2: int
    y2: int
    
    def to_dict(self) -> Dict:
        return {
            'frame': self.frame,
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2
        }
    
    def to_xyxy(self) -> np.ndarray:
        """Return as [x1, y1, x2, y2] array"""
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    def to_xywh(self) -> np.ndarray:
        """Return as [x, y, w, h] array"""
        return np.array([self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1])


@dataclass
class VideoSample:
    """Single video sample with reference images and annotations"""
    video_id: str
    video_path: Path
    reference_images: List[np.ndarray]
    reference_image_paths: List[Path]
    annotations: List[List[BBox]]  # List of detection sequences
    
    def __post_init__(self):
        # Load reference images if not already loaded
        if not self.reference_images and self.reference_image_paths:
            self.reference_images = [
                cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                for p in self.reference_image_paths
            ]


class DroneDataset:
    """
    Dataset loader for drone object detection challenge
    """
    
    def __init__(self, data_root: str):
        """
        Initialize dataset
        
        Args:
            data_root: Path to training data root (e.g., 'observing/train')
        """
        self.data_root = Path(data_root)
        self.samples_dir = self.data_root / "samples"
        self.annotations_file = self.data_root / "annotations" / "annotations.json"
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Get all video samples
        self.samples = self._load_samples()
        
    def _load_annotations(self) -> Dict:
        """Load annotations from JSON file"""
        with open(self.annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Convert to dictionary keyed by video_id
        annotations_dict = {}
        for item in annotations:
            video_id = item['video_id']
            annotations_dict[video_id] = item['annotations']
        
        return annotations_dict
    
    def _load_samples(self) -> List[VideoSample]:
        """Load all video samples"""
        samples = []
        
        for video_dir in sorted(self.samples_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            
            video_id = video_dir.name
            video_path = video_dir / "drone_video.mp4"
            
            if not video_path.exists():
                print(f"Warning: Video not found for {video_id}")
                continue
            
            # Load reference image paths
            ref_img_dir = video_dir / "object_images"
            ref_img_paths = sorted(ref_img_dir.glob("*.jpg"))
            
            if len(ref_img_paths) != 3:
                print(f"Warning: Expected 3 reference images for {video_id}, found {len(ref_img_paths)}")
            
            # Parse annotations
            annotations = self._parse_annotations(video_id)
            
            sample = VideoSample(
                video_id=video_id,
                video_path=video_path,
                reference_images=[],
                reference_image_paths=ref_img_paths,
                annotations=annotations
            )
            
            samples.append(sample)
        
        return samples
    
    def _parse_annotations(self, video_id: str) -> List[List[BBox]]:
        """
        Parse annotations for a video
        
        Returns:
            List of detection sequences (each sequence is a list of BBox)
        """
        if video_id not in self.annotations:
            return []
        
        sequences = []
        for ann in self.annotations[video_id]:
            bboxes = []
            for bbox_dict in ann['bboxes']:
                bbox = BBox(
                    frame=bbox_dict['frame'],
                    x1=bbox_dict['x1'],
                    y1=bbox_dict['y1'],
                    x2=bbox_dict['x2'],
                    y2=bbox_dict['y2']
                )
                bboxes.append(bbox)
            sequences.append(bboxes)
        
        return sequences
    
    def get_sample(self, video_id: str) -> Optional[VideoSample]:
        """Get sample by video ID"""
        for sample in self.samples:
            if sample.video_id == video_id:
                # Load reference images
                if not sample.reference_images:
                    sample.reference_images = [
                        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                        for p in sample.reference_image_paths
                    ]
                return sample
        return None
    
    def get_sample_by_index(self, idx: int) -> VideoSample:
        """Get sample by index"""
        sample = self.samples[idx]
        # Load reference images
        if not sample.reference_images:
            sample.reference_images = [
                cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                for p in sample.reference_image_paths
            ]
        return sample
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> VideoSample:
        return self.get_sample_by_index(idx)
    
    def get_video_ids(self) -> List[str]:
        """Get list of all video IDs"""
        return [sample.video_id for sample in self.samples]
    
    def get_categories(self) -> List[str]:
        """Get unique object categories"""
        categories = set()
        for sample in self.samples:
            # Extract category from video_id (e.g., "Backpack_0" -> "Backpack")
            category = '_'.join(sample.video_id.split('_')[:-1])
            categories.add(category)
        return sorted(list(categories))
    
    def split_train_val(self, val_ratio: float = 0.2, random_seed: int = 42) -> Tuple[List[int], List[int]]:
        """
        Split dataset into train and validation sets
        
        Args:
            val_ratio: Ratio of validation samples
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices)
        """
        np.random.seed(random_seed)
        
        # Group by category to ensure balanced split
        category_indices = {}
        for idx, sample in enumerate(self.samples):
            category = '_'.join(sample.video_id.split('_')[:-1])
            if category not in category_indices:
                category_indices[category] = []
            category_indices[category].append(idx)
        
        train_indices = []
        val_indices = []
        
        # Split each category
        for category, indices in category_indices.items():
            indices = np.array(indices)
            np.random.shuffle(indices)
            
            n_val = max(1, int(len(indices) * val_ratio))
            val_indices.extend(indices[:n_val].tolist())
            train_indices.extend(indices[n_val:].tolist())
        
        return sorted(train_indices), sorted(val_indices)


def load_reference_images(video_dir: Path) -> List[np.ndarray]:
    """
    Load reference images for a video
    
    Args:
        video_dir: Path to video directory
        
    Returns:
        List of reference images (RGB)
    """
    ref_img_dir = video_dir / "object_images"
    ref_img_paths = sorted(ref_img_dir.glob("*.jpg"))
    
    images = []
    for path in ref_img_paths:
        img = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    
    return images


def annotations_to_dict_format(predictions: List[List[BBox]], video_id: str) -> Dict:
    """
    Convert predictions to submission format
    
    Args:
        predictions: List of detection sequences
        video_id: Video identifier
        
    Returns:
        Dictionary in submission format
    """
    detections = []
    for sequence in predictions:
        if sequence:
            bboxes = [bbox.to_dict() for bbox in sequence]
            detections.append({'bboxes': bboxes})
    
    return {
        'video_id': video_id,
        'detections': detections
    }


def save_predictions(predictions: List[Dict], output_path: str) -> None:
    """
    Save predictions to JSON file
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)


def load_predictions(predictions_path: str) -> Dict[str, List[List[BBox]]]:
    """
    Load predictions from JSON file
    
    Args:
        predictions_path: Path to predictions JSON
        
    Returns:
        Dictionary mapping video_id to list of detection sequences
    """
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    predictions_dict = {}
    for item in data:
        video_id = item['video_id']
        sequences = []
        
        for detection in item.get('detections', []):
            bboxes = []
            for bbox_dict in detection['bboxes']:
                bbox = BBox(
                    frame=bbox_dict['frame'],
                    x1=bbox_dict['x1'],
                    y1=bbox_dict['y1'],
                    x2=bbox_dict['x2'],
                    y2=bbox_dict['y2']
                )
                bboxes.append(bbox)
            sequences.append(bboxes)
        
        predictions_dict[video_id] = sequences
    
    return predictions_dict

