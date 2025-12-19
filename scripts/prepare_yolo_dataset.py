"""
Convert annotations.json to YOLO format for training

YOLO format:
- images/train/*.jpg
- images/val/*.jpg
- labels/train/*.txt (one per image)
- labels/val/*.txt

Each label line: class x_center y_center width height (normalized 0-1)
"""
import json
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_bbox_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """Convert (x1, y1, x2, y2) to YOLO format (x_center, y_center, w, h) normalized"""
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Clip to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def extract_frames_with_annotations(video_path, annotations, output_dir):
    """Extract frames that have annotations"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create dict: frame_id -> list of bboxes
    frame_annotations = {}
    for bbox in annotations:
        frame_id = bbox['frame']
        if frame_id not in frame_annotations:
            frame_annotations[frame_id] = []
        frame_annotations[frame_id].append(bbox)
    
    # Extract frames
    extracted = []
    frame_ids = sorted(frame_annotations.keys())
    
    print(f"  Extracting {len(frame_ids)} annotated frames from {total_frames} total frames...")
    
    for frame_id in tqdm(frame_ids, desc="  Frames", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  Warning: Cannot read frame {frame_id}")
            continue
        
        # Save frame
        frame_filename = f"frame_{frame_id:06d}.jpg"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        # Convert bboxes to YOLO format
        yolo_labels = []
        for bbox in frame_annotations[frame_id]:
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
                img_width, img_height
            )
            # Class 0 (single class per object type)
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        extracted.append({
            'frame_id': frame_id,
            'filename': frame_filename,
            'labels': yolo_labels,
            'img_width': img_width,
            'img_height': img_height
        })
    
    cap.release()
    return extracted


def prepare_yolo_dataset(data_root, output_root, object_name=None):
    """
    Prepare YOLO dataset from annotations
    
    Args:
        data_root: Path to observing/train/
        output_root: Path to output YOLO dataset
        object_name: If specified, only process this object (e.g., "Backpack")
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    # Load annotations
    annotations_path = data_root / "annotations" / "annotations.json"
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r') as f:
        all_annotations = json.load(f)
    
    # Group by object type
    object_groups = {}
    for entry in all_annotations:
        video_id = entry['video_id']
        # Extract object name and index (e.g., "Backpack_0" -> "Backpack", 0)
        parts = video_id.rsplit('_', 1)
        if len(parts) != 2:
            print(f"Warning: Unexpected video_id format: {video_id}")
            continue
        
        obj_name, idx = parts[0], int(parts[1])
        
        if object_name and obj_name != object_name:
            continue
        
        if obj_name not in object_groups:
            object_groups[obj_name] = {}
        
        object_groups[obj_name][idx] = entry
    
    print(f"\nFound {len(object_groups)} object types to process")
    
    # Process each object
    for obj_name, videos in object_groups.items():
        print(f"\n{'='*60}")
        print(f"Processing: {obj_name}")
        print(f"{'='*60}")
        
        if 0 not in videos or 1 not in videos:
            print(f"  Warning: {obj_name} doesn't have both video 0 and 1. Skipping.")
            continue
        
        # Create output directories
        obj_output_dir = output_root / obj_name
        for split in ['train', 'val']:
            (obj_output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (obj_output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Process train (video_0) and val (video_1)
        for idx, split in [(0, 'train'), (1, 'val')]:
            print(f"\n[{split.upper()}] Processing {obj_name}_{idx}...")
            
            video_path = data_root / "samples" / f"{obj_name}_{idx}" / "drone_video.mp4"
            if not video_path.exists():
                print(f"  Error: Video not found: {video_path}")
                continue
            
            annotations = videos[idx]['annotations'][0]['bboxes']
            print(f"  Found {len(annotations)} annotations")
            
            # Extract frames
            img_output_dir = obj_output_dir / 'images' / split
            extracted = extract_frames_with_annotations(video_path, annotations, img_output_dir)
            
            # Save labels
            label_output_dir = obj_output_dir / 'labels' / split
            for item in extracted:
                label_filename = item['filename'].replace('.jpg', '.txt')
                label_path = label_output_dir / label_filename
                with open(label_path, 'w') as f:
                    f.write('\n'.join(item['labels']))
            
            print(f"  ✓ Extracted {len(extracted)} frames to {split}")
        
        # Create data.yaml
        yaml_path = obj_output_dir / 'data.yaml'
        yaml_content = f"""# YOLO dataset config for {obj_name}
path: {obj_output_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: 1
names: ['{obj_name}']
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Dataset prepared for {obj_name}")
        print(f"  Location: {obj_output_dir}")
        print(f"  Config: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO training dataset from annotations")
    parser.add_argument(
        '--data-root',
        type=str,
        default='observing/train',
        help='Path to observing/train/ directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='yolo_datasets',
        help='Output directory for YOLO datasets'
    )
    parser.add_argument(
        '--object',
        type=str,
        default=None,
        help='Specific object to process (e.g., Backpack). If not specified, process all.'
    )
    
    args = parser.parse_args()
    
    prepare_yolo_dataset(
        data_root=args.data_root,
        output_root=args.output,
        object_name=args.object
    )
    
    print("\n" + "="*60)
    print("DONE! Dataset preparation complete")
    print("="*60)


if __name__ == '__main__':
    main()

