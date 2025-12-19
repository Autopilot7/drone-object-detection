"""
Prepare UNIFIED YOLO dataset with ALL objects as different classes

Instead of 7 separate models, train 1 model with 7 classes:
  0: Backpack
  1: Jacket
  2: Laptop
  3: Lifering
  4: MobilePhone
  5: Person1
  6: WaterBottle
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


def extract_frames_with_annotations(video_path, annotations, output_dir, class_id):
    """Extract frames that have annotations"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    
    for frame_id in tqdm(frame_ids, desc="    Frames", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert bboxes to YOLO format with correct class_id
        yolo_labels = []
        for bbox in frame_annotations[frame_id]:
            x_center, y_center, width, height = convert_bbox_to_yolo(
                bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
                img_width, img_height
            )
            # Use the object's class_id
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        extracted.append({
            'frame_id': frame_id,
            'frame': frame,
            'labels': yolo_labels,
        })
    
    cap.release()
    return extracted


def prepare_unified_dataset(data_root, output_root):
    """
    Prepare unified YOLO dataset with all objects as different classes
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    # Class mapping
    OBJECT_CLASSES = {
        'Backpack': 0,
        'Jacket': 1,
        'Laptop': 2,
        'Lifering': 3,
        'MobilePhone': 4,
        'Person1': 5,
        'WaterBottle': 6
    }
    
    # Create output directories
    for split in ['train', 'val']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotations_path = data_root / "annotations" / "annotations.json"
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r') as f:
        all_annotations = json.load(f)
    
    # Group by object type
    object_groups = {}
    for entry in all_annotations:
        video_id = entry['video_id']
        parts = video_id.rsplit('_', 1)
        if len(parts) != 2:
            continue
        
        obj_name, idx = parts[0], int(parts[1])
        
        if obj_name not in object_groups:
            object_groups[obj_name] = {}
        
        object_groups[obj_name][idx] = entry
    
    print(f"\nFound {len(object_groups)} object types")
    print(f"Classes: {list(OBJECT_CLASSES.keys())}")
    
    # Process each object
    frame_counter = 0
    stats = {'train': 0, 'val': 0}
    
    for obj_name in sorted(OBJECT_CLASSES.keys()):
        if obj_name not in object_groups:
            print(f"  Warning: {obj_name} not found in dataset")
            continue
        
        class_id = OBJECT_CLASSES[obj_name]
        videos = object_groups[obj_name]
        
        print(f"\n{'='*60}")
        print(f"Processing: {obj_name} (class {class_id})")
        print(f"{'='*60}")
        
        if 0 not in videos or 1 not in videos:
            print(f"  Warning: Missing videos for {obj_name}")
            continue
        
        # Process train (video_0) and val (video_1)
        for idx, split in [(0, 'train'), (1, 'val')]:
            print(f"\n  [{split.upper()}] {obj_name}_{idx}")
            
            video_path = data_root / "samples" / f"{obj_name}_{idx}" / "drone_video.mp4"
            if not video_path.exists():
                print(f"    Error: Video not found")
                continue
            
            annotations = videos[idx]['annotations'][0]['bboxes']
            print(f"    Annotations: {len(annotations)} frames")
            
            # Extract frames
            extracted = extract_frames_with_annotations(
                video_path, annotations, None, class_id
            )
            
            # Save frames and labels
            img_output_dir = output_root / 'images' / split
            label_output_dir = output_root / 'labels' / split
            
            for item in extracted:
                # Unique filename across all objects
                filename = f"frame_{frame_counter:06d}"
                frame_counter += 1
                
                # Save image
                img_path = img_output_dir / f"{filename}.jpg"
                cv2.imwrite(str(img_path), item['frame'])
                
                # Save label
                label_path = label_output_dir / f"{filename}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(item['labels']))
            
            stats[split] += len(extracted)
            print(f"    ✓ Extracted {len(extracted)} frames")
    
    # Create data.yaml
    yaml_path = output_root / 'data.yaml'
    yaml_content = f"""# Unified YOLO dataset for all objects
path: {output_root.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(OBJECT_CLASSES)}
names: {list(OBJECT_CLASSES.keys())}
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"Train frames: {stats['train']}")
    print(f"Val frames: {stats['val']}")
    print(f"Total: {stats['train'] + stats['val']}")
    print(f"\nClasses: {list(OBJECT_CLASSES.keys())}")
    print(f"Config: {yaml_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Prepare unified YOLO dataset")
    parser.add_argument(
        '--data-root',
        type=str,
        default='observing/train',
        help='Path to observing/train/ directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='yolo_unified',
        help='Output directory for unified dataset'
    )
    
    args = parser.parse_args()
    
    prepare_unified_dataset(
        data_root=args.data_root,
        output_root=args.output
    )


if __name__ == '__main__':
    main()

