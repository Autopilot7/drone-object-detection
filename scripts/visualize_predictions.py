"""
Visualize model predictions vs ground truth
"""
import cv2
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DroneDataset
from ultralytics import YOLO

OBJECT_CLASSES = {
    'Backpack': 0,
    'Jacket': 1,
    'Laptop': 2,
    'Lifering': 3,
    'MobilePhone': 4,
    'Person1': 5,
    'WaterBottle': 6
}

def visualize_video(video_id, model_path, output_dir, max_frames=100):
    """Visualize predictions on video"""
    dataset = DroneDataset('observing/train')
    sample = dataset.get_sample(video_id)
    
    # Load model
    model = YOLO(model_path)
    
    # Get object type
    obj_name = video_id.rsplit('_', 1)[0]
    target_class = OBJECT_CLASSES[obj_name]
    
    # Read video
    cap = cv2.VideoCapture(str(sample.video_path))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get GT frames
    gt_frames = set()
    for seq in sample.annotations:
        for bbox in seq:
            gt_frames.add(bbox.frame)
    
    print(f"Video: {video_id}")
    print(f"Target class: {obj_name} (class {target_class})")
    print(f"GT frames: {sorted(list(gt_frames))[:20]}")
    print(f"Total GT frames: {len(gt_frames)}")
    
    frame_idx = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip if not GT frame
        if frame_idx not in gt_frames:
            frame_idx += 1
            continue
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Draw predictions
        vis_frame = frame.copy()
        
        # Draw GT
        for seq in sample.annotations:
            for bbox in seq:
                if bbox.frame == frame_idx:
                    cv2.rectangle(vis_frame, 
                                (bbox.x1, bbox.y1), 
                                (bbox.x2, bbox.y2), 
                                (0, 255, 0), 2)  # Green = GT
                    cv2.putText(vis_frame, "GT", 
                              (bbox.x1, bbox.y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw predictions
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Color based on class match
                if cls == target_class:
                    color = (0, 0, 255)  # Red = Correct class
                    label = f"{obj_name} {conf:.2f}"
                else:
                    color = (255, 0, 0)  # Blue = Wrong class
                    class_name = [k for k, v in OBJECT_CLASSES.items() if v == cls][0]
                    label = f"{class_name} {conf:.2f}"
                
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, label,
                          (x1, y2+20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame info
        cv2.putText(vis_frame, f"Frame {frame_idx}",
                  (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save
        output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), vis_frame)
        print(f"  Saved: {output_path}")
        
        saved_count += 1
        frame_idx += 1
    
    cap.release()
    print(f"\n✓ Saved {saved_count} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, 
                       help='Video ID (e.g., WaterBottle_1)')
    parser.add_argument('--model', type=str, 
                       default='runs/train_unified/drone_detector/weights/best.pt')
    parser.add_argument('--output', type=str, default='output/visualizations')
    parser.add_argument('--max-frames', type=int, default=50)
    
    args = parser.parse_args()
    
    visualize_video(args.video, args.model, args.output, args.max_frames)


if __name__ == '__main__':
    main()

