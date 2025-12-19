"""
Evaluate unified multi-class YOLO model

One model detects all 7 object types!
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DroneDataset, BBox
from src.evaluation.st_iou import compute_st_iou
from src.utils.video_utils import extract_frames


# Class mapping (must match training)
OBJECT_CLASSES = {
    'Backpack': 0,
    'Jacket': 1,
    'Laptop': 2,
    'Lifering': 3,
    'MobilePhone': 4,
    'Person1': 5,
    'WaterBottle': 6
}

CLASS_TO_NAME = {v: k for k, v in OBJECT_CLASSES.items()}


def load_unified_model(model_path):
    """Load unified YOLO model"""
    from ultralytics import YOLO
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    return model


def predict_with_unified_model(
    model,
    video_path,
    target_class_id,
    frame_skip=1,
    confidence_threshold=0.25
):
    """
    Run inference with unified model, filter by target class
    
    Args:
        model: Trained YOLO model
        video_path: Path to video
        target_class_id: Class ID to filter (0-6)
        frame_skip: Frame skip
        confidence_threshold: Detection confidence
        
    Returns:
        List of BBox detections for target class only
    """
    detections = []
    
    # Extract frames
    frame_generator = extract_frames(video_path, frame_skip=frame_skip)
    
    for frame_idx, frame in frame_generator:
        # YOLO inference
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        # Parse results - filter by class
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Only keep detections of target class
                if cls == target_class_id:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    detections.append(BBox(
                        frame=frame_idx,
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=conf
                    ))
    
    return detections


def evaluate_unified_model(
    model_path,
    dataset,
    frame_skip=1,
    confidence_threshold=0.25
):
    """Evaluate unified model on all validation videos"""
    print(f"Loading model: {model_path}")
    model = load_unified_model(model_path)
    
    all_results = []
    st_ious = []
    
    # Evaluate each object type on its validation video (video_1)
    for obj_name, class_id in sorted(OBJECT_CLASSES.items(), key=lambda x: x[1]):
        video_id = f"{obj_name}_1"  # Validation video
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {obj_name} (class {class_id})")
        print(f"{'='*60}")
        
        try:
            sample = dataset.get_sample(video_id)
        except KeyError:
            print(f"  ⚠️  Video not found: {video_id}")
            continue
        
        print(f"  Video: {sample.video_id}")
        print(f"  Ground truth: {len(sample.annotations)} frames")
        
        # Run inference
        print(f"  Running inference...")
        predictions = predict_with_unified_model(
            model=model,
            video_path=sample.video_path,
            target_class_id=class_id,
            frame_skip=frame_skip,
            confidence_threshold=confidence_threshold
        )
        
        print(f"  Predictions: {len(predictions)} detections")
        
        # Compute ST-IoU
        st_iou = compute_st_iou(
            ground_truth=sample.annotations,
            predictions=predictions
        )
        
        print(f"  ST-IoU: {st_iou:.4f}")
        
        all_results.append({
            'object': obj_name,
            'class_id': class_id,
            'video_id': video_id,
            'num_predictions': len(predictions),
            'num_ground_truth': len(sample.annotations),
            'st_iou': st_iou,
            'predictions': [pred.to_dict() for pred in predictions]
        })
        
        st_ious.append(st_iou)
    
    return all_results, st_ious


def main():
    parser = argparse.ArgumentParser(description="Evaluate unified YOLO model")
    parser.add_argument(
        '--model',
        type=str,
        default='models/trained/unified.pt',
        help='Path to unified model'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='observing/train',
        help='Path to data root'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Frame skip for inference'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/unified_results.json',
        help='Output file'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    dataset = DroneDataset(data_root=args.data_root)
    print(f"Loaded {len(dataset)} videos")
    
    # Evaluate
    all_results, st_ious = evaluate_unified_model(
        model_path=args.model,
        dataset=dataset,
        frame_skip=args.frame_skip,
        confidence_threshold=args.confidence
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY (Unified Model)")
    print("="*60)
    
    if st_ious:
        print(f"Objects evaluated: {len(st_ious)}")
        print(f"Mean ST-IoU: {np.mean(st_ious):.4f}")
        print(f"Std ST-IoU: {np.std(st_ious):.4f}")
        print(f"Min ST-IoU: {np.min(st_ious):.4f}")
        print(f"Max ST-IoU: {np.max(st_ious):.4f}")
        
        print("\nPer-object results:")
        for result in all_results:
            print(f"  {result['object']:15s} (class {result['class_id']}): ST-IoU = {result['st_iou']:.4f}")
    
    print(f"\nResults saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()

