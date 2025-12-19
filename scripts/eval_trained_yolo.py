"""
Evaluate trained YOLO models on validation set and compare with baseline

Usage:
    python scripts/eval_trained_yolo.py --object Backpack
    python scripts/eval_trained_yolo.py --object all
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DroneDataset, BBox
from src.evaluation.st_iou import compute_st_iou
from src.utils.video_utils import extract_frames


def load_trained_model(model_path, device='cuda'):
    """Load trained YOLO model"""
    from ultralytics import YOLO
    import torch
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    return model


def predict_with_trained_yolo(
    model,
    video_path,
    frame_skip=1,
    confidence_threshold=0.25
):
    """
    Run inference with trained YOLO model
    
    Returns:
        List of BBox detections
    """
    detections = []
    
    # Extract frames
    frame_generator = extract_frames(video_path, frame_skip=frame_skip)
    
    for frame_idx, frame in frame_generator:
        # YOLO inference
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        # Parse results
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
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


def evaluate_object(
    object_name,
    model_path,
    dataset,
    frame_skip=1,
    confidence_threshold=0.25
):
    """Evaluate trained model on validation video"""
    print(f"\nEvaluating: {object_name}")
    print(f"Model: {model_path}")
    
    # Load model
    model = load_trained_model(model_path)
    
    # Get validation video (video_1)
    video_id = f"{object_name}_1"
    try:
        sample = dataset.get_sample(video_id)
    except KeyError:
        print(f"  ⚠️  Validation video not found: {video_id}")
        return None
    
    print(f"  Video: {sample.video_id}")
    print(f"  Ground truth: {len(sample.annotations)} frames")
    
    # Run inference
    print(f"  Running inference...")
    predictions = predict_with_trained_yolo(
        model=model,
        video_path=sample.video_path,
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
    
    return {
        'object': object_name,
        'video_id': video_id,
        'num_predictions': len(predictions),
        'num_ground_truth': len(sample.annotations),
        'st_iou': st_iou,
        'predictions': [pred.to_dict() for pred in predictions]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO models")
    parser.add_argument(
        '--object',
        type=str,
        required=True,
        help='Object to evaluate (e.g., Backpack) or "all"'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models/trained',
        help='Directory containing trained models'
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
        help='Confidence threshold for detection'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/trained_yolo_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    dataset = DroneDataset(data_root=args.data_root)
    print(f"Loaded {len(dataset)} videos")
    
    models_dir = Path(args.models_dir)
    
    # Find objects to evaluate
    if args.object.lower() == 'all':
        # Find all trained models
        model_files = list(models_dir.glob('*.pt'))
        objects = [f.stem for f in model_files]
        print(f"\nFound {len(objects)} trained models: {objects}")
    else:
        objects = [args.object]
    
    # Evaluate each object
    all_results = []
    st_ious = []
    
    for obj_name in tqdm(objects, desc="Evaluating"):
        model_path = models_dir / f"{obj_name}.pt"
        
        if not model_path.exists():
            print(f"\n⚠️  Model not found: {model_path}, skipping...")
            continue
        
        try:
            result = evaluate_object(
                object_name=obj_name,
                model_path=model_path,
                dataset=dataset,
                frame_skip=args.frame_skip,
                confidence_threshold=args.confidence
            )
            
            if result:
                all_results.append(result)
                st_ious.append(result['st_iou'])
        
        except Exception as e:
            print(f"\n❌ Error evaluating {obj_name}: {e}")
            continue
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if st_ious:
        import numpy as np
        print(f"Objects evaluated: {len(st_ious)}")
        print(f"Mean ST-IoU: {np.mean(st_ious):.4f}")
        print(f"Std ST-IoU: {np.std(st_ious):.4f}")
        print(f"Min ST-IoU: {np.min(st_ious):.4f}")
        print(f"Max ST-IoU: {np.max(st_ious):.4f}")
        
        print("\nPer-object results:")
        for result in all_results:
            print(f"  {result['object']:15s}: ST-IoU = {result['st_iou']:.4f}")
    else:
        print("No results to report")
    
    print(f"\nResults saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()

