"""
Fine-tune YOLOv8 on prepared dataset

Usage:
    python scripts/train_yolo.py --object Backpack --epochs 50
    python scripts/train_yolo.py --object all --epochs 50  # Train all objects
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def train_yolo_for_object(
    object_name,
    data_yaml_path,
    base_model='yolov8n.pt',
    epochs=50,
    imgsz=640,
    batch_size=16,
    patience=20,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project='runs/train',
    name=None
):
    """
    Fine-tune YOLO on specific object
    
    Args:
        object_name: Name of object (e.g., "Backpack")
        data_yaml_path: Path to data.yaml
        base_model: Base YOLO model to fine-tune
        epochs: Number of training epochs
        imgsz: Image size
        batch_size: Batch size
        patience: Early stopping patience (epochs without improvement)
        device: cuda or cpu
        project: Project directory for saving results
        name: Run name (default: object_name)
    """
    if name is None:
        name = object_name
    
    print(f"\n{'='*60}")
    print(f"Training YOLO for: {object_name}")
    print(f"{'='*60}")
    print(f"Data config: {data_yaml_path}")
    print(f"Base model: {base_model}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load pre-trained model
    model = YOLO(base_model)
    
    # Train
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        # Optimization settings
        patience=patience,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every N epochs
        # Augmentation (help with small dataset)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    print(f"\n✓ Training complete for {object_name}")
    print(f"Best model saved to: {results.save_dir / 'weights' / 'best.pt'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on prepared dataset")
    parser.add_argument(
        '--object',
        type=str,
        required=True,
        help='Object to train (e.g., Backpack) or "all" for all objects'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='yolo_datasets',
        help='Root directory of prepared YOLO datasets'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='Base YOLO model (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20 epochs)'
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    if args.object.lower() == 'all':
        # Train all objects
        object_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
        objects = [d.name for d in object_dirs if (d / 'data.yaml').exists()]
        print(f"\nFound {len(objects)} objects to train: {objects}")
    else:
        objects = [args.object]
    
    # Train each object
    for obj_name in objects:
        data_yaml = dataset_root / obj_name / 'data.yaml'
        
        if not data_yaml.exists():
            print(f"\n⚠️  Warning: data.yaml not found for {obj_name}, skipping...")
            continue
        
        try:
            train_yolo_for_object(
                object_name=obj_name,
                data_yaml_path=data_yaml,
                base_model=args.base_model,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch_size=args.batch_size,
                patience=args.patience,
                device=args.device,
                project='runs/train',
                name=obj_name
            )
        except Exception as e:
            print(f"\n❌ Error training {obj_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nTrained models saved to: runs/train/<object_name>/weights/best.pt")
    print("\nTo use trained models:")
    print("  1. Copy best.pt to models/trained/<object>.pt")
    print("  2. Run evaluation: python scripts/eval_trained_yolo.py")


if __name__ == '__main__':
    main()

