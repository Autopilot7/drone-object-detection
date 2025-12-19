"""
Train UNIFIED multi-class YOLO model (optimized for RTX 3060)

All 7 objects in 1 model!

RTX 3060 Optimizations:
- Batch size: 32-48 (12GB VRAM)
- Mixed precision (FP16): 2x faster
- Model: yolov8s (good balance)
- Cache: images in RAM for faster loading
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def train_unified_yolo(
    data_yaml_path,
    base_model='yolov8s.pt',
    epochs=50,
    imgsz=640,
    batch_size=32,
    patience=10,
    device='cuda',
    cache=True,
    amp=True  # Mixed precision
):
    """
    Train unified YOLO model for all objects
    
    Optimized for RTX 3060:
    - batch_size=32-48 (12GB VRAM)
    - amp=True (FP16 mixed precision)
    - cache=True (RAM caching for speed)
    """
    print(f"\n{'='*60}")
    print(f"TRAINING UNIFIED YOLO MODEL")
    print(f"{'='*60}")
    print(f"Data config: {data_yaml_path}")
    print(f"Base model: {base_model}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Device: {device}")
    print(f"Mixed precision (FP16): {amp}")
    print(f"Cache images: {cache}")
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
        project='runs/train_unified',
        name='drone_detector',
        exist_ok=True,
        
        # Speed optimizations
        cache=cache,  # Cache images in RAM (32GB is plenty!)
        amp=amp,      # Mixed precision (FP16) - 2x faster on RTX
        workers=8,    # Data loading workers
        
        # Optimization settings
        patience=patience,
        save=True,
        save_period=10,
        
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
        
        # Training hyperparameters
        lr0=0.01,     # Initial learning rate
        lrf=0.01,     # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )
    
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    print(f"\n✓ Training complete!")
    print(f"Best model: {best_model_path}")
    print(f"Results: {results.save_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train unified multi-class YOLO")
    parser.add_argument(
        '--data',
        type=str,
        default='yolo_unified/data.yaml',
        help='Path to data.yaml'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='yolov8s.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
        help='Base YOLO model (s=recommended for RTX 3060)'
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
        default=32,
        help='Batch size (32-48 for RTX 3060 12GB)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable image caching (use if low RAM)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable mixed precision (slower but more stable)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Check data.yaml exists
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"❌ Error: {data_yaml} not found!")
        print(f"\nRun first: python scripts/prepare_unified_dataset.py")
        return
    
    # Train
    train_unified_yolo(
        data_yaml_path=data_yaml,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        cache=not args.no_cache,
        amp=not args.no_amp
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Copy model:")
    print("   mkdir -p models/trained")
    print("   cp runs/train_unified/drone_detector/weights/best.pt models/trained/unified.pt")
    print("")
    print("2. Evaluate:")
    print("   python scripts/eval_unified.py")
    print("="*60)


if __name__ == '__main__':
    main()

