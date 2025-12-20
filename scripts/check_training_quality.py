"""
Verify training data: check if images match labels
"""
import cv2
from pathlib import Path
import sys

def check_training_data(dataset_root='yolo_unified', num_samples=5):
    """Verify training images và labels match"""
    dataset_root = Path(dataset_root)
    
    # Check train set
    img_dir = dataset_root / 'images' / 'train'
    label_dir = dataset_root / 'labels' / 'train'
    
    img_files = sorted(list(img_dir.glob('*.jpg')))[:num_samples]
    
    print("="*60)
    print("CHECKING TRAINING DATA QUALITY")
    print("="*60)
    
    for img_file in img_files:
        label_file = label_dir / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            print(f"\n❌ No label for {img_file.name}")
            continue
        
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"\n❌ Cannot read {img_file.name}")
            continue
        
        h, w = img.shape[:2]
        
        print(f"\n{img_file.name}:")
        print(f"  Image size: {w}×{h}")
        
        # Read label
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        print(f"  Annotations: {len(lines)}")
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            cls, x_center, y_center, width, height = map(float, parts)
            
            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            box_w = x2 - x1
            box_h = y2 - y1
            
            print(f"    Class {int(cls)}: [{x1}, {y1}, {x2}, {y2}] = {box_w}×{box_h} px")
            
            # Check if box is reasonable
            if box_w < 5 or box_h < 5:
                print(f"      ⚠️  Box too small! Possibly wrong scale")
            elif box_w > w * 0.8 or box_h > h * 0.8:
                print(f"      ⚠️  Box too large! Possibly wrong scale")
            elif x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                print(f"      ⚠️  Box out of bounds!")
            else:
                print(f"      ✅ Box looks reasonable")

if __name__ == '__main__':
    check_training_data(num_samples=10)

