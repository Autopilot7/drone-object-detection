"""
Check prepared YOLO dataset format
"""
from pathlib import Path
import yaml

def check_dataset(dataset_root='yolo_unified'):
    """Check dataset structure and format"""
    dataset_root = Path(dataset_root)
    
    # Check data.yaml
    yaml_path = dataset_root / 'data.yaml'
    if not yaml_path.exists():
        print(f"❌ data.yaml not found at {yaml_path}")
        return
    
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("="*60)
    print("DATASET CONFIGURATION")
    print("="*60)
    print(f"Path: {data_config.get('path')}")
    print(f"Train: {data_config.get('train')}")
    print(f"Val: {data_config.get('val')}")
    print(f"Classes: {data_config.get('nc')}")
    print(f"Names: {data_config.get('names')}")
    print()
    
    # Check images and labels
    for split in ['train', 'val']:
        img_dir = dataset_root / 'images' / split
        label_dir = dataset_root / 'labels' / split
        
        if not img_dir.exists():
            print(f"❌ {split} images not found: {img_dir}")
            continue
        
        if not label_dir.exists():
            print(f"❌ {split} labels not found: {label_dir}")
            continue
        
        img_files = list(img_dir.glob('*.jpg'))
        label_files = list(label_dir.glob('*.txt'))
        
        print(f"\n{split.upper()} SET:")
        print(f"  Images: {len(img_files)}")
        print(f"  Labels: {len(label_files)}")
        
        # Check class distribution
        class_counts = {i: 0 for i in range(data_config['nc'])}
        
        for label_file in label_files[:100]:  # Sample first 100
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        cls = int(line.split()[0])
                        class_counts[cls] += 1
        
        print(f"  Class distribution (first 100 labels):")
        for cls_name, cls_id in zip(data_config['names'], range(data_config['nc'])):
            print(f"    {cls_id}: {cls_name:15s} - {class_counts[cls_id]} annotations")
        
        # Check sample label
        if label_files:
            print(f"\n  Sample label ({label_files[0].name}):")
            with open(label_files[0], 'r') as f:
                lines = f.readlines()[:5]
                for line in lines:
                    print(f"    {line.strip()}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    check_dataset()

