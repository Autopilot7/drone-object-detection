# YOLO Training Scripts

Scripts for fine-tuning YOLOv8 on drone object detection dataset.

## 📋 **Prerequisites**

- Python 3.8+
- CUDA GPU (recommended)
- All dependencies from `requirements.txt`

## 🚀 **Quick Start**

### **Full Pipeline (All Objects)**

```bash
# Step 1: Prepare dataset (convert annotations to YOLO format)
python scripts/prepare_yolo_dataset.py

# Step 2: Train YOLO for all objects (~2-3 hours/object on GPU)
python scripts/train_yolo.py --object all --epochs 50

# Step 3: Copy trained models to models/trained/
mkdir -p models/trained
cp runs/train/*/weights/best.pt models/trained/

# Step 4: Evaluate
python scripts/eval_trained_yolo.py --object all
```

### **Single Object (Faster, for Testing)**

```bash
# Prepare dataset for WaterBottle only
python scripts/prepare_yolo_dataset.py --object WaterBottle

# Train YOLO for WaterBottle (~2-3 hours on GPU)
python scripts/train_yolo.py --object WaterBottle --epochs 50

# Copy trained model
mkdir -p models/trained
cp runs/train/WaterBottle/weights/best.pt models/trained/WaterBottle.pt

# Evaluate
python scripts/eval_trained_yolo.py --object WaterBottle
```

---

## 📖 **Detailed Usage**

### **1. prepare_yolo_dataset.py**

Converts `annotations.json` to YOLO training format.

**Options:**
```bash
--data-root PATH      # Path to observing/train/ (default: observing/train)
--output PATH         # Output directory (default: yolo_datasets)
--object NAME         # Specific object or all (default: all)
```

**Output Structure:**
```
yolo_datasets/
├── Backpack/
│   ├── data.yaml
│   ├── images/
│   │   ├── train/  # Frames from Backpack_0
│   │   └── val/    # Frames from Backpack_1
│   └── labels/
│       ├── train/  # YOLO format labels
│       └── val/
├── Jacket/
│   └── ...
└── ...
```

**Example:**
```bash
# Process all objects
python scripts/prepare_yolo_dataset.py

# Process only WaterBottle
python scripts/prepare_yolo_dataset.py --object WaterBottle
```

---

### **2. train_yolo.py**

Fine-tunes YOLOv8 on prepared dataset.

**Options:**
```bash
--object NAME           # Object to train or "all"
--dataset-root PATH     # Root of YOLO datasets (default: yolo_datasets)
--base-model MODEL      # Base YOLO model (default: yolov8n.pt)
                        # Options: yolov8n/s/m/l/x.pt (n=fastest, x=most accurate)
--epochs N              # Training epochs (default: 50)
--batch-size N          # Batch size (default: 16)
--imgsz SIZE            # Image size (default: 640)
--device cuda/cpu       # Device (auto-detected)
```

**Example:**
```bash
# Train WaterBottle with default settings
python scripts/train_yolo.py --object WaterBottle

# Train all objects with larger model
python scripts/train_yolo.py --object all --base-model yolov8s.pt --epochs 100

# Train on CPU (slow!)
python scripts/train_yolo.py --object Backpack --device cpu --epochs 30
```

**Training Time (on GPU):**
- yolov8n: ~2-3 hours/object (fastest, good enough)
- yolov8s: ~4-5 hours/object (better accuracy)
- yolov8m: ~6-8 hours/object (even better)

**Output:**
```
runs/train/
├── Backpack/
│   ├── weights/
│   │   ├── best.pt     # Best model (use this!)
│   │   └── last.pt     # Last epoch
│   ├── results.png     # Training curves
│   └── ...
└── ...
```

---

### **3. eval_trained_yolo.py**

Evaluates trained models on validation set.

**Options:**
```bash
--object NAME           # Object to evaluate or "all"
--models-dir PATH       # Directory with trained models (default: models/trained)
--data-root PATH        # Data root (default: observing/train)
--frame-skip N          # Frame skip for inference (default: 1)
--confidence FLOAT      # Detection confidence threshold (default: 0.25)
--output PATH           # Output JSON file (default: output/trained_yolo_results.json)
```

**Example:**
```bash
# Evaluate WaterBottle
python scripts/eval_trained_yolo.py --object WaterBottle

# Evaluate all objects
python scripts/eval_trained_yolo.py --object all

# With frame skip (faster)
python scripts/eval_trained_yolo.py --object all --frame-skip 2
```

**Output:**
- JSON file with predictions and ST-IoU scores
- Console summary with mean/std/min/max ST-IoU

---

## 🎯 **Expected Results**

### **Baseline (Zero-Shot YOLO + CLIP)**
```
Mean ST-IoU: ~0.05 (5%)
```

### **Fine-tuned YOLO (Expected)**
```
Mean ST-IoU: ~0.30-0.50 (30-50%)
```

**Why improvement?**
- Fine-tuned on specific objects and drone footage
- No need for similarity matching (direct detection)
- Better handling of small objects and occlusions

---

## 📝 **Notes**

### **Training Tips:**

1. **Start with 1 object** to verify pipeline works
2. **Use yolov8n.pt** for faster training (good enough!)
3. **Monitor training**: Check `runs/train/<object>/results.png`
4. **Early stopping**: Training stops automatically if no improvement for 20 epochs

### **If Training Fails:**

```bash
# Check GPU memory
nvidia-smi

# Reduce batch size
python scripts/train_yolo.py --object Backpack --batch-size 8

# Use smaller model
python scripts/train_yolo.py --object Backpack --base-model yolov8n.pt

# Reduce image size
python scripts/train_yolo.py --object Backpack --imgsz 416
```

### **After Training:**

```bash
# Copy best models to models/trained/
mkdir -p models/trained
for obj in runs/train/*/; do
    name=$(basename $obj)
    cp $obj/weights/best.pt models/trained/$name.pt
done
```

---

## 🔄 **Comparison with Baseline**

To compare trained YOLO with baseline (YOLO+CLIP):

```bash
# 1. Run baseline
python run_all_approaches.py --approaches deep_learning --encoder clip

# 2. Run trained YOLO
python scripts/eval_trained_yolo.py --object all

# 3. Compare results
# - Baseline: output/predictions_deep_learning.json
# - Trained: output/trained_yolo_results.json
```

---

## 🐛 **Troubleshooting**

**Issue: "CUDA out of memory"**
```bash
# Solution: Reduce batch size
python scripts/train_yolo.py --object Backpack --batch-size 8
```

**Issue: "Model not found"**
```bash
# Solution: Check model was copied correctly
ls -la models/trained/
# Should see: Backpack.pt, Jacket.pt, etc.
```

**Issue: "Training very slow"**
```bash
# Solution 1: Use smaller model
python scripts/train_yolo.py --object Backpack --base-model yolov8n.pt

# Solution 2: Reduce epochs
python scripts/train_yolo.py --object Backpack --epochs 30
```

---

## 📊 **Monitoring Training**

### **During Training:**
```bash
# Terminal output shows:
# - Epoch progress
# - Loss values (box, cls, dfl)
# - Metrics (precision, recall, mAP)
# - ETA

# Check training curves:
tensorboard --logdir runs/train
# Or view: runs/train/<object>/results.png
```

### **After Training:**
```bash
# View results
ls runs/train/<object>/
# - results.png: Training curves
# - confusion_matrix.png: Predictions breakdown
# - val_batch*.jpg: Validation predictions
```

---

## ✅ **Expected Timeline**

```
Task                           Time (GPU)    Time (CPU)
────────────────────────────────────────────────────────
Prepare dataset (all)          5-10 min      5-10 min
Train 1 object (50 epochs)     2-3 hours     20-30 hours
Train all objects (7)          14-21 hours   140-210 hours
Evaluation (all)               10-15 min     30-60 min
────────────────────────────────────────────────────────
Total (GPU)                    ~15-22 hours
Total (CPU)                    ~141-211 hours
```

**Recommendation:** Use GPU! CPU training is too slow.

