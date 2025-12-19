# 🎯 YOLO Training Guide - Quick Start

## 📋 **Overview**

Train YOLOv8 to directly detect target objects (no similarity matching needed!).

**Approach:**
- **Train set:** Video_0 (Backpack_0, Jacket_0, ...)
- **Val set:** Video_1 (Backpack_1, Jacket_1, ...)
- **Method:** Fine-tune YOLOv8 with ground-truth annotations

**Expected Improvement:**
- Baseline (YOLO+CLIP): ST-IoU ~5%
- Trained YOLO: ST-IoU ~30-50%

---

## 🚀 **Quick Start (1 Object for Testing)**

### **On Server (with GPU):**

```bash
# 1. Prepare dataset for WaterBottle
python scripts/prepare_yolo_dataset.py --object WaterBottle

# 2. Train YOLO (~2-3 hours on GPU)
python scripts/train_yolo.py \
    --object WaterBottle \
    --epochs 50 \
    --batch-size 16

# 3. Copy trained model
mkdir -p models/trained
cp runs/train/WaterBottle/weights/best.pt models/trained/WaterBottle.pt

# 4. Evaluate
python scripts/eval_trained_yolo.py --object WaterBottle
```

**Expected Output:**
```
Processing: WaterBottle
Training... [Progress bar]
✓ Training complete
Best model: runs/train/WaterBottle/weights/best.pt

Evaluating: WaterBottle
ST-IoU: 0.35-0.45  ← Much better than 0.05!
```

---

## 🔥 **Full Pipeline (All Objects)**

### **Step 1: Prepare All Datasets (~5-10 min)**

```bash
python scripts/prepare_yolo_dataset.py
```

**Output:**
```
yolo_datasets/
├── Backpack/data.yaml
├── Jacket/data.yaml
├── Laptop/data.yaml
├── Lifering/data.yaml
├── MobilePhone/data.yaml
├── Person1/data.yaml
└── WaterBottle/data.yaml
```

### **Step 2: Train All Objects (~15-20 hours on GPU)**

**Option A: Sequential (safer)**
```bash
python scripts/train_yolo.py --object all --epochs 50
```

**Option B: Parallel (faster, if multiple GPUs)**
```bash
# Terminal 1
python scripts/train_yolo.py --object Backpack --epochs 50 --device cuda:0

# Terminal 2
python scripts/train_yolo.py --object Jacket --epochs 50 --device cuda:1

# ... etc
```

**Monitor training:**
```bash
# Watch training progress
watch -n 60 "ls -lh runs/train/*/weights/best.pt"

# Check results
cat runs/train/WaterBottle/results.csv | tail -n 5
```

### **Step 3: Copy Trained Models**

**Linux/Mac:**
```bash
bash scripts/copy_trained_models.sh
```

**Windows:**
```bash
scripts\copy_trained_models.bat
```

**Manual:**
```bash
mkdir -p models/trained
cp runs/train/Backpack/weights/best.pt models/trained/Backpack.pt
cp runs/train/Jacket/weights/best.pt models/trained/Jacket.pt
cp runs/train/Laptop/weights/best.pt models/trained/Laptop.pt
cp runs/train/Lifering/weights/best.pt models/trained/Lifering.pt
cp runs/train/MobilePhone/weights/best.pt models/trained/MobilePhone.pt
cp runs/train/Person1/weights/best.pt models/trained/Person1.pt
cp runs/train/WaterBottle/weights/best.pt models/trained/WaterBottle.pt
```

### **Step 4: Evaluate All**

```bash
python scripts/eval_trained_yolo.py --object all --frame-skip 2
```

---

## 📊 **Comparing Results**

### **Baseline (YOLO + CLIP - Zero-Shot)**
```bash
python run_all_approaches.py --approaches deep_learning --encoder clip --frame-skip 2
```

### **Trained YOLO**
```bash
python scripts/eval_trained_yolo.py --object all --frame-skip 2
```

### **Compare:**
```python
import json
import pandas as pd

# Load results
with open('output/predictions_deep_learning.json') as f:
    baseline = json.load(f)

with open('output/trained_yolo_results.json') as f:
    trained = json.load(f)

# Compare
# (Add comparison script here if needed)
```

---

## 🎯 **Expected Results**

| Object | Baseline (CLIP) | Trained YOLO | Improvement |
|--------|----------------|--------------|-------------|
| WaterBottle | 0.05 | 0.35-0.45 | 7-9x better |
| Backpack | 0.03 | 0.30-0.40 | 10-13x better |
| Jacket | 0.04 | 0.25-0.35 | 6-9x better |
| ... | ... | ... | ... |
| **Mean** | **0.05** | **0.30-0.50** | **6-10x better** |

---

## ⚡ **Tips for Faster Training**

### **1. Use Smaller Model (yolov8n)**
```bash
python scripts/train_yolo.py --object all --base-model yolov8n.pt
# Fastest, still good results
```

### **2. Reduce Epochs**
```bash
python scripts/train_yolo.py --object all --epochs 30
# With early stopping, often enough
```

### **3. Increase Batch Size (if GPU memory allows)**
```bash
python scripts/train_yolo.py --object all --batch-size 32
# Faster training, but needs more GPU RAM
```

### **4. Train Overnight**
```bash
# Run in tmux/screen so it continues after disconnect
tmux new -s yolo_training
python scripts/train_yolo.py --object all --epochs 50
# Ctrl+B, D to detach
# tmux attach -t yolo_training to reattach
```

---

## 🐛 **Troubleshooting**

### **Issue: CUDA Out of Memory**
```bash
# Solution: Reduce batch size
python scripts/train_yolo.py --object Backpack --batch-size 8
```

### **Issue: Training Stuck/Slow**
```bash
# Check GPU usage
nvidia-smi

# Should see:
# - GPU Util: 80-100%
# - Memory: 50-80% used
```

### **Issue: Low Validation mAP**
This is normal! The validation set is very different from train (different video).
- Focus on ST-IoU, not mAP
- mAP@50 around 0.3-0.5 is OK

### **Issue: Model Overfitting**
If validation loss increases while train loss decreases:
- Already handled by early stopping (patience=20)
- Try more augmentation (already enabled)
- Reduce epochs

---

## 📝 **Important Notes**

1. **Each object = separate model**
   - Backpack.pt only detects backpacks
   - Jacket.pt only detects jackets
   - etc.

2. **No similarity matching needed**
   - Trained YOLO directly detects target object
   - Much simpler and faster than CLIP approach

3. **Small dataset is OK**
   - YOLO pre-trained on COCO
   - We just fine-tune on specific objects
   - ~1000-3000 frames per object is enough

4. **Training time varies**
   - Depends on: GPU, model size, epochs, dataset size
   - WaterBottle (shortest video): ~1.5 hours
   - Backpack (longest video): ~3.5 hours

---

## ✅ **Next Steps After Training**

1. **Analyze Results**
   ```bash
   # View training curves
   ls runs/train/WaterBottle/results.png
   
   # Check validation predictions
   ls runs/train/WaterBottle/val_batch*.jpg
   ```

2. **Test Inference**
   ```bash
   # Quick test on single video
   from ultralytics import YOLO
   model = YOLO('models/trained/WaterBottle.pt')
   results = model('observing/train/samples/WaterBottle_1/drone_video.mp4')
   ```

3. **Compare with Baseline**
   - If Trained YOLO >> Baseline: Great! Use trained models
   - If similar: Baseline zero-shot is already good enough
   - If worse: Check training curves, may need more epochs

4. **Optimize Thresholds**
   ```bash
   # Try different confidence thresholds
   python scripts/eval_trained_yolo.py --object all --confidence 0.15
   python scripts/eval_trained_yolo.py --object all --confidence 0.25
   python scripts/eval_trained_yolo.py --object all --confidence 0.35
   ```

---

## 🎓 **Understanding Training Output**

```
Epoch   Box Loss   Cls Loss   DFL Loss   Precision   Recall   mAP@50
────────────────────────────────────────────────────────────────────
1/50    1.234      0.567      0.890      0.123       0.234    0.100
10/50   0.789      0.234      0.456      0.456       0.567    0.234
30/50   0.456      0.123      0.234      0.678       0.789    0.456
50/50   0.234      0.067      0.123      0.789       0.890    0.567
```

**What to look for:**
- **Losses decrease**: Good! Model is learning
- **Precision/Recall increase**: Good! Better detection
- **mAP@50 increases**: Good! Better overall performance
- **Validation < Train**: Normal (slight overfitting is OK)

---

## 📞 **Questions?**

See detailed documentation in `scripts/README.md`

