# GPU Server Setup Guide

## 📋 Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPU with CUDA support
- Python 3.8+
- Git installed

## 🚀 Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/CV.git
cd CV
```

### 2. Download Dataset

**Option A: Download from original source**
```bash
# Download your dataset to observing/train/
# Structure should be:
# observing/train/samples/
# observing/train/annotations/
```

**Option B: Transfer from your computer**
```bash
# On your computer (Windows):
# Compress the data
tar -czf observing.tar.gz observing/

# Upload to server using scp:
scp observing.tar.gz username@server_ip:/path/to/CV/

# On server: Extract
tar -xzf observing.tar.gz
```

**Option C: Use cloud storage**
```bash
# Upload to Google Drive, Dropbox, or AWS S3
# Then download on server:
wget "YOUR_DOWNLOAD_LINK" -O observing.tar.gz
tar -xzf observing.tar.gz
```

### 3. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 4. Verify GPU

```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
```

### 5. Run Processing

```bash
# Test with one video first
python test_single_video.py

# Run all approaches
python run_all_approaches.py --frame-skip 3

# Run specific approach
python run_all_approaches.py --approaches deep_learning --frame-skip 2
```

## ⚡ Performance Tips

### Use tmux/screen (recommended)

Processing takes hours, use tmux to keep session alive:

```bash
# Install tmux
sudo apt install tmux

# Start tmux session
tmux new -s cv_processing

# Run your command
python run_all_approaches.py --frame-skip 2

# Detach: Ctrl+B then D
# Reattach: tmux attach -t cv_processing
```

### Monitor GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Redirect Output to File

```bash
# Save all output to log file
python run_all_approaches.py --frame-skip 2 2>&1 | tee run.log
```

## 🐛 Troubleshooting

### CUDA Out of Memory

Edit `src/config.py`:
```python
YOLO_MODEL = "yolov8m.pt"  # Use smaller model
BATCH_SIZE = 4  # Reduce batch size
```

### Missing Libraries

```bash
# Install system dependencies
sudo apt install -y python3-opencv
sudo apt install -y libgl1-mesa-glx
```

### PyTorch CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Install correct PyTorch version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📊 Expected Performance

With GPU (e.g., RTX 3090, A100):
- Traditional CV: 5-10 minutes
- Deep Learning: 20-40 minutes (frame-skip 2)
- Hybrid: 15-30 minutes

## 📥 Download Results

```bash
# On server: compress results
tar -czf results.tar.gz output/

# Download to your computer:
scp username@server_ip:/path/to/CV/results.tar.gz .
```

## 🔄 Update Code

```bash
# On server
cd CV
git pull origin main
pip install -r requirements.txt  # If requirements changed
```

