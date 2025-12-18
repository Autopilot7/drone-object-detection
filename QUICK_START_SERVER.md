# ⚡ Hướng Dẫn Nhanh: Upload GitHub & Chạy Trên GPU Server

## 📤 Phần 1: Upload Lên GitHub (Trên Windows)

### Bước 1: Khởi Tạo Git
```powershell
cd C:\Users\Surface1\Documents\CV
git init
git add .
git commit -m "Initial commit"
```

### Bước 2: Tạo Repo Trên GitHub
1. Vào https://github.com → New repository
2. Tên: `drone-object-detection`
3. Public hoặc Private
4. Create repository

### Bước 3: Push Code
```powershell
git remote add origin https://github.com/YOUR_USERNAME/drone-object-detection.git
git push -u origin main
```

### Bước 4: Upload Data Lên Google Drive
```powershell
# Nén data
Compress-Archive -Path observing -DestinationPath observing.zip

# Upload observing.zip lên Google Drive
# Click chuột phải → Get link → Copy link
# Lấy FILE_ID từ link (phần giữa /d/ và /view)
```

---

## 🖥️ Phần 2: Setup Trên GPU Server

### Bước 1: Clone Code
```bash
git clone https://github.com/YOUR_USERNAME/drone-object-detection.git
cd drone-object-detection
```

### Bước 2: Download Data
```bash
# Install gdown
pip install gdown

# Download từ Google Drive (thay YOUR_FILE_ID)
gdown https://drive.google.com/uc?id=YOUR_FILE_ID

# Giải nén
unzip observing.zip
```

### Bước 3: Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Verify GPU
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Bước 5: Chạy (Với Tmux)
```bash
# Cài tmux
sudo apt install tmux -y

# Tạo session
tmux new -s cv

# Chạy script
python run_all_approaches.py --frame-skip 2 2>&1 | tee run.log

# Detach: Ctrl+B rồi nhấn D
# Reattach sau: tmux attach -t cv
```

---

## 📋 Commands Tóm Tắt

### Trên Windows (Upload):
```powershell
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/drone-object-detection.git
git push -u origin main
```

### Trên Server (Setup):
```bash
git clone https://github.com/YOUR_USERNAME/drone-object-detection.git
cd drone-object-detection
pip install gdown
gdown https://drive.google.com/uc?id=YOUR_FILE_ID
unzip observing.zip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
tmux new -s cv
python run_all_approaches.py --frame-skip 2 2>&1 | tee run.log
```

---

## 🎯 Lưu Ý Quan Trọng

1. **KHÔNG** push folder `observing/` lên GitHub (quá lớn)
2. Upload data riêng qua Google Drive/Dropbox
3. Dùng **tmux** để chạy background (tránh mất kết nối)
4. Redirect output sang file log: `2>&1 | tee run.log`
5. Monitor GPU: `watch -n 1 nvidia-smi`

---

## 📊 Thời Gian Chạy Trên GPU

| Approach | Frame-skip 2 | Frame-skip 3 |
|----------|--------------|--------------|
| Traditional CV | ~5-10 phút | ~3-5 phút |
| Deep Learning | ~30-40 phút | ~20-25 phút |
| Hybrid | ~20-30 phút | ~15-20 phút |
| **Tổng** | **~1-1.5 giờ** | **~40-50 phút** |

---

## 🔄 Workflow Hoàn Chỉnh

```
[Windows]                [GitHub]              [GPU Server]
   |                        |                       |
   | git push              |                       |
   |---------------------> |                       |
   |                       |                       |
   | upload data           |                       |
   | to Google Drive       |                       |
   |                       |                       |
   |                       | git clone             |
   |                       |--------------------->|
   |                       |                       |
   | share Drive link      |                       |
   |-------------------------------------->| download data
   |                       |                       |
   |                       |              setup & run
   |                       |                       |
   |                       |                  get results
   |<----------------------------------------------|
```

---

## ✅ Checklist

- [ ] Upload code lên GitHub
- [ ] Upload data lên Google Drive
- [ ] Clone code trên server
- [ ] Download data trên server
- [ ] Setup Python environment
- [ ] Verify GPU hoạt động
- [ ] Chạy với tmux
- [ ] Monitor progress
- [ ] Download results về

