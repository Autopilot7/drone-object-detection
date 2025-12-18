# 📤 Hướng Dẫn Upload Project Lên GitHub

## 🎯 Chuẩn Bị

### 1. Tạo Repository Trên GitHub

1. Truy cập https://github.com
2. Đăng nhập
3. Click nút **"+"** góc trên phải → **"New repository"**
4. Điền thông tin:
   - **Repository name**: `drone-object-detection`
   - **Description**: `Multi-approach drone object detection with Traditional CV, Deep Learning, and Hybrid methods`
   - **Public** hoặc **Private** (tùy bạn)
   - **KHÔNG** chọn "Add a README" (đã có rồi)
5. Click **"Create repository"**

### 2. Cài Git Trên Windows (Nếu Chưa Có)

```powershell
# Kiểm tra đã có git chưa
git --version

# Nếu chưa có, tải tại: https://git-scm.com/download/win
```

## 📤 Upload Code Lên GitHub

### Bước 1: Khởi Tạo Git Repository

```powershell
# Mở PowerShell tại thư mục project
cd C:\Users\Surface1\Documents\CV

# Khởi tạo git
git init

# Thêm tất cả files (trừ những file trong .gitignore)
git add .

# Commit
git commit -m "Initial commit: Complete implementation of 3 approaches"
```

### Bước 2: Kết Nối Với GitHub

```powershell
# Thay YOUR_USERNAME bằng username GitHub của bạn
git remote add origin https://github.com/YOUR_USERNAME/drone-object-detection.git

# Kiểm tra
git remote -v
```

### Bước 3: Push Code Lên GitHub

```powershell
# Push lên GitHub
git push -u origin main

# Nếu lỗi, thử:
git push -u origin master
```

**Lưu ý**: Lần đầu push, GitHub sẽ yêu cầu đăng nhập:
- Username: username GitHub của bạn
- Password: **Personal Access Token** (KHÔNG phải password thông thường)

#### Tạo Personal Access Token:

1. GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. **Generate new token** → **Generate new token (classic)**
3. Chọn: `repo` (full control)
4. **Generate token**
5. **COPY token** (chỉ hiện 1 lần!)
6. Dùng token này làm password khi push

## ✅ Kiểm Tra

Truy cập: `https://github.com/YOUR_USERNAME/drone-object-detection`

Bạn sẽ thấy:
- ✅ Tất cả code files
- ✅ README.md
- ✅ requirements.txt
- ❌ KHÔNG có folder `observing/` (quá lớn)
- ❌ KHÔNG có folder `venv/`
- ❌ KHÔNG có folder `output/`

## 📊 Files Đã Upload

```
✅ src/ (tất cả code)
✅ notebooks/
✅ requirements.txt
✅ README.md
✅ USAGE.md
✅ SERVER_SETUP.md
✅ run_all_approaches.py
✅ test_single_video.py
✅ .gitignore

❌ observing/ (data - sẽ upload riêng)
❌ venv/ (bỏ qua)
❌ output/ (bỏ qua)
```

## 🔄 Cập Nhật Code Sau Này

```powershell
# Sau khi sửa code
git add .
git commit -m "Update: thêm logging chi tiết"
git push
```

---

# 📥 Cách Xử Lý Data Videos

## Option 1: Google Drive (Dễ Nhất)

### Trên Windows:
```powershell
# Compress data
Compress-Archive -Path observing -DestinationPath observing.zip

# Upload observing.zip lên Google Drive
# Lấy link share
```

### Trên Server:
```bash
# Install gdown
pip install gdown

# Download (thay FILE_ID bằng ID từ link Google Drive)
gdown https://drive.google.com/uc?id=FILE_ID

# Extract
unzip observing.zip
```

## Option 2: SCP (Nếu Có SSH Access)

```powershell
# Trên Windows (trong PowerShell hoặc Git Bash)
scp -r observing username@server_ip:/path/to/CV/
```

## Option 3: Cloud Storage (S3, Dropbox, etc.)

Upload lên cloud storage và download trên server.

---

# 🖥️ Setup Trên GPU Server

## Bước 1: Clone Repository

```bash
# SSH vào server
ssh username@server_ip

# Clone project
git clone https://github.com/YOUR_USERNAME/drone-object-detection.git
cd drone-object-detection
```

## Bước 2: Download Data

```bash
# Chọn một trong các option ở trên
# Ví dụ với Google Drive:
pip install gdown
gdown https://drive.google.com/uc?id=YOUR_FILE_ID
unzip observing.zip
```

## Bước 3: Setup Environment

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Bước 4: Verify GPU

```bash
# Kiểm tra GPU
nvidia-smi

# Kiểm tra PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Bước 5: Chạy

```bash
# Dùng tmux để chạy background
tmux new -s cv

# Chạy
python run_all_approaches.py --frame-skip 2

# Detach: Ctrl+B rồi nhấn D
# Reattach: tmux attach -t cv
```

---

# 📝 Checklist

### Trước Khi Upload:
- [x] ✅ Đã tạo .gitignore
- [x] ✅ Đã tạo GitHub repository
- [ ] ⬜ Đã có Personal Access Token
- [ ] ⬜ Đã test git push

### Sau Khi Upload:
- [ ] ⬜ Kiểm tra files trên GitHub
- [ ] ⬜ Upload data lên cloud storage
- [ ] ⬜ Clone trên server
- [ ] ⬜ Download data trên server
- [ ] ⬜ Setup environment trên server
- [ ] ⬜ Test chạy trên server

---

# 🆘 Troubleshooting

### "Permission denied"
```bash
# Tạo SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add key vào GitHub Settings → SSH keys
```

### "Large files"
```bash
# Nếu vô tình add file lớn
git rm --cached observing -r
git commit -m "Remove large files"
git push
```

### "Authentication failed"
- Dùng Personal Access Token thay vì password
- Hoặc setup SSH key

