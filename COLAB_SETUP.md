# Phase 1 Training on Google Colab

## Quick Start (5 min setup)

### Option A: Upload Data to Drive (Easiest)

**Step 1: Prepare Data Locally**
```bash
# On your machine
zip -r phase1_data.zip data/processed_real/
```

**Step 2: Upload to Google Drive**
- Go to https://drive.google.com
- Create folder: `liss4_colab`
- Upload `phase1_data.zip`
- Right-click → Extract

**Step 3: Open Colab Notebook**
- Go to https://colab.research.google.com
- New notebook
- Cell 1:
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/liss4_colab')
!ls -la
```

**Step 4: Clone Repository**
- Cell 2:
```python
!git clone https://github.com/YOUR_USERNAME/liss4_cloud_removal_final.git
os.chdir('liss4_cloud_removal_final')
```

**Step 5: Install Dependencies**
- Cell 3:
```python
!pip install -q torch rasterio albumentations pyyaml scikit-image opencv-python-headless tqdm
```

**Step 6: Copy Data**
- Cell 4:
```python
import shutil
shutil.copytree('/content/drive/MyDrive/liss4_colab/processed_real', 'data/processed_real', dirs_exist_ok=True)
!ls data/processed_real/clear | wc -l  # Should show 2447
```

**Step 7: Run Training**
- Cell 5:
```python
import yaml
from src.training.trainer import Trainer

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["training"]["epochs"] = 20
cfg["training"]["batch_size"] = 16

trainer = Trainer(cfg)
history = trainer.train("data/processed_real")

print(f"Best PSNR: {max(history['val_psnr']):.2f} dB")
print(f"Best SSIM: {max(history['val_ssim']):.4f}")
```

**Step 8: Download Results**
- Cell 6:
```python
!zip -r /tmp/results.zip models/best_model.pth results/metrics/training_history.csv

from google.colab import files
files.download('/tmp/results.zip')
files.download('/tmp/training_curves.png')  # If plotted
```

---

## Option B: Use Provided Colab Script

**Step 1-4: Same as above** (mount Drive, clone repo, install deps)

**Step 5: Copy & Run Colab Script**

```python
# Copy-paste entire colab_phase1_train.py content
# Or run directly:
!curl -s https://raw.githubusercontent.com/YOUR_REPO/liss4_cloud_removal_final/main/colab_phase1_train.py > colab_run.py
exec(open('colab_run.py').read())
```

---

## Option C: Use This All-in-One Colab Cell

```python
# ============ ALL-IN-ONE COLAB CELL ============

# Mount Drive
from google.colab import drive, files
drive.mount('/content/drive')

# Setup
import os
import sys
os.chdir('/content/drive/MyDrive')

# Install
!pip install -q torch rasterio albumentations pyyaml scikit-image opencv-python-headless tqdm >/dev/null 2>&1

# Clone or extract
repo_path = '/content/drive/MyDrive/liss4_cloud_removal_final'
if not os.path.exists(repo_path):
    !git clone https://github.com/YOUR_USERNAME/liss4_cloud_removal_final.git
os.chdir(repo_path)

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Check data
from pathlib import Path
data_path = Path("data/processed_real")
n_clear = len(list((data_path / "clear").glob("*.npy")))
print(f"Training pairs: {n_clear}")

if n_clear > 0:
    # Train
    import yaml
    from src.training.trainer import Trainer
    
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    cfg["training"]["epochs"] = 20
    cfg["training"]["batch_size"] = 16
    
    trainer = Trainer(cfg)
    history = trainer.train("data/processed_real")
    
    # Download
    import shutil
    shutil.make_archive('/tmp/phase1_results', 'zip', 
                       '/content/drive/MyDrive/liss4_cloud_removal_final/models',
                       '.')
    files.download('/tmp/phase1_results.zip')
    
    print(f"\n=== RESULTS ===")
    print(f"PSNR: {max(history['val_psnr']):.2f} dB")
    print(f"SSIM: {max(history['val_ssim']):.4f}")
    print(f"Model saved to: models/best_model.pth")
else:
    print("ERROR: No data found in data/processed_real/")
    print("Upload data/ folder to Drive first!")
```

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
cfg["training"]["batch_size"] = 8  # or even 4
```

### CUDA Error
```python
# Use CPU (slower but works)
cfg["device"] = "cpu"
```

### Data Not Found
```python
# Check structure
!find /content/drive/MyDrive/liss4_colab -name "*.npy" | head -5
```

### Slow Training
- Use GPU (Runtime → Change runtime type → GPU)
- Colab T4 GPU: ~10 sec/epoch for 2,447 samples

---

## Expected Output

```
Phase 1 Training: Real LISS-IV Data
============================================================
Data directory: data/processed_real
Training pairs: 2447
Epochs: 20
Batch size: 16
============================================================

Device: cuda
Model parameters: 1,968,513
Starting training for 20 epochs...

Epoch [001/020] Train Loss: 0.1234 | Val Loss: 0.1089 | PSNR: 28.45dB | SSIM: 0.8234 | Time: 12.3s
Epoch [002/020] Train Loss: 0.1015 | Val Loss: 0.0987 | PSNR: 29.12dB | SSIM: 0.8456 | Time: 12.1s
...

=== RESULTS ===
Best Val Loss: 0.0856
Final PSNR: 31.23 dB
Final SSIM: 0.8723
Final MAE: 0.0324

Model saved: models/best_model.pth
Metrics saved: results/metrics/training_history.csv
```

---

## Download & Use Model Locally

After training completes:

```bash
# Extract downloaded zip
unzip phase1_results.zip

# Copy to local project
cp best_model.pth ./models/

# Run inference
python -c "
import torch
from src.model.unet import build_model
import yaml

cfg = yaml.safe_load(open('configs/config.yaml'))
model = build_model(cfg)
ckpt = torch.load('models/best_model.pth')
model.load_state_dict(ckpt['model_state'])
print('Model loaded successfully!')
"
```

