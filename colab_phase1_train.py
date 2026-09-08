"""
Google Colab Training Script for Phase 1 Real Data
Paste this into a Colab cell
"""

# ============ CELL 1: Mount Drive & Install ============
from google.colab import drive
drive.mount('/content/drive')

# Install deps
!pip install -q torch torchvision torchaudio rasterio albumentations pyyaml scikit-image opencv-python-headless matplotlib tqdm

# ============ CELL 2: Clone & Setup ============
import os
os.chdir('/content/drive/MyDrive')

# Clone repo (or upload manually)
!git clone https://github.com/YOUR_REPO/liss4_cloud_removal_final.git 2>/dev/null || echo "Repo exists"
os.chdir('/content/drive/MyDrive/liss4_cloud_removal_final')

!ls -la

# ============ CELL 3: Check GPU ============
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============ CELL 4: Verify Data ============
from pathlib import Path
data_path = Path("data/processed_real")
clear_count = len(list((data_path / "clear").glob("*.npy")))
cloudy_count = len(list((data_path / "cloudy").glob("*.npy")))

print(f"\n{'='*60}")
print(f"Dataset Status")
print(f"{'='*60}")
print(f"Clear patches: {clear_count}")
print(f"Cloudy patches: {cloudy_count}")
print(f"Masks: {len(list((data_path / 'masks').glob('*.npy')))}")

if clear_count == 0:
    print("\n[WARNING] No processed data found!")
    print("Need to run data loader first or upload data")
else:
    print(f"[OK] Data ready!")

# ============ CELL 5: Train ============
import yaml
from src.training.trainer import Trainer

# Load config
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# Override for Colab
cfg["training"]["epochs"] = 20  # Change as needed
cfg["training"]["batch_size"] = 16
cfg["training"]["learning_rate"] = 1e-4

print(f"\n{'='*60}")
print(f"Phase 1 Real Data Training")
print(f"{'='*60}")
print(f"Epochs: {cfg['training']['epochs']}")
print(f"Batch size: {cfg['training']['batch_size']}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"{'='*60}\n")

# Train
trainer = Trainer(cfg)
history = trainer.train("data/processed_real")

# ============ CELL 6: Save Results & Download ============
import shutil
import json

# Create output zip
!zip -r /tmp/phase1_results.zip models/best_model.pth results/metrics/training_history.csv

# Download
from google.colab import files
files.download('/tmp/phase1_results.zip')
print("\n[OK] Downloaded: phase1_results.zip")

# Print summary
print(f"\n{'='*60}")
print(f"Training Summary")
print(f"{'='*60}")
print(f"Best Val Loss: {trainer.best_val_loss:.4f}")
print(f"Final PSNR: {history['val_psnr'][-1]:.2f} dB")
print(f"Final SSIM: {history['val_ssim'][-1]:.4f}")
print(f"Final MAE: {history['val_mae'][-1]:.4f}")
print(f"\nModel: models/best_model.pth (89 MB)")
print(f"Metrics: results/metrics/training_history.csv")
print(f"{'='*60}\n")

# ============ CELL 7: Plot Results ============
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/metrics/training_history.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(df['epoch'], df['train_loss'], label='Train', marker='o')
axes[0].plot(df['epoch'], df['val_loss'], label='Val', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(df['epoch'], df['val_psnr'], marker='o', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title('Validation PSNR')
axes[1].grid(alpha=0.3)

axes[2].plot(df['epoch'], df['val_ssim'], marker='o', color='orange')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('SSIM')
axes[2].set_title('Validation SSIM')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/training_curves.png', dpi=100, bbox_inches='tight')
plt.show()

# Download plot
files.download('/tmp/training_curves.png')
print("[OK] Downloaded: training_curves.png")
