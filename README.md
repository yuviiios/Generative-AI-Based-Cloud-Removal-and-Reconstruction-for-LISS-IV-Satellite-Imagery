# 🛰️ Generative AI-Based Cloud Removal for LISS-IV Satellite Imagery

Deep learning framework for automated cloud removal and reconstruction on LISS-IV satellite imagery from ISRO's Bhoonidhi portal.

**Status:** Prototype v1.0 — Real data pipeline operational

## 🎯 Features

- **Real LISS-IV Data:** 2,447 training patches from 20 scenes (10 cloudy + 10 clear)
- **U-Net Architecture:** Encoder-decoder with skip connections optimized for cloud removal
- **Multi-metric Evaluation:** PSNR, SSIM, MAE, SAM for quantitative assessment
- **Streamlit Dashboard:** Interactive training & inference UI
- **Cloud Detection:** Automated cloud masking using brightness + morphological ops
- **Data Augmentation:** Albumentations pipeline for robust training
- **CLI & Notebook Support:** Both CLI and Colab-compatible training scripts

## 📋 Roadmap

- ✅ **Phase 1:** Real data integration & U-Net baseline
- 🔄 **Phase 2:** Multi-sensor fusion (Sentinel-1/2), advanced cloud masking
- 📋 **Phase 3:** Architecture upgrade (Diffusion/Transformer models)
- 📋 **Phase 4:** Production deployment (Docker, REST API)


## 🗂️ Project Structure

```
liss4_cloud_removal/
├── app.py                       # Streamlit UI for training & inference
├── phase1_real_data_train.py    # CLI trainer (real data)
├── colab_phase1_train.py        # Google Colab training notebook
├── bhoonidhi_downloader.py      # LISS-IV data downloader
├── configs/
│   └── config.yaml              # Hyperparameters, paths, loss weights
├── src/
│   ├── preprocessing/
│   │   ├── real_data_loader.py      # GeoTIFF → patch extraction
│   │   ├── data_generator.py        # Synthetic data generation
│   │   ├── cloud_simulator.py       # Cloud mask simulation
│   │   ├── augmentation.py          # Albumentations pipeline
│   │   └── __init__.py
│   ├── model/
│   │   ├── unet.py              # U-Net encoder-decoder
│   │   ├── losses.py            # Perceptual + L1 + SSIM losses
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py           # Training loop & validation
│   │   ├── dataset.py           # PyTorch Dataset class
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── metrics.py           # PSNR, SSIM, MAE, SAM
│   │   ├── visualizer.py        # Plotting & comparison visuals
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── data/
│   ├── raw/clear/               # Reference GeoTIFF scenes
│   ├── raw/cloudy/              # Cloudy GeoTIFF scenes
│   ├── processed_real/          # Extracted patches
│   └── (outputs generated during training)
├── models/                      # Saved model checkpoints
├── results/                     # Metrics & visualizations
├── tests/
│   └── test_pipeline.py         # Unit tests
├── requirements.txt
├── setup.py
├── PHASE1_REAL_DATA.md          # Phase 1 detailed docs
└── COLAB_SETUP.md               # Colab setup guide
```

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- GPU recommended (CUDA)
- ~3GB disk space (includes data & models)

### Setup

```bash
git clone https://github.com/yuviiios/Generative-AI-Based-Cloud-Removal-and-Reconstruction-for-LISS-IV-Satellite-Imagery.git
cd liss4_cloud_removal_final

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Option 1: Streamlit UI (Recommended)

```bash
streamlit run app.py
```

Open http://localhost:8501. Choose:
- **Real Data tab:** Use preprocessing + training on actual LISS-IV imagery
- **Synthetic tab:** Generate synthetic clouds for quick prototyping
- Configure epochs, batch size, patch size from sidebar

### Option 2: CLI Training

**Preprocess real data (one-time):**
```bash
python -m src.preprocessing.real_data_loader
# Outputs: data/processed_real/{clear,cloudy,masks}/
```

**Train on real data:**
```bash
python phase1_real_data_train.py --epochs 30 --batch_size 16
# Saves: models/best_model.pth, results/metrics/training_history.csv
```

### Option 3: Google Colab

```bash
# Open in Colab and run:
!git clone <repo-url>
%run colab_phase1_train.py
```
See `COLAB_SETUP.md` for details.

### Inference

Use Streamlit app → **Inference** tab to:
1. Upload cloudy .tif or .jpg
2. Select model checkpoint
3. View removed clouds + metrics

Or programmatic:
```python
from src.model.unet import UNet
from src.preprocessing.augmentation import get_transforms
from PIL import Image
import torch

model = UNet(3, 3)
model.load_state_dict(torch.load("models/best_model.pth"))
# ... process image, run inference
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Baseline Results (Prototype v1.0)

Trained on 2,447 real LISS-IV patches (256×256):

| Metric | Baseline | Target |
|--------|----------|--------|
| PSNR   | 28–32 dB | 32–38  |
| SSIM   | 0.80–0.88| 0.88–0.95 |
| MAE    | 0.03–0.08| <0.03  |

---

## ⚙️ Configuration

Edit `configs/config.yaml`:
```yaml
data:
  patch_size: 256         # Input patch size
  batch_size: 16
  mode: "real"           # "real" or "synthetic"

training:
  epochs: 30
  learning_rate: 1e-4
  loss_weights:
    l1: 1.0
    ssim: 0.1
    perceptual: 0.1

model:
  channels: [64, 128, 256, 512]  # U-Net depth
```

---

## 📥 Get LISS-IV Data

1. **Bhoonidhi Portal:** https://bhoonidhi.nrsc.gov.in
   - Download clear scenes (cloud-free) & cloudy scenes
   - Format: GeoTIFF (.tif)

2. **Organize:**
```
data/raw/
├── clear/     # 10 cloud-free scenes
└── cloudy/    # 10 cloudy scenes
```

3. **Preprocess:**
```bash
python -m src.preprocessing.real_data_loader
```

Future phases: Sentinel-1/2 SAR-optical fusion, multi-temporal stacking.

---

## 📄 License

MIT — Educational & research use.




