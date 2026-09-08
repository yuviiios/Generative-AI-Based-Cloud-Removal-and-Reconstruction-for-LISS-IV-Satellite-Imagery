# Phase 1: Real LISS-IV Data Integration

## Overview
Phase 1 transitions from synthetic to real satellite data. Uses actual LISS-IV imagery from the Bhoonidhi portal with real cloud and cloud-free pairs.

## Dataset

### Source
- **Clear images:** `data/raw/clear/` (10 GeoTIFF scenes)
- **Cloudy images:** `data/raw/cloudy/` (10 GeoTIFF scenes)

### Preprocessing Pipeline

**Step 1: Extract Patches**
```bash
python -m src.preprocessing.real_data_loader
```

This script:
1. Reads GeoTIFF files using rasterio
2. Normalizes each band to [0, 1] using 2-98 percentile clipping
3. Extracts overlapping 256×256 patches (stride 128 px)
4. Detects cloud masks using brightness + morphology
5. Filters patches with 5-85% cloud coverage
6. Saves triplets (clear, cloudy, mask) as .npy files

**Output:** 
- `data/processed_real/clear/` — 2,447 reference patches
- `data/processed_real/cloudy/` — 2,447 cloudy patches
- `data/processed_real/masks/` — 2,447 cloud masks

**Metrics:**
- Clear patches extracted: 3,771
- Valid cloudy patches (5-85% coverage): 2,447
- Patch size: 256×256 px
- Stride: 128 px (50% overlap)

## Training

### CLI Training
```bash
python phase1_real_data_train.py --epochs 30 --batch_size 16
```

Arguments:
- `--epochs`: Training epochs (default 30)
- `--batch_size`: Batch size (default 8, use 16 for faster training on GPU)
- `--config`: Config YAML path (default `configs/config.yaml`)
- `--data_dir`: Processed data directory (default `data/processed_real`)

### Streamlit Dashboard
```bash
streamlit run app.py
```

1. Select "Real Data" from sidebar radio button
2. Set hyperparameters (epochs, batch size, patch size)
3. Click "Train Model" tab
4. Click "🚀 Run Full Training Pipeline"
5. Monitor loss curves and metrics in real-time

### Training Details

**Model:** U-Net (encoder-decoder with skip connections)
- Input: 3-band satellite image (C, 256, 256)
- Output: Reconstructed cloud-free image
- Parameters: ~1.9M

**Loss Function:** L1 + SSIM combined loss
- Emphasizes both pixel-level accuracy and structural similarity
- Cloud mask weighted regions for focused learning

**Optimizer:** Adam
- Learning rate: 1e-4
- Weight decay: 1e-5

**Data Augmentation:**
- Random rotation (±10°)
- Random flip (H/V)
- Random brightness/contrast
- Applied only to training set

**Validation Split:** 70% train / 15% val / 15% test

## Inference & Evaluation

### Run Inference
```bash
python -c "
import torch
import numpy as np
from pathlib import Path
from src.model.unet import build_model
import yaml

with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(cfg).to(device)
ckpt = torch.load('models/best_model.pth', map_location=device)
model.load_state_dict(ckpt['model_state'])

# Load test image
cloudy = np.load('data/processed_real/cloudy/cloudy_0000.npy')
inp = torch.from_numpy(cloudy.transpose(2, 0, 1)).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    pred = model(inp).cpu().numpy()[0].transpose(1, 2, 0)

print('Prediction shape:', pred.shape)
print('Value range:', pred.min(), pred.max())
"
```

### Metrics

After training, metrics are saved to `results/metrics/training_history.csv`:

| Metric | Description | Target |
|--------|---|---|
| PSNR | Peak Signal-to-Noise Ratio (dB) | >28 dB |
| SSIM | Structural Similarity Index | >0.85 |
| MAE | Mean Absolute Error | <0.05 |
| SAM | Spectral Angle Mapper (°) | <10° |

### Interactive Dashboard
```bash
streamlit run app.py
# Go to "Results" tab → Set test seed → Click "Run Inference"
# View side-by-side: Cloudy → Predicted → Ground Truth
```

## Directory Structure

```
data/
├── raw/
│   ├── clear/          # 10 original GeoTIFF scenes
│   └── cloudy/         # 10 original GeoTIFF scenes
└── processed_real/
    ├── clear/          # 2,447 patches
    ├── cloudy/         # 2,447 patches
    └── masks/          # 2,447 cloud masks

models/
└── best_model.pth      # Trained U-Net weights

results/
├── metrics/
│   └── training_history.csv
└── visualizations/     # Output inference images
```

## Next Steps (Phase 2)

- Integrate Sentinel-1 SAR fusion
- Multi-temporal stacking (Sentinel-2)
- Advanced cloud masking (Fmask, Sen2Cor)
- DEM-based topographic correction
- Expand training data with more Bhoonidhi scenes

