# 🛰️ Generative AI-Based Cloud Removal for LISS-IV Satellite Imagery

> **ISRO / NER Cloud Reconstruction Framework** 
> Prototype v1.0 — Step 1 of a multi-phase development roadmap

## 📋 Full Project Plan

### Phase 1 (NOW): Real Data Integration ✅
- **Real LISS-IV data:** 10 cloudy + 10 clear scenes → 2,447 training patches
- Data preprocessing: GeoTIFF reading, normalization, cloud detection
- U-Net based cloud removal model
- Training + inference on real satellite data
- Quantitative evaluation (PSNR, SSIM, MAE, SAM)
- Streamlit interactive dashboard
- CLI training script: `python phase1_real_data_train.py --epochs 30`

### Phase 2 (Next): Real Data Integration
- Integrate actual LISS-IV scenes from Bhoonidhi portal
- Sentinel-1 SAR auxiliary fusion
- Sentinel-2 multi-temporal stacking
- Advanced cloud masking (Fmask, Sen2Cor)
- DEM-based topographic correction

### Phase 3: Advanced Architecture
- Replace U-Net with Diffusion Model / Transformer (SwinIR / MAT)
- Multi-sensor fusion (SAR + Optical)
- Temporal attention module
- Conditional image generation with GAN discriminator

### Phase 4: Production Deployment
- End-to-end pipeline automation
- Docker containerization
- REST API (FastAPI)
- GEE integration for large-scale processing
- Final evaluation report & comparative analysis


## 🗂️ Project Structure

liss4_cloud_removal/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── configs/
│   └── config.yaml              # All hyperparameters & paths
├── src/
│   ├── preprocessing/
│   │   ├── data_generator.py    # Synthetic LISS-IV data generator
│   │   ├── cloud_simulator.py   # Realistic cloud mask simulation
│   │   └── augmentation.py      # Albumentations pipeline
│   ├── model/
│   │   ├── unet.py              # U-Net architecture
│   │   ├── losses.py            # Perceptual + L1 + SSIM losses
│   │   └── blocks.py            # Reusable conv/attention blocks
│   ├── training/
│   │   ├── trainer.py           # Training loop with validation
│   │   └── callbacks.py         # Checkpointing, early stopping
│   ├── evaluation/
│   │   ├── metrics.py           # PSNR, SSIM, MAE, SAM
│   │   └── visualizer.py        # Side-by-side result plots
│   └── utils/
│       ├── io_utils.py          # Image I/O (GDAL/rasterio)
│       └── logger.py            # Logging setup
├── data/
│   ├── raw/                     # Original cloudy images
│   ├── processed/               # Preprocessed patches
│   ├── masks/                   # Cloud masks
│   └── outputs/                 # Model predictions
├── models/                      # Saved model checkpoints
├── results/
│   ├── metrics/                 # CSV metric logs
│   └── visualizations/          # Output images
├── notebooks/
│   └── demo.ipynb               # Interactive demo notebook
├── tests/
│   └── test_pipeline.py         # Unit tests
├── app.py                       # Streamlit dashboard
└── run_demo.py                  # Single-command demo runner
```

## ⚡ Quick Start (Phase 1: Real Data)

### Prerequisites
- Python 3.9+
- CUDA GPU (optional, CPU works for prototype)
- ~2GB disk space

### 1. Clone / Extract & Setup Environment

```bash
cd liss4_cloud_removal
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Preprocess Real Data (One Time)

```bash
python -m src.preprocessing.real_data_loader
```

This will:
1. Read GeoTIFF files from `data/raw/clear/` and `data/raw/cloudy/`
2. Extract 256×256 patches (stride 128)
3. Normalize band values to [0, 1]
4. Detect cloud masks (brightness + morphology)
5. Save 2,447 training triplets to `data/processed_real/`

**Output:** `data/processed_real/` with subdirs: `clear/`, `cloudy/`, `masks/`

### 2b. Train Model on Real Data

```bash
python phase1_real_data_train.py --epochs 30 --batch_size 16
```

Or use Streamlit dashboard:

```bash
streamlit run app.py
# Select "Real Data" tab → Click "Run Full Training Pipeline"
```

Training will:
1. Load 2,447 real training pairs
2. Train U-Net for 30 epochs
3. Compute val metrics (PSNR, SSIM, MAE) 
4. Save best model to `models/best_model.pth`
5. Log history to `results/metrics/training_history.csv`

### 3. Launch Interactive Dashboard

```bash
streamlit run app.py
```
7.  Print summary report
Open http://localhost:8501 in your browser.

### 4. Train with Custom Config

```bash
python -m src.training.trainer --config configs/config.yaml --epochs 50
```

### 5. Run Inference on Your Image

```bash
python -m src.evaluation.visualizer --input path/to/cloudy.tif --output results/
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Expected Prototype Results

| Metric | Expected Range (Prototype) |
|--------|---------------------------|
| PSNR   | 28–34 dB                  |
| SSIM   | 0.82–0.92                 |
| MAE    | 0.02–0.06                 |

---

## 🔧 Configuration

Edit `configs/config.yaml` to change:
- Image patch size
- Model architecture depth
- Training epochs / batch size
- Loss function weights
- Data paths

---

## 📦 Dataset (for Real Data in Phase 2)

1. **LISS-IV**: Download from [Bhoonidhi](https://bhoonidhi.nrsc.gov.in)
2. **Sentinel-1**: [Copernicus Hub](https://scihub.copernicus.eu)
3. **Sentinel-2**: [Copernicus Hub](https://scihub.copernicus.eu)
4. Place raw `.tif` files in `data/raw/`

---

## 📄 License
MIT License — For research and academic use.




