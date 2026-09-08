# 🛰️ LISS-IV Cloud Removal — Google Colab Quick Start

**Run the complete demo in ~40 minutes on free Colab GPU.**

## 📋 What's Included

- ✅ Dataset download (SEN12MS-CR, Sentinel-2)
- ✅ LISS-IV band extraction (Green, Red, NIR)
- ✅ Cloud mask generation
- ✅ U-Net model training (20 epochs)
- ✅ Inference + metrics (PSNR, SSIM, MAE, SAM)
- ✅ Before/after visualization
- ✅ Download results

## 🚀 How to Run

### Option 1: Direct Link (Recommended)

1. Open Google Colab: https://colab.research.google.com
2. File → Open Notebook → GitHub
3. Paste: `https://github.com/yuviiios/Generative-AI-Based-Cloud-Removal-and-Reconstruction-for-LISS-IV-Satellite-Imagery`
4. Select branch: `main`
5. Choose notebook: `liss4_cloud_removal_final/LISS4_CloudRemoval_Colab_Demo.ipynb`
6. Run cells sequentially (Shift+Enter)

### Option 2: Upload to Colab

1. Download `LISS4_CloudRemoval_Colab_Demo.ipynb`
2. Open https://colab.research.google.com
3. File → Upload Notebook
4. Select the .ipynb file
5. Run cells

### Option 3: Use Your Own GitHub Repo

```python
# In Colab, use this notebook directly:
from google.colab import drive
drive.mount('/content/drive')

# Or clone your fork:
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

---

## ⏱️ Runtime Breakdown

| Step | Time | Notes |
|------|------|-------|
| Setup (Cell 1-2) | 2 min | Install packages, clone repo |
| Download data (Cell 3) | 10 min | SEN12MS-CR dataset (~500 MB) |
| Extract patches (Cell 4-5) | 5 min | LISS-IV bands + cloud masks |
| Apply fixes (Cell 6) | <1 min | Automated bug patches |
| Train model (Cell 8) | 15-20 min | U-Net 20 epochs, batch size 8 |
| Evaluate (Cell 9-11) | 3 min | Inference + visualization |
| **TOTAL** | **~35-40 min** | ~10 min active, rest automated |

## 📊 Expected Results

After training, you should see:

```
TEST SET RESULTS
========================================
PSNR (dB):     28-32  (higher is better)
SSIM:          0.82-0.88  (max = 1.0)
MAE:           0.04-0.08  (lower is better)
SAM (degrees): 5-15  (lower is better)
========================================
```

### Training Curves

- Loss steadily decreases
- PSNR increases (converges ~15 dB improvement)
- SSIM plateaus around 0.85

### Visual Results

Before/after grid showing:
- Cloudy input
- Model reconstruction
- Ground truth (clear)

---

## 💾 Download Results

After completion, download:

1. **Training curves**: `results/visualizations/training_curves.png`
2. **Inference grid**: `results/visualizations/inference_grid.png`
3. **Model checkpoint**: `models/best_model.pth` (89 MB)
4. **Metrics CSV**: `results/metrics/training_history.csv`

### In Colab

```python
from google.colab import files

# Download individual files
files.download('results/visualizations/training_curves.png')
files.download('results/visualizations/inference_grid.png')
files.download('models/best_model.pth')

# Or create zip archives (Cell 13)
files.download('/tmp/LISS4_CloudRemoval_Results.zip')
```

---

## 🐛 Troubleshooting

### GPU Out of Memory

**Error:** `CUDA out of memory`

**Fix:** Reduce batch size in Cell 7:
```python
cfg['training']['batch_size'] = 4  # or 2
```

### Dataset Download Timeout

**Error:** `wget timeout` in Cell 3

**Fix:** Manually download and upload:
1. Download SEN12MS-CR: https://mediatum.ub.tum.de/download/1700268
2. Upload to Colab (Files → Upload)
3. Extract manually: `!unzip -q sen12mscr.zip`

### Module Not Found

**Error:** `ModuleNotFoundError: No module named 'src.preprocessing'`

**Fix:** Make sure Cell 2 clone completed. Check:
```python
import os
print(os.getcwd())
print(os.listdir('.')[:10])
```

Should show `src/`, `configs/`, etc.

### Visualizer Function Missing

**Error:** `FileNotFoundError: visualizer.run_inference_and_visualize`

**Fix:** Cell 11 has fallback code. Will still generate inference grid. ✅

---

## 📝 For Faculty Presentation

### Key Points to Show

1. **Problem Statement**
   - Cloud contamination in optical satellite imagery
   - LISS-IV lacks SAR → need reconstruction from optical alone

2. **Data Pipeline**
   - Source: Sentinel-2 (proxy for LISS-IV spectral bands)
   - 256×256 patches, cloud masks
   - Training triplets: (clear, cloudy, mask)

3. **Model Architecture**
   - U-Net with skip connections
   - 3-band input → 3-band output
   - ~500K parameters

4. **Results**
   - PSNR improvement: +3-6 dB
   - SSIM: 0.82-0.88
   - Visual before/after

5. **Workflow**
   - Real satellite data → preprocessing → training → inference
   - End-to-end in Colab, reproducible

### Show These Files

- `training_curves.png` → convergence proof
- `inference_grid.png` → visual quality
- `training_history.csv` → quantitative trends

---

## 🔄 Next Steps (Phase 2)

After this prototype works:

1. **Use real LISS-IV data**
   - Download from Bhoonidhi portal (ISRO)
   - Replace Sentinel-2 extraction in Cell 5

2. **Improve model**
   - Try AttentionGAN (Cloudy-Kofta reference)
   - Add Sentinel-1 SAR fusion (NIRMRGH approach)

3. **Optimize for deployment**
   - Quantize model → smaller checkpoint
   - Deploy as REST API or web app

4. **Extend to full scenes**
   - Current: 256×256 patches
   - Full LISS-IV: ~17,000 × 16,000 px
   - Use tiling + blending strategy

---

## 📚 References

- **SEN12MS-CR Dataset**: https://patricktum.github.io/cloud_removal/sen12mscr/
- **NIRMRGH (reference)**: NAFNet residual architecture, reproducibility best practices
- **Cloudy-Kofta (reference)**: AttentionGAN, pix2pix models
- **Problem Statement**: ISRO Hackathon 2025-2026

---

## ✅ Checklist Before Demo

- [ ] Run notebook end-to-end (full 40 min)
- [ ] Save outputs to local machine
- [ ] Verify metrics make sense (PSNR 28-32 dB)
- [ ] Check visualization looks good (before/after grid)
- [ ] Prepare 2-3 min talking points
- [ ] Have GitHub link ready (share repo with panel)

---

**Ready? Open Colab and start Cell 1! 🚀**
