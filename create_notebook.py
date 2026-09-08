#!/usr/bin/env python3
import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# 🛰️ LISS-IV Cloud Removal — Colab Demo\n", "**40 min end-to-end demo on free GPU**"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"colab": {"base_uri": "https://localhost:8080/"}, "id": "setup", "outputId": "out1"},
            "outputs": [],
            "source": ["!pip -q install torch torchvision rasterio scikit-image albumentations pyyaml\n", "import torch\n", "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import os\n", "os.chdir('/content')\n", "!git clone https://github.com/yuviiios/Generative-AI-Based-Cloud-Removal-and-Reconstruction-for-LISS-IV-Satellite-Imagery.git 2>/dev/null || echo exists\n", "os.chdir('Generative-AI-Based-Cloud-Removal-and-Reconstruction-for-LISS-IV-Satellite-Imagery/liss4_cloud_removal_final')\n", "import sys\n", "sys.path.insert(0, os.getcwd())\n", "print(f'Project ready: {os.getcwd()}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import subprocess, glob\n", "print('Downloading SEN12MS-CR...')\n", "subprocess.run(['wget', '-q', 'https://mediatum.ub.tum.de/download/1700268', '-O', 'sen12mscr.zip'], timeout=600)\n", "subprocess.run(['unzip', '-q', 'sen12mscr.zip'])\n", "s2_files = sorted(glob.glob('SEN12MSCR/ROIs*_s2/*.tif'))\n", "print(f'Found {len(s2_files)} scenes')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import rasterio, numpy as np\n", "def extract_liss4(tif):\n", "    with rasterio.open(tif) as src:\n", "        g = np.clip(src.read(3).astype(np.float32) / 10000, 0, 1)\n", "        r = np.clip(src.read(4).astype(np.float32) / 10000, 0, 1)\n", "        n = np.clip(src.read(8).astype(np.float32) / 10000, 0, 1)\n", "    return np.stack([g, r, n], axis=-1)\n", "test = extract_liss4(s2_files[0])\n", "print(f'Shape: {test.shape}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["from pathlib import Path\n", "from tqdm import tqdm\n", "from src.preprocessing.real_data_loader import extract_patches, detect_cloud_mask\n", "from src.preprocessing.cloud_simulator import CloudSimulator\n", "output_dir = Path('data/colab_dataset')\n", "output_dir.mkdir(parents=True, exist_ok=True)\n", "for d in ['clear', 'cloudy', 'masks']:\n", "    (output_dir / d).mkdir(exist_ok=True)\n", "print('Building dataset...')\n", "triplet_count = 0\n", "simulator = CloudSimulator(256)\n", "for idx, tif in enumerate(tqdm(s2_files[:15])):\n", "    img = extract_liss4(tif)\n", "    patches = extract_patches(img, 256, 192)\n", "    for p in patches[:10]:\n", "        mask = detect_cloud_mask(p)\n", "        if 0.05 < mask.mean() < 0.85:\n", "            np.save(output_dir / 'clear' / f'clear_{triplet_count:05d}.npy', p)\n", "            cm, sm = simulator.generate_cloud_mask(triplet_count)\n", "            cloudy = simulator.apply_clouds(p, cm, sm)\n", "            np.save(output_dir / 'cloudy' / f'cloudy_{triplet_count:05d}.npy', cloudy)\n", "            np.save(output_dir / 'masks' / f'mask_{triplet_count:05d}.npy', mask)\n", "            triplet_count += 1\n", "print(f'Created {triplet_count} triplets')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import yaml\n", "from src.training.trainer import Trainer\n", "with open('configs/config.yaml') as f:\n", "    cfg = yaml.safe_load(f)\n", "cfg['training']['epochs'] = 20\n", "cfg['training']['batch_size'] = 8\n", "trainer = Trainer(cfg)\n", "print(f'Training {cfg[\"training\"][\"epochs\"]} epochs...')\n", "history = trainer.train(str(output_dir))\n", "print(f'PSNR: {history[\"val_psnr\"][-1]:.2f} dB')\n", "print(f'SSIM: {history[\"val_ssim\"][-1]:.4f}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import torch, matplotlib.pyplot as plt\n", "from src.training.dataset import build_dataloaders\n", "from src.evaluation.metrics import evaluate_model\n", "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n", "_, _, test_loader = build_dataloaders(str(output_dir), 0.7, 0.15, 8, 42)\n", "results = evaluate_model(trainer.model, test_loader, device)\n", "print(f'PSNR: {results[\"psnr\"]:.2f} dB')\n", "print(f'SSIM: {results[\"ssim\"]:.4f}')\n", "fig, ax = plt.subplots()\n", "ax.plot(range(1, len(history[\"train_loss\"])+1), history[\"train_loss\"])\n", "ax.plot(range(1, len(history[\"val_loss\"])+1), history[\"val_loss\"])\n", "ax.set_title('Loss')\n", "plt.savefig('results/visualizations/curves.png')\n", "plt.show()"]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

Path("LISS4_CloudRemoval_Colab_Demo.ipynb").write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print("[OK] Notebook created: LISS4_CloudRemoval_Colab_Demo.ipynb")
