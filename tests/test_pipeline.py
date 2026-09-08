"""
Unit tests for the LISS-IV cloud removal pipeline.
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Data Generation Tests ────────────────────────────────────────
class TestDataGenerator:
    def test_scene_shape(self):
        from src.preprocessing.data_generator import generate_liss4_scene
        img = generate_liss4_scene(size=64, seed=0)
        assert img.shape == (64, 64, 3), f"Expected (64,64,3), got {img.shape}"

    def test_scene_range(self):
        from src.preprocessing.data_generator import generate_liss4_scene
        img = generate_liss4_scene(size=64, seed=1)
        assert img.min() >= 0.0, "Pixel values below 0"
        assert img.max() <= 1.0, "Pixel values above 1"

    def test_scene_dtype(self):
        from src.preprocessing.data_generator import generate_liss4_scene
        img = generate_liss4_scene(size=64, seed=2)
        assert img.dtype == np.float32


# ── Cloud Simulator Tests ────────────────────────────────────────
class TestCloudSimulator:
    def test_mask_shape(self):
        from src.preprocessing.cloud_simulator import CloudSimulator
        sim = CloudSimulator(patch_size=64)
        mask, shadow = sim.generate_cloud_mask(seed=0)
        assert mask.shape == (64, 64)
        assert shadow.shape == (64, 64)

    def test_mask_binary(self):
        from src.preprocessing.cloud_simulator import CloudSimulator
        sim = CloudSimulator(patch_size=64)
        mask, shadow = sim.generate_cloud_mask(seed=3)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_cloudy_image_range(self):
        from src.preprocessing.data_generator import generate_liss4_scene
        from src.preprocessing.cloud_simulator import CloudSimulator
        sim = CloudSimulator(patch_size=64)
        clear = generate_liss4_scene(size=64, seed=0)
        mask, shadow = sim.generate_cloud_mask(seed=0)
        cloudy = sim.apply_clouds(clear, mask, shadow)
        assert cloudy.min() >= 0.0
        assert cloudy.max() <= 1.0

    def test_no_mutual_overlap(self):
        """Cloud and shadow masks should not overlap."""
        from src.preprocessing.cloud_simulator import CloudSimulator
        sim = CloudSimulator(patch_size=64)
        mask, shadow = sim.generate_cloud_mask(seed=5)
        overlap = (mask == 1) & (shadow == 1)
        assert not overlap.any(), "Cloud and shadow masks overlap"


# ── Model Tests ──────────────────────────────────────────────────
class TestUNet:
    def test_output_shape(self):
        from src.model.unet import UNet
        model = UNet(in_channels=3, out_channels=3, base_features=8, depth=2)
        x = torch.randn(2, 3, 64, 64)
        y = model(x)
        assert y.shape == (2, 3, 64, 64)

    def test_output_range(self):
        from src.model.unet import UNet
        model = UNet(in_channels=3, out_channels=3, base_features=8, depth=2)
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_parameter_count(self):
        from src.model.unet import UNet
        model = UNet(base_features=8, depth=2)
        assert model.count_parameters() > 0


# ── Metrics Tests ────────────────────────────────────────────────
class TestMetrics:
    def test_psnr_identical(self):
        from src.evaluation.metrics import psnr
        x = np.random.rand(3, 64, 64).astype(np.float32)
        assert psnr(x, x) > 60.0, "PSNR of identical images should be very high"

    def test_ssim_identical(self):
        from src.evaluation.metrics import ssim
        x = np.random.rand(3, 64, 64).astype(np.float32)
        score = ssim(x, x)
        assert score > 0.99

    def test_mae_zero(self):
        from src.evaluation.metrics import mae
        x = np.random.rand(3, 64, 64).astype(np.float32)
        assert mae(x, x) < 1e-6

    def test_sam_zero(self):
        from src.evaluation.metrics import sam
        x = np.random.rand(3, 64, 64).astype(np.float32)
        score = sam(x, x)
        assert score < 1e-3, f"SAM of identical images should be ~0, got {score}"

    def test_batch_metrics(self):
        from src.evaluation.metrics import compute_metrics
        pred   = np.random.rand(4, 3, 64, 64).astype(np.float32)
        target = np.random.rand(4, 3, 64, 64).astype(np.float32)
        m = compute_metrics(pred, target)
        assert "psnr" in m and "ssim" in m and "mae" in m and "sam" in m


# ── Loss Tests ───────────────────────────────────────────────────
class TestLoss:
    def test_loss_zero_for_identical(self):
        from src.model.losses import CloudRemovalLoss
        criterion = CloudRemovalLoss()
        x = torch.rand(2, 3, 64, 64)
        loss = criterion(x, x)
        assert loss.item() < 0.05

    def test_loss_positive(self):
        from src.model.losses import CloudRemovalLoss
        criterion = CloudRemovalLoss()
        pred   = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        loss = criterion(pred, target)
        assert loss.item() > 0
