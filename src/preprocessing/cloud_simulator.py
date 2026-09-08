"""
Cloud & Shadow Simulator for LISS-IV imagery.
Generates realistic cloud masks using multi-scale blob simulation.
"""

import numpy as np
import cv2
from typing import Tuple


class CloudSimulator:
    """Simulates realistic cloud contamination patterns over satellite imagery."""

    def __init__(
        self,
        patch_size: int = 256,
        min_coverage: float = 0.1,
        max_coverage: float = 0.7,
        shadow_offset: Tuple[int, int] = (15, 20),
        shadow_intensity: float = 0.4,
    ):
        self.patch_size = patch_size
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.shadow_offset = shadow_offset
        self.shadow_intensity = shadow_intensity

    def _perlin_noise(self, size: int, freq: int, seed: int) -> np.ndarray:
        np.random.seed(seed)
        small = np.random.rand(freq, freq).astype(np.float32)
        return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)

    def generate_cloud_mask(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate binary cloud and shadow masks.
        Returns:
            cloud_mask: (H, W) binary uint8
            shadow_mask: (H, W) binary uint8
        """
        size = self.patch_size
        np.random.seed(seed)

        # Multi-scale noise to form realistic cloud shapes
        noise = np.zeros((size, size), dtype=np.float32)
        for scale, freq in zip([1.0, 0.5, 0.25], [4, 8, 16]):
            noise += scale * self._perlin_noise(size, freq, seed + int(scale * 10))
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

        # Target coverage
        target = np.random.uniform(self.min_coverage, self.max_coverage)
        threshold = np.percentile(noise, (1 - target) * 100)
        cloud_mask = (noise >= threshold).astype(np.uint8)

        # Morphological smoothing for natural cloud edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
        cloud_mask = cv2.GaussianBlur(cloud_mask.astype(np.float32), (11, 11), 0)
        cloud_mask = (cloud_mask > 0.5).astype(np.uint8)

        # Shadow: offset version of cloud mask
        ox, oy = self.shadow_offset
        ox += np.random.randint(-5, 5)
        oy += np.random.randint(-5, 5)
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        shadow_raw = cv2.warpAffine(cloud_mask.astype(np.float32), M, (size, size))
        shadow_mask = ((shadow_raw > 0.5) & (cloud_mask == 0)).astype(np.uint8)

        return cloud_mask, shadow_mask

    def apply_clouds(
        self,
        image: np.ndarray,
        cloud_mask: np.ndarray,
        shadow_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply cloud and shadow masks to a clear image.
        Args:
            image: (H, W, C) float32 in [0, 1]
            cloud_mask: (H, W) uint8 binary
            shadow_mask: (H, W) uint8 binary
        Returns:
            cloudy_image: (H, W, C) float32 in [0, 1]
        """
        cloudy = image.copy()
        H, W, C = image.shape

        # Cloud pixels: near-white with slight texture
        cloud_noise = np.random.rand(H, W, C).astype(np.float32) * 0.05
        cloud_pixels = 0.88 + cloud_noise
        cloud_pixels = np.clip(cloud_pixels, 0, 1)

        cm = cloud_mask[:, :, np.newaxis].astype(np.float32)
        sm = shadow_mask[:, :, np.newaxis].astype(np.float32)

        # Apply semi-transparent cloud blending
        alpha = 0.90
        cloudy = cloudy * (1 - cm) + (alpha * cloud_pixels + (1 - alpha) * cloudy) * cm

        # Apply shadow darkening
        cloudy = cloudy * (1 - sm * self.shadow_intensity)
        cloudy = np.clip(cloudy, 0, 1)

        return cloudy.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.preprocessing.data_generator import generate_liss4_scene

    sim = CloudSimulator(patch_size=256)
    clear = generate_liss4_scene(size=256, seed=7)
    mask, shadow = sim.generate_cloud_mask(seed=7)
    cloudy = sim.apply_clouds(clear, mask, shadow)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clear[:, :, :3])
    axes[0].set_title("Clear (RGB)")
    axes[1].imshow(cloudy[:, :, :3])
    axes[1].set_title("Cloudy (RGB)")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Cloud Mask")
    plt.tight_layout()
    plt.savefig("results/visualizations/cloud_simulation_demo.png", dpi=150)
    print("Saved demo visualization.")
