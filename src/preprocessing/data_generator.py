"""
Synthetic LISS-IV Data Generator
Simulates 3-band (Green, Red, NIR) satellite imagery for prototype testing.
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, List
import cv2
from tqdm import tqdm


def generate_terrain_band(size: int, scale: float = 0.005, seed: int = None) -> np.ndarray:
    """Generate realistic terrain-like reflectance using Perlin-like noise."""
    if seed is not None:
        np.random.seed(seed)
    
    # Multi-scale noise for realistic terrain
    noise = np.zeros((size, size))
    for i, freq in enumerate([1, 2, 4, 8, 16]):
        amplitude = 1.0 / (2 ** i)
        small = np.random.rand(size // freq + 2, size // freq + 2)
        zoomed = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        noise += amplitude * zoomed
    
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise.astype(np.float32)


def generate_liss4_scene(size: int = 256, seed: int = None) -> np.ndarray:
    """
    Generate a synthetic LISS-IV-like 3-band image.
    Bands: [Green (0.52-0.59µm), Red (0.62-0.68µm), NIR (0.77-0.86µm)]
    Returns: (H, W, 3) float32 array in [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    base = generate_terrain_band(size, seed=seed)
    
    # Simulate different land cover types
    land_type = np.zeros((size, size), dtype=np.int32)
    
    # Water bodies (low reflectance)
    water_mask = base < 0.2
    # Vegetation (high NIR, moderate Red, low Green)
    veg_mask = (base >= 0.2) & (base < 0.5)
    # Urban/bare (moderate across bands)
    urban_mask = (base >= 0.5) & (base < 0.75)
    # Cloud-bright areas or snow (high across all)
    bright_mask = base >= 0.75

    green = np.zeros((size, size), dtype=np.float32)
    red   = np.zeros((size, size), dtype=np.float32)
    nir   = np.zeros((size, size), dtype=np.float32)

    # Water: low reflectance, slightly higher in blue-green
    green[water_mask] = base[water_mask] * 0.15 + np.random.rand(*base[water_mask].shape) * 0.05
    red[water_mask]   = base[water_mask] * 0.08 + np.random.rand(*base[water_mask].shape) * 0.03
    nir[water_mask]   = base[water_mask] * 0.04 + np.random.rand(*base[water_mask].shape) * 0.02

    # Vegetation: red-edge effect, high NIR
    green[veg_mask] = base[veg_mask] * 0.12 + 0.05 + np.random.rand(*base[veg_mask].shape) * 0.03
    red[veg_mask]   = base[veg_mask] * 0.08 + 0.03 + np.random.rand(*base[veg_mask].shape) * 0.02
    nir[veg_mask]   = base[veg_mask] * 0.45 + 0.15 + np.random.rand(*base[veg_mask].shape) * 0.05

    # Urban/bare soil
    green[urban_mask] = base[urban_mask] * 0.30 + 0.10 + np.random.rand(*base[urban_mask].shape) * 0.05
    red[urban_mask]   = base[urban_mask] * 0.35 + 0.12 + np.random.rand(*base[urban_mask].shape) * 0.05
    nir[urban_mask]   = base[urban_mask] * 0.28 + 0.10 + np.random.rand(*base[urban_mask].shape) * 0.05

    # Bright (roads, sand, etc.)
    green[bright_mask] = base[bright_mask] * 0.40 + 0.20
    red[bright_mask]   = base[bright_mask] * 0.42 + 0.20
    nir[bright_mask]   = base[bright_mask] * 0.38 + 0.18

    # Add spatial texture with small noise
    noise_g = np.random.rand(size, size).astype(np.float32) * 0.02
    noise_r = np.random.rand(size, size).astype(np.float32) * 0.02
    noise_n = np.random.rand(size, size).astype(np.float32) * 0.02

    green = np.clip(green + noise_g, 0, 1)
    red   = np.clip(red   + noise_r, 0, 1)
    nir   = np.clip(nir   + noise_n, 0, 1)

    # Stack to (H, W, 3)
    scene = np.stack([green, red, nir], axis=-1)
    return scene.astype(np.float32)


def generate_dataset(
    output_dir: str,
    n_images: int = 200,
    patch_size: int = 256,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Generate full synthetic dataset: cloud-free + cloudy pairs.
    Returns lists of saved (cloudless, cloudy) image paths.
    """
    from src.preprocessing.cloud_simulator import CloudSimulator

    clear_dir = Path(output_dir) / "clear"
    cloudy_dir = Path(output_dir) / "cloudy"
    mask_dir = Path(output_dir) / "masks"
    
    clear_dir.mkdir(parents=True, exist_ok=True)
    cloudy_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    simulator = CloudSimulator(patch_size=patch_size)
    clear_paths, cloudy_paths = [], []

    print(f"[DataGen] Generating {n_images} synthetic LISS-IV image pairs...")
    for i in tqdm(range(n_images), desc="Generating"):
        clear_img = generate_liss4_scene(size=patch_size, seed=seed + i)
        cloud_mask, shadow_mask = simulator.generate_cloud_mask(seed=seed + i)
        cloudy_img = simulator.apply_clouds(clear_img, cloud_mask, shadow_mask)

        # Save as .npy for speed (swap to .tif with rasterio in Phase 2)
        c_path = str(clear_dir / f"clear_{i:04d}.npy")
        cy_path = str(cloudy_dir / f"cloudy_{i:04d}.npy")
        m_path = str(mask_dir / f"mask_{i:04d}.npy")

        np.save(c_path, clear_img)
        np.save(cy_path, cloudy_img)
        np.save(m_path, cloud_mask)

        clear_paths.append(c_path)
        cloudy_paths.append(cy_path)

    print(f"[DataGen] Saved {n_images} pairs to {output_dir}")
    return clear_paths, cloudy_paths


if __name__ == "__main__":
    generate_dataset("data/processed", n_images=10)
