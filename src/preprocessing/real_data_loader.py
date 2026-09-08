"""
Phase 2: Real Satellite Data Loader
Reads real GeoTIFF files and prepares
them for training.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import List, Tuple
import cv2
from tqdm import tqdm


def read_geotiff(filepath: str) -> np.ndarray:
    """Read a GeoTIFF and return as (H, W, C) float32 array in [0,1]."""
    with rasterio.open(filepath) as src:
        img = src.read().astype(np.float32)
        img = np.transpose(img, (1, 2, 0))

    # Normalize each band to [0, 1]
    for c in range(img.shape[-1]):
        band = img[:, :, c]
        p2, p98 = np.percentile(band, [2, 98])
        if p98 > p2:
            img[:, :, c] = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            img[:, :, c] = 0
    return img.astype(np.float32)


def extract_patches(
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 128
) -> List[np.ndarray]:
    """Split large satellite image into small training patches."""
    H, W, C = image.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size, :]
            if patch.std() > 0.01:
                patches.append(patch)
    if len(patches) == 0:
        resized = cv2.resize(image, (patch_size, patch_size))
        patches.append(resized)
    return patches


def detect_cloud_mask(image: np.ndarray) -> np.ndarray:
    """
    Detect clouds using brightness thresholding.
    Clouds = very bright across all bands.
    """
    brightness = image.mean(axis=-1)
    band_std = image.std(axis=-1)
    cloud_mask = (
        (brightness > 0.75) & (band_std < 0.15)
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    cloud_mask = cv2.morphologyEx(
        cloud_mask, cv2.MORPH_OPEN, kernel
    )
    cloud_mask = cv2.morphologyEx(
        cloud_mask, cv2.MORPH_CLOSE, kernel
    )
    return cloud_mask


def build_real_dataset(
    clear_dir: str,
    cloudy_dir: str,
    output_dir: str,
    patch_size: int = 256,
    stride: int = 128,
):
    """
    Process raw GeoTIFFs into training-ready patches.
    Saves (clear, cloudy, mask) triplets as .npy files.
    """
    clear_dir  = Path(clear_dir)
    cloudy_dir = Path(cloudy_dir)
    out_dir    = Path(output_dir)

    out_clear  = out_dir / "clear"
    out_cloudy = out_dir / "cloudy"
    out_masks  = out_dir / "masks"

    for d in [out_clear, out_cloudy, out_masks]:
        d.mkdir(parents=True, exist_ok=True)

    clear_files = sorted(
        list(clear_dir.glob("*.tif")) +
        list(clear_dir.glob("*.tiff"))
    )
    cloudy_files = sorted(
        list(cloudy_dir.glob("*.tif")) +
        list(cloudy_dir.glob("*.tiff"))
    )

    print(f"Found {len(clear_files)} clear scenes")
    print(f"Found {len(cloudy_files)} cloudy scenes")

    # Extract all clear patches first
    clear_patches = []
    for f in tqdm(clear_files, desc="Reading clear"):
        img = read_geotiff(str(f))
        patches = extract_patches(img, patch_size, stride)
        clear_patches.extend(patches)
    print(f"Total clear patches: {len(clear_patches)}")

    # Extract cloudy patches + detect masks
    idx = 0
    for f in tqdm(cloudy_files, desc="Reading cloudy"):
        img = read_geotiff(str(f))
        patches = extract_patches(img, patch_size, stride)
        for patch in patches:
            mask = detect_cloud_mask(patch)
            coverage = mask.mean()
            # Only keep patches with 5-85% cloud cover
            if 0.05 < coverage < 0.85:
                np.save(
                    out_cloudy / f"cloudy_{idx:04d}.npy",
                    patch
                )
                np.save(
                    out_masks / f"mask_{idx:04d}.npy",
                    mask
                )
                # Pair with a clear patch
                ref = clear_patches[idx % len(clear_patches)]
                np.save(
                    out_clear / f"clear_{idx:04d}.npy",
                    ref
                )
                idx += 1

    print(f"Saved {idx} training triplets to {output_dir}")
    return idx


if __name__ == "__main__":
    build_real_dataset(
        clear_dir="data/raw/clear",
        cloudy_dir="data/raw/cloudy",
        output_dir="data/processed_real",
        patch_size=256,
        stride=128,
    )