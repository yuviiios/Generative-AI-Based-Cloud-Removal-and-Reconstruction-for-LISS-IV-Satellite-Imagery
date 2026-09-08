"""
Augmentation pipeline for LISS-IV cloud removal training.
"""

import numpy as np
import cv2
from typing import Tuple


class SatelliteAugmentor:
    """Augmentation for satellite image pairs (cloudy, clear, mask)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        cloudy: np.ndarray,
        clear: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply synchronized augmentations to all three inputs."""

        # Random horizontal flip
        if np.random.rand() < self.p:
            cloudy = np.fliplr(cloudy).copy()
            clear  = np.fliplr(clear).copy()
            mask   = np.fliplr(mask).copy()

        # Random vertical flip
        if np.random.rand() < self.p:
            cloudy = np.flipud(cloudy).copy()
            clear  = np.flipud(clear).copy()
            mask   = np.flipud(mask).copy()

        # Random 90° rotation
        if np.random.rand() < self.p:
            k = np.random.randint(1, 4)
            cloudy = np.rot90(cloudy, k).copy()
            clear  = np.rot90(clear,  k).copy()
            mask   = np.rot90(mask,   k).copy()

        # Random brightness shift (spectral jitter)
        if np.random.rand() < self.p * 0.5:
            delta = np.random.uniform(-0.05, 0.05)
            cloudy = np.clip(cloudy + delta, 0, 1)
            clear  = np.clip(clear  + delta, 0, 1)

        return cloudy, clear, mask
