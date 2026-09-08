"""
Quantitative metrics for cloud removal evaluation.
"""

import numpy as np
from typing import Dict
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (higher is better)."""
    return peak_signal_noise_ratio(target, pred, data_range=1.0)


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Structural Similarity Index (higher is better, max=1)."""
    # pred/target: (C, H, W) or (H, W, C)
    if pred.ndim == 3 and pred.shape[0] <= 4:
        pred   = pred.transpose(1, 2, 0)
        target = target.transpose(1, 2, 0)
    return structural_similarity(target, pred, data_range=1.0, channel_axis=-1)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error (lower is better)."""
    return float(np.mean(np.abs(pred - target)))


def sam(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Spectral Angle Mapper in degrees (lower is better).
    Measures spectral fidelity between reconstructed and reference.
    """
    if pred.ndim == 3 and pred.shape[0] <= 4:
        pred   = pred.transpose(1, 2, 0)   # (H, W, C)
        target = target.transpose(1, 2, 0)

    dot   = np.sum(pred * target, axis=-1)
    norm1 = np.linalg.norm(pred,   axis=-1) + eps
    norm2 = np.linalg.norm(target, axis=-1) + eps
    cos_angle = np.clip(dot / (norm1 * norm2), -1, 1)
    angles = np.arccos(cos_angle) * (180.0 / np.pi)
    return float(np.mean(angles))


def compute_metrics(
    pred_batch: np.ndarray,
    target_batch: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all metrics over a batch.
    Args:
        pred_batch:   (B, C, H, W) numpy
        target_batch: (B, C, H, W) numpy
    Returns:
        dict of averaged metrics
    """
    results = {"psnr": [], "ssim": [], "mae": [], "sam": []}
    B = pred_batch.shape[0]

    for i in range(B):
        p = pred_batch[i]
        t = target_batch[i]
        results["psnr"].append(psnr(p, t))
        results["ssim"].append(ssim(p, t))
        results["mae"].append(mae(p, t))
        results["sam"].append(sam(p, t))

    return {k: float(np.mean(v)) for k, v in results.items()}


def evaluate_model(model, test_loader, device) -> Dict[str, float]:
    """Full evaluation on test set."""
    import torch
    model.eval()
    all_psnr, all_ssim, all_mae, all_sam = [], [], [], []

    with torch.no_grad():
        for cloudy, clear, mask in test_loader:
            cloudy = cloudy.to(device)
            pred = model(cloudy).cpu().numpy()
            target = clear.numpy()
            m = compute_metrics(pred, target)
            all_psnr.append(m["psnr"])
            all_ssim.append(m["ssim"])
            all_mae.append(m["mae"])
            all_sam.append(m["sam"])

    results = {
        "psnr": float(np.mean(all_psnr)),
        "ssim": float(np.mean(all_ssim)),
        "mae":  float(np.mean(all_mae)),
        "sam":  float(np.mean(all_sam)),
    }
    return results
