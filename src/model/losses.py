"""
Loss functions for cloud removal:
- L1 (pixel-wise reconstruction)
- SSIM-based structural loss
- Combined loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable SSIM loss (1 - SSIM)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, C, H, W = pred.shape

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    pad = window_size // 2
    mu1 = F.conv2d(pred,   window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    s1  = F.conv2d(pred   * pred,   window, padding=pad, groups=C) - mu1_sq
    s2  = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(pred   * target, window, padding=pad, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2) + 1e-8)

    return 1.0 - ssim_map.mean()


class CloudRemovalLoss(nn.Module):
    """
    Combined loss for cloud removal:
      total = l1_w * L1 + ssim_w * SSIM_loss
    Optionally weighted by cloud mask (focus on cloud regions).
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.5):
        super().__init__()
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H, W) predicted clear image
            target: (B, C, H, W) ground truth clear image
            mask:   (B, 1, H, W) cloud mask (optional, for region weighting)
        """
        l1 = self.l1(pred, target)
        ss = ssim_loss(pred, target)

        if mask is not None:
            # Extra penalty inside cloud regions
            masked_l1 = (torch.abs(pred - target) * mask).mean()
            l1 = l1 + 0.5 * masked_l1

        return self.l1_weight * l1 + self.ssim_weight * ss


def build_loss(cfg: dict) -> CloudRemovalLoss:
    loss_cfg = cfg.get("loss", {})
    return CloudRemovalLoss(
        l1_weight=loss_cfg.get("l1_weight", 1.0),
        ssim_weight=loss_cfg.get("ssim_weight", 0.5),
    )
