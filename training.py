from __future__ import annotations

import numpy as np
import torch


def _augment_pair(rng: np.random.Generator, y: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(rng.integers(0, 4))
    if k:
        y = torch.rot90(y, k=k, dims=(-2, -1))
        w = torch.rot90(w, k=k, dims=(-2, -1))
    if bool(rng.integers(0, 2)):
        y = torch.flip(y, dims=(-1,))
        w = torch.flip(w, dims=(-1,))
    if bool(rng.integers(0, 2)):
        y = torch.flip(y, dims=(-2,))
        w = torch.flip(w, dims=(-2,))
    return y, w


def sample_patches(
    *,
    rng: np.random.Generator,
    y_full: torch.Tensor,
    anchor: torch.Tensor,
    patch: int,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample patches from noisy image and anchor.

    Args:
        rng: Random number generator
        y_full: Noisy image tensor (1, 1, H, W)
        anchor: Denoised anchor (1, 1, H, W)
        patch: Patch size
        batch: Batch size
        device: Device

    Returns:
        Tuple of (noisy_patches, anchor_patches)
    """
    h, w = int(y_full.shape[-2]), int(y_full.shape[-1])
    patch = min(patch, h, w)

    top = torch.randint(0, h - patch + 1, (batch,), device=device)
    left = torch.randint(0, w - patch + 1, (batch,), device=device)

    yp = torch.empty((batch, 1, patch, patch), device=device)
    wp = torch.empty_like(yp)

    for i in range(batch):
        ys = y_full[..., top[i] : top[i] + patch, left[i] : left[i] + patch]
        ws = anchor[..., top[i] : top[i] + patch, left[i] : left[i] + patch]

        ys, ws = _augment_pair(rng, ys, ws)
        yp[i] = ys
        wp[i] = ws

    return yp, wp
