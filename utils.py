from __future__ import annotations

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HW or CHW numpy array to BCHW tensor."""
    if img.ndim == 2:
        img = img[np.newaxis, np.newaxis, ...]
    elif img.ndim == 3:
        img = img[np.newaxis, ...]
    return torch.from_numpy(img.astype(np.float32)).to(device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert BCHW tensor to HW numpy array."""
    return tensor.squeeze().cpu().numpy().astype(np.float32)


def normalize_to_01(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def create_placeholder_image(
    height: int = 256,
    width: int = 256,
    spatial_noise_level: float = 0.1,
    direct_noise_level: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create placeholder noisy and clean image pair for testing.

    Noise mode:
      1. downsample x2 -> add Gaussian noise -> upsample x2 (spatially correlated noise)
      2. add direct Gaussian noise at original resolution

    Args:
        height: Image height
        width: Image width
        spatial_noise_level: Spatially correlated noise level (downsampled then upsampled)
        direct_noise_level: Direct Gaussian noise level at original resolution

    Returns:
        noisy: Noisy image in [0, 1] range
        clean: Clean image in [0, 1] range
    """
    from scipy.ndimage import zoom

    # Create a simple pattern as clean image
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    # Combine patterns: circles + gradients + hard edges
    clean = np.zeros((height, width), dtype=np.float32)

    # Add circular patterns (soft)
    for cx, cy, r in [(0.3, 0.3, 0.15), (0.7, 0.7, 0.2), (0.5, 0.5, 0.25)]:
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        clean += 0.3 * np.exp(-dist**2 / (2 * r**2))

    # Add gradient
    clean += 0.2 * xx + 0.1 * yy

    # Add hard edge shapes
    # Rectangle 1
    rect1 = ((xx >= 0.1) & (xx <= 0.25) & (yy >= 0.6) & (yy <= 0.85)).astype(np.float32)
    clean += 0.4 * rect1

    # Rectangle 2
    rect2 = ((xx >= 0.75) & (xx <= 0.9) & (yy >= 0.1) & (yy <= 0.35)).astype(np.float32)
    clean += 0.3 * rect2

    # Hard circle (disk)
    disk_dist = np.sqrt((xx - 0.15)**2 + (yy - 0.2)**2)
    disk = (disk_dist <= 0.08).astype(np.float32)
    clean += 0.5 * disk

    # Triangle
    tri_mask = ((yy >= 0.7) & (yy <= 0.95) & (xx >= 0.6) & (xx <= 0.6 + (0.95 - yy) * 0.8)).astype(np.float32)
    clean += 0.35 * tri_mask

    # Normalize to [0, 1]
    clean = normalize_to_01(clean)

    # Step 1: Downsample clean image by factor of 2
    clean_down = zoom(clean, 0.5, order=1)

    # Step 2: Add spatially correlated Gaussian noise at downsampled resolution
    spatial_noise = np.random.randn(*clean_down.shape).astype(np.float32) * spatial_noise_level
    noisy_down = clean_down + spatial_noise

    # Step 3: Upsample back to original resolution
    noisy = zoom(noisy_down, 2.0, order=1).astype(np.float32)

    # Ensure output has exact original size (in case of rounding issues)
    noisy = noisy[:height, :width]

    # Step 4: Add direct Gaussian noise at original resolution
    if direct_noise_level > 0:
        direct_noise = np.random.randn(height, width).astype(np.float32) * direct_noise_level
        noisy = noisy + direct_noise

    # Clip to [0, 1]
    noisy = np.clip(noisy, 0, 1).astype(np.float32)

    return noisy, clean
