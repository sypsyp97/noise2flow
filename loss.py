from __future__ import annotations

import torch


def rms_per_sample(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute RMS per sample in a batch. Input shape: (B, C, H, W)."""
    return torch.sqrt(torch.mean(x * x, dim=(1, 2, 3)) + eps)


def weighted_dir_loss(v: torch.Tensor, d: torch.Tensor, eps: float = 1e-8, delta: float = 1e-6) -> torch.Tensor:
    """Direction-only loss via weighted cosine similarity (L_dir in the paper)."""
    # Importance map m = |d| + δ
    w = torch.abs(d).detach() + delta

    v_flat = v.flatten(1)
    d_flat = d.flatten(1)
    w_flat = w.flatten(1)

    # Weighted inner product and norms
    dot = (w_flat * v_flat * d_flat).sum(dim=1)
    v_norm = torch.sqrt((w_flat * (v_flat * v_flat)).sum(dim=1) + eps)
    d_norm = torch.sqrt((w_flat * (d_flat * d_flat)).sum(dim=1) + eps)

    cos = dot / (v_norm * d_norm + eps)
    cos = torch.clamp(cos, -1.0, 1.0)

    # L_dir = E[1 - cos]
    return (1.0 - cos).mean()
