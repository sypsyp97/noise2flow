from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import FlowInferConfig


def infer_flow_tensor(
    *,
    net: nn.Module,
    y: torch.Tensor,
    strength: float,
    mode: str = "onestep",
    clamp01: bool = True,
) -> torch.Tensor:
    """
    Inference that returns a tensor (for gradient computation during strength learning).
    """
    def v_field(x_in: torch.Tensor) -> torch.Tensor:
        v_raw = net(torch.cat([x_in, y], dim=1))
        return torch.tanh(v_raw)

    if mode == "onestep":
        x = y + strength * v_field(y)
        if clamp01:
            x = torch.clamp(x, 0.0, 1.0)
        return x

    raise ValueError(f"infer_flow_tensor only supports 'onestep' mode, got '{mode}'")


@torch.no_grad()
def infer_flow(
    *,
    net: nn.Module,
    y: torch.Tensor,
    strength: float,
    infer_cfg: FlowInferConfig,
) -> np.ndarray:
    net.eval()

    mode = str(infer_cfg.mode).lower().strip()
    integ = str(infer_cfg.integrator).lower().strip()
    do_clamp = bool(infer_cfg.clamp01)

    def v_field(x_in: torch.Tensor) -> torch.Tensor:
        v_raw = net(torch.cat([x_in, y], dim=1))
        return torch.tanh(v_raw)

    if mode == "onestep":
        x = y + strength * v_field(y)
        if do_clamp:
            x = torch.clamp(x, 0.0, 1.0)
        return x.squeeze().cpu().numpy().astype(np.float32)

    if mode != "integrate":
        print(f"[warn] Unknown flow_infer.mode '{infer_cfg.mode}', using 'onestep'")
        x = y + strength * v_field(y)
        if do_clamp:
            x = torch.clamp(x, 0.0, 1.0)
        return x.squeeze().cpu().numpy().astype(np.float32)

    K = max(1, int(infer_cfg.infer_steps))
    h = strength / float(K)
    x = y.clone()

    for _ in range(K):
        if integ == "euler":
            x = x + h * v_field(x)
            if do_clamp:
                x = torch.clamp(x, 0.0, 1.0)
        else:
            # Heun method (optional multi-step inference in the paper)
            # x_{n+1} = Π(x_n + h/2 * (v(x_n) + v(x_n + h*v(x_n))))
            v1 = v_field(x)
            x_e = x + h * v1  # no clamp on intermediate step
            v2 = v_field(x_e)
            x = x + 0.5 * h * (v1 + v2)
            if do_clamp:
                x = torch.clamp(x, 0.0, 1.0)

    return x.squeeze().cpu().numpy().astype(np.float32)
