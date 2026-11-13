from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    seed: int = 42


@dataclass
class Noise2DetailConfig:
    chan_embed: int = 48
    iters: int = 1000
    lr: float = 1e-3


@dataclass
class FlowModelConfig:
    base_channels: int = 48


@dataclass
class FlowTrainConfig:
    steps: int = 1000
    lr: float = 1e-4
    patch: int = 128
    batch: int = 8
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    rho: float = 0.50        # tube radius scale factor (rho in the paper)
    sigma_cap: float = 0.03  # tube radius upper bound (sigma_cap in the paper)


@dataclass
class StrengthLearnConfig:
    """Configuration for learning optimal strength B after flow net training."""
    steps: int = 1000
    lr: float = 1e-2
    init_strength: float = 0.20
    infer_multiplier: float = 1.15  # Use B * 1.15 for inference


@dataclass
class FlowInferConfig:
    mode: str = "onestep" # "integrate" or "onestep"

    # only for integrate mode
    infer_steps: int = 10
    integrator: str = "heun"

    clamp01: bool = True


@dataclass
class Config:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    n2d: Noise2DetailConfig = field(default_factory=Noise2DetailConfig)
    flow_model: FlowModelConfig = field(default_factory=FlowModelConfig)
    flow_train: FlowTrainConfig = field(default_factory=FlowTrainConfig)
    strength_learn: StrengthLearnConfig = field(default_factory=StrengthLearnConfig)
    flow_infer: FlowInferConfig = field(default_factory=FlowInferConfig)
