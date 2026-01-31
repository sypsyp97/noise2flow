# Noise2Flow

Single-image denoising via learning mean denoising flows.

## Overview

Noise2Flow learns a mean velocity field from a single noisy image using a weak denoiser as directional anchor. Key features:

- **Direction-only learning**: Supervises only restoration direction via cosine similarity, decoupling from magnitude bias
- **Monte Carlo tube sampling**: Regularizes the learned field by sampling around the anchor trajectory
- **Single-step inference (1-NFE)**: No ODE integration required at test time

## Requirements

```
torch
numpy
scipy
tqdm
matplotlib
scikit-image
```

## Usage

Quick test:
```bash
python main.py
```

## Configuration

Edit `config.py` to adjust:

- `flow_train.steps`: Training iterations (default: 1000)
- `flow_train.patch`: Patch size (default: 128)
- `flow_train.rho`: Tube radius scale (default: 0.5)
- `flow_infer.mode`: "onestep" or "integrate"
- `flow_infer.infer_steps`: Steps for multi-step inference (default: 10)
- `flow_infer.integrator`: "euler" or "heun"

## License

Anonymous submission - license to be added upon acceptance.
