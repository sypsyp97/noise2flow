from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class N2D_DenoisingNetwork(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 48):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _create_downsampling_filters(channels: int, device: torch.device):
    f1 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=device).repeat(channels, 1, 1, 1)
    f2 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=device).repeat(channels, 1, 1, 1)
    return f1, f2


def pair_downsample(image: torch.Tensor):
    c = int(image.shape[1])
    f1, f2 = _create_downsampling_filters(c, image.device)
    d1 = F.conv2d(image, f1, stride=2, groups=c)
    d2 = F.conv2d(image, f2, stride=2, groups=c)
    return d1, d2


def pixel_unshuffle(x: torch.Tensor, factor: int):
    b, c, h, w = x.shape
    oh, ow = h // factor, w // factor
    x = x.contiguous().view(b, c, oh, factor, ow, factor)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    return x.view(b * factor * factor, c, oh, ow)


def pixel_shuffle(x: torch.Tensor, factor: int):
    b, c, h, w = x.shape
    ob = b // (factor * factor)
    x = x.contiguous().view(ob, factor, factor, c, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    return x.view(ob, c, h * factor, w * factor)


def n2d_loss_stage1(noisy: torch.Tensor, model: nn.Module) -> torch.Tensor:
    noisy1, noisy2 = pair_downsample(noisy)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))

    denoised = noisy - model(noisy)
    den1, den2 = pair_downsample(denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, den1) + F.mse_loss(pred2, den2))
    return loss_res + loss_cons


def n2d_loss_stage3(semi: torch.Tensor, noisy: torch.Tensor, model: nn.Module) -> torch.Tensor:
    noisy1, noisy2 = pair_downsample(noisy)
    semi1, semi2 = pair_downsample(semi)
    pred1 = model(semi1)
    pred2 = model(semi2)
    loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))

    denoised = model(semi)
    den1, den2 = pair_downsample(denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, den1) + F.mse_loss(pred2, den2))
    return loss_res + loss_cons


def noise2detail_stage3_anchor(noisy01: torch.Tensor, *, chan_embed: int, iters: int, lr: float) -> torch.Tensor:
    device = noisy01.device
    c = int(noisy01.shape[1])
    if c != 1:
        print(f"[warn] Expected grayscale (C=1), got C={c}. Using first channel only.")
        noisy01 = noisy01[:, :1]
        c = 1

    m1 = N2D_DenoisingNetwork(c, hidden_channels=chan_embed).to(device)
    opt1 = torch.optim.Adam(m1.parameters(), lr=lr)
    m1.train()
    for _ in tqdm(range(int(iters)), desc="N2D stage1", dynamic_ncols=True, leave=False):
        opt1.zero_grad(set_to_none=True)
        loss = n2d_loss_stage1(noisy01, m1)
        loss.backward()
        opt1.step()

    with torch.no_grad():
        den1 = torch.clamp(noisy01 - m1(noisy01), 0.0, 1.0)

        input_2x = pixel_unshuffle(noisy01, 2)
        pseudo1 = torch.clamp(input_2x - m1(input_2x), 0.0, 1.0)
        pseudo1 = pixel_shuffle(pseudo1, 2)
        pseudo1 = torch.clamp(pseudo1 - m1(pseudo1), 0.0, 1.0)

        input_4x = pixel_unshuffle(noisy01, 4)
        pseudo2 = torch.clamp(input_4x - m1(input_4x), 0.0, 1.0)
        pseudo2 = pixel_shuffle(pseudo2, 4)
        pseudo2 = torch.clamp(pseudo2 - m1(pseudo2), 0.0, 1.0)

        den2 = (den1 + pseudo1 + pseudo2) / 3.0

    m2 = N2D_DenoisingNetwork(c, hidden_channels=chan_embed).to(device)
    opt2 = torch.optim.Adam(m2.parameters(), lr=lr)
    m2.train()
    for _ in tqdm(range(int(iters)), desc="N2D stage3", dynamic_ncols=True, leave=False):
        opt2.zero_grad(set_to_none=True)
        loss = n2d_loss_stage3(den2, noisy01, m2)
        loss.backward()
        opt2.step()

    with torch.no_grad():
        den3 = torch.clamp(m2(den2), 0.0, 1.0)
    return den3.detach()
