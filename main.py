from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn.functional as F

from config import Config
from inference import infer_flow, infer_flow_tensor
from loss import weighted_dir_loss, rms_per_sample
from model import UNet
from n2d import noise2detail_stage3_anchor
from training import sample_patches
from utils import set_seed, to_tensor, to_numpy, create_placeholder_image


def denoise(
    noisy: np.ndarray,
    cfg: Config | None = None,
    clean: np.ndarray | None = None,
) -> dict:
    """
    Denoise a single image using Noise2Flow.

    Args:
        noisy: Noisy input image, shape (H, W), values in [0, 1]
        cfg: Configuration object. If None, uses defaults.
        clean: Optional clean ground truth for evaluation.

    Returns:
        Dictionary containing:
            - 'denoised': Denoised image
            - 'anchor_n2d': N2D anchor image
            - 'metrics': PSNR/SSIM if clean provided
    """
    if cfg is None:
        cfg = Config()

    set_seed(cfg.runtime.seed)

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    y = to_tensor(noisy, device)

    # Stage 1: Generate anchor from weak denoiser
    print("[info] Noise2Detail -> anchor ...")
    anchor_n2d = noise2detail_stage3_anchor(
        y,
        chan_embed=cfg.n2d.chan_embed,
        iters=cfg.n2d.iters,
        lr=cfg.n2d.lr,
    )

    # Stage 2: Train flow network
    print("[info] Training flow network ...")
    net = UNet(
        in_ch=2,
        out_ch=1,
        base=cfg.flow_model.base_channels,
        bilinear=True,
    ).to(device)

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.flow_train.lr,
        weight_decay=cfg.flow_train.weight_decay,
    )

    rng = np.random.default_rng(cfg.runtime.seed)
    pbar = tqdm(range(cfg.flow_train.steps), dynamic_ncols=True)

    for step in pbar:
        net.train()
        y_p, w_p = sample_patches(
            rng=rng,
            y_full=y,
            anchor=anchor_n2d,
            patch=cfg.flow_train.patch,
            batch=cfg.flow_train.batch,
            device=device,
        )

        d = w_p - y_p

        # λ ~ U[0, 1] as in paper Eq. 6
        lam = torch.empty((cfg.flow_train.batch, 1, 1, 1), device=device).uniform_(0.0, 1.0)

        d_rms = rms_per_sample(d)
        sigma = cfg.flow_train.rho * d_rms.mean().item()
        sigma = min(sigma, cfg.flow_train.sigma_cap)
        eps = torch.randn_like(y_p) * sigma

        x_in = (1.0 - lam) * y_p + lam * w_p + eps
        x_in = torch.clamp(x_in, 0.0, 1.0)

        v_raw = net(torch.cat([x_in, y_p], dim=1))
        v = torch.tanh(v_raw)

        loss = weighted_dir_loss(v, d)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.flow_train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.flow_train.grad_clip)
        opt.step()

        if (step + 1) % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    # Stage 3: Learn optimal strength B (align with N2D anchor)
    print("[info] Learning optimal strength B ...")
    # Freeze flow network parameters
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    target_anchor = anchor_n2d

    # Create learnable strength parameter B
    B = torch.nn.Parameter(
        torch.tensor(cfg.strength_learn.init_strength, device=device)
    )
    opt_B = torch.optim.Adam([B], lr=cfg.strength_learn.lr)

    pbar_B = tqdm(range(cfg.strength_learn.steps), dynamic_ncols=True, desc="Learn B")
    for step in pbar_B:
        opt_B.zero_grad()
        # Forward pass with current strength B
        denoised_tensor = infer_flow_tensor(
            net=net,
            y=y,
            strength=B,
            mode="onestep",
            clamp01=True,
        )
        # MSE loss to align with better anchor
        loss_B = F.mse_loss(denoised_tensor, target_anchor)
        loss_B.backward()
        opt_B.step()

        # Clamp B to be positive
        with torch.no_grad():
            B.data.clamp_(min=0.01)

        if (step + 1) % 50 == 0:
            pbar_B.set_postfix(B=f"{B.item():.4f}", loss=f"{loss_B.item():.6f}")

    learned_B = B.item()
    print(f"[info] Learned strength B = {learned_B:.4f}")

    # Stage 4: Inference with B * multiplier
    infer_strength = learned_B * cfg.strength_learn.infer_multiplier
    print(f"[info] Inference with strength = {learned_B:.4f} * {cfg.strength_learn.infer_multiplier} = {infer_strength:.4f}")
    denoised = infer_flow(net=net, y=y, strength=infer_strength, infer_cfg=cfg.flow_infer)

    result = {
        'denoised': denoised,
        'anchor_n2d': to_numpy(anchor_n2d),
        'learned_B': learned_B,
        'infer_strength': infer_strength,
    }

    # Compute metrics if clean image provided
    if clean is not None:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        result['metrics'] = {
            'noisy_psnr': peak_signal_noise_ratio(clean, noisy, data_range=1.0),
            'noisy_ssim': structural_similarity(clean, noisy, data_range=1.0),
            'denoised_psnr': peak_signal_noise_ratio(clean, denoised, data_range=1.0),
            'denoised_ssim': structural_similarity(clean, denoised, data_range=1.0),
            'n2d_psnr': peak_signal_noise_ratio(clean, result['anchor_n2d'], data_range=1.0),
            'n2d_ssim': structural_similarity(clean, result['anchor_n2d'], data_range=1.0),
        }
        print(f"  Noisy      PSNR: {result['metrics']['noisy_psnr']:.2f} dB, SSIM: {result['metrics']['noisy_ssim']:.4f}")
        print(f"  N2D Anchor PSNR: {result['metrics']['n2d_psnr']:.2f} dB, SSIM: {result['metrics']['n2d_ssim']:.4f}")
        print(f"  Noise2Flow PSNR: {result['metrics']['denoised_psnr']:.2f} dB, SSIM: {result['metrics']['denoised_ssim']:.4f}")

    return result


def save_comparison_matrix(
    noisy: np.ndarray,
    anchor_n2d: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray | None = None,
    metrics: dict | None = None,
    save_path: str = "comparison_matrix.png",
    show: bool = True,
) -> None:
    """
    Save and display comparison matrix: noisy vs N2D anchor vs noise2flow.

    Args:
        noisy: Noisy input image
        anchor_n2d: N2D denoised anchor
        denoised: Noise2Flow denoised result
        clean: Optional clean ground truth image
        metrics: Optional dict with PSNR values (noisy_psnr, n2d_psnr, denoised_psnr)
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    if clean is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        images = [noisy, anchor_n2d, denoised, clean]
        if metrics is not None:
            titles = [
                f"Noisy\nPSNR: {metrics['noisy_psnr']:.2f} dB\nSSIM: {metrics['noisy_ssim']:.4f}",
                f"Anchor (N2D)\nPSNR: {metrics['n2d_psnr']:.2f} dB\nSSIM: {metrics['n2d_ssim']:.4f}",
                f"Noise2Flow\nPSNR: {metrics['denoised_psnr']:.2f} dB\nSSIM: {metrics['denoised_ssim']:.4f}",
                "Clean (GT)"
            ]
        else:
            titles = ['Noisy', 'Anchor (N2D)', 'Noise2Flow', 'Clean (GT)']
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        images = [noisy, anchor_n2d, denoised]
        titles = ['Noisy', 'Anchor (N2D)', 'Noise2Flow']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[info] Comparison matrix saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Demo with placeholder image."""
    print("[info] Creating placeholder test image ...")
    noisy, clean = create_placeholder_image(
        height=256, width=256,
        spatial_noise_level=0.05,  # spatially correlated noise
        direct_noise_level=0.05   # direct Gaussian noise
    )

    cfg = Config()
    cfg.flow_train.steps = 1000  # Reduce for demo

    result = denoise(noisy, cfg, clean)

    print("[done] Denoising complete.")
    print(f"  Learned B: {result['learned_B']:.4f}, Infer strength: {result['infer_strength']:.4f}")
    if 'metrics' in result:
        print(f"  PSNR improvement: {result['metrics']['denoised_psnr'] - result['metrics']['noisy_psnr']:.2f} dB")

    # Save comparison matrix: noisy vs N2D anchor vs noise2flow
    save_comparison_matrix(
        noisy=noisy,
        anchor_n2d=result['anchor_n2d'],
        denoised=result['denoised'],
        clean=clean,
        metrics=result.get('metrics'),
        save_path="comparison_matrix.png",
        show=True,
    )


if __name__ == "__main__":
    main()
