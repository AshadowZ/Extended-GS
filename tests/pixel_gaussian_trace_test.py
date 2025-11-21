"""
Test script for validating pixel-to-Gaussian tracking.

This script mirrors the scene / camera preparation used in
``profiling/single_dn_test.py`` but enables ``track_pixel_gaussians`` to ensure
that the CUDA logic records meaningful (gaussian_id, pixel_id) pairs.
"""

from pathlib import Path
import os
import random

import torch

from gsplat._helper import load_test_data
from gsplat.rendering import rasterization


def prepare_scene(device: torch.device):
    """Load the test garden scene and pick a manageable subset."""

    data_path = Path(__file__).resolve().parents[1] / "assets" / "test_garden.npz"

    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(data_path=str(data_path), device=device, scene_grid=1)

    n_gaussians = 110_000
    n_cameras = 3

    means = means[:n_gaussians].clone().contiguous().requires_grad_(True)
    quats = quats[:n_gaussians].clone().contiguous().requires_grad_(True)
    scales = scales[:n_gaussians].clone().contiguous().requires_grad_(True)
    opacities = opacities[:n_gaussians].clone().contiguous().requires_grad_(True)
    colors = colors[:n_gaussians].clone().contiguous().requires_grad_(True)

    viewmats = viewmats[:n_cameras].clone().contiguous()
    Ks = Ks[:n_cameras].clone().contiguous()

    render_width, render_height = 320, 180
    Ks[..., 0, :] *= render_width / width
    Ks[..., 1, :] *= render_height / height

    return (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        render_width,
        render_height,
        n_cameras,
    )


def decode_pixel_id(pixel_id: int, width: int, height: int):
    """Convert flattened pixel id back to (camera_idx, row, col)."""
    pixels_per_image = width * height
    cam_idx = pixel_id // pixels_per_image
    within_image = pixel_id % pixels_per_image
    row = within_image // width
    col = within_image % width
    return cam_idx, row, col


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        render_width,
        render_height,
        n_cameras,
    ) = prepare_scene(device)

    print("Running rasterization with pixel-gaussian tracking...")
    render_colors, render_alphas, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        render_width,
        render_height,
        packed=False,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=3.0,
        render_mode="RGB",
        distributed=False,
        with_ut=False,
        with_eval3d=False,
        track_pixel_gaussians=True,
        max_gaussians_per_pixel=8,
        pixel_gaussian_threshold=0.05,
    )

    pixel_pairs = meta.get("pixel_gaussians", torch.empty(0, 2, device=device))
    print(f"Render colors shape: {render_colors.shape}")
    print(f"Render alphas shape: {render_alphas.shape}")
    print(f"Tracked pairs: {pixel_pairs.shape[0]}")

    if pixel_pairs.numel() == 0:
        raise RuntimeError("No pixel-gaussian pairs recorded. Check tracking logic.")

    N = means.shape[-2]
    flatten_stride = N
    total_pixels = n_cameras * render_width * render_height

    invalid_gaussians = pixel_pairs[:, 0] < 0
    invalid_pixels = (pixel_pairs[:, 1] < 0) | (pixel_pairs[:, 1] >= total_pixels)

    assert not invalid_gaussians.any(), "Found negative gaussian ids."
    assert not invalid_pixels.any(), "Found pixel ids outside valid range."

    print("Sampled pixel-gaussian pairs:")
    pairs_cpu = pixel_pairs.cpu()
    for idx in random.sample(range(len(pairs_cpu)), k=min(5, len(pairs_cpu))):
        g_id, pix_id = pairs_cpu[idx].tolist()
        cam_idx_px, row, col = decode_pixel_id(int(pix_id), render_width, render_height)
        view_idx = g_id // flatten_stride
        gaussian_idx = g_id % flatten_stride
        print(
            "  Gaussian {:>6} (global id {}) contributes to "
            "camera {} (pixel cam {} row {} col {})".format(
                gaussian_idx, g_id, view_idx, cam_idx_px, row, col
            )
        )

    print("All checks passed. Pixel-to-gaussian tracking works on the sample scene.")


if __name__ == "__main__":
    main()
