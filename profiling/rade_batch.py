"""Visualize rasterization outputs as images.

This script loads the rasterization outputs and visualizes them as images:
- RGB images
- Expected depths (colormap)
- Median depths (colormap)
- Expected normals (RGB representation)
"""

import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from gsplat._helper import load_test_data
from gsplat.rendering import rasterization

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def visualize_outputs():
    # Load test data from assets/test_garden.npz
    # Use absolute path to avoid issues when running from profiling directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/test_garden.npz")
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
    ) = load_test_data(data_path=data_path, device=device, scene_grid=1)

    print(f"Loaded test data:")
    print(f"  means shape: {means.shape}")
    print(f"  quats shape: {quats.shape}")
    print(f"  scales shape: {scales.shape}")
    print(f"  opacities shape: {opacities.shape}")
    print(f"  colors shape: {colors.shape}")
    print(f"  viewmats shape: {viewmats.shape}")
    print(f"  Ks shape: {Ks.shape}")
    print(f"  width: {width}, height: {height}")

    # Set parameters: batch=4, cameras=3
    n_batches = 4
    n_cameras = 3

    # Prepare tensors for batch processing
    tensors = [means, quats, scales, opacities, colors]
    for i, tensor in enumerate(tensors):
        tensor = tensor[:1000]  # Use first 1000 gaussians for testing
        tensor = torch.broadcast_to(tensor, (n_batches, *tensor.shape)).contiguous()
        tensor.requires_grad = True
        tensors[i] = tensor
    means, quats, scales, opacities, colors = tensors

    # Prepare viewmats and Ks for multiple cameras
    # Use all 3 viewmats from load_test_data for the 3 cameras
    viewmats = torch.broadcast_to(
        viewmats[None, :n_cameras, ...], (n_batches, n_cameras, *viewmats.shape[1:])
    ).clone()
    Ks = torch.broadcast_to(
        Ks[None, :n_cameras, ...], (n_batches, n_cameras, *Ks.shape[1:])
    ).clone()

    print(f"\nPrepared tensors for batch processing:")
    print(f"  means shape: {means.shape}")
    print(f"  quats shape: {quats.shape}")
    print(f"  scales shape: {scales.shape}")
    print(f"  opacities shape: {opacities.shape}")
    print(f"  colors shape: {colors.shape}")
    print(f"  viewmats shape: {viewmats.shape}")
    print(f"  Ks shape: {Ks.shape}")

    # Set render resolution
    render_width, render_height = 640, 360  # 360p resolution
    Ks[..., 0, :] *= render_width / width
    Ks[..., 1, :] *= render_height / height

    print(f"\nRendering with:")
    print(f"  batch size: {n_batches}")
    print(f"  cameras: {n_cameras}")
    print(f"  resolution: {render_width}x{render_height}")

    # Run rasterization
    try:
        (
            render_colors,
            render_alphas,
            expected_depths,
            median_depths,
            expected_normals,
            meta,
        ) = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, K, 3]
            viewmats,  # [C, 4, 4]
            Ks,  # [C, 3, 3]
            render_width,
            render_height,
            packed=False,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=3.0,
            distributed=False,
            with_ut=False,
            with_eval3d=False,
        )

        print(f"\nRasterization successful!")
        print(f"  render_colors shape: {render_colors.shape}")
        print(f"  render_alphas shape: {render_alphas.shape}")
        print(f"  expected_depths shape: {expected_depths.shape}")
        print(f"  median_depths shape: {median_depths.shape}")
        print(f"  expected_normals shape: {expected_normals.shape}")

        # Extract RGB from render_colors (first 3 channels)
        rgb = render_colors[..., :3]

        print(f"\nExtracted outputs:")
        print(f"  rgb shape: {rgb.shape}")
        print(f"  expected_depths shape: {expected_depths.shape}")
        print(f"  median_depths shape: {median_depths.shape}")
        print(f"  expected_normals shape: {expected_normals.shape}")

        # Create output directory for images
        output_dir = Path("output/rade_batch_test_outputs")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Convert tensors to numpy for visualization
        rgb_np = rgb.cpu().detach().numpy()
        expected_depths_np = expected_depths.cpu().detach().numpy()
        median_depths_np = median_depths.cpu().detach().numpy()
        expected_normals_np = expected_normals.cpu().detach().numpy()

        print(f"\nVisualizing outputs for {n_batches} batches and {n_cameras} cameras...")

        # Visualize each batch - create one PNG per batch showing all cameras
        for batch_idx in range(n_batches):
            print(f"\nProcessing batch {batch_idx + 1}/{n_batches}...")

            # Create figure for this batch - layout: cameras × modalities
            fig = plt.figure(figsize=(6 * n_cameras, 4 * 4))  # width: 6 per camera, height: 4 per modality

            # For each camera in this batch
            for cam_idx in range(n_cameras):
                # Get data for this batch and camera
                rgb_img = np.clip(rgb_np[batch_idx, cam_idx], 0, 1)
                expected_depth_img = expected_depths_np[batch_idx, cam_idx, ..., 0]
                median_depth_img = median_depths_np[batch_idx, cam_idx, ..., 0]
                normals_img = (expected_normals_np[batch_idx, cam_idx] + 1) / 2  # Convert from [-1,1] to [0,1]
                normals_img = np.clip(normals_img, 0, 1)

                # Calculate depth ranges for consistent colormap scaling
                valid_mask_expected = expected_depth_img > 1e-6
                valid_mask_median = median_depth_img > 1e-6

                if np.any(valid_mask_expected):
                    vmin_expected = np.percentile(expected_depth_img[valid_mask_expected], 5)
                    vmax_expected = np.percentile(expected_depth_img[valid_mask_expected], 95)
                else:
                    vmin_expected, vmax_expected = 0, 1

                if np.any(valid_mask_median):
                    vmin_median = np.percentile(median_depth_img[valid_mask_median], 5)
                    vmax_median = np.percentile(median_depth_img[valid_mask_median], 95)
                else:
                    vmin_median, vmax_median = 0, 1

                # Row 1: RGB
                ax1 = plt.subplot(4, n_cameras, cam_idx + 1)
                plt.imshow(rgb_img)
                plt.title(f"Camera {cam_idx}: RGB")
                plt.axis('off')

                # Row 2: Expected Depth
                ax2 = plt.subplot(4, n_cameras, cam_idx + 1 + n_cameras)
                im2 = plt.imshow(expected_depth_img, cmap='viridis', vmin=vmin_expected, vmax=vmax_expected)
                plt.colorbar(im2, fraction=0.046, pad=0.04)
                plt.title(f"Camera {cam_idx}: Expected Depth")
                plt.axis('off')

                # Row 3: Median Depth
                ax3 = plt.subplot(4, n_cameras, cam_idx + 1 + 2 * n_cameras)
                im3 = plt.imshow(median_depth_img, cmap='plasma', vmin=vmin_median, vmax=vmax_median)
                plt.colorbar(im3, fraction=0.046, pad=0.04)
                plt.title(f"Camera {cam_idx}: Median Depth")
                plt.axis('off')

                # Row 4: Expected Normals
                ax4 = plt.subplot(4, n_cameras, cam_idx + 1 + 3 * n_cameras)
                plt.imshow(normals_img)
                plt.title(f"Camera {cam_idx}: Expected Normals")
                plt.axis('off')

            plt.suptitle(f"Batch {batch_idx} - All {n_cameras} Cameras", fontsize=16, y=0.98)
            plt.tight_layout()
            plt.savefig(output_dir / f"batch{batch_idx}.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved batch {batch_idx} visualization to {output_dir}/batch{batch_idx}.png")

        print(f"\n✅ All batch visualizations saved to {output_dir}/")
        print(f"Each batch has one PNG file showing all cameras and modalities.")

        # Print statistics
        print(f"\nOutput statistics:")
        print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"  Expected depths range: [{expected_depths.min():.3f}, {expected_depths.max():.3f}]")
        print(f"  Median depths range: [{median_depths.min():.3f}, {median_depths.max():.3f}]")
        print(f"  Expected normals range: [{expected_normals.min():.3f}, {expected_normals.max():.3f}]")

        # Test gradient propagation
        print(f"\nTesting gradient propagation...")

        # Create a dummy loss function that uses all outputs
        dummy_loss = (
            render_colors.sum() * 0.1 +
            render_alphas.sum() * 0.2 +
            expected_depths.sum() * 0.3 +
            median_depths.sum() * 0.4 +
            expected_normals.sum() * 0.5
        )

        # Compute gradients
        dummy_loss.backward()

        # Check gradients for each input tensor
        print(f"\nGradient statistics:")
        print(f"  means gradients: {means.grad is not None}, shape: {means.grad.shape if means.grad is not None else 'None'}")
        print(f"  quats gradients: {quats.grad is not None}, shape: {quats.grad.shape if quats.grad is not None else 'None'}")
        print(f"  scales gradients: {scales.grad is not None}, shape: {scales.grad.shape if scales.grad is not None else 'None'}")
        print(f"  opacities gradients: {opacities.grad is not None}, shape: {opacities.grad.shape if opacities.grad is not None else 'None'}")
        print(f"  colors gradients: {colors.grad is not None}, shape: {colors.grad.shape if colors.grad is not None else 'None'}")

        # Check if gradients are non-zero (indicating successful backprop)
        # Move gradients to CPU to avoid CUDA memory access issues
        if means.grad is not None:
            means_grad_cpu = means.grad.cpu()
            print(f"  means grad range: [{means_grad_cpu.min():.6f}, {means_grad_cpu.max():.6f}]")
        if quats.grad is not None:
            quats_grad_cpu = quats.grad.cpu()
            print(f"  quats grad range: [{quats_grad_cpu.min():.6f}, {quats_grad_cpu.max():.6f}]")
        if scales.grad is not None:
            scales_grad_cpu = scales.grad.cpu()
            print(f"  scales grad range: [{scales_grad_cpu.min():.6f}, {scales_grad_cpu.max():.6f}]")
        if opacities.grad is not None:
            opacities_grad_cpu = opacities.grad.cpu()
            print(f"  opacities grad range: [{opacities_grad_cpu.min():.6f}, {opacities_grad_cpu.max():.6f}]")
        if colors.grad is not None:
            colors_grad_cpu = colors.grad.cpu()
            print(f"  colors grad range: [{colors_grad_cpu.min():.6f}, {colors_grad_cpu.max():.6f}]")

        # Verify gradient propagation across batches
        print(f"\nBatch gradient verification:")
        for batch_idx in range(n_batches):
            if means.grad is not None:
                batch_grad_norm = means.grad[batch_idx].cpu().norm().item()
                print(f"  Batch {batch_idx}: means grad norm = {batch_grad_norm:.6f}")

        # Verify that all batches have non-zero gradients (indicating proper batch processing)
        if means.grad is not None:
            batch_grad_norms = [means.grad[batch_idx].cpu().norm().item() for batch_idx in range(n_batches)]
            print(f"\nBatch gradient norms: {batch_grad_norms}")
            print(f"  Mean batch grad norm: {np.mean(batch_grad_norms):.6f}")
            print(f"  Std batch grad norm: {np.std(batch_grad_norms):.6f}")

        print(f"\n✅ Gradient propagation test completed!")

        return True

    except Exception as e:
        print(f"\nError during rasterization or visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = visualize_outputs()
    if success:
        print("\n✅ Visualization completed successfully!")
    else:
        print("\n❌ Visualization failed!")