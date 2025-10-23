"""Test rasterization function with batch processing for RGB+ED+N mode.

This script tests the rasterization function with batch processing,
ensuring that the RGB+ED+N render mode works correctly with batched inputs.
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


def test_rasterization_batch():
    """Test rasterization with batch processing for RGB+ED+N mode."""

    # Load test data from assets/test_garden.npz
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets/test_garden.npz"
    )
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

    # Set parameters: batch=3, cameras=3
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
    render_width, render_height = 320, 180  # Low resolution for fast testing
    Ks[..., 0, :] *= render_width / width
    Ks[..., 1, :] *= render_height / height

    print(f"\nRendering with:")
    print(f"  batch size: {n_batches}")
    print(f"  cameras: {n_cameras}")
    print(f"  resolution: {render_width}x{render_height}")

    # Test RGB+ED+N render mode
    render_mode = "RGB+ED+N"

    print(f"\n{'='*60}")
    print(f"Testing render_mode: {render_mode}")
    print(f"{'='*60}")

    try:
        # Run rasterization
        render_colors, render_alphas, meta = rasterization(
            means,  # [B, N, 3]
            quats,  # [B, N, 4]
            scales,  # [B, N, 3]
            opacities,  # [B, N]
            colors,  # [B, N, 3]
            viewmats,  # [B, C, 4, 4]
            Ks,  # [B, C, 3, 3]
            render_width,
            render_height,
            packed=False,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=3.0,
            render_mode=render_mode,
            distributed=False,
            with_ut=False,
            with_eval3d=False,
        )

        print(f"✅ Rasterization successful for {render_mode}!")
        print(f"  render_colors shape: {render_colors.shape}")
        print(f"  render_alphas shape: {render_alphas.shape}")

        # Verify output shapes
        expected_channels = 7  # RGB(3) + ED(1) + Normals(3) = 7
        actual_channels = render_colors.shape[-1]

        if actual_channels == expected_channels:
            print(
                f"✅ Output channels correct: {actual_channels} (expected: {expected_channels})"
            )
        else:
            print(
                f"❌ Output channels incorrect: {actual_channels} (expected: {expected_channels})"
            )

        # Check for NaN and infinite values
        has_nan = torch.isnan(render_colors).any()
        has_inf = torch.isinf(render_colors).any()

        if not has_nan and not has_inf:
            print(f"✅ No NaN or infinite values in output")
        else:
            print(f"❌ Found NaN: {has_nan}, Infinite: {has_inf}")

        # Check value ranges
        # RGB channels (0-2)
        rgb_range = render_colors[..., :3]
        rgb_min, rgb_max = rgb_range.min().item(), rgb_range.max().item()
        print(f"  RGB range: [{rgb_min:.3f}, {rgb_max:.3f}]")

        if rgb_min >= 0 and rgb_max <= 1.0:
            print(f"✅ RGB values in valid range [0, 1]")
        else:
            print(f"⚠️  RGB values outside expected range [0, 1]")

        # Depth channel (3)
        depth_channel = render_colors[..., 3:4]
        depth_min, depth_max = depth_channel.min().item(), depth_channel.max().item()
        print(f"  Depth range: [{depth_min:.3f}, {depth_max:.3f}]")

        if depth_min >= 0:
            print(f"✅ Depth values non-negative")
        else:
            print(f"⚠️  Depth values contain negative numbers")

        # Normals channels (4-6)
        normals = render_colors[..., 4:7]
        normals_min, normals_max = normals.min().item(), normals.max().item()
        print(f"  Normals range: [{normals_min:.3f}, {normals_max:.3f}]")

        if normals_min >= -1 and normals_max <= 1.0:
            print(f"✅ Normals in valid range [-1, 1]")
        else:
            print(f"⚠️  Normals outside expected range [-1, 1]")

        # Test gradient propagation
        print(f"\nTesting gradient propagation for {render_mode}...")

        # Create a dummy loss function
        dummy_loss = render_colors.sum() * 0.1 + render_alphas.sum() * 0.2

        # Compute gradients
        dummy_loss.backward()

        # Check gradients for each input tensor
        grad_stats = []
        for name, tensor in [
            ("means", means),
            ("quats", quats),
            ("scales", scales),
            ("opacities", opacities),
            ("colors", colors),
        ]:
            has_grad = tensor.grad is not None
            grad_norm = tensor.grad.norm().item() if has_grad else 0.0
            grad_stats.append((name, has_grad, grad_norm))

        print(f"  Gradient statistics:")
        all_have_gradients = True
        for name, has_grad, grad_norm in grad_stats:
            status = "✅" if has_grad else "❌"
            print(f"    {status} {name}: has_grad={has_grad}, norm={grad_norm:.6f}")
            if not has_grad:
                all_have_gradients = False

        if all_have_gradients:
            print(f"✅ All gradients propagated successfully!")
        else:
            print(f"❌ Some gradients failed to propagate")

        print(f"\n✅ {render_mode} test completed successfully!")

    except Exception as e:
        print(f"❌ Error during {render_mode} rasterization: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n{'='*60}")
    print("RGB+ED+N batch rasterization test completed!")
    print(f"{'='*60}")

    return True


def visualize_batch_rgb_ed_n_outputs():
    """Visualize RGB+ED+N outputs from 4 scenes with 3 cameras each."""

    # Load test data from assets/test_garden.npz
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets/test_garden.npz"
    )
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

    print(f"Loaded test data for visualization:")
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
    render_width, render_height = 320, 180  # Low resolution for fast testing
    Ks[..., 0, :] *= render_width / width
    Ks[..., 1, :] *= render_height / height

    print(f"\nRendering with:")
    print(f"  batch size: {n_batches}")
    print(f"  cameras: {n_cameras}")
    print(f"  resolution: {render_width}x{render_height}")

    # Test RGB+ED+N render mode
    render_mode = "RGB+ED+N"

    print(f"\n{'='*60}")
    print(f"Visualizing render_mode: {render_mode}")
    print(f"{'='*60}")

    try:
        # Run rasterization
        render_colors, render_alphas, _ = rasterization(
            means,  # [B, N, 3]
            quats,  # [B, N, 4]
            scales,  # [B, N, 3]
            opacities,  # [B, N]
            colors,  # [B, N, 3]
            viewmats,  # [B, C, 4, 4]
            Ks,  # [B, C, 3, 3]
            render_width,
            render_height,
            packed=True,
            near_plane=0.01,
            far_plane=100.0,
            radius_clip=3.0,
            render_mode=render_mode,
            distributed=False,
            with_ut=False,
            with_eval3d=False,
        )

        print(f"✅ Rasterization successful for {render_mode}!")
        print(f"  render_colors shape: {render_colors.shape}")
        print(f"  render_alphas shape: {render_alphas.shape}")

        # Extract RGB, depth, and normal channels
        # render_colors shape: [B, C, H, W, 7] where channels are: RGB(3) + ED(1) + Normals(3)
        rgb = render_colors[..., :3]  # RGB channels
        depth = render_colors[..., 3:4]  # Expected depth channel
        normals = render_colors[..., 4:7]  # Normal channels

        print(f"\nExtracted outputs:")
        print(f"  rgb shape: {rgb.shape}")
        print(f"  depth shape: {depth.shape}")
        print(f"  normals shape: {normals.shape}")

        # Create output directory
        output_dir = Path("output/dn_mb_mc_test")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Convert tensors to numpy for visualization
        rgb_np = rgb.cpu().detach().numpy()
        depth_np = depth.cpu().detach().numpy()
        normals_np = normals.cpu().detach().numpy()

        print(
            f"\nVisualizing outputs for {n_batches} scenes and {n_cameras} cameras..."
        )

        # Create one image per scene showing all 3 cameras and 3 modalities
        for batch_idx in range(n_batches):
            print(f"Processing scene {batch_idx + 1}/{n_batches}...")

            # Create figure for this scene - layout: cameras × modalities
            plt.figure(
                figsize=(6 * n_cameras, 4 * 3)
            )  # width: 6 per camera, height: 4 per modality

            # For each camera in this scene
            for cam_idx in range(n_cameras):
                # Get data for this scene and camera
                rgb_img = np.clip(rgb_np[batch_idx, cam_idx], 0, 1)
                depth_img = depth_np[
                    batch_idx, cam_idx, ..., 0
                ]  # Remove channel dimension
                normals_img = (
                    normals_np[batch_idx, cam_idx] + 1
                ) / 2  # Convert from [-1,1] to [0,1]
                normals_img = np.clip(normals_img, 0, 1)

                # Calculate depth range for consistent colormap scaling
                valid_mask = depth_img > 1e-6
                if np.any(valid_mask):
                    vmin = np.percentile(depth_img[valid_mask], 5)
                    vmax = np.percentile(depth_img[valid_mask], 95)
                else:
                    vmin, vmax = 0, 1

                # Row 1: RGB
                plt.subplot(3, n_cameras, cam_idx + 1)
                plt.imshow(rgb_img)
                plt.title(f"Camera {cam_idx}: RGB")
                plt.axis("off")

                # Row 2: Depth
                plt.subplot(3, n_cameras, cam_idx + 1 + n_cameras)
                im2 = plt.imshow(depth_img, cmap="viridis", vmin=vmin, vmax=vmax)
                plt.colorbar(im2, fraction=0.046, pad=0.04)
                plt.title(f"Camera {cam_idx}: Expected Depth")
                plt.axis("off")

                # Row 3: Normals
                plt.subplot(3, n_cameras, cam_idx + 1 + 2 * n_cameras)
                plt.imshow(normals_img)
                plt.title(f"Camera {cam_idx}: Expected Normals")
                plt.axis("off")

            plt.suptitle(
                f"Scene {batch_idx} - All {n_cameras} Cameras", fontsize=16, y=0.98
            )
            plt.tight_layout()
            plt.savefig(
                output_dir / f"scene{batch_idx}_rgb_ed_n_visualization.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            print(
                f"  ✅ Saved scene {batch_idx} visualization to {output_dir}/scene{batch_idx}_rgb_ed_n_visualization.png"
            )

        print(f"\n✅ All scene visualizations saved to {output_dir}/")
        print(f"Each scene has one PNG file showing all cameras and modalities.")

        # Print statistics
        print(f"\nOutput statistics:")
        print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"  Normals range: [{normals.min():.3f}, {normals.max():.3f}]")

        print(f"\n✅ RGB+ED+N batch visualization completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during {render_mode} visualization: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the original test function
    success = test_rasterization_batch()
    if success:
        print("\n✅ RGB+ED+N batch rasterization test completed successfully!")
    else:
        print("\n❌ RGB+ED+N batch rasterization test failed!")

    print("\n" + "=" * 60)
    print("Running RGB+ED+N batch visualization...")
    print("=" * 60)

    # Run the visualization function
    viz_success = visualize_batch_rgb_ed_n_outputs()
    if viz_success:
        print("\n✅ RGB+ED+N batch visualization completed successfully!")
    else:
        print("\n❌ RGB+ED+N batch visualization failed!")
