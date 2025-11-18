import argparse
import math
import torch
import torch.nn.functional as F
from typing import List
import tqdm  # For progress bars
import os
import sys  # Import sys
from pathlib import Path  # Import Path
import numpy as np
import open3d as o3d
import cv2
import copy  # <-- Added import for post-processing
import shutil

# --- Add parent directory to Python path ---
# Get the directory of the current script
script_dir = Path(__file__).resolve().parent
# Get the parent directory
parent_dir = script_dir.parent
# Add the parent directory to sys.path
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# -------------------------------------

# --- Assume these imports are from your project structure ---
# Ensure the 'datasets' directory is in your PYTHONPATH, or in the same directory as this script
try:
    from datasets.colmap import Dataset, Parser
except ImportError:
    print(
        f"Error: Failed to import 'datasets.colmap.Dataset' or 'datasets.colmap.Parser' from '{parent_dir}'."
    )
    print(
        "Please ensure the 'datasets' directory is in the parent directory of your script."
    )
    print(f"Current Python path: {sys.path}")
    exit(1)
# --- Requires gsplat library ---
try:
    from gsplat.rendering import rasterization, rasterization_2dgs
except ImportError:
    print("Error: Failed to import required gsplat rendering functions.")
    print("Please ensure 'gsplat' library is installed (e.g., pip install gsplat)")
    exit(1)


def load_params_from_ckpt(ckpt_paths: List[str], device: torch.device):
    """
    Load Gaussian parameters from one or more checkpoint files.

    Args:
        ckpt_paths: List of paths to checkpoint files.
        device: Target device to load tensors onto (e.g., 'cuda', 'cpu').

    Returns:
        A tuple containing the loaded parameters:
        (means, quats, scales, opacities, colors, sh_degree)
    """
    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []

    print(f"Loading from {len(ckpt_paths)} files to {device}...")

    for ckpt_path in ckpt_paths:
        try:
            # Load checkpoint, assuming the 'splats' key contains parameters
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]

            # Extract, process, and add to lists
            means.append(ckpt["means"])
            quats.append(
                F.normalize(ckpt["quats"], p=2, dim=-1)
            )  # Normalize quaternions
            scales.append(
                torch.exp(ckpt["scales"])
            )  # Convert log scales to real scales
            opacities.append(
                torch.sigmoid(ckpt["opacities"])
            )  # Convert logits to [0, 1] range
            sh0.append(ckpt["sh0"])  # SH 0th order (DC)
            shN.append(ckpt["shN"])  # SH higher orders (AC)

            print(f"  - Successfully loaded: {ckpt_path}")

        except Exception as e:
            print(f"  - Failed to load: {ckpt_path} (Error: {e})")
            continue

    if not means:
        print("Failed to load any valid Gaussian parameters.")
        return None, None, None, None, None, -1

    # Concatenate tensors from lists into one large tensor
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)

    # Combine SH coefficients
    # colors tensor shape will be [N, S, 3], where S = (sh_degree + 1)^2
    colors = torch.cat([sh0, shN], dim=-2)

    # Calculate SH degree
    # S = (sh_degree + 1)^2  =>  sh_degree = sqrt(S) - 1
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    print("--- Gaussian parameters loaded successfully ---")
    print(f"Total Gaussians: {means.shape[0]}")
    print(f"SH degree (sh_degree): {sh_degree}")
    print("--------------------------")

    return means, quats, scales, opacities, colors, sh_degree


# --- Helper functions added from reference script ---


def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert torch.Tensor to NumPy array; return directly if already NumPy array.
    Automatically handles device (CPU/CUDA) and gradients (detach).
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


class DiskNpyArray:
    """
    Lightweight wrapper that exposes a list of .npy files like an array,
    loading data on-demand to avoid holding everything in memory.
    """

    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return np.load(self.paths[idx])


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist

    # Ensure poses are numpy arrays
    poses_np = [to_numpy(p) for p in poses]
    return np.median(pdist([p[:3, 3] for p in poses_np]))


def create_tsdf_mesh(
    depths_np,
    colors_np,
    ixts,
    camtoworlds,
    hws,
    voxel_length=None,
    sdf_trunc=None,
    depth_trunc=None,  # <-- ADDED
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
):
    """
    Modified TSDF reconstruction function
    depths_np: (N, H, W) float numpy depth maps (unit: meters)
    colors_np: (N, H, W, 3) float numpy RGB images, range [0, 1]
    ixts: (N, 3, 3) intrinsic matrices (numpy)
    camtoworlds: (N, 4, 4) camera-to-world 4x4 transformation matrices (numpy)
    hws: (N, 2) list/array of (h,w) for each frame
    voxel_length: tsdf voxel size (meters), auto-estimated if None
    sdf_trunc: truncation distance (meters), uses 3 * voxel_length if None
    color_type: open3d.integration.TSDFVolumeColorType
    """
    N = len(depths_np)
    assert N == len(colors_np) == len(camtoworlds) == len(ixts) == len(hws)

    # Estimate scene scale (based on camera poses)
    cam_centers = np.asarray([c[0:3, 3] for c in camtoworlds])
    if len(cam_centers) > 1:
        med_dist = float(np.median(get_med_dist_between_poses(camtoworlds)))
    else:
        med_dist = 1.0

    # Default voxel size
    if voxel_length is None:
        voxel_length = med_dist / 192  # reference value
    if sdf_trunc is None:
        sdf_trunc = voxel_length * 5.0

    print(
        f"[TSDF] N={N}, med_camera_dist={med_dist:.3f} m, voxel_length={voxel_length:.6f} m, sdf_trunc={sdf_trunc:.6f} m"
    )
    # --- ADDED ---
    if depth_trunc is not None:
        print(
            f"[TSDF] Applying max depth truncation at {depth_trunc:.3f} m for input images."
        )
    else:
        print("[TSDF] No maximum depth truncation applied for input images.")
    # --- END ADDED ---

    # Use ScalableTSDFVolume (supports large scenes)
    tsdf_vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=sdf_trunc, color_type=color_type
    )

    for i in tqdm.tqdm(range(N), desc="TSDF Integration"):
        depth = depths_np[i].astype(np.float32)  # HxW
        color_data = colors_np[i]
        if color_data.dtype == np.uint8:
            color_float = color_data.astype(np.float32) / 255.0
        else:
            color_float = color_data.astype(np.float32)  # HxW, 3, float [0,1]
        h, w = hws[i]  # colmap's (height, width)

        # Ensure dimensions match
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        if color_float.shape[0] != h or color_float.shape[1] != w:
            color_float = cv2.resize(
                color_float, (w, h), interpolation=cv2.INTER_LINEAR
            )

        # Convert float [0,1] to uint8 [0,255]
        color_uint8 = (color_float * 255).astype(np.uint8)

        # Open3D expects depth as float image
        o3d_depth = o3d.geometry.Image(depth)
        o3d_color = o3d.geometry.Image(color_uint8)

        # --- ADDED ---
        # Use user-specified depth_trunc if provided, otherwise use a large value (pseudo-infinity)
        integration_depth_trunc = depth_trunc if depth_trunc is not None else 10000.0
        # --- END ADDED ---

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,  # since depth is already float in meters, scale=1.0
            depth_trunc=integration_depth_trunc,  # <-- MODIFIED (was 1000.0)
            convert_rgb_to_intensity=False,
        )

        # Construct Open3D camera intrinsic object
        K = ixts[i]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
        )

        # Open3D requires extrinsic as 4x4 world_to_camera matrix
        extrinsic_world_to_cam = np.linalg.inv(camtoworlds[i])

        # integrate
        tsdf_vol.integrate(rgbd, intrinsic, extrinsic_world_to_cam)

    print("[TSDF] Extracting triangle mesh ...")
    mesh = tsdf_vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    return mesh


# --- New: Mesh post-processing function ---
def post_process_mesh(mesh, cluster_to_keep=50):
    """
    Post-process the mesh to filter out floaters and disconnected parts.
    Default is to keep the 50 largest connected components.
    """
    print(
        f"Post-processing mesh to keep the {cluster_to_keep} largest connected components..."
    )
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        (
            triangle_clusters,
            cluster_n_triangles,
            cluster_area,
        ) = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    # cluster_area = np.asarray(cluster_area) # Seems unused

    # If total clusters is less than clusters to keep, set cluster_to_keep to total clusters
    # (prevents np.sort...[-cluster_to_keep] index out of bounds)
    total_clusters = len(cluster_n_triangles)
    if total_clusters == 0:
        print("Warning: No triangles found in mesh.")
        return mesh_0

    actual_cluster_to_keep = min(cluster_to_keep, total_clusters)

    if actual_cluster_to_keep == 0:
        print("Number of clusters to keep is 0, returning original mesh.")
        return mesh_0

    # Find the triangle count of the K-th largest cluster (K = actual_cluster_to_keep)
    n_cluster_threshold = np.sort(cluster_n_triangles.copy())[-actual_cluster_to_keep]

    # Original logic: ensure minimum cluster size is at least 50
    n_cluster_threshold = max(n_cluster_threshold, 50)  # filter meshes smaller than 50

    print(f"  - Total clusters: {total_clusters}")
    print(
        f"  - Will keep: {actual_cluster_to_keep} clusters (or more, if tied in size)"
    )
    print(f"  - Minimum triangle count threshold: {n_cluster_threshold}")

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster_threshold
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()

    print(f"  - Original vertex count: {len(mesh.vertices)}")
    print(f"  - Post-processed vertex count: {len(mesh_0.vertices)}")
    return mesh_0


# --- End of new function ---


def main():
    parser = argparse.ArgumentParser(
        description="Load Gaussian model from CKPT and render the entire dataset."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="Path to one or more .pt checkpoint files.",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the Mip-NeRF 360 dataset."
    )
    parser.add_argument(
        "--data_factor",
        type=int,
        default=1,
        help="Downsampling factor for the dataset.",
    )
    parser.add_argument(
        "--normalize_world_space",
        action="store_true",
        help="Normalize world space (if used during training).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Render interval, render one frame every N frames. Default is 1 (render all frames).",
    )
    # --- New TSDF parameters ---
    parser.add_argument(
        "--voxel_length",
        type=float,
        default=None,
        help="TSDF voxel size (meters), auto-estimated by default.",
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=None,
        help="TSDF truncation distance (meters), default 3 * voxel_length.",
    )
    # --- ADDED ---
    parser.add_argument(
        "--depth_trunc",
        type=float,
        default=None,
        help="Maximum depth (meters) for TSDF integration. Input depth values beyond this range will be ignored. Default: None (no truncation).",
    )
    # --- END ADDED ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Save directory for the output mesh .ply file. Default uses --data_dir path.",
    )
    # --- New: Post-processing parameters ---
    parser.add_argument(
        "--post_process_clusters",
        type=int,
        default=50,
        help="Maximum number of connected components to keep. Default is 50. Set to 0 to skip post-processing.",
    )
    parser.add_argument(
        "--gs_type",
        type=str,
        choices=["3dgs", "2dgs"],
        default="3dgs",
        help="Type of Gaussian splats saved in the checkpoint. Use '3dgs' for 3D Gaussians or '2dgs' for 2D Gaussians.",
    )

    args = parser.parse_args()

    # Prepare output directory and cache path (clear cache each run)
    output_directory = args.output_dir if args.output_dir else args.data_dir
    os.makedirs(output_directory, exist_ok=True)
    cache_dir = os.path.join(output_directory, "cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[Cache] Writing intermediate RGB/Depth to: {cache_dir}")

    # 1. Auto-select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Gaussian parameters (Logic 1)
    means, quats, scales, opacities, colors, sh_degree = load_params_from_ckpt(
        args.ckpt, device
    )

    if means is None:
        print("Failed to load model parameters, exiting.")
        return

    # 3. Create Trainset (Logic 2, Part 1)
    print(f"\nLoading train set from '{args.data_dir}'...")
    try:
        parser = Parser(
            data_dir=args.data_dir,
            factor=args.data_factor,
            normalize=args.normalize_world_space,
            test_every=8,  # This value does not affect the 'train' split
        )
        trainset = Dataset(
            parser,
            split="train",
        )
        # Use DataLoader for iteration
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=1,  # Process one image at a time
            shuffle=False,  # Render in order
            num_workers=8,  # Simplified operation, set to 8
        )
        print(f"Successfully loaded {len(trainset)} training images.")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 4. Render loop (Logic 2, Part 2)
    print("\nStarting render of 'trainset' (RGB+Depth)...")

    # These lists will store metadata and cached file paths
    color_cache_paths = []
    depth_cache_paths = []
    all_camtoworlds = []  # New: for TSDF
    all_Ks = []  # New: for TSDF
    all_hws = []  # New: for TSDF
    sample_color_shape = None
    sample_depth_shape = None
    color_cache_dtype = "uint8"
    depth_cache_dtype = "float16"

    total_rendered = 0

    # -->> Modified progress bar logic <<--
    # 1. Calculate the total number of frames to actually render
    total_to_render = math.ceil(len(trainset) / args.interval)

    # 2. Create pbar outside the loop
    pbar = tqdm.tqdm(total=total_to_render, desc="Rendering images")

    with torch.no_grad():  # No gradients needed for rendering
        # 3. Iterate trainloader (no longer wrapped with tqdm here)
        for i, data in enumerate(trainloader):
            # -->> Interval check <<--
            if i % args.interval != 0:
                continue

            # Get camera parameters from data loader
            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            gt_pixels = data["image"].to(device)  # [1, H, W, 3]
            height, width = gt_pixels.shape[1:3]

            # Calculate view matrices (gsplat needs viewmats)
            viewmats = torch.linalg.inv(camtoworlds)

            # Render RGB + Expected Depth (ED)
            if args.gs_type == "3dgs":
                renders, _, meta = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    render_mode="RGB+ED",  # Request RGB and Depth
                    packed=False,
                )

                # Separate color and depth
                color = renders[..., 0:3].clamp(0.0, 1.0)  # [1, H, W, 3]
                depth = meta["render_median"]  # [1, H, W, 1]
            else:
                (render_colors, _, _, _, render_median, _,) = rasterization_2dgs(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    render_mode="RGB+ED",
                    packed=False,
                )

                # 2DGS returns RGB channels plus median depth in the last channel
                color = render_colors[..., 0:3].clamp(0.0, 1.0)
                depth = render_median

            # Cache RGB/Depth to disk to avoid holding every frame in memory
            color_np = (
                (color.squeeze(0).cpu().numpy() * 255.0)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )
            depth_np = depth.squeeze(0).squeeze(-1).cpu().numpy().astype(np.float16)
            if sample_color_shape is None:
                sample_color_shape = color_np.shape
            if sample_depth_shape is None:
                sample_depth_shape = depth_np.shape
            cache_idx = total_rendered
            color_path = os.path.join(cache_dir, f"color_{cache_idx:06d}.npy")
            depth_path = os.path.join(cache_dir, f"depth_{cache_idx:06d}.npy")
            np.save(color_path, color_np)
            np.save(depth_path, depth_np)
            color_cache_paths.append(color_path)
            depth_cache_paths.append(depth_path)
            # --- New: Save camera parameters ---
            all_camtoworlds.append(camtoworlds.cpu())
            all_Ks.append(Ks.cpu())
            all_hws.append((height, width))
            # --------------------------
            total_rendered += 1

            # 4. Update pbar only when rendering
            pbar.update(1)

    # 5. Close pbar after the loop
    pbar.close()

    print("\n--- Rendering complete ---")
    print(
        f"Rendered {total_rendered} / {len(trainset)} total images (interval {args.interval})."
    )
    if total_rendered > 0:
        print(f"Cached {total_rendered} rendered frames to disk under '{cache_dir}'.")
        if sample_color_shape is not None:
            print(
                f"  - Sample color tensor shape (saved): {sample_color_shape} [{color_cache_dtype}]"
            )
        if sample_depth_shape is not None:
            print(
                f"  - Sample depth tensor shape (saved): {sample_depth_shape} [{depth_cache_dtype}]"
            )

        # Release 3DGS parameter tensors to free GPU memory before TSDF
        del means, quats, scales, opacities, colors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Memory] Released Gaussian parameter tensors and cleared CUDA cache.")

    else:
        print("No images rendered, skipping TSDF reconstruction.")
        return

    # 5. TSDF Reconstruction (New logic)
    print("\n--- Starting TSDF Reconstruction ---")

    # 5.1 Prepare data (Convert from Tensor list to Numpy array)
    print("Preparing data for TSDF reconstruction...")

    # 1. Depths/Colors: use disk-backed arrays so TSDF loads per frame
    depths_np = DiskNpyArray(depth_cache_paths)
    colors_np = DiskNpyArray(color_cache_paths)

    # 3. Intrinsics: List[Tensor[1,3,3]] -> Array[N,3,3]
    ixts_tensor = torch.cat(all_Ks, dim=0)  # [N, 3, 3]
    ixts_np = ixts_tensor.numpy()

    # 4. Extrinsics: List[Tensor[1,4,4]] -> Array[N,4,4]
    camtoworlds_tensor = torch.cat(all_camtoworlds, dim=0)  # [N, 4, 4]
    camtoworlds_np = camtoworlds_tensor.numpy()

    # 5. H/W: List[(H,W)] -> Array[N, 2]
    hws_np = np.array(all_hws)

    print(
        f"Data sources: frames={len(depths_np)}, ixts={ixts_np.shape}, camtoworlds={camtoworlds_np.shape}, hws={hws_np.shape}"
    )
    if sample_color_shape is not None and sample_depth_shape is not None:
        print(
            f"  - Cached sample shapes -> color: {sample_color_shape} [{color_cache_dtype}], depth: {sample_depth_shape} [{depth_cache_dtype}]"
        )

    # 5.2 Call TSDF reconstruction
    mesh = create_tsdf_mesh(
        depths_np,
        colors_np,
        ixts_np,
        camtoworlds_np,
        hws_np,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,  # <-- ADDED
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # --- New: Call mesh post-processing ---
    # 5.2.1 (Optional) Mesh post-processing
    if args.post_process_clusters > 0:
        print(
            f"\n--- Starting mesh post-processing (keeping {args.post_process_clusters} largest components) ---"
        )
        mesh = post_process_mesh(mesh, cluster_to_keep=args.post_process_clusters)
    else:
        print("\n--- Skipping mesh post-processing ---")
    # --- End of new logic ---

    # 5.3 Restore mesh to original (pre-normalization) coordinate system if needed
    if args.normalize_world_space:
        print("\n--- Restoring mesh to original COLMAP coordinate system ---")
        try:
            normalization_transform = getattr(parser, "transform", None)
            if normalization_transform is None:
                raise AttributeError(
                    "Parser does not expose a normalization transform."
                )
            inv_transform = np.linalg.inv(normalization_transform).astype(np.float64)
            mesh.transform(inv_transform)
            print(
                "[Denormalize] Applied inverse normalization transform "
                f"(scale factor: {getattr(parser, 'scale', 'unknown')})."
            )
        except Exception as exc:
            print(
                f"[Warning] Failed to restore mesh to original coordinates: {exc}. "
                "Saving normalized mesh instead."
            )

    # 5.4 Save Mesh
    # Ensure directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Construct the final file path
    mesh_out_path = os.path.join(output_directory, "reconstructed_mesh.ply")

    o3d.io.write_triangle_mesh(mesh_out_path, mesh)
    print(f"\n--- Reconstruction complete ---")
    print(f"Mesh saved to: {mesh_out_path}")


if __name__ == "__main__":
    main()
