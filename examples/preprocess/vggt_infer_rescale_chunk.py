import os
import argparse
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import roma
from colmap_util import read_model, get_intrinsics, get_hws, get_extrinsic
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import cv2
from sklearn.linear_model import RANSACRegressor, LinearRegression
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F


def depth_edge(
    depth: torch.Tensor,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: torch.Tensor = None,
) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.

    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = F.max_pool2d(
            depth, kernel_size, stride=1, padding=kernel_size // 2
        ) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        diff = F.max_pool2d(
            torch.where(mask, depth, -torch.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + F.max_pool2d(
            torch.where(mask, -depth, -torch.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge


def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a torch.Tensor into a NumPy array, or return it directly if it already is one.
    Automatically moves tensors to CPU and detaches gradients where necessary.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist

    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))


def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 100
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps * poses[:, :3, 2]))

    R, T, s = roma.rigid_points_registration(
        center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True
    )
    return s, R, T


def get_sparse_depth(points3d, ixt, ext, point3D_ids, h, w):
    # sparse_depth: Nx3 array, uvd
    if [id for id in point3D_ids if id != -1] == []:
        return []
    points = np.asarray([points3d[id].xyz for id in point3D_ids if id != -1])
    errs = np.asarray([points3d[id].error for id in point3D_ids if id != -1])
    num_views = np.asarray(
        [len(points3d[id].image_ids) for id in point3D_ids if id != -1]
    )
    sparse_points = points @ ext[:3, :3].T + ext[:3, 3:].T
    sparse_points = sparse_points @ ixt.T
    sparse_points[:, :2] = sparse_points[:, :2] / sparse_points[:, 2:]
    sparse_points = np.concatenate(
        [sparse_points, errs[:, None], num_views[:, None]], axis=1
    )

    sdpt = np.zeros((h, w, 3))
    for x, y, z, error, num_views in sparse_points:
        x, y = int(x), int(y)
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        sdpt[y, x, 0] = z
        sdpt[y, x, 1] = error
        sdpt[y, x, 2] = num_views

    return sdpt


def crop_vggt_pred_to_original_dimensions(
    vggt_pred, original_width, original_height, target_size=518
):
    if original_width >= original_height:
        new_width = target_size
        new_height = round(original_height * (new_width / original_width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(original_width * (new_height / original_height) / 14) * 14

    h_padding = target_size - new_height
    w_padding = target_size - new_width

    pad_top = h_padding // 2
    pad_left = w_padding // 2

    if len(vggt_pred.shape) == 3:  # HxWx3
        cropped_pred = vggt_pred[
            pad_top : pad_top + new_height, pad_left : pad_left + new_width, :
        ]
    else:  # HxW
        cropped_pred = vggt_pred[
            pad_top : pad_top + new_height, pad_left : pad_left + new_width
        ]

    return cv2.resize(
        cropped_pred, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )


def read_ixt_ext_hw_pointid(cams, images, points):
    # get image ids
    name2imageid = {img.name: img.id for img in images.values()}
    names = sorted([img.name for img in images.values()])
    imageids = [name2imageid[name] for name in names]

    # ixts
    ixts = np.asarray(
        [get_intrinsics(cams[images[imageid].camera_id]) for imageid in imageids]
    )
    # exts
    exts = np.asarray([get_extrinsic(images[imageid]) for imageid in imageids])
    # hws
    hws = np.asarray([get_hws(cams[images[imageid].camera_id]) for imageid in imageids])
    # point ids
    point_ids = [images[imageid].point3D_ids for imageid in imageids]

    return ixts, exts, hws, point_ids, names


def create_tsdf_mesh(
    depths_resized,
    ixts,
    camtoworlds,
    image_paths,
    hws,
    voxel_length=None,
    sdf_trunc=None,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
):
    """
    depths_resized: (N, H, W) float numpy depth maps in meters (float32/float64)
    ixts: (N, 3, 3) intrinsic matrices (fx, fy, cx, cy)
    camtoworlds: (N, 4, 4) camera-to-world transforms (numpy)
    image_paths: list of N RGB image paths
    hws: list/array of (h, w) per frame (from COLMAP)
    voxel_length: TSDF voxel size in meters; auto-estimated if None
    sdf_trunc: truncation distance in meters; defaults to 5 * voxel_length if None
    color_type: open3d.integration.TSDFVolumeColorType
    """
    N = len(depths_resized)
    assert N == len(image_paths) == len(camtoworlds) == len(ixts) == len(hws)

    # Estimate scene scale (based on camera poses)
    # Use the median distance between camera centers
    cam_centers = np.asarray([c[0:3, 3] for c in camtoworlds])
    if len(cam_centers) > 1:
        from scipy.spatial.distance import pdist

        med_dist = float(np.median(pdist(cam_centers)))
    else:
        med_dist = 1.0
    # Default voxel size is median distance / 64 (adjust as needed)
    if voxel_length is None:
        voxel_length = med_dist / 64
    if sdf_trunc is None:
        sdf_trunc = voxel_length * 3.0

    print(
        f"[TSDF] N={N}, med_camera_dist={med_dist:.3f} m, voxel_length={voxel_length:.6f} m, sdf_trunc={sdf_trunc:.6f} m"
    )

    # Use ScalableTSDFVolume (handles large scenes)
    tsdf_vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=sdf_trunc, color_type=color_type
    )

    for i in range(N):
        depth = depths_resized[i].astype(np.float32)  # HxW
        h, w = hws[i]  # (height, width) from COLMAP
        # Depth maps may occasionally disagree with (h, w); resize as a fallback
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        color = imageio.imread(image_paths[i])[..., :3]
        if color.shape[0] != h or color.shape[1] != w:
            color = np.array(Image.fromarray(color).resize((w, h), Image.BILINEAR))

        # Open3D expects depth as float images whose units match depth_scale; we keep depth in meters with scale 1.0
        o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
        o3d_color = o3d.geometry.Image((color).astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,  # depth is already stored in meters
            depth_trunc=1000.0,  # loose truncation; sdf_trunc governs actual TSDF trunc
            convert_rgb_to_intensity=False,
        )

        # Construct Open3D camera intrinsics
        K = ixts[i]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
        )

        # Open3D expects a 4x4 camera-to-world extrinsic
        extrinsic = np.asarray(camtoworlds[i], dtype=np.float64)

        # integrate
        try:
            tsdf_vol.integrate(
                rgbd, intrinsic, np.linalg.inv(extrinsic)
            )  # NOTE: Open3D historically uses world-to-camera extrinsics
            # Explanation:
            # - Historically Open3D v0.9+ expects extrinsic transform from camera to world (extrinsic_cam_to_world).
            # - If you see the result flipped or wrong, try passing np.linalg.inv(extrinsic) instead.
        except Exception as e:
            # Fall back by swapping the extrinsic convention if integration fails
            try:
                tsdf_vol.integrate(rgbd, intrinsic, extrinsic)
            except Exception as ee:
                print(f"[WARN] Frame {i} integrate failed: {e} | fallback failed: {ee}")
        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"[TSDF] integrated {i+1}/{N}")

    print("[TSDF] extracting triangle mesh ...")
    mesh = tsdf_vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # ======= Remove small floating mesh components =======
    print("[TSDF] filtering small connected components ...")
    (
        triangle_clusters,
        cluster_n_triangles,
        cluster_area,
    ) = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    # Filter by the number of faces per connected component
    min_triangles = int(np.max(cluster_n_triangles) * 0.05)  # keep clusters >=5% of the largest one
    mask = cluster_n_triangles[triangle_clusters] >= min_triangles
    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()

    print(f"[TSDF] kept {mesh.triangles.__len__()} triangles after filtering.")
    # ======= Filtering complete =======

    return mesh


def main():
    parser = argparse.ArgumentParser(description="VGGT -> TSDF Mesh & Visualization")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the COLMAP dataset")
    parser.add_argument("--chunk_size", type=int, default=45, help="Batch size per inference chunk (default: 45)")
    parser.add_argument(
        "--tsdf_frame_interval",
        type=int,
        default=1,
        help="Frame interval for TSDF reconstruction (default 1 = all frames; 2 uses every other frame)",
    )
    parser.add_argument("--save_depth_vis", action="store_true", help="Save depth visualization images")
    parser.add_argument("--save_point_clouds", action="store_true", help="Save colored point clouds")
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=0.0,
        help="Percentile threshold for filtering low-confidence pixels (default: 0)",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference and reuse existing chunk_predictions",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    chunk_size = args.chunk_size
    tsdf_frame_interval = args.tsdf_frame_interval
    skip_inference = args.skip_inference

    ## ====== 1. Load COLMAP metadata start ======
    colmap_dir = os.path.join(data_dir, "sparse/0/")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(data_dir, "sparse")
    assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

    # Extract image metadata from the COLMAP folder
    cams, images, colmap_points = read_model(colmap_dir)
    ixts, exts, hws, point_ids, names = read_ixt_ext_hw_pointid(
        cams, images, colmap_points
    )
    camtoworlds = np.linalg.inv(exts)

    # Load images
    image_names = sorted([img.name for img in images.values()])
    image_dir = os.path.join(data_dir, "images")
    image_paths = [os.path.join(image_dir, name) for name in image_names]
    # Check that files exist
    missing = [p for p in image_paths if not os.path.isfile(p)]
    if missing:
        print(
            f"[WARN] {len(missing)} images not found in {image_dir}, e.g. {missing[:3]}"
        )

    # Decide preprocessing mode based on the first image aspect ratio
    first_image_path = image_paths[0]
    first_image = Image.open(first_image_path)
    first_width, first_height = first_image.size
    if first_height > first_width:
        # Height > width -> use pad mode
        preprocessing_mode = "pad"
        print(f"[INFO] First image {first_width}x{first_height} (H>W); using pad mode")
    else:
        # Width >= height -> use crop mode
        preprocessing_mode = "crop"
        print(f"[INFO] First image {first_width}x{first_height} (W>=H); using crop mode")
    first_image.close()
    ## ====== 1. Load COLMAP metadata end ========

    ## ====== 2. Initialize VGGT start ======
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    model = VGGT()
    _URL = "/home/joker/learning/VGGT-Long/weights/model.pt"
    state_dict = torch.load(_URL, map_location="cpu")  # best-effort load
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    ## ====== 2. Initialize VGGT end ========

    ## ====== 3. Chunked inference start =======
    chunk_dir = os.path.join(data_dir, "chunk_predictions")
    os.makedirs(chunk_dir, exist_ok=True)

    if skip_inference:
        print("[INFO] Skipping inference and reusing existing chunk_predictions")
    else:
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                for start in range(0, len(image_paths), chunk_size):
                    end = min(start + chunk_size, len(image_paths))
                    batch_paths = image_paths[start:end]
                    if preprocessing_mode == "pad":
                        batch_images = load_and_preprocess_images(
                            batch_paths, mode="pad"
                        ).to(device)
                    else:
                        batch_images = load_and_preprocess_images(
                            batch_paths, mode="crop"
                        ).to(device)

                    print(f"[Chunk {start}–{end-1}] Inference ...")
                    predictions = model(batch_images)

                    # Process outputs
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        predictions["pose_enc"], batch_images.shape[-2:]
                    )
                    predictions["extrinsic"] = extrinsic
                    predictions["intrinsic"] = intrinsic
                    for key in predictions.keys():
                        if isinstance(predictions[key], torch.Tensor):
                            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
                    predictions["depth"] = np.squeeze(predictions["depth"])
                    # Drop unused fields
                    for k in [
                        "pose_enc",
                        "pose_enc_list",
                        "world_points",
                        "world_points_conf",
                        "images",
                    ]:
                        if k in predictions:
                            del predictions[k]

                    # Inspect results before saving if needed
                    # print("\nPredictions contains the following variables:")
                    # for key in predictions.keys():
                    #     if isinstance(predictions[key], np.ndarray):
                    #         print(f"- {key}: shape {predictions[key].shape}")
                    #     else:
                    #         print(f"- {key}: type {type(predictions[key])}")

                    # Save predictions
                    out_file = os.path.join(
                        chunk_dir, f"predictions_{start:05d}_{end-1:05d}.npz"
                    )
                    np.savez_compressed(out_file, **predictions)
                    print(f"[Saved] {out_file}")

                    del batch_images, predictions
                    torch.cuda.empty_cache()
    ## ====== 3. Chunked inference end =========

    ## ====== 4. Process chunks start =======
    print("\nStart process chunks ...")
    chunk_files = sorted(os.listdir(chunk_dir))
    all_depths_resized = []
    for chunk_idx, fname in enumerate(chunk_files):
        fpath = os.path.join(chunk_dir, fname)
        predictions = dict(np.load(fpath, allow_pickle=True))
        print(f"[Process] {fname} ...")

        # Parse global start/end indices from filename (predictions_{start:05d}_{end-1:05d}.npz)
        base = os.path.splitext(fname)[0].replace("predictions_", "")
        start_str, end_str = base.split("_")
        start = int(start_str)
        end = int(end_str) + 1

        # ====== 4.1 Align camera poses and rescale depth start ======
        T_wc_list_vggt = []
        for i, E in enumerate(predictions["extrinsic"]):
            R_cw = E[:, :3]  # 3×3 rotation
            t_cw = E[:, 3]  # translation
            R_wc = R_cw.T
            t_wc = -R_cw.T @ t_cw

            T_wc = torch.eye(4, device=device, dtype=torch.float32)
            T_wc[:3, :3] = torch.from_numpy(R_wc).to(device).float()
            T_wc[:3, 3] = torch.from_numpy(t_wc).to(device).float()
            T_wc_list_vggt.append(T_wc)
        T_wc_list_vggt = torch.stack(T_wc_list_vggt, dim=0)  # (N,4,4)

        camtoworlds_chunk = camtoworlds[start:end]  # numpy slice
        camtoworlds_tensor = torch.from_numpy(camtoworlds_chunk).to(device).float()
        s, R, T = align_multiple_poses(T_wc_list_vggt, camtoworlds_tensor)
        print(f"[Chunk {start}-{end-1}] scale factor s = {s}")

        # Resize + stage1 rescale: use per-frame global hws
        depths_resized = []
        for i_local in range(len(predictions["depth"])):
            depth = predictions["depth"][i_local]

            global_idx = start + i_local  # global index
            original_height, original_width = hws[global_idx]

            # Restore resolution according to preprocessing mode
            if preprocessing_mode == "pad":
                # pad mode: use crop_vggt_pred_to_original_dimensions
                depth_resized = crop_vggt_pred_to_original_dimensions(
                    depth, original_width, original_height
                )
            else:
                # crop mode: directly resize with cv2.INTER_NEAREST
                depth_resized = cv2.resize(
                    depth,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            depth_resized = depth_resized * s.item()
            depths_resized.append(depth_resized)
            all_depths_resized.append(depth_resized)  # cache all depth maps
        depths_resized = np.stack(depths_resized, axis=0)

        # Stage 2: refine depths with COLMAP sparse projections (per frame)
        for i_local in range(len(predictions["depth"])):
            global_idx = start + i_local
            h, w = hws[global_idx]
            ixt = ixts[global_idx]
            ext = exts[global_idx]
            point_id = point_ids[global_idx]

            sdpt = get_sparse_depth(colmap_points, ixt, ext, point_id, h, w)
            if len(sdpt) == 0:
                print(f"[WARN] Global Frame {global_idx+1} has no sparse depth points.")
                continue

            depth_img = sdpt[..., 0]
            valid_mask = depth_img > 0
            if np.count_nonzero(valid_mask) < 10:
                print(
                    f"[WARN] Global Frame {global_idx+1} has too few valid sparse points."
                )
                continue

            # Simplest robust median-ratio fallback if RANSAC fails
            # d_vggt = depths_resized[i_local][valid_mask].reshape(-1)
            # d_colmap = depth_img[valid_mask].reshape(-1)
            # ratio = d_colmap / (d_vggt + 1e-8)  # avoid divide-by-zero
            # median_ratio = np.median(ratio)
            # Median absolute deviation
            # abs_dev = np.abs(ratio - median_ratio)
            # mad = np.median(abs_dev) + 1e-8
            # robust_z = abs_dev / mad
            # inlier_mask = robust_z < 3.0
            # a = np.median(ratio[inlier_mask]) if np.any(inlier_mask) else median_ratio
            # inlier_count = int(inlier_mask.sum())
            # b = 0

            # RANSAC-fit affine transform
            d_vggt = depths_resized[i_local][valid_mask].reshape(-1, 1)
            d_colmap = depth_img[valid_mask].reshape(-1, 1)
            try:
                # Trim extreme values
                lo, hi = np.percentile(d_colmap, [5, 95])
                mask_clip = (d_colmap >= lo) & (d_colmap <= hi)
                d_vggt = d_vggt[mask_clip].reshape(-1, 1)
                d_colmap = d_colmap[mask_clip].reshape(-1, 1)
                # Run RANSAC
                ransac = RANSACRegressor(
                    LinearRegression(fit_intercept=True), max_trials=5000
                )
                ransac.fit(d_vggt, d_colmap)
                # Read coefficients
                est = ransac.estimator_
                a = float(est.coef_.ravel()[0])
                b = float(est.intercept_.ravel()[0])
                inlier_count = int(ransac.inlier_mask_.sum())
            except Exception as e:
                print(f"[WARN] RANSAC failed on global frame {global_idx+1}: {e}")
                a, b, inlier_count = 1.0, 0.0, 0

            print(
                f"[Global Frame {global_idx+1:05d}] fit result: a = {a:.6f}, b = {b:.6f}, inliers = {inlier_count}/{len(d_vggt)}"
            )
            depths_resized[i_local] = a * depths_resized[i_local] + b

            # Optional sparse-depth visualization
            # mask = depth_img > 0
            # if np.count_nonzero(mask) == 0:
            #     continue
            # Normalize depth to 0-255
            # depth_vis = np.zeros_like(depth_img, dtype=np.uint8)
            # valid_depths = depth_img[mask]
            # d_min, d_max = np.percentile(valid_depths, [1, 99])  # clamp extremes
            # depth_norm = np.clip((depth_img - d_min) / (d_max - d_min + 1e-6), 0, 1)
            # depth_vis[mask] = (depth_norm[mask] * 255).astype(np.uint8)
            # depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            # Output directory
            # save_dir = os.path.join(data_dir, "sparse_depth_vis")
            # os.makedirs(save_dir, exist_ok=True)
            # name_noext = os.path.splitext(os.path.basename(names[i]))[0]
            # cv2.imwrite(os.path.join(save_dir, f"{name_noext}_depth.png"), depth_color)
        # ====== 4.1 Align camera poses and rescale depth end ========

        # ====== 4.2 Save depth maps and visualizations start ======
        depth_dir = os.path.join(data_dir, "vggt_depth")
        os.makedirs(depth_dir, exist_ok=True)
        depth_vis_dir = None
        if args.save_depth_vis:
            depth_vis_dir = os.path.join(data_dir, "vggt_vis_depth")
            os.makedirs(depth_vis_dir, exist_ok=True)

        for i_local in range(len(predictions["depth"])):
            global_idx = start + i_local
            resized_depth = depths_resized[i_local]  # modify data in place

            # Resize confidence to the original image resolution (mode-dependent)
            depth_conf = predictions["depth_conf"][i_local]
            original_height, original_width = hws[global_idx]

            if preprocessing_mode == "pad":
                resized_depth_conf = crop_vggt_pred_to_original_dimensions(
                    depth_conf, original_width, original_height
                )
            else:
                resized_depth_conf = cv2.resize(
                    depth_conf,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Compute the confidence percentile threshold
            conf_threshold = np.percentile(resized_depth_conf, args.conf_percentile)
            valid_conf_mask = resized_depth_conf >= conf_threshold

            # Detect and filter depth edges
            depth_tensor = (
                torch.from_numpy(resized_depth).float().unsqueeze(0).unsqueeze(0).cuda()
            )
            edge_mask = depth_edge(depth_tensor, atol=None, rtol=0.03, kernel_size=3)
            edge_mask = edge_mask[0, 0].cpu().numpy().astype(bool)

            invalid_mask = (~valid_conf_mask) | edge_mask
            resized_depth[invalid_mask] = 0

            # Save .npy
            depth_npy_path = os.path.join(depth_dir, f"frame_{global_idx+1:05d}.npy")
            np.save(depth_npy_path, resized_depth)

            # Save visualization .png
            if args.save_depth_vis:
                depth_normalized = cv2.normalize(
                    resized_depth, None, 0, 255, cv2.NORM_MINMAX
                )
                depth_normalized = depth_normalized.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                zero_mask = resized_depth == 0
                depth_colored[zero_mask] = (255, 255, 255)

                depth_path = os.path.join(
                    depth_vis_dir, f"depth_{global_idx+1:05d}.png"
                )
                cv2.imwrite(depth_path, depth_colored)

        print(f"[Chunk {start}-{end-1}] saved depth .npy files and visualizations.")
        # ====== 4.2 Save depth maps and visualizations end ========

        # ====== 4.3 Save colored point clouds start ========
        if args.save_point_clouds:
            output_dir = os.path.join(data_dir, "point_clouds")
            os.makedirs(output_dir, exist_ok=True)

            for i_local in range(len(predictions["depth"])):
                global_idx = start + i_local
                resized_depth = depths_resized[i_local]

                original_height, original_width = hws[global_idx]
                u, v = np.meshgrid(
                    np.arange(original_width), np.arange(original_height)
                )
                uv1 = (
                    np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T
                )  # (3, H*W)

                depth_flat = resized_depth.reshape(-1)
                valid_mask = depth_flat > 0
                uv1_valid = uv1[:, valid_mask]
                depth_valid = depth_flat[valid_mask]

                if len(depth_valid) == 0:
                    continue

                K = ixts[global_idx]
                T_wc = camtoworlds[global_idx]

                points_cam = (np.linalg.inv(K) @ uv1_valid).T
                points_cam = points_cam * depth_valid.reshape(-1, 1)
                points_world = (T_wc[:3, :3] @ points_cam.T + T_wc[:3, 3:4]).T

                img_path = image_paths[global_idx]
                img = imageio.imread(img_path)[..., :3]
                colors = img.reshape(-1, 3)[valid_mask] / 255.0

                points = np.concatenate([points_world, colors], axis=1)

                output_path = os.path.join(
                    output_dir, f"point_cloud_{global_idx+1:05d}.ply"
                )
                with open(output_path, "w") as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(points)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                    f.write("end_header\n")
                    for x, y, z, r, g, b in points:
                        f.write(f"{x} {y} {z} {int(r*255)} {int(g*255)} {int(b*255)}\n")

            print(f"[Chunk {start}-{end-1}] point clouds saved to {output_dir}")
        # ====== 4.3 Save colored point clouds end ==========

        # Clean up before processing the next chunk
        del predictions, depths_resized, depth_tensor
        torch.cuda.empty_cache()
    ## ====== 4. Process chunks end =========

    ## ====== 5. TSDF fusion and colored sparse export start ======
    # Fuse TSDF and save the resulting mesh
    all_depths_resized = np.array(all_depths_resized)
    print(f"all_depths_resized shape: {all_depths_resized.shape}")

    if tsdf_frame_interval > 1:
        total_frames = len(all_depths_resized)
        selected_indices = list(range(0, total_frames, tsdf_frame_interval))
        print(
            f"[TSDF] Interval {tsdf_frame_interval}: using {len(selected_indices)} / {total_frames} frames for reconstruction"
        )
        # Use a subset of frames
        selected_depths = all_depths_resized[selected_indices]
        selected_ixts = ixts[selected_indices]
        selected_camtoworlds = camtoworlds[selected_indices]
        selected_image_paths = [image_paths[i] for i in selected_indices]
        selected_hws = hws[selected_indices]
    else:
        # Use all frames
        selected_depths = all_depths_resized
        selected_ixts = ixts
        selected_camtoworlds = camtoworlds
        selected_image_paths = image_paths
        selected_hws = hws

    mesh = create_tsdf_mesh(
        selected_depths,
        selected_ixts,
        selected_camtoworlds,
        selected_image_paths,
        selected_hws,
        voxel_length=None,
        sdf_trunc=None,
    )

    mesh_out_ply = os.path.join(data_dir, "scene_tsdf_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_out_ply, mesh)

    # Save mesh vertices as a colored sparse point cloud
    print("[EXTRACT] extracting vertices and colors from mesh ...")
    verts = np.asarray(mesh.vertices)  # (V,3)
    # Reuse mesh vertex colors when available; otherwise fall back to projection sampling
    if mesh.has_vertex_colors() and np.asarray(mesh.vertex_colors).size != 0:
        colors = np.asarray(mesh.vertex_colors)  # (V,3), assumed to be in [0,1]
        # Some Open3D builds store 0-255, but typically 0-1
        if colors.max() > 1.1:
            colors = colors / 255.0

    # Build an Open3D point cloud and save as a colored PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    # clamp colors to [0,1]
    colors_clamped = np.clip(colors, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors_clamped)
    pointcloud_out_ply = os.path.join(data_dir, "scene_vertices_color.ply")
    o3d.io.write_point_cloud(pointcloud_out_ply, pcd, write_ascii=False)
    print(f"[Saved] colored point cloud saved to: {pointcloud_out_ply}")
    ## ====== 5. TSDF fusion and colored sparse export end ========


if __name__ == "__main__":
    main()
