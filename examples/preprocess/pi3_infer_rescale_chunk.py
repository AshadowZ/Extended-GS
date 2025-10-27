import os
import argparse
import torch
import roma
from colmap_util import read_model, get_intrinsics, get_hws, get_extrinsic
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import cv2
from sklearn.linear_model import RANSACRegressor, LinearRegression
import open3d as o3d
import torch
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_path_as_tensor
from pi3.utils.geometry import depth_edge


def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    把 torch.Tensor 转成 NumPy 数组；如果已经是 NumPy 数组则直接返回。
    会自动处理设备（CPU/CUDA）和梯度（detach）。
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
    depths_resized: (N, H, W) float numpy 深度图（单位：米，float32/float64）
    ixts: (N, 3, 3) 或 (N, 3, 3) 内参矩阵 (fx,fy,cx,cy)
    camtoworlds: (N, 4, 4) 相机到世界的 4x4 变换矩阵（numpy）
    image_paths: list of N RGB 图像路径
    hws: list/array of (h,w) for each frame (来自 colmap)
    voxel_length: tsdf 体素尺寸（米），若为 None 则自动估计
    sdf_trunc: 截断距离（米），若为 None 则使用 5 * voxel_length
    color_type: open3d.integration.TSDFVolumeColorType
    """
    N = len(depths_resized)
    assert N == len(image_paths) == len(camtoworlds) == len(ixts) == len(hws)

    # 估算场景尺度（基于相机位姿）
    # 用相机中心之间的中位距离估计
    cam_centers = np.asarray([c[0:3, 3] for c in camtoworlds])
    if len(cam_centers) > 1:
        from scipy.spatial.distance import pdist

        med_dist = float(np.median(pdist(cam_centers)))
    else:
        med_dist = 1.0
    # 默认体素大小为场景中位距离的 1/256（可调整）
    if voxel_length is None:
        voxel_length = med_dist / 36
    if sdf_trunc is None:
        sdf_trunc = voxel_length * 3.0

    print(
        f"[TSDF] N={N}, med_camera_dist={med_dist:.3f} m, voxel_length={voxel_length:.6f} m, sdf_trunc={sdf_trunc:.6f} m"
    )

    # 使用 ScalableTSDFVolume（支持大场景）
    tsdf_vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=sdf_trunc, color_type=color_type
    )

    for i in range(N):
        depth = depths_resized[i].astype(np.float32)  # HxW
        h, w = hws[i]  # colmap 的 (height, width)
        # 有时 depth 的 shape 可能和 h,w 不一致，尝试 resize（但更好是保证一致）
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        color = imageio.imread(image_paths[i])[..., :3]
        if color.shape[0] != h or color.shape[1] != w:
            color = np.array(Image.fromarray(color).resize((w, h), Image.BILINEAR))

        # Open3D 期望 depth 为 float 图像（单位与 depth_scale 对应），我们把 depth_scale 设为 1.0（depth 单位为米）
        o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
        o3d_color = o3d.geometry.Image((color).astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,  # 因为深度已经是以米为单位的浮点数，scale=1.0
            depth_trunc=1000.0,  # 一个较大的截断（实际被 sdf_trunc 决定）
            convert_rgb_to_intensity=False,
        )

        # 构造相机内参 Open3D 对象
        K = ixts[i]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
        )

        # Open3D 要求 extrinsic 为 4x4 的相机到世界变换（也就是 camera -> world）
        extrinsic = np.asarray(camtoworlds[i], dtype=np.float64)

        # integrate
        try:
            tsdf_vol.integrate(
                rgbd, intrinsic, np.linalg.inv(extrinsic)
            )  # NOTE: Open3D 的 integrate 接受 extrinsic = world_to_camera ? see below
            # Explanation:
            # - Historically Open3D v0.9+ expects extrinsic transform from camera to world (extrinsic_cam_to_world).
            # - If you see the result flipped or wrong, try passing np.linalg.inv(extrinsic) instead.
        except Exception as e:
            # 兼容不同版本的 open3d 对 extrinsic 的期望：尝试反向传入
            try:
                tsdf_vol.integrate(rgbd, intrinsic, extrinsic)
            except Exception as ee:
                print(f"[WARN] Frame {i} integrate failed: {e} | fallback failed: {ee}")
        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"[TSDF] integrated {i+1}/{N}")

    print("[TSDF] extracting triangle mesh ...")
    mesh = tsdf_vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # ======= 新增：去除悬浮小mesh =======
    print("[TSDF] filtering small connected components ...")
    (
        triangle_clusters,
        cluster_n_triangles,
        cluster_area,
    ) = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    # 根据三角面片数量阈值筛选
    min_triangles = int(np.max(cluster_n_triangles) * 0.05)  # 保留最大连通块的5%以上，其余丢弃
    mask = cluster_n_triangles[triangle_clusters] >= min_triangles
    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()

    print(f"[TSDF] kept {mesh.triangles.__len__()} triangles after filtering.")
    # ======= 过滤结束 =======

    return mesh


def main():
    parser = argparse.ArgumentParser(description="Pi3 -> TSDF Mesh & Visualization")
    parser.add_argument("--data_dir", type=str, required=True, help="COLMAP 数据集路径")
    parser.add_argument("--chunk_size", type=int, default=45, help="分块推理的批次大小 (默认: 45)")
    parser.add_argument(
        "--tsdf_frame_interval",
        type=int,
        default=1,
        help="TSDF重建使用的帧间隔 (默认: 1，使用所有帧；2表示每隔一帧使用)",
    )
    parser.add_argument("--save_depth_vis", action="store_true", help="是否保存深度图可视化")
    parser.add_argument("--save_point_clouds", action="store_true", help="是否保存彩色点云")
    parser.add_argument(
        "--conf_percentile", type=float, default=0.0, help="用于过滤低置信度像素的百分位数阈值 (默认: 0)"
    )
    parser.add_argument(
        "--skip_inference", action="store_true", help="跳过推理阶段，直接使用现有的chunk_predictions"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    chunk_size = args.chunk_size
    tsdf_frame_interval = args.tsdf_frame_interval
    skip_inference = args.skip_inference

    ## ====== 1. 读取 COLMAP 元数据 start ======
    colmap_dir = os.path.join(data_dir, "sparse/0/")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(data_dir, "sparse")
    assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

    # 从colmap文件夹中提取图像元数据
    cams, images, colmap_points = read_model(colmap_dir)
    ixts, exts, hws, point_ids, names = read_ixt_ext_hw_pointid(
        cams, images, colmap_points
    )
    camtoworlds = np.linalg.inv(exts)

    # 读取图像
    image_names = sorted([img.name for img in images.values()])
    image_dir = os.path.join(data_dir, "images")
    image_paths = [os.path.join(image_dir, name) for name in image_names]
    # 检查文件是否存在
    missing = [p for p in image_paths if not os.path.isfile(p)]
    if missing:
        print(
            f"[WARN] {len(missing)} images not found in {image_dir}, e.g. {missing[:3]}"
        )
    ## ====== 1. 读取 COLMAP 元数据 end ========

    ## ====== 2. 初始化 Pi3 start ======
    print("Loading model...")
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    ## ====== 2. 初始化 Pi3 end ========

    ## ====== 3. 分chunk推理 start =======
    chunk_dir = os.path.join(data_dir, "pi3_chunk_predictions")
    os.makedirs(chunk_dir, exist_ok=True)

    if skip_inference:
        print("[INFO] 跳过推理阶段，直接使用现有的chunk_predictions")
    else:
        print("Running model inference...")
        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                for start in range(0, len(image_paths), chunk_size):
                    end = min(start + chunk_size, len(image_paths))
                    batch_paths = image_paths[start:end]
                    # Load a sequence of N images into a tensor
                    # imgs shape: (N, 3, H, W).
                    # imgs value: [0, 1]
                    batch_images = load_images_path_as_tensor(
                        batch_paths, interval=1
                    ).to(device)
                    batch_images = batch_images.unsqueeze(0)

                    print(f"[Chunk {start}-{end-1}] Inference ...")
                    predictions = model(batch_images)

                    # 处理输出
                    predictions["conf"] = torch.sigmoid(predictions["conf"])
                    edge = depth_edge(predictions["local_points"][..., 2], rtol=0.03)
                    predictions["conf"][edge] = 0.0
                    for key in predictions.keys():
                        if isinstance(predictions[key], torch.Tensor):
                            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
                    predictions["conf"] = predictions["conf"].squeeze(-1)
                    predictions["depth_map"] = predictions["local_points"][:, :, :, 2]

                    # 删除不需要的字段
                    for k in ["points", "local_points"]:
                        if k in predictions:
                            del predictions[k]

                    # 打印待保存的结果
                    print("\nPredictions contains the following variables:")
                    for key in predictions.keys():
                        if isinstance(predictions[key], np.ndarray):
                            print(f"- {key}: shape {predictions[key].shape}")
                        else:
                            print(f"- {key}: type {type(predictions[key])}")

                    # 保存
                    out_file = os.path.join(
                        chunk_dir, f"predictions_{start:05d}_{end-1:05d}.npz"
                    )
                    np.savez_compressed(out_file, **predictions)
                    print(f"[Saved] {out_file}")

                    del batch_images, predictions
                    torch.cuda.empty_cache()
    ## ====== 3. 分chunk推理 end =========

    ## ====== 4. 分chunk处理数据 start =======
    print("\nStart process chunks ...")
    chunk_files = sorted(os.listdir(chunk_dir))
    all_depths_resized = []
    for chunk_idx, fname in enumerate(chunk_files):
        fpath = os.path.join(chunk_dir, fname)
        predictions = dict(np.load(fpath, allow_pickle=True))
        print(f"[Process] {fname} ...")

        # 从文件名提取全局 start/end（文件名格式: predictions_{start:05d}_{end-1:05d}.npz）
        base = os.path.splitext(fname)[0].replace("predictions_", "")
        start_str, end_str = base.split("_")
        start = int(start_str)
        end = int(end_str) + 1

        # ====== 4.1 对齐相机位姿，rescale 深度 start ======
        T_wc_list_pi3 = []
        for i, E in enumerate(predictions["camera_poses"]):
            R_cw = E[:3, :3]  # 3×3 旋转
            t_cw = E[:3, 3]  # 3   平移
            R_wc = R_cw.T
            t_wc = -R_cw.T @ t_cw

            T_wc = torch.eye(4, device=device, dtype=torch.float32)
            T_wc[:3, :3] = torch.from_numpy(R_wc).to(device).float()
            T_wc[:3, 3] = torch.from_numpy(t_wc).to(device).float()
            T_wc_list_pi3.append(T_wc)
        T_wc_list_pi3 = torch.stack(T_wc_list_pi3, dim=0)  # (N,4,4)

        camtoworlds_chunk = camtoworlds[start:end]  # numpy slice
        camtoworlds_tensor = torch.from_numpy(camtoworlds_chunk).to(device).float()
        s, R, T = align_multiple_poses(T_wc_list_pi3, camtoworlds_tensor)
        print(f"[Chunk {start}-{end-1}] 缩放因子 s = {s}")

        # resize + stage1 rescale：注意使用 chunk 内的每一帧对应的全局 hws
        depths_resized = []
        for i_local in range(len(predictions["depth_map"])):
            depth = predictions["depth_map"][i_local]

            global_idx = start + i_local  # 全局索引
            original_height, original_width = hws[global_idx]

            depth_resized = cv2.resize(
                depth,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            depth_resized = depth_resized * s.item()
            depths_resized.append(depth_resized)
            all_depths_resized.append(depth_resized)  # 保存所有深度
        depths_resized = np.stack(depths_resized, axis=0)

        # stage 2: 用 colmap 投影的稀疏深度进一步 rescale 深度（针对 chunk 中每一帧）
        for i_local in range(len(predictions["depth_map"])):
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

            # # 最简单鲁棒的中指比例，RANSAC优化炸了用这个
            # d_vggt = depths_resized[i_local][valid_mask].reshape(-1)
            # d_colmap = depth_img[valid_mask].reshape(-1)
            # ratio = d_colmap / (d_vggt + 1e-8)  # 避免除零
            # median_ratio = np.median(ratio)
            # # MAD 计算
            # abs_dev = np.abs(ratio - median_ratio)
            # mad = np.median(abs_dev) + 1e-8
            # robust_z = abs_dev / mad
            # inlier_mask = robust_z < 3.0
            # a = np.median(ratio[inlier_mask]) if np.any(inlier_mask) else median_ratio
            # inlier_count = int(inlier_mask.sum())
            # b = 0

            # RANSAC估仿射变换
            d_pi3 = depths_resized[i_local][valid_mask].reshape(-1, 1)
            d_colmap = depth_img[valid_mask].reshape(-1, 1)
            try:
                # 去除极值
                lo, hi = np.percentile(d_colmap, [5, 95])
                mask_clip = (d_colmap >= lo) & (d_colmap <= hi)
                d_pi3 = d_pi3[mask_clip].reshape(-1, 1)
                d_colmap = d_colmap[mask_clip].reshape(-1, 1)
                # RANSAC
                ransac = RANSACRegressor(
                    LinearRegression(fit_intercept=True), max_trials=5000
                )
                ransac.fit(d_pi3, d_colmap)
                # 取出结果
                est = ransac.estimator_
                a = float(est.coef_.ravel()[0])
                b = float(est.intercept_.ravel()[0])
                inlier_count = int(ransac.inlier_mask_.sum())
            except Exception as e:
                print(f"[WARN] RANSAC failed on global frame {global_idx+1}: {e}")
                a, b, inlier_count = 1.0, 0.0, 0

            print(
                f"[Global Frame {global_idx+1:05d}] 拟合结果: a = {a:.6f}, b = {b:.6f}, 内点数 = {inlier_count}/{len(d_pi3)}"
            )
            depths_resized[i_local] = a * depths_resized[i_local] + b

            # # 可视化稀疏深度
            # mask = depth_img > 0
            # if np.count_nonzero(mask) == 0:
            #     continue
            # # 归一化深度到 0-255
            # depth_vis = np.zeros_like(depth_img, dtype=np.uint8)
            # valid_depths = depth_img[mask]
            # d_min, d_max = np.percentile(valid_depths, [1, 99])  # 排除极值
            # depth_norm = np.clip((depth_img - d_min) / (d_max - d_min + 1e-6), 0, 1)
            # depth_vis[mask] = (depth_norm[mask] * 255).astype(np.uint8)
            # depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            # # 保存文件夹
            # save_dir = os.path.join(data_dir, "sparse_depth_vis")
            # os.makedirs(save_dir, exist_ok=True)
            # name_noext = os.path.splitext(os.path.basename(names[i]))[0]
            # cv2.imwrite(os.path.join(save_dir, f"{name_noext}_depth.png"), depth_color)
        # ====== 4.1 对齐相机位姿，rescale 深度 end ========

        # ====== 4.2 保存深度图并可视化 start ======
        depth_dir = os.path.join(data_dir, "pi3_depth")
        os.makedirs(depth_dir, exist_ok=True)
        depth_vis_dir = None
        if args.save_depth_vis:
            depth_vis_dir = os.path.join(data_dir, "pi3_vis_depth")
            os.makedirs(depth_vis_dir, exist_ok=True)

        for i_local in range(len(predictions["depth_map"])):
            global_idx = start + i_local
            resized_depth = depths_resized[i_local]  # 修改原始数据

            # 检查并转换数据类型为float16
            if resized_depth.dtype != np.float16:
                # 先检查深度值范围是否适合float16
                max_depth = np.max(resized_depth[resized_depth > 0])  # 只计算有效深度
                if max_depth > 65504:  # float16最大表示范围
                    print(f"警告：深度值{max_depth:.2f}超过float16范围，已自动截断")
                    resized_depth = np.clip(resized_depth, 0, 65504)
                resized_depth = resized_depth.astype(np.float16)

            # 将 conf 调整到原始图像分辨率（同样根据模式选择恢复方式）
            depth_conf = predictions["conf"][i_local]
            original_height, original_width = hws[global_idx]

            resized_depth_conf = cv2.resize(
                depth_conf,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # 计算置信度的分位数阈值
            conf_threshold = np.percentile(resized_depth_conf, args.conf_percentile)
            valid_conf_mask = (resized_depth_conf >= conf_threshold) & (
                resized_depth_conf > 1e-5
            )

            invalid_mask = ~valid_conf_mask
            resized_depth[invalid_mask] = 0.0

            # 保存 .npy
            depth_npy_path = os.path.join(depth_dir, f"frame_{global_idx+1:05d}.npy")
            np.save(depth_npy_path, resized_depth)

            # 保存可视化 .jpeg
            if args.save_depth_vis:
                # 转换为 float32 以满足 OpenCV 要求
                resized_depth = resized_depth.astype(np.float32)
                # 标准化到 0-255
                depth_normalized = cv2.normalize(
                    resized_depth, None, 0, 255, cv2.NORM_MINMAX
                )
                depth_normalized = depth_normalized.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                zero_mask = resized_depth == 0
                depth_colored[zero_mask] = (255, 255, 255)

                # 保存为JPEG（调整质量参数）
                depth_path = os.path.join(
                    depth_vis_dir, f"depth_{global_idx+1:05d}.jpg"
                )
                cv2.imwrite(
                    depth_path,
                    depth_colored,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90],  # 质量参数（0-100）
                )

        print(f"[Chunk {start}-{end-1}] saved depth .npy and visualizations.")
        # ====== 4.2 保存深度图并可视化 end ========

        # ====== 4.3 保存彩色点云 start ========
        if args.save_point_clouds:
            output_dir = os.path.join(data_dir, "pi3_point_clouds")
            os.makedirs(output_dir, exist_ok=True)

            for i_local in range(len(predictions["depth_map"])):
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
        # ====== 4.3 保存彩色点云 end ==========

        # 清理并继续下一个 chunk
        del predictions, depths_resized
        torch.cuda.empty_cache()
    ## ====== 4. 分chunk处理数据 end =========

    ## ====== 5. TSDF 融合并保存 mesh 及其对应彩色稀疏点云 start ======
    # 调用 TSDF 合并并保存 mesh
    all_depths_resized = np.array(all_depths_resized)
    print(f"all_depths_resized shape: {all_depths_resized.shape}")

    if tsdf_frame_interval > 1:
        total_frames = len(all_depths_resized)
        selected_indices = list(range(0, total_frames, tsdf_frame_interval))
        print(
            f"[TSDF] 使用间隔 {tsdf_frame_interval}，从 {total_frames} 帧中选择 {len(selected_indices)} 帧进行重建"
        )
        # 使用部分帧
        selected_depths = all_depths_resized[selected_indices]
        selected_ixts = ixts[selected_indices]
        selected_camtoworlds = camtoworlds[selected_indices]
        selected_image_paths = [image_paths[i] for i in selected_indices]
        selected_hws = hws[selected_indices]
    else:
        # 使用所有帧
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
    print(f"[Saved] mesh saved to: {mesh_out_ply}")

    # 保存顶点为彩色稀疏点云
    print("[EXTRACT] extracting vertices and colors from mesh ...")
    verts = np.asarray(mesh.vertices)  # (V,3)
    # 如果 mesh 自带顶点颜色，直接用；否则使用后备投影采样办法
    if mesh.has_vertex_colors() and np.asarray(mesh.vertex_colors).size != 0:
        colors = np.asarray(mesh.vertex_colors)  # (V,3), 假定在 0-1 范围
        # 有些 open3d 可能保存 0-255 范围，但通常是 0-1
        if colors.max() > 1.1:
            colors = colors / 255.0

    # 构造 Open3D 点云并保存为彩色 ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    # clamp colors to [0,1]
    colors_clamped = np.clip(colors, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors_clamped)
    pointcloud_out_ply = os.path.join(data_dir, "scene_vertices_color.ply")
    o3d.io.write_point_cloud(pointcloud_out_ply, pcd, write_ascii=False)
    print(f"[Saved] colored point cloud saved to: {pointcloud_out_ply}")
    ## ====== 5. TSDF 融合并保存 mesh 及其对应彩色稀疏点云 end ========


if __name__ == "__main__":
    main()
