import argparse
import os
import threading
import time
from queue import Queue

import cv2
import numpy as np
import torch
import utils3d
from colmap_util import get_extrinsic, get_hws, get_intrinsics, read_model
from moge.model.v2 import MoGeModel  # Use MoGe-2
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr


def get_sparse_depth(points3d, ixt, ext, point3d_ids, h, w):
    if all(pid == -1 for pid in point3d_ids):
        return None

    valid_ids = [pid for pid in point3d_ids if pid != -1 and pid in points3d]
    if not valid_ids:
        return None

    points = np.asarray([points3d[pid].xyz for pid in valid_ids])
    errs = np.asarray([points3d[pid].error for pid in valid_ids])
    num_views = np.asarray([len(points3d[pid].image_ids) for pid in valid_ids])

    sparse_points = points @ ext[:3, :3].T + ext[:3, 3]
    sparse_points = sparse_points @ ixt.T
    sparse_points[:, :2] /= sparse_points[:, 2:]
    sparse_points = np.concatenate(
        [sparse_points, errs[:, None], num_views[:, None]], axis=1
    )

    sdpt = np.zeros((h, w, 3), dtype=np.float32)
    for x, y, z, error, views in sparse_points:
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        if z > 0:
            sdpt[yi, xi, 0] = z
            sdpt[yi, xi, 1] = error
            sdpt[yi, xi, 2] = views
    if not np.any(sdpt[..., 0] > 0):
        return None
    return sdpt


def normalize_image_key(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0].lower()


def prepare_colmap_context(colmap_dir, data_factor=1.0):
    cams, images, points3d = read_model(colmap_dir)
    meta = {}
    meta_norm = {}
    for img in images.values():
        cam = cams[img.camera_id]
        ixt = get_intrinsics(cam)
        h, w = get_hws(cam)
        if data_factor != 1.0:
            ixt = ixt.copy()
            ixt[0, :] /= data_factor
            ixt[1, :] /= data_factor
            h = max(1, int(round(h / data_factor)))
            w = max(1, int(round(w / data_factor)))

        entry = {
            "ixt": ixt,
            "ext": get_extrinsic(img),
            "hw": (h, w),
            "point_ids": img.point3D_ids,
        }
        meta[img.name] = entry
        norm_key = normalize_image_key(img.name)
        if norm_key not in meta_norm:
            meta_norm[norm_key] = entry
    return meta, meta_norm, points3d


def run_inference(
    data_dir,
    image_dir_name="images",
    read_threads=6,
    write_threads=3,
    queue_size=8,
    save_depth=False,
    save_depth_vis=False,
    remove_depth_edge=True,
    depth_edge_rtol=0.04,
    align_depth_with_colmap=True,
    data_factor=1.0,
    verbose=False,
    colmap_min_sparse=30,
    colmap_clip_low=5.0,
    colmap_clip_high=95.0,
    colmap_ransac_trials=2000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model_load_start = time.time()
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    normal_output_dir = os.path.join(data_dir, "moge_normal")
    os.makedirs(normal_output_dir, exist_ok=True)

    depth_output_dir = None
    depth_vis_output_dir = None
    if save_depth:
        depth_output_dir = os.path.join(data_dir, "moge_depth")
        os.makedirs(depth_output_dir, exist_ok=True)
    if save_depth_vis:
        depth_vis_output_dir = os.path.join(data_dir, "moge_depth_vis")
        os.makedirs(depth_vis_output_dir, exist_ok=True)

    colmap_meta = None
    colmap_points = None
    colmap_meta = None
    colmap_meta_norm = None
    colmap_points = None
    if align_depth_with_colmap:
        resolved_colmap_dir = os.path.join(data_dir, "sparse", "0")
        if not os.path.exists(resolved_colmap_dir):
            resolved_colmap_dir = os.path.join(data_dir, "sparse")
        if os.path.exists(resolved_colmap_dir):
            print(f"Loading COLMAP model from {resolved_colmap_dir}")
            colmap_meta, colmap_meta_norm, colmap_points = prepare_colmap_context(
                resolved_colmap_dir, data_factor=data_factor
            )
            print(f"COLMAP alignment ready for {len(colmap_meta)} images")
        else:
            print("[WARN] COLMAP model directory not found; skipping depth alignment.")
            align_depth_with_colmap = False

    image_dir = os.path.join(data_dir, image_dir_name)
    image_files = sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        print(f"No processable images found in {image_dir}.")
        return

    print(f"Found {len(image_files)} images")
    queue_size = max(1, queue_size)

    filename_queue = Queue()
    data_queue = Queue(maxsize=queue_size)
    save_queue = Queue(maxsize=queue_size)

    total_read_time = 0.0
    total_inference_time = 0.0
    stats_lock = threading.Lock()
    save_stats = {"time": 0.0, "count": 0}

    for fname in image_files:
        filename_queue.put(fname)
    for _ in range(read_threads):
        filename_queue.put(None)

    def reader_worker():
        while True:
            fname = filename_queue.get()
            if fname is None:
                filename_queue.task_done()
                break

            img_path = os.path.join(image_dir, fname)
            read_start = time.time()
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                read_time = time.time() - read_start
                data_queue.put((fname, None, read_time))
                print(f"[WARN] Failed to read image {img_path}")
                filename_queue.task_done()
                continue

            img_rgb = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            img_rgb = np.ascontiguousarray(img_rgb)
            read_time = time.time() - read_start
            data_queue.put((fname, img_rgb, read_time))
            filename_queue.task_done()

    def writer_worker():
        while True:
            item = save_queue.get()
            if item is None:
                save_queue.task_done()
                break
            image_name, normal_tensor, raw_mask_tensor, depth_tensor = item
            base_name = os.path.splitext(image_name)[0]
            save_start = time.time()

            mask_np_raw = None
            if raw_mask_tensor is not None:
                mask_np_raw = to_numpy(raw_mask_tensor)
                if mask_np_raw.dtype != np.bool_:
                    mask_np_raw = mask_np_raw > 0.5

            if normal_tensor is not None and normal_tensor.numel() > 0:
                normal_np = normal_tensor.numpy()
                normal_np = normal_np * [-0.5, -0.5, -0.5] + 0.5
                normal_uint8 = np.clip(normal_np * 255.0, 0, 255).astype(np.uint8)

                if mask_np_raw is not None:
                    mask_bool = (
                        mask_np_raw[:, :, None]
                        if mask_np_raw.ndim == 2
                        else mask_np_raw
                    )
                    mask_bool = np.broadcast_to(mask_bool, normal_uint8.shape)
                    normal_uint8 = normal_uint8.copy()
                    normal_uint8[~mask_bool] = 0

                output_path = os.path.join(normal_output_dir, f"{base_name}.png")
                success = cv2.imwrite(
                    output_path, cv2.cvtColor(normal_uint8, cv2.COLOR_RGB2BGR)
                )
                if not success:
                    print(f"[WARN] Failed to write normal map {output_path}")

            depth_mask_np = mask_np_raw.copy() if mask_np_raw is not None else None
            depth_np = depth_tensor.numpy() if depth_tensor is not None else None

            if (
                align_depth_with_colmap
                and depth_np is not None
                and colmap_meta is not None
            ):
                entry = colmap_meta.get(image_name)
                if entry is None and colmap_meta_norm is not None:
                    entry = colmap_meta_norm.get(normalize_image_key(image_name))
                if entry is None:
                    entry = None
            else:
                entry = None

            if entry is not None:
                h, w = entry["hw"]
                if depth_np.shape != (h, w):
                    depth_np = cv2.resize(
                        depth_np, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                if depth_mask_np is not None and depth_mask_np.shape != (h, w):
                    depth_mask_np = cv2.resize(
                        depth_mask_np.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                sdpt = get_sparse_depth(
                    colmap_points,
                    entry["ixt"],
                    entry["ext"],
                    entry["point_ids"],
                    h,
                    w,
                )
                if sdpt is not None:
                    sparse_depth = sdpt[..., 0]
                    valid_sparse = sparse_depth > 0
                    if np.count_nonzero(valid_sparse) >= colmap_min_sparse:
                        d_pred = depth_np[valid_sparse]
                        d_colmap = sparse_depth[valid_sparse]
                        clip_mask = np.ones_like(d_colmap, dtype=bool)
                        if 0 <= colmap_clip_low < colmap_clip_high <= 100:
                            lo, hi = np.percentile(
                                d_colmap, [colmap_clip_low, colmap_clip_high]
                            )
                            if hi > lo:
                                clip_mask = (d_colmap >= lo) & (d_colmap <= hi)
                        d_pred_clip = d_pred[clip_mask].reshape(-1, 1)
                        d_colmap_clip = d_colmap[clip_mask].reshape(-1, 1)
                        if len(d_pred_clip) >= colmap_min_sparse:
                            try:
                                ransac = RANSACRegressor(
                                    LinearRegression(),
                                    max_trials=colmap_ransac_trials,
                                )
                                ransac.fit(d_pred_clip, d_colmap_clip)
                                coef = float(ransac.estimator_.coef_.ravel()[0])
                                intercept = float(
                                    ransac.estimator_.intercept_.ravel()[0]
                                )
                                if verbose:
                                    print(
                                        f"[COLMAP] {image_name}: depth = {coef:.6f} * pred + {intercept:.6f}"
                                    )
                                depth_np = depth_np * coef + intercept
                            except Exception as exc:
                                print(f"[WARN] COLMAP alignment failed for {image_name}: {exc}")

            if depth_np is not None and remove_depth_edge:
                edge_mask = utils3d.numpy.depth_edge(
                    depth_np, rtol=depth_edge_rtol, mask=depth_mask_np
                )
                depth_mask_np = (
                    (~edge_mask)
                    if depth_mask_np is None
                    else (depth_mask_np & (~edge_mask))
                )

            if save_depth and depth_np is not None and depth_output_dir is not None:
                depth_to_save = depth_np.astype(np.float32)
                depth_to_save = np.nan_to_num(
                    depth_to_save, nan=0.0, posinf=0.0, neginf=0.0
                )
                if depth_mask_np is not None:
                    depth_to_save = depth_to_save.copy()
                    depth_to_save[~depth_mask_np] = 0.0
                depth_path = os.path.join(depth_output_dir, f"{base_name}.npy")
                try:
                    np.save(depth_path, depth_to_save.astype(np.float16))
                except Exception as exc:
                    print(f"[WARN] Failed to write depth map {depth_path}: {exc}")

            if (
                save_depth_vis
                and depth_np is not None
                and depth_vis_output_dir is not None
            ):
                valid_mask = (
                    depth_mask_np
                    if depth_mask_np is not None
                    else np.isfinite(depth_np)
                )
                valid_values = depth_np[valid_mask]

                if valid_values.size == 0:
                    depth_norm = np.zeros_like(depth_np, dtype=np.uint8)
                else:
                    vmin = float(valid_values.min())
                    vmax = float(valid_values.max())
                    if vmax - vmin < 1e-6:
                        depth_norm = np.zeros_like(depth_np, dtype=np.uint8)
                    else:
                        depth_norm = (depth_np - vmin) / (vmax - vmin)
                        depth_norm = np.clip(depth_norm, 0.0, 1.0)
                        depth_norm = np.nan_to_num(
                            depth_norm, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        depth_norm = (depth_norm * 255.0).astype(np.uint8)

                depth_norm = depth_norm.astype(np.uint8)
                if depth_mask_np is not None:
                    depth_norm = depth_norm.copy()
                    depth_norm[~depth_mask_np] = 0
                cmap = (
                    cv2.COLORMAP_TURBO
                    if hasattr(cv2, "COLORMAP_TURBO")
                    else cv2.COLORMAP_JET
                )
                depth_color = cv2.applyColorMap(depth_norm, cmap)
                if depth_mask_np is not None:
                    depth_color = depth_color.copy()
                    depth_color[~depth_mask_np] = 255
                vis_path = os.path.join(depth_vis_output_dir, f"{base_name}.png")
                success = cv2.imwrite(vis_path, depth_color)
                if not success:
                    print(f"[WARN] Failed to write depth visualization {vis_path}")

            save_time = time.time() - save_start
            with stats_lock:
                save_stats["time"] += save_time
                save_stats["count"] += 1
            save_queue.task_done()

    reader_threads = [
        threading.Thread(target=reader_worker, name=f"reader-{idx}", daemon=True)
        for idx in range(read_threads)
    ]
    writer_threads = [
        threading.Thread(target=writer_worker, name=f"writer-{idx}", daemon=True)
        for idx in range(write_threads)
    ]

    for t in reader_threads + writer_threads:
        t.start()

    processed = 0
    pbar = tqdm(total=len(image_files), desc="MoGe inference", unit="img")

    try:
        while processed < len(image_files):
            fname, img_rgb, read_time = data_queue.get()
            total_read_time += read_time

            if img_rgb is None:
                tqdm.write(f"Skipping {fname} (failed to read)")
                processed += 1
                pbar.update(1)
                continue

            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device)

            inference_start = time.time()
            with torch.no_grad():
                output = model.infer(tensor)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            """
            `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
            The maps are in the same size as the input image. 
            {
                "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
                "depth": (H, W),        # depth map
                "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
                "mask": (H, W),         # a binary mask for valid pixels. 
                "intrinsics": (3, 3),   # normalized camera intrinsics
            }
            """

            mask_tensor = output.get("mask")
            mask_cpu = mask_tensor.detach().cpu() if mask_tensor is not None else None

            normal = output.get("normal")
            normal_cpu = normal.detach().cpu() if normal is not None else None

            depth_tensor = None
            if (save_depth or save_depth_vis) and output.get("depth") is not None:
                depth_tensor = output["depth"].detach().cpu()

            if normal_cpu is None and depth_tensor is None:
                processed += 1
                pbar.update(1)
                continue
            save_queue.put((fname, normal_cpu, mask_cpu, depth_tensor))

            processed += 1
            pbar.update(1)

            if verbose and (processed % 10 == 0 or processed == len(image_files)):
                with stats_lock:
                    avg_save = (
                        save_stats["time"] / save_stats["count"]
                        if save_stats["count"] > 0
                        else 0.0
                    )
                tqdm.write(
                    f"Image {fname}: read {read_time:.3f}s, inference {inference_time:.3f}s, "
                    f"avg save {avg_save:.3f}s"
                )

    finally:
        pbar.close()

    filename_queue.join()

    save_queue.join()
    for _ in writer_threads:
        save_queue.put(None)
    for t in writer_threads:
        t.join()

    for t in reader_threads:
        t.join()

    total_save_time = save_stats["time"]

    print("\n" + "=" * 50)
    print("Processing complete! Detailed timing stats:")
    print(f"Model load: {model_load_time:.2f}s")
    print(f"Total image read time: {total_read_time:.2f}s")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Total save time: {total_save_time:.2f}s")
    print(f"Images processed: {len(image_files)}")
    if len(image_files) > 0:
        total_avg = (total_read_time + total_inference_time + total_save_time) / len(
            image_files
        )
        print(f"Average per-image time: {total_avg:.2f}s")
        print(f"Average inference time: {total_inference_time / len(image_files):.2f}s")
        if save_stats["count"]:
            print(f"Average save time: {total_save_time / save_stats['count']:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded MoGe-2 normal inference")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data root containing image subdirectories, e.g., /path/to/data",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="Subdirectory under data_dir that stores images (default: images)",
    )
    parser.add_argument(
        "--read_threads",
        type=int,
        default=2,
        help="Number of threads for parallel image reading (default: 2)",
    )
    parser.add_argument(
        "--write_threads",
        type=int,
        default=5,
        help="Number of threads for writing normals (default: 2)",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=8,
        help="Reader/writer queue capacity (default: 8)",
    )
    parser.add_argument(
        "--save_depth",
        action="store_true",
        help="Save model depth outputs as float16 .npy (default: disabled)",
    )
    parser.add_argument(
        "--save_depth_vis",
        action="store_true",
        help="Save depth visualization PNGs (default: disabled)",
    )
    parser.add_argument(
        "--remove_depth_edge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable removal of depth edge pixels (default on; disable with --no-remove_depth_edge)",
    )
    parser.add_argument(
        "--depth_edge_rtol",
        type=float,
        default=0.04,
        help="Relative threshold rtol for edge detection (default: 0.04)",
    )
    parser.add_argument(
        "--align_depth_with_colmap",
        action="store_true",
        help="Align depth with COLMAP sparse depths (default: off)",
    )
    parser.add_argument(
        "--data_factor",
        type=float,
        default=1.0,
        help="Scale factor between dataset images and COLMAP originals (>1 means downsampled)",
    )
    parser.add_argument(
        "--colmap_min_sparse",
        type=int,
        default=30,
        help="Minimum sparse points needed for RANSAC alignment (default: 30)",
    )
    parser.add_argument(
        "--colmap_clip_low",
        type=float,
        default=5.0,
        help="Lower percentile clip for sparse depth (percent), default 5%",
    )
    parser.add_argument(
        "--colmap_clip_high",
        type=float,
        default=95.0,
        help="Upper percentile clip for sparse depth (percent), default 95%",
    )
    parser.add_argument(
        "--colmap_ransac_trials",
        type=int,
        default=2000,
        help="Maximum RANSAC iterations (default: 2000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose timing and alignment stats",
    )
    args = parser.parse_args()

    run_inference(
        data_dir=args.data_dir,
        image_dir_name=args.image_dir,
        read_threads=args.read_threads,
        write_threads=args.write_threads,
        queue_size=args.queue_size,
        save_depth=args.save_depth,
        save_depth_vis=args.save_depth_vis,
        remove_depth_edge=args.remove_depth_edge,
        depth_edge_rtol=args.depth_edge_rtol,
        align_depth_with_colmap=args.align_depth_with_colmap,
        data_factor=args.data_factor,
        verbose=args.verbose,
        colmap_min_sparse=args.colmap_min_sparse,
        colmap_clip_low=args.colmap_clip_low,
        colmap_clip_high=args.colmap_clip_high,
        colmap_ransac_trials=args.colmap_ransac_trials,
    )
