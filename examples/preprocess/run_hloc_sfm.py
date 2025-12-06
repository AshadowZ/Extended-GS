#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Structure-from-Motion (SfM) using Hierarchical Localization (HLOC).

This script:
1. Copies all images from --input_image_dir into a new folder named 'images'
   in the same parent directory, with standardized names (frame_00001.jpg etc.).
   A progress bar will show copy progress.
2. Runs the HLOC SfM pipeline using the copied images.

Example:
    python run_hloc_sfm.py \
        --input_image_dir /media/joker/HV/3DGS/flyroom/input \
        --camera_model PINHOLE \
        --matching_method sequential \
        --feature_type superpoint_aachen \
        --matcher_type superpoint+lightglue \
        --use_single_camera_mode True

"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from hloc_utils import run_hloc, CameraModel, PANO_CONFIG


def copy_images_fast(
    image_dir: Path, output_root: Path, image_prefix: str = "frame_"
) -> Path:
    """Copy original images into the provided distorted/images directory."""

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in image_exts]
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"📁 Copying {len(image_paths)} images to {output_root} ...")

    def copy_one(idx_path):
        idx, src_path = idx_path
        dst_path = output_root / f"{image_prefix}{idx:05d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst_path)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(
            tqdm(
                ex.map(copy_one, enumerate(image_paths, start=1)),
                total=len(image_paths),
                unit="img",
            )
        )

    print("✅ Copy finished.")
    return output_root


def get_pano_virtual_configs():
    """Return rotations and FOV based on shared PANO_CONFIG."""

    hfov_deg = PANO_CONFIG["fov"]
    vfov_deg = PANO_CONFIG["fov"]
    rots = []
    for yaw, pitch in PANO_CONFIG["views"]:
        rot = Rotation.from_euler("XY", [-pitch, -yaw], degrees=True).as_matrix()
        rots.append(rot)

    return rots, hfov_deg, vfov_deg


def split_panoramas(
    image_dir: Path,
    output_root: Path,
    image_prefix: str = "frame_",
    downscale: float = 1.0,
) -> Path:
    """Split each panorama into five perspective views and save by camera index."""

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    pano_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in image_exts]
    )
    if not pano_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print(
        f"🧩 Detected {len(pano_paths)} panoramas. Splitting into perspective views..."
    )
    print(f"📂 Output Directory: {output_root}")

    first_img = cv2.imread(str(pano_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Cannot read image: {pano_paths[0]}")
    pano_h, pano_w = first_img.shape[:2]

    rots, hfov, vfov = get_pano_virtual_configs()
    w_virt_raw = int(pano_w * hfov / 360)
    h_virt_raw = int(pano_h * vfov / 180)
    w_virt = max(1, int(w_virt_raw / downscale))
    h_virt = max(1, int(h_virt_raw / downscale))
    print(f"📏 Resolution: {w_virt}x{h_virt} (Downscale factor: {downscale})")
    focal = w_virt / (2 * np.tan(np.deg2rad(hfov) / 2))

    cx, cy = w_virt / 2 - 0.5, h_virt / 2 - 0.5
    y_grid, x_grid = np.indices((h_virt, w_virt))
    rays = np.stack(
        [
            (x_grid - cx),
            (y_grid - cy),
            np.full_like(x_grid, focal, dtype=np.float32),
        ],
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    def process_one_pano(idx_path):
        idx, src_path = idx_path
        img = cv2.imread(str(src_path))
        if img is None:
            return

        for cam_idx, rot in enumerate(rots):
            rays_rotated = rays @ rot
            x, y, z = (
                rays_rotated[..., 0],
                rays_rotated[..., 1],
                rays_rotated[..., 2],
            )
            yaw = np.arctan2(x, z)
            pitch = -np.arctan2(y, np.linalg.norm(rays_rotated[..., [0, 2]], axis=-1))

            u = (1 + yaw / np.pi) / 2 * pano_w
            v = (1 - pitch * 2 / np.pi) / 2 * pano_h

            perspective_img = cv2.remap(
                img,
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_WRAP,
            )

            sub_dir = output_root / f"pano_camera{cam_idx}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            save_path = sub_dir / f"{image_prefix}{idx:05d}{src_path.suffix.lower()}"
            cv2.imwrite(str(save_path), perspective_img)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(
            tqdm(
                ex.map(process_one_pano, enumerate(pano_paths, start=1)),
                total=len(pano_paths),
                unit="pano",
            )
        )

    print("✅ Panorama splitting finished.")
    return output_root


def main():
    parser = argparse.ArgumentParser(description="Run SfM using HLOC toolbox")

    parser.add_argument(
        "--input_image_dir", type=Path, required=True, help="Path to original images."
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        required=True,
        choices=[m.name for m in CameraModel],
        help="Camera model (e.g., PINHOLE, OPENCV, OPENCV_FISHEYE).",
    )
    parser.add_argument(
        "--matching_method",
        type=str,
        default="sequential",
        choices=["exhaustive", "sequential", "retrieval"],
        help="Method for image matching.",
    )
    parser.add_argument("--feature_type", type=str, default="superpoint_aachen")
    parser.add_argument("--matcher_type", type=str, default="superglue")
    parser.add_argument("--use_single_camera_mode", type=bool, default=True)
    parser.add_argument(
        "--is_panorama",
        action="store_true",
        help="If set, split panoramas into perspective views and exit.",
    )
    parser.add_argument(
        "--pano_downscale",
        type=float,
        default=1.0,
        help="Downscale factor for panorama splitting (>=1).",
    )
    parser.add_argument(
        "--gpu_ba",
        type=str.lower,
        choices=["on", "off"],
        default="off",
        help="Whether to enable GPU bundle adjustment (default: off).",
    )

    args = parser.parse_args()
    distorted_images_dir = args.input_image_dir.parent / "distorted" / "images"
    enable_gpu_ba = args.gpu_ba == "on"

    if args.is_panorama:
        print("\n🔄 Mode: Panorama Processing")
        working_dir = split_panoramas(
            args.input_image_dir,
            distorted_images_dir,
            downscale=max(1.0, args.pano_downscale),
        )
    else:
        working_dir = copy_images_fast(args.input_image_dir, distorted_images_dir)

    # Step 2: configure the output directory (distorted/colmap)
    colmap_dir = working_dir.parent / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: run HLOC SfM
    print(f"\n🚀 Running HLOC SfM on {working_dir} ...")
    run_hloc(
        image_dir=working_dir,
        colmap_dir=colmap_dir,
        camera_model=CameraModel[args.camera_model],
        verbose=False,
        matching_method=args.matching_method,
        feature_type=args.feature_type,
        matcher_type=args.matcher_type,
        use_single_camera_mode=args.use_single_camera_mode,
        is_panorama=args.is_panorama,
        enable_gpu_ba=enable_gpu_ba,
    )

    project_root = args.input_image_dir.parent
    print(f"\n✅ All steps completed!")
    print(f"  [Distorted Data] : {working_dir.parent}")
    print(f"  [3DGS Training Data] :")
    print(f"     - Images: {project_root / 'images'}")
    print(f"     - Sparse: {project_root / 'sparse' / '0'}")


if __name__ == "__main__":
    main()
