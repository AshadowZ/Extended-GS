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
from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import shutil

from hloc_utils import run_hloc, CameraModel


def copy_images_fast(image_dir: Path, image_prefix: str = "frame_") -> Path:
    new_dir = image_dir.parent / "images"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in image_exts]
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"📁 Copying {len(image_paths)} images using fast mode...")

    def copy_one(idx_path):
        idx, src_path = idx_path
        dst_path = new_dir / f"{image_prefix}{idx:05d}{src_path.suffix.lower()}"
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
    return new_dir


def main():
    parser = argparse.ArgumentParser(description="Run SfM using HLOC toolbox")

    parser.add_argument(
        "--input_image_dir",
        type=Path,
        required=True,
        help="Path to the original input image directory.",
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
    parser.add_argument(
        "--feature_type",
        type=str,
        default="superpoint_aachen",
        choices=[
            "superpoint_aachen",
            "superpoint_max",
            "superpoint_inloc",
            "r2d2",
            "d2net-ss",
            "sift",
            "sosnet",
            "disk",
            "aliked-n16",
            "xfeat",
        ],
        help="Feature extractor type.",
    )
    parser.add_argument(
        "--matcher_type",
        type=str,
        default="superglue",
        choices=[
            "superpoint+lightglue",
            "disk+lightglue",
            "aliked+lightglue",
            "xfeat+lighterglue",
            "superglue",
            "superglue-fast",
            "NN-superpoint",
            "NN-ratio",
            "NN-mutual",
            "adalam",
        ],
        help="Feature matcher type.",
    )
    parser.add_argument(
        "--use_single_camera_mode",
        type=bool,
        default=True,
        help="If True, assume all images share one camera model.",
    )

    args = parser.parse_args()

    # ① 拷贝并标准化图像目录（带进度条）
    standardized_dir = copy_images_fast(args.input_image_dir)

    # ② 定义输出目录
    colmap_dir = standardized_dir.parent / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # ③ 运行 HLOC SfM
    print(f"\n🚀 Running HLOC SfM on {standardized_dir} ...")
    run_hloc(
        image_dir=standardized_dir,
        colmap_dir=colmap_dir,
        camera_model=CameraModel[args.camera_model],
        matching_method=args.matching_method,
        feature_type=args.feature_type,
        matcher_type=args.matcher_type,
        use_single_camera_mode=args.use_single_camera_mode,
    )

    print(f"\n✅ SfM completed!\nResults saved to:\n  {colmap_dir}")
    print(f"  sparse output: {standardized_dir.parent / 'sparse' / '0'}")


if __name__ == "__main__":
    main()
