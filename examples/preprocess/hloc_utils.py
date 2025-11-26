"""
Code that uses the hierarchical localization toolbox (hloc)
to extract and match image features, estimate camera poses,
and do sparse reconstruction.
Requires hloc module from : https://github.com/cvg/Hierarchical-Localization
"""

# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import shutil
from pathlib import Path
from typing import Literal

from enum import Enum  # Import Enum since CameraModel extends it
from rich.console import Console


class CameraModel(Enum):
    """Enum for camera types."""

    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"
    RADIAL = "RADIAL"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    OPENCV = "OPENCV"
    FULL_OPENCV = "FULL_OPENCV"
    # pycolmap also supports other fisheye models such as OPENCV_FISHEYE,
    # FOV, etc., but this script assumes a distortion-free camera by default.


CONSOLE = Console(width=120)


def run_hloc(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    matching_method: Literal["exhaustive", "sequential"] = "sequential",
    feature_type: Literal[
        "superpoint_aachen",
        "superpoint_max",
        "superpoint_inloc",
        "r2d2",
        "d2net-ss",
        "sift",
        "sosnet",
        "disk",
        "aliked-n16",
    ] = "superpoint_aachen",
    matcher_type: Literal[
        "superpoint+lightglue",
        "disk+lightglue",
        "aliked+lightglue",
        "superglue",
        "superglue-fast",
        "NN-superpoint",
        "NN-ratio",
        "NN-mutual",
        "adalam",
    ] = "superglue",
    num_matched: int = 50,
    refine_pixsfm: bool = False,
    use_single_camera_mode: bool = True,
) -> None:
    """Runs hloc on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        verbose: If True, logs the output of the command.
        matching_method: Method to use for matching images.
        feature_type: Type of visual features to use.
        matcher_type: Type of feature matcher to use.
        num_matched: Number of image pairs for loc.
        refine_pixsfm: If True, refine the reconstruction using pixel-perfect-sfm.
        use_single_camera_mode: If True, uses one camera for all frames. Otherwise uses one camera per frame.
    """

    try:
        # TODO(1480) un-hide pycolmap import
        import pycolmap
        from hloc import (  # type: ignore
            extract_features,
            match_features,
            pairs_from_exhaustive,
            pairs_from_retrieval,
            pairs_from_sequential,
            reconstruction,
        )
    except ImportError:
        _HAS_HLOC = False
    else:
        _HAS_HLOC = True

    try:
        from pixsfm.refine_hloc import PixSfM  # type: ignore
    except ImportError:
        _HAS_PIXSFM = False
    else:
        _HAS_PIXSFM = True

    if not _HAS_HLOC:
        CONSOLE.print(
            f"[bold red]Error: To use this set of parameters ({feature_type}/{matcher_type}/hloc), "
            "you must install hloc toolbox!!"
        )
        sys.exit(1)

    if refine_pixsfm and not _HAS_PIXSFM:
        CONSOLE.print(
            "[bold red]Error: use refine_pixsfm, you must install pixel-perfect-sfm toolbox!!"
        )
        sys.exit(1)

    outputs = colmap_dir
    sfm_pairs = outputs / "pairs-netvlad.txt"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    # Place sparse/0 alongside the image directory
    sfm_dir = image_dir.parent / "sparse" / "0"

    retrieval_conf = extract_features.confs["netvlad"]  # type: ignore
    feature_conf = extract_features.confs[feature_type]  # type: ignore
    matcher_conf = match_features.confs[matcher_type]  # type: ignore

    references = [p.relative_to(image_dir).as_posix() for p in image_dir.iterdir()]
    extract_features.main(feature_conf, image_dir, image_list=references, feature_path=features)  # type: ignore
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)  # type: ignore
    elif matching_method == "sequential":
        # Use sequential matching with specified parameters
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)
        pairs_from_sequential.main(
            output=sfm_pairs,
            image_list=references,
            window_size=6,
            quadratic_overlap=True,
            use_loop_closure=True,
            retrieval_path=retrieval_path,
            retrieval_interval=2,
            num_loc=5,
        )
    else:
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)  # type: ignore
        num_matched = min(len(references), num_matched)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)  # type: ignore
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)  # type: ignore

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)  # type: ignore

    if use_single_camera_mode:  # one camera per all frames
        camera_mode = pycolmap.CameraMode.SINGLE  # type: ignore
    else:  # one camera per frame
        camera_mode = pycolmap.CameraMode.PER_IMAGE  # type: ignore

    if refine_pixsfm:
        sfm = PixSfM(  # type: ignore
            conf={
                "dense_features": {"use_cache": True},
                "KA": {
                    "dense_features": {"use_cache": True},
                    "max_kps_per_problem": 1000,
                },
                "BA": {"strategy": "costmaps"},
            }
        )
        refined, _ = sfm.reconstruction(
            sfm_dir,
            image_dir,
            sfm_pairs,
            features,
            matches,
            image_list=references,
            camera_mode=camera_mode,  # type: ignore
            image_options=image_options,
            verbose=verbose,
        )
        print("Refined", refined.summary())

    else:
        reconstruction.main(  # type: ignore
            sfm_dir,
            image_dir,
            sfm_pairs,
            features,
            matches,
            camera_mode=camera_mode,  # type: ignore
            image_options=image_options,
            verbose=verbose,
        )

    # =========================================================================
    # Post-processing: Image Undistortion OR Data Migration
    # =========================================================================
    CONSOLE.print("\n[bold yellow]🚀 Starting post-processing...[/bold yellow]")

    if image_dir.parent.name == "distorted":
        project_root = image_dir.parent.parent
    else:
        project_root = image_dir.parent
        CONSOLE.print("[bold red]Warning: Unexpected directory structure. Assuming parent as root.[/bold red]")

    target_images_dir = project_root / "images"
    target_sparse_dir = project_root / "sparse" / "0"

    if camera_model in [CameraModel.PINHOLE, CameraModel.SIMPLE_PINHOLE]:
        CONSOLE.print(f"[bold green]ℹ️ Camera model is {camera_model.name}. Skipping undistortion.[/bold green]")
        CONSOLE.print("🔄 Migrating data to 3DGS standard format...")

        if target_images_dir.exists():
            shutil.rmtree(target_images_dir)
        shutil.copytree(image_dir, target_images_dir)

        if target_sparse_dir.exists():
            shutil.rmtree(target_sparse_dir)
        target_sparse_dir.mkdir(parents=True, exist_ok=True)

        for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
            src = sfm_dir / filename
            dst = target_sparse_dir / filename
            if src.exists():
                shutil.copy2(src, dst)

        CONSOLE.print(f"[bold green]✅ Data migration complete![/bold green]")
    else:
        CONSOLE.print(f"[bold yellow]🔧 Running pycolmap.undistort_images for {camera_model.name}...[/bold yellow]")

        try:
            CONSOLE.print(f"   Input Model:  {sfm_dir}")
            CONSOLE.print(f"   Input Images: {image_dir}")
            CONSOLE.print(f"   Output Root:  {project_root}")

            options = pycolmap.UndistortCameraOptions()  # type: ignore
            options.max_image_size = 2000

            pycolmap.undistort_images(  # type: ignore
                output_path=str(project_root),
                input_path=str(sfm_dir),
                image_path=str(image_dir),
                output_type="COLMAP",
                undistort_options=options,
            )

            sparse_root = project_root / "sparse"
            if sparse_root.exists():
                sparse_zero = sparse_root / "0"
                sparse_zero.mkdir(parents=True, exist_ok=True)
                for item in list(sparse_root.iterdir()):
                    if item == sparse_zero:
                        continue
                    if item.suffix == ".bin":
                        shutil.move(str(item), str(sparse_zero / item.name))

            for script_name in ("run-colmap-geometric.sh", "run-colmap-photometric.sh"):
                script_path = project_root / script_name
                if script_path.exists():
                    try:
                        script_path.unlink()
                        CONSOLE.print(f"  > Removed helper script: {script_path}")
                    except OSError as cleanup_err:
                        CONSOLE.print(
                            f"[bold red]Warning:[/bold red] Failed to remove {script_path}: {cleanup_err}"
                        )

            stereo_dir = project_root / "stereo"
            if stereo_dir.exists():
                try:
                    shutil.rmtree(stereo_dir)
                    CONSOLE.print(f"  > Removed stereo directory: {stereo_dir}")
                except OSError as cleanup_err:
                    CONSOLE.print(f"[bold red]Warning:[/bold red] Failed to remove {stereo_dir}: {cleanup_err}")

            CONSOLE.print(f"[bold green]✅ Undistortion complete![/bold green]")
        except Exception as err:  # pragma: no cover - best effort logging
            CONSOLE.print(f"[bold red]❌ Undistortion failed: {err}[/bold red]")
            return

    CONSOLE.print(f"  > 3DGS Ready Images: {target_images_dir}")
    CONSOLE.print(f"  > 3DGS Ready Sparse: {target_sparse_dir}")
