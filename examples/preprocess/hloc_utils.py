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
from typing import Any, List, Literal, Optional, Tuple, Set

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
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


PANO_CONFIG = {
    "fov": 100.0,
    "views": [
        (0.0, 0.0),
        (90.0, 0.0),
        (180.0, 0.0),
        (-90.0, 0.0),
        (0.0, 90.0),
    ],
}


def get_rig_rotations() -> List[np.ndarray]:
    """Return panorama rig rotations defined in PANO_CONFIG."""

    rotations: List[np.ndarray] = []
    for yaw, pitch in PANO_CONFIG["views"]:
        rot = Rotation.from_euler("XY", [-pitch, -yaw], degrees=True).as_matrix()
        rotations.append(rot)
    return rotations


def create_pano_rig_config(ref_idx: int = 0):
    """Create a pycolmap RigConfig describing the five-view panorama setup."""

    try:
        import pycolmap
    except ImportError:
        return None

    cams_from_pano = get_rig_rotations()
    rig_cameras = []
    for idx, cam_from_pano_R in enumerate(cams_from_pano):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_R = cam_from_pano_R @ cams_from_pano[ref_idx].T
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_R), np.zeros(3)
            )

        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=(idx == ref_idx),
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


MODEL_FILENAMES = (
    "cameras.bin",
    "images.bin",
    "points3D.bin",
    "frames.bin",
    "rigs.bin",
)


def _get_candidate_model_dirs(sparse_root: Path) -> List[Path]:
    """Return all folders under sparse_root that look like COLMAP models."""

    def looks_like_model_dir(path: Path) -> bool:
        return path.is_dir() and (path / "cameras.bin").exists()

    candidates: List[Path] = []
    seen: Set[Path] = set()

    def add_candidate(path: Path) -> None:
        if path in seen or not looks_like_model_dir(path):
            return
        seen.add(path)
        candidates.append(path)

    add_candidate(sparse_root)
    if sparse_root.exists():
        for cameras_file in sorted(sparse_root.rglob("cameras.bin")):
            add_candidate(cameras_file.parent)
    return candidates


def _select_largest_colmap_model(
    pycolmap_module: Any, sparse_root: Path
) -> Tuple[Path, int]:
    """Pick the model with the most registered images and move it to sparse_root."""
    candidates = _get_candidate_model_dirs(sparse_root)
    if not candidates:
        raise RuntimeError(f"No COLMAP models found under {sparse_root}")

    model_stats: List[Tuple[Path, int]] = []
    for model_dir in candidates:
        try:
            reconstruction = pycolmap_module.Reconstruction(str(model_dir))
            model_stats.append((model_dir, reconstruction.num_reg_images()))
        except Exception as exc:  # pragma: no cover - logging best effort
            CONSOLE.print(
                f"[bold red]Warning:[/bold red] Failed to read model {model_dir}: {exc}"
            )
    if not model_stats:
        raise RuntimeError(
            f"Unable to read any COLMAP reconstructions under {sparse_root}"
        )
    best_dir, best_count = max(model_stats, key=lambda item: item[1])

    if best_dir != sparse_root:
        CONSOLE.print(
            f"[bold yellow]↻ Detected largest model in {best_dir}. Moving files to {sparse_root}.[/bold yellow]"
        )
        sparse_root.mkdir(parents=True, exist_ok=True)
        for filename in MODEL_FILENAMES:
            src = best_dir / filename
            if not src.exists():
                continue
            dst = sparse_root / filename
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
        best_dir = sparse_root

    return best_dir, best_count


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
    is_panorama: bool = False,
    enable_gpu_ba: bool = False,
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
        enable_gpu_ba: Whether to enable GPU bundle adjustment in pycolmap.
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

    if is_panorama and camera_model not in (
        CameraModel.PINHOLE,
        CameraModel.SIMPLE_PINHOLE,
    ):
        CONSOLE.print(
            "[bold yellow]⚠️ Panorama mode currently supports PINHOLE/SIMPLE_PINHOLE only."
            " Forcing camera_model=PINHOLE.[/bold yellow]"
        )
        camera_model = CameraModel.PINHOLE

    outputs = colmap_dir
    sfm_pairs = outputs / "pairs-netvlad.txt"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    # Let COLMAP write every model under distorted/sparse
    sfm_root = image_dir.parent / "sparse"
    if sfm_root.exists():
        shutil.rmtree(sfm_root)
    sfm_root.mkdir(parents=True, exist_ok=True)

    retrieval_conf = extract_features.confs["netvlad"]  # type: ignore
    feature_conf = extract_features.confs[feature_type]  # type: ignore
    matcher_conf = match_features.confs[matcher_type]  # type: ignore

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    references = sorted(
        [
            p.relative_to(image_dir).as_posix()
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
    )
    extract_features.main(feature_conf, image_dir, image_list=references, feature_path=features)  # type: ignore
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(  # type: ignore
            sfm_pairs,
            image_list=references,
            groupby_folder=is_panorama,
        )
    elif matching_method == "sequential":
        # Use sequential matching with specified parameters
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)
        pairs_from_sequential.main(
            output=sfm_pairs,
            image_list=references,
            window_size=8,
            quadratic_overlap=True,
            use_loop_closure=True,
            retrieval_path=retrieval_path,
            retrieval_interval=2,
            num_loc=3,
            groupby_folder=is_panorama,
        )
    else:
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)  # type: ignore
        num_matched = min(len(references), num_matched)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)  # type: ignore
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)  # type: ignore

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)  # type: ignore
    mapper_opts: Optional[Any] = None
    rig_config = None

    # Configure IncrementalPipelineOptions following current pycolmap API.
    try:
        mapper_opts_candidate = pycolmap.IncrementalPipelineOptions()
    except Exception:
        mapper_opts_candidate = None

    if mapper_opts_candidate is not None:
        mapper_opts = mapper_opts_candidate
        if hasattr(mapper_opts, "ba_use_gpu"):
            mapper_opts.ba_use_gpu = enable_gpu_ba
        if enable_gpu_ba and hasattr(mapper_opts, "ba_gpu_index"):
            mapper_opts.ba_gpu_index = "0"

        if enable_gpu_ba:
            local_getter = getattr(mapper_opts, "get_local_bundle_adjustment", None)
            global_getter = getattr(mapper_opts, "get_global_bundle_adjustment", None)
            if callable(local_getter) and callable(global_getter):
                for ba_opts in (local_getter(), global_getter()):
                    ba_opts.use_gpu = True
                    ba_opts.gpu_index = "0"
                    ba_opts.min_num_images_gpu_solver = 50
            elif all(
                hasattr(mapper_opts, attr)
                for attr in ("ba_local_bundle_options", "ba_global_bundle_options")
            ):
                for attr in ("ba_local_bundle_options", "ba_global_bundle_options"):
                    ba_opts = getattr(mapper_opts, attr)
                    ba_opts.use_gpu = True
                    ba_opts.gpu_index = "0"
                    ba_opts.min_num_images_gpu_solver = 50
            else:
                CONSOLE.print(
                    "[bold yellow]⚠️ Unable to locate bundle adjustment accessors on pycolmap "
                    "IncrementalPipelineOptions; GPU BA settings will not be customized.[/bold yellow]"
                )
    else:
        mapper_opts = {}

    def _assign_mapper_option(option_name: str, value: Any) -> None:
        nonlocal mapper_opts
        if mapper_opts is None:
            mapper_opts = {}
        if isinstance(mapper_opts, dict):
            mapper_opts[option_name] = value
        else:
            setattr(mapper_opts, option_name, value)

    if is_panorama:
        camera_mode = pycolmap.CameraMode.PER_FOLDER  # type: ignore
        CONSOLE.print("[bold green]ℹ️ Creating Panorama Rig Configuration...[/bold green]")
        rig_config = create_pano_rig_config()

        if references:
            first_img_path = image_dir / references[0]
            img = cv2.imread(str(first_img_path))
            if img is None:
                raise RuntimeError(f"Cannot read calibration image: {first_img_path}")
            h, w = img.shape[:2]
            hfov_deg = PANO_CONFIG["fov"]
            hfov_rad = np.deg2rad(hfov_deg)
            focal = w / (2 * np.tan(hfov_rad / 2))
            cx, cy = w / 2.0 - 0.5, h / 2.0 - 0.5

            if camera_model == CameraModel.PINHOLE:
                image_options.camera_params = f"{focal},{focal},{cx},{cy}"
            elif camera_model == CameraModel.SIMPLE_PINHOLE:
                image_options.camera_params = f"{focal},{cx},{cy}"

            CONSOLE.print(
                f"[bold green]ℹ️ Initialized Intrinsics: {w}x{h}, params={image_options.camera_params}[/bold green]"
            )

        _assign_mapper_option("ba_refine_sensor_from_rig", False)
        _assign_mapper_option("ba_refine_focal_length", True)
        _assign_mapper_option("ba_refine_principal_point", False)
        _assign_mapper_option("ba_refine_extra_params", False)
    elif use_single_camera_mode:  # one camera per all frames
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
            sfm_root,
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
            sfm_root,
            image_dir,
            sfm_pairs,
            features,
            matches,
            camera_mode=camera_mode,  # type: ignore
            image_options=image_options,
            verbose=verbose,
            mapper_options=mapper_opts or None,
            rig_config=rig_config,
        )

    try:
        sfm_dir, num_reg_images = _select_largest_colmap_model(pycolmap, sfm_root)
        CONSOLE.print(
            f"[bold green]ℹ️ Selected COLMAP model {sfm_dir.name if sfm_dir != sfm_root else sfm_dir} "
            f"with {num_reg_images} registered images.[/bold green]"
        )
    except RuntimeError as err:
        CONSOLE.print(f"[bold red]Error:[/bold red] {err}")
        sys.exit(1)

    # =========================================================================
    # Post-processing: Image Undistortion OR Data Migration
    # =========================================================================
    CONSOLE.print("\n[bold yellow]🚀 Starting post-processing...[/bold yellow]")

    if image_dir.parent.name == "distorted":
        project_root = image_dir.parent.parent
    else:
        project_root = image_dir.parent
        CONSOLE.print(
            "[bold red]Warning: Unexpected directory structure. Assuming parent as root.[/bold red]"
        )

    target_images_dir = project_root / "images"
    target_sparse_dir = project_root / "sparse" / "0"

    if is_panorama:
        CONSOLE.print(
            "[bold green]ℹ️ Processing Panorama: Flattening directory structure for 3DGS...[/bold green]"
        )

        if target_images_dir.exists():
            shutil.rmtree(target_images_dir)
        target_images_dir.mkdir(parents=True, exist_ok=True)

        if target_sparse_dir.exists():
            shutil.rmtree(target_sparse_dir)
        target_sparse_dir.mkdir(parents=True, exist_ok=True)

        name_map: dict[str, str] = {}
        src_images = sorted(
            [
                p
                for p in image_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in valid_extensions
            ]
        )
        CONSOLE.print(f"   > Flattening and copying {len(src_images)} images...")
        for src_path in src_images:
            rel_path = src_path.relative_to(image_dir)
            new_name = rel_path.as_posix().replace("/", "__")
            dst_path = target_images_dir / new_name
            shutil.copy2(src_path, dst_path)
            name_map[rel_path.as_posix()] = new_name

        CONSOLE.print("   > Updating COLMAP model...")
        try:
            recon = pycolmap.Reconstruction(sfm_dir)
            for image in recon.images.values():
                if image.name in name_map:
                    image.name = name_map[image.name]
                else:
                    CONSOLE.print(
                        f"[bold yellow]Warning: Model image {image.name} not found in source folders.[/bold yellow]"
                    )
            recon.write(target_sparse_dir)
            CONSOLE.print(
                f"[bold green]✅ Panorama data ready at {target_images_dir}[/bold green]"
            )
        except Exception as err:
            CONSOLE.print(f"[bold red]❌ Failed to update panorama model: {err}[/bold red]")

    elif camera_model in [CameraModel.PINHOLE, CameraModel.SIMPLE_PINHOLE]:
        CONSOLE.print(
            f"[bold green]ℹ️ Camera model is {camera_model.name}. Skipping undistortion.[/bold green]"
        )
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
        CONSOLE.print(
            f"[bold yellow]🔧 Running pycolmap.undistort_images for {camera_model.name}...[/bold yellow]"
        )

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
                    CONSOLE.print(
                        f"[bold red]Warning:[/bold red] Failed to remove {stereo_dir}: {cleanup_err}"
                    )

            CONSOLE.print(f"[bold green]✅ Undistortion complete![/bold green]")
        except Exception as err:  # pragma: no cover - best effort logging
            CONSOLE.print(f"[bold red]❌ Undistortion failed: {err}[/bold red]")
            return

    CONSOLE.print(f"  > 3DGS Ready Images: {target_images_dir}")
    CONSOLE.print(f"  > 3DGS Ready Sparse: {target_sparse_dir}")
