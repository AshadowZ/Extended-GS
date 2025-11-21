import json
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from .colmap_util import read_model, get_intrinsics, get_hws, get_extrinsic
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        depth_dir_name: Optional[str] = None,
        normal_dir_name: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        # Read COLMAP model using colmap_util
        cameras, images, points3D = read_model(colmap_dir)

        # Extract extrinsic matrices in camera-to-world format.
        imdata = images
        c2w_mats = []  # camera-to-world matrices
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        camtype_dict = dict()  # camera_id -> camtype
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for img in images.values():
            ext = get_extrinsic(img)  # This returns world-to-camera (w2c) matrix
            c2w = np.linalg.inv(ext)
            c2w_mats.append(c2w)  # Store as c2w

            # support different camera intrinsics
            camera_id = img.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras[camera_id]
            K = get_intrinsics(cam)
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.model
            # print(f"[Parser] camera type is {type_}")
            if type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == "SIMPLE_RADIAL":
                params = np.array([cam.params[3], 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == "RADIAL":
                params = np.array(
                    [cam.params[3], cam.params[4], 0.0, 0.0], dtype=np.float32
                )
                camtype = "perspective"
            elif type_ == "OPENCV":
                params = np.array(
                    [cam.params[4], cam.params[5], cam.params[6], cam.params[7]],
                    dtype=np.float32,
                )
                camtype = "perspective"
            elif type_ == "OPENCV_FISHEYE":
                params = np.array(
                    [cam.params[4], cam.params[5], cam.params[6], cam.params[7]],
                    dtype=np.float32,
                )
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            camtype_dict[camera_id] = camtype
            h, w = get_hws(cam)
            imsize_dict[camera_id] = (w // factor, h // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        # Check if any camera has distortion (non-PINHOLE models)
        has_distortion = any(  # error
            camtype != "perspective" for camtype in camtype_dict.values()
        )
        if has_distortion:
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        c2w_mats = np.stack(c2w_mats, axis=0)

        # camtoworlds already in camera-to-world format, no conversion needed
        camtoworlds = c2w_mats

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [img.name for img in images.values()]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = np.array([pt.xyz for pt in points3D.values()], dtype=np.float32)
        points_err = np.array([pt.error for pt in points3D.values()], dtype=np.float32)
        points_rgb = np.array([pt.rgb for pt in points3D.values()], dtype=np.uint8)
        point_indices = dict()

        # Create fast mapping from point ID to index to avoid O(n) linear search
        point_id_to_idx = {
            point_id: idx for idx, point_id in enumerate(points3D.keys())
        }

        image_id_to_name = {img.id: img.name for img in images.values()}
        for point_id, point_data in points3D.items():
            for image_id, _ in zip(point_data.image_ids, point_data.point2D_idxs):
                image_name = image_id_to_name[image_id]
                point_idx = point_id_to_idx[point_id]  # O(1) lookup
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1, scale = similarity_from_cameras(camtoworlds, strict_scaling=False)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)
            scale = 1.0

        self.scale = scale
        print(f"[Parser] colmap.py scale: {self.scale}")

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.camtype_dict = camtype_dict  # Dict of camera_id -> camtype
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            camtype = self.camtype_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Process depth map paths (hardcoded .npy extension)
        if depth_dir_name is None:
            print("[Parser] No depth directory name provided. Skipping depth priors.")
            self.depth_paths = None
        else:
            depth_dir = os.path.join(self.data_dir, depth_dir_name)
            self.depth_paths = []
            print(f"[Parser] Building depth paths from: {depth_dir} (ext: .npy)")
            # self.image_names is already a sorted list
            for img_name in self.image_names:
                base_name, _ = os.path.splitext(img_name)
                # Hardcoded .npy extension
                path = os.path.join(depth_dir, base_name + ".npy")
                self.depth_paths.append(path)

        # Process normal map paths (hardcoded .png extension)
        if normal_dir_name is None:
            print("[Parser] No normal directory name provided. Skipping normal priors.")
            self.normal_paths = None
        else:
            normal_dir = os.path.join(self.data_dir, normal_dir_name)
            self.normal_paths = []
            print(f"[Parser] Building normal paths from: {normal_dir} (ext: .png)")
            # self.image_names is already a sorted list
            for img_name in self.image_names:
                base_name, _ = os.path.splitext(img_name)
                # Hardcoded .png extension
                path = os.path.join(normal_dir, base_name + ".png")
                self.normal_paths.append(path)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices
            # self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        depth_data = None
        normal_data = None

        if self.parser.depth_paths is not None:
            depth_path = self.parser.depth_paths[index]
            try:
                depth_data = np.load(depth_path).astype(np.float32)  # Known to be .npy
                depth_data = depth_data * self.parser.scale
            except Exception as e:
                print(f"Warning: Could not load depth {depth_path}: {e}")
        if self.parser.normal_paths is not None:
            normal_path = self.parser.normal_paths[index]
            try:
                normal_data = imageio.imread(normal_path)[..., :3]  # Known to be .png
            except Exception as e:
                print(f"Warning: Could not load normal {normal_path}: {e}")

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

            # Apply to depth (Note: INTER_NEAREST)
            if depth_data is not None:
                depth_data = cv2.remap(depth_data, mapx, mapy, cv2.INTER_NEAREST)
                depth_data = depth_data[y : y + h, x : x + w]
            # Apply to normal
            if normal_data is not None:
                normal_data = cv2.remap(normal_data, mapx, mapy, cv2.INTER_LINEAR)
                normal_data = normal_data[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]

            # Apply to depth (using same x, y)
            if depth_data is not None:
                depth_data = depth_data[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
            # Apply to normal (using same x, y)
            if normal_data is not None:
                normal_data = normal_data[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
            # Apply to mask (if exists)
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]

            K[0, 2] -= x
            K[1, 2] -= y

        if depth_data is not None:
            depth_prior = torch.from_numpy(depth_data).float().unsqueeze(-1)
        else:
            depth_prior = torch.empty(0)
        if normal_data is not None:  # needs to be inverted, to opencv coord
            normal_prior = torch.from_numpy(normal_data).float() / 255.0
            normal_prior = 1.0 - normal_prior * 2.0
        else:
            normal_prior = torch.empty(0)

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "depth_prior": depth_prior,  # [H, W, 1]
            "normal_prior": normal_prior,  # [H, W, 3]
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
