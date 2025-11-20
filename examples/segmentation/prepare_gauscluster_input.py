import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------------------------------------------------------
# Ensure the examples directory (parent of this file) is on sys.path so the
# dataset utilities can be imported just like other example scripts do.
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from datasets.colmap import Dataset, Parser  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare camera, mask, and Gaussian parameters for GausCluster."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root COLMAP-style data directory (e.g. /path/to/scene).",
    )
    parser.add_argument(
        "--data_factor",
        type=int,
        default=1,
        help="Image downscale factor to match Parser behaviour (default: 1).",
    )
    parser.add_argument(
        "--normalize_world_space",
        action="store_true",
        help="Match Parser option to normalize world coordinates.",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="Keep consistent with Parser/Dataset split (default: 8).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="One or more checkpoint paths containing Gaussian parameters.",
    )
    parser.add_argument(
        "--mask_subdir",
        type=str,
        default=None,
        help="Optional override for which subfolder inside data_dir/sam to use.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the prepared payload (.pt). Defaults to <data_dir>/gauscluster_input.pt",
    )
    return parser.parse_args()


def load_gaussians_from_ckpt(
    ckpt_paths: List[str], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load and merge Gaussian parameters from checkpoint files."""
    means, quats, scales, opacities, sh0, shn = [], [], [], [], [], []

    print(f"[Gaussians] Loading {len(ckpt_paths)} checkpoint(s) on {device} ...")
    for ckpt_path in ckpt_paths:
        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        if "splats" not in payload:
            raise KeyError(f"Checkpoint missing 'splats' key: {ckpt_path}")
        splats = payload["splats"]
        means.append(splats["means"])
        quats.append(F.normalize(splats["quats"], dim=-1))
        scales.append(torch.exp(splats["scales"]))
        opacities.append(torch.sigmoid(splats["opacities"]))
        sh0.append(splats["sh0"])
        shn.append(splats["shN"])
        print(f"  • Loaded {ckpt_path}")

    def _cat(items: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(items, dim=0)

    means = _cat(means)
    quats = _cat(quats)
    scales = _cat(scales)
    opacities = _cat(opacities)
    colors = torch.cat([_cat(sh0), _cat(shn)], dim=-2)
    sh_degree = int(np.sqrt(colors.shape[-2]) - 1)

    print(
        f"[Gaussians] Total={means.shape[0]}, feature_dim={(sh_degree + 1) ** 2}, sh_degree={sh_degree}"
    )
    return means, quats, scales, opacities, colors, sh_degree


def discover_mask_dir(data_dir: str, override: Optional[str] = None) -> str:
    """Find the directory that stores SAM masks."""
    sam_root = os.path.join(data_dir, "sam")
    if not os.path.isdir(sam_root):
        raise FileNotFoundError(f"sam directory not found under {data_dir}")

    candidates = []
    if override is not None:
        candidates.append(os.path.join(sam_root, override))
    # Preferred fallbacks.
    candidates.extend(
        [
            os.path.join(sam_root, "mask_sorted"),
            os.path.join(sam_root, "mask"),
            os.path.join(sam_root, "mask_filtered"),
        ]
    )
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            print(f"[Masks] Using SAM masks from: {candidate}")
            return candidate

    raise FileNotFoundError(f"No mask directory found inside {sam_root}")


def read_mask(mask_path: str) -> np.ndarray:
    """Load a mask image/npy as integer labels."""
    if mask_path.endswith(".npy"):
        mask = np.load(mask_path)
    else:
        mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


def collect_view_data(
    parser: Parser, dataset: Dataset, mask_dir: str
) -> List[Dict[str, torch.Tensor]]:
    """Assemble per-view camera parameters and SAM masks."""
    view_data = []
    image_hw_cache: Dict[str, Tuple[int, int]] = {}
    indices = dataset.indices

    for local_idx, parser_idx in enumerate(indices):
        image_path = parser.image_paths[parser_idx]
        image_name = Path(image_path).stem

        camera_id = parser.camera_ids[parser_idx]
        width, height = parser.imsize_dict[camera_id]
        K = parser.Ks_dict[camera_id]
        camtoworld = parser.camtoworlds[parser_idx]

        mask_path = None
        for ext in (".png", ".jpg", ".jpeg", ".npy"):
            candidate = os.path.join(mask_dir, f"{image_name}{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {image_name} in {mask_dir}")

        mask_np = read_mask(mask_path)
        mask_h, mask_w = mask_np.shape

        if mask_h != height or mask_w != width:
            if image_path not in image_hw_cache:
                rgb = imageio.imread(image_path)
                image_hw_cache[image_path] = rgb.shape[:2]
            actual_h, actual_w = image_hw_cache[image_path]
            if mask_h == actual_h and mask_w == actual_w:
                height, width = actual_h, actual_w
            else:
                raise ValueError(
                    f"Mask 尺寸 {mask_h}x{mask_w} 与相机解析到的尺寸 {height}x{width} "
                    f"以及原始图像尺寸 {actual_h}x{actual_w} 均不一致：{mask_path}"
                )
        segmap = torch.from_numpy(mask_np.copy()).long()

        view_data.append(
            {
                "index": int(parser_idx),
                "image_name": image_name,
                "image_path": image_path,
                "width": int(width),
                "height": int(height),
                "K": torch.from_numpy(K).float(),
                "camtoworld": torch.from_numpy(camtoworld).float(),
                "segmap": segmap,
                "mask_path": mask_path,
            }
        )

        if (local_idx + 1) % 20 == 0 or (local_idx + 1) == len(indices):
            print(
                f"[Cameras] Prepared {local_idx + 1}/{len(indices)} views "
                f"(current: {image_name})"
                )

    return view_data


def build_mask_gaussian_tracker(
    gaussians: Tuple[torch.Tensor, ...],
    view_data: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict:
    """构建 mask -> Gaussian 跟踪：为每帧的每个 mask 找到对应的高斯点 id 列表。"""
    means, quats, scales, opacities, colors, sh_degree = gaussians
    num_points = means.shape[0]
    num_frames = len(view_data)

    gaussian_in_frame_matrix = np.zeros((num_points, num_frames), dtype=bool)
    mask_gaussian_pclds: Dict[str, np.ndarray] = {}
    frame_gaussian_ids: List[set] = []
    global_frame_mask_list: List[Tuple[int, int]] = []

    means_d = means.to(device)
    quats_d = quats.to(device)
    scales_d = scales.to(device)
    opacities_d = opacities.to(device)
    colors_d = colors.to(device)

    pbar = tqdm(view_data, desc="构建 mask→Gaussian 跟踪", total=num_frames)
    for frame_idx, view in enumerate(pbar):
        width, height = view["width"], view["height"]
        viewmats = torch.linalg.inv(view["camtoworld"].to(device)).unsqueeze(0)
        Ks = view["K"].to(device).unsqueeze(0)

        with torch.no_grad():
            _, _, meta = rasterization(
                means=means_d,
                quats=quats_d,
                scales=scales_d,
                opacities=opacities_d,
                colors=colors_d,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree,
                render_mode="RGB",
                packed=False,
                radius_clip=3.0,
                distributed=False,
                with_ut=False,
                with_eval3d=False,
                track_pixel_gaussians=True,
                max_gaussians_per_pixel=256,
                pixel_gaussian_threshold=0.0,
            )

        pixel_gaussians = meta.get("pixel_gaussians", None)
        if pixel_gaussians is None or pixel_gaussians.numel() == 0:
            raise RuntimeError(f"未在视角 {view['image_name']} 获取到像素-高斯对应关系。")

        gaus_ids = pixel_gaussians[:, 0].long()
        pixel_ids = pixel_gaussians[:, 1].long()

        segmap_flat = view["segmap"].to(device).view(-1)
        labels = segmap_flat[pixel_ids]
        valid_mask = labels > 0
        if valid_mask.sum() == 0:
            frame_gaussian_ids.append(set())
            continue

        gaus_ids = gaus_ids[valid_mask]
        labels = labels[valid_mask]

        unique_labels = labels.unique()
        frame_gauss_set: set = set()
        for lbl in unique_labels:
            lbl_int = int(lbl.item())
            lbl_mask = labels == lbl
            lbl_gauss = torch.unique(gaus_ids[lbl_mask]).cpu().numpy()
            mask_gaussian_pclds[f"{frame_idx}_{lbl_int}"] = lbl_gauss
            global_frame_mask_list.append((frame_idx, lbl_int))
            frame_gauss_set.update(lbl_gauss.tolist())

        frame_gaussian_ids.append(frame_gauss_set)
        if len(frame_gauss_set) > 0:
            gaussian_in_frame_matrix[list(frame_gauss_set), frame_idx] = True

    return {
        "gaussian_in_frame_matrix": gaussian_in_frame_matrix,
        "mask_gaussian_pclds": mask_gaussian_pclds,
        "frame_gaussian_ids": frame_gaussian_ids,
        "global_frame_mask_list": global_frame_mask_list,
    }


def main():
    args = parse_args()

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else Path(args.data_dir) / "gauscluster_input.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians = load_gaussians_from_ckpt(args.ckpt, device=device)

    print(f"[Dataset] Loading COLMAP data from {args.data_dir}")
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
    )
    dataset = Dataset(parser, split="train")
    print(f"[Dataset] Train images: {len(dataset)}")

    mask_dir = discover_mask_dir(args.data_dir, args.mask_subdir)
    view_data = collect_view_data(parser, dataset, mask_dir)
    print(f"[Summary] Prepared {len(view_data)} camera views with SAM masks.")

    means, quats, scales, opacities, colors, sh_degree = gaussians
    print("\n================ 数据校验 ================")
    print(f"数据目录: {args.data_dir}")
    print(f"Mask 目录: {mask_dir}")
    print(f"相机数量: {len(view_data)}")
    print(f"高斯数量: {means.shape[0]}")
    print(f"SH 阶数: {sh_degree}")
    print("----------------------------------------")
    print("示例视角：")
    for sample in view_data[:3]:
        print(
            f"  • {sample['image_name']} | 分辨率 {sample['width']}x{sample['height']} | "
            f"mask: {sample['mask_path']}"
        )
    print("========================================\n")

    tracker = build_mask_gaussian_tracker(gaussians, view_data, device=device)
    total_masks = len(tracker["global_frame_mask_list"])
    total_points_hit = tracker["gaussian_in_frame_matrix"].sum()
    print("============== 跟踪统计（阶段一） ==============")
    print(f"有效 mask 数量: {total_masks}")
    print(f"有像素贡献的 Gaussian 总计: {total_points_hit}")
    for i, (frame_id, mask_id) in enumerate(tracker["global_frame_mask_list"][:5]):
        gs_ids = tracker["mask_gaussian_pclds"][f"{frame_id}_{mask_id}"]
        print(
            f"  • frame {frame_id:03d}, mask {mask_id}: 高斯数 {len(gs_ids)}"
        )
    print("=============================================")


if __name__ == "__main__":
    main()
