import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import viser
from pathlib import Path
from gsplat.distributed import cli
from gsplat.rendering import rasterization_2dgs

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from utils import get_implied_normal_from_depth


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    means, quats, scales, opacities = [], [], [], []
    colors_list = []
    uses_sh = None
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        has_sh = "sh0" in ckpt and "shN" in ckpt
        if uses_sh is None:
            uses_sh = has_sh
        else:
            assert (
                uses_sh == has_sh
            ), "All checkpoints must either use SH or direct colors consistently."

        if has_sh:
            sh0 = ckpt["sh0"]
            shN = ckpt["shN"]
            colors_list.append(torch.cat([sh0, shN], dim=-2))
        elif "colors" in ckpt:
            colors_list.append(torch.sigmoid(ckpt["colors"]))
        else:
            raise KeyError(
                f"Checkpoint {ckpt_path} is missing color information (expected 'sh0/shN' or 'colors')."
            )
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    colors = torch.cat(colors_list, dim=0).unsqueeze(0)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1) if uses_sh else None
    print("Number of Gaussians:", len(means))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode={
                "rgb": "RGB",
                "expected_depth": "RGB+ED",
                "median_depth": "RGB+ED",
                "render_normal": "RGB+ED",
                "surf_normal": "RGB+ED",
                "alpha": "RGB",
            }[render_tab_state.render_mode],
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
        )
        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "expected_depth":
            depth = render_colors[0, ..., -1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(
                    depth_norm.unsqueeze(-1), render_tab_state.colormap
                )
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "median_depth":
            depth = render_median[0, ..., 0]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(
                    depth_norm.unsqueeze(-1), render_tab_state.colormap
                )
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "render_normal":
            render_normals_vis = 1 - (render_normals[0] * 0.5 + 0.5)
            renders = render_normals_vis.cpu().numpy()
        elif render_tab_state.render_mode == "surf_normal":
            depth = render_colors[0, ..., -1:]
            normals_vis = get_implied_normal_from_depth(depth, K[None])
            normals_vis = 1 - (normals_vis[0] * 0.5 + 0.5)
            renders = normals_vis.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
