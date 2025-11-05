import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
)
from utils_depth import get_implied_normal_from_depth
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import DefaultStrategy, ImprovedStrategy
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 1
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.00004
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1
    # GSs with large projected footprint will be pruned.
    prune_scale2d: float = 0.15
    # Stop refining based on 2D scale after this iteration (0 disables it)
    refine_scale2d_stop_iter: int = 4000
    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 20_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Maximum number of GSs allowed when using ImprovedStrategy
    budget: int = 3_000_000

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = True
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Distortion loss. (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000

    # Apply depth regularization once every N training iterations
    depth_reg_every_n: int = 8
    # Name of the directory containing depth priors relative to each scene
    depth_dir_name: Optional[str] = "pi3_depth" # "pi3_depth"
    # Weight assigned to the depth prior loss term
    depth_loss_weight: float = 0.2
    # Iteration after which depth regularization becomes active
    depth_loss_activation_step: int = 1_000

    # Apply normal regularization once every N training iterations
    normal_reg_every_n: int = 16
    # Name of the directory containing normal priors relative to each scene
    normal_dir_name: Optional[str] = "moge_normal" # "moge_normal"
    # Weight assigned to the rendered normal loss term
    render_normal_loss_weight: float = 0.1
    # Iteration after which rendered normal regularization becomes active
    render_normal_loss_activation_step: int = 7_000
    # Weight assigned to the surface normal loss term derived from depth
    surf_normal_loss_weight: float = 0.1
    # Iteration after which surface normal regularization becomes active
    surf_normal_loss_activation_step: int = 7_000
    # Weight assigned to enforcing consistency between rendered and depth normals
    consistency_normal_loss_weight: float = 0.0
    # Iteration after which normal consistency regularization becomes active
    consistency_normal_loss_activation_step: int = 7_000

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = True
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = True

    # Model for splatting.
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"
    # Densification strategy type.
    strategy_type: Literal["default", "improved"] = "default"
    # Whether to print verbose information from the densification strategy.
    strategy_verbose: bool = True

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)
        if self.refine_scale2d_stop_iter > 0:
            self.refine_scale2d_stop_iter = int(
                self.refine_scale2d_stop_iter * factor
            )
        self.depth_loss_activation_step = int(self.depth_loss_activation_step * factor)
        self.render_normal_loss_activation_step = int(
            self.render_normal_loss_activation_step * factor
        )
        self.surf_normal_loss_activation_step = int(
            self.surf_normal_loss_activation_step * factor
        )
        self.consistency_normal_loss_activation_step = int(
            self.consistency_normal_loss_activation_step * factor
        )


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 4e-5 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            depth_dir_name=cfg.depth_dir_name,
            normal_dir_name=cfg.normal_dir_name,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        if self.model_type == "2dgs":
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        # Densification Strategy
        if cfg.strategy_type == "default":
            self.strategy = DefaultStrategy(
                verbose=cfg.strategy_verbose,
                prune_opa=cfg.prune_opa,
                grow_grad2d=cfg.grow_grad2d,
                grow_scale3d=cfg.grow_scale3d,
                prune_scale3d=cfg.prune_scale3d,
                prune_scale2d=cfg.prune_scale2d,
                refine_scale2d_stop_iter=cfg.refine_scale2d_stop_iter,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=cfg.absgrad,
                revised_opacity=cfg.revised_opacity,
                key_for_gradient=key_for_gradient,
            )
        elif cfg.strategy_type == "improved":
            if not cfg.absgrad:
                print(
                    "[Warning] ImprovedStrategy typically expects absgrad=True for better densification."
                )
            self.strategy = ImprovedStrategy(
                verbose=cfg.strategy_verbose,
                prune_opa=cfg.prune_opa,
                grow_grad2d=cfg.grow_grad2d,
                prune_scale3d=cfg.prune_scale3d,
                prune_scale2d=cfg.prune_scale2d,
                refine_scale2d_stop_iter=cfg.refine_scale2d_stop_iter,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=cfg.absgrad,
                key_for_gradient=key_for_gradient,
                budget=cfg.budget,
            )
        else:
            raise ValueError(f"Unknown strategy_type: {cfg.strategy_type}")
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(
            scene_scale=self.scene_scale
        )

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        self.bil_grids = None
        self.bil_grid_optimizers: List[torch.optim.Optimizer] = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                )
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        render_mode = kwargs.get("render_mode", "RGB")

        if self.model_type == "2dgs":
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
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "2dgs-inria":
            renders, info = rasterization_2dgs_inria_wrapper(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
            render_colors, render_alphas = renders
            render_normals = info["normals_rend"]
            render_distort = info["render_distloss"]
            render_median = render_colors[..., -1:]

        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            info,
        )

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid and self.bil_grid_optimizers:
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            depth_prior = None
            if cfg.depth_loss_weight > 0.0:
                raw_depth_prior = data.get("depth_prior")
                if raw_depth_prior is not None and raw_depth_prior.numel() > 0:
                    depth_prior = raw_depth_prior.to(device)
                    if depth_prior.dim() == 3:
                        depth_prior = depth_prior.unsqueeze(0)

            normal_prior = None
            if cfg.surf_normal_loss_weight > 0.0 or cfg.render_normal_loss_weight > 0.0:
                raw_normal_prior = data.get("normal_prior")
                if raw_normal_prior is not None and raw_normal_prior.numel() > 0:
                    normal_prior = raw_normal_prior.to(device)
                    if normal_prior.dim() == 3:
                        normal_prior = normal_prior.unsqueeze(0)

            need_depth_prior = (
                cfg.depth_loss_weight > 0.0
                and depth_prior is not None
                and step >= cfg.depth_loss_activation_step
                and step % cfg.depth_reg_every_n == 0
            )
            need_normal_prior = (
                normal_prior is not None
                and step % cfg.normal_reg_every_n == 0
                and (
                    (
                        cfg.surf_normal_loss_weight > 0.0
                        and step >= cfg.surf_normal_loss_activation_step
                    )
                    or (
                        cfg.render_normal_loss_weight > 0.0
                        and step >= cfg.render_normal_loss_activation_step
                    )
                )
            )
            need_consistency_normal = (
                cfg.consistency_normal_loss_weight > 0.0
                and step >= cfg.consistency_normal_loss_activation_step
                and step % cfg.normal_reg_every_n == 0
            )

            need_depth = need_depth_prior or need_normal_prior or need_consistency_normal
            render_mode = "RGB+ED" if need_depth else "RGB"

            (
                renders,
                alphas,
                normals,
                render_distort,
                render_median,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode=render_mode,
                distloss=self.cfg.dist_loss,
            )
            if render_mode == "RGB+ED":
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid and self.bil_grids is not None:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=device) + 0.5) / height,
                    (torch.arange(width, device=device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.long().unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            tvloss = None
            if cfg.use_bilateral_grid and self.bil_grids is not None:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            depth_loss_value = torch.tensor(0.0, device=device)
            if need_depth_prior and depths is not None:
                depth_loss_value = self.compute_depth_loss(depths, depth_prior)
                loss += cfg.depth_loss_weight * depth_loss_value

            consistency_norm_loss = torch.tensor(0.0, device=device)
            surf_normal_loss = torch.tensor(0.0, device=device)
            render_normal_loss = torch.tensor(0.0, device=device)
            if depths is not None and (need_consistency_normal or need_normal_prior):
                surf_normals = get_implied_normal_from_depth(depths, Ks)

                if need_consistency_normal:
                    valid_mask = torch.ones_like(alphas)
                    consistency_norm_loss = self.compute_normal_loss(
                        F.normalize(normals, dim=-1), surf_normals, valid_mask
                    )
                    loss += cfg.consistency_normal_loss_weight * consistency_norm_loss

                if need_normal_prior:
                    mask_prior = torch.ones_like(depths)
                    if (
                        cfg.surf_normal_loss_weight > 0.0
                        and step >= cfg.surf_normal_loss_activation_step
                    ):
                        surf_normal_loss = self.compute_normal_loss(
                            surf_normals, normal_prior, mask_prior
                        )
                        loss += cfg.surf_normal_loss_weight * surf_normal_loss
                    if (
                        cfg.render_normal_loss_weight > 0.0
                        and step >= cfg.render_normal_loss_activation_step
                    ):
                        render_normal_loss = self.compute_normal_loss(
                            F.normalize(normals, dim=-1), normal_prior, mask_prior
                        )
                        loss += cfg.render_normal_loss_weight * render_normal_loss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if need_depth_prior:
                    self.writer.add_scalar(
                        "train/depthloss", depth_loss_value.item(), step
                    )
                if need_normal_prior and cfg.surf_normal_loss_weight > 0.0:
                    self.writer.add_scalar(
                        "train/surf_normalloss", surf_normal_loss.item(), step
                    )
                if need_normal_prior and cfg.render_normal_loss_weight > 0.0:
                    self.writer.add_scalar(
                        "train/render_normalloss", render_normal_loss.item(), step
                    )
                if need_consistency_normal:
                    self.writer.add_scalar(
                        "train/consistency_normalloss",
                        consistency_norm_loss.item(),
                        step,
                    )
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                if cfg.use_bilateral_grid and tvloss is not None:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            image_ids = data["image_id"].to(device).long()
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            (
                render_colors,
                alphas,
                normals,
                render_distort,
                render_median,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            normals_from_depth = get_implied_normal_from_depth(
                render_colors[..., 3:4], Ks
            )
            render_rgb = render_colors[..., :3]
            if cfg.use_bilateral_grid and self.bil_grids is not None:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=device) + 0.5) / height,
                    (torch.arange(width, device=device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                render_rgb = slice(
                    self.bil_grids,
                    grid_xy.expand(render_rgb.shape[0], -1, -1, -1),
                    render_rgb,
                    image_ids.unsqueeze(-1),
                )["rgb"]
            colors = torch.clamp(render_rgb, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            # write median depths
            denom = render_median.max() - render_median.min()
            render_median = (render_median - render_median.min()) / (denom + 1e-8)
            # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
            render_median = (
                apply_float_colormap(render_median).detach().cpu().squeeze(0).numpy()
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_median_depth_{step}.png",
                (render_median * 255).astype(np.uint8),
            )

            # write normals
            normals_img = (normals[0] * 0.5 + 0.5).cpu().numpy()
            normals_output = (normals_img * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normal_{step}.png", normals_output
            )

            # write normals from depth
            if normals_from_depth is not None:
                normals_depth = normals_from_depth * alphas.detach()
                normals_depth = (normals_depth[0] * 0.5 + 0.5).cpu().numpy()
                normals_depth = (normals_depth - normals_depth.min()) / (
                    normals_depth.max() - normals_depth.min() + 1e-8
                )
                normals_from_depth_output = (normals_depth * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                    normals_from_depth_output,
                )

            # write distortions

            render_dist = render_distort
            dist_max = torch.max(render_dist)
            dist_min = torch.min(render_dist)
            render_dist = (render_dist - dist_min) / (dist_max - dist_min)
            render_dist = (
                apply_float_colormap(render_dist).detach().cpu().squeeze(0).numpy()
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_distortions_{step}.png",
                (render_dist * 255).astype(np.uint8),
            )

            pixels_c = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_c = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors_c, pixels_c))
            metrics["ssim"].append(self.ssim(colors_c, pixels_c))
            metrics["lpips"].append(self.lpips(colors_c, pixels_c))
            if cfg.use_bilateral_grid and self.bil_grids is not None:
                cc_colors = color_correct(colors, pixels)
                cc_colors_c = cc_colors.permute(0, 3, 1, 2)
                metrics["cc_psnr"].append(self.psnr(cc_colors_c, pixels_c))
                metrics["cc_ssim"].append(self.ssim(cc_colors_c, pixels_c))
                metrics["cc_lpips"].append(self.lpips(cc_colors_c, pixels_c))

        ellipse_time /= len(valloader)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update(
            {
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
        )
        if cfg.use_bilateral_grid and self.bil_grids is not None:
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
        else:
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
        # save stats as json
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_mode_map = {
            "rgb": "RGB",
            "expected_depth": "RGB+ED",
            "median_depth": "RGB+ED",
            "render_normal": "RGB+ED",
            "surf_normal": "RGB+ED",
            "alpha": "RGB",
        }

        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            info,
        ) = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode=render_mode_map[render_tab_state.render_mode],
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "expected_depth":
            # normalize depth to [0, 1]
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
            renders = apply_float_colormap(
                depth_norm.unsqueeze(-1), render_tab_state.colormap
            ).cpu().numpy()
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
            renders = apply_float_colormap(
                depth_norm.unsqueeze(-1), render_tab_state.colormap
            ).cpu().numpy()
        elif render_tab_state.render_mode == "render_normal":
            render_normals = render_normals[0] * 0.5 + 0.5  # normalize to [0, 1]
            render_normals = 1 - render_normals # for better vis
            renders = render_normals.cpu().numpy()
        elif render_tab_state.render_mode == "surf_normal":
            depth = render_colors[..., -1:]
            surf_normals = get_implied_normal_from_depth(depth, K[None])
            surf_normals = surf_normals[0] * 0.5 + 0.5
            surf_normals = 1 - surf_normals  # for better vis
            renders = surf_normals.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
        else:
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders


    def compute_depth_loss(
        self, pred_depth: torch.Tensor, gt_depth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the mean absolute depth loss between predicted and ground-truth depth maps,
        supporting batched input.

        Args:
            pred_depth (torch.Tensor): Predicted depth maps of shape [B, H, W, 1] or [H, W, 1].
            gt_depth (torch.Tensor): Ground-truth depth maps of shape [B, H, W, 1] or [H, W, 1].

        Returns:
            torch.Tensor: A scalar tensor representing the averaged absolute depth loss across the batch.
        """

        if gt_depth is None:
            return torch.tensor(0.0, device=pred_depth.device)

        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(0)
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(0)

        valid_pix = gt_depth > 0.0
        if not valid_pix.any():
            return torch.tensor(0.0, device=pred_depth.device)

        diff = torch.where(valid_pix, pred_depth - gt_depth, 0.0)
        abs_diff = diff.abs()

        per_image_loss = abs_diff.sum(dim=(1, 2, 3)) / valid_pix.sum(dim=(1, 2, 3)).clamp(min=1)
        return per_image_loss.mean()


    def compute_normal_loss(
        self,
        pred_normals_bhw3: torch.Tensor,
        gt_normals_bhw3: torch.Tensor,
        masks_bhw1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the mean cosine similarity loss between predicted and ground-truth surface normals,
        supporting batched input.

        Args:
            pred_normals_bhw3 (torch.Tensor): Predicted normals of shape [B, H, W, 3] or [H, W, 3].
            gt_normals_bhw3 (torch.Tensor): Ground-truth normals of shape [B, H, W, 3] or [H, W, 3].
            masks_bhw1 (torch.Tensor): Binary masks of shape [B, H, W, 1] or [H, W, 1] indicating valid pixels.

        Returns:
            torch.Tensor: A scalar tensor representing the averaged cosine loss across the batch.
        """

        if gt_normals_bhw3 is None or pred_normals_bhw3 is None:
            return torch.tensor(0.0, device=self.device)

        if pred_normals_bhw3.dim() == 3:
            pred_normals_bhw3 = pred_normals_bhw3.unsqueeze(0)
        if gt_normals_bhw3.dim() == 3:
            gt_normals_bhw3 = gt_normals_bhw3.unsqueeze(0)
        if masks_bhw1.dim() == 3:
            masks_bhw1 = masks_bhw1.unsqueeze(0)

        pred_normals_bhw3 = F.normalize(pred_normals_bhw3, dim=-1)
        gt_normals_bhw3 = F.normalize(gt_normals_bhw3, dim=-1)

        dot = (gt_normals_bhw3 * pred_normals_bhw3).sum(dim=-1).clamp(-1.0, 1.0)
        mask = masks_bhw1.squeeze(-1)
        masked_dot = dot * mask
        valid_counts = mask.sum(dim=(1, 2)).clamp(min=1)
        per_batch_loss = 1 - (masked_dot.sum(dim=(1, 2)) / valid_counts)
        return per_batch_loss.mean()


def main(cfg: Config):
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        global BilateralGrid, color_correct, slice, total_variation_loss
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
