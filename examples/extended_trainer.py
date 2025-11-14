import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy, ImprovedStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
from utils_depth import get_implied_normal_from_depth


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
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
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = True
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = True

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
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Densification strategy selection (default / mcmc / improved)
    strategy_type: Literal["default", "mcmc", "improved"] = "default"
    # Verbosity for densification logs
    strategy_verbose: bool = True
    # Densification hyper-parameters (see notes below; shared unless marked otherwise)
    prune_opa: float = 0.05
    grow_grad2d: float = 0.0002
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 4000
    refine_start_iter: int = 500
    refine_stop_iter: int = 20000
    reset_every: int = 3000
    refine_every: int = 100
    absgrad: bool = True
    # DefaultStrategy-only hyper-parameters
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    pause_refine_after_reset: int = 0
    revised_opacity: bool = False
    # ImprovedStrategy-only hyper-parameter
    budget: int = 2_000_000
    # MCMCStrategy-only hyper-parameters
    mcmc_cap_max: int = 1_000_000
    mcmc_noise_lr: float = 5e5
    mcmc_min_opacity: float = 0.005

    # Strategy instance (constructed from the type/params above)
    strategy: Union[DefaultStrategy, MCMCStrategy, ImprovedStrategy] = field(
        init=False, repr=False
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

    # LR for 3D point positions
    means_lr: float = 4e-5
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0

    ### Scale regularization
    """Weight of the regularisation loss encouraging gaussians to be flat, i.e. set their minimum
    scale to be small"""
    flat_reg: float = 1.0
    """If scale regularization is enabled, a scale regularization introduced in PhysGauss
    (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians.
    This implementation adapts the PhysGauss loss to use the ratio of max to median scale
    instead of max to min, as implemented in mvsanywhere/regsplatfacto. This modification
    has been found to work better at encouraging Gaussians to be disks.
    """
    scale_reg: float = 1.0
    """Threshold of ratio of Gaussian's max to median scale before applying regularization
    loss. This is adapted from the PhysGauss paper (there they used ratio of max to min).
    """
    max_gauss_ratio: float = 6.0

    ### depth and normal regularization
    """Specifies applying depth regularization once every N iterations"""
    depth_reg_every_n: int = 4
    """If not None, the code will look for a folder named 'depth_dir_name' at the same level as
    the 'images' directory, load the dense depth maps from it, and use their depth values 
    for regularization.
    """
    depth_dir_name: Optional[str] = "moge_depth"  # "pi3_depth"
    """Weight of the depth loss"""
    depth_loss_weight: float = 0.25
    """Starting iteration for depth regularization"""
    depth_loss_activation_step: int = 1000

    """Specifies applying normal regularization once every N iterations"""
    normal_reg_every_n: int = 8
    """If not None, the code will look for a folder named 'normal_dir_name' at the same level as
    the 'images' directory, load the dense normal maps from it, and use their normal values 
    for regularization.
    """
    normal_dir_name: Optional[str] = "moge_normal"  # "moge_normal"
    """Weight of the render_normal_loss"""
    render_normal_loss_weight: float = 0.1
    """Starting iteration for render_normal regularization"""
    render_normal_loss_activation_step: int = 7000
    """Weight of the surf_normal_loss"""
    surf_normal_loss_weight: float = 0.1
    """Starting iteration for surf_normal regularization"""
    surf_normal_loss_activation_step: int = 7000
    """Weight of the consistency_normal_loss"""
    consistency_normal_loss_weight: float = 0.0
    """Starting iteration for normal consistency regularization"""
    consistency_normal_loss_activation_step: int = 7000

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

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = True
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = True

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def __post_init__(self):
        self.rebuild_strategy()

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, ImprovedStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

    def rebuild_strategy(self):
        if self.strategy_type == "default":
            self.strategy = DefaultStrategy(
                prune_opa=self.prune_opa,
                grow_grad2d=self.grow_grad2d,
                grow_scale3d=self.grow_scale3d,
                grow_scale2d=self.grow_scale2d,
                prune_scale3d=self.prune_scale3d,
                prune_scale2d=self.prune_scale2d,
                refine_scale2d_stop_iter=self.refine_scale2d_stop_iter,
                refine_start_iter=self.refine_start_iter,
                refine_stop_iter=self.refine_stop_iter,
                reset_every=self.reset_every,
                refine_every=self.refine_every,
                pause_refine_after_reset=self.pause_refine_after_reset,
                absgrad=self.absgrad,
                revised_opacity=self.revised_opacity,
                verbose=self.strategy_verbose,
            )
        elif self.strategy_type == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.mcmc_cap_max,
                noise_lr=self.mcmc_noise_lr,
                refine_start_iter=self.refine_start_iter,
                refine_stop_iter=self.refine_stop_iter,
                refine_every=self.refine_every,
                min_opacity=self.mcmc_min_opacity,
                verbose=self.strategy_verbose,
            )
        elif self.strategy_type == "improved":
            self.strategy = ImprovedStrategy(
                prune_opa=self.prune_opa,
                grow_grad2d=self.grow_grad2d,
                prune_scale3d=self.prune_scale3d,
                prune_scale2d=self.prune_scale2d,
                refine_scale2d_stop_iter=self.refine_scale2d_stop_iter,
                refine_start_iter=self.refine_start_iter,
                refine_stop_iter=self.refine_stop_iter,
                reset_every=self.reset_every,
                refine_every=self.refine_every,
                absgrad=self.absgrad,
                verbose=self.strategy_verbose,
                budget=self.budget,
            )
        else:
            assert_never(self.strategy_type)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

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
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        elif isinstance(self.cfg.strategy, ImprovedStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

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
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
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
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
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
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

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
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
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

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
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
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, (DefaultStrategy, ImprovedStrategy))
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

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
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
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
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]

            # Optional priors
            depth_prior = None
            raw_depth_prior = data.get("depth_prior")
            if (
                raw_depth_prior is not None
                and isinstance(raw_depth_prior, torch.Tensor)
                and raw_depth_prior.numel() > 0
            ):
                depth_prior = raw_depth_prior.to(device)
                if depth_prior.dim() == 3:
                    depth_prior = depth_prior.unsqueeze(0)

            normal_prior = None
            normal_prior_mask = None
            raw_normal_prior = data.get("normal_prior")
            if (
                raw_normal_prior is not None
                and isinstance(raw_normal_prior, torch.Tensor)
                and raw_normal_prior.numel() > 0
            ):
                normal_prior = raw_normal_prior.to(device)
                if normal_prior.dim() == 3:
                    normal_prior = normal_prior.unsqueeze(0)
                ones_like = torch.ones_like(normal_prior)
                is_invalid = torch.isclose(normal_prior, ones_like).all(
                    dim=-1, keepdim=True
                )
                normal_prior_mask = (~is_invalid).float()

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Determine required render mode based on active regularizations
            need_depth_prior = (
                depth_prior is not None
                and cfg.depth_loss_weight > 0.0
                and step >= cfg.depth_loss_activation_step
                and step % cfg.depth_reg_every_n == 0
            )
            need_normal_prior = (
                normal_prior is not None
                and step % cfg.normal_reg_every_n == 0
                and (
                    (
                        cfg.surf_normal_loss_weight > 0
                        and step >= cfg.surf_normal_loss_activation_step
                    )
                    or (
                        cfg.render_normal_loss_weight > 0
                        and step >= cfg.render_normal_loss_activation_step
                    )
                )
            )
            need_consistency_normal = (
                cfg.consistency_normal_loss_weight > 0.0
                and step >= cfg.consistency_normal_loss_activation_step
                and step % cfg.normal_reg_every_n == 0
            )

            need_depth_for_render = (
                need_depth_prior or need_normal_prior or need_consistency_normal
            )
            need_normals_for_render = need_normal_prior or need_consistency_normal

            # Select render mode
            if need_normals_for_render:
                render_mode = "RGB+ED+N"
            elif need_depth_for_render:
                render_mode = "RGB+ED"
            else:
                render_mode = "RGB"

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode=render_mode,
                masks=masks,
            )

            # Parse render outputs based on mode
            if render_mode == "RGB+ED+N":
                colors, depths, render_normals = (
                    renders[..., 0:3],
                    renders[..., 3:4],
                    renders[..., 4:7],
                )
            elif render_mode == "RGB+ED":
                colors, depths, render_normals = (
                    renders[..., 0:3],
                    renders[..., 3:4],
                    None,
                )
            else:
                colors, depths, render_normals = renders, None, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # depth loss
            if need_depth_prior:
                depth_loss = self.compute_depth_loss(depths, depth_prior)
                loss += cfg.depth_loss_weight * depth_loss

            consistency_norm_loss = None
            surf_normal_loss = None
            render_normal_loss = None

            surf_normals_from_depth = None
            if (need_consistency_normal or need_normal_prior) and depths is not None:
                surf_normals_from_depth = get_implied_normal_from_depth(depths, Ks)

            # consistency normal loss
            if (
                need_consistency_normal
                and surf_normals_from_depth is not None
                and render_normals is not None
            ):
                mask_consistency = torch.ones_like(alphas)
                consistency_norm_loss = self.compute_normal_loss(
                    F.normalize(render_normals, dim=-1),
                    surf_normals_from_depth,
                    mask_consistency,
                )
                loss += cfg.consistency_normal_loss_weight * consistency_norm_loss

            # normal loss
            if need_normal_prior and surf_normals_from_depth is not None:
                mask = torch.ones_like(depths).float()  # [B,H,W,1]
                if normal_prior_mask is not None:
                    mask = mask * normal_prior_mask

                # Surface normal loss (from depth)
                if (
                    cfg.surf_normal_loss_weight > 0
                    and step >= cfg.surf_normal_loss_activation_step
                ):
                    surf_normal_loss = self.compute_normal_loss(
                        surf_normals_from_depth, normal_prior, mask
                    )
                    loss += cfg.surf_normal_loss_weight * surf_normal_loss

                # Rendered normal loss
                if (
                    cfg.render_normal_loss_weight > 0
                    and step >= cfg.render_normal_loss_activation_step
                    and render_normals is not None
                ):
                    render_normal_loss = self.compute_normal_loss(
                        render_normals, normal_prior, mask
                    )
                    loss += cfg.render_normal_loss_weight * render_normal_loss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()

            # the smallest scale is always near 0
            if cfg.flat_reg > 0.0:
                loss += cfg.flat_reg * self.compute_flat_loss()
            # We follow the original SplatFacto implementation here and only apply this loss every 10 steps
            if cfg.scale_reg > 0.0 and step % 10 == 0:
                loss += cfg.scale_reg * self.compute_scale_regularisation_loss_median()

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if need_depth_prior:
                    self.writer.add_scalar("train/depthloss", depth_loss.item(), step)
                if (
                    need_normal_prior
                    and cfg.surf_normal_loss_weight > 0
                    and surf_normal_loss is not None
                ):
                    self.writer.add_scalar(
                        "train/surf_normalloss", surf_normal_loss.item(), step
                    )
                if (
                    need_normal_prior
                    and cfg.render_normal_loss_weight > 0
                    and render_normal_loss is not None
                ):
                    self.writer.add_scalar(
                        "train/render_normalloss", render_normal_loss.item(), step
                    )
                if need_consistency_normal and consistency_norm_loss is not None:
                    self.writer.add_scalar(
                        "train/consistency_normalloss",
                        consistency_norm_loss.item(),
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
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

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # optimize with gradient accumulation
            # Implementation of gradient accumulation trick from:
            # "Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering"
            # https://arxiv.org/abs/2508.12313v1
            # if step <= 15000:
            #     # Every step update for first 15000 iterations
            #     for optimizer in self.optimizers.values():
            #         if cfg.visible_adam:
            #             optimizer.step(visibility_mask)
            #         else:
            #             optimizer.step()
            #         optimizer.zero_grad(set_to_none=True)
            # elif step <= 22500:
            #     # Accumulate 5 steps, update every 5 steps for 15000-22500 iterations
            #     if step % 5 == 0:
            #         for optimizer in self.optimizers.values():
            #             if cfg.visible_adam:
            #                 optimizer.step(visibility_mask)
            #             else:
            #                 optimizer.step()
            #             optimizer.zero_grad(set_to_none=True)
            # else:
            #     # Accumulate 20 steps, update every 20 steps after 22500 iterations
            #     if step % 20 == 0:
            #         for optimizer in self.optimizers.values():
            #             if cfg.visible_adam:
            #                 optimizer.step(visibility_mask)
            #             else:
            #                 optimizer.step()
            #             optimizer.zero_grad(set_to_none=True)
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

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, (DefaultStrategy, ImprovedStrategy)):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

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
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_bilateral_grid:
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
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

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

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
            "render_normal": "RGB+ED+N",
            "surf_normal": "ED",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            # print("[Debug] depth range:", float(depth.min()), float(depth.max()))

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
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        elif render_tab_state.render_mode == "render_normal":
            normals_t = render_colors[0, ..., 4:7]  # -> Tensor, shape [H, W, 3]
            normals_t = (normals_t + 1) * 0.5
            normals_t = 1.0 - normals_t  # Better visualization
            # Ensure range and convert to numpy uint8
            normals_t = (
                normals_t.clamp(0.0, 1.0).detach().cpu().numpy()
            )  # float in [0,1]
            renders = (normals_t * 255.0).astype(np.uint8)  # numpy array
        elif render_tab_state.render_mode == "surf_normal":
            depth = render_colors[..., 0:1]
            normals_t = get_implied_normal_from_depth(depth, K)
            normals_t = normals_t.squeeze()
            normals_t = (normals_t + 1) * 0.5
            normals_t = 1.0 - normals_t  # Better visualization
            # Ensure range and convert to numpy uint8
            normals_t = (
                normals_t.clamp(0.0, 1.0).detach().cpu().numpy()
            )  # float in [0,1]
            renders = (normals_t * 255.0).astype(np.uint8)  # numpy array
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
        valid_pix = gt_depth > 0.0
        if not valid_pix.any():
            return torch.tensor(0.0, device=pred_depth.device)

        diff = torch.where(valid_pix, pred_depth - gt_depth, 0)
        abs_diff = torch.abs(diff)

        # Automatically infer batch dimension
        if pred_depth.ndim == 4:  # [B,H,W,1]
            per_image_loss = abs_diff.sum(dim=(1, 2, 3)) / valid_pix.sum(
                dim=(1, 2, 3)
            ).clamp(min=1)
            depth_loss = per_image_loss.mean()
        else:  # Single image [H,W] or [H,W,1]
            depth_loss = abs_diff.sum() / valid_pix.sum().clamp(min=1)

        return depth_loss

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
            pred_normals_bhw3 (torch.Tensor): Predicted normal maps of shape [B, H, W, 3],
                normalized to unit vectors.
            gt_normals_bhw3 (torch.Tensor): Ground-truth normal maps of shape [B, H, W, 3],
                normalized to unit vectors.
            masks_bhw1 (torch.Tensor): Binary masks of shape [B, H, W, 1] indicating valid pixels.

        Returns:
            torch.Tensor: A scalar tensor representing the averaged cosine loss across the batch.
        """
        # Ensure batch dimension exists (handles single-image input)
        if pred_normals_bhw3.dim() == 3:
            pred_normals_bhw3 = pred_normals_bhw3.unsqueeze(0)
            gt_normals_bhw3 = gt_normals_bhw3.unsqueeze(0)
            masks_bhw1 = masks_bhw1.unsqueeze(0)

        # Compute dot product between predicted and ground-truth normals
        # Shape: [B, H, W]
        dot = (gt_normals_bhw3 * pred_normals_bhw3).sum(dim=-1).clamp(-1.0, 1.0)

        # Remove channel dimension from mask → [B, H, W]
        mask = masks_bhw1.squeeze(-1)

        # Avoid sparse access pattern from masked_select by zeroing invalid pixels
        masked_dot = dot * mask  # Zero out invalid pixels (faster than masked_select)

        # Count valid pixels per image; clamp to avoid division by zero
        valid_counts = mask.sum(dim=(1, 2)).clamp(min=1)

        # Compute per-image cosine loss: 1 - mean(cos(theta))
        per_batch_loss = 1 - (masked_dot.sum(dim=(1, 2)) / valid_counts)

        # Return mean loss across the batch
        return per_batch_loss.mean()

    def compute_flat_loss(self) -> torch.Tensor:
        """
        Computes the flatness loss. This encourages the smallest scale of each Gaussian to be small.

        This should have the effect of encouraging Gaussians to be disks (or spikes) rather than
        balls. There is a separate `scale_regularisation_loss` which encourages the Gaussians to
        be disks rather than spikes.

        Returns:
            torch.Tensor: The flatness loss as a scalar.
        """
        flat_loss = torch.exp(self.splats["scales"]).amin(dim=-1).mean()
        return flat_loss

    def compute_scale_regularisation_loss_median(self) -> torch.Tensor:
        """
        Computes the scale regularisation loss as the ratio between the maximum and median
        scale of the Gaussians. This is only applied to Gaussians with ratios above
        self.config.max_gauss_ratio.

        This is adapted from the PhysGauss paper (https://xpandora.github.io/PhysGaussian/).
        In that paper, they used the ratio of max to min scale. The max-to-median ratio
        modification comes from mvsanywhere/regsplatfacto and has been found to work better
        at encouraging disk-shaped Gaussians.

        Returns:
            torch.Tensor: The scale regularisation loss as a scalar.
        """
        # For each Gaussian, compute the ratio between the maximum and median (middle) dimension
        scale_exp = torch.exp(self.splats["scales"])
        ratio = scale_exp.amax(dim=-1) / scale_exp.median(dim=-1).values

        # Gaussians with ratios below max_gauss_ratio have no loss applied to them.
        # Gaussians with ratios above max_gauss_ratio have their ratio minimised.
        # The following diagram shows how the scale_reg loss varies as the ratio varies:
        #
        #           ▲
        #           │
        #           │       max_gauss_ratio       x
        #           │               │           x
        # scale_reg │               │         x
        #           │               │       x
        #           │               │     x
        #           │               │   x
        #           │               ▼ x
        #       0.0 └xxxxxxxxxxxxxxxx──────────────────►
        #           1.0
        #                          ratio
        #
        max_gauss_ratio = torch.tensor(self.cfg.max_gauss_ratio)
        scale_reg = torch.maximum(ratio, max_gauss_ratio) - max_gauss_ratio
        return scale_reg.mean()  # this has a weighting applied in get_loss_dict


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python -m examples.extended_trainer \
        --strategy_type improved \
        --data_dir data/360_v2/garden \
        --result_dir results/garden
    ```

    """

    cfg = tyro.cli(Config)
    cfg.rebuild_strategy()
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
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

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
