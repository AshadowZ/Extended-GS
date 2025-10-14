from dataclasses import dataclass
from typing import Any, Dict, Union
import math

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import remove, reset_opa, long_axis_split


@dataclass
class ImprovedStrategy(Strategy):
    """An improved strategy with budget-based Gaussian splitting.

    This strategy is based on the paper:
    "Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering"
    https://arxiv.org/abs/2508.12313v1

    The strategy will:

    - Periodically split GSs along their long axis based on importance sampling from high image plane gradients.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.
    - Perform quantile-based pruning after the first two resets.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          candidates for splitting. Default is 0.0002.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".
        budget (int): Maximum number of Gaussians allowed. Default is 1000000.

    Examples:

        >>> from gsplat import ImprovedStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = ImprovedStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    absgrad: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    budget: int = 2000000

    def __post_init__(self):
        """Initialize instance variables after dataclass initialization."""
        self.reset_count = 0

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        state = {"grad2d": None, "count": None}
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            # grow GSs
            n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 10.0,
            )
            self.reset_count += 1

        # After the first two resets, perform quantile pruning 300 steps after reset
        if self.reset_count <= 2 and step % self.reset_every == 300 and step > 300:
            n_quantile_prune = self._quantile_prune_gs(
                params=params,
                optimizers=optimizers,
                state=state,
                percentile=0.2,
            )
            if self.verbose:
                print(
                    f"Step {step}: {n_quantile_prune} GSs pruned by quantile (20%). "
                    f"Now having {len(params['means'])} GSs."
                )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d

        startI = self.refine_start_iter
        endI = self.refine_stop_iter - 500
        den = endI - startI
        # 计算 rate，防止除以 0，并确保为 float
        if den == 0:
            rate = 1.0
        else:
            rate = float((step - startI) / den)
        # clamp 到 [0, 1]，避免负值或 >1 的奇异情况
        rate = max(0.0, min(1.0, rate))

        if rate >= 1.0:
            budget = int(self.budget)
        else:
            # 使用 math.sqrt 对 float 做开方
            budget = int(math.sqrt(rate) * float(self.budget))

        total_qualified = int(torch.sum(is_grad_high).item())
        curr_points = params["means"].shape[0]
        theoretical_max = total_qualified + curr_points
        final_budget = min(budget, theoretical_max)
        new_points_needed = final_budget - curr_points

        # 初始化is_split掩码，全为False
        is_split = torch.zeros_like(is_grad_high, dtype=torch.bool, device=device)
        # 创建重要性分数向量，只考虑高梯度区域
        importance_scores = grads.clone()
        importance_scores[~is_grad_high] = 0.0  # 屏蔽非高梯度区域
        # 确保所有分数非负且至少有一个有效候选
        if torch.any(importance_scores > 0):
            num_available = (importance_scores > 0).sum().item()
            actual_split_count = min(max(new_points_needed, 0), num_available)
            if actual_split_count > 0:
                selected_indices = torch.multinomial(
                    importance_scores, actual_split_count, replacement=False
                )
                is_split[selected_indices] = True

        n_split = is_split.sum().item()

        # then split
        if n_split > 0:
            long_axis_split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
            )
        return n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

    @torch.no_grad()
    def _quantile_prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        percentile: float,
    ) -> int:
        """Prune Gaussians with opacities below the given percentile.

        This method implements the quantile-based pruning strategy from:
        "Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering"
        https://arxiv.org/abs/2508.12313v1

        Args:
            params: The parameters dictionary containing "opacities".
            optimizers: The optimizers for the parameters.
            state: The running state dictionary.
            percentile: The percentile threshold (e.g., 0.2 for 20%).
                Gaussians with opacities below this percentile will be pruned.

        Returns:
            Number of Gaussians pruned.
        """
        opacities = torch.sigmoid(params["opacities"].flatten())
        threshold = torch.quantile(opacities, percentile)
        is_prune = opacities < threshold

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
