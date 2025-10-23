""" Taken from SimpleRecon"""
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

OPENGL_TO_OPENCV = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
).astype(float)


def to_homogeneous(input_tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified
    dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output_bkN = torch.cat([input_tensor, ones], dim=dim)
    return output_bkN


class BackprojectDepth(nn.Module):
    """
    Layer that projects points from 2D camera to 3D space. The 3D points are
    represented in homogeneous coordinates.
    """

    def __init__(self, height: int, width: int) -> None:
        super().__init__()

        self.height = height
        self.width = width

        xx, yy = torch.meshgrid(
            torch.arange(self.width),
            torch.arange(self.height),
            indexing="xy",
        )
        pix_coords_2hw = torch.stack((xx, yy), dim=0) + 0.5

        pix_coords_13N = to_homogeneous(pix_coords_2hw, dim=0).flatten(1).unsqueeze(0)

        # make these tensors into buffers so they are put on the correct GPU
        # automatically
        self.register_buffer("pix_coords_13N", pix_coords_13N)

    def forward(self, depth_b1hw: torch.Tensor, invK_b44: torch.Tensor) -> torch.Tensor:
        """
        Backprojects spatial points in 2D image space to world space using
        invK_b44 at the depths defined in depth_b1hw.
        """

        cam_points_b3N = torch.matmul(invK_b44[:, :3, :3], self.pix_coords_13N)
        cam_points_b3N = depth_b1hw.flatten(start_dim=2) * cam_points_b3N
        cam_points_b4N = to_homogeneous(cam_points_b3N, dim=1)
        return cam_points_b4N


class Project3D(nn.Module):
    """
    Layer that projects 3D points into the 2D camera
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps).view(1, 1, 1))

    def forward(
        self,
        points_b4N: torch.Tensor,
        K_b44: torch.Tensor,
        cam_T_world_b44: torch.Tensor,
    ) -> torch.Tensor:
        """
        Projects spatial points in 3D world space to camera image space using
        the extrinsics matrix cam_T_world_b44 and intrinsics K_b44.
        """
        P_b44 = K_b44 @ cam_T_world_b44

        cam_points_b3N = P_b44[:, :3] @ points_b4N

        # from Kornia and OpenCV, https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#convert_points_from_homogeneous
        mask = torch.abs(cam_points_b3N[:, 2:]) > self.eps
        depth_b1N = cam_points_b3N[:, 2:] + self.eps
        scale = torch.where(
            mask, 1.0 / depth_b1N, torch.tensor(1.0, device=depth_b1N.device)
        )

        pix_coords_b2N = cam_points_b3N[:, :2] * scale

        return torch.cat([pix_coords_b2N, depth_b1N], dim=1)


class NormalGenerator(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
    ):
        """
        Estimates normals from depth maps.
        """
        super().__init__()
        self.height = height
        self.width = width

        self.backproject = BackprojectDepth(self.height, self.width)

    def forward(self, depth_b1hw: torch.Tensor, invK_b44: torch.Tensor) -> torch.Tensor:
        cam_points_b4N = self.backproject(depth_b1hw, invK_b44)
        cam_points_b3hw = cam_points_b4N[:, :3].view(-1, 3, self.height, self.width)

        gradients_b32hw = kornia.filters.spatial_gradient(cam_points_b3hw)

        # 反转方向以匹配OpenCV坐标定义
        normal_b3hw = -torch.cross(
            gradients_b32hw[:, :, 0],  # ∂P/∂x
            gradients_b32hw[:, :, 1],  # ∂P/∂y
            dim=1,
        )
        normal_b3hw = F.normalize(normal_b3hw, dim=1)
        return normal_b3hw
