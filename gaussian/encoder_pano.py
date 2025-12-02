from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Optional, Union

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

import open3d as o3d
import plotly.graph_objs as go
import torch.nn.functional as F

from backbone.backbone_pano import BackboneDino
from gaussian.build_gaussians import sample_image_grid

from .build_gaussians import build_covariance

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    features: Float[Tensor, "batch gaussian dim"]
    confidence: Float[Tensor, "batch gaussian 1"]
    rgbs: Float[Tensor, "batch gaussian 3"]


def equirectangular_to_xyz(width, height, device):
    """Convert equirectangular coordinates to spherical 3D coordinates in OpenCV convention"""
    # 创建 theta 和 phi 为 1D 张量
    theta = torch.linspace(0, 2 * torch.pi, width, device=device)  # 方位角 [0, 2π]
    phi = torch.linspace(0, torch.pi, height, device=device)       # 仰角 [0, π]
    
    # 生成网格，调整 indexing='ij' 确保符合 PyTorch 约定
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')
    theta = theta # (H, W)
    phi = phi     # (H, W)
    # 计算 OpenCV 形式的 X, Y, Z 坐标
    x = -torch.sin(phi) * torch.sin(theta)   # OpenCV X: 右
    y = -torch.cos(phi)                     # OpenCV Y: 下
    z = -torch.sin(phi) * torch.cos(theta)  # OpenCV Z: 前

    # 将 x, y, z 堆叠在一起，并调整维度 (height, width, 3)
    xyz = torch.stack((x, y, z), dim=-1)  # (H, W, 3)

    return xyz


class GaussianEncoder(nn.Module):
    def __init__(self, gs_dim=11, area='same') -> None:
        super(GaussianEncoder, self).__init__()
        self.backbone = BackboneDino()
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out + 1 + 3, 128),
        )
        # self.backbone_projection = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(self.backbone.d_out, 128),
        # )
        self.gpv = 3
        self.to_opacity = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                128,
                gs_dim*self.gpv,
            ),
        )

        self.pos_act = nn.Tanh()
        self.scale_act = nn.Sigmoid()
        self.opacity_act = nn.Sigmoid()
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        # self.to_gaussians_feat = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(
        #         128,
        #         148,
        #     ),
        # )
        # High resolution skip only required in case of now downscaling
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 3),
            nn.ReLU(),
        )
        if area == 'same':
            self.offset_max = [3.0] * 3
            self.scale_max = [0.3] * 3
        else:
            self.offset_max = [3.0] * 3
            self.scale_max = [0.3] * 3

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
    ) -> Float[Tensor, " *batch"]:
        exponent = 1.0
        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        img: Float[Tensor, "batch view channels height width"],
        grd_feat: Union[Float[Tensor, "batch channels height width"] , None],
        grd_conf: Union[Float[Tensor, "batch channels height width"] , None],
        depth_map: Float[Tensor, "batch 1 height width"],
    ) -> Gaussians:
        b, _, h, w = img.shape
        features = self.backbone(img)
        device = img.device
        # h, w = features.shape[-2:]
        features = torch.cat((img, depth_map / 20.0, features), dim=1)
        features = rearrange(features, "b c h w -> b h w c").contiguous()
        features = self.backbone_projection(features)
        features = rearrange(features, "b h w c -> b c h w").contiguous()

        if self.high_resolution_skip is not None:
            # Add the high-resolution skip connection.
            skip = self.high_resolution_skip(img)
            features = features + skip

        # Sample depths from the resulting features.
        features = rearrange(features, "b c h w -> b (h w) c")
        depth_map = rearrange(depth_map, "b c h w -> b (h w) c")
        # fake_depth = depths.mean(dim=-1).reshape(b, 1, 80, 160)  # 假设深度张量，形状为 [1, 80, 160]
        # Convert the features and depths into Gaussians.
        # xy_ray, _ = sample_image_grid((h, w), device)
        # xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = self.to_gaussians(features)
        gaussians = gaussians.view(b, h*w, self.gpv, -1)
        xyz_coords = equirectangular_to_xyz(w, h, device)
        coords = rearrange(xyz_coords, "h w xyz -> (h w) xyz").contiguous()
        coords = coords.unsqueeze(0).repeat(b, 1, 1)

        gs_offsets_x = self.pos_act(gaussians[..., :1]) * self.offset_max[0]
        gs_offsets_y = self.pos_act(gaussians[..., 1:2]) * self.offset_max[1]
        gs_offsets_z = self.pos_act(gaussians[..., 2:3]) * self.offset_max[1]

        opacities = self.opacity_act(gaussians[..., 3:4])

        rotations = self.rot_act(gaussians[..., 4:8])
        scale_x = self.scale_act(gaussians[..., 8:9]) * self.scale_max[0]
        scale_y = self.scale_act(gaussians[..., 9:10]) * self.scale_max[1]
        scale_z = self.scale_act(gaussians[..., 10:11]) * self.scale_max[2]

        scales = torch.cat([scale_x, scale_y, scale_z], dim=-1)
        offset_xyz = torch.cat([gs_offsets_x, gs_offsets_y, gs_offsets_z], dim=-1)
        
        means = coords * depth_map
        means = means[:,:,None,:] + offset_xyz
        covariances = build_covariance(scales, rotations)

        gs_features = rearrange(grd_feat, "batch channels height width -> batch (height width) channels").unsqueeze(-2)
        gs_confidences = rearrange(grd_conf, "batch channels height width -> batch (height width) channels").unsqueeze(-2)
        gs_rgbs = rearrange(img, "batch channels height width -> batch (height width) channels").unsqueeze(-2)
        
        gs_features = gs_features.broadcast_to((*opacities.shape[:-1], 32))
        gs_confidences = gs_confidences.broadcast_to((*opacities.shape[:-1], 1))
        gs_rgbs = gs_rgbs.broadcast_to((*opacities.shape[:-1], 3))
        # 假设你有以下张量
        # image_tensor = img[:,0]  # 图像张量，形状为 [1, 3, 80, 160]
        # fake_depth = depths[:, :, :, 0].reshape(1, 1, 80, 160)  # 假设深度张量，形状为 [1, 80, 160]
        # coords_tensor = xyz_coords.unsqueeze(0) * real_depth.squeeze(1).unsqueeze(-1) # 三维坐标点张量，形状为 [1, 80, 160, 3]
        # # coords_tensor = xyz_coords.unsqueeze(0) * fake_depth.squeeze(1).unsqueeze(-1) # 三维坐标点张量，形状为 [1, 80, 160, 3]
        # # 提取 3D 坐标点 (x, y, z)
        # points = coords_tensor[0].reshape(-1, 3).cpu().detach().numpy()  # 将坐标点张量展平为 (80*160, 3)

        # # 提取颜色 (这里假设使用图像的 RGB 值作为颜色)
        # colors = image_tensor[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()  # 将图像张量展平为 (80*160, 3)

        # colors_rgb = ['rgb({},{},{})'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]  # 转换为字符串形式的 RGB

        # fig = go.Figure(data=[go.Scatter3d(
        #     x=points[:, 0],
        #     y=points[:, 1],
        #     z=points[:, 2],
        #     mode='markers',
        #     marker=dict(
        #         size=3,
        #         color=colors_rgb,  # 设置颜色
        #     )
        # )])
        # fig.update_layout(scene=dict(
        #     xaxis=dict(title='X', tick0=0, dtick=1),  # X 轴方向正确
        #     yaxis=dict(title='Y (Down)', tick0=0, dtick=-1),  # 翻转Y轴
        #     zaxis=dict(title='Z', tick0=0, dtick=1),  # Z 轴方向正确
        #     aspectmode='cube'  # 确保XYZ比例一致
        # ))
        # # fig.show()

        # # 保存为 HTML 文件，下载后用浏览器打开
        # fig.write_html("point_cloud1.html")

        return Gaussians(
            rearrange(
                means,
                "b r spp xyz -> b (r spp) xyz",
            ),
            rearrange(
                covariances,
                "b r spp i j -> b (r spp) i j",
            ),
            rearrange(
                opacities,
                "b r spp c -> b (r spp) c",
            ),
            rearrange(
                gs_features,
                "b r spp c -> b (r spp) c",
            ),
            rearrange(
                gs_confidences,
                "b r spp c -> b (r spp) c",
            ),
            rearrange(
                gs_rgbs,
                "b r spp c -> b (r spp) c",
            )
        )

    @property
    def last_layer_weights(self) -> Tensor:
        return self.to_gaussians[-1].weight
