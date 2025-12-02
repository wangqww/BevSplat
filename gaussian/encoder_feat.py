from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Optional, Union

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


from backbone.backbone_dino import BackboneDino
from depth_predictor.depth_predictor_monocular import DepthPredictorMonocular
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from gaussian.build_gaussians import sample_image_grid
from gaussian.gaussian_adapter_feat import GaussianAdapter

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    features: Float[Tensor, "batch gaussian dim"]
    confidence: Float[Tensor, "batch gaussian 1"]

class GaussianFeatEncoder(nn.Module):
    def __init__(self, n_feature_channels) -> None:
        super(GaussianFeatEncoder, self).__init__()
        self.backbone = BackboneDino()
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, 128),
        )
        self.depth_predictor = DepthPredictorMonocular()
        self.gaussian_adapter = GaussianAdapter()
        
        self.to_opacity = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                128,
                9,
            ),
        )
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
        grd_feat: Float[Tensor, "batch view channels height width"],
        grd_conf: Float[Tensor, "batch view channels height width"],
        camera_k: Float[Tensor, "batch view 3 3"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        deterministic: bool = False,
    ) -> Gaussians:
        b, v, _, h, w = img.shape
        features = self.backbone(img)
        device = features.device
        h, w = features.shape[-2:]
        features = rearrange(features, "b v c h w -> b v h w c").contiguous()
        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w").contiguous()

        if self.high_resolution_skip is not None:
            # Add the high-resolution skip connection.
            skip = rearrange(img, "b v c h w -> (b v) c h w")
            skip = self.high_resolution_skip(skip)
            features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)

        # Sample depths from the resulting features.
        features = rearrange(features, "b v c h w -> b v (h w) c")
        depths, densities = self.depth_predictor.forward(
            features,
            near,
            far,
            deterministic,
            1 if deterministic else 3,
        )

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = self.to_gaussians(features).unsqueeze(-2)
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = 3
        gaussians = self.gaussian_adapter.forward(
            extrinsics,
            camera_k,
            xy_ray,
            depths,
            self.map_pdf_to_opacity(densities) / gpp,
            gaussians[..., 2:],
            grd_feat,
            grd_conf,
            (h, w),
        )

        # Dump visualizations if needed.
        # if visualization_dump is not None:
        #     visualization_dump["depth"] = rearrange(
        #         depths, "b (h w) s -> b h w s", h=h, w=w
        #     )
        #     visualization_dump["scales"] = rearrange(
        #         gaussians.scales, "b r spp xyz -> b (r spp) xyz"
        #     )
        #     visualization_dump["rotations"] = rearrange(
        #         gaussians.rotations, "b r spp xyzw -> b (r spp) xyzw"
        #     )

        # Optionally apply a per-pixel opacity.
        # opacity_multiplier = (
        #     rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
        #     if self.cfg.predict_opacity
        #     else 1
        # )

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r spp xyz -> b (v r spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r spp i j -> b (v r spp) i j",
            ),
            rearrange(
                gaussians.opacities,
                "b v r spp -> b (v r spp)",
            ),
            rearrange(
                gaussians.features,
                "b v r spp c -> b (v r spp) c",
            ),
            rearrange(
                gaussians.confidence,
                "b v r spp c -> b (v r spp) c",
            )
        )

    @property
    def last_layer_weights(self) -> Tensor:
        return self.to_gaussians[-1].weight
