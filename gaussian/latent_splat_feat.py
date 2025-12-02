from dataclasses import dataclass
from math import isqrt
from typing import Literal, Optional, Tuple, Union

import torch

# from diff_gaussian_tw import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )

from feat_gaussian import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from .sh_utils import eval_sh

def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


@dataclass
class RenderOutput:
    color: Float[Tensor, "batch 3 height width"]
    feature: Float[Tensor, "batch channels height width"]
    confidence: Float[Tensor, "batch channels height width"]
    mask: Float[Tensor, "batch height width"]
    depth: Float[Tensor, "batch height width"]

def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, "batch"],
    far: Float[Tensor, "batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_color_sh_coefficients: Union[Float[Tensor, "batch gaussian 3 d_sh"], None],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_feature: Union[Float[Tensor, "batch gaussian channels"], None] = None,
    gaussian_confidence: Union[Float[Tensor, "batch gaussian"], None] = None,
    scale_invariant: bool = True,
    use_sh: bool = True
) -> RenderOutput:
    assert gaussian_color_sh_coefficients is not None or gaussian_feature is not None

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    color_sh_degree = 0
    shs = None
    features = None
    confidence = None
    colors_precomp = None
    if use_sh:
        if gaussian_color_sh_coefficients is not None:
            color_sh_degree = isqrt(gaussian_color_sh_coefficients.shape[-1]) - 1
            shs = rearrange(gaussian_color_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()
        if gaussian_feature is not None:
            # TODO implement general feature SH conversion in CUDA rasterizer
            # campos = extrinsics[:, :3, 3]
            # dir_pp = gaussian_means - campos.unsqueeze(1)
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=-1, keepdim=True)
            features = gaussian_feature
        if gaussian_confidence is not None:
            confidence = gaussian_confidence
    else:
        if gaussian_color_sh_coefficients is not None:
            colors_precomp = gaussian_color_sh_coefficients[..., 0]
        if gaussian_feature is not None:
            features = gaussian_feature
        if gaussian_confidence is not None:
            confidence = gaussian_confidence

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_feature_maps = []
    all_confidence_maps = []
    all_masks = []
    all_depth_maps = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=color_sh_degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, feature_map, confidence_map, mask, depth_map, _ = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if shs is not None else None,
            colors_precomp=colors_precomp[i] if colors_precomp is not None else None,
            features=features[i] if features is not None else None,
            confidence=confidence[i] if confidence is not None else None,
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_feature_maps.append(feature_map)
        all_confidence_maps.append(confidence_map)
        all_masks.append(mask.squeeze(0))
        all_depth_maps.append(depth_map.squeeze(0))
    all_images = torch.stack(all_images) if all_images[0] is not None else None
    all_feature_maps = torch.stack(all_feature_maps) if all_feature_maps[0] is not None else None
    all_confidence_maps = torch.stack(all_confidence_maps) if all_confidence_maps[0] is not None else None
    all_masks = torch.stack(all_masks)
    all_depth_maps = torch.stack(all_depth_maps)
    return RenderOutput(all_images, all_feature_maps, all_confidence_maps, all_masks, all_depth_maps)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_features: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_color_sh_coefficients: Union[Float[Tensor, "batch gaussian 3 d_sh"], None],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_feature: Union[Float[Tensor, "batch gaussian channels"], None] = None,
    gaussian_confidence: Union[Float[Tensor, "batch gaussian 1"], None] = None,
    gaussian_rgbs: Union[Float[Tensor, "batch gaussian 3"], None] = None,
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: Union[dict, None] = None,
) -> RenderOutput:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    
    color_sh_degree = 0
    shs = None
    features = None
    confidence = None
    colors_precomp = None
    if use_sh:
        if gaussian_color_sh_coefficients is not None:
            color_sh_degree = isqrt(gaussian_color_sh_coefficients.shape[-1]) - 1
            shs = rearrange(gaussian_color_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()
        if gaussian_feature is not None:
            # TODO implement general feature SH conversion in CUDA rasterizer
            # campos = extrinsics[:, :3, 3]
            # dir_pp = gaussian_means - campos.unsqueeze(1)
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=-1, keepdim=True)
            features = gaussian_feature
        if gaussian_confidence is not None:
            confidence = gaussian_confidence
        if gaussian_rgbs is not None:
            colors_precomp = gaussian_rgbs
    else:
        if gaussian_rgbs is not None:
            colors_precomp = gaussian_rgbs
        if gaussian_feature is not None:
            features = gaussian_feature
        if gaussian_confidence is not None:
            confidence = gaussian_confidence
    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_feature_maps = []
    all_confidence_maps = []
    all_masks = []
    all_depth_maps = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_features[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=color_sh_degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, feature_map, confidence_map, mask, depth_map, _ = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if shs is not None else None,
            colors_precomp=colors_precomp[i] if colors_precomp is not None else None,
            features=features[i] if features is not None else None,
            confidence=confidence[i] if confidence is not None else None,
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_feature_maps.append(feature_map)
        all_confidence_maps.append(confidence_map)
        all_masks.append(mask.squeeze(0))
        all_depth_maps.append(depth_map.squeeze(0))
    all_images = torch.stack(all_images) if all_images[0] is not None else None
    all_feature_maps = torch.stack(all_feature_maps) if all_feature_maps[0] is not None else None
    all_confidence_maps = torch.stack(all_confidence_maps) if all_confidence_maps[0] is not None else None
    all_masks = torch.stack(all_masks)
    all_depth_maps = torch.stack(all_depth_maps)
    return RenderOutput(all_images, all_feature_maps, all_confidence_maps, all_masks, all_depth_maps)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]
