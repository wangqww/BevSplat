from jaxtyping import Float, Shaped
from dataclasses import dataclass
from torch import Tensor
from pathlib import Path
from typing import Any, Generator, Iterable, Literal, Optional, Union
import torch
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange
from string import ascii_letters, digits, punctuation
import numpy as np
from torchvision import transforms
from gaussian.decoder import DecoderOutput
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from gaussian.latent_splat_feat import render_cuda_orthographic
# from gaussian.nopo_cuda_splatting import render_cuda_orthographic
to_pil_image = transforms.ToPILImage()


Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Float[Tensor, "#channel"],
    Float[Tensor, ""],
]
EXPECTED_CHARACTERS = digits + punctuation + ascii_letters


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    color_harmonics: Union[Float[Tensor, "batch gaussian 3 d_sh"], None]
    features: Float[Tensor, "batch gaussian dim"]
    confidence: Float[Tensor, "batch gaussian 1"]
    rgbs: Union[Float[Tensor, "batch gaussian 3"], None]

def _sanitize_color(color: Color) -> Float[Tensor, "#channel"]:
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)

def overlay(
    base: Float[Tensor, "channel base_height base_width"],
    overlay: Float[Tensor, "channel overlay_height overlay_width"],
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Float[Tensor, "channel base_height base_width"]:
    # The overlay must be smaller than the base.
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result

def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]

def compute_equal_aabb_with_margin(
    minima: Float[Tensor, "*#batch 3"],
    maxima: Float[Tensor, "*#batch 3"],
    margin: float = 0.1,
) -> tuple[
    Float[Tensor, "*batch 3"],  # minima of the scene
    Float[Tensor, "*batch 3"],  # maxima of the scene
]:
    midpoint = (maxima + minima) * 0.5
    span = (maxima - minima).max() * (1 + margin)
    scene_minima = midpoint - 0.5 * span
    scene_maxima = midpoint + 0.5 * span
    return scene_minima, scene_maxima

def cat(
    main_axis: Axis,
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Float[Tensor, "channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((c, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def vcat(
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def draw_label(
    text: str,
    font: Path,
    font_size: int,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    """Draw a black label on a white background with no border."""
    try:
        font = ImageFont.truetype(str(font), font_size)
    except OSError:
        font = ImageFont.load_default()
    left, _, right, _ = font.getbbox(text)
    width = right - left
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    image = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    return rearrange(image, "h w c -> c h w")

def add_label(
    image: Float[Tensor, "3 width height"],
    label: str,
    font: Path = Path("assets/Inter-Regular.otf"),
    font_size: int = 24,
) -> Float[Tensor, "3 width_with_label height_with_label"]:
    return vcat(
        draw_label(label, font, font_size, image.device),
        image,
        align="left",
        gap=4,
    )

def pad(images: list[Shaped[Tensor, "..."]]) -> list[Shaped[Tensor, "..."]]:
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [
        torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device)
        for x in images
    ]
    for image, result in zip(images, results):
        slices = [slice(0, x) for x in image.shape]
        result[slices] = image[slices]
    return results

def render_projections(
    gaussians: Gaussians,
    resolution: tuple[int, int],
    heading_shift_left,
    shift_u,
    shift_v,
    margin: float = 0.1,
    heading: Union[Tensor, None] = None,
    look_axis = 1,
    width = 101.0 / 2,
    height = 101.0 / 2,
) -> Float[Tensor, "batch 3 3 height width"]:
    device = gaussians.means.device
    B, V = heading_shift_left.shape
    gaussians.means = gaussians.means.view(B,V,-1,3)
    gaussians.covariances = gaussians.covariances.view(B,V,-1,3,3)
    gaussians.rgbs = gaussians.rgbs.view(B,V,-1,3)
    gaussians.opacities = gaussians.opacities.view(B,V,-1)
    gaussians.features = gaussians.features.view(B,V,-1,32)
    gaussians.confidence = gaussians.confidence.view(B,V,-1,1)
    if heading == None:
        heading = torch.zeros([B, 1], dtype=torch.float32, device=gaussians.means.device)
    color_out_batch = []
    feature_out_batch = []
    confidence_out_batch = []

    for b in range(B):
        color_out_view = []
        feature_out_view = []
        confidence_out_view = []
        for v in range(V):
            # Compute the minima and maxima of the scene.
            minima = gaussians.means[b:b+1,v].min(dim=1).values
            maxima = gaussians.means[b:b+1,v].max(dim=1).values
            scene_minima, scene_maxima = compute_equal_aabb_with_margin(
                minima, maxima, margin=margin / 2
            )

            # look = ["x", "y", "z"]
            # for look_axis in range(3):
            # look_axis = 0
            right_axis = (look_axis + 1) % 3
            down_axis = (look_axis + 2) % 3

            # Define the extrinsics for rendering.
            extrinsics = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
            extrinsics[:, right_axis, 0] = 1
            extrinsics[:, down_axis, 1] = 1
            extrinsics[:, look_axis, 2] = 1
            # extrinsics[:, right_axis, 3] = 0.5 * (
            #     scene_minima[:, right_axis] + scene_maxima[:, right_axis]
            # )
            # extrinsics[:, down_axis, 3] = 0.5 * (
            #     scene_minima[:, down_axis] + scene_maxima[:, down_axis]
            # )

            extrinsics[:, look_axis, 3] = scene_minima[:, look_axis]
            extrinsics[:, 3, 3] = 1
            cos = torch.cos(-heading_shift_left[b:b+1,v:v+1])
            sin = torch.sin(-heading_shift_left[b:b+1,v:v+1])
            zeros = torch.zeros_like(cos)
            ones = torch.ones_like(cos)
            R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
            R = R.view(1, 3, 3)  # shape = [B,3,3]
            # 将 R 扩展为 4x4 矩阵，形状为 [B, 4, 4]
            R_4x4 = torch.eye(4, device=device).unsqueeze(0)  # [1,4,4]
            R_4x4[:, :3, :3] = R  # 替换上半部分为旋转矩阵
            # 添加平移，x轴平移为-shift_v，z轴平移为-shift_u
            R_4x4[:, 0, 3] = -shift_u[b,v]  # x轴平移
            R_4x4[:, 2, 3] = -shift_v[b,v]  # z轴平移
            extrinsics_rotated = torch.bmm(R_4x4, extrinsics)  # [1,4,4]
            # Define the intrinsics for rendering.
            extents = scene_maxima - scene_minima
            far = extents[:, look_axis]
            near = torch.zeros_like(far)
            # width = extents[:, right_axis]
            # height = extents[:, down_axis]
            # extrinsics[:, right_axis, 3] = 0
            # extrinsics[:, down_axis, 3] = 0

            render_out = render_cuda_orthographic(
                extrinsics_rotated,
                width,
                height,
                near,
                far,
                resolution,
                torch.zeros((1, 3), dtype=torch.float32, device=device),
                gaussians.means[b:b+1,v],
                gaussians.covariances[b:b+1,v],
                gaussians.color_harmonics[b:b+1,v] if hasattr(gaussians, 'color_harmonics') else None,
                gaussians.opacities[b:b+1,v],
                gaussians.features[b:b+1,v],
                gaussians.confidence[b:b+1,v],
                gaussians.rgbs[b:b+1,v] if hasattr(gaussians, 'rgbs') else None,
                fov_degrees=0.1,
                use_sh=True,
            )
            color = render_out.color
            feature = render_out.feature
            confidence = render_out.confidence
            color_out_view.append(color)
            feature_out_view.append(feature)
            confidence_out_view.append(confidence)
        color_out_batch.append(torch.cat(color_out_view, dim=0))
        feature_out_batch.append(torch.cat(feature_out_view, dim=0))
        confidence_out_batch.append(torch.cat(confidence_out_view, dim=0))
    return torch.stack(color_out_batch, dim=0), torch.stack(feature_out_batch, dim=0), torch.stack(confidence_out_batch, dim=0)

def render_to_decoder_output(
    render_output,
    b: int,
) -> DecoderOutput:
    if render_output.feature is not None:
        features = render_output.feature
        # NOTE background feature = 0 = mean = logvar (of normal distribution)
        mean, logvar = (features, (1-rearrange(render_output.mask.detach(), "b h w -> b () h w", b=b)).log().expand_as(features))
        feature_posterior = DiagonalGaussianDistribution(mean, logvar)
    else:
        feature_posterior = None
    return DecoderOutput(
        color=render_output.color if render_output.color is not None else None,
        feature_posterior=feature_posterior,
        mask=render_output.mask,
        depth=render_output.depth
    )