from dataclasses import dataclass
from functools import partial
import math
from typing import Callable

import torch
import pandas as pd

# -------------------------------------------------------------------
# Types and registry spec
# -------------------------------------------------------------------

MaskFn = Callable[
    [tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor] | pd.Series],
    torch.Tensor,
]
"""Mask Function type"""


@dataclass(frozen=True)
class LeafMaskSpec:
    """Leaf mask data class"""

    fn: MaskFn
    """Leaf mask function, mapping shape parameters to a leaf mask."""
    required: set[str] | list[set[str]]
    """Parameters of the leaf shape."""
    bbox: Callable[[dict[str, torch.Tensor], str], tuple[int, int, int, int]]
    """Axis-aligned bounding box function."""


# -------------------------------------------------------------------
# Registry function / Overview
# -------------------------------------------------------------------


def get_leaf_mask_kw() -> dict[str, LeafMaskSpec]:
    """
    Return dictionary mapping leaf shapes to their mask functions
    and required parameters.
    """
    return {
        "circular": LeafMaskSpec(
            fn=circular,
            required=[{"x_pos", "y_pos", "area"}, {"x_pos", "y_pos", "radius"}],
            bbox=partial(leaf_aabb, leaf_shape="circular"),
        ),
        "ellipsoid": LeafMaskSpec(
            fn=ellipsoid,
            required={"x_pos", "y_pos", "area", "aspect_ratio", "orientation"},
            bbox=partial(leaf_aabb, leaf_shape="ellipsoid"),
        ),
        "rectangular": LeafMaskSpec(
            fn=rectangular,
            required={"x_pos", "y_pos", "area", "aspect_ratio", "orientation"},
            bbox=partial(leaf_aabb, leaf_shape="rectangular"),
        ),
        "polygon": LeafMaskSpec(
            fn=polygon,
            required={"x_pos", "y_pos", "area", "n_vertices"},
            bbox=partial(leaf_aabb, leaf_shape="polygon"),
        ),
    }


# -------------------------------------------------------------------
# Individual functions
# -------------------------------------------------------------------


def circular(
    index_grid: tuple[torch.Tensor, torch.Tensor],
    params: dict[str, torch.Tensor] | pd.Series,
) -> torch.Tensor:
    """Generate mask of circle from given area and x-y-position on tensor.

    Args:
        index_grid (tuple[tensor, tensor]):
            x and y indices of area to be masked.
        params (dict[str, tensor]):
            Value for each parameter.

    Returns:
        torch.Tensor:
            Leaf mask.
    """
    X, Y = index_grid
    keys = params.keys() if isinstance(params, dict) else params.index
    if not (("area" in keys) ^ ("radius" in keys)):
        raise ValueError("Either radius or area must be provided.")
    for key in ("area", "radius"):
        if key in keys and params[key] < 0:
            raise ValueError(f"Provided {key} must be non-negative.")
    dist_from_center = torch.sqrt(
        (X - params["x_pos"]) ** 2 + (Y - params["y_pos"]) ** 2
    )
    if "area" in keys:
        mask = dist_from_center <= torch.sqrt(params["area"] / torch.pi)
    else:
        mask = dist_from_center <= params["radius"]
    return mask


def rectangular(
    index_grid: tuple[torch.Tensor, torch.Tensor],
    params: dict[str, torch.Tensor] | pd.Series,
) -> torch.Tensor:
    """Generate mask of rectangle from given area, aspect ratio, orientation,
    and x-y-position on tensor.

    Args:
        index_grid (tuple[tensor, tensor]):
            x and y indices of area to be masked.
        params (dict[str, tensor]):
            Value for each parameter.

    Returns:
        torch.Tensor:
            Leaf mask.
    """
    X, Y = index_grid
    if params["area"] < 0:
        raise ValueError("Provided area must be non-negative.")
    if params["aspect_ratio"] <= 0:
        raise ValueError("Provided aspect_ratio must be positive.")
    with X.device:
        height = torch.sqrt(params["area"] / params["aspect_ratio"])
        width = height * params["aspect_ratio"]
        sin = torch.sin(params["orientation"])
        cos = torch.cos(params["orientation"])
        dx = X - params["x_pos"]
        dy = Y - params["y_pos"]
        X = dx * cos - dy * sin
        Y = dx * sin + dy * cos
        mask = (torch.abs(X) <= width / 2) & (torch.abs(Y) <= height / 2)
    return mask


def ellipsoid(
    index_grid: tuple[torch.Tensor, torch.Tensor],
    params: dict[str, torch.Tensor] | pd.Series,
) -> torch.Tensor:
    """Generate mask of ellipsoid from given area, aspect ratio, orientation,
    and x-y-position on tensor.

    Args:
        index_grid (tuple[tensor, tensor]):
            x and y indices of area to be masked.
        params (dict[str, tensor]):
            Value for each parameter.

    Returns:
        torch.Tensor:
            Leaf mask.
    """
    X, Y = index_grid
    if params["area"] <= 0:
        raise ValueError("Provided area must be positive.")
    if params["aspect_ratio"] <= 0:
        raise ValueError("Provided aspect_ratio must be positive.")
    with X.device:
        a = torch.sqrt((params["area"] * params["aspect_ratio"]) / torch.pi)
        b = torch.sqrt(params["area"] / (torch.pi * params["aspect_ratio"]))
        sin = torch.sin(params["orientation"])
        cos = torch.cos(params["orientation"])
        dx = X - params["x_pos"]
        dy = Y - params["y_pos"]
        X = dx * cos - dy * sin
        Y = dx * sin + dy * cos
        mask = (X / a) ** 2 + (Y / b) ** 2 <= 1
    return mask


def polygon(
    index_grid: tuple[torch.Tensor, torch.Tensor],
    params: dict[str, torch.Tensor] | pd.Series,
) -> torch.Tensor:
    """Generate mask of regular polygon from given area, number of vertices
    and x-y-position on tensor.

    Args:
        index_grid (tuple[tensor, tensor]):
            x and y indices of area to be masked.
        params (dict[str, tensor]):
            Value for each parameter.

    Returns:
        torch.Tensor:
            Leaf mask.
    """
    X, Y = index_grid
    if params["area"] <= 0:
        raise ValueError("Provided area must be positive.")
    if params["n_vertices"] <= 0:
        raise ValueError("Provided number of vertices must be positive.")
    if params["n_vertices"] != int(params["n_vertices"]):
        raise ValueError("Provided number of vertices must be an integer.")
    with X.device:
        radius = torch.sqrt(
            2
            * params["area"]
            / (params["n_vertices"] * torch.sin(2 * torch.pi / params["n_vertices"]))
        )
        angles = torch.linspace(0.0, 2 * torch.pi, int(params["n_vertices"]))
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        vertices = torch.stack(
            (
                params["x_pos"] + radius * cos_angles,
                params["y_pos"] + radius * sin_angles,
            ),
            dim=1,
        )
        n = vertices.size(0)

        x_coords, y_coords = X.ravel(), Y.ravel()
        mask = torch.zeros(x_coords.shape[0], dtype=torch.bool)

        # ray casting algorithm
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]

            y_range_condition = (v1[1] > y_coords) != (v2[1] > y_coords)
            x_intersection = (v2[0] - v1[0]) * (y_coords - v1[1]) / (
                v2[1] - v1[1]
            ) + v1[0]
            x_range_condition = x_coords < x_intersection

            mask ^= y_range_condition & x_range_condition

    return mask.reshape(X.shape)


# -----------------------------------------
# Function to get axis-aligned bounding box
# -----------------------------------------


def leaf_aabb(
    params: dict[str, torch.Tensor],
    leaf_shape: str,
) -> tuple[int, int, int, int]:
    """Compute the axis-aligned bounding box of a leaf.

    Args:
        params:
            Sampled leaf parameters (x_pos, y_pos, area / radius, …).
        leaf_shape:
            One of "circular", "ellipsoid", "rectangular", "polygon".

    Returns:
        (y_min, x_min, y_max, x_max) as ints, not yet clipped to canvas.
    """
    cx = float(params["x_pos"])
    cy = float(params["y_pos"])

    if leaf_shape == "circular":
        keys = params.keys() if isinstance(params, dict) else params.index
        if "radius" in keys:
            r = float(params["radius"])
        else:
            r = math.sqrt(float(params["area"]) / math.pi)
        return (
            math.floor(cy - r),
            math.floor(cx - r),
            math.ceil(cy + r),
            math.ceil(cx + r),
        )

    if leaf_shape in ("ellipsoid", "rectangular"):
        area = float(params["area"])
        aspect = float(params["aspect_ratio"])
        theta = float(params["orientation"])

        if leaf_shape == "ellipsoid":
            a = math.sqrt(area * aspect / math.pi)
            b = math.sqrt(area / (math.pi * aspect))
            # Ellipse: bounding half-widths use Pythagorean formula
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            hx = math.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
            hy = math.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)
        else:  # rectangular
            h = math.sqrt(area / aspect)
            w = h * aspect
            a = w / 2
            b = h / 2
            # Rectangle: bounding half-widths use absolute-value sum
            abs_cos = abs(math.cos(theta))
            abs_sin = abs(math.sin(theta))
            hx = a * abs_cos + b * abs_sin
            hy = a * abs_sin + b * abs_cos
        return (
            math.floor(cy - hy),
            math.floor(cx - hx),
            math.ceil(cy + hy),
            math.ceil(cx + hx),
        )

    if leaf_shape == "polygon":
        area = float(params["area"])
        n_v = int(params["n_vertices"])
        r = math.sqrt(2 * area / (n_v * math.sin(2 * math.pi / n_v)))
        return (
            math.floor(cy - r),
            math.floor(cx - r),
            math.ceil(cy + r),
            math.ceil(cx + r),
        )

    # Fallback — unknown shape, return infinite box (no acceleration)
    return (-(10**9), -(10**9), 10**9, 10**9)
