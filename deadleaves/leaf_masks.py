from dataclasses import dataclass
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
    bbox: Callable[[dict[str, torch.Tensor]], tuple[int, int, int, int]]
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
            bbox=circular_aabb,
        ),
        "ellipsoid": LeafMaskSpec(
            fn=ellipsoid,
            required={"x_pos", "y_pos", "area", "aspect_ratio", "orientation"},
            bbox=ellipsoid_aabb,
        ),
        "rectangular": LeafMaskSpec(
            fn=rectangular,
            required={"x_pos", "y_pos", "area", "aspect_ratio", "orientation"},
            bbox=rectangular_aabb,
        ),
        "polygon": LeafMaskSpec(
            fn=polygon,
            required={"x_pos", "y_pos", "area", "n_vertices"},
            bbox=polygon_aabb,
        ),
    }


# -------------------------------------------------------------------
# Individual functions
# -------------------------------------------------------------------

# -----------------------------------------------
# Circle
# -----------------------------------------------


def circular_validated_params(params: dict[str, torch.Tensor] | pd.Series) -> dict:
    """Validate parameters for a circular shape and return a canonical dict.

    Args:
        params (dict[str, torch.Tensor] | pd.Series):
            Circle parameter values.

    Raises:
        ValueError:
            Neither radius nor area or both are provided.
        ValueError:
            Area or radius is negative.

    Returns:
        dict:
            Canonical dict for circle parameter values.
    """
    keys = params.keys() if isinstance(params, dict) else params.index
    if not (("area" in keys) ^ ("radius" in keys)):
        raise ValueError("Either radius or area must be provided.")
    for key in ("area", "radius"):
        if key in keys and params[key] < 0:
            raise ValueError(f"Provided {key} must be non-negative.")

    p = dict(params) if isinstance(params, dict) else params.to_dict()

    if "radius" not in p:
        p["radius"] = torch.sqrt(p["area"] / torch.pi)
    return p


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
    p = circular_validated_params(params)
    dist_from_center = torch.sqrt(
        (X - params["x_pos"]) ** 2 + (Y - params["y_pos"]) ** 2
    )
    mask = dist_from_center <= p["radius"]
    return mask


def circular_aabb(
    params: dict[str, torch.Tensor] | pd.Series
) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box for a circle.

    Args:
        params (dict[str, torch.Tensor] | pd.Series):
            Circle parameter values.

    Returns:
        tuple[int, int, int, int]:
            (y_min, x_min, y_max, x_max) as ints, not yet clipped to canvas.
    """
    p = circular_validated_params(params)

    cx = float(p["x_pos"])
    cy = float(p["y_pos"])
    r = float(p["radius"])

    return (
        math.floor(cy - r),
        math.floor(cx - r),
        math.ceil(cy + r),
        math.ceil(cx + r),
    )


# -----------------------------------------------
# Rectangle
# -----------------------------------------------


def rectangular_validated_params(params: dict[str, torch.Tensor] | pd.Series) -> dict:
    """Validate parameters for a rectangular shape and return a canonical dict.

    Args:
        params (dict[str, torch.Tensor] | pd.Series):
            Rectangle parameter values.

    Raises:
        ValueError:
            Area is negative.
        ValueError:
            Aspect ratio is non-positive.

    Returns:
        dict:
            Canonical dict for rectangle parameter values.
    """
    if params["area"] < 0:
        raise ValueError("Provided area must be non-negative.")
    if params["aspect_ratio"] <= 0:
        raise ValueError("Provided aspect_ratio must be positive.")
    p = dict(params) if isinstance(params, dict) else params.to_dict()
    return p


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
    p = rectangular_validated_params(params)
    with X.device:
        height = torch.sqrt(p["area"] / p["aspect_ratio"])
        width = height * p["aspect_ratio"]
        sin = torch.sin(p["orientation"])
        cos = torch.cos(p["orientation"])
        dx = X - p["x_pos"]
        dy = Y - p["y_pos"]
        X = dx * cos - dy * sin
        Y = dx * sin + dy * cos
        mask = (torch.abs(X) <= width / 2) & (torch.abs(Y) <= height / 2)
    return mask


def rectangular_aabb(
    params: dict[str, torch.Tensor] | pd.Series
) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box for a rectangle.

    Args:
        params (dict[str, torch.Tensor] | pd.Series):
            Rectangle parameter values.

    Returns:
        tuple[int, int, int, int]:
            (y_min, x_min, y_max, x_max) as ints, not yet clipped to canvas.
    """
    p = rectangular_validated_params(params)

    cx = float(p["x_pos"])
    cy = float(p["y_pos"])
    area = float(p["area"])
    aspect = float(p["aspect_ratio"])
    theta = float(p["orientation"])

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


# -----------------------------------------------
# Ellipsoid
# -----------------------------------------------


def ellipsoid_validated_params(params: dict[str, torch.Tensor] | pd.Series) -> dict:
    """Validate parameters for a ellipsoidal shape and return a canonical dict.

    Args:
        params (dict[str, torch.Tensor] | pd.Series):
            Ellipsoid parameter values.

    Raises:
        ValueError:
            Area is negative.
        ValueError:
            Aspect ratio is non-positive.

    Returns:
        dict:
            Canonical dict for ellipsoid parameter values.
    """
    if params["area"] <= 0:
        raise ValueError("Provided area must be positive.")
    if params["aspect_ratio"] <= 0:
        raise ValueError("Provided aspect_ratio must be positive.")
    p = dict(params) if isinstance(params, dict) else params.to_dict()
    return p


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
    p = ellipsoid_validated_params(params)
    with X.device:
        a = torch.sqrt((p["area"] * p["aspect_ratio"]) / torch.pi)
        b = torch.sqrt(p["area"] / (torch.pi * p["aspect_ratio"]))
        sin = torch.sin(p["orientation"])
        cos = torch.cos(p["orientation"])
        dx = X - p["x_pos"]
        dy = Y - p["y_pos"]
        X = dx * cos - dy * sin
        Y = dx * sin + dy * cos
        mask = (X / a) ** 2 + (Y / b) ** 2 <= 1
    return mask


def ellipsoid_aabb(
    params: dict[str, torch.Tensor] | pd.Series
) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box for a ellipsoid.

    Args:
        params (dict[str, torch.Tensor] | pd.Series):
            Ellipsoid parameter values.

    Returns:
        tuple[int, int, int, int]:
            (y_min, x_min, y_max, x_max) as ints, not yet clipped to canvas.
    """
    p = ellipsoid_validated_params(params)

    cx = float(p["x_pos"])
    cy = float(p["y_pos"])
    area = float(p["area"])
    aspect = float(p["aspect_ratio"])
    theta = float(p["orientation"])

    a = math.sqrt(area * aspect / math.pi)
    b = math.sqrt(area / (math.pi * aspect))
    # Ellipse: bounding half-widths use Pythagorean formula
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    hx = math.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
    hy = math.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)
    return (
        math.floor(cy - hy),
        math.floor(cx - hx),
        math.ceil(cy + hy),
        math.ceil(cx + hx),
    )


# -----------------------------------------------
# Regular Polygon
# -----------------------------------------------


def polygon_validated_params(params: dict[str, torch.Tensor] | pd.Series) -> dict:
    """Validate parameters for a polygon shape and return a canonical dict.

    Args:
        params (dict[str, torch.Tensor] | pd.Series):
            Polygon parameter values.

    Raises:
        ValueError:
            Area is negative.
        ValueError:
            Number of vertices is smaller than 3.
        ValueError:
            Number of vertices is non-integer.

    Returns:
        dict:
            Canonical dict for polygon parameter values.
    """
    if params["area"] <= 0:
        raise ValueError("Provided area must be positive.")
    if params["n_vertices"] < 3:
        raise ValueError("Provided number of vertices must be at least 3.")
    if params["n_vertices"] != int(params["n_vertices"]):
        raise ValueError("Provided number of vertices must be an integer.")
    p = dict(params) if isinstance(params, dict) else params.to_dict()
    return p


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
    p = polygon_validated_params(params)
    with X.device:
        radius = torch.sqrt(
            2
            * p["area"]
            / (p["n_vertices"] * torch.sin(2 * torch.pi / p["n_vertices"]))
        )
        angles = torch.linspace(0.0, 2 * torch.pi, int(p["n_vertices"]) + 1)[:-1]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        vertices = torch.stack(
            (
                p["x_pos"] + radius * cos_angles,
                p["y_pos"] + radius * sin_angles,
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


def polygon_aabb(
    params: dict[str, torch.Tensor] | pd.Series
) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box for a polygon.

    Args:
        params (dict[str, torch.Tensor] | pd.Series):
            Polygon parameter values.

    Returns:
        tuple[int, int, int, int]:
            (y_min, x_min, y_max, x_max) as ints, not yet clipped to canvas.
    """
    p = polygon_validated_params(params)

    cx = float(p["x_pos"])
    cy = float(p["y_pos"])
    area = float(p["area"])
    n_v = int(p["n_vertices"])

    r = math.sqrt(2 * area / (n_v * math.sin(2 * math.pi / n_v)))
    return (
        math.floor(cy - r),
        math.floor(cx - r),
        math.ceil(cy + r),
        math.ceil(cx + r),
    )
