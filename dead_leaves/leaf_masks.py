import torch


def circular(
    index_grid: tuple[torch.Tensor, torch.Tensor], params: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Generate mask of circle from given area and x-y-position on tensor.

    Args:
        index_grid (tuple[tensor, tensor]): x and y indices of area to be masked.
        params (dict[str, tensor]): Value for each parameter.

    Returns:
        torch.Tensor: Leaf mask.
    """
    X, Y = index_grid
    dist_from_center = torch.sqrt(
        (X - params["x_pos"]) ** 2 + (Y - params["y_pos"]) ** 2
    )
    mask = dist_from_center <= torch.sqrt(params["area"] / torch.pi)
    return mask


def rectangular(
    index_grid: tuple[torch.Tensor, torch.Tensor], params: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Generate mask of rectangle from given area, aspect ratio, orientation,
    and x-y-position on tensor.

    Args:
        index_grid (tuple[tensor, tensor]): x and y indices of area to be masked.
        params (dict[str, tensor]): Value for each parameter.

    Returns:
        torch.Tensor: Leaf mask.
    """
    X, Y = index_grid
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
    index_grid: tuple[torch.Tensor, torch.Tensor], params: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Generate mask of ellipsoid from given area, aspect ratio, orientation,
    and x-y-position on tensor.

    Args:
        index_grid (tuple[tensor, tensor]): x and y indices of area to be masked.
        params (dict[str, tensor]): Value for each parameter.

    Returns:
        torch.Tensor: Leaf mask.
    """
    X, Y = index_grid
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


def regular_polygon(
    index_grid: tuple[torch.Tensor, torch.Tensor], params: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Generate mask of regular polygon from given area, number of vertices
    and x-y-position on tensor.

    Args:
        index_grid (tuple[tensor, tensor]): x and y indices of area to be masked.
        params (dict[str, tensor]): Value for each parameter.

    Returns:
        torch.Tensor: Leaf mask.
    """
    X, Y = index_grid
    radius = torch.sqrt(
        2
        * params["area"]
        / (params["n_vertices"] * torch.sin(2 * torch.pi / params["n_vertices"]))
    )
    angles = torch.linspace(0, 2 * torch.pi, params["n_vertices"].int())
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
        x_intersection = (v2[0] - v1[0]) * (y_coords - v1[1]) / (v2[1] - v1[1]) + v1[0]
        x_range_condition = x_coords < x_intersection

        mask ^= y_range_condition & x_range_condition

    return mask.reshape(X.shape)
