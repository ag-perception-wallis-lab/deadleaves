"""
Circular motion
===========================
"""

import pandas as pd
import torch

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from deadleaves import (
    LeafGeometryGenerator,
    LeafAppearanceSampler,
    ImageRenderer,
    LeafTopology,
)

image_shape = (512, 731)
area = image_shape[0] * image_shape[1]

geometry = LeafGeometryGenerator(
    leaf_shape="circular",
    shape_param_distributions={
        "area": {"uniform": {"low": area * 0.005, "high": area * 0.01}},
    },
    image_shape=image_shape,
    n_sample=250,
    position_mask={
        "shape": "circular",
        "params": {
            "area": area * 0.35,
        },
    },
)

leaf_table, segmentation_map = geometry.generate_segmentation()

color_params = {
    "H": {"uniform": {"low": 0.0, "high": 0.2}},
    "S": {"uniform": {"low": 0.0, "high": 1.0}},
    "V": {"uniform": {"low": 0.0, "high": 1.0}},
}

appearance = LeafAppearanceSampler(
    leaf_table=leaf_table,
)

table = appearance.sample_color(color_param_distributions=color_params)

topology = LeafTopology(image_shape=image_shape)


def apply_circular_motion(
    leaf_table: pd.DataFrame,
    image_size: tuple[int, int] | torch.Size,
    angle_step: float,
    angular_jitter: float = 0.0,
    radial_jitter: float = 0.0,
) -> pd.DataFrame:

    table = leaf_table.copy()

    cy = image_size[0] / 2
    cx = image_size[1] / 2

    x = torch.tensor(table["x_pos"]) - cx
    y = torch.tensor(table["y_pos"]) - cy

    # Polar coordinates
    r = torch.sqrt(x**2 + y**2)
    theta = torch.arctan2(y, x)

    # update angle
    theta += angle_step
    if angular_jitter > 0:
        theta += torch.distributions.normal.Normal(0.0, angular_jitter).sample(
            (len(theta),)
        )

    # update radius
    if radial_jitter > 0:
        r += torch.distributions.normal.Normal(0.0, radial_jitter).sample((len(r),))
        r = torch.clip(r, 0.0, None)

    # back to Cartesian
    table["x_pos"] = cx + r * torch.cos(theta)
    table["y_pos"] = cy + r * torch.sin(theta)

    return table


frames = []
for t in range(10):
    table = apply_circular_motion(
        leaf_table=table,
        image_size=segmentation_map.shape,
        angle_step=torch.pi / 10,
        angular_jitter=0.1,
    )

    renderer = ImageRenderer(
        leaf_table=table,
        image_shape=image_shape,
        background_color=torch.tensor(1.0),
    )
    image = renderer.render_image()
    frames.append(image.cpu())


# Generate gif
fig, ax = plt.subplots(figsize=(7.31, 5.12))
im = ax.imshow(frames[0])
ax.axis("off")
ax.set_position((0.0, 0.0, 1.0, 1.0))


def update(i):
    im.set_data(frames[i])
    return [im]


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=100,
    blit=True,
)

plt.close(fig)
HTML(ani.to_jshtml())
