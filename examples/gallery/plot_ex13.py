"""
Proto-object in motion
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

shared_params = {
    "leaf_shape": "circular",
    "shape_param_distributions": {
        "area": {"uniform": {"low": area * 0.005, "high": area * 0.01}}
    },
    "image_shape": image_shape,
}

table_background, _ = LeafGeometryGenerator(
    **shared_params,
    n_sample=200,
).generate_segmentation()

table_target, _ = LeafGeometryGenerator(
    **shared_params,
    n_sample=10,
    position_mask={"shape": "circular", "params": {"area": area * 0.05}},
).generate_segmentation()

table_background = LeafAppearanceSampler(leaf_table=table_background).sample_color(
    color_param_distributions={"gray": {"normal": {"loc": 0.5, "scale": 0.2}}},
)

table_target = LeafAppearanceSampler(leaf_table=table_target).sample_color(
    color_param_distributions={
        "R": {"uniform": {"low": 0.2, "high": 1.0}},
        "G": {"constant": {"value": 0.0}},
        "B": {"constant": {"value": 0.0}},
    },
)

table_background["type"] = "background"
table_target["type"] = "target"

topology = LeafTopology(image_shape=image_shape)

table_merged = topology.merge_leaf_tables(table_background, table_target)
table_merged = LeafTopology.randomize_index(table_merged, seed=42)


def move_leaves_one_step(
    leaf_table: pd.DataFrame,
    frame_idx: int,
    image_shape: tuple[int, int],
    radius: float = 5.0,
    target_velocity: tuple[float, float] = (1.0, 0.0),
    bg_angles: torch.Tensor | None = None,
    bg_angular_velocities: torch.Tensor | None = None,
) -> pd.DataFrame:
    table = leaf_table.copy()

    # Identify background and target leaves
    bg_mask = table["type"] != "target"
    target_mask = table["type"] == "target"

    n_bg = bg_mask.sum()
    bg_indices = table.index[bg_mask]

    # Initialize angles and angular velocities if not provided
    if bg_angles is None:
        bg_angles = torch.distributions.uniform.Uniform(0, 2 * torch.pi).sample((n_bg,))
    if bg_angular_velocities is None:
        bg_angular_velocities = torch.distributions.uniform.Uniform(0.02, 0.05).sample(
            (n_bg,)
        )

    # Move background leaves
    for i, idx in enumerate(bg_indices):
        angle = bg_angles[i] + frame_idx * bg_angular_velocities[i]
        dx = radius * torch.cos(angle)
        dy = radius * torch.sin(angle)
        table.loc[idx, "x_pos"] = table.loc[idx, "x_pos"] + dx
        table.loc[idx, "y_pos"] = table.loc[idx, "y_pos"] + dy

    # Move target leaves linearly
    table.loc[target_mask, "x_pos"] = (
        leaf_table.loc[target_mask, "x_pos"] + frame_idx * target_velocity[0]
    ) % image_shape[1]
    table.loc[target_mask, "y_pos"] = (
        leaf_table.loc[target_mask, "y_pos"] + frame_idx * target_velocity[1]
    ) % image_shape[0]
    return table


frames = []
n_frames = 10
for t in range(n_frames):
    table = move_leaves_one_step(
        leaf_table=table_merged,
        image_shape=image_shape,
        frame_idx=t,
        radius=5.0,
        target_velocity=(image_shape[1] / n_frames, 0.0),
    )

    renderer = ImageRenderer(
        leaf_table=table,
        image_shape=image_shape,
        background_color=torch.tensor(1.0),
    )
    image = renderer.render_image()
    frames.append(image.cpu())

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
    frames=n_frames,
    interval=100,
    blit=True,
)

plt.close(fig)
HTML(ani.to_jshtml())
