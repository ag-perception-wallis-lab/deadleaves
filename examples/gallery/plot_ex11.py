"""
Iris
===========================
"""

from dead_leaves import DeadLeavesModel, DeadLeavesImage
import torch


def angle_by_position(d):
    center_x = 365.5
    center_y = 255.5
    alpha = torch.arctan2(center_x - d["x_pos"], center_y - d["y_pos"])
    return alpha


position_mask = torch.zeros((512, 731))
Y, X = torch.meshgrid(
    torch.arange(731),
    torch.arange(512),
    indexing="xy",
)
dist_from_center = torch.sqrt((X - 255.5) ** 2 + (Y - 365.5) ** 2)
position_mask = (dist_from_center <= 200) & (dist_from_center >= 50)
position_mask = position_mask.to(bool)

model = DeadLeavesModel(
    "ellipsoid",
    {
        "area": {"uniform": {"low": 100.0, "high": 500.0}},
        "aspect_ratio": {"constant": {"value": 0.3}},
        "orientation": {
            "constant": {"value": {"from": ["x_pos", "y_pos"], "fn": angle_by_position}}
        },
    },
    (512, 731),
    position_mask,
)
leaves, partition = model.sample_partition()

color_params = {"gray": {"normal": {"loc": 0.5, "scale": 0.2}}}

colormodel = DeadLeavesImage(
    leaves,
    partition,
    color_params,
    background_color=torch.tensor([1.0, 1.0, 1.0]),
)
image = colormodel.sample_image()
colormodel.show(image)
