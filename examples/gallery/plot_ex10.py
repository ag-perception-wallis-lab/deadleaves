"""
Example 10
===========================
"""

import torch
from dead_leaves import DeadLeavesModel, DeadLeavesImage
from dead_leaves.utils import choose_compute_backend

position_mask = torch.zeros((512, 512))
X, Y = torch.meshgrid(
    torch.arange(512),
    torch.arange(512),
    indexing="xy",
)
dist_from_center = torch.sqrt((X - 255.5) ** 2 + (Y - 255.5) ** 2)
position_mask = (dist_from_center <= 200) & (dist_from_center >= 50)
position_mask = position_mask.to(device=choose_compute_backend()).to(bool)

model = DeadLeavesModel(
    "ellipsoid",
    {
        "area": {"uniform": {"low": 100.0, "high": 200.0}},
        "aspect_ratio": {"uniform": {"low": 0.5, "high": 2}},
        "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
    },
    (512, 512),
    position_mask,
    n_sample=1000,
)
leaves, partition = model.sample_partition()

color_params = {
    "H": {"normal": {"loc": 0.4, "scale": 0.05}},
    "S": {"normal": {"loc": 0.5, "scale": 0.1}},
    "V": {"normal": {"loc": 0.5, "scale": 0.1}},
}

colormodel = DeadLeavesImage(
    leaves,
    partition,
    color_params,
    background_color=torch.tensor([0.5, 0.5, 0.5], device=choose_compute_backend()),
)
image = colormodel.sample_image()
colormodel.show(image)

colormodel.save(image, "images/sphx_glr_plot_ex10_001.png")
colormodel.save(image, "images/thumb/sphx_glr_plot_ex10_thumb.png")
