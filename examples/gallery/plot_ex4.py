"""
Example 4
===========================

Rotated square leaves (Baradad et al., 2022)
"""

import torch
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    "rectangular",
    {
        "area": {"powerlaw": {"low": 500.0, "high": 10000.0, "k": 1.5}},
        "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
        "aspect_ratio": {"constant": {"value": 1}},
    },
    (512, 731),
)
leaves, partition = model.sample_partition()

color_params = {
    "R": {"uniform": {"low": 0.0, "high": 1.0}},
    "G": {"uniform": {"low": 0.0, "high": 1.0}},
    "B": {"uniform": {"low": 0.0, "high": 1.0}},
}

colormodel = DeadLeavesImage(leaves, partition, color_params)
image = colormodel.sample_image()
colormodel.show(image)
