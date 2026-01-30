"""
Leaves in natural images
===========================

Replication of Wallis and Bex, 2012
"""

import torch
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
from dead_leaves.utils import choose_compute_backend
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

reference_image = Image.open("../../examples/images/ulb_5136-2516.jpg").resize(
    (731, 512), box=(0, 0, 3810, 2667)
)
image_tensor = (
    pil_to_tensor(pic=reference_image).to(device=choose_compute_backend()) / 255
)

position_mask = torch.zeros((512, 731), device=choose_compute_backend())
Y, X = torch.meshgrid(
    torch.arange(731),
    torch.arange(512),
    indexing="xy",
)
for radius in [50, 120, 200]:
    for angle in [0, 0.5 * torch.pi, torch.pi, 1.5 * torch.pi]:
        x_pos = int(radius * torch.cos(torch.Tensor([angle]))) + 256
        y_pos = int(radius * torch.sin(torch.Tensor([angle]))) + 365.5
        area = torch.Tensor([radius * 10])
        dist_from_center = torch.sqrt((X - x_pos) ** 2 + (Y - y_pos) ** 2)
        mask = dist_from_center <= torch.sqrt(area / torch.pi)
        position_mask += mask

model = LeafGeometryGenerator(
    leaf_shape="ellipsoid",
    shape_param_distributions={
        "area": {"uniform": {"low": 100.0, "high": 200.0}},
        "aspect_ratio": {"uniform": {"low": 0.5, "high": 2}},
        "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
    },
    image_shape=(512, 731),
    position_mask=position_mask,
)
leaf_table, segmentation_map = model.generate_segmentation()

color_params = {
    "R": {
        "constant": {
            "value": {
                "from": ["x_pos", "y_pos"],
                "fn": lambda x: image_tensor[
                    0, x["y_pos"].astype(int), x["x_pos"].astype(int)
                ],
            }
        }
    },
    "G": {
        "constant": {
            "value": {
                "from": ["x_pos", "y_pos"],
                "fn": lambda x: image_tensor[
                    1, x["y_pos"].astype(int), x["x_pos"].astype(int)
                ],
            }
        }
    },
    "B": {
        "constant": {
            "value": {
                "from": ["x_pos", "y_pos"],
                "fn": lambda x: image_tensor[
                    2, x["y_pos"].astype(int), x["x_pos"].astype(int)
                ],
            }
        }
    },
}

colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(color_param_distributions=color_params)

renderer = ImageRenderer(
    leaf_table=colormodel.leaf_table, segmentation_map=segmentation_map
)
image = renderer.render_image()
mask = segmentation_map == 0
image[mask] = image_tensor.permute(1, 2, 0)[mask]
renderer.show(image)
