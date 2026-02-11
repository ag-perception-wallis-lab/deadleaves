"""
Leaves in natural images
===========================

Replication of Wallis and Bex, 2012
"""

import torch
import pandas as pd
from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
from deadleaves.utils import choose_compute_backend
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

device = choose_compute_backend()
reference_image = Image.open("../../examples/images/ulb_5136-2516.jpg").resize(
    (731, 512), box=(0, 0, 3810, 2667)
)
image_tensor = pil_to_tensor(pic=reference_image).to(device=device) / 255

Y, X = torch.meshgrid(
    torch.arange(731),
    torch.arange(512),
    indexing="xy",
)

combined_leaf_table = pd.DataFrame()
for radius in [50, 120, 200]:
    for angle in [0, 0.5 * torch.pi, torch.pi, 1.5 * torch.pi]:
        x_pos = int(radius * torch.cos(torch.Tensor([angle]))) + 256
        y_pos = int(radius * torch.sin(torch.Tensor([angle]))) + 365.5
        area = torch.Tensor([radius * 10])
        dist_from_center = torch.sqrt((X - x_pos) ** 2 + (Y - y_pos) ** 2)
        mask = (dist_from_center <= torch.sqrt(area / torch.pi)).to(device)

        model = LeafGeometryGenerator(
            leaf_shape="ellipsoid",
            shape_param_distributions={
                "area": {"uniform": {"low": 100.0, "high": 200.0}},
                "aspect_ratio": {"uniform": {"low": 0.5, "high": 2}},
                "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
            },
            image_shape=(512, 731),
            position_mask=mask,
        )
        leaf_table, segmentation_map = model.generate_segmentation()
        combined_leaf_table = pd.concat(
            [combined_leaf_table, leaf_table], ignore_index=True
        )

combined_leaf_table.leaf_idx = combined_leaf_table.index + 1

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

colormodel = LeafAppearanceSampler(leaf_table=combined_leaf_table)
colormodel.sample_color(color_param_distributions=color_params)

renderer = ImageRenderer(leaf_table=colormodel.leaf_table, image_shape=(512, 731))
image = renderer.render_image()
mask = renderer.segmentation_map == 0
image[mask] = image_tensor.permute(1, 2, 0)[mask]
renderer.show(image)
