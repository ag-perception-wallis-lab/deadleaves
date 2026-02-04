"""
Example 10
===========================
"""

import torch
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
from dead_leaves.utils import choose_compute_backend

position_mask = torch.zeros((512, 731), device=choose_compute_backend())
Y, X = torch.meshgrid(
    torch.arange(731),
    torch.arange(512),
    indexing="xy",
)
dist_from_center = torch.sqrt((X - 255.5) ** 2 + (Y - 365.5) ** 2)
position_mask = (dist_from_center <= 200) & (dist_from_center >= 50)

model = LeafGeometryGenerator(
    leaf_shape="ellipsoid",
    shape_param_distributions={
        "area": {"uniform": {"low": 100.0, "high": 200.0}},
        "aspect_ratio": {"uniform": {"low": 0.5, "high": 2}},
        "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
    },
    image_shape=(512, 731),
    position_mask=position_mask,
    n_sample=1000,
)
leaf_table, segmentation_map = model.generate_segmentation()

color_params = {
    "H": {"normal": {"loc": 0.4, "scale": 0.05}},
    "S": {"normal": {"loc": 0.5, "scale": 0.1}},
    "V": {"normal": {"loc": 0.5, "scale": 0.1}},
}

colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(color_param_distributions=color_params)

renderer = ImageRenderer(
    leaf_table=colormodel.leaf_table,
    segmentation_map=segmentation_map,
    background_color=torch.tensor([0.5, 0.5, 0.5]),
)
renderer.render_image()
renderer.show()
