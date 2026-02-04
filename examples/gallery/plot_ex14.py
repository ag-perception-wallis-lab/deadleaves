"""
Sampling animation
===========================
"""

from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
from IPython.display import HTML
import torch

model = LeafGeometryGenerator(
    leaf_shape="circular",
    shape_param_distributions={
        "area": {"powerlaw": {"low": 5000.0, "high": 10000.0, "k": 1.5}}
    },
    image_shape=(512, 731),
    n_sample=500,
)
leaf_table, segmentation_map = model.generate_segmentation()
colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(
    color_param_distributions={"gray": {"uniform": {"low": 0.0, "high": 1.0}}}
)

renderer = ImageRenderer(
    colormodel.leaf_table, segmentation_map, background_color=torch.tensor(1)
)

ani = renderer.animate()
HTML(ani.to_jshtml())
