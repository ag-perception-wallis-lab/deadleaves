"""
Example 2
===========================
"""

from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

reference_image = Image.open("../../examples/images/ulb_5136-2516.jpg").resize(
    (731, 512), box=(0, 0, 3810, 2667)
)

image_tensor = pil_to_tensor(pic=reference_image) / 255

model = LeafGeometryGenerator(
    leaf_shape="circular",
    shape_param_distributions={
        "area": {"powerlaw": {"low": 100.0, "high": 5000.0, "k": 1.5}}
    },
    image_shape=(512, 731),
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
renderer.render_image()
renderer.show()
