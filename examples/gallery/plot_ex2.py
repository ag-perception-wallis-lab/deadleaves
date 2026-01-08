"""
Example 2
===========================
"""

from dead_leaves import DeadLeavesModel, DeadLeavesImage
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

reference_image = Image.open("ulb_5136-2516.jpg").resize(
    (731, 512), box=(0, 0, 3810, 2667)
)

image_tensor = pil_to_tensor(pic=reference_image) / 255

model = DeadLeavesModel(
    "circular",
    {"area": {"powerlaw": {"low": 100.0, "high": 5000.0, "k": 1.5}}},
    (512, 731),
)
leaves, partition = model.sample_partition()

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

colormodel = DeadLeavesImage(leaves, partition, color_params)
image = colormodel.sample_image()
colormodel.show(image)
