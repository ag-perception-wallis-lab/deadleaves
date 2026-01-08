"""
Example 5
===========================
"""

from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape="rectangular",
    param_distributions={
        "area": {"powerlaw": {"low": 100.0, "high": 5000.0, "k": 1.5}},
        "aspect_ratio": {"uniform": {"low": 0.001, "high": 10}},
        "orientation": {"constant": {"value": 0.0}},
    },
    size=(512, 731),
)
leaves, partition = model.sample_partition()
colormodel = DeadLeavesImage(
    leaves=leaves,
    partition=partition,
    color_param_distributions={"gray": {"uniform": {"low": 0.0, "high": 1.0}}},
)
image = colormodel.sample_image()
colormodel.show(image)
