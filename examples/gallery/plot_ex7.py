"""
Example 7
===========================
"""

from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape="circular",
    param_distributions={
        "area": {"powerlaw": {"low": 100.0, "high": 5000.0, "k": 1.5}}
    },
    size=(512, 512),
)
leaves, partition = model.sample_partition()
colormodel = DeadLeavesImage(
    leaves=leaves,
    partition=partition,
    color_param_distributions={
        "H": {"normal": {"loc": 0.6, "scale": 0.05}},
        "S": {"normal": {"loc": 0.5, "scale": 0.1}},
        "V": {"normal": {"loc": 0.5, "scale": 0.2}},
    },
    texture_param_distributions={
        "gray": {"normal": {"loc": 0, "scale": {"uniform": {"low": 0.01, "high": 0.1}}}}
    },
)
image = colormodel.sample_image()
colormodel.show(image)
