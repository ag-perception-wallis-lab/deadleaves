"""
Natural colors and textures
===========================

replication of Madhusudana et al., 2022
"""

from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape="circular",
    param_distributions={
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
    },
    size=(512, 731),
)
leaves, partition = model.sample_partition()
colormodel = DeadLeavesImage(
    leaves=leaves,
    partition=partition,
    color_param_distributions={"source": {"image": {"dir": "../../examples/images"}}},
    texture_param_distributions={
        "source": {"image": {"dir": "../../examples/textures/brodatz"}},
        "alpha": {"normal": {"loc": 0.0, "scale": 0.4}},
    },
)
image = colormodel.sample_image()
colormodel.show(image)
