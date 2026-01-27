"""
3D Texture
===========================

replication of Groen et al., 2012
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
    color_param_distributions={"gray": {"constant": {"value": 0.0}}},
    texture_param_distributions={
        "source": {
            "image": {"dir": "../../examples/textures/sphere"}
        },  # this folder only contains a single texture file which will be used for all leaves
        "alpha": {"constant": {"value": 1.0}},
    },
)
image = colormodel.sample_image()
colormodel.show(image)
