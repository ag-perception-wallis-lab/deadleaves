"""
RGB squares
===========================

replication of Baradad et al., 2022
"""

from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    "rectangular",
    {
        "area": {"powerlaw": {"low": 500.0, "high": 10000.0, "k": 1.5}},
        "orientation": {"constant": {"value": 0.0}},
        "aspect_ratio": {"constant": {"value": 1}},
    },
    (512, 731),
)
leaves, partition = model.sample_partition()

color_params = {
    "R": {"uniform": {"low": 0.0, "high": 1.0}},
    "G": {"uniform": {"low": 0.0, "high": 1.0}},
    "B": {"uniform": {"low": 0.0, "high": 1.0}},
}

colormodel = DeadLeavesImage(leaves, partition, color_params)
image = colormodel.sample_image()
colormodel.show(image)
