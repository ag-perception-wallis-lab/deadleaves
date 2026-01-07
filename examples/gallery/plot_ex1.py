"""
Example 1
===========================
"""

from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape="circular",
    param_distributions={
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}
    },
    size=(512, 512),
)
leaves, partition = model.sample_partition()
colormodel = DeadLeavesImage(
    leaves=leaves,
    partition=partition,
    color_param_distributions={"gray": {"normal": {"loc": 0.5, "scale": 0.2}}},
    texture_param_distributions={"gray": {"normal": {"loc": 0, "scale": 0.05}}},
)
image = colormodel.sample_image()
colormodel.show(image)

colormodel.save(image, "images/sphx_glr_plot_ex1_001.png")
colormodel.save(image, "images/thumb/sphx_glr_plot_ex1_thumb.png")
