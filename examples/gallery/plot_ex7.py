"""
HSV color and leaf-wise noise
===========================
"""

from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    leaf_shape="circular",
    shape_param_distributions={
        "area": {"powerlaw": {"low": 100.0, "high": 5000.0, "k": 1.5}}
    },
    image_shape=(512, 731),
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(
    color_param_distributions={
        "H": {"normal": {"loc": 0.6, "scale": 0.05}},
        "S": {"normal": {"loc": 0.5, "scale": 0.1}},
        "V": {"normal": {"loc": 0.5, "scale": 0.2}},
    }
)
colormodel.sample_texture(
    texture_param_distributions={
        "gray": {"normal": {"loc": 0, "scale": {"uniform": {"low": 0.01, "high": 0.1}}}}
    }
)

renderer = ImageRenderer(
    leaf_table=colormodel.leaf_table, segmentation_map=segmentation_map
)
image = renderer.render_image()
renderer.show(image)
