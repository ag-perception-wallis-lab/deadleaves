"""
Gray rectangles
===========================
"""

from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    leaf_shape="rectangular",
    shape_param_distributions={
        "area": {"powerlaw": {"low": 1000.0, "high": 10000.0, "k": 1.5}},
        "aspect_ratio": {"uniform": {"low": 0.001, "high": 10}},
        "orientation": {"constant": {"value": 0.0}},
    },
    image_shape=(512, 731),
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(
    color_param_distributions={"gray": {"uniform": {"low": 0.0, "high": 1.0}}}
)

renderer = ImageRenderer(
    leaf_table=colormodel.leaf_table, segmentation_map=segmentation_map
)
renderer.render_image()
renderer.show()
