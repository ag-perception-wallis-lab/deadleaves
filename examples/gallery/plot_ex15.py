"""
Gray Mondrian
===========================

Inspired by MaxEvoy and Paradiso (2001)
"""

from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    leaf_shape="rectangular",
    shape_param_distributions={
        "area": {"uniform": {"low": 10000.0, "high": 50000.0}},
        "aspect_ratio": {"constant": {"value": 1}},
        "orientation": {"constant": {"value": 0}},
    },
    image_shape=(512, 731),
)
leaf_table, segmentation_map = model.generate_segmentation()
colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(
    color_param_distributions={"gray": {"uniform": {"low": 0.1, "high": 0.9}}}
)

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show()
