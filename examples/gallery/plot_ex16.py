"""
Color Mondrian
===========================

Inspired by Land (1985) and Barbur et al. (2004)
"""

from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    leaf_shape="rectangular",
    shape_param_distributions={
        "area": {"uniform": {"low": 50000.0, "high": 100000.0}},
        "aspect_ratio": {"uniform": {"low": 0.1, "high": 2}},
        "orientation": {"constant": {"value": 0}},
    },
    image_shape=(512, 731),
)
leaf_table, segmentation_map = model.generate_segmentation()
colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(
    color_param_distributions={
        "H": {"uniform": {"low": 0.0, "high": 1.0}},
        "S": {"constant": {"value": 0.6}},
        "V": {"constant": {"value": 0.8}},
    }
)

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show()
