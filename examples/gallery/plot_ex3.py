"""
RGB squares
===========================

Replication of Baradad et al., 2022
"""

from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    leaf_shape="rectangular",
    shape_param_distributions={
        "area": {"powerlaw": {"low": 500.0, "high": 10000.0, "k": 1.5}},
        "orientation": {"constant": {"value": 0.0}},
        "aspect_ratio": {"constant": {"value": 1}},
    },
    image_shape=(512, 731),
)
leaf_table, segmentation_map = model.generate_segmentation()

color_params = {
    "R": {"uniform": {"low": 0.0, "high": 1.0}},
    "G": {"uniform": {"low": 0.0, "high": 1.0}},
    "B": {"uniform": {"low": 0.0, "high": 1.0}},
}

colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(color_param_distributions=color_params)

renderer = ImageRenderer(
    leaf_table=colormodel.leaf_table, segmentation_map=segmentation_map
)
renderer.render_image()
renderer.show()
