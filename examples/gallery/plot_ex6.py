"""
Natural colors and textures
===========================

Replication of Madhusudana et al., 2022
"""

from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    leaf_shape="circular",
    shape_param_distributions={
        "area": {"powerlaw": {"low": 1000.0, "high": 10000.0, "k": 1.5}},
    },
    image_shape=(512, 731),
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table=leaf_table)
colormodel.sample_color(
    color_param_distributions={"source": {"image": {"dir": "../../examples/images"}}}
)
colormodel.sample_texture(
    texture_param_distributions={
        "source": {"image": {"dir": "../../examples/textures/brodatz"}},
        "alpha": {"normal": {"loc": 0.0, "scale": 0.4}},
    }
)

renderer = ImageRenderer(
    leaf_table=colormodel.leaf_table, segmentation_map=segmentation_map
)
renderer.render_image()
renderer.show()
