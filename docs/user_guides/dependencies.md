---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Dependencies

You can define dependencies between leaf features, allowing one parameter to influence another.
For example, you may want larger leaves to have different color distributions than smaller leaves, or size and orientation to be correlated.
Dependencies are specified when defining parameter distributions by replacing the parameter value with a dictionary defining the dependency:

```python
    {"from": <value>, "fn": <callable>}
```

## Sampling Order and Valid Dependencies

Dependencies must respect the internal sampling order of the model:
- Color can depend on geometry, but geometry cannot depend on color.
- Shape parameters may depend on one another.
- Cyclic dependencies are not allowed.

## Defining Dependency Functions

### Single-Feature Dependencies

If a parameter depends on a single feature, the value of `"from"` is  that feature name, and the function `fn` receives a single value, returning the dependent parameter.

```python
    {"from": "x_pos", "fn": lambda x: x * 0.01}
```

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"powerlaw": {"low": 100.0, "high": 5000.0, "k": 1.5}}},
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color(
    {
        "H": {"normal": {
            "loc": {"from": "x_pos", "fn": lambda x: 1/512*x * 0.3 + (1-1/512*x) * 0.6}, 
            "scale": 0.05
        }},
        "S": {"normal": {"loc": 0.6, "scale": 0.1}},
        "V": {"normal": {"loc": 0.6, "scale": 0.1}}
        }
)

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

### Multi-Feature Dependencies

If a parameter depends on multiple features, `"from"` must be a list of feature names.
In this case, `fn` receives a dictionary mapping feature names to their sampled values:

```python
{
    "from": ["x_pos","y_pos"],
    "fn": lambda d: (d["x_pos"]**2 + d["y_pos"]**2)**0.5
}
```

This enables defining complex dependencies between spatial, geometric, and visual properties.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
import torch

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"powerlaw": {"low": 100.0, "high": 5000.0, "k": 1.5}}},
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

def fn(d):
    distance_from_center = torch.sqrt(torch.tensor((256 - d["x_pos"]) ** 2 + (256 - d["y_pos"]) ** 2))
    return torch.where(distance_from_center <= 128, 0.5, 0.8)

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color(
    {
        "H": {"normal": {
            "loc": {"from": ["x_pos","y_pos"], "fn": fn}, 
            "scale": 0.05
        }},
        "S": {"normal": {"loc": 0.6, "scale": 0.1}},
        "V": {"normal": {"loc": 0.6, "scale": 0.1}}
        }
)

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```