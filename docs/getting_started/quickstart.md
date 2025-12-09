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

# Quickstart

This package generates synthetic images using the Dead Leaves Model.
A scene is formed by repeatedly layering random shapes (*leaves*) behind each other until the image is filled.
The workflow has two steps:
1. The `DeadLeavesModel` defines the geometry of the leaves, e.g. shape, size, orientation.
2. The `DeadLeavesImage` defines how leaves are rendered, i.e. color and texture specifications.

This Quickstart-Guide will run you through the basic usage of the package.

## Import

```{code-cell}
from dead_leaves import DeadLeavesModel, DeadLeavesImage
```

## Setting up a Model

```{code-cell}
model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (256,256)
)
```

```{code-cell}
leaves, partition = model.sample_partition()
```

Each shape has required parameters for each of which you supply a distribution.

## Using distributions

The package includes common sampling distributions:

| **Name** | **Example use** | **Notes** |
|--|--|--|
| `"uniform"` |  | define `low`, `high` |
| `"normal"` |  | define `loc`, `scale` |

## Render an image

`DeadLeavesImage` turns a geometric partition into an actual image.
You specify color mode and color distributions.

```{code-cell}
colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"normal": {"loc": 0.5, "scale": 0.2}}}
    )
image = colormodel.sample_image()
colormodel.show(image)
```

## Adding texture

Textures are parameterized similarly to color

TODO add example

## Full example

## Next steps

See for all shape parameters
See for details on color
See for details on texture
Browse examples in the Gallery (sphinx-gallery)
See for all distributions