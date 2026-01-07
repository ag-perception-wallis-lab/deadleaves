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

This package generates synthetic images using the Dead Leaves Model, a process in which random shapes (*leaves*) are repeatedly layered until the canvas is fully covered.

The workflow consists of two steps:

1. **Geometry**: A `DeadLeavesModel` defines the geometry of the leaves, e.g. shape, size, orientation.
2. **Rendering**: The `DeadLeavesImage` defines how leaves are rendered, i.e. color and texture specifications.

This Quickstart walks you through the basic usage.

## Import

Begin by importing the two core classes:

```{code-cell}
from dead_leaves import DeadLeavesModel, DeadLeavesImage
```

## Setting up a Model

The `DeadLeavesModel` controls only the **geometry** of the leaves and the **canvas size**.
You choose a leaf shape (circles, ellipses, rectangles, or regular polygons) and assign sampling distributions to that shape’s parameters.
Leaf positions are sampled uniformly on the canvas.

For example, circles require only one parameter: `area`.
Here we use a power-law distribution for the area:

```{code-cell}
model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512)
)
```

To generate a sample, use `.sample_partition()`.
This returns:
- a **pandas DataFrame** describing all generated leaves (location + shape parameters), and
- a **partition**, i.e. a segmentation map assigning each pixel to the leaf on top at that location.

```{code-cell}
leaves, partition = model.sample_partition()
```

Each shape type has required parameters, and each of those parameters must be assigned a sampling distribution.

## Using distributions

You can specify parameter distributions using dictionaries.
The package includes common PyTorch distributions and several custom ones:

| **Name** | **Required keys** | **Notes** |
|--|--|--|
| `"uniform"` | `low`, `high` | |
| `"normal"` | `loc`, `scale` | |
| `"beta"` | `concentration0`, `concentration1` | |
| `"poisson"` | `rate` | |
| `"powerlaw"` | `low`, `high`, `k` | |
| `"cosine"` | `amplitude`, `frequency` | |
| `"expcosine"` | `frequency`, `exponential_constant` | |
| `"image"` | `dir` | |
| `"constant"` | `value` | deterministic |

Each parameter is defined by a dictionary of the form:

```
{"distribution_name": { ... parameters ...}}
```

## Render an image

`DeadLeavesImage` converts a geometric partition into an actual image.
For this second stage, you specify the color mode and the color distributions.

```{code-cell}
colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"normal": {"loc": 0.5, "scale": 0.2}}}
    )
```

You then sample an image using `.sample_image()`:

```{code-cell}
image = colormodel.sample_image()
colormodel.show(image)
```

## Adding texture

You can optionally add texture on top of the leaf colors.
The simplest example uses pixel-wise Gaussian noise:

```{code-cell}
colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"normal": {"loc": 0.5, "scale": 0.2}}},
    texture_param_distributions = {"gray": {"normal": {"loc": 0, "scale": 0.05}}}
    )
image = colormodel.sample_image()
colormodel.show(image)
```

```{note}
This color model reuses the same geometric partition as above; only the sampled leaf colors and textures change.
```

## Full example

Putting everything together, a complete workflow looks like this:

```{code-cell}
# Define geometric model
model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512)
)

# Sample partition of canvas
leaves, partition = model.sample_partition()

# Define color and texture model
colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"normal": {"loc": 0.5, "scale": 0.2}}},
    texture_param_distributions = {"gray": {"normal": {"loc": 0, "scale": 0.05}}}
    )

# Sample colors and textures
image = colormodel.sample_image()

# Display the result
colormodel.show(image)
```

## Next steps

See the documentation on [shape parameters](../user_guides/shapes.md).  
Learn more about [color models](../user_guides/colors.md).  
Learn more about [texture models](../user_guides/textures.md).  
Browse examples in the Gallery (sphinx-gallery) for example scripts.  
Consult the full list of available [distributions](../user_guides/distributions.md).  