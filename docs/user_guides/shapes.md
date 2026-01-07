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

# Shapes

The shape of the objects in a Dead Leaves image are specified via the `shape`argument of the `DeadLeavesModel`.
Currently support shapes are:
- `circular`
- `ellipsoid`
- `rectangular`
- `polygon`

Shape-specific parameters are passed through the `param_distributions`dictionary.
The required parameters depend on the chosen shape.

## Circles

Circular leaves (`shape = "circular"`) are the simplest, requiring only the `area` distribution:

```python
{
    "area": <distribution>
}
```

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}
        },
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"uniform": {"low": 0.0, "high": 1.0}}}
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## Ellipsoids

Ellipsoidal leaves (`shape = "ellipsoid"`) require distributions for
- `area`: size of the ellipse
- `aspect_ratio`: ratio of minor to major axis
- `orientation`: rotation angle.

```python
{
    "area": <distribution>, 
    "orientation": <distribution>, 
    "aspect_ratio": <distribution>
}
```

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage
import torch

model = DeadLeavesModel(
    shape = "ellipsoid", 
    param_distributions = {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
        "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
        "aspect_ratio": {"uniform": {"low": 0.5, "high": 2}}
        },
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"uniform": {"low": 0.0, "high": 1.0}}}
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## Rectangles

Rectangular leaves (`shape = "rectangular"`) use the same parameters as ellipsoids:

```python
{
    "area": <distribution>, 
    "orientation": <distribution>, 
    "aspect_ratio": <distribution>
}
```

**Example**

```{code-cell}
:tags: [hide-input]
:class: output-scale-50
from dead_leaves import DeadLeavesModel, DeadLeavesImage
import torch

model = DeadLeavesModel(
    shape = "rectangular", 
    param_distributions = {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
        "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
        "aspect_ratio": {"uniform": {"low": 0.5, "high": 2}}
        },
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"uniform": {"low": 0.0, "high": 1.0}}}
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## Regular polygons

Currently only regular polygons with fixed orientation are supported (`shape = "polygon"`).
The parameters are `area` and number of vertices `n_vertices`:

```python
{
    "area": <distribution>, 
    "n_vertices": <distribution>
}
```

**Example**

```{code-cell}
:tags: [hide-input]
:class: output-scale-50
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "polygon", 
    param_distributions = {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
        "n_vertices": {"poisson": {"rate": 5}},
        },
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"uniform": {"low": 0.0, "high": 1.0}}}
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```