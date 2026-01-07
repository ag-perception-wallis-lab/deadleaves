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

# Colors

In the Dead Leaves Model, the color of each object can be sampled from different types of distributions.
The colors are specified via the `color_param_distribution` dictionary when creating a `DeadLeavesImage`.
Different color spaces or sources are supported, as described below.

## Gray-scale

Gray-scale leaves are defined using a single channel, `"gray"`.
You can specify any supported [distribution](distributions.md) for the gray values. For example, a normal distribution centered at 0.5:

```python
{
    "gray": <distribution>
}
```
- **Range**: 0 (black) to 1 (white), values sampled outside this range are clipped.
- **Use case**: Simplest and common setup for monochromatic images.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {
        "gray": {"uniform": {"low": 0.0, "high": 1.0}}
        }
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## RGB

RGB colors are defined using three channels: `"R"`, `"G"`, and `"B"`.
Each channel has its own distribution, allowing for fully random or correlated color sampling:

```python
{
    "R": <distribution>, 
    "G": <distribution>, 
    "B": <distribution>
}
```

- **Range for each channel**: 0 to 1
- **Use case**: Full-color images where red, green, and blue components are sampled independently or according to specific distributions.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {
        "R": {"uniform": {"low": 0.0, "high": 1.0}},
        "G": {"uniform": {"low": 0.0, "high": 1.0}},
        "B": {"uniform": {"low": 0.0, "high": 1.0}}
        }
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## HSV

HSV is an alternative color space that can be useful for separating hue from saturation and luminance:

```python
{
    "H": <distribution>, 
    "S": <distribution>, 
    "V": <distribution>
}
```

- **Range for each channel**: 0 to 1
- **Use case**: Easily sample colors with controlled hue.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {
        "H": {"normal": {"loc": 0.5, "scale": 0.1}},
        "S": {"normal": {"loc": 0.5, "scale": 0.1}},
        "V": {"normal": {"loc": 0.5, "scale": 0.1}}
        }
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## From Image

You can also sample colors from an existing image. This allows the Dead Leaves to imitate real color distributions from a source:

```python
{
    "source": {"image": {"dir": <value>}}
}
```

- `dir`: Path to the folder of images to sample from
- **Use case**: Generate synthetic images that match the palette of a real image.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {
        "source": {"image": {"dir": "/home/swantje/datasets/places365"}},
        }
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```