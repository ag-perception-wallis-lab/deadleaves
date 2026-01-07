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

# Distributions

Distributions are used in the Dead Leaves Model to sample object parameters such as size, aspect ratio, orientation, color, and texture.
They are specified as dictionaries with the distribution type as key and its parameters as values.

## Constant

The `Constant` distribution returns a fixed deterministic value every time it is sampled.
Use this when you want a parameter to remain unchanged.

```python
{
    "constant": {
        "value": <value>
    }
}
```

**Use case**: Fixed parameter for all leaves.

**Example: Constant size**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {
        "area": {"constant": {"value": 1000.0}}
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

## Uniform (from PyTorch)

The `Uniform` distribution samples values evenly from a range `[low, high]`.

$$
    f(x) = \begin{cases} \frac{1}{b-a}, & \text{for } x\in[a,b] \\ 0, &\text{else}.\end{cases}
$$

```python
{
    "uniform": {
        "low": <value>, 
        "high": <value>
    }
}
```

**Use case**: Random but equally likely values, e.g. random orientation or hue.

**Example: Uniform size and luminance**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {
        "area": {"uniform": {"low": 100.0, "high": 5000.0}}
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

## Normal (from PyTorch)

The `Normal` distribution samples values from a Gaussian (bell-shaped) distribution with mean `loc` and standard deviation `scale`.

$$
    f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)
$$

```python
{
    "normal": {
        "loc": <value>, 
        "scale": <value>
    }
}
```

**Use case**: Gaussian noise or color.

**Example: Normal size and luminance**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {
        "area": {"normal": {"loc": 5000.0, "scale": 2000.0}}
        },
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"normal": {"loc": 0.5, "scale": 0.2}}}
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## Beta (from PyTorch)

The `Beta` distribution samples values in the range `[0,1]` and is controlled by two concentration parameters.

$$
    f(x) = \begin{cases} \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, &\text{for } x\in[0,1] \\ 0, &\text{else.} \end{cases}
$$

```python
{
    "beta": {
        "concentration0": <value>, 
        "concentration1": <value>
    }
}
```

**Use case**: Random proportions or normalized parameters, e.g. aspect ratio or blending factors.

**Example: Beta aspect ratio**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage
import torch

model = DeadLeavesModel(
    shape = "ellipsoid", 
    param_distributions = {
        "area": {"constant": {"value": 1000.0}},
        "orientation": {"uniform": {"low": 0.0, "high": 2*torch.pi}},
        "aspect_ratio": {"beta": {"concentration0": 5, "concentration1": 13}}
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

## Poisson (from PyTorch)

The `Poisson` distribution generates integer counts based on a given rate.

$$
    P(k) = e^{-\lambda}\frac{\lambda^k}{k!}
$$

```python
{
    "poisson": {
        "rate": <value>
    }
}
```

**Use case**: Polygons with random number of vertices.

**Example: Polygons with Poisson vertices number**

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

## Powerlaw

The `PowerLaw` distribution is useful for heavy-tailed sizes, common in natural phenomena.

$$
f(x) = \begin{cases} \frac{k-1}{x_{\min}^{1-k}-x_{\max}^{1-k}}\cdot x^{-k}, & \text{for } x\in[x_{\min}, x_{\max}] \\ 0, & \text{else.}  \end{cases}
$$

```python
{
    "powerlaw": {
        "low": <value>, 
        "high": <value>, 
        "k": <value>
    }
}
```

**Use case**: Generating leaf sizes with many small and few large objects.

**Example: Powerlaw size and saturation**

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
    color_param_distributions = {
        "H": {"normal": {"loc": 0.6, "scale": 0.1}},
        "S": {"powerlaw": {"low": 0.2, "high": 1.0, "k": 3}},
        "V": {"normal": {"loc": 0.6, "scale": 0.1}}
        }
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## Cosine

The `Cosine` distribution produces periodic variations:

$$
f(x) = \begin{cases} \frac{1}{2\pi} \left(1+A\cdot\cos(F\cdot x)\right), & \text{for } x\in[-\pi,\pi] \\ 0, & \text{else.}  \end{cases}
$$

`frequency` has to be an integer and `amplitude` needs to be between `0.0` and `1.0`.

```python
{
    "cosine": {
        "amplitude": <value>, 
        "frequency": <value>
    }
}
```

**Use case**: Orientations of phase-like variations with periodic structure.

**Example: Cosine orientation**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "ellipsoid", 
    param_distributions = {
        "area": {"constant": {"value": 1000.0}},
        "orientation": {"cosine": {"amplitude": 0.5, "frequency": 4}},
        "aspect_ratio": {"constant": {"value": 0.5}}
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

## Expcosine

The `ExpCosine` distribution is a sharply peaked periodic distribution, useful for strongly directional parameters.

$$
    f(x) = \begin{cases} \frac{A \cdot \exp\left(-c \cdot \sqrt{1 - \cos(F \cdot x)}\right)}{\int_{-\pi}^{\pi} f(x) dx}, & \text{for } x\in[-\pi,\pi] \\ 0, & \text{else.}  \end{cases}
$$

All parameters must be positive with the `frequency` being an integer.

```python
{
    "expcosine": {
        "amplitude": <value>, 
        "frequency": <value>,
        "exponential_constant": <value>
    }
}
```

**Use case**: Leaf orientations with a strong preferred direction, e.g. cardinal bias.

**Example: Exponential cosine orientation**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "ellipsoid", 
    param_distributions = {
        "area": {"constant": {"value": 1000.0}},
        "orientation": {"expcosine": {"amplitude": 1.0, "frequency": 4, "exponential_constant": 3}},
        "aspect_ratio": {"constant": {"value": 0.5}}
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

## Image

The `Image` distribution samples from a set of image files in a given directory.
The class will discover all image type files in `dir` and uniformly sample images from the list.

```python
{
    "image": {
        "dir": <value>
    }
}
```

**Use case**: Assign color or texture by sampling existing images or texture patches, respectively.

**Example: Color and texture from images**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {
        "area": {"constant": {"value": 5000.0}}
        },
    size = (512,512)
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions={
        "source": {"image": {"dir": "/home/swantje/datasets/places365"}}
    },
    texture_param_distributions={
        "source": {"image": {"dir": "/home/swantje/datasets/brodatz"}},
        "alpha": {"normal": {"loc": 0.0, "scale": 0.4}},
    },
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```