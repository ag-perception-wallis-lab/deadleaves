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

The shape of the objects in a Dead Leaves image are specified via the `leaf_shape`argument of the `LeafGeometryGenerator`.
Currently support shapes are:
- `circular`
- `ellipsoid`
- `rectangular`
- `polygon`

Shape-specific parameters are passed through the `shape_param_distributions`dictionary.
The required parameters depend on the chosen shape.

```{note}
For comparability all shape sizes are parameterized via an `area` parameter, which describes the distribution of the leaves size in terms of number of pixels (i.e. squared pixels).
```

## Circles

Circular leaves (`leaf_shape = "circular"`) are the simplest, requiring only the `area` distribution:

```python
{
    "area": <distribution>
}
```

The leaf mask for a circular leaf with position $(\bar{x},\bar{y})$ and area $A$ is then generated via

$$
    L(x,y) = \begin{cases} 
    1, & \text{if } \sqrt{(x-\bar{x})^2 + (y-\bar{y})^2} \leq \sqrt{\frac{A}{\pi}} \\
    0, & \text{else.}
    \end{cases}
$$

```{tip}
To generate images with powerlaw distributed leaf radius (for a $1/f$ power spectrum) simply use a powerlaw distributed area with exponent `k` half of the value you would use for the radius.
```

**Example**

```{code-cell}
:tags: [hide-input]
from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    (256, 256)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Ellipsoids

Ellipsoidal leaves (`leaf_shape = "ellipsoid"`) require distributions for
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

The leaf mask for a ellipsoidal leaf with position $(\bar{x},\bar{y})$, area $A$, aspect ratio $\rho$, and orientation $\phi$ is then generated via

$$
    a = \sqrt{A \cdot \frac{\rho}{\pi}} \qquad
    b = \sqrt{\frac{A}{\pi \cdot \rho}} \\
    u = (x-\bar{x}) \cdot \cos(\phi) - (y-\bar{y}) \cdot \sin(\phi) \\
    v = (x-\bar{x}) \cdot \sin(\phi) + (y-\bar{y}) \cdot \cos(\phi)\\
    L(x,y) = \begin{cases} 
    1, & \text{if } \sqrt{\left(\frac{u}{a}\right)^2 + \left(\frac{v}{b}\right)^2} \leq 1 \\
    0, & \text{else.}
    \end{cases}
$$

**Example**

```{code-cell}
:tags: [hide-input]
from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
import torch


model = LeafGeometryGenerator(
    "ellipsoid", 
    {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
        "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
        "aspect_ratio": {"uniform": {"low": 0.5, "high": 2}}
        },
    (256, 256)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Rectangles

Rectangular leaves (`leaf_shape = "rectangular"`) use the same parameters as ellipsoids:

```python
{
    "area": <distribution>, 
    "orientation": <distribution>, 
    "aspect_ratio": <distribution>
}
```

The leaf mask for a rectangular leaf with position $(\bar{x},\bar{y})$, area $A$, aspect ratio $R$, and orientation $\phi$ is then generated via

$$
    h = \sqrt{\frac{A}{\rho}} \qquad
    w = h \cdot \rho \\
    u = (x-\bar{x})\cdot \cos(\phi) - (y-\bar{y})\cdot \sin(\phi) \\
    v = (x-\bar{x})\cdot \sin(\phi) - (y-\bar{y})\cdot \cos(\phi) \\
    L(x,y) = \begin{cases} 
    1, & \text{if } \vert u \vert \leq \frac{w}{2} \text{ and } \vert v \vert \leq \frac{h}{2} \\
    0, & \text{else.}
    \end{cases}
$$

**Example**

```{code-cell}
:tags: [hide-input]
from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
import torch

model = LeafGeometryGenerator(
    "rectangular", 
    {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
        "orientation": {"uniform": {"low": 0.0, "high": 2 * torch.pi}},
        "aspect_ratio": {"uniform": {"low": 0.5, "high": 2}}
        },
    (256, 256)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Regular polygons

Currently only regular polygons with fixed orientation are supported (`leaf_shape = "polygon"`).
The parameters are `area` and number of vertices `n_vertices`:

```python
{
    "area": <distribution>, 
    "n_vertices": <distribution>
}
```

The leaf mask for a regular polygon leaf with position $(\bar{x},\bar{y})$, area $A$, and number of vertices $n$ is then generated by computing the positions of the vertices via

$$  
    r = \sqrt{2\cdot \frac{A}{n\cdot \sin(2\cdot\frac{\pi}{n})}} \\
    \psi_k = \frac{2\pi k}{n} \qquad
    v_k = \begin{pmatrix}\bar{x} + r\cdot \cos(\psi_k) \\ \bar{y} + r\cdot \sin(\psi_k) \end{pmatrix}.
$$

We then use a simple [Ray-casting algorithm](https://rosettacode.org/wiki/Ray-casting_algorithm#) to check if a pixel $(x,y)$ is with in the convex hull of the vertices, i.e. the polygon.

**Example**

```{code-cell}
:tags: [hide-input]
from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "polygon", 
    {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
        "n_vertices": {"poisson": {"rate": 5}},
        },
    (256, 256)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```