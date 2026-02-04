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

**Use case**: Fixed parameter for all leaves.

```python
{
    "constant": {
        "value": <value>
    }
}
```

**Example: Constant size**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"constant": {"value": 5000.0}}},
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Uniform (from PyTorch)

The `Uniform` distribution samples values evenly from a range `[low, high]` (where $a=$`low` and $b=$`high`).

$$
    f(x) = \begin{cases} \frac{1}{b-a}, & \text{for } x\in[a,b] \\ 0, &\text{else}.\end{cases}
$$

**Use case**: Random but equally likely values, e.g. random orientation or hue.

```python
{
    "uniform": {
        "low": <value>, 
        "high": <value>
    }
}
```

**Example: Uniform size and luminance**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"uniform": {"low": 100.0, "high": 10000.0}}},
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Normal (from PyTorch)

The `Normal` distribution samples values from a Gaussian (bell-shaped) distribution with mean $\mu=$`loc` and standard deviation $\sigma=$`scale`.

$$
    f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)
$$

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import torch

params = [
    (0,1),
    (0.5,0.2),
    (1,0.5),
    (0.5,0.1)
]

fig, ax = plt.subplots(figsize=(5,3))

for loc, scale in params:
    x = torch.linspace(-2,2, 200)
    dist = torch.distributions.normal.Normal(loc=loc,scale=scale)

    ax.plot(x,torch.exp(dist.log_prob(x)), label=fr"$\mu = {loc}$, $\sigma = {scale}$")
ax.legend()
ax.set_xlabel('Value')
ax.set_ylabel('Probability density')
plt.show()
```

**Use case**: Gaussian noise or color.

```python
{
    "normal": {
        "loc": <value>, 
        "scale": <value>
    }
}
```

**Example: Normal size and luminance**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"normal": {"loc": 10000.0, "scale": 2000.0}}},
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"normal": {"loc": 0.5, "scale": 0.25}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Beta (from PyTorch)

The `Beta` distribution samples values in the range `[0,1]` and is controlled by two concentration parameters $\alpha=$`concentration0` and $\beta=$`concentration1`.

$$
    f(x) = \begin{cases} \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, &\text{for } x\in[0,1] \\ 0, &\text{else.} \end{cases}
$$

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import torch

params = [
    (2,5),
    (4,2),
    (1,10),
    (5,10)
]

fig, ax = plt.subplots(figsize=(5,3))

for alpha, beta in params:
    x = torch.linspace(0,1, 200)
    dist = torch.distributions.beta.Beta(alpha, beta)

    ax.plot(x,torch.exp(dist.log_prob(x)), label=fr"$\alpha$ = {alpha}, $\beta$ = {beta}")
ax.legend()
ax.set_xlabel('Value')
ax.set_ylabel('Probability density')
plt.show()
```

**Use case**: Random proportions or normalized parameters, e.g. aspect ratio or blending factors.

```python
{
    "beta": {
        "concentration0": <value>, 
        "concentration1": <value>
    }
}
```

**Example: Beta aspect ratio**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
import torch

model = LeafGeometryGenerator(
    "ellipsoid", 
    {
        "area": {"constant": {"value": 5000.0}},
        "orientation": {"uniform": {"low": 0.0, "high": 2*torch.pi}},
        "aspect_ratio": {"beta": {"concentration0": 5, "concentration1": 13}}
        },
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table,segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Poisson (from PyTorch)

The `Poisson` distribution generates integer counts based on a given `rate` $\lambda$.

$$
    P(k) = e^{-\lambda}\frac{\lambda^k}{k!}
$$

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import torch

params = [
    3,5,10,20
]

fig, ax = plt.subplots(figsize=(5,3))

for rate in params:
    x = torch.arange(0,50)
    dist = torch.distributions.poisson.Poisson(rate)

    ax.scatter(x,torch.exp(dist.log_prob(x)), label=fr"$\lambda$ = {rate}", s=2)
ax.legend()
ax.set_xlabel('Value')
ax.set_ylabel('Probability')
plt.show()
```

**Use case**: Polygons with random number of vertices.

```python
{
    "poisson": {
        "rate": <value>
    }
}
```

**Example: Polygons with Poisson vertices number**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "polygon", 
    {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
        "n_vertices": {"poisson": {"rate": 5}},
        },
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Powerlaw

The `PowerLaw` distribution is useful for heavy-tailed sizes, common in natural phenomena.
It is parameterized via a lower (`low`, $x_\min$) and an upper bound (`high`, $x_\max$) and the exponent $k$.

$$
f(x) = \begin{cases} \frac{k-1}{x_{\min}^{1-k}-x_{\max}^{1-k}}\cdot x^{-k}, & \text{for } x\in[x_{\min}, x_{\max}] \\ 0, & \text{else.}  \end{cases}
$$

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import torch
from dead_leaves.distributions import PowerLaw

params = [
    (10,150,1.5),
    (10,150,3),
    (20,150,1.5),
    (20,300,3)
]

fig, ax = plt.subplots(ncols=2, figsize=(10,3))

for low, high, k in params:
    x = torch.arange(low,high)
    dist = PowerLaw(low,high,k)

    ax[0].plot(x,dist.pdf(x), label=fr"$x_\min = {low}$, $x_\max = {high}$, $k = {k}$")
    ax[1].plot(x,dist.pdf(x), label=fr"$x_\min = {low}$, $x_\max = {high}$, $k = {k}$")
ax[0].legend()
ax[0].set_xlabel('Value')
ax[0].set_ylabel('Probability density')
ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_xlabel('Value')
ax[1].set_ylabel('Probability density')
plt.show()
```

**Use case**: Generating leaf sizes with many small and few large objects.

```python
{
    "powerlaw": {
        "low": <value>, 
        "high": <value>, 
        "k": <value>
    }
}
```

**Example: Powerlaw size and saturation**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "circular", 
    {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}
        },
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color(
    {
        "H": {"normal": {"loc": 0.6, "scale": 0.1}},
        "S": {"powerlaw": {"low": 0.2, "high": 1.0, "k": 3}},
        "V": {"normal": {"loc": 0.6, "scale": 0.1}}
        }
)

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Cosine

The `Cosine` distribution produces periodic variations for values between $-\pi$ and $\pi$:

$$
f(x) = \begin{cases} \frac{1}{2\pi} \left(1+A\cdot\cos(F\cdot x)\right), & \text{for } x\in[-\pi,\pi] \\ 0, & \text{else.}  \end{cases}
$$

The `frequency`, $F$ has to be an integer and specifies the number of phases on the value range.
The `amplitude`, $A$ changes the intensity between `0.0` and `1.0`.

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import torch
from dead_leaves.distributions import Cosine

params = [
    (1,4),
    (0.5,4),
    (1,2),
    (0.5,2)
]

fig, ax = plt.subplots(figsize=(5,3))

for amplitude, frequency in params:
    x = torch.linspace(-torch.pi,torch.pi, steps=200)
    dist = Cosine(amplitude,frequency)

    ax.plot(x,dist.pdf(x), label=fr"$A = {amplitude}$, $F = {frequency}$")
ax.legend()
ax.set_xlabel('Value')
ax.set_ylabel('Probability density')
plt.show()
```

**Use case**: Orientations of phase-like variations with periodic structure.

```python
{
    "cosine": {
        "amplitude": <value>, 
        "frequency": <value>
    }
}
```

```{Note}
Random variables with this distribution always produce values between $-\pi$ and $\pi$ and are mainly useful to parameterized orientation distributions.
```

**Example: Cosine orientation**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "ellipsoid", 
    {
        "area": {"constant": {"value": 1000.0}},
        "orientation": {"cosine": {"amplitude": 0.5, "frequency": 4}},
        "aspect_ratio": {"constant": {"value": 0.5}}
        },
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Expcosine

The `ExpCosine` distribution is a sharply peaked periodic distribution, useful for strongly directional parameters.

$$
    f(x) = \begin{cases} \frac{\exp\left(-c \cdot \sqrt{1 - \cos(F \cdot x)}\right)}{\int_{-\pi}^{\pi} f(x) dx}, & \text{for } x\in[-\pi,\pi] \\ 0, & \text{else.}  \end{cases}
$$

The `frequency`, $F$ has to be an integer and specifies the number of periodic peaks on the value range.
The positive `exponential_constant`, $c$ sets the strength of the peak.

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
import torch
from dead_leaves.distributions import ExpCosine

params = [
    (4, 3),
    (4, 1),
    (2, 3),
    (2, 1)
]

fig, ax = plt.subplots(figsize=(5,3))

for frequency, exponential_constant in params:
    x = torch.linspace(-torch.pi,torch.pi, steps=200)
    dist = ExpCosine(frequency, exponential_constant)

    ax.plot(x,dist.pdf(x), label=fr"$F = {frequency}$, $c = {exponential_constant}$")
ax.legend()
ax.set_xlabel('Value')
ax.set_ylabel('Probability density')
plt.show()
```

**Use case**: Leaf orientations with a strong preferred direction, e.g. cardinal bias.

```python
{
    "expcosine": {
        "frequency": <value>,
        "exponential_constant": <value>
    }
}
```

```{Note}
Random variables with this distribution always produce values between $-\pi$ and $\pi$ and are mainly useful to parameterized orientation distributions.
```

**Example: Exponential cosine orientation**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "ellipsoid", 
    {
        "area": {"constant": {"value": 1000.0}},
        "orientation": {"expcosine": {"frequency": 4, "exponential_constant": 3}},
        "aspect_ratio": {"constant": {"value": 0.5}}
        },
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```

## Image

The `Image` distribution samples from a set of image files in a given directory.
The class will discover all image type files in `dir` and uniformly sample images from the list.

**Use case**: Assign color or texture by sampling existing images or texture patches, respectively.

```python
{
    "image": {
        "dir": <value>
    }
}
```

```{Note}
Sampling from this distribution will return one or multiple paths to image(s) in `dir`.
In particular, it will **sample from all available image files** in the directory provided.
```

**Example: Color and texture from images**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "circular", 
    {
        "area": {"constant": {"value": 5000.0}}
        },
    (512,512)
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({
        "source": {"image": {"dir": "../../examples/images"}}
    })
colormodel.sample_texture({
        "source": {"image": {"dir": "../../examples/textures/brodatz"}},
        "alpha": {"normal": {"loc": 0.0, "scale": 0.4}},
    })

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show(figsize = (3,3))
```