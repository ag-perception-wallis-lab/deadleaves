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

## Cosine

The `Cosine` distribution produces periodic variations:

$$
f(x) = \begin{cases} \frac{1}{2\pi} \left(1+A\cdot\cos(F\cdot x)\right), & \text{for } x\in[-\pi,\pi] \\ 0, & \text{else.}  \end{cases}
$$

```python
{
    "cosine": {
        "amplitude": <value>, 
        "frequency": <value>
    }
}
```

**Use case**: Orientations of phase-like variations with periodic structure.

## Expcosine

The `ExpCosine` distribution is a sharply peaked periodic distribution, useful for strongly directional parameters.

$$
    f(x) = \begin{cases} \frac{A \cdot \exp\left(-c \cdot \sqrt{1 - \cos(F \cdot x)}\right)}{\int_{-\pi}^{\pi} f(x) dx}, & \text{for } x\in[-\pi,\pi] \\ 0, & \text{else.}  \end{cases}
$$

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
