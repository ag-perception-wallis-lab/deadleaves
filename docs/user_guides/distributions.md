# Distributions

## Normal

$\mu$ - `loc`  
$\sigma$ - `scale`

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

torch Normal

## Image

This distribution class will discover all image type files in the given directory `dir` and uniformly sample an image from the list.

```python
{
    "image": {
        "dir": <value>
    }
}
```


## Beta

torch Beta

```python
{
    "beta": {
        "concentration0": <value>, 
        "concentration1": <value>
    }
}
```

## Uniform

```python
{
    "uniform": {
        "low": <value>, 
        "high": <value>
    }
}
```

torch Uniform

## Poisson

torch Poisson

$$
    P(k) = e^{-\text{rate}}\frac{\text{rate}^k}{k!}
$$

```python
{
    "poisson": {
        "rate": <value>
    }
}
```

## Powerlaw

$$
f(x) = \begin{cases} \frac{k-1}{x_{\text{low}}^{1-k}-x_{\text{high}}^{1-k}}\cdot x^{-k}, & \text{for } x_{\text{low}} \leq x \leq x_{\text{high}}, \\ 0, & \text{else.}  \end{cases}
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

## Cosine

$$
f(x) = \begin{cases} \frac{1}{2\pi} \left(1+A\cdot\cos(F\cdot x)\right), & \text{for } -\pi \leq x \leq \pi, \\ 0, & \text{else.}  \end{cases}
$$

```python
{
    "cosine": {
        "amplitude": <value>, 
        "frequency": <value>
    }
}
```

## Expcosine

```python
{
    "expcosine": {
        "amplitude": <value>, 
        "frequency": <value>,
        "exponential_constant": <value>
    }
}
```

## Constant

The distribution class `Constant` is a dummy class that return a fixed deterministic value every time when sampling. 
It therefore only as one input parameter which is the value. 

```python
{
    "constant": {
        "value": <value>
    }
}
```