# Shapes

The shape of the objects in the image is set by passing as string to the `shape` argument of the `DeadLeavesModel`.
We currently support the shapes `circular`, `ellipsoid`, `rectangular`, and `polygon`.

The distributions for the shape parameters a passed to the `DeadLeavesModel` through the dictionary `param_distributions`.
Depending on the chosen shape this dictionary needs to contain information for different parameters.

## Circles

The shape with the least number of parameters is the circle (`shape = 'circular'`).
A dead leaves model with circular objects only requires information about the distribution of the `area`.

```python
{
    "area": <distribution>
}
```



## Ellipsoids

```python
{
    "area": <distribution>, 
    "orientation": <distribution>, 
    "aspect_ratio": <distribution>
}
```

## Rectangles

```python
{
    "area": <distribution>, 
    "orientation": <distribution>, 
    "aspect_ratio": <distribution>
}
```

## Regular polygons

```python
{
    "area": <distribution>, 
    "n_vertices": <distribution>
}
```