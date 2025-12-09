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

## Rectangles

Rectangular leaves (`shape = "rectangular"`) use the same parameters as ellipsoids:

```python
{
    "area": <distribution>, 
    "orientation": <distribution>, 
    "aspect_ratio": <distribution>
}
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