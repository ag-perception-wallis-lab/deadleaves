# Dependencies

You can define dependencies between leaf features, allowing one parameter to influence another.
For example, you may want larger leaves to have different color distributions than smaller leaves, or size and orientation to be correlated.
Dependencies are specified when defining parameter distributions by replacing the parameter value with a dictionary defining the dependency:

```python
    {"from": <value>, "fn": <callable>}
```

## Sampling Order and Valid Dependencies

Dependencies must respect the internal sampling order of the model:
- Color can depend on geometry, but geometry cannot depend on color.
- Shape parameters may depend on one another.
- Cyclic dependencies are not allowed.

## Defining Dependency Functions

### Single-Feature Dependencies

If a parameter depends on a single feature, the value of `"from"` is  that feature name, and the function `fn` receives a single value, returning the dependent parameter.

```python
    {"from": "x_pos", "fn": lambda x: x * 0.01}
```

### Multi-Feature Dependencies

If a parameter depends on multiple features, `"from"` must be a list of feature names.
In this case, `fn` receives a dictionary mapping feature names to their sampled values:

```python
{
    "from": ["x_pos","y_pos"],
    "fn": lambda d: (d["x_pos"]**2 + d["y_pos"]**2)**0.5
}
```

This enables defining complex dependencies between spatial, geometric, and visual properties.