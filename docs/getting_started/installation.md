# Installation

## Installing

There are several ways to install `deadleaves`.
For most usecases we recommend installing from PyPI using `pip`.

To adapt or contribute code, you will have to get the source code from GitHub.

:::::{tab-set}

::::{tab-item} pip {fab}`python`
You may install `deadleaves` from PyPI using `pip`:

```python
pip install deadleaves
```

:::{admonition} Install a different version
:class: dropdown

`pip` by default install the latest version of the package.
To install a different version, simply specify the version number, either an exact version:
```python
pip install "deadleaves==0.1.1"
```
or a conditional version:
```python
pip install "deadleaves<=1.0.0"
```
(for any version before `1.0.0`).

:::
::::


::::{tab-item} source {fab}`github`

1. Clone the repository from GitHub:

    ```bash
    git clone git@github.com/ag-perception-wallis-lab/deadleaves.git
    ```

2. Install `deadleaves` to your local python library using pip, by running from the top-level directory:

    ```python
    pip install .
    ```

:::{admonition} For developers
:class: dropdown

```python
pip install -e .dev
```

for an editable install (`-e`) which makes changes to files immediately usable,
rather than having to reinstall the package after every change;
and to install the development dependencies.
:::
::::

:::::

## Dependencies

We recommend using a Python version `3.12` or newer.
The dependencies should be automatically installed (at least using `pip`). `deadleaves`s required dependencies are: 

- [PyTorch](https://pytorch.org/)
- [Torchquad](https://torchquad.readthedocs.io/en/main/)
- [Torchvision](https://docs.pytorch.org/vision/stable/index.html)
- [Pandas](https://pandas.pydata.org/)

```{note}
If possible we advise to setup [CUDA](https://developer.nvidia.com/cuda-downloads) for GPU use. On CPU sampling a single high-resolution image may take multiple minutes.
```