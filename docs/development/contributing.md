# Contributing

We are happy for anyone who is interested in contributing to the `deadleaves` project.
Head over to GitHub to:

::::{grid} 2 
:gutter: 3
 
:::{grid-item-card} Ask a question ❓
:link: https://github.com/ag-perception-wallis-lab/deadleaves/issues/new
:text-align: center 

:::

:::{grid-item-card} Report a bug 🪲
:link: https://github.com/ag-perception-wallis-lab/deadleaves/issues/new?template=bug_report.md
:text-align: center 

:::
::::

## Development

If you would like to contribute directly to the code:

1. **Fork** the GitHub repository.
2. **Clone** the forked repository to your local machine.
3. **Install** `deadleaves` with the development dependencies.
    - We recommend creating an *editable* installation using `pip install -e ".[dev]"`
    - Use `".[dev,docs]"` to also edit and build the documentation

The `deadleaves` project uses a couple of development tools to improve code quality:

- `pytest` for unit and integration tests
- `black` for consistent code formatting
- `flake8` for linting

Before submitting a pull request, please ensure that all tests pass and that your code is formatted with `black`.

```{toctree}
:hidden:
:maxdepth: 1

Ask a question <https://github.com/ag-perception-wallis-lab/deadleaves/issues>
Report a bug <https://github.com/ag-perception-wallis-lab/deadleaves/issues/new?template=bug_report.md>
```