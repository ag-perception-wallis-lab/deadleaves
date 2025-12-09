# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Dead Leaves"
author = "Swantje Mahncke"
copyright = f"2025, {author}"
release = "0.1"

root_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_nb", "sphinx.ext.autodoc"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["docs/_build"]  # avoid recursive rebuilding on change

# MyST-NB
nb_execution_mode = "force"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_end": [
        "search-button",
        "theme-switcher",
    ],
    "navbar_persistent": [],
}
html_title = f"{project} v{release} Manual"
