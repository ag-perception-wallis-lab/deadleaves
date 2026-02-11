# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import warnings

project = "Dead Leaves"
author = "Swantje Mahncke, Lynn Schmittwilken"
copyright = f"2025, {author}"
release = "0.1"

root_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",  # execute code-cells in markdown files
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",  # support LaTeX
    "sphinx_gallery.gen_gallery",  # sphinx gallery
    "autoapi.extension",  # automatic api reference from docstrings
    "sphinx_design",  # allow sphinx style in markdown files
    "sphinx_copybutton",  # copy buttons on code cells
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]  # avoid recursive rebuilding on change

# MyST-NB
nb_execution_mode = "auto"

myst_enable_extensions = ["dollarmath", "colon_fence"]

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="myst_nb",
)

# LaTeX
latex_engine = "lualatex"

# Sphinx Gallery
sys.path.insert(0, os.path.dirname(__file__))

sphinx_gallery_conf = {
    "examples_dirs": "../examples/gallery",  # path to example scripts
    "gallery_dirs": "gallery",  # path to where to save gallery generated output
    "image_scrapers": ("dead_leaves_scraper.dead_leaves_scraper"),
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
}

# Auto API
autoapi_dirs = ["../deadleaves"]  # path to package
autoapi_root = "reference"  # path to where to save the auto api generated output
autoapi_keep_files = True

if os.environ.get("SPHINX_AUTOBUILD") == "1":
    autoapi_generate_api_docs = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ag-perception-wallis-lab/deadleaves",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_end": ["search-button", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
}
html_title = f"{project} v{release} Manual"
html_logo = "_static/logo_dead_leaves.png"
html_favicon = "_static/logo_dead_leaves.png"
