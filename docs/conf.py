"""Sphinx configuration for pvtend documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from pvtend._version import __version__

project = "pvtend"
copyright = "2026, Xingjian Yan"
author = "Xingjian Yan"
version = __version__          # short X.Y
release = __version__          # full X.Y.Z

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "display_version": True,
}
html_context = {
    "display_github": True,
    "github_user": "yanxingjianken",
    "github_repo": "pvtend",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
html_static_path = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

nbsphinx_execute = "never"
