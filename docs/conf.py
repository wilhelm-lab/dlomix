# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src/"))
sys.path.insert(0, os.path.abspath("../src/dlomix/"))


# -- Project information -----------------------------------------------------

from src.dlomix._metadata import (
    __author__,
    __copyright__,
    __github_url__,
    __package__,
    __version__,
)

project = __package__
copyright = __copyright__
author = __author__

# The full version, including alpha/beta/rc tags

release = __version__


# -- Backend configuration ---------------------------------------------------

from src.dlomix.config import (
    _BACKEND,
    BACKEND_PRETTY_NAME,
    PYTORCH_BACKEND,
    TENSORFLOW_BACKEND,
)

# Add to context for use in templates
html_context = {
    "backend": _BACKEND,
    "other_backend": PYTORCH_BACKEND[0]
    if _BACKEND in TENSORFLOW_BACKEND
    else TENSORFLOW_BACKEND[0],
    "backend_label": BACKEND_PRETTY_NAME,
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx.ext.viewcode"]

autodoc_default_options = {"members": True, "undoc-members": False}

highlight_language = "pycon"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

# book theme
html_theme_options = {
    "repository_url": __github_url__,
    "use_repository_button": True,
    "logo": {
        "image_dark": "assets/logo_dark.jpg",
    },
    # toc on the right side
    "show_toc_level": 1,
    "toc_title": "",
    "home_page_in_toc": True,
    # nav bar depth on the left
    "show_navbar_depth": 2,
    "path_to_docs": "docs/",
    "use_edit_page_button": True,
}

html_logo = "assets/logo.jpg"

html_title = f"DLOmix | v{release}"
html_use_index = False  # Don't create index
html_domain_indices = False  # Don't need module indices
html_copy_source = False  # Don't need sources
html_permalinks = True
html_permalinks_icon = "🔗"
add_module_names = True
option_emphasise_placeholders = True
html_show_sphinx = False

python_use_unqualified_type_names = True


sitemap_url_scheme = "{link}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Extension configuration -------------------------------------------------
