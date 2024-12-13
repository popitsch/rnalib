# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))  # this is needed to find the rnalib package
print(f"Executable: {sys.executable}, path: {sys.path}")
import rnalib

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "rnalib"
copyright = "2024, Niko Popitsch"
author = "Niko Popitsch"
version = rnalib.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_mdinclude",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinxcontrib.apidoc",
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_param = False
napoleon_use_ivar = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

autodoc_default_flags = ["members", "undoc-members"]
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst", ".md"]

apidoc_module_dir = "../rnalib"
apidoc_output_dir = "_api"
apidoc_excluded_paths = ["tests"]
apidoc_separate_modules = True
apidoc_toc_file = False
apidoc_module_first = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = '_static/rnalib_logo.png'
