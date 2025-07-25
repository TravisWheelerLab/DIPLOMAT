from pathlib import Path
import sys

# noinspection PyUnresolvedReferences
from sphinx.ext.autodoc.mock import mock
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

# Add project root directory to python path...
sys.path.insert(0, str(Path(__file__).resolve().parent / "ext"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _get_version() -> str:
    with mock(autodoc_mock_imports):
        import diplomat

        return diplomat.__version__


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DIPLOMAT"
copyright = "2024, Isaac Robinson, George Glidden, Nathan Insel, Travis Wheeler"
author = "Isaac Robinson, George Glidden, Nathan Insel, Travis Wheeler"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "sphinx_toolbox.collapse",
    "sphinx_tabs.tabs",
    "plugin_docgen",
    "enum_tools.autoenum",
]

imgmath_image_format = "svg"

templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = [
    "sleap",
    "tensorflow",
    "pandas",
    "numba",
    "deeplabcut",
    "wx",
    "tf2onnx",
    "onnxruntime",
    "onnx",
]

autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

version = _get_version()
release = version

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]
