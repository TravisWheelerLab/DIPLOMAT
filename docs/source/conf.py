from pathlib import Path
import sys
from typing import Any, Literal, Dict

from sphinx.application import Sphinx
# noinspection PyUnresolvedReferences
from sphinx.ext.autodoc.mock import mock
import os
import inspect


os.environ["NUMBA_DISABLE_JIT"] = "1"

# Add project root directory to python path...
sys.path.insert(0, str(Path(__file__).resolve().parent / "ext"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_MOCKED_PACKAGES = [
    "tensorflow",
    "numba",
    "wx",
    "tf2onnx",
    "onnxruntime",
    "onnx",
]


def _get_version() -> str:
    with mock(_MOCKED_PACKAGES):
        # We hack numba so we can see numba functions properly documented...
        import numba
        numba.njit = lambda sig: sig if callable(sig) else (lambda x: x)

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

autodoc_mock_imports = _MOCKED_PACKAGES

autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

version = _get_version()
release = version

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"{project} {version}"

html_static_path = ["_static"]
html_favicon = "_static/imgs/logos/DIPLOMAT_icon_small.svg"
html_theme_options = {
    "light_logo": "imgs/logos/DIPLOMAT_icon_large_dark.svg",
    "dark_logo": "imgs/logos/DIPLOMAT_icon_large_light.svg",
    "sidebar_hide_name": True,
}

html_css_files = [
    "css/custom.css",
]


WhatType = Literal['module', 'class', 'exception', 'function', 'method', 'attribute']


def _resolve_class(func):
    cls = sys.modules.get(func.__module__)
    if cls is None:
        return None
    for name in func.__qualname__.split('.')[:-1]:
        cls = getattr(cls, name, None)
        if cls is None:
            return None
    if not inspect.isclass(cls):
        return None
    return cls


def custom_skip_function(app: Sphinx, what: WhatType, name: str, obj: Any, skip: bool, options: Dict[str, bool]):
    import inspect
    # print(app, what, name, obj, skip, options)
    if what == "class":
        if not name.startswith("_") or name == "__init__":
            return skip
        doc = getattr(obj, "__doc__", None)
        if inspect.isfunction(obj) and doc is not None and len(doc) > 0:
            if not obj.__module__.startswith("diplomat"):
                return True
            cls = _resolve_class(obj)
            obj_name = obj.__name__
            # Only document if explicitly defined in this class (not from parent class...)
            if cls is not None and (obj_name in cls.__dict__ or obj_name in getattr(cls, "__slots__", [])):
                return False
            return True
        return skip

    return None


def setup(app):
    app.connect("autodoc-skip-member", custom_skip_function)
