from pathlib import Path
import sys

# Add project root directory to python path...
sys.path.insert(0, str(Path(__file__).resolve().parent / "ext"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DIPLOMAT'
copyright = '2022, Isaac Robinson, Travis Wheeler, Nathan Insel'
author = 'Isaac Robinson, Travis Wheeler, Nathan Insel'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "plugin_docgen"
]

imgmath_image_format = "svg"

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["tensorflow", "pandas", "scipy", "deeplabcut"]

autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]