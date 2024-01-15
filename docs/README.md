# DIPLOMAT Documentation

This folder contains documentation for DIPLOMAT. These are automatically built using Github actions/read the docs when
pushing to the main branch of diplomat, but can also be built locally. 

DIPLOMAT uses [Sphinx](https://www.sphinx-doc.org/en/master/) for handling documentation. All the docs can be found in
the `source` folder. Sphinx uses restructured text (`.rst`) for handling documentation. 
The index of the documentation is `source/index.rst`, so if you add a file to DIPLOMAT's documentation, make sure to
include it in the index file under its table of contents (under the `toctree` element). 

Configuration of Sphinx is found under `source/conf.py`. DIPLOMAT uses some custom styling (`source/_static/css`), and
also includes a custom extension for properly documenting and building links for the API, CLI, and Plugins. This
extension can be found under `source/ext/plugin_docgen.py`. The extension adds custom types and links to Sphinx, so
that CLI commands, and plugins can be linked to easily in documentation and doc strings in the code. 
Some examples are shown below:
```rst
To link to a plugin, use the plugin prefix like :plugin:`~diplomat.predictors.FastPlotterArgmax`, 
:plugin:`~diplomat.frontends.SLEAPFrontend`, or :plugin:`~diplomat.predictors.fpe.frame_pass.MITViterbi`.

To link to a CLI commands, use the cli prefix, like :cli:`diplomat track`, or :cli:`diplomat frontends list loaded`.
```

The extension also comes with custom templates, which are a mix of restructured text and python f-strings for 
dynamically injecting info from the extension. These are found in `source/_templates`, and exclude the custom class
and custom module templates (which are just standard sphinx/jinja templates).

## Building Documentation Locally

To build documentation on your local machine, you'll need to create a python environment for docs and activate it. 
This can be done using venv as below. (You can also use other environments, such as conda or mamba environments)

```bash
# On unix
python -m venv venv
source venv/bin/activate
# On windows (cmd)
python -m venv venv
cd venv/Scripts
Activate.bat
cd ../..
```

Once a virtual environment is activated, you can install the documentation dependencies:

```bash
pip install -r requirements.txt
```

And build the documentation (as html) use the following command:

```bash
make html
```

Now you can navigate to [build/html/index.html](build/html/index.html) relative to this folder, to preview the 
locally built docs in your browser.

DIPLOMAT's documentation can be built into many formats (epub, pdf, latex, man pages, etc). 
To see all the other formats DIPLOMAT's docs can build to, simply run the command `make help`.
