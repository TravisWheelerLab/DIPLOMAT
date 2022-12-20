Advanced Usage of DIPLOMAT
==========================

.. contents:: Contents

Inspecting DIPLOMAT Plugins
---------------------------

DIPLOMAT was developed with extensibility in mind, so core functionality can be extended via
plugins. DIPLOMAT has two kinds of plugins:

 - Predictors: Plugins that take in model outputs and predict poses, or animal locations from them. Some of these also have additional side effects such as plotting or frame export.
 - Frontends: These are plugins that grab frames from another tracking software and pipe them into the predictor the user has selected.
   Currently, there is two for `DeepLabCut <https://github.com/DeepLabCut/DeepLabCut>`_ and `SLEAP <https://sleap.ai/>`_.

To get information about predictors, one can use the commands of the ``diplomat predictors`` sub-command space:

.. code-block:: sh

    # List predictor names and their descriptions (Names are passed to -p flag of track).
    diplomat predictors list
    # List the settings of a predictor plugin (Can be passed to -ps flag of track to configure them).
    diplomat predictors list_settings PredictorName

To get information about frontends, use commands :py:cli:`diplomat frontends list all` and :py:cli:`diplomat frontends list loaded`:

.. code-block:: sh

    # List all frontends available
    diplomat frontends list all
    # List loaded frontends...
    diplomat frontends list loaded

Development Usage
-----------------

DIPLOMAT is written entirely in python. To set up an environment for developing DIPLOMAT, you can simply
pull down this repository and create an environment for it using the commands shown below.

.. code-block:: sh

    git clone https://github.com/TravisWheelerLab/DIPLOMAT.git
    cd DIPLOMAT
    # Create a python environment
    python -m venv venv
    # On windows use the command: venv/Scripts/activate.bat
    source venv/bin/activate

    # Install DIPLOMAT dependencies with all extras (sleap, dlc, and gui) You may want to change this to only install some extras.
    pip install -e .[all]

For most development, you'll most likely want to add additional predictor plugins.
Predictors can be found in the ``diplomat/predictors`` directory. Classes that extend Predictor are automatically
loaded from this directory. To test predictors, you can use the :py:cli:`diplomat predictors test` command:

.. code-block:: sh

    diplomat predictors test PredictorName