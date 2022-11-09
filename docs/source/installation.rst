Installation
============

DIPLOMAT currently supports being installed as a normal python package on Windows, Linux, and MacOS.
DIPLOMAT and can be installed by following the installation guide below.

.. contents:: Contents



Installing Python
-----------------

If you have not already, you'll need to install python to utilize DIPLOMAT. It is recommend that you use
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ which provides python environment
and install process that is consistent across platforms. To install miniconda:

 - Visit `https://docs.conda.io/en/latest/miniconda.html <https://docs.conda.io/en/latest/miniconda.html>`_.
 - Select the installer for your OS from the list of installers.
 - Run the installer and follow the installation instructions.

Installing DIPLOMAT
-------------------

Using Conda or Miniconda
^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a anaconda installed, you'll want to open a terminal and type:

.. code-block:: sh

    conda env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-DEEPLABCUT.yaml

.. hint::

    Both running and installing diplomat requires access to a terminal. To access one:

    **Windows:** Open the start menu and search for *Anaconda Command Prompt*. If a program by that
    name is not found, search for *Command Prompt*.

    **Linux:** Press :kbd:`CTRL` + :kbd:`ALT` + :kbd:`T`. This will open a terminal window.

    **Mac:** Select the search icon in the top right corner of the screen to open Spotlight, and
    then search for *Terminal*.

Once done, simply activate the brand new environment.

.. code-block:: sh

    conda activate DIPLOMAT-DEEPLABCUT

From here, the ``diplomat`` command will be available from the command line.

Using PIP
^^^^^^^^^

If you are using an alternative package for managing python environments, you can install
DIPLOMAT by simply using pip, using one of the two commands below:

.. code-block:: sh

    # Install DIPLOMAT with GUI support.
    pip install "diplomat-track[gui] @ git+https://github.com/TravisWheelerLab/DIPLOMAT.git"
    # Install DIPLOMAT without UI support.
    pip install git+https://github.com/TravisWheelerLab/DIPLOMAT.git


