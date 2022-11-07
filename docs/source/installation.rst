Installation
============

DIPLOMAT currently supports being installed as a normal python package on Windows, Linux, and MacOS.
DIPLOMAT and can be installed to

Installing Python
-----------------

If you have not already, you'll need to install python to utilize DIPLOMAT. We recommend using
:ref:`Miniconda <https://docs.conda.io/en/latest/miniconda.html>` which provides python environment
and install process that is consistent across platforms. To install miniconda:

 - Visit :ref:`https://docs.conda.io/en/latest/miniconda.html <https://docs.conda.io/en/latest/miniconda.html>`.
 - Select the installer for your OS from the list of installers to download it.
 - Run the instructions an follow the installation instructions.

Installing DIPLOMAT
-------------------

Using Conda or Miniconda
^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a anaconda installed, you'll want to open a terminal and type:

.. code-block: shell

    conda env create -f some/magic/url


Using PIP
^^^^^^^^^

Once you are using an alternative package for managing python environments, you can still install
DIPLOMAT by simply using pip, using one of the two commands below:

.. code-block: shell

    # Install DIPLOMAT with GUI support.
    pip install "diplomat-track[gui] @ git+https://github.com/TravisWheelerLab/DIPLOMAT.git"
    # Install DIPLOMAT without UI support.
    pip install git+https://github.com/TravisWheelerLab/DIPLOMAT.git

