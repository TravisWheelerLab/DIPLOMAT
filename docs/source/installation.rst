Installation
============

DIPLOMAT currently supports being installed as a normal python package on Windows, Linux, and MacOS.
DIPLOMAT and can be installed by following the installation guide below.

.. contents:: Contents



Installing Python
-----------------

If you have not already, you'll need to install python to utilize DIPLOMAT. It is recommend that you use
`Miniforge <https://github.com/conda-forge/miniforge>`_ which provides a python environment
and install process that is consistent across platforms. To install Miniforge:

 - Visit `https://github.com/conda-forge/miniforge <https://github.com/conda-forge/miniforge>`_.
 - Select the installer for your OS from the list of installers.
 - Run the installer and follow the installation instructions.

.. hint::

    In these instructions, we use the commands ``python3`` and ``python3 -m pip`` to invoke Python and PIP, respectively.

    Your machine may vary! If the commands are not recognized, try ``python`` for Python, and ``python -m pip`` or just ``pip`` for PIP.

.. hint::

    We recommend using a Python version between 3.8 and 3.10. Check your python version by running ``python3 --version``.
    If your version of Python falls outside of this range, you can use Miniforge or conda to create an appropriately-versioned environment.
    For example, to create a Python 3.10 environment, run ``conda create -n py310 python==3.10``, and activate it with ``conda activate py310``.
    Now check again that your Python has the correct version with ``python3 --version``.

Installing DIPLOMAT
-------------------

.. hint::

    Both running and installing diplomat requires access to a terminal. To access one:

    **Windows:** Open the start menu and search for *Miniforge Prompt*.

    **Linux:** Press :kbd:`CTRL` + :kbd:`ALT` + :kbd:`T`. This will open a terminal window.

    **Mac:** Select the search icon in the top right corner of the screen to open Spotlight, and
    then search for *Terminal*.

MacOS and Linux
^^^^^^^^^^^^^^^

With support for SLEAP projects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create the environment and activate.
**You will need to activate this environment every time you use DIPLOMAT.**

.. code-block:: sh

    # create a Python 3.10 virtual environment
    python3 -m venv venv

    # activate the environment
    source venv/bin/activate


Next, you'll install SLEAP.
For more information about the SLEAP installation process, 
refer to the `SLEAP installation guide <https://sleap.ai/installation.html>`_.

.. code-block:: sh

    # install SLEAP and verify
    python3 -m pip install "sleap[pypi]"
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    python3 -c "import sleap; sleap.versions()"
    python3 -c "import sleap; sleap.system_summary()"

Install DIPLOMAT. 
Omit the `[gui]` option if you are installing on HPC or other headless systems.

.. code-block:: sh

    # install DIPLOMAT and verify
    python3 -m pip install "diplomat-track[gui]"
    diplomat --version

In order to verify the installation, download the testing resources 
**N5PZS.avi** and **SLEAP_5bp.zip** from our Zenodo record: `Zenodo record <https://zenodo.org/records/14232002>`_.
Unzip **SLEAP_5bp.zip** and put the **test_sleap_5** folder in the same directory as **N5PZS.avi**. 
Alternatively, use these `curl` commands to download and unzip the resources. 

.. code-block:: sh

    # download and unzip files from https://zenodo.org/records/14232002,
    # or do it in the terminal with curl:
    curl https://zenodo.org/records/14232002/files/SLEAP_5bp.zip --output SLEAP_5bp.zip && unzip SLEAP_5bp.zip
    curl https://zenodo.org/records/14232002/files/N5PZS.avi --output N5PZS.avi

Finally, verify the tracking functionality for DIPLOMAT-SLEAP.
**Make sure both the video file `N5PZS.avi` and the SLEAP project folder `test_sleap_5` are in your current directory.**

.. code-block:: sh

    # verify that tracking works
    diplomat track -c test_sleap_5/ -v N5PZS.avi -no 3

With support for DeepLabCut projects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create the environment and activate.
**You will need to activate this environment every time you use DIPLOMAT.**

.. code-block:: sh

    # create a Python 3.10 virtual environment
    python3 -m venv venv

    # activate the environment
    source venv/bin/activate

Next, you'll install DeepLabCut.
For more information about the DeepLabCut installation process, 
refer to the `DeepLabCut installation guide <https://deeplabcut.github.io/DeepLabCut/README.html>`_.

.. code-block:: sh

    # install DeepLabCut and verify
    python3 -m pip install "numpy<1.24.0"
    python3 -m pip install "deeplabcut[tf]"
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Install DIPLOMAT. 
Omit the `[gui]` option if you are installing on HPC or other headless systems.

.. code-block:: sh

    # install DIPLOMAT and verify
    python3 -m pip install "diplomat-track[gui]"
    diplomat --version

In order to verify the installation, download the testing resources 
**N5PZS.avi** and **DLC_5bp.zip** from our Zenodo record: `Zenodo record <https://zenodo.org/records/14232002>`_.
Unzip **DLC_5bp.zip** and put the **test_dlc_5** folder in the same directory as **N5PZS.avi**. 
Alternatively, use these `curl` commands to download and unzip the resources. 

.. code-block:: sh

    # download and unzip files from https://zenodo.org/records/14232002,
    # or do it in the terminal with curl:
    curl https://zenodo.org/records/14232002/files/DLC_5bp.zip --output DLC_5bp.zip && unzip DLC_5bp.zip
    curl https://zenodo.org/records/14232002/files/N5PZS.avi --output N5PZS.avi

    # your working directory should now contain "test_dlc_5" and "N5PZS.avi".

Finally, verify the tracking functionality for DIPLOMAT-DLC.
**Make sure both the video file `N5PZS.avi` and the DLC project folder `test_dlc_5` are in your current directory.**

.. code-block:: sh

    # verify that tracking works
    diplomat track -c test_dlc_5/config.yaml -v N5PZS.avi -no 3

Windows
^^^^^^^

With support for SLEAP projects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create the environment and activate.
**You will need to activate this environment every time you use DIPLOMAT.**

.. code-block:: sh

    # create the environment
    ## with GPU
    mamba env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-SLEAP.yaml
    ## with CPU
    mamba env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-SLEAP-CPU.yaml
    
    # activate the environment
    mamba activate DIPLOMAT-SLEAP

    # fix the Numpy version
    python3 -m pip install "numpy<1.23.0"

    # verify
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    python3 -c "import sleap; sleap.versions()"
    python3 -c "import sleap; sleap.system_summary()"
    diplomat --version

In order to verify the installation, download the testing resources 
**N5PZS.avi** and **SLEAP_5bp.zip** from our Zenodo record: `Zenodo record <https://zenodo.org/records/14232002>`_.
Unzip **SLEAP_5bp.zip** and put the **test_sleap_5** folder in the same directory as **N5PZS.avi**. 
Alternatively, use these `curl` commands to download and unzip the resources. 

.. code-block:: sh

    # download and unzip files from https://zenodo.org/records/14232002,
    # or do it in the terminal with curl:
    curl https://zenodo.org/records/14232002/files/SLEAP_5bp.zip --output SLEAP_5bp.zip && unzip SLEAP_5bp.zip
    curl https://zenodo.org/records/14232002/files/N5PZS.avi --output N5PZS.avi

Finally, verify the tracking functionality for DIPLOMAT-SLEAP.
**Make sure both the video file `N5PZS.avi` and the SLEAP project folder `test_sleap_5` are in your current directory.**

.. code-block:: sh

    # verify that tracking works
    diplomat track -c test_sleap_5/ -v N5PZS.avi -no 3

With support for DeepLabCut projects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create the environment and activate.
**You will need to activate this environment every time you use DIPLOMAT.**

.. code-block:: sh

    # create the environment
    conda create -n diplomat_dlc python==3.10
    
    # activate the environment
    conda activate diplomat_dlc

Next, you'll install DeepLabCut.
For more information about the DeepLabCut installation process, 
refer to the `DeepLabCut installation guide <https://deeplabcut.github.io/DeepLabCut/README.html>`_.

.. code-block:: sh

    # install DLC and verify
    python3 -m pip install "numpy<1.24.0"
    python3 -m pip install "deeplabcut[tf]"
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Install DIPLOMAT. 
Omit the `[gui]` option if you are installing on HPC or other headless systems.

.. code-block:: sh

    # install DIPLOMAT and verify
    python3 -m pip install "diplomat-track[gui]"
    diplomat --version

In order to verify the installation, download the testing resources 
**N5PZS.avi** and **DLC_5bp.zip** from our Zenodo record: `Zenodo record <https://zenodo.org/records/14232002>`_.
Unzip **DLC_5bp.zip** and put the **test_dlc_5** folder in the same directory as **N5PZS.avi**. 
Alternatively, use these `curl` commands to download and unzip the resources. 

.. code-block:: sh

    # download and unzip files from https://zenodo.org/records/14232002,
    # or do it in the terminal with curl:
    curl https://zenodo.org/records/14232002/files/DLC_5bp.zip --output DLC_5bp.zip && unzip DLC_5bp.zip
    curl https://zenodo.org/records/14232002/files/N5PZS.avi --output N5PZS.avi

    # your working directory should now contain "test_dlc_5" and "N5PZS.avi".

Finally, verify the tracking functionality for DIPLOMAT-DLC.
**Make sure both the video file `N5PZS.avi` and the DLC project folder `test_dlc_5` are in your current directory.**

.. code-block:: sh

    # verify that tracking works
    diplomat track -c test_dlc_5/config.yaml -v N5PZS.avi -no 3

Alternate Methods
^^^^^^^^^^^^^^^^^

If the standard methods do not work, consider installing DIPLOMAT from source with the `developer installation method <advanced_usage.html>`_.
