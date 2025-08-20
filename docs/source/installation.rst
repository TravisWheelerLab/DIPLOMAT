Installation
============

DIPLOMAT currently supports being installed as a normal python package on Windows, Linux, and MacOS.
DIPLOMAT and can be installed by following the installation guide below.

Installing Python
-----------------

If you have not already, you'll need to install python to utilize DIPLOMAT. It is recommend that you use
`Miniforge <https://github.com/conda-forge/miniforge>`_ which provides a python environment
and install process that is consistent across platforms. To install Miniforge:

 - Visit `https://github.com/conda-forge/miniforge <https://github.com/conda-forge/miniforge>`_.
 - Select the installer for your OS from the list of installers.
 - Run the installer and follow the installation instructions.

.. hint::

    In these instructions, we use the commands ``python`` and ``pip`` to invoke Python and PIP, respectively.

    Your machine may vary! If the commands are not recognized, try ``python3`` for Python, and ``python -m pip`` or ``python3 -m pip`` for PIP.

.. hint::

    DIPLOMAT requires a Python version of at least 3.8 or higher. Check your python version by running ``python3 --version``.
    If your version of Python is not at least 3.8, you'll need to install using the conda/Miniforge approach, or install a
    supported version of python before using the pip based install approach.

Installing DIPLOMAT
-------------------

Overview
^^^^^^^^

DIPLOMAT works independently of the animal tracking packages it can run inference on top of (SLEAP and DeepLabCut).
Although it is technically possible to install diplomat alongside these packages, it is strongly discouraged as it
leads to package conflicts in most scenarios. Instead, it is strongly recommended to install diplomat in it's
own separate python environment. This can be done by following the directions below.

It you want to create new SLEAP or DeepLabCut projects or train new models, you'll need to install their
software packages. This can be done by following their installation guides at the links below:

* SLEAP: `<https://sleap.ai/installation.html>`_
* DeepLabCut: `<https://deeplabcut.github.io/DeepLabCut/README.html>`_

Note that neither is required to run diplomat, if you just want to run diplomat you can skip this section
and follow one of the installation procedures below based on your platform.

Installation
^^^^^^^^^^^^

.. hint::

    Both running and installing diplomat requires access to a terminal. To access one:

    **Windows:** Open the start menu and search for *Miniforge Prompt*.

    **Linux:** Press :kbd:`CTRL` + :kbd:`ALT` + :kbd:`T`. This will open a terminal window.

    **Mac:** Select the search icon in the top right corner of the screen to open Spotlight, and
    then search for *Terminal*.



.. tabs::

    .. group-tab:: Windows

        .. tabs::

            .. group-tab:: Miniforge or Conda

                Open the terminal and run one of the two commands below:

                .. code-block::

                    # On systems with an NVIDIA GPU:
                    conda env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-NVIDIA.yaml
                    # On any other system:
                    conda env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT.yaml

                Once the installation finishes, you can test the installation by running the commands below.

                .. code-block::

                    # Activate the diplomat environment.
                    conda activate diplomat
                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded

            .. group-tab:: Pip

                Open the terminal, with access to the python environment you would like to install diplomat in.
                Then run one of the commands below.

                .. code-block::

                    # Install with all frontends and gui support on system with a NVIDIA GPU
                    pip install diplomat-track[all-nvidia]
                    # Install with all frontends and gui support on any other system
                    pip install diplomat-track[all]

                If more granular control is needed of what parts of diplomat should be installed,
                you can mix and match the frontend specific and ui optional dependency flags, all listed
                in the commands below.

                .. code-block::

                    # Equivalent to all-nvidia, remove parts you don't want.
                    pip install diplomat-track[sleap-nvidia, dlc-nvidia, gui]
                    # Equivalent to all, remove parts you don't want.
                    pip install diplomat-track[sleap, dlc, gui]

                Once installed, you can test diplomat is installed correctly by running the command below.

                .. code-block::

                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded


    .. group-tab:: MacOS

        .. tabs::

            .. group-tab:: Miniforge or Conda

                Open the terminal and run the command below:

                .. code-block::

                    conda env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT.yaml

                Once the installation finishes, you can test the installation by running the commands below.

                .. code-block::

                    # Activate the diplomat environment.
                    conda activate diplomat
                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded

            .. group-tab:: Pip

                Open the terminal, with access to the python environment you would like to install diplomat in.
                Then run the command below.

                .. code-block::

                    # Install with all frontends and gui support on any other system
                    pip install diplomat-track[all]

                If more granular control is needed of what parts of diplomat should be installed,
                you can mix and match the frontend specific and ui optional dependency flags, all listed
                in the commands below.

                .. code-block::

                    # Equivalent to all, remove parts you don't want.
                    pip install diplomat-track[sleap, dlc, gui]

                Once installed, you can test diplomat is installed correctly by running the command below.

                .. code-block::

                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded


    .. group-tab:: Linux

        .. tabs::

            .. group-tab:: Miniforge or Conda

                Open the terminal and run one of the two commands below:

                .. code-block::

                    # On systems with an NVIDIA GPU:
                    conda env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-NVIDIA.yaml
                    # On any other system:
                    conda env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT.yaml

                Once the installation finishes, you can test the installation by running the commands below.

                .. code-block::

                    # Activate the diplomat environment.
                    conda activate diplomat
                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded

            .. group-tab:: Pip

                Open the terminal, with access to the python environment you would like to install diplomat in.
                Then run one of the commands below.

                .. code-block::

                    # Install with all frontends and gui support on system with a NVIDIA GPU
                    pip install diplomat-track[all-nvidia]
                    # Install with all frontends and gui support on any other system
                    pip install diplomat-track[all]

                If more granular control is needed of what parts of diplomat should be installed,
                you can mix and match the frontend specific and ui optional dependency flags, all listed
                in the commands below.

                .. code-block::

                    # Equivalent to all-nvidia, remove parts you don't want.
                    pip install diplomat-track[sleap-nvidia, dlc-nvidia, gui]
                    # Equivalent to all, remove parts you don't want.
                    pip install diplomat-track[sleap, dlc, gui]

                Once installed, you can test diplomat is installed correctly by running the command below.

                .. code-block::

                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded


Development Installation Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. collapse:: DIPLOMAT Installation for Development

    |

    * If you plan on developing frontends or predictors for DIPLOMAT, consider installing DIPLOMAT from source with the `developer installation method <advanced_usage.html>`_.

|

Verifying your DIPLOMAT Installation
------------------------------------

We have created a  `Zenodo record <https://zenodo.org/records/14232002>`_ with pretrained SLEAP and DeepLabCut projects and a short video clip
with which you can check your DIPLOMAT installation.

.. collapse:: Verify with SLEAP

    |

    In order to verify the installation, download the testing resources
    **N5PZS.avi** and **SLEAP_5bp.zip** from our `Zenodo record <https://zenodo.org/records/14232002>`_.
    Unzip **SLEAP_5bp.zip** and put the **test_sleap_5** folder in the same directory as **N5PZS.avi**.
    Alternatively, use these `curl` commands to download and unzip the resources.

    .. code-block:: sh

        # download and unzip files from https://zenodo.org/records/14232002,
        # or do it in the terminal with curl:
        curl https://zenodo.org/records/14232002/files/SLEAP_5bp.zip --output SLEAP_5bp.zip && unzip SLEAP_5bp.zip
        curl https://zenodo.org/records/14232002/files/N5PZS.avi --output N5PZS.avi

    Finally, verify the tracking functionality for DIPLOMAT-SLEAP.
    **Make sure both the video file `N5PZS.avi` and the SLEAP project folder `test_sleap_5` are in your current directory.**


    Verify that DIPLOMAT's primary tracking functionality works.
	
    .. code-block:: sh

        # verify that tracking works
        diplomat track -c test_sleap_5/ -v N5PZS.avi -no 3
	
    If you installed diplomat with ``"all"``, or ui support, verify that the Interact GUI appears after this command completes.

    .. code-block:: sh

        # verify that interact works
        diplomat track_and_interact -c test_sleap_5/ -v N5PZS.avi -no 3

|

.. collapse:: Verify with DeepLabCut

    |

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

    Verify that DIPLOMAT's primary tracking functionality works.

    .. code-block:: sh

        # verify that tracking works
        diplomat track -c test_dlc_5/config.yaml -v N5PZS.avi -no 3
	
    If you installed If you installed diplomat with ``"all"``, or ui support, verify that the Interact GUI appears after this command completes.
	
    .. code-block:: sh

        # verify that tracking works
        diplomat track_and_interact -c test_dlc_5/config.yaml -v N5PZS.avi -no 3

|
