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

Installing DIPLOMAT
-------------------

With Support for DeepLabCut Projects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using Mamba or Conda
~~~~~~~~~~~~~~~~~~~~

Once you have mamba or a mamba compatible CLI installed, you'll want to open a terminal and type one of these
two commands:

.. code-block:: sh

    # Install diplomat with GPU support...
    mamba env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-DEEPLABCUT.yaml
    # Install diplomat with CPU support only...
    mamba env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-DEEPLABCUT-CPU.yaml

.. hint::

    Both running and installing diplomat requires access to a terminal. To access one:

    **Windows:** Open the start menu and search for *Miniforge Prompt*.

    **Linux:** Press :kbd:`CTRL` + :kbd:`ALT` + :kbd:`T`. This will open a terminal window.

    **Mac:** Select the search icon in the top right corner of the screen to open Spotlight, and
    then search for *Terminal*.

Once done, simply activate the brand new environment.

.. code-block:: sh

    mamba activate DIPLOMAT-DEEPLABCUT

From here, the ``diplomat`` command will be available from the command line.

Using PIP
~~~~~~~~~

If you are using an alternative package for managing python environments, you can install
DIPLOMAT with DeepLabCut support by simply using pip, using one of the two commands below:

.. code-block:: sh

    # Install DIPLOMAT with DeepLabCut with GUI support.
    pip install diplomat-track[dlc, gui]
    # Install DIPLOMAT with DeepLabCut without UI support.
    pip install diplomat-track[dlc]

Troubleshooting for DIPLOMAT-DLC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the standard methods fail to install DIPLOMAT and DLC, you can install from 
the Github source code. This method requires `git <https://git-scm.com/downloads>`_, as well 
as Miniforge. 

.. code-block:: sh

    # Clone the DIPLOMAT repository and navigate into it.
    git clone https://github.com/TravisWheelerLab/DIPLOMAT
    cd DIPLOMAT
    # With Miniforge, create a Python 3.10 environment and activate it.
    conda create -n py310 python==3.10
    conda activate py310
    # Use the environment you just activated to create a virtual environment ("venv") containing Python 3.10.
    python -m venv venv
    # Fully deactivate the Miniforge environment.
    # (run the command twice)
    conda deactivate
    conda deactivate 
    # Now, activate the virtual environment.
    ## On Windows, the first time you activate the venv, you may need to configure your execution policy. 
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ## Activate the venv on Windows.
    venv/scripts/Activate
    ## Activate the venv on Mac/Linux.
    source venv/bin/activate
    # Finally, install DIPLOMAT and DLC. The installation may take several minutes to complete.
    python -m pip install -e .[dlc,gui] --ignore-installed
    # Verify that the installation was successful. The following command should output the current version number.
    diplomat --version

On Windows, if DIPLOMAT crashes with "OSError: [WinError 126]", you need the libomp DLL. 
Download the .zip from https://www.dllme.com/dll/files/libomp140_x86_64/versions, extract 
it, and copy the .dll file to the torch libraries folder of your virtual environment, which 
should be located at ``.\venv\lib\site-packages\torch\lib`` within the DIPLOMAT directory. 
If you named your virtual environment something other than ``venv``, change the path accordingly.

With Support for SLEAP Projects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using Mamba or Conda
~~~~~~~~~~~~~~~~~~~~

Once you have a mamba installed, you'll want to open a terminal and type one of these two commands:

.. code-block:: sh

    # Install diplomat with GPU support...
    mamba env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-SLEAP.yaml
    # Install diplomat with CPU support only...
    mamba env create -f https://raw.githubusercontent.com/TravisWheelerLab/DIPLOMAT/main/conda-environments/DIPLOMAT-SLEAP-CPU.yaml

.. hint::

    Both running and installing diplomat requires access to a terminal. To access one:

    **Windows:** Open the start menu and search for *Miniforge Prompt*.

    **Linux:** Press :kbd:`CTRL` + :kbd:`ALT` + :kbd:`T`. This will open a terminal window.

    **Mac:** Select the search icon in the top right corner of the screen to open Spotlight, and
    then search for *Terminal*.

Once done, simply activate the brand new environment.

.. code-block:: sh

    mamba activate DIPLOMAT-SLEAP

From here, the ``diplomat`` command will be available from the command line.

Using PIP
~~~~~~~~~

If you are using an alternative package for managing python environments, you can install
DIPLOMAT with SLEAP support by simply using pip, using one of the two commands below:

NOTE: SLEAP is known to have installation issues on Windows when attempting to use pip. If you're
trying to install DIPLOMAT with SLEAP support on Windows, prefer using the mamba/miniforge method above.

.. code-block:: sh

    # Install DIPLOMAT with SLEAP with GUI support.
    pip install diplomat-track[sleap, gui]
    # Install DIPLOMAT with SLEAP without UI support.
    pip install diplomat-track[sleap]

Troubleshooting for DIPLOMAT-SLEAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the Mamba method fails to install DIPLOMAT and SLEAP, you may need to downgrade the 
numpy version manually. Activate the mamba environment with ``mamba activate DIPLOMAT-SLEAP``,
then downgrade numpy with ``pip install numpy<1.23.0``. 

Verifying the DIPLOMAT installation
-----------------------------------

Downloading the Sample data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sample models and a video are provided `on Zenodo <https://zenodo.org/records/14232002>`_ for
verifying the installation. Download the video clip `N5PZS.avi` and the model corresponding to
your installation (DLC_5bp.zip for DeepLabCut, SLEAP_5bp.zip for SLEAP.) Unzip the model. Your
working directory should now contain both the video file `N5PZS.avi` and the model folder, either 
`test_dlc_5/` or `test_sleap_5/`. Verify that both are present by running ``ls``.

Activating the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, you will activate the environment for DIPLOMAT. Unless you installed with the 
`Using PIP` method, you have to activate the environment that was created for DIPLOMAT in 
a previous step.

Activating with Mamba
~~~~~~~~~~~~~~~~~~~~~
If you used the mamba installation 
process, you'll run 
``mamba activate DIPLOMAT-DEEPLABCUT`` or
``mamba activate DIPLOMAT-SLEAP``. 

Activating with venv
~~~~~~~~~~~~~~~~~~
If you followed the virtual environment-based methods (DLC troubleshooting or developer 
install) you'll run ``venv/scripts/Activate`` on Windows or ``source venv/bin/activate`` 
on Mac/Linux (replacing `venv` with whatever you named the virtual environment.) 

Activating with PIP
~~~~~~~~~~~~~~~~~~~
If you followed the PIP-only method and installed DIPLOMAT to your default environment, 
no action is necessary.

Verify
^^^^^^
In the directory containing the sample video and model, you can run track to verify that 
all of DIPLOMAT's functionality were installed properly.

Verify tracking without GUI
~~~~~~~~~~~~~~~~~~~~
For DeepLabCut, run 
``diplomat track -c test_dlc_5 -v N5PZS.avi``. 

For SLEAP, run 
``diplomat track -c test_sleap_5 -v N5PZS.avi``. 

If the tracking completes successfully, a new file ending with extension either `.h5` or 
`.slp` will now be present.

Verify tracking with GUI
~~~~~~~~~~~~~~~~~
For DeepLabCut, run 
``diplomat track_and_interact -c test_dlc_5 -v N5PZS.avi``. 

For SLEAP, run 
``diplomat track_and_interact -c test_sleap_5 -v N5PZS.avi``. 

After tracking completes, the manual annotation window will be opened and you should be 
able to make changes to the automated results.