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

    DIPLOMAT requires a Python version between 3.8 and 3.10. Check your python version by running ``python3 --version``.
    If your version of Python falls outside of this range, you can use Miniforge or conda to create an appropriately-versioned environment.
    For example, to create a Python 3.10 environment, run ``conda create -n py310 python==3.10``, and activate it with ``conda activate py310``.
    Now check again that your Python has the correct version with ``python3 --version``.

Installing DIPLOMAT
-------------------

Overview
^^^^^^^^

The DIPLOMAT installation process can be broken down into two steps: 
first, installing one of the front-end networks, SLEAP or DeepLabCut (DLC), 
and second, installing DIPLOMAT itself.

Note that in order to keep both DIPLOMAT and the front-end contained in one place,
you will create an environment either with PIP or conda,
and **this environment must be activated every time you wish to use DIPLOMAT**.

In the first step, you will install whichever front-end you want to use. 
We currently support `SLEAP <https://sleap.ai/installation.html>`_ and `DeepLabCut <https://deeplabcut.github.io/DeepLabCut/README.html>`_. 
Be aware that these are third-party tools with potentially complex installation procedures.
The installation tutorials we provide use the methods that we found to be most reliable across most systems, 
but *we cannot guarantee that any third-party tool will work on your computer*.
If you have difficulty installing a particular front-end, you are encouraged to refer to its internal documentation.

After the front-end is installed, you will install DIPLOMAT **in the same environment**. 
The ``diplomat-track`` package is available on PIP and offers a streamlined installation.
If you are installing with GUI, i.e., ``diplomat-track[gui]``, 
the UI requirements may take several minutes to install on some systems.
In general, if you create an environment containing the front-end network, 
you will be able to install DIPLOMAT to that environment.

.. hint::

    Both running and installing diplomat requires access to a terminal. To access one:

    **Windows:** Open the start menu and search for *Miniforge Prompt*.

    **Linux:** Press :kbd:`CTRL` + :kbd:`ALT` + :kbd:`T`. This will open a terminal window.

    **Mac:** Select the search icon in the top right corner of the screen to open Spotlight, and
    then search for *Terminal*.

Install on Linux (with PIP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. collapse:: Install DIPLOMAT-SLEAP on Linux

	|

	First, create the environment and activate.
	**You will need to activate this environment every time you use DIPLOMAT.**

	.. code-block:: sh

	    # create a Python 3.10 virtual environment
	    python3 -m venv venv-sleap

	    # activate the environment
	    source venv-sleap/bin/activate

	    # upgrade pip
	    python3 -m pip upgrade pip

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

|

.. collapse:: Install DIPLOMAT-DLC on Linux

	|

	First, create the environment and activate.
	**You will need to activate this environment every time you use DIPLOMAT.**

	.. code-block:: sh

	    # create a Python 3.10 virtual environment
	    python3 -m venv venv-dlc

	    # activate the environment
	    source venv-dlc/bin/activate

	    # upgrade pip
	    python3 -m pip upgrade pip

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

|

These procedures were tested primarily on Ubuntu and CentOS. 
If you're running a different distro, we trust that you can make the appropriate changes.
The *MacOS (with conda)* tutorials below may also work on some Linux systems. 

|

Install on MacOS (with conda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. collapse:: Install DIPLOMAT-SLEAP on MacOS

	|

	First, create the environment and activate.
	**You will need to activate this environment every time you use DIPLOMAT.**

	.. code-block:: sh

	    # create the environment
	    conda create -n diplomat-sleap -c conda-forge "python==3.9"

	    # activate the environment
	    conda activate diplomat-sleap


	Next, you'll install SLEAP.
	For more information about the SLEAP installation process, 
	refer to the `SLEAP installation guide <https://sleap.ai/installation.html>`_.

	.. code-block:: sh

	    # install SLEAP and verify
	    conda install -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.3.3
	    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
	    python3 -c "import sleap; sleap.versions()"
	    python3 -c "import sleap; sleap.system_summary()"

	Install DIPLOMAT. 
	Omit the `[gui]` option if you are installing on HPC or other headless systems.

	.. code-block:: sh

	    # install DIPLOMAT and verify
	    python3 -m pip install "diplomat-track[gui]"
	    diplomat --version

|

.. collapse:: Install DIPLOMAT-DLC on MacOS

	|

	First, create the environment and activate.
	**You will need to activate this environment every time you use DIPLOMAT.**

	.. code-block:: sh

	    # create the environment
	    conda create -n diplomat-dlc -c conda-forge "python==3.10"

	    # activate the environment
	    conda activate diplomat-dlc

	Next, you'll install DeepLabCut.
	For more information about the DeepLabCut installation process, 
	refer to the `DeepLabCut installation guide <https://deeplabcut.github.io/DeepLabCut/README.html>`_.

	.. code-block:: sh

	    # install DeepLabCut and verify
	    conda install "numpy<1.24.0"
	    conda install "deeplabcut[tf]"
	    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

	Install DIPLOMAT. 
	Omit the `[gui]` option if you are installing on HPC or other headless systems.

	.. code-block:: sh

	    # install DIPLOMAT and verify
	    python3 -m pip install "diplomat-track[gui]"
	    diplomat --version
|

Install on Windows
^^^^^^^^^^^^^^^^^^

.. collapse:: Install DIPLOMAT-SLEAP on Windows

	|

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

	You may need to fix the Numpy version in order for SLEAP to run properly.

	.. code-block:: sh

	    # fix the Numpy version
	    python3 -m pip install "numpy<1.23.0"
	    # verify
	    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
	    python3 -c "import sleap; sleap.versions()"
	    python3 -c "import sleap; sleap.system_summary()"

	Finally, install DIPLOMAT.
	Omit the `[gui]` option if you are installing on HPC or other headless systems.
	
	.. code-block:: sh

	    # install DIPLOMAT and verify
	    python3 -m pip install "diplomat-track[gui]"
	    diplomat --version

|

.. collapse:: Install DIPLOMAT-DLC on Windows

	|

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

|

Troubleshooting
---------------

.. collapse:: MacOS Tips
	
	|

	* If the DLC installation crashes with an error about `tables` or `hdf5`, make sure your system has the prerequisite packages installed: ``brew install hdf5 c-blosc lzo bzip2``.
	
	|

	* In some cases, it may be necessary to pre-configure conda before a Python environment can be created:

		.. code-block:: sh 

			% conda config --add channels conda-forge
			% conda config --set channel_priority strict

	|
	
	* Some users have reported success using SLEAP's default mamba installation method to create an environment: 
		
		.. code-block:: 

			mamba create -y -n diplomat-sleap -c conda-forge -c anaconda -c sleap sleap=1.3.3
			conda activate diplomat-sleap
			pip install "diplomat-track[gui]"

		* However, on many systems this will result in a Python 3.7 environment, which is incompatible with DIPLOMAT.

| 

.. collapse:: Linux Tips

	|

	* On some systems it may be easier to use a conda environment rather than a PIP venv. If you can't get DIPLOMAT working with the PIP methods in the Linux section, try using the conda methods in the MacOS section.

	|

	* When installing DLC, you may need to run ``pip install "numpy<1.24.0"`` *after* installing DLC in addition to running it before.

| 

.. collapse:: GPU Troubles

	|

	* Your system may not have the necessary NVIDIA, CUDA, and cuDNN libraries pre-installed. In this case, refer to `TensorFlow's Software requirements <https://www.tensorflow.org/install/pip#software_requirements>`_ for links to the relevant libraries. For an installation tutorial, refer to the `NVIDIA installation docs <https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html>`_. Alternatively, if the conda (MacOS) method works on your machine, you can run ``conda install -c conda-forge cudatoolkit cudnn`` before proceeding to the DLC or SLEAP installation step.

|

.. collapse:: Installing SLEAP and DLC to the same environment

	|
	
	* If you want to install both SLEAP and DLC to the same environment, **the SLEAP installation must be performed before the DLC installation!** Generally, we recommend creating a distinct environment for each front-end.

| 

.. collapse:: Alternate Installation Methods

	|

	* If the standard methods do not work, consider installing DIPLOMAT from source with the `developer installation method <advanced_usage.html>`_.

|

Verifying your DIPLOMAT installation
------------------------------------

We have created a  `Zenodo record <https://zenodo.org/records/14232002>`_ with pretrained SLEAP and DeepLabCut projects and a short video clip
with which you can check your DIPLOMAT installation.

.. collapse:: Verify DIPLOMAT-SLEAP

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
	
	If you installed ``diplomat-track[gui]``, verify that the Interact GUI appears after this command completes.

	.. code-block:: sh

	    # verify that interact works 
	    diplomat track_and_interact -c test_sleap_5/ -v N5PZS.avi -no 3

|

.. collapse:: Verify DIPLOMAT-DLC

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
	
	If you installed ``diplomat-track[gui]``, verify that the Interact GUI appears after this command completes.
	
	.. code-block:: sh

	    # verify that tracking works
	    diplomat track_and_interact -c test_dlc_5/config.yaml -v N5PZS.avi -no 3

|
