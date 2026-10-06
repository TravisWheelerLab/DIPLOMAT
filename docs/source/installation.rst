Installation
============

DIPLOMAT currently supports being installed as a normal python package on Windows, Linux, and MacOS.
DIPLOMAT and can be installed by following the installation guide below.

Installing Python
-----------------

If you have not already, you'll need to install python to utilize DIPLOMAT. It is recommend that you use
`uv <https://docs.astral.sh/uv/>`_ which provides a python environment
and install process that is consistent across platforms. To install uv:

 - Visit `https://docs.astral.sh/uv/getting-started/installation/ <https://docs.astral.sh/uv/getting-started/installation/>`_.
 - Copy and paste the installation command in the terminal for your operating system, and press :kbd:`ENTER`.


.. hint::

     Both running and installing diplomat requires access to a terminal. To access one:

     **Windows:** Open the start menu and search for *Terminal*.

     **Linux:** Press :kbd:`CTRL` + :kbd:`ALT` + :kbd:`T`. This will open a terminal window.

     **Mac:** Select the search icon in the top right corner of the screen to open Spotlight, and
     then search for *Terminal*.

.. hint::

    In these instructions, we use the commands ``python`` and ``pip`` to invoke Python and PIP, respectively.

    Your machine may vary! If the commands are not recognized, try ``python3`` for Python, and ``python -m pip`` or ``python3 -m pip`` for PIP.

.. hint::

    DIPLOMAT requires a Python version of at least 3.11 or higher. Check your python version by running ``python3 --version``.
    If your version of Python is not at least 3.11, you'll need to install using the conda/Miniforge approach, or install a
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

* SLEAP: `<https://docs.sleap.ai/latest/installation/>`_
* DeepLabCut: `<https://deeplabcut.github.io/DeepLabCut/docs/installation.html>`_

.. warning::

    DeepLabCut's install guide still recommends using an outdated conda based installation, which we don't
    recommend due to conda based installations being buggy, especially with newer python versions. We recommend
    installing it with uv, which can be done with this single command:

    .. code-block:: sh

        # You can launch the UI after installation by typing 'dlc' into your terminal.
        uv tool install "deeplabcut[gui]"


Note that neither is required to run diplomat, if you just want to run diplomat you can skip this section
and follow one of the installation procedures below based on your platform.

Installation
^^^^^^^^^^^^

.. tabs::

    .. group-tab:: Windows

        .. tabs::

            .. group-tab:: uv

                Open the terminal and run the command below:

                .. code-block::

                    uv tool install "diplomat-track[all]"

                If more granular control is needed of what parts of diplomat should be installed,
                you can mix and match the frontend specific and ui optional dependency flags, all listed
                in the command below.

                .. code-block::

                    # Equivalent to all, remove parts you don't want.
                    uv tool install "diplomat-track[sleap, dlc, gui]"

                Once the installation finishes, you can test the installation by running the command below.

                .. code-block::

                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded

            .. group-tab:: pip

                Open the terminal, with access to the python environment you would like to install diplomat in.
                Then run the command below.

                .. code-block::

                    pip install diplomat-track[all]

                If more granular control is needed of what parts of diplomat should be installed,
                you can mix and match the frontend specific and ui optional dependency flags, all listed
                in the command below.

                .. code-block::

                    # Equivalent to all, remove parts you don't want.
                    pip install diplomat-track[sleap, dlc, gui]

                Once installed, you can test diplomat is installed correctly by running the command below.

                .. code-block::

                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded


    .. group-tab:: MacOS

        .. tabs::

            .. group-tab:: uv

                Open the terminal and run the command below:

                .. code-block::

                    uv tool install "diplomat-track[all]"

                If more granular control is needed of what parts of diplomat should be installed,
                you can mix and match the frontend specific and ui optional dependency flags, all listed
                in the command below.

                .. code-block::

                    # Equivalent to all, remove parts you don't want.
                    uv tool install "diplomat-track[sleap, dlc, gui]"

                Once the installation finishes, you can test the installation by running the command below.

                .. code-block::

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
                in the command below.

                .. code-block::

                    # Equivalent to all, remove parts you don't want.
                    pip install diplomat-track[sleap, dlc, gui]

                Once installed, you can test diplomat is installed correctly by running the command below.

                .. code-block::

                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded


    .. group-tab:: Linux

        .. tabs::

            .. group-tab:: uv

                Open the terminal and run the command below:

                .. code-block::

                    uv tool install "diplomat-track[all]"

                If more granular control is needed of what parts of diplomat should be installed,
                you can mix and match the frontend specific and ui optional dependency flags, all listed
                in the command below.

                .. code-block::

                    # Equivalent to all, remove parts you don't want.
                    uv tool install "diplomat-track[sleap, dlc, gui]"

                Once the installation finishes, you can test the installation by running the command below.

                .. code-block::

                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded

            .. group-tab:: pip

                Open the terminal, with access to the python environment you would like to install diplomat in.
                Then run the command below.

                .. code-block::

                    pip install diplomat-track[all]

                If more granular control is needed of what parts of diplomat should be installed,
                you can mix and match the frontend specific and ui optional dependency flags, all listed
                in the command below.

                .. code-block::

                    # Equivalent to all, remove parts you don't want.
                    pip install diplomat-track[sleap, dlc, gui]

                Once installed, you can test diplomat is installed correctly by running the command below.

                .. code-block::

                    # Test diplomat can access the frontends it needs...
                    diplomat frontends list loaded


Legacy Project Support
^^^^^^^^^^^^^^^^^^^^^^

DIPLOMAT has support for running old DeepLabCut and SLEAP models and projects that still use tensorflow,
but the additional packages for doing so are not included by default. They can be installed using the
`legacy` and `legacy-nvidia` dependency flags. Note these only install legacy support, so you still need
to include other flags to get the UI or modern pytorch project support. Some examples of custom installation
combinations are included below:

.. code-block::

    # Install support for legacy, and new SLEAP and DLC projects (includes UI).
    uv tool install "diplomat-track[all, legacy]"
    # Install support for support for legacy, and new SLEAP and DLC projects on system with an nvidia GPU (includes UI)...
    uv tool install "diplomat-track[all, legacy-nvidia]"
    # Install support for just legacy projects, and include UI support...
    uv tool install "diplomat-track[gui, legacy]"
    # Install support for just legacy projects...
    uv tool install "diplomat-track[all, legacy]"


Development Installation Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan on developing frontends or predictors for DIPLOMAT, consider installing DIPLOMAT from source with the `developer installation method <advanced_usage.html#development-usage>`_.
