Basic Usage of DIPLOMAT
=======================

.. contents:: Contents

Using DIPLOMAT with DeepLabCut Projects
---------------------------------------

Setting up a Project
^^^^^^^^^^^^^^^^^^^^

DIPLOMAT works on both single-animal and multi-animal DeepLabCut based projects, with no adjustments.
So if you plan on using the DeepLabCut frontend, you'll need to first setup a DeepLabCut project. This
can be done by following the
`Single Animal DLC Project <https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html>`_
or
`Multi-Animal DLC Project <https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html>`_
guides included in the DeepLabCut documentation. It is recommended you follow the multi-animal
guide as it will allow you to create a skeleton and label multiple individuals (DIPLOMAT will
automatically pull the skeleton from DeepLabCut if one isn't manually specified when running
the ``diplomat`` tracking commands).

You'll want to follow these user guides until you have finished
`training the network <https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#train-the-network>`_.
Once the network is trained, you can begin using DIPLOMAT to begin tracking videos.

Tracking a Video
^^^^^^^^^^^^^^^^

To run DIPLOMAT on a video, simply pass run either the :py:cli:`diplomat supervised`
or :py:cli:`diplomat unsupervised` command, as shown below (you'll need to replace the paths below
with actual paths to the project config file and the video):

.. code-block:: sh

    # Run diplomat with UI intervention...
    diplomat supervised -c path/to/dlc/project/config.yaml -v path/to/video/to/run/on.mp4
    # Run without the UI...
    diplomat unsupervised -c path/to/dlc/project/config.yaml -v path/to/video/to/run/on.mp4

Both of these commands support running on a list videos by simply passing paths using a comma
separated list:

.. code-block:: sh

    diplomat supervised -c path/to/dlc/project/config.yaml -v [path/to/video1.mp4, path/to/video2.webm, path/to/video3.mkv]

They also support working on more or less individuals by specifying the ``--num_outputs`` or ``-no`` flag:

.. code-block:: sh

    # Video has 2 individuals.
    diplomat supervised -c path/to/dlc/project/config.yaml -no 2 -v path/to/video1.mp4
    # Video has 5 individuals.
    diplomat supervised -c path/to/dlc/project/config.yaml -no 5 -v path/to/video2.mp4


Producing a Labeled Video
^^^^^^^^^^^^^^^^^^^^^^^^^

Once tracking is done, one can produce a labeled video using the :py:cli:`diplomat annotate`
command and passing a video to it, as shown below:

.. code-block:: sh

    diplomat annotate -c path/to/dlc/project/config.yaml -v path/to/video.mp4

This will cause DIPLOMAT to produce another video placed next to the original video with
``_labeled`` tacked on to the end of its name. Solid markers indicate a tracked and detected part,
hollow markers indicate the part is occluded or hidden.

Making Minor Tweaks to Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DIPLOMAT provides a stripped down version of the UI editor, which allows you to make minor
modifications to results and also view results after tracking has already been done.
This can be done using the :py:cli:`diplomat tweak` command.

.. code-block:: sh

    diplomat tweak -c path/to/dlc/project/config.yaml -v path/to/video.mp4


Saving Model Outputs for Later Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DIPLOMAT is capable of grabbing model outputs (confidence maps and location references) and
dumping them to a file, which can improve performance when analyzing the same video multiple
times or allow analysis to be completed somewhere else on a machine that lacks a GPU. To create
a frame store for later analysis, run tracking with the frame store exporting predictor:

.. code-block:: sh

    diplomat track -c path/to/config -v path/to/video -p FrameExporter

The above command will generate a .dlfs file next to the video. To run tracking on it, run one of
DIPLOMAT's tracking methods, but with the ``-fs`` flag passing in the frame store(s) instead of the video.

.. code-block:: sh

    # Run DIPLOMAT with no UI...
    diplomat unsupervised -c path/to/config -fs path/to/fstore.dlfs
    # Run DIPLOMAT with UI...
    diplomat supervised -c path/to/config -fs path/to/fstore.dlfs
    # Run DIPLOMAT with some other prediction algorithm
    diplomat track -c path/to/config -fs path/to/fstore.dlfs -p NameOfPredictorPlugin

Video Utilities
---------------

The :py:cli:`diplomat split_videos` command provides functionality for both trimming and splitting
videos into segments. It allows for splitting the video into fixed length segments or at exact
second based offsets, as shown below:

.. code-block:: sh

    # Split a video into 2 minute (120 second) chunks (-sps stands for seconds per segment).
    diplomat split_videos -v path/to/video.mp4 -sps 120

    # Split a video at exactly 30, 70, and 500 seconds in.
    diplomat split_videos -v path/to/video.mp4 -sps [30, 70, 500]

    # Like all other commands, multiple videos can be passed.
    diplomat split_videos -v [path/to/video1.mov, path/to/video2.avi] -sps 120

    # Can specify an alternative output format via fourcc code and file extension...
    diplomat split_videos -v path/to/video1.mov -sps 120 -ofs mp4v -oe .mp4


Inspecting DIPLOMAT Plugins
---------------------------

DIPLOMAT was developed with extensibility in mind, so core functionality can be extended via
plugins. DIPLOMAT has two kinds of plugins:

 - Predictors: Plugins that take in model outputs and predict poses, or animal locations from them. Some of these also have additional side effects such as plotting or frame export.
 - Frontends: These are plugins that grab frames from another tracking software and pipe them into the predictor the user has selected. Currently, there is only one for `DeepLabCut <https://github.com/DeepLabCut/DeepLabCut>`_.

To get information about predictors, one can use the commands of diplomat predictors:

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

DIPLOMAT is written entirely in python. To set up an environment for developing DIPLOMAT, you
can simply pull down this repository and install its requirements.txt dependencies to your
virtual environment.

.. code-block:: sh

    git clone https://github.com/TravisWheelerLab/DIPLOMAT.git
    cd DIPLOMAT
    pip install -r requirements.txt

For most development, you'll most likely want to add additional predictor plugins.
Predictors can be found in diplomat/predictors. Classes that extend Predictor are automatically
loaded from this directory. To test predictors, you can use the :py:cli:`diplomat predictors test` command:

.. code-block:: sh

    diplomat predictors test PredictorName

