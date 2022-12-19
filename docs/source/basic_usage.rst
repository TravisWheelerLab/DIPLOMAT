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


Using DIPLOMAT with SLEAP Projects
----------------------------------

Setting up a Project
^^^^^^^^^^^^^^^^^^^^

DIPLOMAT works with all of SLEAP's models, with the exception of SLEAP's top-down based
models. To setup a SLEAP project, one can simply use SLEAP's UI to create a project
and label frames. To setup a SLEAP project, you can follow the SLEAP tutorial at
`https://sleap.ai/tutorials/tutorial.html <https://sleap.ai/tutorials/tutorial.html>`_
all the way through the "Start Training" section.

Tracking a Video
^^^^^^^^^^^^^^^^

To run DIPLOMAT on a video, simply pass run either the :py:cli:`diplomat supervised`
or :py:cli:`diplomat unsupervised` command, as shown below (you'll need to replace the paths below
with actual paths to the project config file and the video):

.. code-block:: sh

    # Run diplomat with UI intervention...
    diplomat supervised -c path/to/sleap/model/folder/or/zip -v path/to/video/to/run/on.mp4
    # Run without the UI...
    diplomat unsupervised -c path/to/sleap/model/folder/or/zip -v path/to/video/to/run/on.mp4

Model paths are typically placed in a folder called "models" placed next to the .slp file for your SLEAP project. Both of the above commands will
produce a ".slp" file with a prefix matching the name video. Both of these commands support running on a list videos by simply passing paths
using a comma separated list:

.. code-block:: sh

    diplomat supervised -c path/to/sleap/model/folder/or/zip -v [path/to/video1.mp4, path/to/video2.webm, path/to/video3.mkv]

The above commands also support working on more or less individuals by specifying the ``--num_outputs`` or ``-no`` flag, just like for DeepLabCut.

.. code-block:: sh

    # Video has 2 individuals.
    diplomat supervised -c path/to/sleap/model/folder/or/zip -no 2 -v path/to/video1.mp4
    # Video has 5 individuals.
    diplomat supervised -c path/to/sleap/model/folder/or/zip -no 5 -v path/to/video2.mp4


Producing a Labeled Video
^^^^^^^^^^^^^^^^^^^^^^^^^

Once tracking is done, one can produce a labeled video using the :py:cli:`diplomat annotate`
command and passing a video to it, as shown below:

.. code-block:: sh

    diplomat annotate -c path/to/sleap/model/folder/or/zip -v path/to/final/labels.slp

Notice that the video parameter (-v flag) does not accept a list of videos, but rather a list of
SLEAP files generated by one of DIPLOMAT's tracking commands (:py:cli:`diplomat track`,
:py:cli:`diplomat supervised`, or :py:cli:`diplomat unsupervised`).
This will cause DIPLOMAT to produce video placed next to the labels with the same name. Solid markers indicate a tracked and detected part,
hollow markers indicate the part is occluded or hidden.

Making Minor Tweaks to Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DIPLOMAT provides a stripped down version of the UI editor, which allows you to make minor
modifications to results and also view results after tracking has already been done.
This can be done using the :py:cli:`diplomat tweak` command.

.. code-block:: sh

    # NOTICE: Does not take videos, but a list of output labels for SLEAP...
    diplomat tweak -c path/to/dlc/project/config.yaml -v path/to/final/labels.slp


Saving Model Outputs for Later Analysis (All Frontends)
-------------------------------------------------------

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