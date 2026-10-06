Basic Usage
===========

Setting up a Project
--------------------

.. tabs::

    .. group-tab:: DeepLabCut

        DIPLOMAT works on both single-animal and multi-animal DeepLabCut based projects, with no adjustments.
        So if you plan on using the DeepLabCut frontend, you'll need to first setup a DeepLabCut project. This
        can be done by following the
        `Single Animal DLC Project <https://deeplabcut.github.io/DeepLabCut/docs/quick-start/single_animal_quick_guide.html>`_
        or
        `Multi-Animal DLC Project <https://deeplabcut.github.io/DeepLabCut/docs/quick-start/tutorial_maDLC.html>`_
        guides included in the DeepLabCut documentation. Both project types can also be created and managed easily
        using the included `GUI <https://deeplabcut.github.io/DeepLabCut/docs/main-workflows/user-guide.html#gui-recommended-for-beginners>`_
        in new versions. It is recommended you create a multi-animal project as it will allow you to create a skeleton and label multiple
        individuals (DIPLOMAT will automatically pull the skeleton from DeepLabCut if one isn't manually specified when running
        the ``diplomat`` tracking commands).

        You'll want to follow these user guides until you have finished
        `training the network <https://deeplabcut.github.io/DeepLabCut/docs/quick-start/tutorial_maDLC.html#train-the-network>`_.
        Once the network is trained, you can begin using DIPLOMAT to begin tracking videos.

    .. group-tab:: SLEAP

        DIPLOMAT works with all of SLEAP's models, but it's recommended to use their bottom-up models with diplomat.
        To setup a SLEAP project, you can use SLEAP's UI. You can follow the SLEAP tutorial at
        `https://docs.sleap.ai/latest/tutorial/overview/ <https://docs.sleap.ai/latest/tutorial/overview/>`_
        all the way to and including the "Training a Model" section.



Tracking a Video
----------------

.. tabs::

    .. group-tab:: DeepLabCut

        To run DIPLOMAT on a video, simply run either the :py:cli:`diplomat track_and_interact`
        or :py:cli:`diplomat track` command, as shown below (you'll need to replace the paths below
        with actual paths to the project config file and the video):

        .. code-block:: sh

            # Run diplomat with UI intervention...
            diplomat track_and_interact -c path/to/dlc/project/config.yaml/or/zipped/dlc/project -v path/to/video/to/run/on.mp4
            # Run without the UI...
            diplomat track -c path/to/dlc/project/config.yaml/or/zipped/dlc/project -v path/to/video/to/run/on.mp4

        These commands will produce a .csv and .dipui file after tracking is done. Both of these commands support running
        on a list videos by simply passing paths using a comma separated list encased in square brackets:

        .. code-block:: sh

            diplomat track_and_interact -c path/to/dlc/project/config.yaml/or/zipped/dlc/project -v [path/to/video1.mp4, path/to/video2.webm, path/to/video3.mkv]

        They also support working on more or less individuals by specifying the ``--num_outputs`` or ``-no`` flag:

        .. code-block:: sh

            # Video has 2 individuals.
            diplomat track_and_interact -c path/to/dlc/project/config.yaml/or/zipped/dlc/project -no 2 -v path/to/video1.mp4
            # Video has 5 individuals.
            diplomat track_and_interact -c path/to/dlc/project/config.yaml/or/zipped/dlc/project -no 5 -v path/to/video2.mp4

    .. group-tab:: SLEAP

        To run DIPLOMAT on a video, simply run either the :py:cli:`diplomat track_and_interact`
        or :py:cli:`diplomat track` command, as shown below (you'll need to replace the paths below
        with actual paths to the SLEAP model folder or zip file and the video):

        .. code-block:: sh

            # Run diplomat with UI intervention...
            diplomat track_and_interact -c path/to/sleap/model/folder/or/zip -v path/to/video/to/run/on.mp4 -no <num_bodies>
            # Run without the UI...
            diplomat track -c path/to/sleap/model/folder/or/zip -v path/to/video/to/run/on.mp4 -no <num_bodies>

        Models are typically placed in a folder called "models" placed next to the .slp file for your SLEAP project. Both of the above commands will
        produce a .csv and .dipui file once tracking is done. Both of these commands support running on a list videos by simply passing paths
        using a comma separated list:

        .. code-block:: sh

            diplomat track_and_interact -c path/to/sleap/model/folder/or/zip -v [path/to/video1.mp4, path/to/video2.webm, path/to/video3.mkv] -no <num_bodies>

        The above commands also support working on more or less individuals by specifying the ``--num_outputs`` or ``-no`` flag, just like for DeepLabCut.
        Unlike DeepLabCut, this argument is required.

        .. code-block:: sh

            # Video has 2 individuals.
            diplomat track_and_interact -c path/to/sleap/model/folder/or/zip -no 2 -v path/to/video1.mp4
            # Video has 5 individuals.
            diplomat track_and_interact -c path/to/sleap/model/folder/or/zip -no 5 -v path/to/video2.mp4


Producing a Labeled Video
-------------------------

Once tracking is done, one can produce a labeled video using the :py:cli:`diplomat annotate`
command and passing video/csv pairs to it, as shown below:

.. code-block:: sh

    diplomat annotate -v path/to/video.mp4 -c path/to/video-track-results.csv

This will cause DIPLOMAT to produce another video placed next to the original video with
``_labeled`` tacked on to the end of its name. Solid markers indicate a tracked and detected part,
hollow markers indicate the part is occluded or hidden.

.. note::

    This command supports the following additonal final trace formats on top of DIPLOMAT csv files:
     - DeepLabCut .csv
     - DeepLabCut .h5
     - SLEAP .h5

Restoring UI to Make Major Adjustments
--------------------------------------

DIPLOMAT, when run in interactive or non-interactive mode with the `"storage_mode"` set to `"disk"` or `"hybrid"` (`"hybrid"` is
the default setting), will save the video, all run session info, and frame data to a `.dipui`
file. If the DIPLOMAT UI either crashes or you would like to edit your saved results in the
feature complete version of the UI, you can restore the UI state using the :py:cli:`diplomat interact`
command, as shown below:

.. code-block:: sh

    diplomat interact -s path/to/ui/state/file.dipui


Making Minor Tweaks to Results
------------------------------

DIPLOMAT provides a stripped down version of the UI editor, which allows you to make minor
modifications to results and also view results after tracking has already been done.
This can be done passing video/csv pairs :py:cli:`diplomat tweak` command.

.. code-block:: sh

    diplomat tweak -v path/to/video.mp4 -c path/to/video-track-results.csv

.. note::

    This command supports the following additonal final trace formats on top of DIPLOMAT csv files:
     - DeepLabCut .csv
     - DeepLabCut .h5
     - SLEAP .h5

Saving Model Outputs for Later Analysis (All Frontends)
-------------------------------------------------------

DIPLOMAT is capable of grabbing model outputs (confidence maps and location references) and
dumping them to a file, which can improve performance when analyzing the same video multiple
times or allow analysis to be completed somewhere else on a machine that lacks a GPU. To create
a frame store for later analysis, run tracking with the frame store exporting predictor:

.. code-block:: sh

    diplomat track_with -c path/to/config -v path/to/video -p FrameExporter -no 1

The above command will generate a .dlfs file next to the video. To run tracking on it, run one of
DIPLOMAT's tracking methods, but with the ``-fs`` flag passing in the frame store(s) instead of the video.
Also, the project config is not needed when running on frame stores.

.. code-block:: sh

    # Run DIPLOMAT with no UI...
    diplomat track -fs path/to/fstore.dlfs -no <num_bodies>
    # Run DIPLOMAT with UI...
    diplomat track_and_interact -fs path/to/fstore.dlfs -no <num_bodies>
    # Run DIPLOMAT with some other prediction algorithm
    diplomat track_with -fs path/to/fstore.dlfs -p NameOfPredictorPlugin -no <num_bodies>

Saving DIPLOMAT Run Settings
----------------------------

DIPLOMAT does not have a global or project-like configuration file, but instead provides a special command,
:py:cli:`diplomat yaml`, that allows for running any DIPLOMAT CLI command with a custom configuration by specifying the options in a YAML file.

To use it, simply create a yaml file, similar to the one below. The yaml below runs the :py:cli:`diplomat track`
command with some of the settings modified for the main tracking algorithm.

.. code-block:: yaml

    # The name of the sub-command to run.
    command: track
    # The arguments to pass to the command. Note you can't use abbreviations here, but instead you need to use the full setting names.
    arguments:
      # Change this to your actual project path...
      config: path/to/project/model/or/config
      num_outputs: 2
      output_suffix: "boost-occluded"
      segmented_passes:
        - - MITViterbi
          - occluded_transition_probability: 0.75
            lowest_value: 1e-8
            obscured_survival_max: 100

Then, you can run it using:

.. code-block:: sh

    diplomat yaml name_of_yaml_file.yaml --video path/to/video.mp4


Notice, since we don't include a video path in the yaml file, we include it as an extra argument to the
:py:cli:`diplomat yaml` command. :py:cli:`diplomat yaml` will automatically detect and indicate when required
arguments to the command you are attempting to run haven't been specified. You can also override arguments
in the yaml file by passing a new value to the command, as shown below. Note that you must use the full name
for the option, abbreviations are not supported.

.. code-block:: sh

    # Using '-no' does not work...
    diplomat yaml name_of_yaml_file.yaml --video path/to/video.mp4 --num_outputs 3


Video Utilities
---------------

The :py:cli:`diplomat split_videos` command provides functionality for both trimming and splitting
videos into segments. It allows for splitting a video into fixed length segments or at exact
second based offsets, as shown below:

.. code-block:: sh

    # Split a video into 2 minute (120 second) chunks (-sps stands for seconds per segment).
    diplomat split_videos -v path/to/video.mp4 -sps 120

    # Split a video at exactly 30.25, 70.001, and 500 seconds in.
    diplomat split_videos -v path/to/video.mp4 -sps [30.25, 70.001, 500]

    # Like all other commands, multiple videos can be passed.
    diplomat split_videos -v [path/to/video1.mov, path/to/video2.avi] -sps 120

    # Can specify an alternative output format via fourcc code and file extension...
    diplomat split_videos -v path/to/video1.mov -sps 120 -ofs mp4v -oe .mp4
