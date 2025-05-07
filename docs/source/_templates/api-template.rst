DIPLOMAT API
============

DIPLOMAT includes the following python interfaces and plugins below.

Core Functions
--------------

Python functions that are equivalents for DIPLOMAT's command line interface.

.. autosummary::
    :toctree: _autosummary
    :recursive:

{diplomat.core}

Predictors
----------

A list of the :class:`~diplomat.processing.Predictor` plugins included with DIPLOMAT by default.
Predictors predict the exact locations of objects given probabilistic model outputs.

.. toctree::
    :hidden:

{diplomat.files.predictors}


.. list-table::
    :widths: auto

{diplomat.predictors}


.. _Frame Passes:

Frame Passes
------------

A list of the ``FramePass`` plugins included by default
with DIPLOMAT. Some :class:`~diplomat.processing.Predictor` plugins use frame passes to perform
pose prediction, including the :plugin:`~diplomat.predictors.SegmentedFramePassEngine` and
:plugin:`~diplomat.predictors.SupervisedSegmentedFramePassEngine` based predictors.

.. toctree::
    :hidden:

{diplomat.files.frame_passes}


.. list-table::
    :widths: auto

{diplomat.frame_passes}

Processing Module
-----------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:

    diplomat.processing

Utilities
---------

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:

    diplomat.utils


Frontends
---------

A list of frontends that come included with DIPLOMAT. Frontends enable DIPLOMAT to run
off of model results produced by a specific tracking software package and project.

.. toctree::
    :hidden:

{diplomat.files.frontends}


.. list-table::
    :widths: auto

{diplomat.frontends}


WX GUI Components
-----------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:

    diplomat.wx_gui