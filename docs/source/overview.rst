Overview
========

DIPLOMAT (Deep learning-based Identity-Preserving Labeled-Object Multi Animal Tracking) is a software
package that provides algorithms for enhanced multi-animal tracking on top of other CNN based animal
tracking packages. Currently, DIPLOMAT has support for running on DeepLabCut and SLEAP based projects.
Unlike other multi-animal tracking packages, DIPLOMAT's algorithms work directly off confidence maps
instead of running peak detection, allowing for more nuanced tracking results compared to other methods.

.. list-table::
    :widths: 50 50
    :align: center

    * - .. image:: /_static/imgs/example1.png
            :align: center

      - .. image:: /_static/imgs/example2.png
            :align: center

DIPLOMAT supports two primary modes for performing animal tracking, an unsupervised mode, which
simply runs DIPLOMAT on a video and outputs a file containing tracking information per body part
and individual, and supervised mode, which displays the tracked results in a user interface, where
the user can make adjustments and rerun DIPLOMAT as many times as needed on the modified results
before saving them. DIPLOMAT also contains additional tools for visualizing, storing, and tweaking
tracking results.

.. figure:: /_static/imgs/UIDemo.png
    :align: center
    :alt: A gif demonstrating applying corrections in DIPLOMAT's supervised interface.

    Example of using DIPLOMAT's supervised interface. Minor corrections can be made, and then tracking algorithms rerun to produce better results.


To learn how to install DIPLOMAT, see the :doc:`Installation <installation>` page.

To see a deeper exploration of DIPLOMAT's capabilities, see the :doc:`Basic Usage <basic_usage>` page.
