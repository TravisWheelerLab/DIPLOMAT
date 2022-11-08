Overview
========

DIPLOMAT (Deeplabcut-based Identity-Preserving Labeled-Object Multi Animal Tracking) is a software
package that provides algorithms for enhanced multi-animal tracking on top of other CNN based animal
tracking packages. Currently, DIPLOMAT only has support for running on DeepLabCut based projects,
but support for additional packages is planned.

DIPLOMAT supports two primary modes for performing animal tracking, an unsupervised mode, which
simply runs DIPLOMAT on a video and outputs a file containing tracking information per body part
and individual, and supervised mode, which displays the tracked results in a user interface, where
the user can make adjustments and rerun DIPLOMAT as many times as needed on the modified results
before saving them. DIPLOMAT also contains additional tools for visualizing, storing, and tweaking
tracking results.

To learn how to install DIPLOMAT, see the :doc:`Installation <installation>` page.

To see a deeper exploration of DIPLOMAT's capabilities, see the :doc:`Basic Usage <basic_usage>` page.
