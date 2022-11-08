=================
DIPLOMAT Commands
=================

DIPLOMAT supports the following list of subcommands when running it's CLI.

Tracking and Analysis Commands
==============================

.. toctree::
    :hidden:

{entries.files.track_commands}

.. list-table::
    :widths: auto

{entries.track_commands}

Development Commands
====================

.. toctree::
    :hidden:

{entries.files.dev_commands}

.. list-table::
    :widths: auto

{entries.dev_commands}

Frontend Specific Commands (Advanced)
=====================================

These are the frontend specific implementations of DIPLOMAT's core functionalities.
When you run :cli:`diplomat track`, :cli:`diplomat tweak`, :cli:`diplomat annotate`,
:cli:`diplomat supervised`, or :cli:`diplomat unsupervised`, you are calling these functions,
as those commands determine the correct implementation based on the config file you pass.
Therefore, you will likely almost never call these functions directly, but they are included as
they list all settings that a backend supports.

.. toctree::
    :hidden:

{entries.files.frontend_commands}

.. list-table::
    :widths: auto

{entries.frontend_commands}
