"""
This module defines the abstract base class for predictor plugins, and additional data structures, classes, and functions used for
processing network outputs into body part pose predictions.
"""
# Used for type hints
from typing import Type, Set
# Used by get_predictor for loading plugins
from diplomat.utils import pluginloader
from diplomat import predictors

# Imports for other stuff in this module...
from diplomat.processing.predictor import Predictor, TestFunction
from diplomat.processing.track_data import TrackingData
from diplomat.processing.progress_bar import ProgressBar, TQDMProgressBar
from diplomat.processing.pose import Pose
from diplomat.processing import type_casters
from diplomat.processing.type_casters import TypeCaster
from diplomat.processing.containers import Config, ConfigSpec

__all__ = [
    "Predictor",
    "TrackingData",
    "ProgressBar",
    "TQDMProgressBar",
    "Pose",
    "type_casters",
    "TypeCaster",
    "Config",
    "ConfigSpec",
    "TestFunction",
    "get_predictor",
    "get_predictor_plugins"
]


def get_predictor(name: str) -> Type[Predictor]:
    """
    Get the predictor plugin by the specified name.

    :param name: The name of this plugin, should be a string

    :returns: The plugin class that has a name that matches the specified name
    """
    # Load the plugins from the directory: "deeplabcut/pose_estimation_tensorflow/nnet/predictors"
    plugins = get_predictor_plugins()
    # Iterate the plugins until we find one with a matching name, otherwise throw a ValueError if we don't find one.
    for plugin in plugins:
        if plugin.get_name() == name:
            return plugin
    else:
        raise ValueError(
            f"Predictor plugin {name} does not exist, try another plugin name..."
        )


def get_predictor_plugins() -> Set[Type[Predictor]]:
    """
    Get and retrieve all predictor plugins currently available to the DeepLabCut implementation.

    :returns: A Set of Predictors, being the all classes that extend the Predictor class currently loaded visible to the python interpreter.
    """
    return pluginloader.load_plugin_classes(predictors, Predictor)
