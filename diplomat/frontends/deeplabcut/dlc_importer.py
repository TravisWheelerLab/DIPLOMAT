import warnings
import builtins
import os
from importlib import import_module


def _dummy_print(*args, **kwargs):
    pass


def _silent_import(name, pkg=None):
    with warnings.catch_warnings():
        # Keep deeplabcut from flooding diplomat with warning messages and print statements...
        debug_mode = os.environ.get("DIPLOMAT_DEBUG", False)

        if(not debug_mode):
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')

            warnings.filterwarnings("ignore")
            true_print = builtins.print
            builtins.print = _dummy_print
        try:
            return import_module(name, pkg)
        finally:
            if(not debug_mode):
                builtins.print = true_print
                del os.environ["TF_CPP_MIN_LOG_LEVEL"]


class LazyImporter:

    NOTHING = object()

    def __init__(self, name, pkg=None):
        self._name = name
        self._pkg = pkg
        self._mod = self.NOTHING

    def __getattr__(self, item):
        return type(self)(f"{self._name}.{item}")

    def __call__(self, *args, **kwargs):
        if(self._mod is self.NOTHING):
            self._mod = _silent_import(self._name, self._pkg)

        return self._mod(*args, **kwargs)


deeplabcut = LazyImporter("deeplabcut")
predict = LazyImporter("deeplabcut.pose_estimation_tensorflow.core.predict")
checkcropping = LazyImporter("deeplabcut.pose_estimation_tensorflow.predict_videos.checkcropping")
load_config = LazyImporter("deeplabcut.pose_estimation_tensorflow.config.load_config")
auxiliaryfunctions = LazyImporter("deeplabcut.pose_estimation_tensorflow.auxiliaryfunctions")

