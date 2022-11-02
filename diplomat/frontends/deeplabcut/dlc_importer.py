import warnings
import builtins
import os

def _dummy_print(*args, **kwargs):
    pass

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
        import matplotlib
        import deeplabcut
        from deeplabcut import auxiliaryfunctions
        from deeplabcut.pose_estimation_tensorflow.core import predict
        from deeplabcut.pose_estimation_tensorflow.predict_videos import checkcropping
        from deeplabcut.pose_estimation_tensorflow.config import load_config
        from deeplabcut.pose_estimation_tensorflow import auxiliaryfunctions
    except ImportError:
        deeplabcut = None

    if(not debug_mode):
        builtins.print = true_print
        del os.environ["TF_CPP_MIN_LOG_LEVEL"]
