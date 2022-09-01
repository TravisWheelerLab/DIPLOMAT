import warnings
import builtins

def dummy_print(*args, **kwargs):
    pass

with warnings.catch_warnings():
    # Keep deeplabcut from flooding diplomat with warning messages and print statements...
    warnings.filterwarnings("ignore")
    true_print = builtins.print
    builtins.print = dummy_print
    try:
        import deeplabcut
        from deeplabcut import auxiliaryfunctions
        from deeplabcut.pose_estimation_tensorflow.core import predict
        from deeplabcut.pose_estimation_tensorflow.predict_videos import checkcropping
        from deeplabcut.pose_estimation_tensorflow.config import load_config
        from deeplabcut.pose_estimation_tensorflow import auxiliaryfunctions
    except ImportError:
        deeplabcut = None
    builtins.print = true_print
