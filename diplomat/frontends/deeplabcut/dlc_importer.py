from diplomat.utils.lazy_import import SilentImports, verify_existence_of

# This enforces dlc exists so this module can't be imported when DLC doesn't exist, but still avoids
# executing DLC's code which has a bunch of side effects...
verify_existence_of("tensorflow")
verify_existence_of("tf2onnx")
verify_existence_of("onnx")
verify_existence_of("onnxruntime")

with SilentImports():
    import tensorflow as tf
    import onnxruntime as ort
    import onnx
    import tf2onnx
