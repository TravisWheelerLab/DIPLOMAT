from diplomat.utils.lazy_import import verify_existence_of, SilentImports

verify_existence_of("tensorflow")
verify_existence_of("onnx")
verify_existence_of("onnxruntime")
verify_existence_of("tf2onnx")

with SilentImports():
    import tensorflow as tf
    import onnxruntime as ort
    import onnx
    import tf2onnx