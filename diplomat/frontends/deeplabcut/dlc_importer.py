from diplomat.utils.lazy_import import (
    LazyImporter,
    verify_existence_of,
    onnx_preload_import,
)

# This enforces dlc exists so this module can't be imported when DLC doesn't exist, but still avoids
# executing DLC's code which has a bunch of side effects...
verify_existence_of("tensorflow")
verify_existence_of("tf2onnx")
verify_existence_of("onnx")
verify_existence_of("onnxruntime")

tf = LazyImporter("tensorflow")
ort = LazyImporter("onnxruntime", import_function=onnx_preload_import)
onnx = LazyImporter("onnx")
tf2onnx = LazyImporter("tf2onnx")
