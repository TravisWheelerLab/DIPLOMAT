from diplomat.utils.lazy_import import LazyImporter, verify_existence_of

verify_existence_of("tensorflow")
verify_existence_of("onnx")
verify_existence_of("onnxruntime")
verify_existence_of("tf2onnx")

tf = LazyImporter("tensorflow")
ort = LazyImporter("onnxruntime")
onnx = LazyImporter("onnx")
tf2onnx = LazyImporter("tf2onnx")
