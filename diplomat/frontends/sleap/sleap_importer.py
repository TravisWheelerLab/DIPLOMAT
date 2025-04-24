from diplomat.utils.lazy_import import LazyImporter, verify_existence_of

verify_existence_of("tensorflow")
verify_existence_of("onnx")
verify_existence_of("onnxruntime")

tf = LazyImporter("tensorflow")