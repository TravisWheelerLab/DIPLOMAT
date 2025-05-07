from diplomat.utils.lazy_import import verify_existence_of, LazyImporter, onnx_preload_import

verify_existence_of("tensorflow")
verify_existence_of("onnx")
verify_existence_of("onnxruntime")
verify_existence_of("tf2onnx")
verify_existence_of("h5py")

tf = LazyImporter("tensorflow")
ort = LazyImporter("onnxruntime", import_function=onnx_preload_import)
onnx = LazyImporter("onnx")
tf2onnx = LazyImporter("tf2onnx")
h5py = LazyImporter("h5py")
