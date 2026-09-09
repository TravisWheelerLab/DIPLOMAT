from diplomat.utils.lazy_import import LazyImporter, verify_existence_of

verify_existence_of("torch")
verify_existence_of("sleap_nn")
verify_existence_of("sleap_io")

torch = LazyImporter("torch")
sleap_nn = LazyImporter("sleap_nn")
sleap_io = LazyImporter("sleap_io")
