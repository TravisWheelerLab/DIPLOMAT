from diplomat.utils.lazy_import import LazyImporter, verify_existence_of

verify_existence_of("sleap")


sleap = LazyImporter("sleap")
tf = LazyImporter("tensorflow")

SleapDataConfig = LazyImporter("sleap.nn.config.DataConfig")
SleapVideo = LazyImporter("sleap.Video")
Provider = LazyImporter("sleap.nn.data.pipelines.Provider")
SleapVideoReader = LazyImporter("sleap.nn.data.providers.VideoReader")
SleapPredictor = LazyImporter("sleap.nn.inference.Predictor")
SleapInferenceLayer = LazyImporter("sleap.nn.inference.InferenceLayer")
SleapSkeleton = LazyImporter("sleap.skeleton.Skeleton")