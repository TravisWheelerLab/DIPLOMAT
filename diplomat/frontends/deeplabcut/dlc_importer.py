from diplomat.utils.lazy_import import LazyImporter, verify_existence_of


# This enforces dlc exists so this module can't be imported when DLC doesn't exist, but still avoids
# executing DLC's code which has a bunch of side effects...
verify_existence_of("deeplabcut")

deeplabcut = LazyImporter("deeplabcut")
tf = LazyImporter("tensorflow")
predict = LazyImporter("deeplabcut.pose_estimation_tensorflow.core.predict")
checkcropping = LazyImporter("deeplabcut.pose_estimation_tensorflow.predict_videos.checkcropping")
load_config = LazyImporter("deeplabcut.pose_estimation_tensorflow.config.load_config")
auxiliaryfunctions = LazyImporter("deeplabcut.utils.auxiliaryfunctions")
