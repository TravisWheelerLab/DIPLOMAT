from diplomat.utils.lazy_import import (
    LazyImporter,
    verify_existence_of,
)

# This enforces dlc exists so this module can't be imported when DLC doesn't exist, but still avoids
# executing DLC's code which has a bunch of side effects...
verify_existence_of("deeplabcut")

dlc_config = LazyImporter("deeplabcut.core.config")
dlc_torch = LazyImporter("deeplabcut.pose_estimation_pytorch")
