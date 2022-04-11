# For types in methods
from typing import Optional
# Plugin base class
from diplomat.processing import *


class ArgMax(Predictor):
    """
    Default processor for DeepLabCut, and the code originally used by DeepLabCut for prediction of points. Predicts
    the point from the probability frames simply by selecting the max probability in each frame.
    """
    def on_frames(self, scmap: TrackingData) -> Optional[Pose]:
        # Using new object library to get the max... Drastically simplified logic...
        return scmap.get_poses_for(
            scmap.get_max_scmap_points(num_max=self.num_outputs)
        )

    def on_end(self, pbar) -> Optional[Pose]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True
