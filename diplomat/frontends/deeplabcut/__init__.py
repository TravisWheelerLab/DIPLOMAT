from typing import Optional
from diplomat.frontends import DIPLOMATFrontend, DIPLOMATBaselineCommands

class DEEPLABCUTFrontend(DIPLOMATFrontend):
    @classmethod
    def init(cls) -> Optional[DIPLOMATBaselineCommands]:
        try:
            from diplomat.frontends.deeplabcut.predict_videos_dlc import analyze_videos
            from diplomat.frontends.deeplabcut.predict_frames_dlc import analyze_frame_store
            from diplomat.frontends.deeplabcut.label_videos_dlc import create_labeled_videos
        except ImportError:
            return None

        return DIPLOMATBaselineCommands(
            analyze_video=analyze_videos,
            analyze_frames=analyze_frame_store,
            label_video=create_labeled_videos
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "deeplabcut"