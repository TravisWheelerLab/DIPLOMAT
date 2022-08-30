from typing import Optional
from diplomat.frontends import DIPLOMATFrontend, DIPLOMATBaselineCommands

class DEEPLABCUTFrontend(DIPLOMATFrontend):
    """
    The DEEPLABCUT frontend for DIPLOMAT.
    """
    @classmethod
    def init(cls) -> Optional[DIPLOMATBaselineCommands]:
        try:
            from diplomat.frontends.deeplabcut._verify_func import _verify_dlc_like
            from diplomat.frontends.deeplabcut.predict_videos_dlc import analyze_videos
            from diplomat.frontends.deeplabcut.predict_frames_dlc import analyze_frames
            from diplomat.frontends.deeplabcut.label_videos_dlc import label_videos
        except ImportError:
            return None

        return DIPLOMATBaselineCommands(
            _verify_analyze_videos=_verify_dlc_like,
            analyze_videos=analyze_videos,
            _verify_analyze_frames=_verify_dlc_like,
            analyze_frames=analyze_frames,
            _verify_label_videos=_verify_dlc_like,
            label_videos=label_videos
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "deeplabcut"