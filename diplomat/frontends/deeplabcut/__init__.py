from typing import Optional
from diplomat.frontends import DIPLOMATFrontend, DIPLOMATBaselineCommands


class DEEPLABCUTFrontend(DIPLOMATFrontend):
    """
    The DEEPLABCUT frontend for DIPLOMAT. Contains functions for running DIPLOMAT on DEEPLABCUT projects.
    """
    @classmethod
    def init(cls) -> Optional[DIPLOMATBaselineCommands]:
        try:
            from diplomat.frontends.deeplabcut._verify_func import _verify_dlc_like
            from diplomat.frontends.deeplabcut.predict_videos_dlc import analyze_videos
            from diplomat.frontends.deeplabcut.predict_frames_dlc import analyze_frames
            from diplomat.frontends.deeplabcut.label_videos_dlc import label_videos
            from diplomat.frontends.deeplabcut.tweak_results import tweak_videos
            from diplomat.frontends.deeplabcut.convert_results_dlc import convert_results
        except ImportError:
            return None

        return DIPLOMATBaselineCommands(
            _verifier=_verify_dlc_like,
            analyze_videos=analyze_videos,
            analyze_frames=analyze_frames,
            label_videos=label_videos,
            tweak_videos=tweak_videos,
            convert_results=convert_results
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "deeplabcut"


