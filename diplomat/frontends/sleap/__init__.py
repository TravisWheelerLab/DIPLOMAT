from typing import Optional
from diplomat.frontends import DIPLOMATFrontend, DIPLOMATBaselineCommands


class SLEAPFrontend(DIPLOMATFrontend):
    """
    The SLEAP frontend for DIPLOMAT. Contains functions for running DIPLOMAT on SLEAP projects.
    """
    @classmethod
    def init(cls) -> Optional[DIPLOMATBaselineCommands]:
        try:
            from diplomat.frontends.sleap._verify_func import _verify_sleap_like
            from diplomat.frontends.sleap.predict_videos_sleap import analyze_videos
            from diplomat.frontends.sleap.predict_frames_sleap import analyze_frames
            from diplomat.frontends.sleap.label_videos_sleap import label_videos
            from diplomat.frontends.sleap.tweak_results_sleap import tweak_videos
        except ImportError:
            return None

        return DIPLOMATBaselineCommands(
            _verifier=_verify_sleap_like,
            analyze_videos=analyze_videos,
            analyze_frames=analyze_frames,
            label_videos=label_videos,
            tweak_videos=tweak_videos
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "sleap"


