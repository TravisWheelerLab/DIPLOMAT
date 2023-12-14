from typing import Optional
from diplomat.frontends import DIPLOMATFrontend, DIPLOMATCommands


class SLEAPFrontend(DIPLOMATFrontend):
    """
    The SLEAP frontend for DIPLOMAT. Contains functions for running DIPLOMAT on SLEAP projects.
    """
    @classmethod
    def init(cls) -> Optional[DIPLOMATCommands]:
        try:
            from diplomat.frontends.sleap._verify_func import _verify_sleap_like
            from diplomat.frontends.sleap.predict_videos_sleap import analyze_videos
            from diplomat.frontends.sleap.predict_frames_sleap import analyze_frames
            from diplomat.frontends.sleap.label_videos_sleap import label_videos
            from diplomat.frontends.sleap.tweak_results_sleap import tweak_videos
            from diplomat.frontends.sleap.convert_results_sleap import convert_results
            from diplomat.frontends.sleap.save_from_restore import _save_from_restore
        except ImportError:
            return None

        return DIPLOMATCommands(
            _verifier=_verify_sleap_like,
            _save_from_restore=_save_from_restore,
            analyze_videos=analyze_videos,
            analyze_frames=analyze_frames,
            label_videos=label_videos,
            tweak_videos=tweak_videos,
            convert_results=convert_results
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "sleap"


