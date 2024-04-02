from typing import Optional
from diplomat.frontends import DIPLOMATFrontend, DIPLOMATCommands
from diplomat.utils import colormaps

class DEEPLABCUTFrontend(DIPLOMATFrontend):
    """
    The DEEPLABCUT frontend for DIPLOMAT. Contains functions for running DIPLOMAT on DEEPLABCUT projects.
    """
    @classmethod
    def init(cls) -> Optional[DIPLOMATCommands]:
        try:
            from diplomat.frontends.deeplabcut._verify_func import _verify_dlc_like
            from diplomat.frontends.deeplabcut.predict_videos_dlc import analyze_videos
            from diplomat.frontends.deeplabcut.predict_frames_dlc import analyze_frames
            from diplomat.frontends.deeplabcut.label_videos_dlc import label_videos
            from diplomat.frontends.deeplabcut.tweak_results import tweak_videos
            from diplomat.frontends.deeplabcut.convert_results_dlc import convert_results
            from diplomat.frontends.deeplabcut.save_from_restore import _save_from_restore
        except ImportError:
            return None

        return DIPLOMATCommands(
            _verifier=_verify_dlc_like,
            _save_from_restore=_save_from_restore,
            analyze_videos=analyze_videos,
            analyze_frames=analyze_frames,
            label_videos=label_videos,
            tweak_videos=tweak_videos,
            convert_results=convert_results
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "deeplabcut"


