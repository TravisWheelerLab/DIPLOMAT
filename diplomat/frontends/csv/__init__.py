from typing import Optional
from diplomat.frontends import DIPLOMATFrontend, DIPLOMATCommands


class DEEPLABCUTFrontend(DIPLOMATFrontend):
    """
    The CSV frontend for DIPLOMAT. Contains functions for running some DIPLOMAT operations on csv trajectory files.
    Supports video creation, and tweak UI commands.
    """
    @classmethod
    def init(cls) -> Optional[DIPLOMATCommands]:
        try:
            from diplomat.frontends.csv._verify_func import _verify
            from diplomat.frontends.csv.label_videos import label_videos
            from diplomat.frontends.csv.tweak_results import tweak_videos
        except ImportError:
            return None

        return DIPLOMATCommands(
            _verifier=_verify,
            label_videos=label_videos,
            tweak_videos=tweak_videos
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "csv"


