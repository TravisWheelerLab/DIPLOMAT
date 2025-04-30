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
            from diplomat.frontends.deeplabcut.load_model import load_model
        except ImportError:
            return None

        return DIPLOMATCommands(
            _verifier=_verify_dlc_like,
            _load_model=load_model
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "deeplabcut"


