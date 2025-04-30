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
            from diplomat.frontends.sleap.load_model import load_models
        except ImportError:
            return None

        return DIPLOMATCommands(
            _verifier=_verify_sleap_like,
            _load_model=load_models
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "sleap"


