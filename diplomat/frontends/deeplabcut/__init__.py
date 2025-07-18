from typing import Optional
from diplomat.frontends import DIPLOMATFrontend, DIPLOMATCommands


class DEEPLABCUTFrontend(DIPLOMATFrontend):
    """
    The DEEPLABCUT frontend for DIPLOMAT. Contains functions for running DIPLOMAT on DEEPLABCUT projects.
    """

    @classmethod
    def init(cls) -> Optional[DIPLOMATCommands]:
        try:
            from diplomat.frontends.deeplabcut._verify_func import _verify_dlc_like
            from diplomat.frontends.deeplabcut.load_model import load_model
            from diplomat.frontends.deeplabcut.convert_tracks import (
                _dlc_hdf_to_diplomat_table,
            )
        except ImportError:
            return None

        return DIPLOMATCommands(
            _verifier=_verify_dlc_like,
            _load_model=load_model,
            _load_tracks=_dlc_hdf_to_diplomat_table,
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "deeplabcut"
