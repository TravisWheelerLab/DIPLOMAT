from diplomat.frontends import DIPLOMATCommands, DIPLOMATFrontend


class DEEPLABCUTLegacyFrontend(DIPLOMATFrontend):
    """
    The DEEPLABCUT legacy frontend for DIPLOMAT. Contains functions for running DIPLOMAT on DEEPLABCUT tensorflow projects.
    """

    @classmethod
    def init(cls) -> DIPLOMATCommands:
        from diplomat.frontends.deeplabcut_legacy._verify_func import _verify_dlc_like
        from diplomat.frontends.deeplabcut_legacy.convert_tracks import (
            _dlc_hdf_to_diplomat_table,
        )
        from diplomat.frontends.deeplabcut_legacy.load_model import load_model

        return DIPLOMATCommands(
            _verifier=_verify_dlc_like,
            _load_model=load_model,
            _load_tracks=_dlc_hdf_to_diplomat_table,
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "deeplabcut_legacy"
