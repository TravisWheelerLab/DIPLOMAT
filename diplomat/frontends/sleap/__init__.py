from diplomat.frontends import DIPLOMATCommands, DIPLOMATFrontend


class SLEAPFrontend(DIPLOMATFrontend):
    """
    The SLEAP frontend for DIPLOMAT. Contains functions for running DIPLOMAT on SLEAP projects.
    """

    @classmethod
    def init(cls) -> DIPLOMATCommands:
        from diplomat.frontends.sleap._verify_func import _verify_sleap_like
        from diplomat.frontends.sleap.convert_tracks import (
            _sleap_nn_tracks_to_diplomat_table,
        )
        from diplomat.frontends.sleap.load_model import load_models

        return DIPLOMATCommands(
            _verifier=_verify_sleap_like,
            _load_model=load_models,
            _load_tracks=_sleap_nn_tracks_to_diplomat_table,
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "sleap"
