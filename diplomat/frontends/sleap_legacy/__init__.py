from diplomat.frontends import DIPLOMATCommands, DIPLOMATFrontend


class SLEAPLegacyFrontend(DIPLOMATFrontend):
    """
    The SLEAP Legacy frontend for DIPLOMAT. Contains functions for running DIPLOMAT on SLEAP Legacy projects.
    """

    @classmethod
    def init(cls) -> DIPLOMATCommands:
        from diplomat.frontends.sleap_legacy._verify_func import _verify_sleap_like
        from diplomat.frontends.sleap_legacy.convert_tracks import (
            _sleap_analysis_h5_to_diplomat_table,
        )
        from diplomat.frontends.sleap_legacy.load_model import load_models

        return DIPLOMATCommands(
            _verifier=_verify_sleap_like,
            _load_model=load_models,
            _load_tracks=_sleap_analysis_h5_to_diplomat_table,
        )

    @classmethod
    def get_package_name(cls) -> str:
        return "sleap_legacy"
