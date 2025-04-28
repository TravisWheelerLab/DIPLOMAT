import diplomat.processing.type_casters as tc
from diplomat.frontends import ModelInfo, ModelLike
from diplomat.frontends.sleap.run_utils import _load_configs
from diplomat.frontends.sleap.sleap_providers import PredictorExtractor
from diplomat.utils.cli_tools import extra_cli_args, Flag


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def load_models(
    config: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    batch_size: tc.Optional[int] = None,
    num_outputs: tc.Optional[int] = None,
    gpu_index: tc.Optional[int] = None,
    output_suffix: str = "",
    refinement_kernel_size: int = 5,
    use_cpu: Flag = False,
    **kwargs
) -> tc.Tuple[ModelInfo, ModelLike]:
    configs = _load_configs(config)

    provider = PredictorExtractor(
        configs,
        refinement_kernel_size,

    )

