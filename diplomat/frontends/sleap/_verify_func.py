from diplomat.processing.type_casters import Union, List, PathLike, typecaster_function
from .run_utils import _paths_to_str, _load_configs


@typecaster_function
def _verify_sleap_like(
    config: Union[List[PathLike], PathLike],
    **kwargs
) -> bool:
    try:
        # Config for sleap is always a sleap model, so try to load it...
        config = _paths_to_str(config)
        cfgs = _load_configs(config, include_models=False)
        return len(cfgs) >= 1
    except (IOError, ValueError):
        return False
