from diplomat.processing.type_casters import List, PathLike, Union, typecaster_function

from .run_utils import _load_model_configs, _paths_to_str


@typecaster_function
def _verify_sleap_like(config: Union[List[PathLike], PathLike], **kwargs) -> bool:
    config = _paths_to_str(config)
    if len(config) < 1:
        raise ValueError("Empty list passed as configuration files!")
    _load_model_configs(config)
    return True
