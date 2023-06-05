from diplomat.processing.type_casters import typecaster_function, Union, List, PathLike
from .csv_utils import _fix_paths, _header_check


@typecaster_function
def _verify(
    config: Union[List[PathLike], PathLike],
    **kwargs
) -> bool:
    if("videos" not in kwargs):
        return False

    try:
        config, videos = _fix_paths(config, kwargs["videos"])
        return all(_header_check(c) for c in config)
    except (IOError, ValueError):
        return False
