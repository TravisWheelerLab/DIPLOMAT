import sleap

from diplomat.processing.type_casters import Union, List, PathLike, typecaster_function
from .run_utils import _paths_to_str

@typecaster_function
def _verify_sleap_like(
    config: Union[List[PathLike], PathLike],
    **kwargs
) -> bool:
    try:
        config = _paths_to_str(config)
        __ = sleap.load_model(config)
        return True
    except:
        return False