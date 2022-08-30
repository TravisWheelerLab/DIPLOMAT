from deeplabcut import auxiliaryfunctions
from diplomat.processing.type_casters import PathLike, Union, Optional, List, Dict, Any, typecaster_function

@typecaster_function
def _verify_dlc_like(
    config: PathLike,
    videos: Union[List[PathLike], PathLike] = None,
    frame_stores: Union[List[PathLike], PathLike] = None,
    predictor: Optional[str] = None,
    predictor_settings: Optional[Dict[str, Any]] = None,
    num_outputs: Optional[int] = None,
    **kwargs
) -> bool:
    try:
        cfg = auxiliaryfunctions.read_config(str(config))
        # Check the config for DLC based keys...
        expected_keys = {"Task", "scorer", "date", "project_path", "video_sets", "bodyparts"}
        for key in expected_keys:
            if(key not in cfg):
                return False

        return True
    except Exception:
        return False