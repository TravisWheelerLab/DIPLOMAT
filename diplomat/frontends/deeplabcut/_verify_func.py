from .dlc_importer import auxiliaryfunctions
from diplomat.processing.type_casters import PathLike, typecaster_function

@typecaster_function
def _verify_dlc_like(
    config: PathLike,
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