from diplomat.processing.type_casters import Union, List, PathLike, typecaster_function


@typecaster_function
def _verify_dlc_like(
    config: Union[List[PathLike], PathLike],
    **kwargs
) -> bool:
    try:
        # DLC functions only accept a single path for the config, the path to the config.yaml...
        if(isinstance(config, (list, tuple))):
            if(len(config) > 1):
                return False
            config = config[0]

        import yaml
        with open(str(config)) as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        # Check the config for DLC based keys...
        expected_keys = {"Task", "scorer", "date", "project_path", "video_sets", ("bodyparts", "multianimalbodyparts")}
        for key in expected_keys:
            if(isinstance(key, str)):
                key = (key,)

            if(not any((sub_key in cfg) for sub_key in key)):
                return False

        return True
    except Exception:
        return False
