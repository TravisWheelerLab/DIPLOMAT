from pathlib import PurePosixPath
from typing import Tuple
from zipfile import ZipFile, is_zipfile
from diplomat.processing.type_casters import Union, List, PathLike, typecaster_function
import yaml


def _load_dlc_like_zip_file(z: ZipFile) -> Tuple[PurePosixPath, dict]:
    config_files = []
    for path in z.namelist():
        path = PurePosixPath(path)
        if path.name == "config.yaml":
            config_files.append(path)

    if len(config_files) < 1:
        raise ValueError("Could not find config file in the passed zip file!")
    elif len(config_files) > 1:
        raise ValueError("Found multiple config files in the passed zip file!")

    content = yaml.load(z.read(str(config_files[0])), yaml.SafeLoader)
    return config_files[0], content


@typecaster_function
def _verify_dlc_like(config: Union[List[PathLike], PathLike], **kwargs) -> bool:
    try:
        # DLC functions only accept a single path for the config, the path to the config.yaml...
        if isinstance(config, (list, tuple)):
            if len(config) > 1:
                return False
            config = config[0]

        if is_zipfile(config):
            with ZipFile(config, "r") as z:
                __, cfg = _load_dlc_like_zip_file(z)
        else:
            with open(str(config)) as f:
                cfg = yaml.load(f, yaml.SafeLoader)
        # Check the config for DLC based keys...
        expected_keys = {
            "Task",
            "scorer",
            "date",
            "project_path",
            "video_sets",
            ("bodyparts", "multianimalbodyparts"),
        }
        for key in expected_keys:
            if isinstance(key, str):
                key = (key,)

            if not any((sub_key in cfg) for sub_key in key):
                return False

        return True
    except Exception:
        return False
