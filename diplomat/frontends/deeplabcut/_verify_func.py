from pathlib import Path, PurePosixPath
from typing import Tuple
from zipfile import ZipFile, is_zipfile

import yaml

from diplomat.processing.type_casters import List, PathLike, Union, typecaster_function


def _load_dlc_like_zip_file(z: ZipFile) -> Tuple[PurePosixPath, dict]:
    config_files = []
    all_dirs = {
        PurePosixPath(info.filename)
        if info.is_dir()
        else PurePosixPath(info.filename).parent
        for info in z.infolist()
    }
    for path in z.namelist():
        path = PurePosixPath(path)
        if (
            path.name == "config.yaml"
            and (path.parent / "dlc-models-pytorch") in all_dirs
        ):
            config_files.append(path)

    if len(config_files) < 1:
        raise ValueError("Could not find config file in the passed zip file!")
    elif len(config_files) > 1:
        raise ValueError("Found multiple config files in the passed zip file!")

    content = yaml.load(z.read(str(config_files[0])), yaml.SafeLoader)
    return config_files[0], content


@typecaster_function
def _verify_dlc_like(config: Union[List[PathLike], PathLike], **kwargs) -> bool:
    # DLC functions only accept a single path for the config, the path to the config.yaml...
    if isinstance(config, (list, tuple)):
        if len(config) > 1:
            raise ValueError("DeepLabCut can only accept 1 config file, not multiple.")
        config = config[0]

    if is_zipfile(config):
        with ZipFile(config, "r") as z:
            __, cfg = _load_dlc_like_zip_file(z)
    else:
        with open(str(config)) as f:
            cfg = yaml.load(f, yaml.SafeLoader)

        if not (Path(str(config)).parent / "dlc-models-pytorch").exists():
            raise ValueError(
                "Project is missing 'dlc-models-pytorch', may be old tensorflow deeplabcut project or a model hasn't been trained yet."
            )
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
            raise ValueError(f"Config is missing required keys: {key}")

    engine = cfg.get("engine", "pytorch")
    if engine != "pytorch":
        raise ValueError(f"Project is '{engine}', must be 'pytorch'!")

    return True
