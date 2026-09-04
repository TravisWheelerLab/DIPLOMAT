import zipfile
from pathlib import Path, PurePosixPath

import yaml

from diplomat.utils.lazy_import import resolve_lazy_imports


def _dict_get_path(dict_obj, key, default=None):
    for k_p in key:
        if not isinstance(dict_obj, dict) or k_p not in dict_obj:
            return default
        dict_obj = dict_obj[k_p]
    return dict_obj


def _find_key_nested(data: dict, key: str, default=None):
    for k, v in data.items():
        if k == key:
            return v
        if isinstance(v, dict):
            guess = _find_key_nested(v, key)
            if guess is not None:
                return guess

    return default


def _paths_to_str(paths):
    if isinstance(paths, (list, tuple)):
        return [str(p) for p in paths]
    else:
        return [str(paths)]


def _resolve_model_path(files):
    models = [f for f in files if f.suffix == ".ckpt"]

    for model_option in ["best", "final", "latest"]:
        full_name = f"{model_option}.ckpt"
        for model in models:
            if model.name == full_name:
                return model

    max_model = None
    max_val = 0
    for model in models:
        try:
            val = int(model.split("_")[-1])
        except ValueError:
            continue
        if max_val < val:
            max_model = model
            max_val = val

    if max_model is None:
        raise ValueError(
            "Unable to find a model to load in the configuration directory!"
        )

    return max_model


@resolve_lazy_imports
def _load_configs_from_zip(zip_path: Path, unpack_zips_to=None):
    cfg_lst = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.infolist():
            if file.filename.split("/")[-1].endswith("training_config.yaml"):
                inner_path = PurePosixPath(file.filename)
                config_dir = inner_path.parent

                cfg = yaml.load(z.read(str(inner_path)))
                model_path = _resolve_model_path(
                    PurePosixPath(name)
                    for name in z.namelist()
                    if (PurePosixPath(name).parent == config_dir)
                )
                if unpack_zips_to is not None:
                    unpack_zips_to = Path(unpack_zips_to)
                    data_dir = unpack_zips_to / "i"
                else:
                    cfg_lst.append((cfg, f"{}:{str(inner_path)}", f"{}:{model_path}"))

        if len(cfg_lst) == 0:
            raise IOError(
                "Sleap model zip file does not contain a training configuration file!"
            )

    return cfg_lst


def _load_config_and_model(path, unpack_zips_to):
    path = Path(path)
    if zipfile.is_zipfile(path):
        return _load_configs_from_zip(path, unpack_zips_to)

    if path.is_dir():
        path = path / "training_config.yaml"
    path = path.resolve()

    with path.open("rb") as f:
        cfg = yaml.load(f)
    model_path = _resolve_model_path(path.parent.iterdir())
    return [(cfg, path, model_path)]


@resolve_lazy_imports
def _load_model_configs(model_paths):
    # from sleap_nn.config.utils import resolve_model_dir
    # from sleap_nn.inference.loaders import _load_training_config
    cfgs = []
    for mp in model_paths:
        cfg, _ = _load_training_config(mp)
        cfgs.append((mp, cfg))

    return cfgs
