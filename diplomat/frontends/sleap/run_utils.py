import shutil
import zipfile
from pathlib import Path, PurePosixPath

import yaml


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


def _load_configs_from_zip(cfg_lst: list, zip_path: Path, unpack_zips_to=None):
    def files_in_dir(directory, as_path=False):
        return (
            PurePosixPath(name) if as_path else str(name)
            for name in z.namelist()
            if (PurePosixPath(name).parent == directory)
        )

    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.infolist():
            if file.filename.split("/")[-1].endswith("training_config.yaml"):
                inner_path = PurePosixPath(file.filename)
                config_dir = inner_path.parent

                cfg = yaml.safe_load(z.read(str(inner_path)))
                model_path = _resolve_model_path(files_in_dir(config_dir, True))
                if unpack_zips_to is not None:
                    unpack_zips_to = Path(unpack_zips_to)
                    data_dir = unpack_zips_to / f"model{len(cfg_lst)}"
                    data_dir.mkdir(exist_ok=True)
                    for name in files_in_dir(config_dir):
                        with (
                            z.open(name, "r") as r,
                            open(data_dir / PurePosixPath(name).name, "wb") as w,
                        ):
                            shutil.copyfileobj(r, w)
                    cfg_lst.append(
                        (
                            cfg,
                            str(data_dir / "training_config.yaml"),
                            str(data_dir / model_path.name),
                        )
                    )
                else:
                    cfg_lst.append(
                        (
                            cfg,
                            f"{zip_path}:{str(inner_path)}",
                            f"{zip_path}:{model_path}",
                        )
                    )

        if len(cfg_lst) == 0:
            raise IOError(
                "Sleap model zip file does not contain a training configuration file!"
            )


def _load_config_and_model(cfg_lst, path, unpack_zips_to=None):
    path = Path(path)
    if zipfile.is_zipfile(path):
        return _load_configs_from_zip(cfg_lst, path, unpack_zips_to)

    if path.is_dir():
        path = path / "training_config.yaml"
    path = path.resolve()

    with path.open("rb") as f:
        cfg = yaml.safe_load(f)
    model_path = _resolve_model_path(path.parent.iterdir())
    cfg_lst.append((cfg, path, model_path))


def _load_model_configs(model_paths, unpack_zips_to=None):
    cfgs = []
    for mp in model_paths:
        _load_config_and_model(cfgs, mp, unpack_zips_to)

    return cfgs
