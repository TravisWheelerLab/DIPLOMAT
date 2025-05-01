from .sleap_imports import tf, h5py
import json
import zipfile
from io import BytesIO
from inspect import signature
from pathlib import Path, PurePosixPath
from typing import Optional, Type
from diplomat.processing import Predictor, Config
from ...utils.lazy_import import resolve_lazy_imports


def _paths_to_str(paths):
    if(isinstance(paths, (list, tuple))):
        return [str(p) for p in paths]
    else:
        return str(paths)


_INT_TO_EDGE_TYPE = {
    1: "BODY",
    2: "SYMMETRY"
}


def _decode_skeleton(skeleton_dict, obj_memory = None):
    if(obj_memory is None):
        obj_memory = []
    new_skeleton = {}

    if("py/object" in skeleton_dict):
        # Extract the data
        data = skeleton_dict["py/state"]["py/tuple"]
        obj_memory.append(data)
        return data
    if("py/reduce" in skeleton_dict):
        data = _INT_TO_EDGE_TYPE[skeleton_dict["py/reduce"][1]["py/tuple"][0]]
        obj_memory.append(data)
        return data
    if("py/id" in skeleton_dict):
        return obj_memory[skeleton_dict["py/id"] - 1]

    for k, sub_item in skeleton_dict.items():
        if(isinstance(sub_item, dict)):
            new_skeleton[k] = _decode_skeleton(sub_item, obj_memory)
        elif(isinstance(sub_item, list)):
            new_skeleton[k] = [_decode_skeleton(v, obj_memory) if(isinstance(v, dict)) else v for v in sub_item]
        else:
            new_skeleton[k] = sub_item

    return new_skeleton


def _resolve_model_path(files):
    models = [f for f in files if "model" in f.stem and f.suffix == ".h5"]

    for model_option in ["best_model", "final_model", "latest_model"]:
        full_name = f"{model_option}.h5"
        for model in models:
            if(model.name == full_name):
                return model

    max_model = None
    max_val = 0
    for model in models:
        val = int(model.split("_")[-1])
        if(max_val < val):
            max_model = model
            max_val = val

    if(max_model is None):
        raise ValueError("Unable to find a model to load in the configuration directory!")

    return max_model


def _dict_has_path(dict_obj, key):
    for k_p in key:
        if(not isinstance(dict_obj, dict) or k_p not in dict_obj):
            return False
        dict_obj = dict_obj[k_p]
    return True


def _dict_get_path(dict_obj, key, default = None):
    for k_p in key:
        if(not isinstance(dict_obj, dict) or k_p not in dict_obj):
            return default
        dict_obj = dict_obj[k_p]
    return dict_obj


def _correct_skeletons_in_config(cfg):
    if _dict_has_path(cfg, ("data", "labels", "skeletons")):
        cfg["data"]["labels"]["skeletons"] = [
            _decode_skeleton(s) for s in cfg["data"]["labels"]["skeletons"]
        ]


@resolve_lazy_imports
def _load_configs_from_zip(z: zipfile.ZipFile, include_model = True):
    cfg_lst = []

    for file in z.infolist():
        if (file.filename.split("/")[-1].endswith("training_config.yaml")):
            inner_path = PurePosixPath(file.filename)
            config_dir = inner_path.parent

            cfg = json.loads(z.read(str(inner_path)))
            _correct_skeletons_in_config(cfg)
            model_path = _resolve_model_path(
                PurePosixPath(name) for name in z.namelist() if (PurePosixPath(name).parent == config_dir)
            )
            if (include_model):
                model = tf.keras.models.load_model(
                    h5py.File(BytesIO(z.read(str(model_path))), "r"),
                    compile=False
                )
                return cfg_lst.append((cfg, model))
            else:
                return cfg_lst.append(cfg)

    if len(cfg_lst) == 0:
        raise IOError("Sleap model zip file does not contain a training configuration file!")

    return cfg_lst


@resolve_lazy_imports
def _load_config_and_model(path, include_model = True):
    path = Path(path)
    if(zipfile.is_zipfile(path)):
        with zipfile.ZipFile(path, "r") as z:
            return _load_configs_from_zip(z, include_model)

    if(path.is_dir()):
        path = path / "training_config.json"
    path = path.resolve()

    with path.open("rb") as f:
        cfg = json.load(f)
        _correct_skeletons_in_config(cfg)
    model_path = _resolve_model_path(path.parent.iterdir())
    if(include_model):
        model = tf.keras.models.load_model(model_path, compile=False)
        return [(cfg, model)]
    else:
        return [cfg]


def _load_configs(paths, include_models: bool = True):
    paths = [paths] if(isinstance(paths, (str, Path))) else paths

    if(len(paths) < 1):
        raise ValueError(f"No configuration files passed to open!")

    configs = []

    for p in paths:
        configs.extend(_load_config_and_model(p, include_models))

    return configs


def _get_default_value(func, attr, fallback):
    param = signature(func).parameters.get(attr, None)
    return fallback if(param is None) else param.default


def _get_predictor_settings(predictor_cls: Type[Predictor], user_passed_settings) -> Optional[Config]:
    settings_backing = predictor_cls.get_settings()

    if(settings_backing is None):
        return None

    return Config(user_passed_settings, settings_backing)

