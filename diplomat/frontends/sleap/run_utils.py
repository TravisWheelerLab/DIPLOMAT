from inspect import signature
from typing import Optional, Type

from diplomat.processing import Config, Predictor
from diplomat.utils.lazy_import import resolve_lazy_imports

from .sleap_imports import omegaconf, sleap_nn


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


@resolve_lazy_imports
def _load_model_configs(model_paths):
    resolve_model_dir = sleap_nn.config.utils.resolve_model_dir
    _load_training_config = sleap_nn.inference.loaders._load_training_config

    model_paths = [resolve_model_dir(mp) for mp in model_paths]

    cfgs = []
    for mp in model_paths:
        cfg, _ = _load_training_config(mp)
        cfgs.append((mp, cfg))

    return cfgs


def _get_default_value(func, attr, fallback):
    param = signature(func).parameters.get(attr, None)
    return fallback if (param is None) else param.default


def _get_predictor_settings(
    predictor_cls: Type[Predictor], user_passed_settings
) -> Optional[Config]:
    settings_backing = predictor_cls.get_settings()

    if settings_backing is None:
        return None

    return Config(user_passed_settings, settings_backing)
