from inspect import signature
from typing import Optional, Type

from diplomat.processing import Config, Predictor
from diplomat.utils.lazy_import import resolve_lazy_imports

from .sleap_imports import omegaconf, sleap_nn


def _paths_to_str(paths):
    if isinstance(paths, (list, tuple)):
        return [str(p) for p in paths]
    else:
        return [str(paths)]


@resolve_lazy_imports
def _load_model_configs(model_paths):
    resolve_model_dir = sleap_nn.config.utils.resolve_model_dir
    _load_training_config = sleap_nn.inference.loaders._load_training_config

    preprocess_config = omegaconf.OmegaConf.create(
        {
            "ensure_rgb": None,
            "ensure_grayscale": None,
            "crop_size": None,
            "max_width": None,
            "max_height": None,
            "scale": None,
        }
    )

    model_paths = [resolve_model_dir(mp) for mp in model_paths]

    cfgs = []
    for mp in model_paths:
        cfg, _ = _load_training_config(mp)
        cfgs.append(cfg)

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
