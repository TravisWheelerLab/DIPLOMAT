"""
A tool providing multi-animal tracking capabilities on top of DeepLabCut.
"""

__version__ = "0.0.1"

from diplomat.predictor_ops import list_predictor_plugins, get_predictor_settings, test_predictor_plugin
from diplomat.utils.video_splitter import split_videos

# Attempt to load all frontends, putting their public functions into submodules of diplomat.
def load_frontends():
    from diplomat import frontends
    from diplomat.frontends import DIPLOMATFrontend
    from diplomat.utils.pluginloader import load_plugin_classes
    from types import ModuleType
    from dataclasses import asdict

    frontends = load_plugin_classes(frontends, DIPLOMATFrontend)
    loaded_funcs = {}

    for frontend in frontends:
        try:
            res = frontend.init()
        except Exception as e:
            res = None

        if(res is not None):
            name = frontend.get_package_name()
            mod = ModuleType(__name__ + "." + name)

            globals()[name] = mod
            loaded_funcs[name] = res

            if(hasattr(frontend, "__doc__")):
                mod.__doc__ = frontend.__doc__

            for (name, func) in asdict(res).items():
                if(not name.startswith("_")):
                    setattr(mod, name, func)

    return frontends, loaded_funcs

_FRONTENDS, _LOADED_FRONTENDS = load_frontends()