"""
A tool providing multi-animal tracking capabilities on top of other Deep learning based tracking software.
"""

__version__ = "0.1.8"
# Can be used by functions to determine if diplomat was invoked through it's CLI interface.
CLI_RUN = False

from diplomat.predictor_ops import list_predictor_plugins, get_predictor_settings, test_predictor_plugin
from diplomat.frontend_ops import list_all_frontends, list_loaded_frontends
from diplomat.utils.video_splitter import split_videos
from diplomat.core_ops import track_with, track, track_and_interact, annotate, tweak, yaml, convert, interact

__all__ = [
    "list_predictor_plugins",
    "get_predictor_settings",
    "test_predictor_plugin",
    "list_all_frontends",
    "list_loaded_frontends",
    "split_videos",
    "track_with",
    "track_and_interact",
    "track",
    "annotate",
    "tweak",
    "yaml",
    "convert",
    "interact"
]


# Attempt to load all frontends, putting their public functions into submodules of diplomat.
def _load_frontends():
    from diplomat import frontends
    from diplomat.frontends import DIPLOMATFrontend
    from diplomat.utils.pluginloader import load_plugin_classes
    from diplomat.utils._function_tools import replace_function_name_and_module
    from types import ModuleType
    from multiprocessing import current_process

    if(current_process().name != "MainProcess"):
        # If something in this package is using multiprocessing, disable the automatic frontend loading code.
        # This is done because some frontends (DEEPLABCUT) use about 1/3 a Gig of memory on import, which can
        # cause memory issues if a lot of processes are started by a predictor...
        return (set(), {})

    frontends = load_plugin_classes(frontends, DIPLOMATFrontend, recursive=False)
    loaded_funcs = {}

    for frontend in frontends:
        res = frontend.init()

        if(res is not None):
            name = frontend.get_package_name()
            mod = ModuleType(__name__ + "." + name)
            mod.__all__ = []

            globals()[name] = mod
            loaded_funcs[name] = res

            if(hasattr(frontend, "__doc__")):
                mod.__doc__ = frontend.__doc__

            for (name, func) in res:
                if(not name.startswith("_")):
                    func = replace_function_name_and_module(func, name, mod.__name__)
                    setattr(mod, name, func)
                    mod.__all__.append(name)

    return frontends, loaded_funcs


_FRONTENDS, _LOADED_FRONTENDS = _load_frontends()
