"""
Module includes methods useful to loading all plugins placed in a folder, or module.
"""

import importlib
import logging
import pkgutil
import sys
from types import ModuleType
from typing import Callable, Set, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# Generic type for method below
T = TypeVar("T")


def load_plugin_classes(
    plugin_dir: ModuleType,
    plugin_metaclass: Type[T],
    do_reload: bool = False,
    display_error: Union[bool, Callable[[str, Exception], bool]] = True,
    recursive: bool = True,
) -> Set[Type[T]]:
    """
    Loads all plugins, or classes, within the specified module folder and submodules that extend the provided metaclass
    type.

    :param plugin_dir: A module object representing the path containing plugins... Can get a module object
                       using import...
    :param plugin_metaclass: The metaclass that all plugins extend. Please note this is the class type, not the
                             instance of the class, so if the base class is Foo just type Foo as this argument.
    :param do_reload: Boolean, Determines if plugins should be reloaded if they already exist. Defaults to True.
    :param display_error: Boolean or callable.
                          If a boolean, determines if import errors are sent using python's logging system when they occur.
                          Defaults to True. By default, these warnings are visible.
                          If a callable, it should accept the package name and an exception, and return a boolean determining
                          if an exception should be thrown. This function can also perform logging.
    :param recursive: Boolean, if true recursively search subpackages for the class. Otherwise, only the first level is
                      searched.

    :return: A list of class types that directly extend the provided base class and where found in the specified
             module folder.
    """
    if isinstance(display_error, bool):
        if display_error:

            def _display_error_default(p, e):
                logger.warning(f"Can't load '{p}'.", exc_info=e)
                return False

            display_error = _display_error_default
        else:
            display_error = lambda _p, _e: False
    # Get absolute and relative package paths for this module...
    path = list(iter(plugin_dir.__path__))[0]
    rel_path = plugin_dir.__name__

    plugins: Set[Type[T]] = set()

    # Iterate all modules in specified directory using pkgutil, importing them if they are not in sys.modules
    for importer, package_name, ispkg in pkgutil.iter_modules([path], rel_path + "."):
        # If the module name is not in system modules or the 'reload' flag is set to true, perform a full load of the
        # modules...
        if (package_name not in sys.modules) or do_reload:
            try:
                if package_name in sys.modules:
                    del sys.modules[package_name]
                sub_module = importlib.import_module(package_name)
            except Exception as e:
                if display_error(package_name, e):
                    raise
                continue
        else:
            sub_module = sys.modules[package_name]

        # Now we check if the module is a package, and if so, recursively call this method...
        if ispkg and recursive:
            plugins = plugins | load_plugin_classes(
                sub_module, plugin_metaclass, do_reload, display_error, recursive
            )

        # We begin looking for plugin classes
        for item in dir(sub_module):
            field = getattr(sub_module, item)

            try:
                # We check if the field is a type or class,
                # it's module matches the current module (it was created here, this makes sure the location classes
                # are loaded from is consistent), it extends or is the base class for this type of plugin,
                # and it is not the plugin base class.
                if (
                    isinstance(field, type)
                    and (field.__module__ == sub_module.__name__)
                    and issubclass(field, plugin_metaclass)
                    and (field != plugin_metaclass)
                ):
                    # It is a plugin, add it to the list...
                    plugins.add(field)
            except Exception:
                # Some classes throw an error when passed to issubclass, just ignore them as they're
                # clearly not a plugin.
                pass

    return plugins
