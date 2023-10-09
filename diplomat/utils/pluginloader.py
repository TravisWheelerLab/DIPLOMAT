"""
Module includes methods useful to loading all plugins placed in a folder, or module.
"""

from typing import Set
from typing import Type
from typing import TypeVar
from types import ModuleType
import pkgutil
import sys
import warnings


# Generic type for method below
T = TypeVar("T")


def load_plugin_classes(
    plugin_dir: ModuleType,
    plugin_metaclass: Type[T],
    do_reload: bool = False,
    display_error: bool = True,
    recursive: bool = True
) -> Set[Type[T]]:
    """
    Loads all plugins, or classes, within the specified module folder and submodules that extend the provided metaclass
    type.

    :param plugin_dir: A module object representing the path containing plugins... Can get a module object
                       using import...
    :param plugin_metaclass: The metaclass that all plugins extend. Please note this is the class type, not the
                             instance of the class, so if the base class is Foo just type Foo as this argument.
    :param do_reload: Boolean, Determines if plugins should be reloaded if they already exist. Defaults to True.
    :param display_error: Boolean, determines if import errors are sent using python's warning system when they occur.
                          Defaults to True. Note these warnings won't be visible unless you set up a filter for them,
                          such as below:

                          import warnings
                          warnings.simplefilter("always", ImportWarning)
    :param recursive: Boolean, if true recursively search subpackages for the class. Otherwise, only the first level is
                      searched.

    :return: A list of class types that directly extend the provided base class and where found in the specified
             module folder.
    """
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
                sub_module = importer.find_module(package_name).load_module(
                    package_name
                )
                sys.modules[package_name] = sub_module
            except Exception as e:
                if(display_error):
                    import traceback
                    warnings.warn(
                        f"Can't load '{package_name}'. Due to issue below: \n {traceback.format_exc()}",
                        ImportWarning
                    )
                continue
        else:
            sub_module = sys.modules[package_name]

        # Now we check if the module is a package, and if so, recursively call this method...
        if ispkg and recursive:
            plugins = plugins | load_plugin_classes(
                sub_module, plugin_metaclass, do_reload
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
