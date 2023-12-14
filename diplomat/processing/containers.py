"""
Provides the config container implementation, used for storing configuration parameters.
"""

from collections import UserDict
from typing import TypeVar, Dict, Tuple, Any, Mapping
from diplomat.processing.type_casters import TypeCaster

T = TypeVar("T")
ConfigSpec = Dict[str, Tuple[T, TypeCaster[T], str]]


class Config(UserDict):
    """
    Represents a configuration of settings. Is a dictionary with support for
    attribute style access of keys, and also allows for a backing dictionary,
    which can control what values are allowed to be stored in this
    configuration.
    """
    def __init__(self, *args, **kwargs):
        """
        Create a new configuration dictionary:

        :param args: Only accepts 2 non-keyword arguments, the data, and a backing
               dictionary. See set_backing for more info.
        :param kwargs: Optional, additional keys and values to add to the dictionary.
        """
        args = list(args)
        self._back_dict = None
        if(len(args) > 2):
            raise TypeError("Only accepts up to 2 non-keyword arguments, dict and back_dict")
        if(len(args) == 2):
            self._back_dict = args.pop()
        super().__init__(*args, **kwargs)


    def set_backing(self, back_dict: ConfigSpec):
        """
        Set the backing dictionary for this config object.

        :param back_dict: A dictionary of strings (key names) to length 3 tuples. The tuples contain
                          the following values in order:

                          - E: The default value for this setting.
                          - TypeCaster[E]: A casting function, to convert or check passed values are of the right type.
                          - str: The description of this setting, not used here...
        """
        self._back_dict = back_dict
        self.update(self.data)

    def __missing__(self, key: str) -> Any:
        # Return default setting value if it is missing from the dictionary.
        if(self._back_dict is None):
            raise KeyError(key)
        else:
            def_val, conv_meth, desc = self._back_dict[key]
            return conv_meth(def_val)

    def extract(self, data: Mapping):
        if(self._back_dict is None):
            self.update(data)
        else:
            for key in self._back_dict:
                if(key in data):
                    self[key] = data[key]

    def __setitem__(self, key: str, value: Any):
        # We only allow for settings in the backing dict to be set, and only
        # if they type cast correctly.
        if(self._back_dict is None):
            super().__setitem__(key, value)
        else:
            def_val, conv_meth, desc = self._back_dict.get(key, (None, None, None))
            if(conv_meth is None):
                raise KeyError(f"Attempted to set non-existent setting: '{key}'.\n"
                               f"Supported settings are: {self._back_dict.keys()}")
            super().__setitem__(key, conv_meth(value))

    def __getattr__(self, key: str) -> Any:
        # Allow for access to dictionary values using the '.' operator...
        if (key.startswith("_") or (key == "data")):
            return super().__getattribute__(key)
        return self.__getitem__(key)

    def __setattr__(self, key: str, value: Any):
        # Allow for setting dictionary values using the '.' operator...
        if(key.startswith("_") or (key == "data")):
            return super().__setattr__(key, value)
        return self.__setitem__(key, value)