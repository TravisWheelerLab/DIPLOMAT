"""
Provides tools for replacing a functions module and name, while still allowing the function to be pickled and
transferred around...
"""
from types import FunctionType
from typing import Any, Callable
from functools import update_wrapper


class MovedFunction:
    def __init__(self, func: FunctionType, name: str, module: str):
        self.__name__ = name
        self.__module__ = module
        update_wrapper(self, func)
        self._f = func

    def __getattr__(self, item: str) -> Any:
        return getattr(self._f, item)

    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)

    def __repr__(self):
        return self._f.__repr__()

    def __str__(self):
        return self._f.__str__()


def replace_function_name_and_module(func: FunctionType, name: str, module: str) -> Callable:
    return MovedFunction(func, name, module)