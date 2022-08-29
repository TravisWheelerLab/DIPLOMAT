from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Type, Union, List
import inspect
import os

Pathy = Union[os.PathLike, str]


class ArgumentMatchingFunction:
    def __init__(self, **kwargs: Type):
        self._params = kwargs

    def enforce(self, callable):
        sig = inspect.signature(callable)
        for name in sig.parameters.items():
            pass


def function_enforcing_dataclass(clazz):
    enforcers = {}

    for name, t in inspect.get_annotations(clazz).items():
        if(isinstance(t, ArgumentMatchingFunction)):
            clazz.__annotations__[name] = Callable
            enforcers[name] = t

    # Insert our custom post-init method...
    def pi(self):
        for name, enf in self._enforcers:
            enf.enforce(getattr(self, name))

    clazz.__post_init__ = pi
    clazz = dataclass(frozen=True)(clazz)
    clazz._enforcers = enforcers

    return clazz

@function_enforcing_dataclass
class DIPLOMATBaselineCommands:
    """
    The baseline set of functions each DIPLOMAT backend must implement. Backends can add additional commands
    by extending this base class...
    """
    description: str
    analyze_video: ArgumentMatchingFunction(config=Pathy, videos=Union[Pathy, List[Pathy]])
    analyze_frames: Callable
    label_video: Callable



class DIPLOMATFrontend(ABC):
    """
    Represents a diplomat frontend. Frontends can define commands or functions that can be used in the CLI frontend.
    """
    @classmethod
    @abstractmethod
    def init(cls) -> Optional[DIPLOMATBaselineCommands]:
        """
        Attempt to initialize the frontend, returning a list of api functions. If the backend can't initialize due to missing imports/requirements,
        this function should return None.

        :return: A DIPLOMATBaselineCommands or subclass of it, which is simply a dataclass contained all the required frontend functions.
        """
        pass

    @classmethod
    @abstractmethod
    def get_package_name(cls) -> str:
        """
        Return the name to give the subpackage and subcommand for this frontend.

        :return: A string, the sub-package and sub-command name.
        """
        pass
