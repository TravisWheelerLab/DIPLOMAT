from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import OrderedDict

import numpy as np
import pandas as pd

from diplomat.processing import Pose, TrackingData
from diplomat.processing.type_casters import (
    StrictCallable,
    PathLike,
    Union,
    List,
    Dict,
    Any,
    Optional,
    TypeCaster,
    NoneType,
    Tuple, TypedDict, RangedInteger, RangedFloat
)
import typing

from diplomat.utils.colormaps import to_colormap
from diplomat.utils.shapes import shape_iterator


class Select(Union):
    def __init__(self, *types: TypeCaster):
        super().__init__(*types)
        self._all_valid_types = []
        for tp in types:
            if(isinstance(tp, Union)):
                self._all_valid_types.extend(tp._valid_types)
            else:
                self._all_valid_types.append(tp)

    def __eq__(self, other: TypeCaster):
        if(isinstance(other, Union)):
            # Subset check...
            return all(self.__eq__(val) for val in other._valid_types)
        else:
            if(other in self._all_valid_types):
                return True
            else:
                res = super().__eq__(other)
                if(res is NotImplemented):
                    return False
                return res

    def to_metavar(self) -> str:
        raise ValueError("Select should only be used for type enforcement, not type hints!")

    def to_type_hint(self) -> typing.Type:
        raise ValueError("Select should only be used for type enforcement, not type hints!")


# Config argument can be a list of paths, single path, or union of the two...
ConfigPathLikeArgument = Select[List[PathLike], PathLike]

VerifierFunction = StrictCallable(
    config=Union[List[PathLike], PathLike],
    _kwargs=True,
    _return=bool
)

ConvertResultsFunction = lambda ret: StrictCallable(
    config=ConfigPathLikeArgument,
    videos=Union[List[PathLike], PathLike],
    csvs=Union[List[PathLike], PathLike],
    _return=ret
)

ModelInfo = TypedDict(
    "ModelInfo",
    num_outputs=Optional(RangedInteger(1, np.inf)),
    batch_size=Optional(RangedInteger(1, np.inf)),
    dotsize=RangedInteger(1, np.inf),
    colormap=to_colormap,
    shape_list=shape_iterator,
    alphavalue=RangedFloat(0, 1),
    pcutoff=RangedFloat(0, 1),
    line_thickness=RangedInteger(1, np.inf),
    bp_names=List(str),
    skeleton=Union(NoneType, List(Tuple(str, str))),
    frontend=str
)

ModelLike = StrictCallable(
    frames=np.ndarray,
    _return=TrackingData
)
ModelLoaderFunction = StrictCallable(
    config=ConfigPathLikeArgument,
    num_outputs=Optional[int],
    batch_size=Optional[int],
    _return=Tuple(ModelInfo, ModelLike)
)

TracksLoaderFunction = StrictCallable(
    path=PathLike,
    _return=pd.DataFrame
)


@dataclass(frozen=True)
class DIPLOMATContract:
    """
    Represents a 'contract'
    """
    method_name: str
    method_type: StrictCallable


class CommandManager(type):

    __no_type_check__ = False

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)

        annotations = typing.get_type_hints(obj)

        for name, annot in annotations.items():
            if(name in obj.__dict__):
                raise TypeError(f"Command annotation '{name}' has default value, which is not allowed.")

        return obj

    def __getattr__(self, item) -> DIPLOMATContract:
        annot = typing.get_type_hints(self)[item]
        return DIPLOMATContract(item, annot)


def required(typecaster: TypeCaster) -> TypeCaster:
    typecaster._required = True
    return typecaster


class DIPLOMATCommands(metaclass=CommandManager):
    """
    The baseline set of functions each DIPLOMAT backend must implement. Backends can add additional commands
    by passing the methods to this classes constructor.
    """
    _verifier: required(VerifierFunction)
    _load_model: required(ModelLoaderFunction)
    _load_tracks: TracksLoaderFunction

    def __init__(self, **kwargs):
        missing = object()
        self._commands = OrderedDict()

        annotations = typing.get_type_hints(type(self))

        for name, annot in annotations.items():
            value = kwargs.get(name, missing)

            if(value is missing):
                if(getattr(annot, "_required", False)):
                    raise ValueError(f"Command '{name}' is required, but was not provided.")
                continue
            if(annot is None or (not isinstance(annot, TypeCaster))):
                raise TypeError("DIPLOMAT Command Struct can only contain typecaster types.")

            self._commands[name] = annot(value)

        for name, value in kwargs.items():
            if(name not in annotations):
                self._commands[name] = value

    def __iter__(self):
        return iter(self._commands.items())

    def __getattr__(self, item: str):
        return self._commands.get(item)

    def __contains__(self, item: str):
        return item in self._commands

    def verify(self, contract: DIPLOMATContract, config: Union[List[PathLike], PathLike], **kwargs: Any) -> bool:
        """
        Verify this backend can handle the provided command type, config file, and arguments.

        :param contract: The contract for the command. Includes the name of the method and the type of the method,
                         which will typically be a strict callable.
        :param config: The configuration file, checks if the backend can handle this configuration file.
        :param kwargs: Any additional arguments to pass to the backends verifier.

        :return: A boolean, True if the backend can handle the provided command and arguments, otherwise False.
        """
        if(self.verify_contract(contract)):
            return self._verifier(config, **kwargs)
        return False

    def verify_contract(self, contract: DIPLOMATContract):
        """
        Verify this frontend has the provided contract, or function with a specified name and arguments.

        :param contract: The contract for the command. Includes the name of the method and the type of the method,
                         which will typically be a strict callable.
        """
        if(contract.method_name in self._commands):
            func = self._commands[contract.method_name]
            try:
                contract.method_type(func)
                return True
            except Exception:
                return False

        return False


class DIPLOMATFrontend(ABC):
    """
    Represents a diplomat frontend. Frontends can define commands or functions that can be used in the CLI frontend.
    """
    @classmethod
    @abstractmethod
    def init(cls) -> typing.Optional[DIPLOMATCommands]:
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
