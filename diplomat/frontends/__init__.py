from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from diplomat.processing.type_casters import StrictCallable, PathLike, Union, List, Dict, Any, Optional, TypeCaster, NoneType
import typing


class Select(Union):
    def __eq__(self, other: TypeCaster):
        if(isinstance(other, Union)):
            # Subset check...
            return all(self.__eq__(val) for val in other._valid_types)
        else:
            if(other in self._valid_types):
                return True
            else:
                return super().__eq__(other)

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

AnalyzeVideosFunction = lambda ret: StrictCallable(
    config=ConfigPathLikeArgument,
    videos=Union[List[PathLike], PathLike],
    predictor=Optional[str],
    predictor_settings=Optional[Dict[str, Any]],
    num_outputs=Optional[int],
    _return=ret
)

AnalyzeFramesFunction = lambda ret: StrictCallable(
    config=ConfigPathLikeArgument,
    frame_stores=Union[List[PathLike], PathLike],
    predictor=Optional[str],
    predictor_settings=Optional[Dict[str, Any]],
    num_outputs=Optional[int],
    _return=ret
)

LabelVideosFunction = lambda ret: StrictCallable(
    config=ConfigPathLikeArgument,
    videos=Union[List[PathLike], PathLike],
    _return=ret
)

ConvertResultsFunction = lambda ret: StrictCallable(
    config=ConfigPathLikeArgument,
    videos=Union[List[PathLike], PathLike],
    _return=ret
)


@dataclass(frozen=False)
class DIPLOMATBaselineCommands:
    """
    The baseline set of functions each DIPLOMAT backend must implement. Backends can add additional commands
    by extending this base class...
    """
    _verifier: VerifierFunction
    analyze_videos: AnalyzeVideosFunction(NoneType)
    analyze_frames: AnalyzeFramesFunction(NoneType)
    label_videos: LabelVideosFunction(NoneType)
    tweak_videos: LabelVideosFunction(NoneType)
    convert_results: ConvertResultsFunction(NoneType)

    def __post_init__(self):
        annotations = typing.get_type_hints(type(self))

        for name, value in asdict(self).items():
            annot = annotations.get(name, None)

            if(annot is None or (not isinstance(annot, TypeCaster))):
                raise TypeError("DIPLOMAT Command Struct can only contain typecaster types.")

            setattr(self, name, annot(value))


class DIPLOMATFrontend(ABC):
    """
    Represents a diplomat frontend. Frontends can define commands or functions that can be used in the CLI frontend.
    """
    @classmethod
    @abstractmethod
    def init(cls) -> typing.Optional[DIPLOMATBaselineCommands]:
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
