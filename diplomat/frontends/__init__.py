from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from inspect import get_annotations
from diplomat.processing.type_casters import StrictCallable, PathLike, Union, Sequence, Dict, Any, Optional, TypeCaster, NoneType
import typing


AnalyzeVideosFunction = lambda ret: StrictCallable(
    config=PathLike,
    videos=Union[Sequence[PathLike], PathLike],
    predictor=Optional[str],
    predictor_settings=Optional[Dict[str, Any]],
    _return=ret
)
AnalyzeFramesFunction = lambda ret: StrictCallable(
    config=PathLike,
    frame_stores=Union[Sequence[PathLike], PathLike],
    predictor=Optional[str],
    predictor_settings=Optional[Dict[str, Any]],
    _return=ret
)
LabelVideosFunction = lambda ret: StrictCallable(config=PathLike, videos=Union[Sequence[PathLike], PathLike], _return=ret)


@dataclass(frozen=False)
class DIPLOMATBaselineCommands:
    """
    The baseline set of functions each DIPLOMAT backend must implement. Backends can add additional commands
    by extending this base class...
    """
    _verify_analyze_videos: AnalyzeVideosFunction(bool)
    analyze_videos: AnalyzeVideosFunction(NoneType)
    _verify_analyze_frames: AnalyzeFramesFunction(bool)
    analyze_frames: AnalyzeFramesFunction(NoneType)
    _verify_label_videos: LabelVideosFunction(bool)
    label_videos: LabelVideosFunction(NoneType)

    def __post_init__(self):
        annotations = get_annotations(type(self))

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
