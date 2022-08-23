from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DIPLOMATBaselineCommands:
    """
    The baseline set of functions each DIPLOMAT backend must implement. Backends can add additional commands
    by extending this base class...
    """
    analyze_video: Callable
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
