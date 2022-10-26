"""
Provides an abstract progress bar interface that is passed to predictors for tracking progress.
"""

from abc import ABC, abstractmethod
from typing import Optional
import tqdm


class ProgressBar(ABC):
    """
    Abstract API for representing a progress bar. Used by predictors for displaying progress info.
    """
    @abstractmethod
    def __init__(self, total: Optional[int] = None):
        """
        Create a new progress bar.

        :param total: An optional integer, the total number of steps that need to be completed. If set to None
                      progress bar disables completion percentage features and simply keeps track of the number
                      of iterations done.
        """
        pass

    @abstractmethod
    def reset(self, total: Optional[int] = None):
        """
        Reset the progress bar with a new total value to reach.

        :param total: An optional integer, the total number of steps that need to be completed. If set to None
                      progress bar disables completion percentage features and simply keeps track of the number
                      of iterations done.
        """
        pass

    @abstractmethod
    def update(self, amt: int = 1):
        """
        Perform a progress bar update, increasing the number of iterations done.

        :param amt: The number of iterations or steps done, to increase the progress bar by. Defaults to 1.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the progress bar, disabling any additional updates.
        """
        pass

    @abstractmethod
    def message(self, message: str):
        """
        Set the description message for the progress bar.

        :param message: A string, the message to include with or above the progress bar. Can be used to describe
                        the current state a process is in.
        """
        pass

    def __del__(self):
        """
        Delete the progress bar, closing it.
        """
        self.__exit__(None, None, None)

    def __enter__(self):
        """
        Allows for progress bar usage with a context manager. 'Opens' the progress bar.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Allows for progress bar usage with a context manager. Closes the progress bar.
        """
        try:
            self.close()
        except Exception:
            pass


class TQDMProgressBar(ProgressBar):
    """
    A Concrete implementation of the ProgressBar API, uses TQDM to display progress.
    """
    def __init__(self, total: Optional[int] = None, tqdm_prior: Optional[tqdm.tqdm] = None):
        """
        Create a new tqdm progress bar.

        :param total: An optional integer, the total number of steps that need to be completed. If set to None
                      progress bar disables completion percentage features and simply keeps track of the number
                      of iterations done.
        :param tqdm_prior: Optional tqdm progress bar. If not set, constructs a new tqdm progress bar using total to
                           utilize internally. Otherwise, uses this as the internal tqdm progress bar, and ignores
                           the total argument.
        """
        super().__init__(total)
        if(tqdm_prior is not None):
            self._tqdm = tqdm_prior
        else:
            self._tqdm = tqdm.tqdm(total=total)

    def reset(self, total: Optional[int] = None):
        self._tqdm.reset(total)

    def update(self, amt: int = 1):
        self._tqdm.update(amt)

    def message(self, message: str):
        self._tqdm.set_description(message)

    def close(self):
        self._tqdm.close()