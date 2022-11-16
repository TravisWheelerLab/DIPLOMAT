from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Generator, Iterator

import numpy as np
from sleap.nn.data.pipelines import Provider
from sleap.nn.inference import InferenceModel, Predictor
from diplomat.processing import TrackingData

def poses_to_sleap(poses)

class SleapProvider(ABC):
    """
    Takes a SLEAP Predictor, and modifies it so that it outputs TrackingData instead of SLEAP predictions.
    """
    supported_models: Optional[frozenset] = None

    @abstractmethod
    def __init__(self, model: InferenceModel):
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str]:
        pass

    @abstractmethod
    def __call__(self, data: Union[]) -> Generator[TrackingData]:
        pass


class WrappingPredictor(Predictor):
    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def _predict_generator(
        self, data_provider: Provider
    ) -> Iterator[Dict[str, np.ndarray]]:

