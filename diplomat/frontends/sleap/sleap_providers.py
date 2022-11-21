from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Iterator, Set, Type, List, Tuple
from typing_extensions import TypedDict
import numpy as np
import tensorflow as tf
from sleap import Video as SleapVideo
from sleap.nn.data.pipelines import Provider
from sleap.nn.data.providers import VideoReader as SleapVideoReader
from sleap.nn.inference import Predictor as SleapPredictor
from diplomat.processing import TrackingData


class SleapMetadata(TypedDict):
    bp_names: List[str]
    skeleton: Optional[List[Tuple[str, str]]]


class SleapModelExtractor(ABC):
    """
    Takes a SLEAP Predictor, and modifies it so that it outputs TrackingData instead of SLEAP predictions.
    """
    supported_models: Optional[Set[Type[SleapPredictor]]] = None

    @abstractmethod
    def __init__(self, model: SleapPredictor):
        pass

    @abstractmethod
    def get_metadata(self, data: Union[Provider]) -> SleapMetadata:
        pass

    @abstractmethod
    def extract(self, data: Union[Provider]) -> Iterator[TrackingData]:
        pass


class BottomUpModelExtractor(SleapModelExtractor):
    from sleap.nn.inference import BottomUpPredictor, BottomUpMultiClassPredictor
    supported_models: Optional[Set[SleapPredictor]] = {BottomUpPredictor, BottomUpMultiClassPredictor}

    def __init__(self, model: Union[BottomUpPredictor, BottomUpMultiClassPredictor]):
        super().__init__(model)
        self._predictor = model

    def get_metadata(self, data: Union[Provider]) -> SleapMetadata:
        pass

    def extract(self, data: Union[Dict, np.ndarray]) -> TrackingData:
        inf_layer = self._predictor.inference_model.inference_layer
        conf_map, __, offsets = inf_layer.forward_pass(data)
        return TrackingData(
            conf_map.numpy(),
            offsets if(offsets is None) else offsets.numpy(),
            inf_layer.input_scale * inf_layer.cm_output_stride
        )


class TopDownModelExtractor(SleapModelExtractor):
    from sleap.nn.inference import TopDownPredictor, TopDownMultiClassPredictor
    supported_models: Optional[Set[SleapPredictor]] = {TopDownPredictor, TopDownMultiClassPredictor}

    def __init__(self, model: Union[TopDownPredictor, TopDownMultiClassPredictor]):
        super().__init__(model)
        self._predictor = model

    def get_metadata(self, data: Union[Provider]) -> SleapMetadata:
        pass

    def extract(self, data: Union[Dict, np.ndarray]) -> TrackingData:
        inf_layer = self._predictor.inference_model.instance_peaks

        # Split image into slices to cover entire image....
        print(inf_layer.keras_model.input[0].shape)
        return None



class SingleInstanceModelExtractor(SleapModelExtractor):
    from sleap.nn.inference import SingleInstancePredictor
    supported_models: Optional[Set[SleapPredictor]] = {SingleInstancePredictor}

    def __init__(self, model: Union[SingleInstancePredictor]):
        super().__init__(model)
        self._predictor = model

    def get_metadata(self, data: Union[Provider]) -> SleapMetadata:
        pass

    def extract(self, data: Union[Dict, np.ndarray]) -> TrackingData:
        inf_layer = self._predictor.inference_model.single_instance_layer

        imgs = data["image"] if(isinstance(data, dict)) else data
        model_outputs = inf_layer.keras_model(inf_layer.preprocess(imgs))

        conf_map = model_outputs
        offset_map = None

        if(isinstance(model_outputs, list)):
            conf_map = model_outputs[inf_layer.confmaps_ind]
            if(inf_layer.offsets_ind is not None):
                offset_map = model_outputs[inf_layer.offsets_ind]

        return TrackingData(
            conf_map.numpy(),
            offset_map if(offset_map is None) else offset_map.numpy(),
            inf_layer.input_scale * inf_layer.output_stride
        )


EXTRACTORS = [BottomUpModelExtractor, SingleInstanceModelExtractor, TopDownModelExtractor]


class PredictorExtractor:
    def __init__(self, predictor: SleapPredictor):
        super().__init__()
        self._predictor = predictor

        for model_extractor in EXTRACTORS:
            if(type(predictor) in model_extractor.supported_models):
                self._model_extractor = model_extractor(self._predictor)
                break
        else:
            raise NotImplementedError(f"The provided predictor, '{type(predictor).__name__}', is not supported yet!")

    def extract(self, data: Union[Provider, SleapVideo]) -> Iterator[TrackingData]:
        if(isinstance(data, SleapVideo)):
            data = SleapVideoReader(data)

        pred = self._predictor

        pred.make_pipeline(data)
        if(pred.inference_model is None):
            pred._initialize_inference_model()

        print(pred.pipeline)
        for ex in pred.pipeline.make_dataset():
            yield self._model_extractor.extract(ex)



def _main_test():
    import sleap
    sleap.disable_preallocation()

    import tensorflow as tf
    if(not tf.executing_eagerly()):
        tf.compat.v1.enable_eager_execution()

    mdl = sleap.load_model("/home/isaac/Code/sleap-data/flies13/models/training_config.json")
    vid = sleap.load_video("/home/isaac/Code/sleap-data/flies13/talk_title_slide@13150-14500.mp4")
    extractor = PredictorExtractor(mdl)

    for frame in extractor.extract(vid):
        print(frame)
        print(frame.get_source_map())
        return



if(__name__ == "__main__"):
    _main_test()




