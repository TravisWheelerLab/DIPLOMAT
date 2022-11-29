from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Iterator, Set, Type, List, Tuple
from typing_extensions import TypedDict
import numpy as np
import tensorflow as tf
from sleap import Video as SleapVideo
from sleap.nn.data.pipelines import Provider
from sleap.nn.data.providers import VideoReader as SleapVideoReader
from sleap.nn.inference import Predictor as SleapPredictor
from sleap.nn.inference import InferenceLayer as SleapInferenceLayer
from sleap.skeleton import Skeleton as SleapSkeleton
from diplomat.processing import TrackingData


class SleapMetadata(TypedDict):
    bp_names: List[str]
    skeleton: Optional[List[Tuple[str, str]]]
    orig_skeleton: SleapSkeleton

def _extract_metadata(predictor: SleapPredictor) -> SleapMetadata:
    skel_list = predictor.data_config.labels.skeletons

    if(len(skel_list) < 1):
        raise ValueError("No part information for this SLEAP project, can't run diplomat!")

    skeleton1 = skel_list[0]
    edge_name_list = skeleton1.edge_names

    return SleapMetadata(
        bp_names=skeleton1.node_names,
        skeleton=edge_name_list if(len(edge_name_list) > 0) else None,
        orig_skeleton=skeleton1
    )


class SleapModelExtractor(ABC):
    """
    Takes a SLEAP Predictor, and modifies it so that it outputs TrackingData instead of SLEAP predictions.
    """
    supported_models: Optional[Set[Type[SleapPredictor]]] = None

    @abstractmethod
    def __init__(self, model: SleapPredictor):
        self.__p = model

    def get_metadata(self) -> SleapMetadata:
        return _extract_metadata(self.__p)

    @abstractmethod
    def extract(self, data: Union[Provider]) -> Iterator[TrackingData]:
        pass


class BottomUpModelExtractor(SleapModelExtractor):
    from sleap.nn.inference import BottomUpPredictor, BottomUpMultiClassPredictor
    supported_models: Optional[Set[SleapPredictor]] = {BottomUpPredictor, BottomUpMultiClassPredictor}

    def __init__(self, model: Union[BottomUpPredictor, BottomUpMultiClassPredictor]):
        super().__init__(model)
        self._predictor = model

    def extract(self, data: Union[Dict, np.ndarray]) -> TrackingData:
        inf_layer = self._predictor.inference_model.inference_layer
        conf_map, __, offsets = inf_layer.forward_pass(data)
        return TrackingData(
            conf_map.numpy(),
            offsets if(offsets is None) else offsets.numpy(),
            (1 / inf_layer.input_scale) * inf_layer.cm_output_stride
        )


def _extract_model_outputs(inf_layer: SleapInferenceLayer, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    images = inf_layer.keras_model(images)

    conf_map = images
    offset_map = None

    if(isinstance(images, list)):
        conf_map = images[inf_layer.confmaps_ind]
        if(inf_layer.offsets_ind is not None):
            offset_map = images[inf_layer.offsets_ind]

    return (conf_map, offset_map)


class TopDownModelExtractor(SleapModelExtractor):
    from sleap.nn.inference import TopDownPredictor, TopDownMultiClassPredictor
    supported_models: Optional[Set[SleapPredictor]] = {TopDownPredictor, TopDownMultiClassPredictor}

    def __init__(self, model: Union[TopDownPredictor, TopDownMultiClassPredictor]):
        super().__init__(model)
        self._predictor = model

    @staticmethod
    def _merge_tiles(result: Optional[tf.Tensor], batch_sz: int, tile_counts: tuple, orig_im_sz: tuple, d_scale: int) -> Optional[np.ndarray]:
        if(result is None):
            return None

        ceil = lambda n: int(np.ceil(n))
        tiles_wide, tiles_high = tile_counts
        og_w, og_h = orig_im_sz

        __, out_h, out_w, out_d = result.shape

        result = tf.reshape(result, [batch_sz, tiles_high, tiles_wide, out_h, out_w, out_d])
        result = tf.reshape(tf.transpose(result, [0, 1, 3, 2, 4, 5]), [batch_sz, tiles_high * out_h, tiles_wide * out_w, out_d])

        return result[:, :ceil(og_h / d_scale), :ceil(og_w / d_scale)].numpy()

    def extract(self, data: Union[Dict, np.ndarray]) -> TrackingData:
        inf_layer = self._predictor.inference_model.instance_peaks

        imgs = data["image"] if(isinstance(data, dict)) else data
        imgs = inf_layer.preprocess(imgs)

        # Split image into slices to cover entire image....
        batch_size, orig_h, orig_w, orig_d = imgs.shape
        h, w, __ = inf_layer.keras_model.input[0].shape

        imgs = tf.image.extract_patches(
            imgs,
            sizes=[1, h, w, 1],
            strides=[1, h, w, 1],
            rates=[1, 1, 1, 1],
            padding="SAME"
        )

        tiles_high, tiles_wide = imgs.shape[1:3]
        imgs = tf.reshape(imgs, [batch_size * tiles_high * tiles_wide, h, w, orig_d])

        conf_map, offset_map = _extract_model_outputs(inf_layer, imgs)

        return TrackingData(
            self._merge_tiles(conf_map, batch_size, (tiles_wide, tiles_high), (orig_w, orig_h), inf_layer.output_stride),
            self._merge_tiles(offset_map, batch_size, (tiles_wide, tiles_high), (orig_w, orig_h), inf_layer.output_stride),
            (1 / inf_layer.input_scale) * inf_layer.output_stride
        )


class SingleInstanceModelExtractor(SleapModelExtractor):
    from sleap.nn.inference import SingleInstancePredictor
    supported_models: Optional[Set[SleapPredictor]] = {SingleInstancePredictor}

    def __init__(self, model: Union[SingleInstancePredictor]):
        super().__init__(model)
        self._predictor = model

    def extract(self, data: Union[Dict, np.ndarray]) -> TrackingData:
        inf_layer = self._predictor.inference_model.single_instance_layer

        imgs = data["image"] if(isinstance(data, dict)) else data
        conf_map, offset_map = _extract_model_outputs(inf_layer, inf_layer.preprocess(imgs))

        return TrackingData(
            conf_map.numpy(),
            offset_map if(offset_map is None) else offset_map.numpy(),
            (1 / inf_layer.input_scale) * inf_layer.output_stride
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

    def get_metadata(self) -> SleapMetadata:
        return self._model_extractor.get_metadata()

    def extract(self, data: Union[Provider, SleapVideo]) -> Iterator[TrackingData]:
        if(isinstance(data, SleapVideo)):
            data = SleapVideoReader(data)

        pred = self._predictor

        pred.make_pipeline(data)
        if(pred.inference_model is None):
            pred._initialize_inference_model()

        for ex in pred.pipeline.make_dataset():
            yield self._model_extractor.extract(ex)



def _main_test():
    import sleap
    sleap.disable_preallocation()

    import tensorflow as tf
    if(not tf.executing_eagerly()):
        tf.compat.v1.enable_eager_execution()

    mdl = sleap.load_model([
        "/home/isaac/Code/sleap-data/flies13/models/221121_212441.centroid.n=1600",
        "/home/isaac/Code/sleap-data/flies13/models/221121_215804.centered_instance.n=1600"
    ])
    vid = sleap.load_video("/home/isaac/Code/sleap-data/flies13/talk_title_slide@13150-14500.mp4")
    extractor = PredictorExtractor(mdl)

    for frame in extractor.extract(vid):
        print(frame)
        from diplomat.utils.extract_frames import pretty_print_frame
        print(pretty_print_frame(frame, 0, 0, False))
        return



if(__name__ == "__main__"):
    _main_test()




