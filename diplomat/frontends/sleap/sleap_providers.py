from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Iterator, Set, Type, List, Tuple

from typing_extensions import TypedDict
import numpy as np

from .sleap_importer import (
    tf,
    SleapDataConfig,
    SleapVideo,
    Provider,
    SleapVideoReader,
    SleapPredictor,
    SleapInferenceLayer,
    SleapSkeleton
)

from diplomat.processing import TrackingData


class SleapMetadata(TypedDict):
    bp_names: List[str]
    skeleton: Optional[List[Tuple[str, str]]]
    orig_skeleton: SleapSkeleton


def sleap_metadata_from_config(config: SleapDataConfig) -> SleapMetadata:
    skel_list = config.labels.skeletons

    if (len(skel_list) < 1):
        raise ValueError("No part information for this SLEAP project, can't run diplomat!")

    skeleton1 = skel_list[0]
    edge_name_list = skeleton1.edge_names

    return SleapMetadata(
        bp_names=skeleton1.node_names,
        skeleton=edge_name_list if (len(edge_name_list) > 0) else None,
        orig_skeleton=skeleton1
    )


class SleapModelExtractor(ABC):
    """
    Takes a SLEAP Predictor, and modifies it so that it outputs TrackingData instead of SLEAP predictions.
    """
    @classmethod
    def supported_models(cls) -> Set[SleapPredictor]:
        return set()

    @abstractmethod
    def __init__(self, model: SleapPredictor):
        self.__p = model

    def get_metadata(self) -> SleapMetadata:
        return sleap_metadata_from_config(self.__p.data_config)

    @abstractmethod
    def extract(self, data: Union[Provider]) -> Tuple[tf.Tensor, Optional[tf.Tensor], float]:
        pass


def _fix_conf_map(conf_map: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(conf_map, 0, 1)


class BottomUpModelExtractor(SleapModelExtractor):
    @classmethod
    def supported_models(cls) -> Set[SleapPredictor]:
        from sleap.nn.inference import BottomUpPredictor, BottomUpMultiClassPredictor
        return {BottomUpPredictor, BottomUpMultiClassPredictor}

    def __init__(self, model: SleapPredictor):
        super().__init__(model)
        self._predictor = model

    def extract(self, data: Union[Dict, np.ndarray]) -> Tuple[tf.Tensor, Optional[tf.Tensor], float]:
        inf_layer = self._predictor.inference_model.inference_layer
        conf_map, __, offsets = inf_layer.forward_pass(data)

        first_scale_val = float(tf.reshape(data["scale"], -1)[0])
        if(not tf.experimental.numpy.allclose(data["scale"], first_scale_val)):
            raise ValueError("Scaling is not consistent!")

        return (
            _fix_conf_map(conf_map),
            offsets if(offsets is None) else offsets,
            (1 / first_scale_val) * (1 / inf_layer.input_scale) * inf_layer.cm_output_stride
        )


def _extract_model_outputs(inf_layer: SleapInferenceLayer, images: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    images = inf_layer.keras_model(images)

    conf_map = images
    offset_map = None

    if(isinstance(images, list)):
        conf_map = images[inf_layer.confmaps_ind]
        if(inf_layer.offsets_ind is not None):
            offset_map = images[inf_layer.offsets_ind]

    return (conf_map, offset_map)


class TopDownModelExtractor(SleapModelExtractor):
    @classmethod
    def supported_models(cls) -> Set[SleapPredictor]:
        from sleap.nn.inference import TopDownPredictor, TopDownMultiClassPredictor
        return {TopDownPredictor, TopDownMultiClassPredictor}

    def __init__(self, model: SleapPredictor):
        super().__init__(model)
        self._predictor = model

    @staticmethod
    def _merge_tiles(
        result: Optional[tf.Tensor],
        batch_sz: int,
        tile_counts: tuple,
        orig_im_sz: tuple,
        d_scale: int,
    ) -> Union[np.ndarray, tf.Tensor, None]:
        if(result is None):
            return None

        ceil = lambda n: int(np.ceil(n))
        tiles_wide, tiles_high = tile_counts
        og_w, og_h = orig_im_sz

        __, out_h, out_w, out_d = result.shape

        result = tf.reshape(result, [batch_sz, tiles_high, tiles_wide, out_h, out_w, out_d])
        result = tf.reshape(tf.transpose(result, [0, 1, 3, 2, 4, 5]), [batch_sz, tiles_high * out_h, tiles_wide * out_w, out_d])
        result = result[:, :ceil(og_h / d_scale), :ceil(og_w / d_scale)]

        return result

    @classmethod
    def _interpolate_crop(cls, x: tf.Tensor, y: tf.Tensor, crops: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x % 1, [-1, 1, 1, 1])
        y = tf.reshape(y % 1, [-1, 1, 1, 1])

        return (
            x * y * crops[:, :-1, :-1]
            + (1 - x) * y * crops[:, :-1, 1:]
            + x * (1 - y) * crops[:, 1:, :-1]
            + (1 - x) * (1 - y) * crops[:, 1:, 1:]
        )

    def extract(self, data: Union[Dict, np.ndarray]) -> Tuple[tf.Tensor, Optional[tf.Tensor], float]:
        import tensorflow as tf

        inf_layer = self._predictor.inference_model.instance_peaks

        imgs = data["image"] if(isinstance(data, dict)) else data

        # Compute centroid confidence map...
        centroid_model = self._predictor.inference_model.centroid_crop
        centroid_model.return_crops = True
        centroid_results = centroid_model.call(imgs)

        first_scale_val = float(tf.reshape(data["scale"], -1)[0])
        if(not tf.experimental.numpy.allclose(data["scale"], first_scale_val)):
            raise ValueError("Scaling is not consistent!")

        crops, crop_offsets = centroid_results["crops"], centroid_results["crop_offsets"]

        batch_indexes = crops.value_rowids()
        batch_size = crops.nrows()

        crops = crops.merge_dims(0, 1)
        crop_offsets = crop_offsets.merge_dims(0, 1)

        crops = inf_layer.preprocess(crops)
        sub_conf_maps, __ = _extract_model_outputs(inf_layer, crops)

        model_downscaling = (1 / inf_layer.input_scale) * inf_layer.output_stride

        conf_h = int(np.ceil(imgs.shape[1] / model_downscaling) + 1)
        conf_w = int(np.ceil(imgs.shape[2] / model_downscaling) + 1)
        img_buffer = tf.zeros([batch_size, conf_h, conf_w, sub_conf_maps.shape[-1]], dtype=tf.float32)

        if(len(crops) == 0):
            return (img_buffer, None,  (1 / first_scale_val) * model_downscaling)

        sub_conf_maps = tf.pad(sub_conf_maps, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
        crop_coords = tf.stack([
            tf.repeat(tf.range(sub_conf_maps.shape[1] - 1), sub_conf_maps.shape[2] - 1),
            tf.tile(tf.range(sub_conf_maps.shape[2] - 1), [sub_conf_maps.shape[1] - 1])
        ], axis=-1)

        crop_offsets = crop_offsets / model_downscaling
        lookup_coords = (
            tf.expand_dims(tf.cast(crop_offsets[..., ::-1], dtype=tf.int32), 1) + crop_coords
        )

        index_vectors = tf.concat([
            tf.broadcast_to(tf.reshape(batch_indexes, [-1, 1, 1]), [*lookup_coords.shape[:-1], 1]),
            lookup_coords
        ], axis=-1)

        img_buffer = tf.tensor_scatter_nd_update(
            img_buffer,
            index_vectors,
            tf.reshape(
                self._interpolate_crop(crop_offsets[..., 0], crop_offsets[..., 1], sub_conf_maps),
                [sub_conf_maps.shape[0], -1, sub_conf_maps.shape[-1]]
            )
        )

        return (
            img_buffer,
            None,
            (1 / first_scale_val) * model_downscaling
        )


class SingleInstanceModelExtractor(SleapModelExtractor):
    @classmethod
    def supported_models(cls) -> Set[SleapPredictor]:
        from sleap.nn.inference import SingleInstancePredictor
        return {SingleInstancePredictor}

    def __init__(self, model: SleapPredictor):
        super().__init__(model)
        self._predictor = model

    def extract(self, data: Union[Dict, np.ndarray]) -> Tuple[tf.Tensor, Optional[tf.Tensor], float]:
        inf_layer = self._predictor.inference_model.single_instance_layer

        imgs = data["image"] if(isinstance(data, dict)) else data
        conf_map, offset_map = _extract_model_outputs(inf_layer, inf_layer.preprocess(imgs))

        first_scale_val = float(tf.reshape(data["scale"], -1)[0])
        if (not tf.experimental.numpy.allclose(data["scale"], first_scale_val)):
            raise ValueError("Scaling is not consistent!")

        return (
            _fix_conf_map(conf_map),
            offset_map,
            (1 / first_scale_val) * (1 / inf_layer.input_scale) * inf_layer.output_stride
        )


EXTRACTORS = [BottomUpModelExtractor, SingleInstanceModelExtractor, TopDownModelExtractor]


def _convolve_2d(img: tf.Tensor, kernel: tf.Tensor):
    """
    Fast-ish manual 2D convolution written using tensorflow's extract_patches, because tf's convolution function gives
    errors when attempting to use directly.
    """
    orig_img_shape = tuple(img.shape)
    img = tf.reshape(tf.transpose(img, perm=[0, 3, 1, 2]), [-1, orig_img_shape[1], orig_img_shape[2]])

    conv_view = tf.image.extract_patches(
        tf.expand_dims(img, -1),
        sizes=[1, kernel.shape[0], kernel.shape[1], 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )

    return tf.transpose(
        tf.reshape(tf.reduce_sum(
            tf.expand_dims(conv_view, -1) * tf.reshape(kernel, [-1, kernel.shape[-1]]),
            -2
        ), [-1, orig_img_shape[3], orig_img_shape[1], orig_img_shape[2], kernel.shape[-1]]),
        perm=[0, 2, 3, 1, 4]
    )


def _create_integral_offsets(probs: tf.Tensor, stride: float, kernel_size: int) -> tf.Tensor:
    """
    Compute estimated offsets for parts based on confidence values in source map. Does this via a
    center-of-mass style calculation locally for each pixel.
    """
    # Concept: We can do localized position integration via 3 convolutions...
    # Two kernels for summing positions * weights in
    if(kernel_size % 2 == 0):
        kernel_size += 1

    # Construct kernels for computing centers of mass computed around a point...
    y_kernel = tf.reshape(
        tf.repeat((tf.range(kernel_size, dtype=probs.dtype) - (kernel_size // 2)) * stride, kernel_size),
        (kernel_size, kernel_size)
    )
    x_kernel = tf.transpose(y_kernel)
    # Simple summation kernel, adds all values in an area...
    ones_kernel = tf.ones((kernel_size, kernel_size), dtype=probs.dtype)

    filters = tf.stack([x_kernel, y_kernel, ones_kernel], axis=-1)

    results = _convolve_2d(probs, filters)

    return tf.math.divide_no_nan(results[:, :, :, :, :2], results[:, :, :, :, 2:])


class PredictorExtractor:
    def __init__(self, predictor: SleapPredictor, refinement_kernel_size: int):
        super().__init__()
        self._predictor = predictor
        self._refinement_kernel_size = refinement_kernel_size

        for model_extractor in EXTRACTORS:
            if(type(predictor) in model_extractor.supported_models()):
                self._model_extractor = model_extractor(self._predictor)
                break
        else:
            raise NotImplementedError(f"The provided predictor, '{type(predictor).__name__}', is not supported yet!")

    def get_metadata(self) -> SleapMetadata:
        return self._model_extractor.get_metadata()

    def extract(self, data: Union[Provider, SleapVideo]) -> Iterator[TrackingData]:
        from sleap import Video as SleapVideo
        if(isinstance(data, SleapVideo)):
            data = SleapVideoReader(data)

        pred = self._predictor

        pred.make_pipeline(data)
        if(pred.inference_model is None):
            pred._initialize_inference_model()

        for ex in pred.pipeline.make_dataset():
            probs, offsets, downscale = self._model_extractor.extract(ex)

            if(offsets is None and self._refinement_kernel_size > 0):
                offsets = _create_integral_offsets(probs, downscale, self._refinement_kernel_size)

            # Trim the resulting outputs so the match expected area for poses from the original video.
            h, w = data.video.shape[1:3]
            trim_h, trim_w = int(np.ceil(h / downscale)), int(np.ceil(w / downscale))
            probs = probs[:, :trim_h, :trim_w]

            offsets = offsets[:, :trim_h, :trim_w]

            yield TrackingData(
                probs.cpu().numpy(),
                None if(offsets is None) else offsets.cpu().numpy(),
                downscale
            )

