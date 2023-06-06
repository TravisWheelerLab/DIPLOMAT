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


def _normalize_conf_map(conf_map: tf.Tensor) -> tf.Tensor:
    conf_map = tf.maximum(conf_map, 0)
    max_val = tf.reduce_max(conf_map, axis=(1, 2), keepdims=True)
    return conf_map / max_val


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
            _normalize_conf_map(conf_map),
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
        # TODO: Eventually fix top down support to actually work.
        raise NotImplementedError("SLEAP's top down model is currently not supported. Please train using a "
                                  "different model type to use DIPLOMAT.")

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

    def extract(self, data: Union[Dict, np.ndarray]) -> Tuple[tf.Tensor, Optional[tf.Tensor], float]:
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

        return (
            _normalize_conf_map(self._merge_tiles(conf_map, batch_size, (tiles_wide, tiles_high), (orig_w, orig_h), inf_layer.output_stride)),
            self._merge_tiles(offset_map, batch_size, (tiles_wide, tiles_high), (orig_w, orig_h), inf_layer.output_stride),
            (1 / inf_layer.input_scale) * inf_layer.output_stride
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
            _normalize_conf_map(conf_map),
            offset_map,
            (1 / first_scale_val) * (1 / inf_layer.input_scale) * inf_layer.output_stride
        )


EXTRACTORS = [BottomUpModelExtractor, SingleInstanceModelExtractor, TopDownModelExtractor]


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


    @staticmethod
    def _convolve_2d(img: tf.Tensor, kernel: tf.Tensor):
        """
        Fast-ish manual 2D convolution written in python, because tf's convolution function gives stupid and unresolvable errors...
        """
        shift_h = kernel.shape[0] // 2
        shift_w = kernel.shape[1] // 2

        padding = tf.transpose(tf.constant([[0, shift_h, shift_w, 0, 0]] * 2))
        img = tf.reshape(tf.repeat(img, kernel.shape[-1], axis=-1), img.shape + (3,))
        result = tf.zeros(img.shape, dtype=img.dtype)
        img = tf.pad(img, padding)

        # We iterate over the kernel for optimal performance...
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                end_i = img.shape[1] - ((shift_h * 2) - i)
                end_j = img.shape[2] - ((shift_w * 2) - j)
                result += kernel[i, j] * img[:, i:end_i, j:end_j]

        return result

    @classmethod
    def _create_integral_offsets(cls, probs: tf.Tensor, stride: float, kernel_size: int) -> tf.Tensor:
        # Concept: We can do localized position integration via 3 convolutions...
        # Two kernels for summing positions * weights in
        if(kernel_size % 2 == 0):
            kernel_size += 1

        # Construct kernels for computing centers of mass computed around a point...
        y_kernel = tf.reshape(tf.repeat((tf.range(kernel_size, dtype=probs.dtype) - (kernel_size // 2)) * stride, kernel_size), (5, 5))
        x_kernel = tf.transpose(y_kernel)
        # Simple summation kernel, adds all values in an area...
        ones_kernel = tf.ones((kernel_size, kernel_size), dtype=probs.dtype)

        filters = tf.stack([x_kernel, y_kernel, ones_kernel], axis=-1)

        results = cls._convolve_2d(probs, filters)

        return tf.math.divide_no_nan(results[:, :, :, :, :2], results[:, :, :, :, 2:])

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
                offsets = self._create_integral_offsets(probs, downscale, self._refinement_kernel_size)

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

