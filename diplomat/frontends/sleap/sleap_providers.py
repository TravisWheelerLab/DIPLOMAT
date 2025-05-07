import functools
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional, Union, List, Tuple

from .onnx_graph_builder import OnnxVar, OnnxOp, to_onnx_graph_def
from .sleap_imports import onnx, tf2onnx, tf, ort
from numpy.lib.stride_tricks import sliding_window_view
from typing_extensions import TypedDict
import numpy as np
from .run_utils import _dict_get_path
from diplomat.processing import TrackingData
from diplomat.utils.lazy_import import resolve_lazy_imports


class SleapMetadata(TypedDict):
    bp_names: List[str]
    skeleton: Optional[List[Tuple[str, str]]]
    input_scaling: float
    sigma: float
    batch_size: int


ConfigAndModels = List[Tuple[dict, tf.Module]]


def _find_key_nested(data: dict, key: str, default = None):
    for k, v in data.items():
        if(k == key):
            return v
        if(isinstance(v, dict)):
            guess = _find_key_nested(v)
            if(guess is not None):
                return guess

    return default


def _normalize_edges(edge_list) -> List[Tuple[str, str]]:
    return sorted(set(tuple(sorted([str(a), str(b)])) for a, b in edge_list))


def sleap_metadata_from_config(configs: ConfigAndModels) -> SleapMetadata:
    parts = None
    edge_list = None

    for cfg, mdl in configs:
        skeletons = _dict_get_path(cfg, ("data", "labels", "skeletons"), None)
        if(skeletons is not None):
            if(len(skeletons) == 0):
                continue
            skel = skeletons[0]
            parts = [n["id"][0] for n in skel["nodes"]]
            edge_list = _normalize_edges((e["source"][0], e["target"][0]) for e in skel["links"] if(e["type"] == "BODY"))
            break
    else:
        # Scenario 2...
        for cfg, mdl in configs:
            parts = _find_key_nested(cfg["model"]["heads"], "part_names")
            if(parts is None):
                continue
            edge_list = _normalize_edges(
                _find_key_nested(cfg["model"]["heads"], "pafs", {"edges": []})["edges"]
            )
            break

    batch_size = 4

    for cfg, mdl in configs:
        input_scaling = _dict_get_path(cfg, ("data", "preprocessing", "input_scaling"), 1.0)
        for sigma_model_type in ["multi_instance", "multi_class_bottomup", "single_instance", "centered_instance", "multi_class_topdown"]:
            sigma = _dict_get_path(cfg, ("model", "heads", sigma_model_type, "sigma"), None)
            if(sigma is None):
                sigma = _dict_get_path(cfg, ("model", "heads", sigma_model_type, "confmaps", "sigma"), None)
            if(sigma is not None):
                batch_size = int(_dict_get_path(cfg, ("optimization", "batch_size"), 4))
                break
        if(sigma is not None):
            break
    else:
        raise ValueError("Unable to find needed model info!")

    if parts is None or edge_list is None:
        raise ValueError("Unable to find a list of parts in the config files passed!")

    return SleapMetadata(
        bp_names=parts,
        skeleton=edge_list,
        input_scaling=input_scaling,
        sigma=sigma,
        batch_size=batch_size,
    )


class SleapModelExtractor(ABC):
    """
    Takes a SLEAP Predictor, and modifies it so that it outputs TrackingData instead of SLEAP predictions.
    """
    @classmethod
    def can_build(cls, models: ConfigAndModels) -> bool:
        return False

    @abstractmethod
    def __init__(self, models: ConfigAndModels, refinement_kernel_size: int, **kwargs):
        if(not self.can_build(models)):
            raise ValueError("Unable to build with passed model configuration!")
        self.__p = models

    def get_metadata(self) -> SleapMetadata:
        return sleap_metadata_from_config(self.__p)

    @abstractmethod
    def extract(self, data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        pass


def _fix_conf_map(conf_map: np.ndarray) -> np.ndarray:
    return np.clip(conf_map, 0, 1)


def _get_config_paths(cfg, paths, default = None):
    return [
        _dict_get_path(cfg, path, default) for path in paths
    ]


class PreProcessingLayer:
    @resolve_lazy_imports
    def __init__(self, config: dict, **kwargs):
        self._config = config
        self._preprocess_config = _dict_get_path(self._config, ("data", "preprocessing"), {})
        p_c = self._preprocess_config

        # Make variables for onnx stuff were using a bunch...
        TensorProto = onnx.TensorProto
        oh = onnx.helper
        onnx_np_helper = onnx.numpy_helper

        input = OnnxVar("INPUT", TensorProto.FLOAT, [None, None, None, 3])

        is_grayscale = p_c.get("ensure_grayscale", False)
        if is_grayscale:
            rgb_to_gray = OnnxOp("Constant", value=onnx_np_helper.from_array(np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)))
            mdl = OnnxOp("Mul", input, rgb_to_gray)
            mdl = OnnxOp("ReduceSum", mdl, axes=[-1], keepdims=1)
        else:
            gray_to_rgb = OnnxOp("Constant", value=onnx_np_helper.from_array(np.array([1.0, 1.0, 1.0], dtype=np.float32)))
            mdl = OnnxOp("Mul", input, gray_to_rgb)

        if(p_c.get("resize_and_pad_to_target", False)):
            self._t_width = p_c.get("target_width", None)
            self._t_height = p_c.get("target_height", None)
            if(self._t_width is not None and self._t_height is not None):
                desired_size = OnnxOp(
                    "Constant",
                    value=onnx_np_helper.from_array(np.array([self._t_height, self._t_width], dtype=np.int64))
                )
                mdl = OnnxOp(
                    "Resize",
                    mdl, None, None, desired_size,
                    axes=[1, 2],
                    coordinate_transformation_mode="asymmetric",
                    keep_aspect_ratio_policy="not_larger"
                )

        self._post_input_scale = p_c.get("input_scaling", 1.0)
        if(self._post_input_scale != 1.0):
            desired_scale = OnnxOp(
                "Constant", value=onnx_np_helper.from_array(np.array([self._post_input_scale, self._post_input_scale], dtype=np.float32))
            )
            mdl = OnnxOp(
                "Resize",
                mdl, None, desired_scale,
                coordinate_transformation_mode="asymmetric",
                keep_aspect_ratio_policy="not_larger"
            )

        graph = to_onnx_graph_def(
            "ImagePreprocessor",
            [mdl.to_var("OUTPUT", TensorProto.FLOAT, (None, None, None, 1 if is_grayscale else 3))]
        )
        self._preprocess_model = oh.make_model(graph, opset_imports=[onnx.OperatorSetIdProto(version=19)])
        onnx.checker.check_model(self._preprocess_model)
        self._preprocess_runner = _onnx_model_to_inference_session(self._preprocess_model, **kwargs)

        for backbone in self._config["model"]["backbone"].values():
            if(backbone is not None):
                # TODO: Calculate this properly from actual model...
                self._pad_to_stride = int(backbone.get("max_stride", backbone.get("output_stride", 1)))


    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        if(not isinstance(img, np.floating)):
            img = img.astype(np.float32) / np.iinfo(img.dtype).max
        img = np.clip(img, 0, 1, dtype=np.float32)
        downscaling = 1

        if(self._t_width is not None and self._t_height is not None):
            downscaling /= min(self._t_height / img.shape[1], self._t_width / img.shape[2])

        downscaling /= self._post_input_scale

        img = self._preprocess_runner.run(None, {"INPUT": img})[0]

        if(self._pad_to_stride > 1):
            pts = self._pad_to_stride
            new_w = ((img.shape[2] // pts) + (img.shape[2] % pts != 0)) * pts
            new_h = ((img.shape[1] // pts) + (img.shape[1] % pts != 0)) * pts
            img = np.pad(
                img,
                ((0, 0), (0, new_h - img.shape[1]), (0, new_w - img.shape[2]), (0, 0)),
                mode="constant",
                constant_values=0.0
            )

        return img, downscaling


@resolve_lazy_imports
def _onnx_model_to_inference_session(onnx_model, **kwargs) -> ort.InferenceSession:
    b = BytesIO()
    onnx.save(onnx_model, b)
    return ort.InferenceSession(b.getvalue(), **kwargs)


@resolve_lazy_imports
def _reset_input_layer(
    keras_model: tf.keras.Model,
    new_shape: Optional[Tuple[Optional[int], Optional[int], Optional[int], int]] = None,
):
    """Returns a copy of `keras_model` with input shape reset to `new_shape`.

    This method was modified from https://stackoverflow.com/a/58485055.

    Args:
        keras_model: `tf.keras.Model` to return a copy of (with input shape reset).
        new_shape: Shape of the returned model's input layer.

    Returns:
        A copy of `keras_model` with input shape `new_shape`.
    """

    if new_shape is None:
        new_shape = (None, None, None, keras_model.input_shape[-1])

    model_config = keras_model.get_config()
    model_config["layers"][0]["config"]["batch_input_shape"] = new_shape
    new_model = tf.keras.Model.from_config(
        model_config, custom_objects={}
    )  # Change custom objects if necessary

    # Iterate over all the layers that we want to get weights from
    weights = [layer.get_weights() for layer in keras_model.layers]
    for layer, weight in zip(new_model.layers, weights):
        if len(weight) > 0:
            layer.set_weights(weight)

    return new_model


@resolve_lazy_imports
def _keras_to_onnx_model(keras_model) -> onnx.ModelProto:
    input_signature = [
        tf.TensorSpec(keras_model.input_shape, tf.float32, name="image")
    ]
    return tf2onnx.convert.from_keras(keras_model, input_signature, opset=17)[0]


def _find_model_output(model: ort.InferenceSession, name: str, required: bool = True):
    for i, out in enumerate(model.get_outputs()):
        if(out.name == name):
            return i

    if(required):
        raise RuntimeError(f"Unable to find required model output layer {name}.")
    return None


def _resolve_heads(outputs, heads):
    return [
        outputs[h_idx] if(h_idx is not None) else None
        for h_idx in heads
    ]

class BottomUpModelExtractor(SleapModelExtractor):
    MODEL_CONFIGS = [
        ("model", "heads", "multi_instance"),
        ("model", "heads", "multi_class_bottomup")
    ]

    @classmethod
    def can_build(cls, models: ConfigAndModels) -> bool:
        return len(models) == 1 and any(_get_config_paths(models[0][0], cls.MODEL_CONFIGS))

    def __init__(self, models: ConfigAndModels, refinement_kernel_size: int, **kwargs):
        super().__init__(models, refinement_kernel_size)
        self._config, self._model = models[0]
        self._preprocessor = PreProcessingLayer(self._config, **kwargs)
        import onnx
        self._model = _keras_to_onnx_model(_reset_input_layer(self._model))

        offset_ref_name = "OffsetRefinementHead"
        if offset_ref_name not in map(lambda x: x.name, self._model.graph.output) and refinement_kernel_size > 1:
            self._refinement_kernel_size = refinement_kernel_size
            post_model = onnx.helper.make_model(
                _get_integral_offsets_model(refinement_kernel_size, 1.0, "conf_maps", offset_ref_name),
                ir_version=self._model.ir_version,
                opset_imports=[onnx.OperatorSetIdProto(version=self._model.opset_import[0].version)]
            )
            self._model = onnx.compose.merge_models(
                self._model,
                post_model,
                [("MultiInstanceConfmapsHead", "conf_maps")],
                outputs=["MultiInstanceConfmapsHead", offset_ref_name]
            )
        else:
            self._refinement_kernel_size = 0
            for out in self._model.graph.output:
                if out.name == "PartAffinityFieldsHead":
                    self._model.graph.output.remove(out)

        self._model = _onnx_model_to_inference_session(self._model, **kwargs)

        self._heads = [
            _find_model_output(self._model, "MultiInstanceConfmapsHead"),
            _find_model_output(self._model, offset_ref_name, False)
        ]

    def extract(self, data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        data, first_scale_val = self._preprocessor(data)
        conf_map, offsets = _resolve_heads(
            self._model.run(None, {self._model.get_inputs()[0].name: data}),
            self._heads
        )
        cmap_dscale = data.shape[1] / conf_map.shape[1]

        if(offsets is not None and self._refinement_kernel_size > 0):
            offsets *= first_scale_val * cmap_dscale

        return (
            _fix_conf_map(conf_map),
            offsets,
            first_scale_val * cmap_dscale
        )


class TopDownModelExtractor(SleapModelExtractor):
    CENTROID_MODELS = [
        ("model", "heads", "centroid"),
    ]
    CENTERED_INST_MODELS = [
        ("model", "heads", "centered_instance"),
        ("model", "heads", "multi_class_topdown")
    ]


    @classmethod
    def can_build(cls, config: ConfigAndModels) -> bool:
        return (
            len(config) == 2 and
            any(c for cfg, mdl in config for c in _get_config_paths(cfg, cls.CENTROID_MODELS)) and
            any(c for cfg, mdl in config for c in _get_config_paths(cfg, cls.CENTERED_INST_MODELS))
        )

    def __init__(self, models: ConfigAndModels, **kwargs):
        super().__init__(models, **kwargs)
        for cfg, mdl in models:
            if any(_get_config_paths(cfg, self.CENTROID_MODELS)):
                self._centroid_model = _onnx_model_to_inference_session(
                    _keras_to_onnx_model(_reset_input_layer(mdl)), **kwargs
                )
                self._centroid_pre = PreProcessingLayer(cfg, **kwargs)
                self._centroid_cfg = cfg
                self._centroid_heads = [
                    _find_model_output(self._centroid_model, "CentroidConfmapsHead"),
                    _find_model_output(self._centroid_model, "OffsetRefinementHead", False)
                ]
            if any(_get_config_paths(cfg, self.CENTERED_INST_MODELS)):
                self._crop_size = _dict_get_path(cfg, ("data", "instance_cropping", "crop_size"))
                if(self._crop_size is None):
                    raise ValueError("Provided top-down model doesn't have crop size!")
                self._cent_inst_model = _onnx_model_to_inference_session(
                    _keras_to_onnx_model(_reset_input_layer(mdl)), **kwargs
                )
                self._cent_inst_pre = PreProcessingLayer(cfg, **kwargs)
                self._cent_inst_cfg = cfg
                self._cent_inst_heads = [
                    _find_model_output(self._centroid_model, "CenteredInstanceConfmapsHead")
                ]

    @staticmethod
    def _merge_tiles(
        result: Optional[np.ndarray],
        batch_sz: int,
        tile_counts: tuple,
        orig_im_sz: tuple,
        d_scale: int,
    ) -> Union[np.ndarray, np.ndarray, None]:
        if(result is None):
            return None

        ceil = lambda n: int(np.ceil(n))
        tiles_wide, tiles_high = tile_counts
        og_w, og_h = orig_im_sz

        __, out_h, out_w, out_d = result.shape

        result = np.reshape(result, [batch_sz, tiles_high, tiles_wide, out_h, out_w, out_d])
        result = np.reshape(np.transpose(result, [0, 1, 3, 2, 4, 5]), [batch_sz, tiles_high * out_h, tiles_wide * out_w, out_d])
        result = result[:, :ceil(og_h / d_scale), :ceil(og_w / d_scale)]

        return result

    def extract(self, orig_img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        # Run the centroid model to find individuals...
        centroid_img, centroid_dscale = self._centroid_pre(orig_img)
        confs, offsets = _resolve_heads(
            self._centroid_model.run(None, {self._centroid_model.get_inputs()[0].name: centroid_img}),
            self._centroid_heads
        )
        centroid_dscale *= centroid_img.shape[1] / confs.shape[1]
        crop_centers = _local_peak_estimation(confs, offsets, centroid_dscale, local_search_area=5, threshold=0.1, integral_refinement=5)

        crops = _extract_crops(orig_img, crop_centers, self._crop_size)
        crops, inst_dscale = self._cent_inst_pre(crops)
        crops_conf = _resolve_heads(
            self._cent_inst_model.run(None, {self._cent_inst_model.get_inputs()[0].name: crops}),
            self._cent_inst_heads
        )[0]
        inst_dscale *= crops.shape[1] / crops_conf.shape[1]

        conf_h = int(np.ceil(orig_img.shape[1] / inst_dscale) + 1)
        conf_w = int(np.ceil(orig_img.shape[2] / inst_dscale) + 1)

        if(len(crops) == 0):
            img = np.zeros((orig_img.shape[0], conf_h, conf_w, orig_img.shape[-1]), dtype=np.float32)
        else:
            img = _restore_crops(
                (orig_img.shape[0], conf_h, conf_w, orig_img.shape[-1]),
                (crop_centers[0], crop_centers[1] / inst_dscale, crop_centers[2] / inst_dscale, crop_centers[3]),
                crops_conf
            )

        return (
            img,
            None,
            inst_dscale
        )


class SingleInstanceModelExtractor(SleapModelExtractor):
    MODEL_CONFIGS = [
        ("model", "heads", "single_instance"),
    ]

    @classmethod
    def can_build(cls, models: ConfigAndModels) -> bool:
        return len(models) == 1 and any(_get_config_paths(models[0][0], cls.MODEL_CONFIGS))

    def __init__(self, models: ConfigAndModels, refinement_kernel_size: int, **kwargs):
        super().__init__(models, refinement_kernel_size)
        self._config, self._model = models[0]
        self._preprocessor = PreProcessingLayer(self._config, **kwargs)
        self._model = _keras_to_onnx_model(_reset_input_layer(self._model))

        offset_ref_name = "OffsetRefinementHead"
        if offset_ref_name not in map(lambda x: x.name, self._model.graph.output) and refinement_kernel_size > 1:
            self._refinement_kernel_size = refinement_kernel_size
            post_model = onnx.helper.make_model(
                _get_integral_offsets_model(self._refinement_kernel_size, 1.0, "conf_maps", offset_ref_name),
                ir_version=self._model.ir_version,
                opset_imports=[onnx.OperatorSetIdProto(version=self._model.opset_import[0].version)]
            )
            self._model = onnx.compose.merge_models(
                self._model,
                post_model,
                [("SingleInstanceConfmapsHead", "conf_maps")],
                outputs=["SingleInstanceConfmapsHead", offset_ref_name]
            )
        else:
            self._refinement_kernel_size = 0

        self._model = _onnx_model_to_inference_session(self._model, **kwargs)

        self._heads = [
            _find_model_output(self._model, "SingleInstanceConfmapsHead"),
            _find_model_output(self._model, "OffsetRefinementHead", False)
        ]

    def extract(self, data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        data, first_scale_val = self._preprocessor(data)
        conf_map, offsets = _resolve_heads(
            self._model.run(None, {self._model.get_inputs()[0].name: data}),
            self._heads
        )
        cmap_dscale = data.shape[1] / conf_map.shape[1]

        if(offsets is not None and self._refinement_kernel_size > 0):
            offsets *= first_scale_val * cmap_dscale

        return (
            _fix_conf_map(conf_map),
            offsets,
            first_scale_val * cmap_dscale
        )


EXTRACTORS = [BottomUpModelExtractor, SingleInstanceModelExtractor, TopDownModelExtractor]


def _convolve_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Fast-ish manual 2D convolution written using numpy's sliding windows implementation.
    """
    pad_h = kernel.shape[-2] - 1
    pad_w = kernel.shape[-1] - 1
    img = np.pad(
        img,
        ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
    )
    conv_view = sliding_window_view(
        img,
        (kernel.shape[0], kernel.shape[1]),
        (1, 2)
    )
    return np.einsum("...ij,...kij->...k", conv_view, kernel)


def _local_peak_estimation(
    img: np.ndarray,
    offsets: Optional[np.ndarray],
    stride: float,
    local_search_area: int,
    threshold: float,
    integral_refinement: int = 0
):
    pad = local_search_area - 1
    h_pad = pad // 2
    img = np.pad(
        img,
        ((0, 0), (h_pad, pad - h_pad), (h_pad, pad - h_pad), (0, 0))
    )
    conv_view = sliding_window_view(
        img,
        (local_search_area, local_search_area),
        (1, 2)
    )
    center_idx = (local_search_area * local_search_area - 1) // 2
    conv_view = conv_view.reshape(conv_view.shape[:-2] + (local_search_area * local_search_area,))

    peaks = (center_idx != np.argmax(conv_view, axis=-1, keepdims=False)) & (img > threshold)
    rb, rx, ry, rp = np.nonzero(peaks)

    if (offsets is not None):
        offsets_per_crop = offsets[rb, rx, ry, rp]
    elif(integral_refinement > 1):
        if(integral_refinement % 2 == 0):
            integral_refinement += 1
        kernel = _get_integral_offset_kernels(integral_refinement, stride, img.dtype)
        neighborhoods = _extract_crops(img, [rb, rx, ry, rp], integral_refinement)
        offsets_per_crop = np.sum(np.expand_dims(neighborhoods, -1) * kernel, axis=[-3, -2])
    else:
        offsets_per_crop = np.zeros((len(rx), 2), dtype=np.float32)

    true_x = (rx + 0.5) * stride + offsets_per_crop[:, 0]
    true_y = (ry + 0.5) * stride + offsets_per_crop[:, 1]

    return (rb, true_y, true_x, rp)


def _interpolate_crop(x: np.ndarray, y: np.ndarray, crops: np.ndarray) -> np.ndarray:
    crops = np.pad(crops, ((0, 0), (1, 1), (1, 1)))
    x = np.reshape(x % 1, [-1, 1, 1])
    y = np.reshape(y % 1, [-1, 1, 1])

    return (
        x * y * crops[:, 1:, 1:]
        + (1 - x) * y * crops[:, 1:, :-1]
        + x * (1 - y) * crops[:, :-1, 1:]
        + (1 - x) * (1 - y) * crops[:, :-1, :-1]
    )


def _restore_crops(img_shape: tuple, crop_centers: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], crops: np.ndarray):
    batch, y, x, part = crop_centers
    crop_h, crop_w = crops.shape[-2:]
    y = np.clip(y, 0, img_shape[1])
    x = np.clip(x, 0, img_shape[2])
    crop_start_x = np.floor().astype(int)
    crop_end_x = crop_start_x + crop_w
    crop_start_y = np.floor(y - crop_h / 2).astype(int)
    crop_end_y = crop_start_y + crop_h

    pad_x = (-min(0, np.min(crop_start_x)), max(img_shape[2], np.max(crop_end_x)) - img_shape[2])
    pad_y = (-min(0, np.min(crop_start_y)), max(img_shape[1], np.max(crop_end_y)) - img_shape[1])

    img = np.zeros((img_shape[0], img_shape[1] + sum(pad_y), img_shape[2] + sum(pad_x), img_shape[3]), dtype=np.float32)

    crop_shift_x = np.reshape(crop_start_x + pad_x[0], (-1, 1, 1))
    crop_shift_y = np.reshape(crop_start_y + pad_y[0], (-1, 1, 1))
    gy, gx = np.ogrid[0:crop_h, 0:crop_w]

    img[
        np.reshape(batch, (-1, 1, 1)),
        crop_shift_y + gy,
        crop_shift_x + gx,
        np.reshape(part, (-1, 1, 1))
    ] = _interpolate_crop(x - crop_w / 2, y - crop_h / 2, crops)

    return img[:, pad_y[0]:img.shape[1] - pad_y[1], pad_x[0]:img.shape[2] - pad_x[1], :]


def _extract_crops(img: np.ndarray, crop_centers: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], crop_size: int):
    batch, y, x, part = crop_centers
    y = np.clip(y, 0, img.shape[1])
    x = np.clip(x, 0, img.shape[2])
    crop_start_x = np.floor(x - crop_size / 2).astype(int)
    crop_end_x = crop_start_x + crop_size
    crop_start_y = np.floor(y - crop_size / 2).astype(int)
    crop_end_y = crop_start_y + crop_size

    pad_x = (-min(0, np.min(crop_start_x)), max(img.shape[2], np.max(crop_end_x)) - img.shape[2])
    pad_y = (-min(0, np.min(crop_start_y)), max(img.shape[1], np.max(crop_end_y)) - img.shape[1])

    if(any(v != 0 for pad in [pad_x, pad_y] for v in pad)):
        img = np.pad(img, ((0, 0), pad_y, pad_x, (0, 0)))

    crop_shift_x = np.reshape(crop_start_x + pad_x[0], (-1, 1, 1))
    crop_shift_y = np.reshape(crop_start_y + pad_y[0], (-1, 1, 1))
    gy, gx = np.ogrid[0:crop_size, 0:crop_size]

    # Indexing magic...
    crops = img[
        np.reshape(batch, (-1, 1, 1)),
        crop_shift_y + gy,
        crop_shift_x + gx,
        np.reshape(part, (-1, 1, 1))
    ]
    return crops


def _resolve_crop_boxes(img: np.ndarray, centers: np.ndarray, sizes: np.ndarray):
    return img


def _get_integral_offset_kernels(kernel_size: int, stride: float, dtype: np.dtype = np.float32):
    # Construct kernels for computing centers of mass computed around a point...
    kernel_half = (kernel_size - 1) // 2
    y_kernel, x_kernel = [v * stride for v in np.mgrid[-kernel_half:kernel_half + 1, -kernel_half:kernel_half + 1]]
    # Simple summation kernel, adds all values in an area...
    ones_kernel = np.ones((kernel_size, kernel_size), dtype=dtype)
    return np.stack([x_kernel, y_kernel, ones_kernel], axis=0).reshape((3, 1, *x_kernel.shape)).astype(dtype)


@resolve_lazy_imports
def _get_integral_offsets_model(kernel_size: int, stride: float, input_name: str = "conf_maps", output_name: str = "offsets"):
    FLOAT = onnx.TensorProto.FLOAT
    input = OnnxVar(input_name, FLOAT, (None, None, None, None))

    # Reshape to (B, C, 1, H, W)
    mdl = OnnxOp("Unsqueeze", input, OnnxOp("Constant", value_ints=[-1]))
    mdl = OnnxOp("Transpose", mdl, perm=[0, 3, 4, 1, 2])
    # Merge batch and channels dimensions.
    full_shape = OnnxOp("Shape", mdl)
    split_shape = OnnxOp("Split", full_shape, OnnxOp("Constant", value_ints=[2, 3]))
    bc_comb_shape = OnnxOp("Concat", OnnxOp("ReduceProd", split_shape[0], keepdims=1), split_shape[1], axis=0)
    mdl = OnnxOp("Reshape", mdl, bc_comb_shape)
    # Run the convolution...
    mdl = OnnxOp(
        "Conv",
        mdl,
        OnnxOp(
            "Constant",
            value=onnx.numpy_helper.from_array(_get_integral_offset_kernels(kernel_size, stride, np.float32))
        ),
        auto_pad="SAME_UPPER",
    )
    # Reshape to original shape and 2 for x and y coordinates...
    full_shape_new = OnnxOp("Shape", mdl)
    split_shape_new = OnnxOp("Split", full_shape_new, OnnxOp("Constant", value_ints=[1, 3]))
    shape_new = OnnxOp("Concat", split_shape[0], split_shape_new[1], axis=0)
    mdl = OnnxOp("Reshape", mdl, shape_new)
    mdl = OnnxOp("Transpose", mdl, perm=[0, 3, 4, 1, 2])

    # Split out coordinate and Center of Mass kernel to divide by it.
    mdl = OnnxOp("Split", mdl, OnnxOp("Constant", value_ints=[2, 1]), axis=-1)
    mdl = OnnxOp("Div", mdl[0], mdl[1])
    # Nan to zero...
    mdl = OnnxOp("Where", OnnxOp("Equal", mdl, mdl), mdl, OnnxOp("Constant", value_floats=[0.0]))
    mdl = OnnxOp("Mul", mdl, OnnxOp("Constant", value_floats=[stride]))

    return to_onnx_graph_def("IntegralOffsets", [mdl.to_var(output_name, FLOAT, (None, None, None, None, 2))])


def _create_integral_offsets(probs: np.ndarray, stride: float, kernel_size: int) -> np.ndarray:
    """
    Compute estimated offsets for parts based on confidence values in source map. Does this via a
    center-of-mass style calculation locally for each pixel.
    """
    # Concept: We can do localized position integration via 3 convolutions...
    # Two kernels for summing positions * weights in
    if(kernel_size % 2 == 0):
        kernel_size += 1

    # Construct kernels for computing centers of mass computed around a point...
    filters = _get_integral_offset_kernels(kernel_size, stride, probs.dtype)
    results = _convolve_2d(probs, filters)

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(results[:, :, :, :, :2] / results[:, :, :, :, 2:])


class PredictorExtractor:
    def __init__(
        self,
        configs: ConfigAndModels,
        refinement_kernel_size: int,
        **kwargs
    ):
        super().__init__()
        self._configs = configs
        self._refinement_kernel_size = refinement_kernel_size

        for model_extractor in EXTRACTORS:
            if(model_extractor.can_build(configs)):
                self._model_extractor = model_extractor(
                    self._configs,
                    refinement_kernel_size=refinement_kernel_size,
                    **kwargs
                )
                break
        else:
            raise NotImplementedError(f"Could not find model handler for provided model type.")

    def get_metadata(self) -> SleapMetadata:
        return self._model_extractor.get_metadata()

    def extract(self, frames: np.ndarray) -> TrackingData:
        probs, offsets, downscale = self._model_extractor.extract(frames)

        # Trim the resulting outputs so the match expected area for poses from the original video.
        h, w = frames.shape[1:3]
        trim_h, trim_w = int(np.ceil(h / downscale)), int(np.ceil(w / downscale))
        probs = probs[:, :trim_h, :trim_w]
        if(offsets is not None):
            offsets = offsets[:, :trim_h, :trim_w]

        return TrackingData(
            probs,
            offsets,
            downscale
        )

