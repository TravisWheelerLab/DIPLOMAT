from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing_extensions import TypedDict

from diplomat.processing import TrackingData
from diplomat.utils.lazy_import import resolve_lazy_imports

from .run_utils import _dict_get_path, _find_key_nested
from .sleap_imports import sleap_nn, torch


class SleapMetadata(TypedDict):
    bp_names: List[str]
    skeleton: Optional[List[Tuple[str, str]]]
    input_scaling: float
    sigma: float
    batch_size: int


Configs = List[Tuple[dict, str, str]]


# noinspection PyTypeChecker
def _normalize_edges(edge_list) -> List[Tuple[str, str]]:
    return sorted({tuple(sorted([str(a), str(b)])) for a, b in edge_list})


def sleap_metadata_from_config(configs: Configs) -> SleapMetadata:
    parts = None
    edge_list = None

    for cfg, _, _ in configs:
        skeletons = _dict_get_path(cfg, ("data_config", "skeletons"), None)
        if skeletons is not None:
            if len(skeletons) == 0:
                continue
            skel = skeletons[0]
            parts = [n["name"] for n in skel["nodes"]]
            edge_list = _normalize_edges(
                (e["source"]["name"], e["destination"]["name"]) for e in skel["edges"]
            )
            break
    else:
        # Scenario 2...
        for cfg, _, _ in configs:
            parts = _find_key_nested(cfg["model_config"]["head_configs"], "part_names")
            if parts is None:
                continue
            edge_list = _normalize_edges(
                _find_key_nested(
                    cfg["model_config"]["head_configs"], "pafs", {"edges": []}
                )["edges"]
            )
            break

    batch_size = 4

    for cfg, _, _ in configs:
        input_scaling = float(
            _dict_get_path(cfg, ("data_config", "preprocessing", "scale"), 1.0)
        )
        for sigma_model_type in [
            "bottomup",
            "multi_class_bottomup",
            "single_instance",
            "centered_instance",
            "multi_class_topdown",
        ]:
            sigma = _dict_get_path(
                cfg,
                ("model_config", "head_configs", sigma_model_type, "confmaps", "sigma"),
                None,
            )
            if sigma is not None:
                batch_size = int(
                    _dict_get_path(
                        cfg, ("trainer_config", "val_data_loader", "batch_size"), 4
                    )
                )
                break
        if sigma is not None:
            break
    else:
        raise ValueError("Unable to find needed model info!")

    if parts is None or edge_list is None:
        raise ValueError("Unable to find a list of parts in the config files passed!")

    return SleapMetadata(
        bp_names=parts,
        skeleton=edge_list,
        input_scaling=input_scaling,
        sigma=float(sigma),
        batch_size=batch_size,
    )


class SleapModelExtractor(ABC):
    """
    Takes a SLEAP Predictor, and modifies it so that it outputs TrackingData instead of SLEAP predictions.
    """

    @classmethod
    def can_build(cls, models: Configs) -> bool:
        return False

    @abstractmethod
    def __init__(
        self, models: Configs, device: str, refinement_kernel_size: int, **kwargs
    ):
        if not self.can_build(models):
            raise ValueError("Unable to build with passed model configuration!")
        self.__p = models

    def get_metadata(self) -> SleapMetadata:
        return sleap_metadata_from_config(self.__p)

    @abstractmethod
    def extract(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        pass


def _fix_conf_map(conf_map: np.ndarray) -> np.ndarray:
    return np.clip(conf_map, 0, 1)


class IntegralOffsets:
    @staticmethod
    @resolve_lazy_imports
    def make_kernel(kernel_size, stride, dtype):
        # Construct kernels for computing centers of mass computed around a point...
        kernel_half = (kernel_size - 1) // 2
        y_kernel, x_kernel = [
            v * stride
            for v in torch.meshgrid(
                torch.arange(-kernel_half, kernel_half + 1, dtype=dtype),
                torch.arange(-kernel_half, kernel_half + 1, dtype=dtype),
                indexing="ij",
            )
        ]
        # Simple summation kernel, adds all values in an area...
        ones_kernel = torch.ones((kernel_size, kernel_size), dtype=dtype)
        return (
            torch.stack([x_kernel, y_kernel, ones_kernel], dim=0)
            .reshape((3, 1, *x_kernel.shape))
            .to(dtype)
        )

    @resolve_lazy_imports
    def __init__(self, kernel_refinement_size: int, stride: float, device: str):
        import torch

        self._kernel_refinement_size = kernel_refinement_size
        self._kernel = self.make_kernel(
            kernel_refinement_size, stride, torch.float32
        ).to(device)

    @resolve_lazy_imports
    def __call__(self, confmap):
        batch, height, width, channels = confmap.shape
        integration_fields = (
            torch.nn.functional.conv2d(
                confmap.permute(0, 3, 1, 2).reshape(batch * channels, 1, height, width),
                self._kernel,
                padding="same",
            )
            .reshape(batch, channels, 3, height, width)
            .permute(0, 3, 4, 1, 2)
        )

        offsets = integration_fields[..., :2] / integration_fields[..., 2:]
        return torch.where(torch.isfinite(offsets), offsets, 0.0)


class BottomUpModelExtractor(SleapModelExtractor):
    MODEL_CONFIGS = {
        "multi_instance": "MultiInstanceConfmapsHead",
        "bottomup": "MultiInstanceConfmapsHead",
        "multi_class_bottomup": "MultiInstanceConfmapsHead",
        "single_instance": "SingleInstanceConfmapsHead",
    }

    @classmethod
    def model_type(cls, config: dict) -> Optional[str]:
        for key in cls.MODEL_CONFIGS:
            if (
                _dict_get_path(config, ("model_config", "head_configs", key))
                is not None
            ):
                return key
        return None

    @classmethod
    def can_build(cls, models: Configs) -> bool:
        return len(models) == 1 and cls.model_type(models[0][0]) is not None

    def __init__(
        self, models: Configs, device: str, refinement_kernel_size: int, **kwargs
    ):
        super().__init__(models, device, refinement_kernel_size, **kwargs)
        self._config, self._config_path, self._model_path = models[0]
        model_type = self.model_type(self._config)
        if model_type is None:
            raise ValueError("Provided config not a supported model!")

        self._predictor = sleap_nn.inference.Predictor.from_model_paths(
            [self._config_path],
            device=device,
            batch_size=1,
            return_confmaps=True,
            integral_refinement="integral",
        )
        self._confmaps_output_name = self.MODEL_CONFIGS[model_type]
        self._integral_offsets = (
            IntegralOffsets(refinement_kernel_size, 1.0, device)
            if refinement_kernel_size > 1
            else None
        )

    def extract(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        x, _info = self._predictor.layer.preprocess(data)
        outputs = self._predictor.layer.backend(x)
        confmaps = outputs[self._confmaps_output_name].detach()  # (B, N, H, W)
        confmaps = confmaps.permute(0, 2, 3, 1)  # (B, H, W, N)
        cmap_dscale = float(
            max(data.shape[1] / confmaps.shape[1], data.shape[2] / confmaps.shape[2])
        )

        if self._integral_offsets is not None:
            offsets = self._integral_offsets(confmaps)
            offsets = (offsets * cmap_dscale).cpu().numpy()
        else:
            offsets = None

        return _fix_conf_map(confmaps.cpu().numpy()), offsets, cmap_dscale


class TopDownModelExtractor(SleapModelExtractor):
    CENTROID_MODELS = ["centroid"]
    CENTERED_INST_MODELS = ["centered_instance", "multi_class_topdown"]

    CENTROID_KEY = "CentroidConfmapsHead"
    CONFMAP_KEY = "CenteredInstanceConfmapsHead"

    @classmethod
    def model_type(cls, config: dict) -> Optional[str]:
        for key in [*cls.CENTROID_MODELS, *cls.CENTERED_INST_MODELS]:
            if (
                _dict_get_path(config, ("model_config", "head_configs", key))
                is not None
            ):
                return key
        return None

    @classmethod
    def can_build(cls, configs: Configs) -> bool:
        model_types = [cls.model_type(cfg) for cfg, _, _ in configs]

        return (
            len(configs) == 2
            and any(cm in model_types for cm in cls.CENTROID_MODELS)
            and any(ci in model_types for ci in cls.CENTERED_INST_MODELS)
        )

    def __init__(
        self, configs: Configs, device: str, refinement_kernel_size: int, **kwargs
    ):
        super().__init__(configs, device, refinement_kernel_size, **kwargs)
        self._predictor = sleap_nn.inference.Predictor.from_model_paths(
            [cfg_path for _, cfg_path, _ in configs],
            device=device,
            batch_size=1,
            return_confmaps=True,
            integral_refinement="integral",
        )

        self._crop_size = None

        for cfg, _, _ in configs:
            if self.model_type(cfg) in self.CENTERED_INST_MODELS:
                self._crop_size = _dict_get_path(
                    cfg, ("data_config", "preprocessing", "crop_size")
                )

        if self._crop_size is None:
            raise ValueError("Provided top-down model doesn't have crop size!")

        self._integral_refinement = (
            IntegralOffsets(refinement_kernel_size, 1.0, device)
            if refinement_kernel_size > 1
            else None
        )
        self._device = device

    @resolve_lazy_imports
    def _local_peak_estimation(
        self,
        img: torch.Tensor,
        stride: float,
        local_search_area: int,
        threshold: float,
    ):
        if local_search_area % 2 == 0:
            local_search_area += 1

        if self._integral_refinement is not None:
            offsets = self._integral_refinement(img.permute(0, 2, 3, 1))
        else:
            offsets = None

        max_neighbors = torch.nn.functional.max_pool2d(
            img, local_search_area, 1, local_search_area // 2
        )

        peaks = (img >= max_neighbors) & (img > threshold)
        rb, rp, ry, rx = torch.nonzero(peaks, as_tuple=True)

        if offsets is not None:
            img_x = (rx + 0.5) * stride + offsets[rb, ry, rx, rp, 0] * stride
            img_y = (ry + 0.5) * stride + offsets[rb, ry, rx, rp, 1] * stride
        else:
            img_x = (rx + 0.5) * stride
            img_y = (ry + 0.5) * stride

        return (rb, img_y, img_x, rp)

    @resolve_lazy_imports
    def _extract_crops(self, img, crop_centers, crop_size):
        batch, y, x, _part = crop_centers
        y = torch.clamp(y, 0, img.shape[1])
        x = torch.clamp(x, 0, img.shape[2])
        crop_start_x = torch.floor(x - crop_size / 2).to(torch.int64)
        crop_end_x = crop_start_x + crop_size
        crop_start_y = torch.floor(y - crop_size / 2).to(torch.int64)
        crop_end_y = crop_start_y + crop_size

        # Add padding to image for crops near the edges to not attempt indexing out of bounds...
        pad_x = (
            -min(0, torch.min(crop_start_x)),
            max(img.shape[2], torch.max(crop_end_x)) - img.shape[2],
        )
        pad_y = (
            -min(0, torch.min(crop_start_y)),
            max(img.shape[1], torch.max(crop_end_y)) - img.shape[1],
        )

        if any(v != 0 for pad in [pad_x, pad_y] for v in pad):
            # Torch padding direction is backwards....
            img = torch.nn.functional.pad(img, (0, 0, *pad_x, *pad_y))

        crop_shift_x = torch.reshape(crop_start_x + pad_x[0], (-1, 1, 1))
        crop_shift_y = torch.reshape(crop_start_y + pad_y[0], (-1, 1, 1))
        gy, gx = [
            v.to(crop_shift_x.device)
            for v in torch.meshgrid(
                torch.arange(crop_size), torch.arange(crop_size), indexing="ij"
            )
        ]

        # Indexing magic...
        crops = img[
            torch.reshape(batch, (-1, 1, 1)), crop_shift_y + gy, crop_shift_x + gx, :
        ]
        return crops

    @resolve_lazy_imports
    def _restore_crops(self, img_shape, crop_centers, crops):
        batch, y, x, _part = crop_centers
        crop_h, crop_w = crops.shape[1:3]
        y = torch.clamp(y, 0, img_shape[1])
        x = torch.clamp(x, 0, img_shape[2])
        crop_start_x = torch.floor(x - crop_w / 2).to(torch.int64)
        crop_end_x = crop_start_x + crop_w
        crop_start_y = torch.floor(y - crop_h / 2).to(torch.int64)
        crop_end_y = crop_start_y + crop_h

        pad_x = (
            -min(0, torch.min(crop_start_x)),
            max(img_shape[2], torch.max(crop_end_x)) - img_shape[2],
        )
        pad_y = (
            -min(0, torch.min(crop_start_y)),
            max(img_shape[1], torch.max(crop_end_y)) - img_shape[1],
        )

        img = torch.zeros(
            (
                img_shape[0],
                img_shape[1] + sum(pad_y),
                img_shape[2] + sum(pad_x),
                img_shape[3],
            ),
            dtype=crops.dtype,
            device=crops.device,
        )

        crop_shift_x = torch.reshape(crop_start_x + pad_x[0], (-1, 1, 1))
        crop_shift_y = torch.reshape(crop_start_y + pad_y[0], (-1, 1, 1))
        gy, gx = [
            v.to(crop_shift_x.device)
            for v in torch.meshgrid(
                torch.arange(crop_h), torch.arange(crop_w), indexing="ij"
            )
        ]

        img[
            torch.reshape(batch, (-1, 1, 1)), crop_shift_y + gy, crop_shift_x + gx, :
        ] = _interpolate_crop(x - crop_w / 2, y - crop_h / 2, crops)

        return img[
            :, pad_y[0] : img.shape[1] - pad_y[1], pad_x[0] : img.shape[2] - pad_x[1], :
        ]

    def extract(
        self, orig_img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        # Run the centroid model to find individuals...
        centroid_layer = self._predictor.layer.centroid_layer
        centered_inst_layer = self._predictor.layer.centered_instance_layer
        cent_out = centroid_layer.backend(centroid_layer.preprocess(orig_img)[0])[
            "output"
        ]
        cent_dscale = float(orig_img.shape[2] / cent_out.shape[2])

        crop_centers = self._local_peak_estimation(
            cent_out, cent_dscale, local_search_area=5, threshold=0.1
        )

        tensor_img = torch.tensor(orig_img).to(self._device)

        extracted_crops = self._extract_crops(tensor_img, crop_centers, self._crop_size)
        extracted_crops = extracted_crops.permute(0, 3, 1, 2)

        center_inst_out = centered_inst_layer.backend(
            centered_inst_layer.preprocess(extracted_crops)[0]
        )["output"].permute(0, 2, 3, 1)

        inst_dscale = float(extracted_crops.shape[2] / center_inst_out.shape[2])

        confmaps = self._restore_crops(
            (
                int(orig_img.shape[0]),
                int(orig_img.shape[1] / inst_dscale),
                int(orig_img.shape[2] / inst_dscale),
                int(center_inst_out.shape[-1]),
            ),
            (
                crop_centers[0],
                crop_centers[1] / inst_dscale,
                crop_centers[2] / inst_dscale,
                crop_centers[3],
            ),
            center_inst_out,
        )

        if self._integral_refinement is not None:
            offsets = self._integral_refinement(confmaps)
            offsets = (offsets * inst_dscale).cpu().numpy()
        else:
            offsets = None

        return _fix_conf_map(confmaps.cpu().numpy()), offsets, inst_dscale


EXTRACTORS = [
    BottomUpModelExtractor,
    TopDownModelExtractor,
]


def _interpolate_crop(x, y, crops):
    crops = torch.nn.functional.pad(crops, (0, 0, 0, 1, 0, 1))
    x = torch.reshape(x % 1, [-1, 1, 1, 1])
    y = torch.reshape(y % 1, [-1, 1, 1, 1])

    return (
        x * y * crops[:, 1:, 1:]
        + (1 - x) * y * crops[:, 1:, :-1]
        + x * (1 - y) * crops[:, :-1, 1:]
        + (1 - x) * (1 - y) * crops[:, :-1, :-1]
    )


class PredictorExtractor:
    def __init__(
        self, configs: Configs, device: str, refinement_kernel_size: int, **kwargs
    ):
        super().__init__()
        self._configs = configs
        self._refinement_kernel_size = refinement_kernel_size

        for model_extractor in EXTRACTORS:
            if model_extractor.can_build(configs):
                self._model_extractor = model_extractor(
                    self._configs,
                    device=device,
                    refinement_kernel_size=refinement_kernel_size,
                    **kwargs,
                )
                break
        else:
            raise NotImplementedError(
                f"Could not find model handler for provided model type."
            )

    def get_metadata(self) -> SleapMetadata:
        return self._model_extractor.get_metadata()

    def extract(self, frames: np.ndarray) -> TrackingData:
        probs, offsets, downscale = self._model_extractor.extract(frames)

        # Trim the resulting outputs so they match expected area for poses from the original video.
        h, w = frames.shape[1:3]
        trim_h, trim_w = int(np.ceil(h / downscale)), int(np.ceil(w / downscale))
        probs = probs[:, :trim_h, :trim_w]
        if offsets is not None:
            offsets = offsets[:, :trim_h, :trim_w]

        return TrackingData(probs, offsets, downscale)
