import contextlib
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Optional, Union
from zipfile import ZipFile, is_zipfile

import numpy as np
import yaml

import diplomat.processing.type_casters as tc
from diplomat.frontends import ModelInfo, ModelLike
from diplomat.processing import TrackingData
from diplomat.utils.cli_tools import Flag
from diplomat.utils.lazy_import import resolve_lazy_imports

from ._verify_func import _load_dlc_like_zip_file
from .dlc_importer import dlc_torch, torch


def _get_model_folder(
    cfg: dict,
    project_root: Path,
    shuffle: int = 1,
    train_fraction: Optional[float] = None,
    model_prefix: str = "",
    is_pytorch: bool = True,
) -> Path:
    task = cfg["Task"]
    date = cfg["date"]
    iterate = f"iteration-{str(cfg['iteration'])}"
    train_fraction = (
        train_fraction if (train_fraction is not None) else cfg["TrainingFraction"][0]
    )
    model_prefix = (
        ""
        if (model_prefix in ["..", "."] or "/" in model_prefix or "\\" in model_prefix)
        else model_prefix
    )

    return Path(project_root) / Path(
        model_prefix,
        "dlc-models-pytorch" if is_pytorch else "dlc-models",
        iterate,
        f"{task}{date}-trainset{str(int(train_fraction * 100))}shuffle{str(shuffle)}",
    )


@resolve_lazy_imports
def _resolve_device(gpu_index: Optional[int], use_cpu: bool) -> str:
    if use_cpu:
        return "cpu"
    if torch.cuda.is_available():
        if gpu_index is not None:
            return f"cuda:{gpu_index}"
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FakeTempDir:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


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


def _interpolate_crop_single(x, y, crops):
    crops = torch.nn.functional.pad(crops, (0, 0, 0, 1, 0, 1))
    x = x % 1
    y = y % 1

    res = (
        x * y * crops[1:, 1:, :]
        + (1 - x) * y * crops[1:, :-1, :]
        + x * (1 - y) * crops[:-1, 1:, :]
        + (1 - x) * (1 - y) * crops[:-1, :-1, :]
    )

    return res


def _correct_crop_range(start, end, max_value):
    clamp = lambda x: max(0, min(max_value, x))
    new_start = -start if start < 0 else 0
    c_start = clamp(start)
    c_end = clamp(end)

    if c_start >= c_end:
        return None, None

    return (
        slice(c_start, c_end),
        slice(new_start, new_start + (c_end - c_start)),
    )


class FrameExtractor:
    @resolve_lazy_imports
    def __init__(
        self,
        model_config,
        pose_runner,
        detector_runner,
        score_threshold=0.25,
        refinement_kernel_size=0,
    ):
        self._config = model_config
        self._pose_runner = pose_runner
        self._detector_runner = detector_runner
        self._integral_refiner = None
        self._locref_stdev = float(
            _dict_get_path(
                model_config, ("model", "bodypart", "predictor", "locref_stdev"), 7.2801
            )
        )
        self._apply_sigmoid = bool(
            _dict_get_path(
                model_config, ("model", "bodypart", "predictor", "apply_sigmoid"), True
            )
        )
        self._clip_scores = bool(
            _dict_get_path(
                model_config, ("model", "bodypart", "predictor", "clip_scores"), False
            )
        )
        self._score_threshold = score_threshold

        if refinement_kernel_size is not None and refinement_kernel_size > 2:
            self._integral_refiner = IntegralOffsets(
                refinement_kernel_size, 1.0, self._pose_runner.device
            )

    @resolve_lazy_imports
    def _cat_values(self, val):
        if isinstance(val[0], np.ndarray):
            return np.concatenate(val, axis=0)
        if isinstance(val[0], torch.Tensor):
            return torch.cat(val, dim=0)
        else:
            raise ValueError(f"Model arguments must be tensors or numpy arrays!")

    def _run_batched(self, func, frames, context, extraction_key=None):
        new_context = {}
        new_frames = []

        def batched_iter(frames_in, ctx):
            if isinstance(ctx, list):
                yield from zip(frames_in, ctx)
            else:
                keys = list(ctx.keys())
                min_len = min(
                    len(frames_in),
                    min((len(v) for v in ctx.values()), default=len(frames_in)),
                )

                for i in range(min_len):
                    yield (frames_in[i], {k: ctx[k][i] for k in keys})

        for img, inner_ctx in batched_iter(frames, context):
            res_img, res_ctx = func(img, inner_ctx)
            if extraction_key is not None:
                res_ctx = res_ctx.pop(extraction_key, {})
            new_frames.append(res_img)
            for key in res_ctx:
                if key not in new_context:
                    new_context[key] = []
                new_context[key].append(res_ctx[key])

        return (
            self._cat_values(new_frames),
            {k: self._cat_values(v) for k, v in new_context.items()},
        )

    @resolve_lazy_imports
    def _bottom_up_impl(self, frames: np.ndarray):
        pr = self._pose_runner

        preproc_frames, model_kwargs = self._run_batched(
            pr.preprocessor, frames, {}, "model_kwargs"
        )

        if pr.inference_cfg.autocast.enabled:
            with torch.autocast(device_type=str(pr.device)):
                outputs = pr.model(preproc_frames.to(pr.device), **model_kwargs)
        else:
            outputs = pr.model(preproc_frames.to(pr.device), **model_kwargs)

        bodypart_data = outputs["bodypart"]
        heatmap = bodypart_data["heatmap"].permute(0, 2, 3, 1)

        if self._apply_sigmoid:
            heatmap = torch.sigmoid(heatmap)
        if self._clip_scores:
            heatmap = torch.clamp(heatmap, 0.0, 1.0)

        downscale = float(
            max(
                frames.shape[1] / heatmap.shape[1],
                frames.shape[2] / heatmap.shape[2],
            )
        )
        locref = bodypart_data.get("locref", None)

        if locref is not None:
            locref = locref.permute(0, 2, 3, 1)
            b, h, w, _c = locref.shape
            locref = locref.reshape(b, h, w, -1, 2) * self._locref_stdev
        elif self._integral_refiner is not None:
            locref = self._integral_refiner(heatmap) * downscale

        return TrackingData(
            heatmap.cpu().numpy(),
            locref.cpu().numpy() if locref is not None else None,
            downscale,
        )

    @resolve_lazy_imports
    def _restore_crop(
        self,
        heatmap_buffer,
        score_buffer,
        crop_x,
        crop_y,
        crop_scale_x,
        crop_scale_y,
        crop,
        score,
    ):
        # Crop is h, w, num_parts...
        crop_old_h, crop_old_w = crop.shape[:2]

        from torchvision.transforms.functional import resize

        crop_proper_size = resize(
            crop.permute(2, 0, 1),
            [int(crop_old_h * crop_scale_y), int(crop_old_w * crop_scale_x)],
        ).permute(1, 2, 0)

        crop_h, crop_w = crop_proper_size.shape[:2]

        crop_start_x = int(np.floor(crop_x))
        crop_end_x = crop_start_x + crop_w
        crop_start_y = int(np.floor(crop_y))
        crop_end_y = crop_start_y + crop_h

        x_dst, x_src = _correct_crop_range(
            crop_start_x, crop_end_x, int(min(crop_w, heatmap_buffer.shape[1]))
        )
        y_dst, y_src = _correct_crop_range(
            crop_start_y, crop_end_y, int(min(crop_h, heatmap_buffer.shape[0]))
        )

        if x_dst is None or y_dst is None:
            return

        score_buffer[y_dst, x_dst, :] += score
        heatmap_buffer[y_dst, x_dst, :] += (
            _interpolate_crop_single(crop_x, crop_y, crop_proper_size) * score
        )[y_src, x_src, :]

    @resolve_lazy_imports
    def _top_down_impl(self, frames: np.ndarray):
        det = self._detector_runner
        pr = self._pose_runner

        det_frames, det_kwargs = self._run_batched(
            det.preprocessor, frames, {}, "model_kwargs"
        )
        det_boxes = det.predict(det_frames, **det_kwargs)

        reconstructed_heatmaps = None
        reconstructed_score_sum = None
        downscale = 1.0

        for batch_i, (frm, detection) in enumerate(zip(frames, det_boxes)):
            detection_data = detection["detection"]
            det_filter = detection_data["scores"] > self._score_threshold
            detection_data = {
                k: v[det_filter]
                for k, v in detection_data.items()
                if k in ("scores", "bboxes")
            }

            pose_frame, detection_data = pr.preprocessor(frm, detection_data)
            pose_kwargs = detection_data.pop("model_kwargs", {})

            if pr.inference_cfg.autocast.enabled:
                with torch.autocast(device_type=str(pr.device)):
                    outputs = pr.model(pose_frame.to(pr.device), **pose_kwargs)
            else:
                outputs = pr.model(pose_frame.to(pr.device), **pose_kwargs)

            bodypart_data = outputs["bodypart"]
            heatmap = bodypart_data["heatmap"].permute(0, 2, 3, 1)

            if self._apply_sigmoid:
                heatmap = torch.sigmoid(heatmap)
            if self._clip_scores:
                heatmap = torch.clamp(heatmap, 0.0, 1.0)

            downscale = float(
                max(
                    pose_frame.shape[1] / heatmap.shape[1],
                    pose_frame.shape[2] / heatmap.shape[2],
                )
            )

            if reconstructed_heatmaps is None:
                reconstructed_heatmaps = torch.zeros(
                    (
                        frames.shape[0],
                        int(np.ceil(frames.shape[1] / downscale)),
                        int(np.ceil(frames.shape[2] / downscale)),
                        heatmap.shape[3],
                    ),
                    device=heatmap.device,
                    dtype=heatmap.dtype,
                )
                reconstructed_score_sum = torch.zeros(
                    (
                        frames.shape[0],
                        int(np.ceil(frames.shape[1] / downscale)),
                        int(np.ceil(frames.shape[2] / downscale)),
                        heatmap.shape[3],
                    ),
                    device=heatmap.device,
                    dtype=heatmap.dtype,
                )

            for indv_hm, offset, scale, score in zip(
                heatmap,
                detection_data["offsets"],
                detection_data["scales"],
                detection_data["scores"],
            ):
                self._restore_crop(
                    reconstructed_heatmaps[batch_i],
                    reconstructed_score_sum[batch_i],
                    offset[0] / downscale,
                    offset[1] / downscale,
                    scale[0],
                    scale[1],
                    indv_hm,
                    score,
                )

        reconstructed_score_sum = torch.clamp(reconstructed_score_sum, min=1)
        heatmaps_final = reconstructed_heatmaps / reconstructed_score_sum

        if self._integral_refiner is not None:
            locref = self._integral_refiner(heatmaps_final) * downscale
        else:
            locref = None

        return TrackingData(
            heatmaps_final.cpu().numpy(),
            locref.cpu().numpy() if locref is not None else None,
            downscale,
        )

    @resolve_lazy_imports
    def __call__(self, frames: np.ndarray) -> TrackingData:
        with torch.no_grad():
            if self._detector_runner is None:
                return self._bottom_up_impl(frames)
            else:
                # TODO: Need to find a new approach, initial sleap-like approach gives really poor results...
                # return self._top_down_impl(frames)
                raise NotImplementedError(
                    "DIPLOMAT doesn't support DeepLabCut top-down models currently. Please use a bottom-up model instead."
                )


class DLCProject(NamedTuple):
    config: dict
    model_train_config: dict
    model_directory: Path


@contextlib.contextmanager
def _load_dlc_project(
    config_path: str,
    model_prefix: str = "",
    shuffle: int = 1,
    training_set_index: int = 0,
):
    if is_zipfile(config_path):
        tmp_dir = tempfile.TemporaryDirectory()
        is_zip = True
    else:
        tmp_dir = FakeTempDir(str(config_path))
        is_zip = False

    with tmp_dir as tmp_dir:
        if is_zip:
            with ZipFile(config_path, "r") as z:
                config_in_zip, config_dict = _load_dlc_like_zip_file(z)
                zip_project_dir = PurePosixPath(config_in_zip).parent
                for zip_info in z.infolist():
                    if zip_info.is_dir():
                        continue
                    zip_path_obj = PurePosixPath(zip_info.filename)
                    try:
                        sub_path = zip_path_obj.relative_to(zip_project_dir)
                        if sub_path.parts[0] not in [
                            "dlc-models-pytorch",
                            "config.yaml",
                        ]:
                            continue
                        dst_path = Path(tmp_dir, sub_path).resolve()
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(zip_info, "r") as fsrc:
                            with dst_path.open("wb") as fdst:
                                shutil.copyfileobj(fsrc, fdst)
                    except ValueError:
                        pass
                project_dir = Path(tmp_dir)
        else:
            project_dir = Path(config_path).resolve().parent
            with open(config_path, "rb") as f:
                config_dict = yaml.load(f, yaml.SafeLoader)

        iteration = config_dict["iteration"]
        train_frac = config_dict["TrainingFraction"][training_set_index]

        model_directory = _get_model_folder(
            config_dict, project_dir, shuffle, train_frac, model_prefix
        )
        model_directory = model_directory.resolve()

        try:
            with (model_directory / "train" / "pytorch_config.yaml").open("rb") as f:
                model_train_config = yaml.load(f, yaml.SafeLoader)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Invalid model selection: (Iteration {iteration}, Training Fraction {train_frac}, Shuffle: {shuffle})"
            )

        yield DLCProject(config_dict, model_train_config, model_directory)


def _select_snapshot(
    snapshot_dir: Union[str, Path],
    prefix: str,
    avoid_prefix: Optional[str] = None,
    snapshot_epoch: Optional[int] = None,
) -> Path:
    snapshots = []

    for file in Path(snapshot_dir).iterdir():
        if (
            not file.is_dir()
            and file.suffix == ".pt"
            and file.stem.startswith(prefix)
            and (avoid_prefix is None or not file.stem.startswith(avoid_prefix))
        ):
            try:
                parts = file.stem.split("-")
                epoch = int(parts[-1])
                is_best = parts[-2] == "best" if len(parts) >= 2 else False
                snapshots.append((is_best, epoch, file))
            except ValueError:
                continue

    if snapshot_epoch is None:
        return max(snapshots)[2]
    else:
        filtered_snapshots = list(filter(lambda v: v[1] == snapshot_epoch, snapshots))
        if len(filtered_snapshots) == 0:
            raise ValueError(
                f"Invalid epoch provided, no model snapshot with epoch {snapshot_epoch} exists!"
            )
        return max(filtered_snapshots)[2]


@resolve_lazy_imports
def _load_dlc_pose_runner(
    dlc_project: DLCProject,
    num_outputs: int,
    batch_size: int,
    device_str: str,
    snapshot_epoch: Optional[int] = None,
):
    # import deeplabcut.pose_estimation_pytorch as dlc_torch
    task = dlc_torch.Task(dlc_project.model_train_config["method"])
    snapshot_path = _select_snapshot(
        dlc_project.model_directory / "train",
        "snapshot-",
        "snapshot-detector-",
        snapshot_epoch,
    )

    pose_runner = dlc_torch.get_pose_inference_runner(
        model_config=dlc_project.model_train_config,
        snapshot_path=snapshot_path,
        max_individuals=num_outputs,
        batch_size=batch_size,
        device=device_str,
    )

    if task == dlc_torch.Task.TOP_DOWN:
        detector_snapshot_path = _select_snapshot(
            dlc_project.model_directory / "train",
            "snapshot-detector-",
            None,
            snapshot_epoch,
        )
        detector_runner = dlc_torch.get_detector_inference_runner(
            model_config=dlc_project.model_train_config,
            snapshot_path=detector_snapshot_path,
            max_individuals=num_outputs,
            batch_size=batch_size,
            device=device_str,
        )
        return (pose_runner, detector_runner)

    return (pose_runner, None)


def _dict_get_path(dict_obj, key, default=None):
    for k_p in key:
        if not isinstance(dict_obj, dict) or k_p not in dict_obj:
            return default
        dict_obj = dict_obj[k_p]
    return dict_obj


@tc.typecaster_function
def load_model(
    config: tc.PathLike,
    num_outputs: tc.Optional[int] = None,
    batch_size: tc.Optional[int] = None,
    gpu_index: tc.Optional[int] = None,
    snapshot_epoch: tc.Optional[int] = None,
    refinement_kernel_size: int = 5,
    model_prefix: str = "",
    shuffle: int = 1,
    training_set_index: int = 0,
    use_cpu: Flag = False,
) -> tc.Tuple[ModelInfo, ModelLike]:
    """
    Run DIPLOMAT tracking on videos using a DEEPLABCUT project and trained network.

    :param config: The path to the DLC config for the DEEPLABCUT project.
    :param shuffle: int, optional. Integer specifying which TrainingsetFraction to use. By default, the first
                    (note that TrainingFraction is a list in config.yaml).
    :param training_set_index: int, optional. Integer specifying which TrainingsetFraction to use. By default the first
                               (note that TrainingFraction is a list in config.yaml).
    :param gpu_index: Integer index of the GPU to use for inference (in tensorflow) defaults to 0, or selecting the first detected GPU if available.
    :param snapshot_epoch: Integer being the epoch of the saved model snapshot to use for inference. If not set, uses the best snapshot if available,
                           otherwise using the latest snapshot.
    :param refinement_kernel_size: Size of refinement kernel used for computing offsets if an offset map is not generated by the model. Defaults to 5.
    :param batch_size: The batch size to use while processing. Defaults to None, which uses the default batch size for the project.
    :param model_prefix: The string prefix of the DEEPLABCUT model to use defaults to no prefix (the default model).
    :param num_outputs: The number of outputs, or bodies to track in the video. Defaults to the value specified in the DLC config, or None if one
                        is not specified.
    :param use_cpu: If True, run on cpu even if a gpu is available. Defaults to False.

    :return: A model info dictionary, and a deeplabcut model wrapper that can be used to estimate poses from video frames.
    """
    if isinstance(config, (tuple, list)):
        if len(config) != 1:
            raise ValueError("Can't pass multiple config files!")
        config = config[0]

    device_str = _resolve_device(gpu_index, use_cpu)

    with _load_dlc_project(
        config, model_prefix, shuffle, training_set_index
    ) as dlc_project:
        # Set the number of outputs...
        if num_outputs is None:
            if "individuals" in dlc_project.config:
                num_outputs = len(dlc_project.config["individuals"])
            elif "individuals" in dlc_project.model_train_config["metadata"]:
                num_outputs = len(
                    dlc_project.model_train_config["metadata"]["individuals"]
                )
            else:
                num_outputs = dlc_project.config.get(
                    "num_outputs",
                    dlc_project.model_train_config["metadata"].get("num_outputs", None),
                )

        if num_outputs is not None:
            num_outputs = int(num_outputs)

        batch_size = (
            batch_size if (batch_size is not None) else dlc_project.config["batch_size"]
        )
        body_parts = list(dlc_project.model_train_config["metadata"]["bodyparts"])
        paf_graph = _dict_get_path(
            dlc_project.model_train_config,
            ("model", "heads", "bodypart", "predictor", "graph"),
        )
        if paf_graph is not None:
            keep_edges = _dict_get_path(
                dlc_project.model_train_config,
                ("model", "heads", "bodypart", "predictor", "edges_to_keep"),
                range(len(paf_graph)),
            )
            skeleton_list = [
                tuple(sorted([body_parts[a], body_parts[b]]))
                for a, b in paf_graph
                if (0 <= a < len(body_parts)) and (0 <= b < len(body_parts))
            ]
            if keep_edges is not None:
                skeleton = sorted({skeleton_list[i] for i in keep_edges})
        else:
            skeleton = []

        return (
            ModelInfo(
                num_outputs=num_outputs,
                batch_size=batch_size,
                dotsize=int(dlc_project.config.get("dotsize", 4)),
                colormap=dlc_project.config.get("colormap", None),
                shape_list=None,
                alphavalue=dlc_project.config.get("alphavalue", 0.7),
                pcutoff=dlc_project.config.get("pcutoff", 0.1),
                line_thickness=1,
                bp_names=body_parts,
                skeleton=skeleton,
                frontend="deeplabcut",
            ),
            FrameExtractor(
                dlc_project.model_train_config,
                *_load_dlc_pose_runner(
                    dlc_project,
                    num_outputs,
                    batch_size,
                    device_str,
                    snapshot_epoch,
                ),
                refinement_kernel_size=refinement_kernel_size,
            ),
        )
