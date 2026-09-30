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


class FrameExtractor:
    def __init__(self, model_config, pose_runner, detector_runner):
        self._config = model_config
        self._pose_runner = pose_runner
        self._detector_runner = detector_runner
        self._device = next(pose_runner.model.parameters()).device

    def _run_batched(self, func, frames, context):
        new_context = {}
        new_frames = []

        def batched_iter(ctx):
            for *items in zip(ctx):
                

        for frame in frames:
            func(frame, context)

    @resolve_lazy_imports
    def __call__(self, frames: np.ndarray) -> TrackingData:
        full_context = {}
        frames_proc = []

        for frame in frames:
            preproc_frames, context = self._pose_runner.preprocessor(frames, {})
            frames.append(frames_proc)
        model_kwargs = context.pop("model_kwargs", {})
        for i in range(10):
            print(preproc_frames.shape)
        print(self._pose_runner.predict(preproc_frames, **model_kwargs))
        raise ValueError()


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
    snapshot_dir: Union[str, Path], prefix: str, snapshot_epoch: Optional[int] = None
) -> Path:
    snapshots = []

    for file in Path(snapshot_dir).iterdir():
        if not file.is_dir() and file.suffix == ".pt" and file.stem.startswith(prefix):
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
        dlc_project.model_directory / "train", "snapshot-", snapshot_epoch
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
            dlc_project.model_directory / "train", "detector-snapshot-", snapshot_epoch
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
            ),
        )
