import tempfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Optional
from zipfile import is_zipfile, ZipFile

import numpy as np
import yaml

import diplomat.processing.type_casters as tc
from diplomat.frontends import ModelInfo, ModelLike
from diplomat.frontends.deeplabcut._verify_func import _load_dlc_like_zip_file
from diplomat.frontends.deeplabcut.dlc_importer import ort, tf, tf2onnx, onnx
from diplomat.processing import TrackingData
from diplomat.utils.cli_tools import Flag


def _get_model_folder(cfg: dict, project_root: Path, shuffle: int = 1, train_fraction: float = None, model_prefix: str = ""):
    task = cfg["Task"]
    date = cfg["date"]
    iterate = f"iteration-{str(cfg['iteration'])}"
    train_fraction = train_fraction if(train_fraction is not None) else cfg["TrainingFraction"][0]
    model_prefix = "" if(model_prefix in ["..", "."] or "/" in model_prefix or "\\" in model_prefix) else model_prefix
    return Path(project_root) / Path(
        model_prefix,
        "dlc-models",
        iterate,
        f"{task}{date}-trainset{str(int(train_fraction * 100))}shuffle{str(shuffle)}"
    )


def _build_provider_ordering(device_index: Optional[int], use_cpu: bool):
    supported_devices = ort.get_available_providers()
    device_config = []

    if(not use_cpu):
        if("CUDAExecutionProvider" in supported_devices):
            device_config.append(("CUDAExecutionProvider", {"device_id": device_index}))
        if("ROCMExecutionProvider" in supported_devices):
            device_config.append(("ROCMExecutionProvider", {"device_id": device_index}))
        if("CoreMLExecutionProvider" in supported_devices):
            device_config.append("CoreMLExecutionProvider")

    # Fallback...
    device_config.append("CPUExecutionProvider")
    return device_config

# TODO: DLC model loading...
# part_pred no grad or Adam in it.
# locref_pred no grad or Adam in it (pose/pairwise_pred/block4/BiasAdd)...
def load_tf_model():
    pass


def _load_and_convert_model(model_dir: Path, device_index: Optional[int], use_cpu: bool):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    meta_files = [file for file in model_dir.iterdir() if file.stem.startswith("snapshot-") and file.suffix == "meta"]
    latest_meta_file = max(meta_files, key=lambda k: int(k.stem.split("-")[-1]))
    checkpoint_file = model_dir / latest_meta_file.stem

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(str(latest_meta_file), clear_devices=True)
        saver.restore(sess, str(checkpoint_file))

        node_set = set(node.name for node in sess.graph.get_operations()[0])
        output_names = ["pose/part_pred/block4/BiasAdd"]
        locref_name = "pose/locref_pred/block4/BiasAdd"
        if(locref_name in node_set):
            output_names.append(locref_name)

        model, __ = tf2onnx.convert.from_graph_def(
            sess.graph.as_graph_def(),
            "DLCModel",
            ["Placeholder"],
            output_names
        )

        b = BytesIO()
        onnx.save(model, b)

    return ort.InferenceSession(
        b.getvalue(),
        providers=_build_provider_ordering(device_index, use_cpu)
    )


class FakeTempDir:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FrameExtractor:
    def __init__(self, onnx_model: ort.InferenceSession):
        self._model = onnx_model

    def __call__(self, frames: np.ndarray) -> TrackingData:
        raise ValueError()


@tc.typecaster_function
def load_model(
    config: tc.PathLike,
    num_outputs: tc.Optional[int] = None,
    batch_size: tc.Optional[int] = None,
    gpu_index: tc.Optional[int] = None,
    model_prefix: str = "",
    shuffle: int = 1,
    training_set_index: int = 0,
    use_cpu: Flag = False
) -> tc.Tuple[ModelInfo, ModelLike]:
    """
    Run DIPLOMAT tracking on videos using a DEEPLABCUT project and trained network.

    :param config: The path to the DLC config for the DEEPLABCUT project.
    :param shuffle: int, optional. Integer specifying which TrainingsetFraction to use. By default, the first
                    (note that TrainingFraction is a list in config.yaml).
    :param training_set_index: int, optional. Integer specifying which TrainingsetFraction to use. By default the first
                               (note that TrainingFraction is a list in config.yaml).
    :param gpu_index: Integer index of the GPU to use for inference (in tensorflow) defaults to 0, or selecting the first detected GPU if available.
    :param batch_size: The batch size to use while processing. Defaults to None, which uses the default batch size for the project.
    :param model_prefix: The string prefix of the DEEPLABCUT model to use defaults to no prefix (the default model).
    :param num_outputs: The number of outputs, or bodies to track in the video. Defaults to the value specified in the DLC config, or None if one
                        is not specified.
    :param use_cpu: If True, run on cpu even if a gpu is available. Defaults to False.

    :return: A model info dictionary, and a deeplabcut model wrapper that can be used to estimate poses from video frames.
    """
    if(isinstance(config, (tuple, list))):
        if(len(config) != 1):
            raise ValueError("Can't pass multiple config files!")
        config = config[0]

    if(is_zipfile(config)):
        tmp_dir = tempfile.TemporaryDirectory()
        is_zip = True
    else:
        tmp_dir = FakeTempDir(str(config))
        is_zip = False

    with tmp_dir as tmp_dir:
        if is_zip:
            with ZipFile(config, "r") as z:
                sub_path, config = _load_dlc_like_zip_file(z)
                z.extractall(tmp_dir.name)
                project_dir = (Path(tmp_dir.name) / sub_path).resolve().parent
        else:
            with open(config, "rb") as f:
                config = yaml.load(f, yaml.SafeLoader)
            project_dir = Path(config).resolve().parent

    iteration = config["iteration"]
    train_frac = config["TrainingFraction"][training_set_index]
    config["project_path"] = project_dir

    model_directory = _get_model_folder(config, project_dir, shuffle, train_frac, model_prefix)
    model_directory = model_directory.resolve()

    try:
        with (model_directory / "test" / "pose_cfg.yaml").open("rb") as f:
            model_config = yaml.load(f, yaml.SafeLoader)
    except FileNotFoundError as e:
        print(e)
        raise FileNotFoundError(f"Invalid model selection: (Iteration {iteration}, Training Fraction {train_frac}, Shuffle: {shuffle})")

    # Set the number of outputs...
    num_outputs = int(
        config.get("num_outputs", model_config.get("num_outputs", None)) if(num_outputs is None) else num_outputs
    )
    batch_size = batch_size if(batch_size is not None) else config["batch_size"]
    body_parts = list(model_config["all_joints_names"])
    if("partaffinityfield_graph" in model_config):
        skeleton = sorted(set(tuple(sorted([body_parts[a], body_parts[b]])) for a, b in model_config["partaffinityfield_graph"]))
    else:
        skeleton = []

    return (
        ModelInfo(
            num_outputs=num_outputs,
            batch_size=batch_size,
            dotsize=int(config.get("dotsize", 4)),
            colormap=config.get("colormap", None),
            shape_list=None,
            alphavalue=config.get("alphavalue", 0.7),
            pcutoff=config.get("pcutoff", 0.1),
            line_thickness=1,
            bp_names=body_parts,
            skeleton=skeleton,
            frontend="deeplabcut"
        ),
        FrameExtractor(_load_and_convert_model(model_directory / "train", gpu_index, bool(use_cpu)))
    )