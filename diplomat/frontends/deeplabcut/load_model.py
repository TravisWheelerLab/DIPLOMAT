import shutil
import tempfile
from io import BytesIO
from pathlib import Path, PosixPath, PurePosixPath
from typing import Optional, List
from zipfile import is_zipfile, ZipFile

import numpy as np
import yaml

import diplomat.processing.type_casters as tc
from diplomat.frontends import ModelInfo, ModelLike
from ._verify_func import _load_dlc_like_zip_file
from .dlc_importer import ort, tf, onnx
from diplomat.processing import TrackingData
from diplomat.utils.cli_tools import Flag


def _get_model_folder(
    cfg: dict,
    project_root: Path,
    shuffle: int = 1,
    train_fraction: float = None,
    model_prefix: str = "",
    is_pytorch: bool = False,
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


def _build_provider_ordering(device_index: Optional[int], use_cpu: bool):
    supported_devices = ort.get_available_providers()
    device_config = []

    def _add(val, extra=None):
        if extra is None:
            extra = {}
        if device_index is not None:
            extra["device_id"] = device_index
        return (val, extra)

    if not use_cpu:
        if "CUDAExecutionProvider" in supported_devices:
            device_config.append(_add("CUDAExecutionProvider"))
        if "ROCMExecutionProvider" in supported_devices:
            device_config.append(_add("ROCMExecutionProvider"))
        if "CoreMLExecutionProvider" in supported_devices:
            device_config.append("CoreMLExecutionProvider")

    # Fallback...
    device_config.append("CPUExecutionProvider")
    return device_config


def _prune_tf_model(graph_def, outputs: List[str]):
    name_to_idx = {n.name: i for i, n in enumerate(graph_def.node)}
    visited = [False] * len(name_to_idx)

    if not all(o in name_to_idx for o in outputs):
        raise ValueError("Not all output nodes exist in the model!")

    stack = []
    stack.extend(outputs)

    while len(stack) > 0:
        node_name = stack.pop()
        idx = name_to_idx[node_name]
        visited[idx] = True
        for input_node_name in graph_def.node[idx].input:
            if (
                input_node_name in name_to_idx
                and not visited[name_to_idx[input_node_name]]
            ):
                stack.append(input_node_name)

    temp_stack = []
    for i in range(len(visited) - 1, -1, -1):
        node = graph_def.node.pop()
        if visited[i]:
            temp_stack.append(node)
    graph_def.node.extend(temp_stack[::-1])

    print(f"Total nodes: {len(visited)}")
    print(f"Removed nodes: {len(visited) - sum(visited)}")

    return graph_def


def _load_meta_graph_def(meta_file):
    meta_graph_def = tf.compat.v1.MetaGraphDef()
    with open(meta_file, "rb") as f:
        meta_graph_def.MergeFromString(f.read())
    return meta_graph_def


def from_checkpoint(model_path, input_names, output_names):
    """Load tensorflow graph from checkpoint."""
    import tensorflow as tf
    import tf2onnx

    tf_v1 = tf.compat.v1
    # make sure we start with clean default graph
    tf_v1.reset_default_graph()
    # model_path = checkpoint/checkpoint.meta
    with tf.device("/cpu:0"):
        with tf_v1.Session() as sess:
            saver = tf_v1.train.import_meta_graph(model_path, clear_devices=True)
            # restore from model_path minus the ".meta"
            sess.run(tf_v1.global_variables_initializer())
            saver.restore(sess, model_path[:-5])
            input_names = tf2onnx.tf_loader.inputs_without_resource(sess, input_names)
            frozen_graph = tf2onnx.tf_loader.freeze_session(
                sess, input_names=input_names, output_names=output_names
            )
            input_names = tf2onnx.tf_loader.remove_redundant_inputs(
                frozen_graph, input_names
            )

        tf_v1.reset_default_graph()
        with tf_v1.Session() as sess:
            frozen_graph = tf2onnx.tf_loader.tf_optimize(
                input_names, output_names, frozen_graph
            )
    tf_v1.reset_default_graph()
    return frozen_graph, input_names, output_names


def _find_direct_consumers(graph_def, node):
    consumers = []
    for n in graph_def.node:
        for i, ins in enumerate(n.input):
            if ins == node:
                consumers.append(f"{n.name}:{i}")

    return consumers


def _get_dlc_inputs_and_outputs(meta_path):
    meta_graph_def = _load_meta_graph_def(meta_path)

    desired_outputs = [
        ("pose/part_pred/block4/BiasAdd:0", True),
        ("pose/locref_pred/block4/BiasAdd:0", False),
    ]
    output_names = []
    op_names = {n.name for n in meta_graph_def.graph_def.node}

    for op_name, is_required in desired_outputs:
        op_only = op_name.split(":")[0]
        if op_only in op_names:
            output_names.append(op_name)
        elif is_required:
            raise ValueError(
                f"Unable to find weights for layer: {op_name} in DLC model, which is required."
            )

    input_names = ["fifo_queue_Dequeue:0"]
    for input_name in input_names:
        if input_name.split(":")[0] not in op_names:
            raise ValueError("Can't find input node!")
    return input_names, output_names


def _load_and_convert_model(
    model_dir: Path, device_index: Optional[int], use_cpu: bool
):
    import tensorflow as tf
    import tensorflow.compat.v1 as tf_v1
    import tf2onnx

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.reset_default_graph()

    meta_files = [
        file
        for file in model_dir.iterdir()
        if file.stem.startswith("snapshot-") and file.suffix == ".meta"
    ]
    if len(meta_files) == 0:
        raise ValueError(
            "No checkpoint files, make sure you've trained a DLC model first!"
        )
    latest_meta_file = max(meta_files, key=lambda k: int(k.stem.split("-")[-1]))

    inputs, outputs = _get_dlc_inputs_and_outputs(str(latest_meta_file))

    graph_def, inputs, outputs = from_checkpoint(str(latest_meta_file), inputs, outputs)

    model, __ = tf2onnx.convert.from_graph_def(
        graph_def,
        name=str(latest_meta_file.name),
        input_names=inputs,
        output_names=outputs,
        shape_override={inputs[0]: [None, None, None, 3]},
        opset=17,
    )

    b = BytesIO()
    onnx.save(model, b)

    return ort.InferenceSession(
        b.getvalue(), providers=_build_provider_ordering(device_index, use_cpu)
    )


class FakeTempDir:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FrameExtractor:
    def __init__(self, onnx_model: ort.InferenceSession, model_config: dict):
        self._model = onnx_model
        self._image_input_name = self._model.get_inputs()[0].name
        self._config = model_config

    def __call__(self, frames: np.ndarray) -> TrackingData:
        outputs = self._model.run(
            None, {self._image_input_name: frames.astype(np.float32)}
        )

        locref = outputs[1] if (len(outputs) > 1) else None

        if locref is not None:
            locref = locref.reshape((*locref.shape[:-1], -1, 2))
            locref *= self._config["locref_stdev"]

        return TrackingData(
            1 / (1 + np.exp(-outputs[0])),
            locref,
            float(
                np.ceil(
                    max(
                        frames.shape[1] / outputs[0].shape[1],
                        frames.shape[2] / outputs[0].shape[2],
                    )
                )
            ),
        )


@tc.typecaster_function
def load_model(
    config: tc.PathLike,
    num_outputs: tc.Optional[int] = None,
    batch_size: tc.Optional[int] = None,
    gpu_index: tc.Optional[int] = None,
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

    if is_zipfile(config):
        tmp_dir = tempfile.TemporaryDirectory()
        is_zip = True
    else:
        tmp_dir = FakeTempDir(str(config))
        is_zip = False

    with tmp_dir as tmp_dir:
        if is_zip:
            with ZipFile(config, "r") as z:
                config_path, config = _load_dlc_like_zip_file(z)
                zip_project_dir = PurePosixPath(config_path).parent
                for zip_info in z.infolist():
                    if zip_info.is_dir():
                        continue
                    zip_path_obj = PurePosixPath(zip_info.filename)
                    try:
                        sub_path = zip_path_obj.relative_to(zip_project_dir)
                        if sub_path.parts[0] not in ["dlc-models", "config.yaml"]:
                            continue
                        dst_path = Path(tmp_dir, sub_path).resolve()
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(zip_info, "r") as fsrc:
                            with Path(tmp_dir, sub_path).open("wb") as fdst:
                                shutil.copyfileobj(fsrc, fdst)
                    except ValueError:
                        pass
                project_dir = tmp_dir
        else:
            project_dir = Path(config).resolve().parent
            with open(config, "rb") as f:
                config = yaml.load(f, yaml.SafeLoader)

        iteration = config["iteration"]
        train_frac = config["TrainingFraction"][training_set_index]

        model_directory = _get_model_folder(
            config, project_dir, shuffle, train_frac, model_prefix
        )
        model_directory = model_directory.resolve()

        try:
            with (model_directory / "test" / "pose_cfg.yaml").open("rb") as f:
                model_config = yaml.load(f, yaml.SafeLoader)
        except FileNotFoundError as e:
            print(e)
            raise FileNotFoundError(
                f"Invalid model selection: (Iteration {iteration}, Training Fraction {train_frac}, Shuffle: {shuffle})"
            )

        # Set the number of outputs...
        num_outputs = (
            config.get("num_outputs", model_config.get("num_outputs", None))
            if (num_outputs is None)
            else num_outputs
        )
        if num_outputs is not None:
            num_outputs = int(num_outputs)
        batch_size = batch_size if (batch_size is not None) else config["batch_size"]
        body_parts = list(model_config["all_joints_names"])
        if "partaffinityfield_graph" in model_config:
            skeleton = sorted(
                set(
                    tuple(sorted([body_parts[a], body_parts[b]]))
                    for a, b in model_config["partaffinityfield_graph"]
                )
            )
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
                frontend="deeplabcut",
            ),
            FrameExtractor(
                _load_and_convert_model(
                    model_directory / "train", gpu_index, bool(use_cpu)
                ),
                model_config,
            ),
        )
