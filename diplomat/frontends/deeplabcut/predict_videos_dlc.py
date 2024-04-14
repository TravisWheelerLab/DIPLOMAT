import time
from typing import List, Dict, Any, Type, Tuple, Optional, Union, Callable

import cv2
import numpy as np
import os
from diplomat.processing import *
from diplomat import processing
from diplomat.utils import video_info
from pathlib import Path
import diplomat.processing.type_casters as tc
from diplomat.utils.shapes import shape_iterator

# DLC Imports
from .dlc_importer import predict, checkcropping, load_config, auxiliaryfunctions, tf
from tqdm import tqdm
import pandas as pd


Pathy = Union[os.PathLike, str]


def _to_str_list(path_list):
    if(isinstance(path_list, (list, tuple))):
        return [str(path) for path in path_list]
    return str(path_list)


@tc.typecaster_function
def analyze_videos(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    video_type: str = "",
    shuffle: int = 1,
    training_set_index: int = 0,
    gpu_index: tc.Optional[int] = None,
    save_as_csv: bool = False,
    destination_folder: tc.Optional[str] = None,
    batch_size: tc.Optional[int] = None,
    cropping: tc.Optional[tc.Tuple[int, int, int, int]] = None,
    model_prefix: str = "",
    num_outputs: tc.Optional[int] = None,
    multi_output_format: tc.Literal["default", "separate"] = "default",
    predictor: tc.Optional[str] = None,
    predictor_settings: tc.Optional[tc.Dict[str, tc.Any]] = None,
):
    """
    Run DIPLOMAT tracking on videos using a DEEPLABCUT project and trained network.

    :param config: The path to the DLC config for the DEEPLABCUT project.

    :param videos: A single path or list of paths, to the location of video files to run analysis on. Can also be a directory.
    :param video_type: Optional string, the video extension to search for if the 'videos' argument is a directory
                       to search inside ('.avi', '.mp4', ...).
    :param shuffle: int, optional. Integer specifying which TrainingsetFraction to use. By default, the first
                    (note that TrainingFraction is a list in config.yaml).
    :param training_set_index: int, optional. Integer specifying which TrainingsetFraction to use. By default the first
                               (note that TrainingFraction is a list in config.yaml).
    :param gpu_index: Integer index of the GPU to use for inference (in tensorflow) defaults to 0, or selecting the first detected GPU if available.
    :param save_as_csv: Boolean, if true save the results to both a HDF5 file (".h5") and also a CSV file, otherwise only save results to a HDF5.
    :param destination_folder: The destination folder to save the resulting HDF5 track files to. Defaults to None, meaning save the HDF5 in the same
                               folder as the video file it was generated from.
    :param batch_size: The batch size to use while processing. Defaults to None, which uses the default batch size for the project.
    :param cropping: A tuple of 4 integers in the format (x1, x2, y1, y2), specifying the boundaries of the cropping box analyze in the video.
                     Defaults to None, which uses the cropping settings in the DLC config file.
    :param model_prefix: The string prefix of the DEEPLABCUT model to use defaults to no prefix (the default model).
    :param num_outputs: The number of outputs, or bodies to track in the video. Defaults to the value specified in the DLC config, or None if one
                        is not specified.
    :param multi_output_format: The format to use when tracking multiple body parts of the same type (multiple bodies, or num_outputs > 1).
                                Defaults to "default", which uses DLC's original multi-output format. Passing "separate" saves additional
                                bodies by tacking on an index onto to body part name (Nose, Nose2, Nose3, ...) instead of storing tracks
                                for the same body part type together.
    :param predictor: A String, the name of the predictor plugin to be used to make predictions. If not specified, defaults to the segmented frame
                  pass engine ("SegmentedFramePassEngine").
    :param predictor_settings: Optional dictionary of strings to any. This will specify what settings a predictor should use,
                    completely ignoring any settings specified in the config.yaml. Default value is None, which
                    tells this method to use the settings specified in the config.yaml.
    """
    if("TF_CUDNN_USE_AUTOTUNE" in os.environ):
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]

    if(gpu_index is not None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    tf.compat.v1.reset_default_graph()

    project_dir = Path(config).resolve().parent
    config = load_config(config)
    iteration = config["iteration"]
    train_frac = config["TrainingFraction"][training_set_index]
    config["project_path"] = project_dir

    model_directory = Path(config["project_path"]) / auxiliaryfunctions.get_model_folder(train_frac, shuffle, config, model_prefix)
    model_directory = model_directory.resolve()

    try:
        model_config = load_config(model_directory / "test" / "pose_cfg.yaml")
    except FileNotFoundError as e:
        print(e)
        raise FileNotFoundError(f"Invalid model selection: (Iteration {iteration}, Training Fraction {train_frac}, Shuffle: {shuffle})")

    try:
        snapshot_list = sorted([
            f.stem for f in (model_directory / "train").iterdir() if(f.suffix == ".index")
        ], key=lambda f: int(f.split("-")[1]))
    except FileNotFoundError:
        raise FileNotFoundError("Snapshots don't exist! Please make sure the model is trained first.")

    selected_snapshot = snapshot_list[config["snapshotindex"] if(config["snapshotindex"] != "all") else -1]
    model_config["init_weights"] = str(model_directory / "train" / selected_snapshot)
    train_iterations = selected_snapshot.split("-")[-1]

    # Set the number of outputs...
    model_config["num_outputs"] = config.get("num_outputs", model_config.get("num_outputs", 1))
    old_num_outputs = model_config["num_outputs"]
    model_config["num_outputs"] = int(max(1, (
        int(num_outputs)
        if ((num_outputs is not None) and (num_outputs >= 1))
        else config.get("num_outputs", len(config.get("individuals", [0])))
    )))

    batch_size = batch_size if(batch_size is not None) else config["batch_size"]
    model_config["batch_size"] = batch_size
    config["batch_size"] = batch_size

    if(cropping is not None):
        config["cropping"] = True
        config["x1"], config["x2"], config["y1"], config["y2"] = cropping

    if(predictor is None):
        predictor = "SegmentedFramePassEngine"
    predictor_cls = processing.get_predictor(predictor)

    if model_config["num_outputs"] > 1 and (not predictor_cls.supports_multi_output()):
        raise NotImplementedError("Predictor plugin does not support num_outputs greater than 1!")

    dlc_scorer, __ = auxiliaryfunctions.GetScorerName(
        config,
        shuffle,
        train_frac,
        trainingsiterations=train_iterations,
        modelprefix=model_prefix,
    )

    sess, inputs, outputs = predict.setup_pose_prediction(model_config)

    table_header = _get_pandas_header(
        model_config["all_joints_names"],
        model_config["num_outputs"],
        multi_output_format,
        dlc_scorer,
    )

    video_list = auxiliaryfunctions.get_list_of_videos(_to_str_list(videos), video_type)

    this_dir = Path.cwd()

    if(len(video_list) > 0):
        for video in video_list:
            _analyze_video(
                video,
                dlc_scorer,
                train_frac,
                config,
                model_config,
                sess,
                inputs,
                outputs,
                table_header,
                save_as_csv,
                destination_folder,
                predictor_cls,
                predictor_settings,
            )
    else:
        print("No videos found!")

    model_config["num_outputs"] = old_num_outputs
    os.chdir(this_dir)

    print("Analysis is done!")

    return None


def _analyze_video(
    video: str,
    dlc_scorer: str,
    training_fraction: float,
    config: Dict[str, Any],
    model_config: Dict[str, Any],
    sess: tf.compat.v1.Session,
    inputs: tf.compat.v1.Tensor,
    outputs: List[tf.compat.v1.Tensor],
    table_header,
    save_as_csv,
    dest_folder=None,
    predictor_cls=None,
    predictor_settings=None,
    dipui_file=None
) -> str:
    print(f"Analyzing video: {video}")

    dest_folder = Path(Path(video).resolve().parent if(dest_folder is None) else dest_folder)
    dest_folder.mkdir(exist_ok=True)

    video_name = Path(video).stem

    h5_path = dest_folder / (video_name + dlc_scorer + ".h5")

    print("Loading the video...")
    cap = cv2.VideoCapture(str(video))
    if(not cap.isOpened()):
        raise IOError(f"Unable to open video: {video}")

    num_frames = video_info.get_frame_count_robust(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = num_frames / fps

    vw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Video Info: frames={num_frames}, fps={fps}, duration={duration}, width={vw}, height={vh}")

    # Passed to the plugin to give it some info about the video...
    video_metadata = Config({
        "fps": fps,
        "duration": duration,
        "size": (vh, vw),
        "output-file-path": str(Path(h5_path).resolve()),
        "orig-video-path": str(Path(video).resolve()),
        "cropping-offset": (int(config["y1"]), int(config["x1"]))
        if (config.get("cropping", False))
        else None,
        "dotsize": config["dotsize"],
        "colormap": config.get("diplomat_colormap", config["colormap"]),
        "shape_list": shape_iterator(config.get("shape_list", None), model_config["num_outputs"]),
        "alphavalue": config["alphavalue"],
        "pcutoff": config["pcutoff"],
        "line_thickness": config.get("line_thickness", 1),
        "skeleton": config.get("skeleton", None),
        "frontend": "deeplabcut"
    })

    # Grab the plugin settings for this plugin...
    predictor_settings = _get_predictor_settings(config, predictor_cls, predictor_settings)
    print(f"Plugin {predictor_cls.get_name()} Settings: {predictor_settings}")

    # Create a predictor plugin instance...
    predictor_inst = predictor_cls(
        model_config["all_joints_names"],
        model_config["num_outputs"],
        num_frames,
        predictor_settings,
        video_metadata,
    )

    start = time.time()

    with predictor_inst as pred:
        predicted_data, num_frames = _get_poses(
            config,
            model_config,
            sess,
            inputs,
            outputs,
            cap,
            num_frames,
            int(model_config["batch_size"]),
            pred,
            cnn_extractor_method=predict.extract_cnn_outputmulti
        )

    stop = time.time()

    if(config["cropping"]):
        coords = [config["x1"], config["x2"], config["y1"], config["y2"]]
    else:
        coords = [0, vw, 0, vh]

    metadata = {
        "data": {
            "start": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": dlc_scorer,
            "DLC-model-config file": model_config,
            "fps": fps,
            "batch_size": model_config["batch_size"],
            "frame_dimensions": (vh, vw),
            "nframes": num_frames,
            "iteration (active-learning)": config["iteration"],
            "training set fraction": training_fraction,
            "cropping": config["cropping"],
            "cropping_parameters": coords
        }
    }

    print(f"Saving results in {dest_folder}...")
    auxiliaryfunctions.SaveData(
        predicted_data[:num_frames, :],
        metadata,
        str(h5_path),
        table_header,
        range(num_frames),
        save_as_csv,
    )

    return dlc_scorer


def _get_pandas_header(body_parts: List[str], num_outputs: int, out_format: str, dlc_scorer: str) -> pd.MultiIndex:
    """
    Creates the pandas data header for the passed body parts and number of outputs.
    body_parts: The list of body part names. List of strings.
    num_outputs: The number of outputs per body part, and integer.
    out_format: The output format, either 'separate-bodyparts' or 'default'.
    dlc_scorer: A string, being the name of the DLC Scorer for this DLC instance.
    Returns: A pandas MultiIndex, being the header entries for the DLC output data.
    """
    # Set this up differently depending on the format...
    if out_format == "separate-bodyparts" and num_outputs > 1:
        # Format which allocates new bodyparts for each prediction by simply adding "__number" to the end of the part's
        # name.
        print("Outputting predictions as separate body parts...")
        suffixes = [f"__{i + 1}" for i in range(num_outputs)]
        suffixes[0] = ""
        all_joints = [bp + s for bp in body_parts for s in suffixes]
        return pd.MultiIndex.from_product(
            [[dlc_scorer], all_joints, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
    else:
        # The original multi output format, multiple predictions stored under each body part
        suffixes = [str(i + 1) for i in range(num_outputs)]
        suffixes[0] = ""
        sub_headers = [
            state + s for s in suffixes for state in ["x", "y", "likelihood"]
        ]
        return pd.MultiIndex.from_product(
            [[dlc_scorer], body_parts, sub_headers],
            names=["scorer", "bodyparts", "coords"],
        )


# Utility method used by AnalyzeVideo, gets the settings for the given predictor plugin
def _get_predictor_settings(
    cfg: Dict[str, Any],
    predictor_cls: Type[Predictor],
    usr_passed_settings: Dict[str, Any] = None
) -> Optional[Config]:
    """ Get the predictor settings from deeplabcut config and return a dictionary for plugin to use... """
    # Grab setting blueprints for predictor plugins(dict of name to default value, type, and description)....
    setting_info = predictor_cls.get_settings()
    name = predictor_cls.get_name()

    if setting_info is None:
        return None

    config = Config({}, setting_info)

    if usr_passed_settings is None:
        # If the dlc config contains a category predictors, and predictors contains a category named after the plugin, load
        # the user cfg for this plugin and merge it with default values
        if (
            ("predictors" in cfg)
            and (cfg["predictors"])
            and (name in cfg["predictors"])
        ):
            config.update(cfg["predictors"][name])
    else:
        # If the user directly passed settings to this method, we ignore the config and use these settings.
        config.update(usr_passed_settings)

    return config

# Utility method used by get_poses, gets a batch of frames, stores them in frame_store and returns the size of the batch
def _get_video_batch(cap: cv2.VideoCapture, batch_size: int, cfg: Dict[str, Any], frame_store: np.ndarray) -> int:
    """ Gets a batch size of frames, and returns them """
    current_frame = 0

    # While the cap is still going and the current frame is less then the batch size...
    while cap.isOpened() and current_frame < batch_size:
        # Read a frame
        ret_val, frame = cap.read()

        # If we got an actual frame, store it in the frame store.
        if ret_val:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if cfg["cropping"]:
                frame_store[current_frame] = frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
            else:
                frame_store[current_frame] = frame
        else:
            # If we don't we have reached the end most likely.
            return current_frame

        current_frame += 1

    return current_frame


# Replaces old system of getting poses and uses a new plugin system for predicting poses...
def _get_poses(
    cfg: Dict[str, Any],
    dlc_cfg: Dict[str, Any],
    sess: tf.compat.v1.Session,
    inputs: tf.compat.v1.Tensor,
    outputs: List[tf.compat.v1.Tensor],
    cap: cv2.VideoCapture,
    num_frames: int,
    batch_size: int,
    predictor: Predictor,
    cnn_extractor_method: Callable[[tuple, dict], Tuple[np.ndarray, np.ndarray]] = predict.extract_cnn_outputmulti
) -> Tuple[np.ndarray, int]:
    """ Gets the poses for any batch size, including batch size of only 1 """
    # Create a numpy array to hold all pose prediction data...
    pose_prediction_data = np.zeros(
        (num_frames, 3 * len(dlc_cfg["all_joints_names"]) * dlc_cfg["num_outputs"])
    )

    pbar = tqdm(total=num_frames)

    ny, nx = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    # Create the temporary batch frame store for storing video frames...
    frame_store = np.empty((batch_size, ny, nx, 3), dtype="ubyte")

    # Create a counter to keep track of the progress bar
    counter = 0
    prog_step = max(10, int(num_frames / 100))
    current_step = 0

    frames_done = 0

    while True:
        size = _get_video_batch(cap, batch_size, cfg, frame_store)
        counter += size

        # If we pass the current step or phase, update the progress bar
        if counter // prog_step > current_step:
            pbar.update(prog_step)
            current_step += 1

        if size > 0:
            # If we received any frames, process them...
            scmap, locref = cnn_extractor_method(
                sess.run(outputs, feed_dict={inputs: frame_store}), dlc_cfg
            )
            down_scale = dlc_cfg["stride"]

            if (
                len(scmap.shape) == 2
            ):  # If there is a single body part, add a dimension at the end
                scmap = np.expand_dims(scmap, axis=2)

            pose = predictor.on_frames(
                TrackingData(scmap[:size], locref, down_scale)
            )

            if pose is not None:
                # If the predictor returned a pose, add it to the final data.
                pose_prediction_data[
                    frames_done : frames_done + pose.get_frame_count()
                ] = pose.get_all()
                frames_done += pose.get_frame_count()

        if size < batch_size:
            # If the output frames by the video capture were less then a full batch, we have reached the end of the
            # video...
            break

    pbar.update(counter - (prog_step * current_step))
    pbar.close()

    # Phase 2: Post processing...

    # Get all of the final poses that are still held by the predictor
    post_pbar = TQDMProgressBar(total=num_frames - frames_done)
    final_poses = predictor.on_end(post_pbar)
    post_pbar.close()

    # Add any post-processed frames
    if final_poses is not None:
        pose_prediction_data[
            frames_done:frames_done + final_poses.get_frame_count()
        ] = final_poses.get_all()
        frames_done += final_poses.get_frame_count()

    # Check and make sure the predictor returned all frames, otherwise throw an error.
    if frames_done != num_frames:
        raise ValueError(
            f"The predictor algorithm did not return the same amount of frames as are in the video.\n"
            f"Expected Amount: {num_frames}, Actual Amount Returned: {frames_done}"
        )

    return pose_prediction_data, num_frames

