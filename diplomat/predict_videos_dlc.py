import time
from typing import List, Dict, Any, Type, Tuple, Optional, Literal, Union

import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from skimage import img_as_ubyte
from diplomat.processing import *
from diplomat import processing
from diplomat.utils import video_info
from pathlib import Path

# DLC Imports
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.predict_videos import checkcropping
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow import predict_multianimal

Pathy = Union[os.PathLike, str]

def analyze_videos(
    config: str,
    videos: List[Pathy],
    video_type: str = "avi",
    shuffle: int = 1,
    training_set_index: int = 0,
    gpu_index: int = None,
    save_as_csv: bool = False,
    destination_folder: str = None,
    batch_size: int = None,
    cropping: Tuple[int, int, int, int] = None,
    model_prefix: str = "",
    num_outputs: int = None,
    multi_output_format: Literal["default", "separate"] = "default",
    predictor: Optional[str] = None,
    predictor_settings: Optional[Dict[str, Any]] = None
) -> str:
    if("TF_CUDNN_USE_AUTOTUNE" in os.environ):
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]

    if(gpu_index is not None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    tf.compat.v1.reset_default_graph()
    this_dir = Path.cwd()

    config = load_config(config)
    iteration = config["iteration"]
    train_frac = config["TrainingFraction"][training_set_index]

    model_directory = Path(config["project_path"]) / auxiliaryfunctions.GetModelFolder(train_frac, shuffle, config, model_prefix)

    try:
        model_config = load_config(model_directory / "test" / "pose_cfg.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"Invalid model selection: (Iteration {iteration}, Training Fraction {train_frac}, Shuffle: {shuffle})")

    try:
        snapshot_list = sorted([
            f.stem for f in (model_directory / "train").iterdir() if(f.suffix == "index")
        ], key=lambda f: int(f.split("-")[1]))
    except FileNotFoundError:
        raise FileNotFoundError("Snapshots don't exist! Please make sure the model is trained first.")

    selected_snapshot = snapshot_list[config["snapshotindex"] if(config["snapshotindex"] != "all") else -1]
    model_config["init_weights"] = model_directory / "train" / selected_snapshot
    train_iterations = selected_snapshot.split("-")[-1]

    # Set the number of outputs...
    model_config["num_outputs"] = config.get("num_outputs", model_config.get("num_outputs", 1))
    old_num_outputs = model_config["num_outputs"]
    model_config["num_outputs"] = (
        int(num_outputs)
        if ((num_outputs is not None) and (num_outputs >= 1))
        else model_config["num_outputs"]
    )

    batch_size = batch_size if(batch_size is not None) else config["batch_size"]
    model_config["batch_size"] = batch_size
    config["batch_size"] = batch_size

    if predictor is not None:
        # If predictor plugin was selected, disable dynamic mode and GPU predictions.
        predictor_cls = processing.get_predictor(predictor)
        print(f"Predictor '{predictor}' selected, disabling GPU predictions and dynamic cropping as both of these are not supported.")

        if model_config["num_outputs"] > 1 and (not predictor_cls.supports_multi_output()):
            raise NotImplementedError(
                "Predictor plugin does not support num_outputs greater than 1."
            )
    else:
        predictor_cls = processing.get_predictor("ArgMax")

    dlc_scorer, __ = auxiliaryfunctions.GetScorerName(
        config,
        shuffle,
        train_frac,
        trainingsiterations=train_iterations,
        modelprefix=model_prefix,
    )

    sess, inputs, outputs = predict.setup_pose_prediction(model_config)

    table_header = get_pandas_header(
        model_config["all_joints_names"],
        model_config["num_outputs"],
        multi_output_format,
        dlc_scorer,
    )

    video_list = auxiliaryfunctions.Getlistofvideos(videos, video_type)

    if(len(video_list) > 0):
        for video in video_list:
            analyze_video(
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
                predictor_settings
            )
    else:
        print("No videos found!")

    model_config["num_outputs"] = old_num_outputs
    os.chdir(this_dir)

    print("Analysis is done!")

    return dlc_scorer


def analyze_video(
    video,
    dlc_scorer,
    training_fraction,
    config,
    model_config,
    sess,
    inputs,
    outputs,
    table_header,
    save_as_csv,
    dest_folder=None,
    predictor_cls=None,
    predictor_settings=None,
) -> str:
    print(f"Analyzing video: {video}")

    dest_folder = Path(Path(video).resolve().parent if(dest_folder is None) else dest_folder)
    dest_folder.mkdir(exist_ok=True)

    video_name = Path(video).stem

    try:
        __ = auxiliaryfunctions.load_analyzed_data(dest_folder, video_name, dlc_scorer)
        print(f"Results for video {video} already exist.")
        return
    except FileNotFoundError:
        pass

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
        "h5-file-name": str(Path(h5_path).resolve()),
        "orig-video-path": str(Path(video).resolve()),
        "cropping-offset": (int(config["y1"]), int(config["x1"]))
        if (config.get("cropping", False))
        else None,
        "dotsize": config["dotsize"],
        "colormap": config["colormap"],
        "alphavalue": config["alphavalue"],
        "pcutoff": config["pcutoff"],
    })

    # Grab the plugin settings for this plugin...
    predictor_settings = get_predictor_settings(config, predictor_cls, predictor_settings)
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

    predicted_data, num_frames = get_poses(
        config,
        model_config,
        sess,
        inputs,
        outputs,
        cap,
        num_frames,
        int(model_config["batch_size"]),
        predictor_inst,
        cnn_extractor_method=(
            predict.extract_cnn_outputmulti
            if ("multi-animal" in model_config["dataset_type"]) else
            lambda *args, **kwargs: predict_multianimal.extract_cnn_outputmulti(*args, **kwargs)[:2]
        )
    )

    stop = time.time()

    if(config["cropping"] == True):
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
        h5_path,
        table_header,
        range(num_frames),
        save_as_csv,
    )

    return dlc_scorer


def get_pandas_header(body_parts: List[str], num_outputs: int, out_format: str, dlc_scorer: str) -> pd.MultiIndex:
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
def get_predictor_settings(
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
def get_video_batch(cap: cv2.VideoCapture, batch_size: int, cfg: Dict[str, Any], frame_store: np.ndarray) -> int:
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
                frame_store[current_frame] = img_as_ubyte(
                    frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )
            else:
                frame_store[current_frame] = img_as_ubyte(frame)
        else:
            # If we don't we have reached the end most likely.
            return current_frame

        current_frame += 1

    return current_frame


# Replaces old system of getting poses and uses a new plugin system for predicting poses...
def get_poses(
    cfg: Dict[str, Any],
    dlc_cfg: Dict[str, Any],
    sess,
    inputs,
    outputs,
    cap: cv2.VideoCapture,
    num_frames: int,
    batch_size: int,
    predictor: Predictor,
    cnn_extractor_method = predict.extract_cnn_outputmulti
) -> Tuple[np.ndarray, int]:
    """ Gets the poses for any batch size, including batch size of only 1 """
    # Create a numpy array to hold all pose prediction data...
    pose_prediction_data = np.zeros(
        (num_frames, 3 * len(dlc_cfg["all_joints_names"]) * dlc_cfg["num_outputs"])
    )

    pbar = tqdm(total=num_frames)

    ny, nx = int(cap.get(4)), int(cap.get(3))

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
        size = get_video_batch(cap, batch_size, cfg, frame_store)
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
            down_scale = dlc_cfg.stride

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

