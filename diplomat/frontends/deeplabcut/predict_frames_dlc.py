from typing import Optional, List, Type, Dict, Any

import cv2

from .dlc_importer import auxiliaryfunctions
from diplomat.processing import Predictor, TQDMProgressBar, Config
import diplomat.processing.type_casters as tc
from diplomat import processing
from diplomat.frontends.deeplabcut.predict_videos_dlc import _get_predictor_settings, _get_pandas_header
from diplomat.utils.video_splitter import _sanitize_path_arg
from pathlib import Path
import numpy as np
import diplomat.utils.frame_store_fmt as frame_store_fmt
import tqdm
import time
from diplomat.utils.shapes import shape_iterator


@tc.typecaster_function
def analyze_frames(
    config: tc.PathLike,
    frame_stores: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    predictor: tc.Optional[str] = None,
    save_as_csv: bool = False,
    multi_output_format: tc.Literal["default", "separate-bodyparts"] = "default",
    video_folders: tc.Optional[tc.Union[tc.List[tc.PathLike], tc.PathLike]] = None,
    num_outputs: tc.Optional[int] = None,
    shuffle: int = 1,
    training_set_index: int = 0,
    predictor_settings: tc.Optional[tc.Dict[str, tc.Any]] = None,
):
    """
    Takes a DIPLOMAT Frame Store file (.dlfs) and makes predictions for the stored frames, using whatever predictor
    plugin is selected. This allows for the video to be run through the Deep Neural Network once, and then run through
    several prediction algorithms as many times as desired, saving time. It also allows for frames to be processed
    on one computer to be transferred to another computer for post-processing and predictions.

    :param config: The path to the DLC config to use to interpret this data. The .DLFS will inherit the neural
                        network of this project, allowing for frame labeling using this project.
    :param frame_stores: The paths to the frame stores (.dlfs files), string or list of strings.
    :param predictor: A String, the name of the predictor plugin to be used to make predictions. If not specified, defaults to the segmented frame
                      pass engine ("SegmentedFramePassEngine").
    :param save_as_csv: A Boolean, True to save the results to the human readable .csv format, otherwise false.
    :param multi_output_format: A string. Determines the multi output format used. "default" uses the default format,
                                while "separate-bodyparts" separates the multi output predictions such that each is its
                                own body part.
    :param video_folders: None, a string, or a list of strings, folders to search through to find videos which
                          correlate to the .dlfs files. If set to None, this method will search for the corresponding
                          videos in the directory each .dlfs file is contained in.
    :param num_outputs: int, default: from config.yaml, or 1 if not set in config.yaml.
                        Allows the user to set the number of predictions for bodypart,
                        overriding the option in the config file.
    :param shuffle: int, optional. An integer specifying the shuffle index of the training dataset used for training
                    the network. The default is 1.
    :param training_set_index: int, optional. Integer specifying which TrainingsetFraction to use. By default the first
                             (note that TrainingFraction is a list in config.yaml).
    :param predictor_settings: Optional dictionary of strings to any. This will specify what settings a predictor should use,
                        completely ignoring any settings specified in the config.yaml. Default value is None, which
                        tells this method to use the settings specified in the config.yaml.
    """
    # Grab the name of the current DLC Scorer, hack as DLCs Plot functions require a scorer, which is dumb. If it fails,
    # we just call the model 'Unknown' :). Simply means user won't be able to use create_labeled_video, data is still
    # 100% valid.
    cfg = auxiliaryfunctions.read_config(config)
    train_frac = cfg["TrainingFraction"][training_set_index]

    try:
        dlc_scorer, dlc_scorer_legacy = auxiliaryfunctions.GetScorerName(
            cfg, shuffle, train_frac
        )
    except Exception:
        dlc_scorer, dlc_scorer_legacy = "Unknown", "Unknown"

    # Convert all of the path lists to Path objects, useful later...
    frame_stores = _sanitize_path_arg(frame_stores)
    if frame_stores is None:
        raise ValueError(
            "Path must be PathLike, a string, a list of PathLike, or a list of strings."
        )

    video_folders = _sanitize_path_arg(video_folders)
    # Video files are not required, but some plugins do rely on them, so this code tries to resolve videos...
    video_files = _resolve_videos(frame_stores, video_folders)

    # Get the number of outputs...
    num_outputs = int(max(1, (
        cfg.get("num_outputs", len(cfg.get("individuals", [0])))
        if ((num_outputs is None) or (int(num_outputs) < 1))
        else int(num_outputs)
    )))

    # Loading the predictor plugin
    if predictor is None:
        predictor = "SegmentedFramePassEngine"
    predictor_cls = processing.get_predictor(predictor)

    # Check and make sure that this predictor supports multi output if we are currently in that mode...
    if (num_outputs > 1) and (not predictor_cls.supports_multi_output()):
        raise NotImplementedError(
            "The selected predictor plugin doesn't support multiple outputs!!!"
        )

    for frame_store_path, video_path in zip(frame_stores, video_files):
        _analyze_frame_store(
            cfg,
            frame_store_path,
            video_path,
            dlc_scorer,
            dlc_scorer_legacy,
            predictor_cls,
            multi_output_format,
            num_outputs,
            train_frac,
            save_as_csv,
            predictor_settings,
        )

    print("Analysis and Predictions are Done! Now your research can truly start!")


def _resolve_videos(
    frame_store_paths: List[Path], video_folders: Optional[List[Path]]
) -> List[Optional[Path]]:
    """
    Private: Resolves the video paths of the frame stores. The entry for a frame store will be None if the original
    video path can't be found.
    """
    video_paths = [None] * len(frame_store_paths)
    expected_video_names = {
        "~".join(path.stem.split("~")[:-1]): idx
        for idx, path in enumerate(frame_store_paths)
    }
    expected_video_names = {
        name: idx for name, idx in expected_video_names.items() if (name.strip() != "")
    }

    # If the user passed video folders to check, check them, searching for all matching videos with the same name.
    if video_folders is not None:
        for path in video_folders:
            if path.is_dir():
                for subpath in path.iterdir():
                    if (subpath.name in expected_video_names) and (
                        video_paths[expected_video_names[subpath.name]] is None
                    ):
                        video_paths[expected_video_names[subpath.name]] = subpath

    # Check if the video exists in the same folder as the .dlfs, if so add it. Overrides video folder search above...
    for idx, path in enumerate(frame_store_paths):
        name = "~".join(path.stem.split("~")[:-1])
        if name.strip() != "":
            suspect_video = (path.resolve().parent) / (name)
            if suspect_video.exists():
                video_paths[idx] = suspect_video

    # Finally, check if the frame store itself can be opened as a video (new dual encoding). Overrides prior searches...
    for idx, path in enumerate(frame_store_paths):
        try:
            test_cap = cv2.VideoCapture(path)
            if(test_cap.grab()):
                video_paths[idx] = path
        except:
            print(f"The frame store at {path} could not be opened as a video.")
    
    return video_paths


def _analyze_frame_store(
    cfg: dict,
    frame_store_path: Path,
    video_name: Optional[str],
    dlc_scorer: str,
    dlc_scorer_legacy: str,
    predictor_cls: Type[Predictor],
    multi_output_format: str,
    num_outputs: int,
    train_frac: str,
    save_as_csv: bool,
    predictor_settings: Optional[Dict[str, Any]],
) -> str:
    # Check if the data was analyzed yet...
    v_name_sanitized = (
        Path(video_name).resolve().stem if (video_name is not None) else "unknownVideo"
    )
    not_analyzed, data_name, dlc_scorer = auxiliaryfunctions.CheckifNotAnalyzed(
        str(frame_store_path.parent), v_name_sanitized, dlc_scorer, dlc_scorer_legacy
    )

    # Read the frame store into memory:
    with frame_store_path.open("rb") as fb:
        print(f"Processing '{frame_store_path.name}'")
        start = time.time()

        # Read in the header, setup the settings.
        frame_reader = frame_store_fmt.DLFSReader(fb)

        (
            num_f,
            f_h,
            f_w,
            f_rate,
            stride,
            vid_h,
            vid_w,
            off_y,
            off_x,
            bp_lst,
        ) = frame_reader.get_header().to_list()

        pd_index = _get_pandas_header(
            bp_lst, num_outputs, multi_output_format, dlc_scorer
        )

        predictor_settings = _get_predictor_settings(
            cfg, predictor_cls, predictor_settings
        )

        video_metadata = Config({
            "fps": f_rate,
            "duration": float(num_f) / f_rate,
            "size": (vid_h, vid_w),
            "output-file-path": data_name,
            # This may be None if we were unable to find the video...
            "orig-video-path": str(video_name) if (video_name is not None) else None,
            "cropping-offset": None if (off_x is None or off_y is None) else (off_y, off_x),
            "dotsize": cfg["dotsize"],
            "colormap": cfg.get("diplomat_colormap", cfg["colormap"]),
            "shape_list": shape_iterator(cfg.get("shape_list", None), num_outputs),
            "alphavalue": cfg["alphavalue"],
            "pcutoff": cfg["pcutoff"],
            "line_thickness": cfg.get("line_thickness", 1),
            "skeleton": cfg.get("skeleton", None),
            "frontend": "deeplabcut"
        })

        # Create the plugin instance...
        print(f"Plugin {predictor_cls.get_name()} Settings: {predictor_settings}")
        predictor = predictor_cls(
            bp_lst, num_outputs, num_f, predictor_settings, video_metadata
        )

        with predictor as predictor_inst:
            # The pose prediction final output array...
            pose_prediction_data = np.zeros((num_f, 3 * len(bp_lst) * num_outputs))

            # Begin running through frames...
            p_bar = tqdm.tqdm(total=num_f)
            frames_done = 0

            while frame_reader.has_next():
                frame = frame_reader.read_frames()
                pose = predictor_inst.on_frames(frame)
                if pose is not None:
                    # If the predictor returned a pose, add it to the final data.
                    pose_prediction_data[
                        frames_done : frames_done + pose.get_frame_count()
                    ] = pose.get_all()
                    frames_done += pose.get_frame_count()

                p_bar.update()

            p_bar.close()

            # Post-Processing Phase:
            # Phase 2: Post processing...

            # Get all of the final poses that are still held by the predictor
            post_pbar = TQDMProgressBar(total=num_f - frames_done)
            final_poses = predictor_inst.on_end(post_pbar)
            post_pbar.close()

            # Add any post-processed frames
            if final_poses is not None:
                pose_prediction_data[
                    frames_done : frames_done + final_poses.get_frame_count()
                ] = final_poses.get_all()
                frames_done += final_poses.get_frame_count()

            # Check and make sure the predictor returned all frames, otherwise throw an error.
            if frames_done != num_f:
                raise ValueError(
                    f"The predictor algorithm did not return the same amount of frames as are in the frame store.\n"
                    f"Expected Amount: {num_f}, Actual Amount Returned: {frames_done}"
                )

        stop = time.time()
        frame_reader.close()

        if cfg["cropping"]:
            coords = [cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]]
        else:
            coords = [0, vid_w, 0, vid_h]

        sub_meta = {
            "start": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": dlc_scorer,
            "DLC-model-config file": None,  # We don't have access to this, so don't even try....
            "fps": f_rate,
            "batch_size": 1,
            "multi_output_format": multi_output_format,
            "frame_dimensions": (f_h * stride, f_w * stride),
            "nframes": num_f,
            "iteration (active-learning)": cfg["iteration"],
            "training set fraction": train_frac,
            "cropping": cfg["cropping"],
            "cropping_parameters": coords,
        }
        metadata = {"data": sub_meta}

        # We are Done!!! Save data and return...
        auxiliaryfunctions.SaveData(
            pose_prediction_data,
            metadata,
            data_name,
            pd_index,
            range(num_f),
            save_as_csv,
        )

    return dlc_scorer
