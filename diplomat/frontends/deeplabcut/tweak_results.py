from .dlc_importer import predict, checkcropping, load_config, auxiliaryfunctions
from .label_videos_dlc import _to_str_list, _get_video_info
import diplomat.processing.type_casters as tc
from pathlib import Path


@tc.typecaster_function
def tweak_videos(
    config: tc.PathLike,
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    shuffle: int = 1,
    training_set_index: int = 0,
    video_type: str = "",
):
    """
    Make minor modifications and tweaks to tracked results produced by DEEPLABCUT (or DIPLOMAT) using the supervised UI.

    :param config: The path to the config.yaml file for the project.
    :param videos: A single video or list of videos to tweak, passed in order to modify.
    :param shuffle: The shuffle index of the model used to track the video. Defaults to 1.
    :param training_set_index: The training index of the model used. Defaults to 0.
    :param video_type: An optional video extension to limit the videos to videos with the specified extension if a directory is passed to the
                       'videos' parameter.
    """
    cfg = auxiliaryfunctions.read_config(config)
    train_frac = cfg["TrainingFraction"][training_set_index]

    dlc_scorer, __ = auxiliaryfunctions.get_scorer_name(
        cfg, shuffle, train_frac
    )

    video_list = auxiliaryfunctions.get_list_of_videos(_to_str_list(videos), video_type)

    for video in video_list:
        try:
            loc_data, metadata, out_path = _get_video_info(video, dlc_scorer)

            if(Path(out_path).exists()):
                print(f"Labeled video {Path(video).name} already exists...")
                continue
        except FileNotFoundError:
            print(f"Unable to find h5 file for video {Path(video).name}. Make sure to run analysis first!")
            continue


def _tweak_single_video(
    args
):
    pass