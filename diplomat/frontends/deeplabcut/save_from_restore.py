from diplomat.processing import Pose
from diplomat.processing.type_casters import typecaster_function, Dict, Any, List
from .dlc_importer import auxiliaryfunctions
from .predict_videos_dlc import _get_pandas_header


@typecaster_function
def _save_from_restore(
    pose: Pose,
    video_metadata: Dict[str, Any],
    num_outputs: int,
    parts: List[str],
    frame_width: int,
    frame_height: int,
    downscaling: float,
    start_time: float,
    end_time: float
):
    pandas_header = _get_pandas_header(parts, num_outputs, "default", "Unknown")

    if(video_metadata["cropping-offset"] is not None):
        coords = [
            video_metadata["cropping-offset"][1],
            int(frame_width * downscaling),
            video_metadata["cropping-offset"][0],
            int(frame_height * downscaling)
        ]
    else:
        coords = [0, video_metadata["size"][1], 0, video_metadata["size"][1]]

    sub_meta = {
        "start": start_time,
        "stop": end_time,
        "run_duration": end_time - start_time,
        "Scorer": "Unknown",
        "DLC-model-config file": None,  # We don't have access to this, so don't even try....
        "fps": video_metadata["fps"],
        "batch_size": 1,
        "multi_output_format": "default",
        "frame_dimensions": video_metadata["size"],
        "nframes": pose.get_frame_count(),
        "iteration (active-learning)": "Unknown",
        "training set fraction": "Unknown",
        "cropping": video_metadata["cropping-offset"] is not None,
        "cropping_parameters": coords,
    }
    metadata = {"data": sub_meta}

    auxiliaryfunctions.SaveData(
        pose.get_all(),
        metadata,
        str(video_metadata["output-file-path"]),
        pandas_header,
        range(pose.get_frame_count()),
        False,
    )
