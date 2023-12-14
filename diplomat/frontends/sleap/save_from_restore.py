import sys
from .run_utils import PoseLabels, _attach_run_info, Timer
from diplomat.processing.type_casters import typecaster_function, Dict, Any, List
from diplomat.processing.pose import Pose
from .sleap_importer import sleap


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
    from sleap.io.video import MediaVideo
    video = sleap.Video(backend=sleap.Video.make_specific_backend(
        MediaVideo,
        dict(
            filename=sleap.Video.fixup_path(str(video_metadata["orig-video-path"])),
            grayscale=False,
            input_format="channels_last",
            dataset=""
        )
    ))

    skeleton = sleap.Skeleton()
    for node in parts:
        skeleton.add_node(node)
    for src, dest in video_metadata["skeleton"]:
        skeleton.add_edge(src, dest)

    labels = PoseLabels(video, num_outputs, skeleton)
    labels.append(pose)

    labels = _attach_run_info(
        labels.to_sleap(),
        Timer(start_time, end_time),
        str(video_metadata["orig-video-path"]),
        str(video_metadata["output-file-path"]),
        sys.argv
    )

    labels.save(str(video_metadata["output-file-path"]))



