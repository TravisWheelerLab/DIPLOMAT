from diplomat.processing import Pose
from typing import Dict, Any, List
from diplomat.utils.track_formats import save_diplomat_table, to_diplomat_table


def _save_from_restore(
    pose: Pose,
    video_metadata: Dict[str, Any],
    num_outputs: int,
    parts: List[str],
    frame_width_pixels: float,
    frame_height_pixels: float,
    start_time: float,
    end_time: float,
):
    save_diplomat_table(
        to_diplomat_table(num_outputs, parts, pose),
        str(video_metadata["output-file-path"]),
    )
