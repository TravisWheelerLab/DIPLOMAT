"""
Provides functions for extracting certain metadata from videos.
"""
from os import PathLike
from diplomat.utils.video_io import ContextVideoCapture


def is_video(video_path: PathLike) -> bool:
    """
    Check if a specified file is a video file.

    :param video_path: The path to the file to check.

    :return: True if the passed path is a video, otherwise False.
    """
    with ContextVideoCapture(str(video_path), throw_on_unopened=False) as cap:
        is_vid = cap.isOpened()
    return is_vid


def get_frame_count_robust(video: PathLike) -> int:
    """
    Get an accurate frame count for a video.

    :param video: The video to get the frame count for.

    :return: An accurate frame count. Accuracy is better as this method opens the video and runs through all the
             frames in the file.
    """
    with ContextVideoCapture(str(video)) as vid:
        output = 0
        while (vid.isOpened() and vid.grab()):
            output += 1

    return int(output)