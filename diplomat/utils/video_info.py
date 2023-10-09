"""
Provides functions for extracting certain metadata from videos.
"""
import cv2
from os import PathLike


def is_video(video_path: PathLike) -> bool:
    """
    Check if a specified file is a video file.

    :param video_path: The path to the file to check.

    :return: True if the passed path is a video, otherwise False.
    """
    cap = cv2.VideoCapture(str(video_path))
    is_vid = cap.isOpened()
    cap.release()
    return is_vid


def get_frame_count_robust(video: PathLike) -> int:
    """
    Get an accurate frame count for a video.

    :param video: The video to get the frame count for.

    :return: An accurate frame count. Accuracy is better as this method opens the video and runs through all the
             frames in the file.
    """
    vid = cv2.VideoCapture(str(video))
    output = 0

    while (vid.isOpened() and vid.grab()):
        output += 1

    vid.release()

    return int(output)