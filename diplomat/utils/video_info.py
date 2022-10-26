"""
Provides functions for extracting certain metadata from videos.
"""
import cv2

def get_frame_count_robust(video: str) -> int:
    """
    Get an accurate frame count for a video.

    :param video: The video to get the frame count for.

    :return: An accurate frame count. Accuracy is better as this method opens the video and runs through all the frames in the file.
    """
    vid = cv2.VideoCapture(video)
    output = 0

    while (vid.isOpened() and vid.grab()):
        output += 1

    vid.release()

    return int(output)