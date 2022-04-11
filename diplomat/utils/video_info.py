import cv2

def get_frame_count_robust(video: str) -> int:
    vid = cv2.VideoCapture(video)
    output = 0

    while (vid.isOpened() and vid.grab()):
        output += 1

    vid.release()

    return int(output)