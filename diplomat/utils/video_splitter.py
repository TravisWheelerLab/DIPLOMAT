"""
Provides a function for splitting videos into consecutive chunks at provided intervals or at exact locations.
"""
from typing import Union, Tuple, List, Optional, Iterable

import diplomat.processing.type_casters as tc
from diplomat.utils.pretty_printer import printer
import cv2
from pathlib import Path
from os import PathLike
import tqdm
import math

FALLBACK_CODEC = "mp4v"
FALLBACK_EXT = ".mp4"


@tc.typecaster_function
def split_videos(
    videos: tc.Union[tc.Path, tc.List[tc.Path]],
    seconds_per_segment: tc.Union[tc.List[int], int] = 300,
    output_fourcc_string: tc.Optional[str] = None,
    output_extension: tc.Optional[str] = None
) -> tc.List[tc.List[Path]]:
    """
    Split a video into even length segments. This will produce a list of videos with "-part{number}" appended to the
    original video name in the same directory as the original video.

    :param videos: Either a single path-like object (string or Path) or a list of path-like objects, being the
                   paths to the videos to split into several segments.
    :param seconds_per_segment: An integer or a list of integers. If a single integer, represents the length of each
                                split segment in seconds (Ex. if 30, split the clip every 30 seconds). If a list of
                                integers, represents the locations to split the video at in seconds. The list can be
                                out of order, as it will be automatically sorted. Also, values that are out of range
                                will be ignored.
                                (Ex. if [10, 400, 100, 30], split the video at 10s, 400s, 100s, and 30s].
    :param output_fourcc_string: Optional, the fourcc string to use for output videos. If not specified uses input video fourcc code.
    :param output_extension: Optional, the file extension to use for output videos (including the dot).
                             Defaults to the input file extension if not specified.

    :returns: A list of lists of Path objects, being the new split video paths for each and every video...
    """
    videos = _sanitize_path_arg(videos)

    if(videos is None):
        raise ValueError("No videos provided!!!")

    return [_split_single_video(video, seconds_per_segment, output_fourcc_string, output_extension) for video in videos]


def _split_single_video(
    video_path: Path,
    seconds_per_segment: Union[int, List[int]],
    output_fourcc_string: Optional[str] = None,
    output_extension: Optional[str] = None
) -> List[Path]:
    """
    PRIVATE: Splits a single video.

    :param video_path: The path of the original video to split.
    :param seconds_per_segment: The duration of each segment, in seconds.
    :param output_fourcc_string: Optional, the output fourcc string. If not specified uses input fourcc code.
    :param output_extension: Optional, the output file extension. Defaults to the input file extension if not specified.

    :returns: The paths of the newly split videos, as a list of "Path"s...
    """
    printer(f"Processing video: {video_path}")
    vid = cv2.VideoCapture(str(video_path))

    width, height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    four_cc = int(vid.get(cv2.CAP_PROP_FOURCC))
    if(four_cc == 0):
        four_cc = -1
    else:
        four_cc = "".join([chr((four_cc >> i) & 255) for i in range(0, 32, 8)])
        four_cc = cv2.VideoWriter_fourcc(*four_cc)

    if(output_fourcc_string is not None):
        four_cc = cv2.VideoWriter_fourcc(*output_fourcc_string)

    extension = video_path.suffix if(output_extension is None) else output_extension

    writer = None
    segment = 0
    frame = 0

    if(isinstance(seconds_per_segment, int)):
        # Yes, a range can be indexed just like an array... :)
        split_loc = range(0, total_frames, int(seconds_per_segment * fps))
    else:
        split_loc = sorted(set([0] + [int(seg * fps) for seg in seconds_per_segment]))

    bar = tqdm.tqdm(total=total_frames)

    zero_pad_amt = int(math.ceil(math.log10(total_frames + 1)))
    paths = []

    while(vid.isOpened()):
        if(writer is None):
            try:
                start = _list_access(split_loc, [total_frames], segment)
                end = _list_access(split_loc, [total_frames], segment + 1) - 1
                writer, p = _new_video_writer(video_path, (start, end), zero_pad_amt, four_cc, fps, (width, height), extension)
                paths.append(p)
            except OSError:
                vid.release()
                bar.close()
                raise
            segment += 1

        res, frm = vid.read()

        if(not res):
            break

        if(writer.isOpened()):
            writer.write(frm)

        if((segment < len(split_loc)) and (frame >= split_loc[segment])):
            writer.release()
            writer = None

        frame += 1
        bar.update(1)

    if(writer is not None):
        writer.release()

    bar.close()
    vid.release()

    return paths


def _list_access(list1, list2, index):
    comb = (list1, list2)
    return comb[index // len(list1)][index % len(list1)]


def _sanitize_path_arg(
    paths: Union[None, Iterable[PathLike], PathLike]
) -> Optional[List[Path]]:
    """
    Sanitizes a pathlike or list of pathlike argument and returns a list of Path, or None if rogue data was passed...
    """
    if isinstance(paths, (PathLike, str)):
        return [Path(str(paths)).resolve()]
    elif isinstance(paths, Iterable):
        paths = list(paths)
        if len(paths) > 0:
            return [Path(str(path)).resolve() for path in paths]
        else:
            return None
    else:
        return None


def _new_video_writer(
    video_path: Path,
    segment: Tuple[int, int],
    padding: int,
    four_cc: int,
    fps: float,
    size: Tuple[int, int],
    ext: str
) -> Tuple[cv2.VideoWriter, Path]:
    """
    PRIVATE: Construct a new video writer. Will try to use the passed codec if possible, otherwise uses a fallback codec and
    format.
    """
    suffix = f"_part{segment[0]:0{padding}d}-{segment[1]:0{padding}d}"
    preferred_path = video_path.parent / f"{video_path.stem}{suffix}{ext}"
    writer = cv2.VideoWriter(str(preferred_path), four_cc, fps, size)
    if(writer.isOpened()):
        return (writer, preferred_path)

    writer.release()
    fallback_path = video_path.parent / f"{video_path.stem}{suffix}{FALLBACK_EXT}"
    writer = cv2.VideoWriter(str(fallback_path), cv2.VideoWriter_fourcc(*FALLBACK_CODEC), fps, size)
    if(writer.isOpened()):
        return (writer, fallback_path)

    writer.release()
    raise OSError("Can't find a codec to write with!!!")

