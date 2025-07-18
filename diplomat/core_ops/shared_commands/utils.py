import functools
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Type, Union, Dict

import cv2

from diplomat.processing import Config, Predictor
from diplomat.utils.frame_store_api import DLFSHeader
from diplomat.utils.shapes import shape_iterator
from diplomat.utils.track_formats import load_diplomat_table
from diplomat.utils.video_info import get_frame_count_robust_fast
from diplomat.utils.video_io import ContextVideoCapture


def _header_check(csv):
    with open(csv, "r") as csv_handle:
        first_lines = [csv_handle.readline().strip("\n").split(",") for i in range(3)]

        header_cols = len(first_lines[0])

        if not all(header_cols == len(line) for line in first_lines):
            return False

        last_header_line = first_lines[-1]
        last_line_exp = ["x", "y", "likelihood"] * (len(last_header_line) // 3)

        if last_header_line != last_line_exp:
            return False

        return True


def _fix_path_pairs(csvs, videos):
    csvs = csvs if (isinstance(csvs, (tuple, list))) else [csvs]
    videos = videos if (isinstance(videos, (tuple, list))) else [videos]

    if len(csvs) == 1:
        csvs = csvs * len(videos)
    if len(videos) == 1:
        videos = videos * len(csvs)

    if len(videos) != len(csvs):
        raise ValueError("Number of videos and csv files passes don't match!")

    return csvs, videos


def _get_predictor_settings(
    predictor_cls: Type[Predictor], user_passed_settings
) -> Optional[Config]:
    settings_backing = predictor_cls.get_settings()

    if settings_backing is None:
        return None

    return Config(user_passed_settings, settings_backing)


def _paths_to_str(paths):
    if isinstance(paths, (list, tuple)):
        return [str(p) for p in paths]
    else:
        return str(paths)


def _convert_user_skeleton(
    skeleton: Union[bool, Dict[str, List[str]], List[str], List[Tuple[str, str]]],
    bp_list: List[str],
) -> List[Tuple[str, str]]:
    if skeleton is False:
        return []

    if skeleton is True:
        skeleton = bp_list

    edge = lambda a, b: (a, b) if a <= b else (b, a)

    if isinstance(skeleton, dict):
        skeleton = [
            edge(x, y) for x, elems in skeleton.items() for y in elems if x != y
        ]
    if (
        isinstance(skeleton, list)
        and len(skeleton) > 0
        and all(isinstance(v, str) for v in skeleton)
    ):
        skeleton = [
            edge(skeleton[i], skeleton[j])
            for i in range(len(skeleton))
            for j in range(i + 1, len(skeleton))
        ]

    if not all(isinstance(v, tuple) and len(v) == 2 for v in skeleton):
        raise ValueError("Invalid skeleton passed!")

    # Validate edge list...
    skeleton_new = set()
    bp_list = set(bp_list)

    for aa, bb in skeleton:
        if aa == bb:
            continue
        if aa not in bp_list:
            raise ValueError(f"Invalid part in skeleton: {aa}, valid parts: {bp_list}")
        if bb not in bp_list:
            raise ValueError(f"Invalid part in skeleton: {bb}, valid parts: {bp_list}")
        skeleton_new.add(edge(aa, bb))

    return sorted(skeleton_new)


def _get_video_metadata(
    frontend: str,
    bp_list: List[str],
    video_path: Optional[Path],
    output_path: Path,
    num_outputs: int,
    visual_settings: Config,
    skeleton: List[Tuple[str, str]] = None,
    crop_loc: Optional[Tuple[int, int]] = None,
    frame_store_header: Optional[DLFSHeader] = None,
) -> Tuple[Config, int]:
    if visual_settings.skeleton is not None:
        skeleton = _convert_user_skeleton(visual_settings.skeleton, bp_list)

    if skeleton is None:
        skeleton = []

    if frame_store_header is None:
        with ContextVideoCapture(str(video_path)) as vid:
            fps = vid.get(cv2.CAP_PROP_FPS)
            w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = get_frame_count_robust_fast(vid)
    else:
        fps = frame_store_header.frame_rate
        w = frame_store_header.frame_width
        h = frame_store_header.frame_height
        frame_count = frame_store_header.number_of_frames

    return (
        Config(
            {
                "fps": fps,
                "duration": frame_count / fps,
                "size": (h, w),
                "output-file-path": str(output_path),
                "orig-video-path": (
                    str(video_path) if (video_path is not None) else None
                ),  # This may be None if we were unable to find the video...
                "cropping-offset": crop_loc,
                "dotsize": visual_settings.dotsize,
                "colormap": visual_settings.colormap,
                "shape_list": shape_iterator(visual_settings.shape_list, num_outputs),
                "alphavalue": visual_settings.alphavalue,
                "pcutoff": visual_settings.pcutoff,
                "line_thickness": visual_settings.get("line_thickness", 1),
                "skeleton": skeleton,
                "frontend": frontend,
            }
        ),
        frame_count,
    )


class Timer:
    def __init__(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ):
        self._start_time = start_time
        self._end_time = end_time

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()

    @property
    def start(self) -> float:
        return self._start_time

    @property
    def end(self) -> float:
        return self._end_time

    @property
    def start_date(self) -> datetime:
        return datetime.fromtimestamp(self._start_time)

    @property
    def end_date(self) -> datetime:
        return datetime.fromtimestamp(self._end_time)

    @property
    def duration(self) -> float:
        return self._end_time - self._start_time


@functools.cache
def _get_track_loaders(include_native: bool = False):
    from diplomat import _LOADED_FRONTENDS
    from diplomat.frontends import DIPLOMATContract, TracksLoaderFunction

    loaders = []

    if include_native:
        loaders.append(lambda path: load_diplomat_table(str(path)))

    for frontend_name, funcs in _LOADED_FRONTENDS.items():
        if funcs.verify_contract(
            DIPLOMATContract("_load_tracks", TracksLoaderFunction)
        ):
            loaders.append(funcs._load_tracks)

    return loaders


def _load_tracks_from_loaders(loaders, input_path):
    old_exp = None
    input_path = Path(str(input_path)).resolve()

    for loader in loaders:
        try:
            return loader(path=input_path)
        except (ValueError, KeyError, TypeError, IOError) as exp:
            try:
                raise exp from old_exp
            except type(exp):
                old_exp = exp
    else:
        raise NotImplementedError(
            f"Unable to find frontend that could load the file: {input_path}"
        ) from old_exp
