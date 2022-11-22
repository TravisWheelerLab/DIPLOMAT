from pathlib import Path
from typing import Optional, Type

import numpy as np

import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args, Flag
from diplomat.processing.progress_bar import TQDMProgressBar
from diplomat.processing import get_predictor, Config, Predictor
from .sleap_providers import PredictorExtractor
from .visual_settings import VISUAL_SETTINGS

import sleap
import tensorflow as tf
from inspect import signature

from ...utils.shapes import shape_iterator


def _paths_to_str(paths):
    if(isinstance(paths, list)):
        return [str(p) for p in paths]
    else:
        return str(paths)


def _get_default_value(func, attr, fallback):
    param = signature(func).parameters.get(attr, None)
    return fallback if(param is None) else param.default


def _setup_gpus(use_cpu: bool, gpu_index: Optional[int]):
    # GPU setup...
    if(use_cpu or not sleap.nn.system.is_gpu_system()):
        sleap.nn.system.use_cpu_only()
    else:
        if(gpu_index is None):
            print("Selecting GPU Automatically")
            gpu_free_mem = sleap.nn.system.get_gpu_memory()
            if(len(gpu_free_mem) == 0):
                print("Unable to get gpu info, selecting first GPU.")
                gpu_index = 0
            else:
                gpu_index = np.argmax(gpu_free_mem)
                print(f"Selecting GPU {gpu_index} with {gpu_free_mem[gpu_index]} MiB of free memory.")
        sleap.nn.system.use_gpu(gpu_index)

    sleap.disable_preallocation()

    # Make sure tensorflow is running in the correct mode for evaluation (this issue can happen when the DLC frontend is loaded)...
    if(not tf.executing_eagerly()):
        tf.compat.v1.enable_eager_execution()


def _get_predictor_settings(predictor_cls: Type[Predictor], user_passed_settings) -> Optional[Config]:
    settings_backing = predictor_cls.get_settings()

    if(settings_backing is None):
        return None

    return Config(user_passed_settings, settings_backing)


def _get_video_metadata(
    video_path: Path,
    output_path: Path,
    num_outputs: int,
    video: sleap.Video,
    visual_settings: Config,
    mdl_metadata: dict
) -> Config:
    fps = getattr(video, "fps", 30)

    return Config({
        "fps": fps,
        "duration": video.num_frames / fps,
        "size": tuple(video.shape[1:3]),
        "output-file-path": str(output_path),
        "orig-video-path": str(video_path),  # This may be None if we were unable to find the video...
        "cropping-offset": None,
        "dotsize": visual_settings.dotsize,
        "colormap": visual_settings.colormap,
        "shape_list": shape_iterator(visual_settings.shape_list, num_outputs),
        "alphavalue": visual_settings.alphavalue,
        "pcutoff": visual_settings.pcutoff,
        "line_thickness": visual_settings.get("line_thickness", 1),
        "skeleton": mdl_metadata["skeleton"]
    })



@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def analyze_videos(
    config: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    videos: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    batch_size: tc.Optional[int] = None,
    num_outputs: tc.Optional[int] = None,
    predictor: tc.Optional[str] = None,
    predictor_settings: tc.Optional[tc.Dict[str, tc.Any]] = None,
    gpu_index: tc.Optional[int] = None,
    use_cpu: Flag = False,
    **kwargs
):
    """


    :param config:
    :param videos:
    :param gpu_index:
    :param batch_size:
    :param cropping:
    :param num_outputs:
    :param predictor:
    :param predictor_settings:
    :param kwargs:

    :return:
    """
    _setup_gpus(use_cpu, gpu_index)

    batch_size = _get_default_value(sleap.load_model, "batch_size", 4) if(batch_size is None) else batch_size
    num_outputs = 1 if(num_outputs is None) else num_outputs

    print("Loading Model...")
    model = sleap.load_model(_paths_to_str(config), batch_size=batch_size)
    # Get the model extractor...
    mdl_extractor = PredictorExtractor(model)
    mdl_metadata = mdl_extractor.get_metadata()

    predictor_cls = get_predictor("SegmentedFramePassEngine" if(predictor is None) else predictor)
    print(f"Using predictor: '{predictor_cls.get_name()}'")

    if(not predictor_cls.supports_multi_output() and num_outputs > 1):
        raise ValueError(f"Predictor '{predictor_cls.get_name()}' doesn't support multiple outputs!")

    visual_settings = Config(kwargs, VISUAL_SETTINGS)

    if(isinstance(videos, str)):
        videos = [videos]

    for video_path in videos:
        print(f"Running analysis on video: '{video_path}'")
        video_path = Path(video_path).resolve()
        video = sleap.load_video(str(video_path))
        output_path = video_path.parent / (video_path.name + ".diplomat.slp")

        video_metadata = _get_video_metadata(video_path, output_path, num_outputs, video, visual_settings, mdl_metadata)
        pred = predictor_cls(mdl_metadata["bp_names"], num_outputs, video.num_frames, _get_predictor_settings(predictor_cls, kwargs), video_metadata)

        prog_bar = TQDMProgressBar(total=video.num_frames)

        for batch in mdl_extractor.extract(video):
            result = pred.on_frames(batch)






