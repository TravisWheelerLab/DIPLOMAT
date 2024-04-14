import sys
from pathlib import Path
from typing import Optional, Type
import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args, Flag
from diplomat.processing.progress_bar import TQDMProgressBar
from diplomat.processing import get_predictor, Config, Predictor

from .sleap_providers import PredictorExtractor
from .visual_settings import VISUAL_SETTINGS
from .run_utils import (
    _attach_run_info,
    _get_predictor_settings,
    _paths_to_str,
    _setup_gpus,
    _get_default_value,
    _get_video_metadata,
    PoseLabels,
    Timer
)


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
    output_suffix: str = "",
    refinement_kernel_size: int = 5,
    use_cpu: Flag = False,
    **kwargs
):
    """
    Run DIPLOMAT tracking on a set of videos or a single video using a SLEAP model, generating results in ".slp" files.

    :param config: The path (or list of paths) to the sleap model(s) used for inference, each as either as a folder or zip file.
    :param videos: A path or list of paths to video files to run DIPLOMAT on using the passed SLEAP model.
    :param batch_size: An integer, the number of images to pass to the model per batch.
    :param num_outputs: An integer, the number of individuals in the video defaults to 1.
    :param predictor: A string, the name of the predictor to use to perform the task of tracking.
    :param predictor_settings: A dictionary of strings to any values, the settings to use for the predictor. Each predictor offers different settings,
                               see :py:cli:`diplomat predictors list_settings` or :py:func:`~diplomat.predictor_ops.get_predictor_settings` to get
                               the settings a predictor plugin supports.
    :param gpu_index: An integer, the index of the GPU to use. If not set DIPLOMAT allows SLEAP to automatically select a GPU.
    :param output_suffix: A string, the suffix to append onto the output .slp file. Defaults to an empty string.
    :param refinement_kernel_size: An integer, the kernel size to use for creating offset maps if they don't exist (via integral refinement).
                                   defaults to False, if set to 0 or a negative integer disables integral refinement.
    :param use_cpu: A boolean, if True force SLEAP to use the CPU to run the model. Defaults to False.
    :param kwargs: The following additional arguments are supported:

                   {extra_cli_args}
    """
    _setup_gpus(use_cpu, gpu_index)

    import sleap
    batch_size = _get_default_value(sleap.load_model, "batch_size", 4) if(batch_size is None) else batch_size
    if(num_outputs is None):
        raise ValueError("'num_outputs' is not set! Please set it to the number of bodies you are tracking.")

    print("Loading Model... TEST")
    model = sleap.load_model(_paths_to_str(config), batch_size=batch_size)
    # Get the model extractor...
    mdl_extractor = PredictorExtractor(model, refinement_kernel_size)
    mdl_metadata = mdl_extractor.get_metadata()

    predictor_cls = get_predictor("SegmentedFramePassEngine" if(predictor is None) else predictor)
    print(f"Using predictor: '{predictor_cls.get_name()}'")

    if(not predictor_cls.supports_multi_output() and num_outputs > 1):
        raise ValueError(f"Predictor '{predictor_cls.get_name()}' doesn't support multiple outputs!")

    visual_settings = Config(kwargs, VISUAL_SETTINGS)

    videos = _paths_to_str(videos)
    if(isinstance(videos, str)):
        videos = [videos]

    for video_path in videos:
        print(f"Running analysis on video: '{video_path}'")
        _analyze_single_video(
            video_path,
            predictor_cls,
            mdl_extractor,
            num_outputs,
            visual_settings,
            mdl_metadata,
            predictor_settings,
            output_suffix
        )


def _analyze_single_video(
    video_path: str,
    predictor_cls: Type[Predictor],
    mdl_extractor: PredictorExtractor,
    num_outputs: int,
    visual_settings: Config,
    mdl_metadata: dict,
    predictor_settings: Optional[dict],
    output_suffix: str
):
    import sleap
    video_path = Path(video_path).resolve()
    video = sleap.load_video(str(video_path))
    output_path = video_path.parent / (video_path.name + f".diplomat_{predictor_cls.get_name()}{output_suffix}.slp")

    video_metadata = _get_video_metadata(video_path, output_path, num_outputs, video, visual_settings, mdl_metadata)
    predictor = predictor_cls(
        mdl_metadata["bp_names"], num_outputs, video.num_frames,
        _get_predictor_settings(predictor_cls, predictor_settings), video_metadata
    )

    labels = PoseLabels(video, num_outputs, mdl_metadata["orig_skeleton"])
    total_frames = 0

    with predictor as pred:
        with Timer() as timer:
            with TQDMProgressBar(total=video.num_frames) as prog_bar:
                print("Running the model...")
                for batch in mdl_extractor.extract(video):
                    result = pred.on_frames(batch)
                    labels.append(result)
                    prog_bar.update(batch.get_frame_count())
                    total_frames += batch.get_frame_count()

            with TQDMProgressBar(total=video.num_frames - len(labels)) as post_pbar:
                print(f"Running post-processing algorithms...")
                result = pred.on_end(post_pbar)
                labels.append(result)

    if (total_frames != len(labels)):
        raise ValueError(
            f"The predictor algorithm did not return the same amount of frames as are in the video.\n"
            f"Expected Amount: {total_frames}, Actual Amount Returned: {len(labels)}"
        )

    print(f"Saving output to: {output_path}")
    labels = _attach_run_info(labels.to_sleap(), timer, str(video_path), str(output_path), sys.argv)
    labels.save(str(output_path))

    print(f"Finished inference at: {timer.end_date}")
    print(f"Total runtime: {timer.duration} secs")
    print()






















