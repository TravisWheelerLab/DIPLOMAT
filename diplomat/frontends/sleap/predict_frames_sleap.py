import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional, Tuple

import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args
from diplomat.processing import get_predictor, Config, TQDMProgressBar, Predictor
from diplomat.utils.video_info import is_video
from diplomat.utils import frame_store_fmt

from .run_utils import (
    _get_default_value,
    _paths_to_str,
    _get_video_metadata,
    _get_predictor_settings,
    PoseLabels,
    Timer,
    _attach_run_info,
    _load_config
)

from .sleap_importer import sleap
from .sleap_providers import sleap_metadata_from_config, SleapMetadata
from .visual_settings import VISUAL_SETTINGS


@dataclass
class _DummyVideo:
    fps: float
    num_frames: float
    shape: Tuple[int, int, int, int]


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False)
@tc.typecaster_function
def analyze_frames(
    config: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    frame_stores: tc.Union[tc.List[tc.PathLike], tc.PathLike],
    batch_size: tc.Optional[int] = None,
    num_outputs: tc.Optional[int] = None,
    predictor: tc.Optional[str] = None,
    predictor_settings: tc.Optional[tc.Dict[str, tc.Any]] = None,
    output_suffix: str = "",
    refinement_kernel_size: int = 5,
    **kwargs
):
    """
    Run DIPLOMAT tracking on a set of frame store files or a single frame store using metadata from a SLEAP model, generating results in ".slp" files.

    :param config: The path (or list of paths) to the sleap model(s) used for inference, each as either as a folder or zip file.
    :param frame_stores: A path or list of paths to frame store ('.dlfs') files to run DIPLOMAT on.
    :param batch_size: An integer, the number of frames to read in from the frame store at a time.
    :param num_outputs: An integer, the number of individuals in the video defaults to 1.
    :param predictor: A string, the name of the predictor to use to perform the task of tracking.
    :param predictor_settings: A dictionary of strings to any values, the settings to use for the predictor. Each predictor offers different settings,
                               see :py:cli:`diplomat predictors list_settings` or :py:func:`~diplomat.predictor_ops.get_predictor_settings` to get
                               the settings a predictor plugin supports.
    :param output_suffix: A string, the suffix to append onto the output .slp file. Defaults to an empty string.
    :param refinement_kernel_size: An integer, the kernel size to use for creating offset maps if they don't exist (via integral refinement).
                                   defaults to False, if set to 0 or a negative integer disables integral refinement.
    :param kwargs: The following additional arguments are supported:

                   {extra_cli_args}
    """
    import sleap
    batch_size = _get_default_value(sleap.load_model, "batch_size", 4) if (batch_size is None) else batch_size
    if(num_outputs is None):
        raise ValueError("'num_outputs' is not set! Please set it to the number of bodies you are tracking.")

    print("Loading Config...")
    config = _load_config(_paths_to_str(config))[0]
    mdl_metadata = sleap_metadata_from_config(config.data)

    predictor_cls = get_predictor("SegmentedFramePassEngine" if (predictor is None) else predictor)
    print(f"Using predictor: '{predictor_cls.get_name()}'")

    if (not predictor_cls.supports_multi_output() and num_outputs > 1):
        raise ValueError(f"Predictor '{predictor_cls.get_name()}' doesn't support multiple outputs!")

    visual_settings = Config(kwargs, VISUAL_SETTINGS)

    frame_stores = _paths_to_str(frame_stores)
    if(isinstance(frame_stores, str)):
        frame_stores = [frame_stores]

    for frame_store_path in frame_stores:
        print(f"Running analysis on frame store: '{frame_store_path}'")
        _analyze_frame_store(
            frame_store_path,
            predictor_cls,
            mdl_metadata,
            num_outputs,
            visual_settings,
            predictor_settings,
            batch_size,
            output_suffix
        )


def _analyze_frame_store(
    frame_store_path: str,
    predictor_cls: Type[Predictor],
    mdl_metadata: SleapMetadata,
    num_outputs: int,
    visual_settings: Config,
    predictor_settings: Optional[dict],
    batch_size: int,
    output_suffix: str
):
    frame_store_path = Path(frame_store_path).resolve()
    video_path = frame_store_path if (is_video(frame_store_path)) else None
    output_path = frame_store_path.parent / (frame_store_path.name + f".diplomat_{predictor_cls.get_name()}{output_suffix}.slp")

    # LOAD the frame store file...
    with frame_store_path.open("rb") as frame_store:
        with frame_store_fmt.DLFSReader(frame_store) as frame_reader:
            (
                num_f,
                f_h,
                f_w,
                f_rate,
                stride,
                vid_h,
                vid_w,
                off_y,
                off_x,
                bp_lst,
            ) = frame_reader.get_header().to_list()

            if(video_path is None):
                fake_video = _DummyVideo(fps=f_rate, num_frames=num_f, shape=(num_f, vid_h, vid_w, len(bp_lst)))
            else:
                from sleap.io.video import MediaVideo
                fake_video = sleap.Video(backend=sleap.Video.make_specific_backend(
                    MediaVideo,
                    dict(
                        filename=sleap.Video.fixup_path(str(frame_store_path)),
                        grayscale=False,
                        input_format="channels_last",
                        dataset=""
                    )
                ))

            video_metadata = _get_video_metadata(
                video_path,
                output_path,
                num_outputs,
                fake_video,
                visual_settings,
                mdl_metadata,
                None if(off_x is None) else (off_y, off_x)
            )

            predictor = predictor_cls(
                mdl_metadata["bp_names"],
                num_outputs,
                num_f,
                _get_predictor_settings(predictor_cls, predictor_settings),
                video_metadata
            )

            bp_to_idx = {val: i for (i, val) in enumerate(bp_lst)}
            idx_to_keep = [bp_to_idx.get(val, None) for val in mdl_metadata["bp_names"]]

            if(any(i is None for i in idx_to_keep)):
                raise ValueError(
                    f"Unable to use frame store, body part(s) "
                    f"{[i for i, val in enumerate(idx_to_keep) if(val is None)]} "
                    f"are missing from the frame store."
                )

            labels = PoseLabels(fake_video, num_outputs, mdl_metadata["orig_skeleton"])
            total_frames = 0

            with predictor as pred:
                with Timer() as timer:
                    with TQDMProgressBar(total=num_f) as prog_bar:
                        print("Running the predictor on frames...")
                        while(frame_reader.has_next()):
                            batch = frame_reader.read_frames(min(batch_size, num_f - total_frames))

                            # Fix the batch to only have what is needed...
                            batch.set_source_map(batch.get_source_map()[:, :, :, idx_to_keep])
                            if(batch.get_offset_map() is not None):
                                batch.set_offset_map(batch.get_offset_map()[:, :, :, idx_to_keep])

                            result = pred.on_frames(batch)
                            labels.append(result)
                            prog_bar.update(batch.get_frame_count())
                            total_frames += batch.get_frame_count()

                    with TQDMProgressBar(total=num_f - len(labels)) as post_pbar:
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