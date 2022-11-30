import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional, Tuple

import diplomat.processing.type_casters as tc
from diplomat.utils.cli_tools import extra_cli_args
from diplomat.processing import get_predictor, Config, TQDMProgressBar, Predictor
from diplomat.utils.video_info import is_video
from diplomat.utils import frame_store_fmt

from .run_utils import _get_default_value, _paths_to_str, _get_video_metadata, _get_predictor_settings, PoseLabels, Timer, _attach_run_info
from .sleap_providers import PredictorExtractor, SleapMetadata
from .visual_settings import VISUAL_SETTINGS

import sleap


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
    **kwargs
):
    batch_size = _get_default_value(sleap.load_model, "batch_size", 4) if (batch_size is None) else batch_size
    num_outputs = 1 if (num_outputs is None) else num_outputs

    print("Loading Model...")
    model = sleap.load_model(_paths_to_str(config), batch_size=batch_size)
    # Get the model extractor...
    mdl_extractor = PredictorExtractor(model)
    mdl_metadata = mdl_extractor.get_metadata()

    predictor_cls = get_predictor("SegmentedFramePassEngine" if (predictor is None) else predictor)
    print(f"Using predictor: '{predictor_cls.get_name()}'")

    if (not predictor_cls.supports_multi_output() and num_outputs > 1):
        raise ValueError(f"Predictor '{predictor_cls.get_name()}' doesn't support multiple outputs!")

    visual_settings = Config(kwargs, VISUAL_SETTINGS)

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
            batch_size
        )


def _analyze_frame_store(
    frame_store_path: str,
    predictor_cls: Type[Predictor],
    mdl_metadata: SleapMetadata,
    num_outputs: int,
    visual_settings: Config,
    predictor_settings: Optional[dict],
    batch_size: int
):
    frame_store_path = Path(frame_store_path).resolve()
    video_path = frame_store_path if (is_video(frame_store_path)) else None
    output_path = frame_store_path.parent / (frame_store_path.name + f".diplomat_{predictor_cls.get_name()}.slp")

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

            fake_video = _DummyVideo(fps=f_rate, num_frames=num_f, shape=(num_f, vid_h, vid_w, len(bp_lst)))

            video_metadata = _get_video_metadata(
                video_path,
                output_path,
                num_outputs,
                fake_video,
                visual_settings,
                mdl_metadata,
                None if(off_x is None) else (off_y, off_x)
            )

            pred = predictor_cls(
                mdl_metadata["bp_names"], num_outputs, num_f, _get_predictor_settings(predictor_cls, predictor_settings), video_metadata
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