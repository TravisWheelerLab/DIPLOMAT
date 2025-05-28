import os
import sys
from diplomat.core_ops.shared_commands.annotate import _label_videos_single
from diplomat.core_ops.shared_commands.save_from_restore import _save_from_restore
from diplomat.core_ops.shared_commands.tracking import analyze_frames, analyze_videos
from diplomat.core_ops.shared_commands.utils import _fix_path_pairs, _get_track_loaders, _load_tracks_from_loaders
from diplomat.core_ops.shared_commands.tweak import _tweak_video_single
from diplomat.core_ops.shared_commands.visual_settings import VISUAL_SETTINGS, FULL_VISUAL_SETTINGS
from diplomat.processing import Config, Predictor, get_predictor
from diplomat.processing.type_casters import (
    typecaster_function,
    PathLike,
    Union,
    Optional,
    List,
    Dict,
    Any,
    get_typecaster_annotations,
    NoneType,
    get_typecaster_required_arguments
)
from diplomat.utils.pretty_printer import printer as print
from diplomat.utils.cli_tools import (func_to_command, allow_arbitrary_flags, Flag, positional_argument_count, CLIError,
                                      extra_cli_args, func_args_to_config_spec, clear_extra_cli_args_and_copy)
from argparse import ArgumentParser
import typing
from types import ModuleType

from diplomat.utils.track_formats import save_diplomat_table, load_diplomat_table
from diplomat.utils.tweak_ui import UIImportError
from diplomat.frontends import DIPLOMATContract, DIPLOMATCommands


class ArgumentError(CLIError):
    """ Error in arguments passed to CLI Command """
    pass


def _get_casted_args(tc_func, extra_args, error_on_miss=True):
    """
    PRIVATE: Get correctly casted extra arguments for the provided typecasting function. Any arguments that don't
    match those in the function raise an ArgumentError, unless error_on_miss is set to false.
    """
    if(isinstance(tc_func, Predictor)):
        def_tcs = tc_func.get_settings()
        def_tcs = def_tcs if(def_tcs is not None) else {}
        def_tcs = {k: v[1] for k, v in def_tcs.items()}
    else:
        def_tcs = get_typecaster_annotations(tc_func)
    extra = getattr(tc_func, "__extra_args", {})
    autocast = getattr(tc_func, "__auto_cast", True)
    allow_arb = getattr(tc_func, "__allow_arbitrary_flags", False)

    new_args = {}
    leftover = {}

    for k, v in extra_args.items():
        if (k in def_tcs):
            new_args[k] = def_tcs[k](v)
        elif (k in extra):
            new_args[k] = extra[k][1](v) if (autocast) else v
        else:
            if (allow_arb):
                new_args[k] = v
                continue
            msg = (
                f"Warning: command '{tc_func.__name__}' does not have "
                f"an argument called '{k}'!"
            )
            if (not error_on_miss):
                print(f"{msg} Ignoring the argument...")
                leftover[k] = v
            else:
                raise ArgumentError(msg)

    return new_args, leftover


def _find_frontend(
    contracts: Union[DIPLOMATContract, List[DIPLOMATContract]],
    config: Union[List[os.PathLike], os.PathLike],
    **kwargs: typing.Any
) -> typing.Tuple[str, ModuleType]:
    from diplomat import _LOADED_FRONTENDS

    contracts = [contracts] if (isinstance(contracts, DIPLOMATContract)) else contracts

    print(f"Loaded frontends: {_LOADED_FRONTENDS}")

    print(f"Config: {config}")

    for name, funcs in _LOADED_FRONTENDS.items():
        print(f"Checking frontend '{name}'...")

        for contract in contracts:
            print(f"Verifying contract '{contract}'...")
            verified = funcs.verify(contract=contract, config=config, **kwargs)
            print(f"Verified: {verified}")

        if (all(funcs.verify(
                contract=c,
                config=config,
                **kwargs
        ) for c in contracts)):
            print(f"Frontend '{name}' selected.")
            return (name, funcs)

    print("Could not find a frontend that correctly handles the passed config and other arguments. Make sure the "
          "config passed is valid.")
    sys.exit(1)


def _display_help(
    frontend_name: str,
    method_type: str,
    calling_command_name: str,
    command_func: typing.Callable,
    is_cli: bool
):
    if (is_cli):
        print(f"\n\nHelp for {frontend_name}'s {method_type} command:\n")
        func_to_command(command_func, ArgumentParser(prog=calling_command_name), allow_short_form=False).print_help()
    else:
        import pydoc
        help_dumper = pydoc.Helper(output=sys.stdout, input=sys.stdin)

        print(f"\n\nDocstring for {frontend_name}'s {method_type} method:\n")
        help_dumper.help(command_func)


@allow_arbitrary_flags
@typecaster_function
@positional_argument_count(1)
def yaml(
    run_config: Union[PathLike, NoneType] = None,
    **extra_args
):
    """
    Run DIPLOMAT based on a passed yaml run script. The yaml script should include a 'command' key specifying the
    DIPLOMAT sub-command to run and an 'arguments' key specifying the list of arguments (key-value pairs) to
    pass to the command.

    :param run_config: The path to the YAML configuration file to use to configure the running of DIPLOMAT.
                       If this value is not specified or is None, read the yaml from standard output.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the selected
                       command, as specified in the YAML file. Additional arguments passed with the same name as ones
                       found in the YAML file will overwrite the values found in the YAML file.

    :return: Returns the value returned by the selected diplomat command.
    """
    import yaml

    if (run_config is None):
        data = yaml.load(sys.stdin, yaml.SafeLoader)
    else:
        with open(str(run_config), "r") as f:
            data = yaml.load(f, yaml.SafeLoader)

    command_name = data.get("command", None)
    arguments = data.get("arguments", {})

    if (not isinstance(command_name, str)):
        raise ArgumentError(f"Yaml file 'command' attribute does not have a value that is a string.")
    if (not isinstance(arguments, dict)):
        raise ArgumentError(f"Yaml file 'arguments' attribute not a list of key-value pairs, or mapping.")

    # Load the command...
    from diplomat._cli_runner import get_dynamic_cli_tree
    cli_tree = get_dynamic_cli_tree()

    sub_tree = cli_tree
    for command_part in command_name.strip().split():
        try:
            sub_tree = cli_tree[command_part]
        except KeyError:
            raise ArgumentError(f"Command '{command_name}' is not a valid diplomat command.")

    try:
        get_typecaster_annotations(sub_tree)
    except TypeError:
        raise ArgumentError(f"Command '{command_name}' is not a valid diplomat command.")

    arguments.update(extra_args)

    for arg in get_typecaster_required_arguments(sub_tree):
        if (arg not in arguments):
            raise ArgumentError(
                f"Command '{command_name}' requires '{arg}' to be passed, include it in the yaml file or pass it as a "
                f"flag to this command."
            )

    return sub_tree(**(_get_casted_args(sub_tree, arguments)[0]))


@allow_arbitrary_flags
@extra_cli_args(VISUAL_SETTINGS, auto_cast=False, doc_header="Additional visual arguments:")
@typecaster_function
def track_with(
    config: Union[List[PathLike], PathLike],
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    frame_stores: Optional[Union[List[PathLike], PathLike]] = None,
    num_outputs: Optional[int] = None,
    batch_size: Optional[int] = None,
    predictor: Optional[str] = None,
    predictor_settings: Optional[Dict[str, Any]] = None,
    output_suffix: str = "",
    help_extra: Flag = False,
    **extra_args
):
    """
    Run DIPLOMAT tracking on videos and/or frame stores. Automatically select a frontend based on the passed arguments.
    Allows for selecting a specific tracker, or predictor.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files to run analysis on.
    :param frame_stores: A single path or list of paths to frame store files to run analysis on.
    :param num_outputs: An integer, the number of bodies to track in the video. If not set the frontend will try to pull it from the project configuration.
    :param batch_size: An integer, the number of frame to process at a single time. If not set the frontend will try to pull it from the project configuration.
    :param predictor: An optional string, specifying the predictor plugin to make predictions with. You can get a list of all available predictors
                      and descriptions using the "diplomat predictors list" command or "diplomat.list_predictor_plugins" function.
    :param predictor_settings: An optional dictionary, listing the settings to use for the specified predictor plugin instead of the defaults.
                               If not specified, the frontend will determine the settings in a frontend specific manner. To see the settings a
                               predictor plugin supports, use the "diplomat predictors list_settings" command or "diplomat.get_predictor_settings"
                               function. To get more information about how a frontend gets settings if not passed, set the help_extra parameter
                               to True to print additional settings for the selected frontend instead of running tracking.
    :param output_suffix: String, a suffix to append to name of the output file. Defaults to no suffix...
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of running tracking.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the frontend, visual settings, or predictor, in that order.
                       To see valid frontend arguments, run track with extra_help flag set to true.
                       {extra_cli_args}
    """
    from diplomat import CLI_RUN

    if (help_extra):
        selected_frontend_name, selected_frontend = _find_frontend(
            contracts=[DIPLOMATCommands._load_model],
            config=config,
            num_outputs=num_outputs,
            batch_size=batch_size,
            **extra_args
        )
        if predictor is None:
            predictor = "SegmentedFramePassEngine"
        predictor_settings = get_predictor(predictor).get_settings()
        if(predictor_settings is None):
            predictor_settings = {}

        predictor_settings.update(VISUAL_SETTINGS)
        predictor_settings.update(
            func_args_to_config_spec(selected_frontend._load_model, track_with)
        )

        _track_with_help = extra_cli_args(predictor_settings, doc_header="Additional frontend, visual, and predictor settings:")(clear_extra_cli_args_and_copy(track_with))

        _display_help(
            selected_frontend_name,
            f"track_with with the {predictor} predictor",
            "diplomat track_with",
            _track_with_help,
            CLI_RUN
        )
        return

    if (videos is None and frame_stores is None):
        print("No frame stores or videos passed, terminating.")
        return

    # If some videos are supplied, run the frontends video analysis function.
    if (videos is not None):
        print("Running on videos...")
        selected_frontend_name, selected_frontend = _find_frontend(
            contracts=[DIPLOMATCommands._load_model],
            config=config,
            num_outputs=num_outputs,
            batch_size=batch_size,
            **extra_args
        )

        model_args, additional_args = _get_casted_args(selected_frontend._load_model, extra_args, error_on_miss=False)
        visual_args, additional_args = _get_casted_args(analyze_videos, additional_args, error_on_miss=False)
        ps_video = {}
        ps_video.update(predictor_settings if(predictor_settings is not None) else {})
        ps_video.update(additional_args)

        model_info, model = selected_frontend._load_model(
            config=config,
            batch_size=batch_size,
            num_outputs=num_outputs,
            **model_args
        )

        analyze_videos(
            model=model,
            model_info=model_info,
            videos=videos,
            predictor=predictor,
            predictor_settings=predictor_settings,
            output_suffix=output_suffix,
            **visual_args
        )

    # If some frame stores are supplied, run the frontends frame analysis function.
    if (frame_stores is not None):
        print("Running on frame stores...")
        visual_args, additional_args = _get_casted_args(analyze_videos, extra_args, error_on_miss=False)
        ps_frames = {}
        ps_frames.update(predictor_settings if(predictor_settings is not None) else {})
        ps_frames.update(additional_args)

        analyze_frames(
            config=config,
            frame_stores=frame_stores,
            num_outputs=num_outputs,
            predictor=predictor,
            predictor_settings=ps_frames,
            **visual_args
        )


@allow_arbitrary_flags
@extra_cli_args(VISUAL_SETTINGS, auto_cast=False, doc_header="Additional visual arguments:")
@typecaster_function
def track(
    config: Union[List[PathLike], PathLike],
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    frame_stores: Optional[Union[List[PathLike], PathLike]] = None,
    num_outputs: Optional[int] = None,
    batch_size: Optional[int] = None,
    settings: Optional[Dict[str, Any]] = None,
    output_suffix: str = "",
    help_extra: Flag = False,
    **extra_args
):
    """
    Run diplomat in a non-interactive tracking mode on the specified config and videos or frame stores. An alias for
    DIPLOMAT track_with with the SegmentedFramePassEngine predictor. The interactive UI can be restored later using
    diplomat interact function or cli command.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files to run analysis on.
    :param frame_stores: A single path or list of paths to frame store files to run analysis on.
    :param num_outputs: An integer, the number of bodies to track in the video. If not set the frontend will try to pull it from the project configuration.
    :param batch_size: An integer, the number of frame to process at a single time. If not set the frontend will try to pull it from the project configuration.
    :param settings: An optional dictionary, listing the settings to use for the specified predictor plugin instead of the defaults.
                     If not specified, the frontend will determine the settings in a frontend specific manner. To see the settings a
                     predictor plugin supports, use the "diplomat predictors list_settings" command or "diplomat.get_predictor_settings"
                     function. To get more information about how a frontend gets settings if not passed, set the help_extra parameter
                     to True to print additional settings for the selected frontend instead of running tracking.
    :param output_suffix: String, a suffix to append to name of the output file. Defaults to no suffix...
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of running tracking.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the frontend, visual settings, or predictor, in that order.
                       To see valid frontend arguments, run track with extra_help flag set to true.
                       {extra_cli_args}
    """
    track_with(
        config=config,
        videos=videos,
        frame_stores=frame_stores,
        num_outputs=num_outputs,
        batch_size=batch_size,
        predictor="SegmentedFramePassEngine",
        predictor_settings=settings,
        output_suffix=output_suffix,
        help_extra=help_extra,
        **extra_args
    )


@allow_arbitrary_flags
@extra_cli_args(VISUAL_SETTINGS, auto_cast=False, doc_header="Additional visual arguments:")
@typecaster_function
def track_and_interact(
    config: Union[List[PathLike], PathLike],
    videos: Optional[Union[List[PathLike], PathLike]] = None,
    frame_stores: Optional[Union[List[PathLike], PathLike]] = None,
    num_outputs: Optional[int] = None,
    batch_size: Optional[int] = None,
    settings: Optional[Dict[str, Any]] = None,
    output_suffix: str = "",
    help_extra: Flag = False,
    **extra_args
):
    """
    Run diplomat in interactive tracking mode on the specified config and videos or frame stores. An alias for
    DIPLOMAT track_with with the SupervisedSegmentedFramePassEngine predictor.

    :param config: The path to the configuration file for the project. The format of this argument will depend on the frontend.
    :param videos: A single path or list of paths to video files to run analysis on.
    :param frame_stores: A single path or list of paths to frame store files to run analysis on.
    :param num_outputs: An integer, the number of bodies to track in the video. If not set the frontend will try to pull it from the project configuration.
    :param batch_size: An integer, the number of frame to process at a single time. If not set the frontend will try to pull it from the project configuration.
    :param settings: An optional dictionary, listing the settings to use for the specified predictor plugin instead of the defaults.
                     If not specified, the frontend will determine the settings in a frontend specific manner. To see the settings a
                     predictor plugin supports, use the "diplomat predictors list_settings" command or "diplomat.get_predictor_settings"
                     function. To get more information about how a frontend gets settings if not passed, set the help_extra parameter
                     to True to print additional settings for the selected frontend instead of running tracking.
    :param output_suffix: String, a suffix to append to name of the output file. Defaults to no suffix...
    :param help_extra: Boolean, if set to true print extra settings for the automatically selected frontend instead of running tracking.
    :param extra_args: Any additional arguments (if the CLI, flags starting with '--') are passed to the frontend, visual settings, or predictor, in that order.
                       To see valid frontend arguments, run track with extra_help flag set to true.
                       {extra_cli_args}
    """
    track_with(
        config=config,
        videos=videos,
        frame_stores=frame_stores,
        num_outputs=num_outputs,
        batch_size=batch_size,
        predictor="SupervisedSegmentedFramePassEngine",
        predictor_settings=settings,
        output_suffix=output_suffix,
        help_extra=help_extra,
        **extra_args
    )


@extra_cli_args(FULL_VISUAL_SETTINGS, auto_cast=False, doc_header="Additional visual arguments:")
@typecaster_function
def annotate(
    videos: Union[List[PathLike], PathLike],
    csvs: Union[List[PathLike], PathLike],
    body_parts_to_plot: Optional[List[str]] = None,
    video_extension: str = "mp4",
    **kwargs
):
    """
    Have diplomat annotate, or label a video given it has already been tracked.

    :param videos: Paths to video file(s) corresponding to the provided csv files.
    :param csvs: The path (or list of paths) to the csv file(s) to label the videos with.
    :param body_parts_to_plot: A set or list of body part names to label, or None, indicating to label all parts.
    :param video_extension: The file extension to use on the created labeled video, excluding the dot.
                            Defaults to 'mp4'.
    :param kwargs: {extra_cli_args}
    """
    csvs, videos = _fix_path_pairs(csvs, videos)
    visual_settings = Config(kwargs, FULL_VISUAL_SETTINGS)

    if len(videos) == 0:
        print("No videos passed, terminating.")
        return

    for c, v in zip(csvs, videos):
        _label_videos_single(str(c), str(v), body_parts_to_plot, video_extension, visual_settings)


@extra_cli_args(VISUAL_SETTINGS, auto_cast=False, doc_header="Additional visual arguments:")
@typecaster_function
def tweak(
    videos: Union[List[PathLike], PathLike],
    csvs: Union[List[PathLike], PathLike],
    **kwargs
):
    """
    Make modifications to DIPLOMAT produced tracking results created for a video using a limited version of the
    interactive labeling UI. Allows for touching up and fixing any minor issues that may arise after tracking and
    saving results.

    :param videos: Paths to video file(s) corresponding to the provided csv files.
    :param csvs: The path (or list of paths) to the csv file(s) to edit.
    :param kwargs: {extra_cli_args}
    """
    csvs, videos = _fix_path_pairs(csvs, videos)
    visual_cfg = Config(kwargs, VISUAL_SETTINGS)

    try:
        for c, v in zip(csvs, videos):
            _tweak_video_single(str(c), str(v), visual_cfg)
    except UIImportError as e:
        print(e)


@typecaster_function
def interact(
    state: Union[List[PathLike], PathLike]
):
    """
    Open diplomat's interactive UI from a .dipui file. Allows for reloading the UI when diplomat crashes, or for
    further editing. Settings and backend will be restored automatically based on the settings and info passed during
    the first run.

    :param state: A path or list of paths to the ui states to restore. Files should be of ".dipui" format.
    """
    from diplomat.predictors.sfpe.file_io import DiplomatFPEState
    from diplomat.processing import TQDMProgressBar, Config
    import time

    try:
        from diplomat.predictors.supervised_sfpe.supervised_segmented_frame_pass_engine \
            import SupervisedSegmentedFramePassEngine
    except ImportError:
        raise UIImportError("Unable to load diplomat UI. Make sure diplomat ui packages are installed")

    if (not isinstance(state, (list, tuple))):
        state = [state]

    for state_file_path in state:
        with open(state_file_path, "r+b") as f:
            with DiplomatFPEState(f) as dip_st:
                meta = dip_st.get_metadata()
                num_frames = len(dip_st) // (len(meta["bodyparts"]) * meta["num_outputs"])

        # Create the UI...
        pred = SupervisedSegmentedFramePassEngine(
            meta["bodyparts"],
            meta["num_outputs"],
            num_frames,
            Config(meta["settings"], SupervisedSegmentedFramePassEngine.get_settings()),
            Config(meta["video_metadata"]),
            restore_path=str(state_file_path)
        )

        with pred as p:
            start_time = time.time()
            with TQDMProgressBar(total=num_frames) as prog_bar:
                poses = p.on_end(prog_bar)
            end_time = time.time()

        if (poses is None):
            raise ValueError("Pass didn't return any data!")

        _save_from_restore(
            pose=poses,
            video_metadata=pred.video_metadata,
            num_outputs=pred.num_outputs,
            parts=pred.bodyparts,
            frame_width_pixels=pred.width,
            frame_height_pixels=pred.height,
            start_time=start_time,
            end_time=end_time
        )


@typecaster_function
def convert_tracks(
    inputs: Union[List[PathLike], PathLike],
    outputs: Union[NoneType, List[PathLike], PathLike] = None
):
    """
    Convert files storing final tracking results for a video from other software to diplomat csv's format that can be
    used with diplomat's tweak and annotate commands.

    :param inputs: A single or list of paths to files to convert to diplomat csvs.
    :param outputs: An optional single path or list of paths, the location to write converted files to. If not
                    specified, places the converted files at same locations as inputs with an extension of .csv
                    instead of the original extension.
    """
    from pathlib import Path

    loaders = _get_track_loaders()
    if(len(loaders) == 0):
        raise ImportError("Unable to find any loaded frontends with csv conversion support.")

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    if outputs is None:
        outputs = []
        for p in inputs:
            p = Path(p).resolve()
            outputs.append(p.parent / (p.stem + ".csv"))
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    if len(inputs) != len(outputs):
        raise ValueError("The provided paths and destinations do not have the same length!")

    for inp, out in zip(inputs, outputs):
        print(f"Converting HDF5 to CSV: {inp}->{out}")
        out = Path(out).resolve()
        diplomat_table = _load_tracks_from_loaders(loaders, inp)
        save_diplomat_table(diplomat_table, str(out))